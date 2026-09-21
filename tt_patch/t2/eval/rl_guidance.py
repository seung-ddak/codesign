"""π_φ residual guidance — DSG 의 `∇_E R` 자리를 학습된 정책으로 교체한다 (D11).

DSG 와의 유일한 차이는 **guidance 를 무엇이 만드는가** 뿐이다.

    DSG : guidance = scale · ∇_{x_t} R̂(x̂_{t+1})      ← 모델 예측 reward 의 해석적 gradient
    RL  : guidance = π_φ(x_t, k)                      ← REINFORCE 로 학습된 residual

나머지(prior·샘플러·평가기·reward 함수)는 전부 동일하게 둔다.

실측 근거 (`tt_codesign/code/probe_guidance.py`):
  - 생성 1 회 = denoising **100 step**.
  - DSG 의 |guidance| 는 앞 90 step 에서 1e-19 (사실상 0), **마지막 10 step 에서만** 4e-3~6e-3
    (max 3e-2). → D11 의 "마지막 10 step" 이 구조적으로 정확히 맞다. `action_scale` 도 이 규모.
  - guidance term 모양: dyna_joint (B,62,29) · fixed_joint (B,100,19) · link (B,78,35)
    · actuator (B,19,14).
"""

from __future__ import annotations

import logging
import math
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn

from t2.eval.diffusion_guidance import HardwareOptimizer


class ResidualGuidancePolicy(nn.Module):
    """토큰별 MLP. (토큰 특징, guided window 내 step) -> pred_noise 와 같은 모양의 residual.

    term 마다 차원이 다르므로 첫 호출 때 지연 생성한다. 토큰 축으로 가중치를 공유하므로
    로봇마다 토큰 수가 달라도 그대로 쓸 수 있다 (RoboToken 의 가변 길이).
    """

    def __init__(
        self,
        hidden: int = 64,
        step_emb_dim: int = 16,
        guided_steps: int = 10,
        action_scale: float = 0.02,
        init_log_std: float = math.log(0.005),
    ):
        super().__init__()
        self.hidden = hidden
        self.step_emb_dim = step_emb_dim
        self.guided_steps = guided_steps
        self.action_scale = action_scale
        self.init_log_std = init_log_std
        self.step_emb = nn.Embedding(guided_steps, step_emb_dim)
        self.trunks = nn.ModuleDict()
        self.log_stds = nn.ParameterDict()

    def _ensure(self, term: str, dim: int, device, dtype) -> None:
        if term in self.trunks:
            return
        self.trunks[term] = nn.Sequential(
            nn.Linear(dim + self.step_emb_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, dim),
        ).to(device=device, dtype=dtype)
        # 마지막 층을 0 으로 시작 -> 학습 전에는 guidance 없음 = unguided 와 동일한 출발점
        nn.init.zeros_(self.trunks[term][-1].weight)
        nn.init.zeros_(self.trunks[term][-1].bias)
        self.log_stds[term] = nn.Parameter(
            torch.full((), self.init_log_std, device=device, dtype=dtype)
        )
        self.step_emb.to(device=device, dtype=dtype)

    def forward(self, term: str, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, D), k: guided window 내 인덱스 -> (mean (B,N,D), std 스칼라)."""
        b, n, d = x.shape
        self._ensure(term, d, x.device, x.dtype)
        emb = self.step_emb(torch.full((1,), k, device=x.device, dtype=torch.long))
        emb = emb.view(1, 1, -1).expand(b, n, -1)
        raw = self.trunks[term](torch.cat([x, emb], dim=-1))
        mean = torch.tanh(raw) * self.action_scale
        return mean, self.log_stds[term].exp()


class GuidanceCritic(nn.Module):
    """Q(s, a) -> 그 설계가 받을 **배치 내 상대 점수**(advantage) 예측.

    왜 critic 인가: action 차원이 step 당 6,694 (10 step = 66,940) 라 REINFORCE 는
    log-prob 합의 분산에 신호가 묻힌다 (try_1 실측: 30 업데이트 정체). critic 은
    `∇_a Q` 로 각 차원에 직접 credit 을 주므로 이 차원수에서도 학습이 된다.
    동시에 **reward 가 미분 불가능해도 된다** — Q 는 결과로부터 배우는 학습된 함수이므로
    2 단(도달 AND 무낙상)에 그대로 쓸 수 있다.

    타깃은 절대 reward 가 아니라 **한 업데이트(=같은 궤적) 안에서 표준화한 값**이다.
    guided term 토큰에는 어느 궤적인지가 들어 있지 않아 절대값 예측은 불가능하고,
    우리가 필요한 것도 "같은 궤적에서 어느 설계가 나은가" 뿐이다.
    """

    def __init__(
        self,
        term_dims: Mapping[str, int],
        hidden: int = 128,
        step_emb_dim: int = 16,
        guided_steps: int = 10,
    ):
        super().__init__()
        self.terms = list(term_dims)
        self.s_enc = nn.ModuleDict(
            {t: nn.Sequential(nn.Linear(d, hidden), nn.SiLU()) for t, d in term_dims.items()}
        )
        self.a_enc = nn.ModuleDict(
            {t: nn.Sequential(nn.Linear(d, hidden), nn.SiLU()) for t, d in term_dims.items()}
        )
        self.step_emb = nn.Embedding(guided_steps, step_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden * len(self.terms) + step_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        k: torch.Tensor,
    ) -> torch.Tensor:
        """states/actions: term -> (B, N, D), k: (B,) -> (B,)."""
        feats = []
        for t in self.terms:
            feats.append(self.s_enc[t](states[t]).mean(dim=1))   # 토큰 평균 풀링
            feats.append(self.a_enc[t](actions[t]).mean(dim=1))
        feats.append(self.step_emb(k))
        return self.head(torch.cat(feats, dim=-1)).squeeze(-1)


class RLResidualGuidance:
    """`guidance_fn` 인터페이스. 마지막 `guided_steps` 구간에서만 residual 을 낸다.

    모드 3 가지:
      - `stochastic=False`            평가. 정책 평균 그대로.
      - `stochastic=True`             데이터 수집. 평균 + 가우시안 잡음, (s, a) 를 기록.
      - `record_log_prob=True`        (구) REINFORCE 용 log-prob 누적.
    샘플러로 돌려주는 텐서는 **detach** 한다 — 확산 그래프 전체가 미분 대상이 되면 안 된다.
    """

    def __init__(
        self,
        policy: ResidualGuidancePolicy,
        reward_fns: Mapping,
        guidance_terms: Sequence[str] = ("dyna_joint", "fixed_joint", "link", "actuator"),
        num_steps: int = 100,
        guided_steps: int = 10,
        stochastic: bool = False,
        explore_std: float = 0.005,
        record_log_prob: bool = False,
        record_transitions: bool = False,
    ):
        self.policy = policy
        self.reward_fns = reward_fns          # DSG 와 키 집합을 맞추기 위한 형식상 필드
        self.guidance_terms = list(guidance_terms)
        self.num_steps = num_steps
        self.guided_steps = guided_steps
        self.stochastic = stochastic
        self.explore_std = explore_std
        self.record_log_prob = record_log_prob
        self.record_transitions = record_transitions
        self.calls = 0
        self.log_probs: list[torch.Tensor] = []
        self.transitions: list[dict] = []     # [{k, states{term->(B,N,D)}, actions{...}}]

    def reset(self) -> None:
        self.calls = 0
        self.log_probs = []
        self.transitions = []

    def __call__(self, x_t_plus_one, x_t) -> dict[str, torch.Tensor]:
        k = self.calls
        self.calls += 1
        first_guided = self.num_steps - self.guided_steps
        if k < first_guided:
            return {}
        kk = k - first_guided

        out: dict[str, torch.Tensor] = {}
        rec_s: dict[str, torch.Tensor] = {}
        rec_a: dict[str, torch.Tensor] = {}
        for term in self.guidance_terms:
            x = x_t.get(term)
            if x is None or not torch.is_tensor(x) or x.ndim != 3:
                continue
            x = x.detach()
            with torch.no_grad():
                mean, std = self.policy(term, x, kk)
            if self.stochastic:
                sigma = std if self.record_log_prob else self.explore_std
                action = (mean + sigma * torch.randn_like(mean)).detach()
            else:
                action = mean.detach()
            if self.record_log_prob:
                mean_g, std_g = self.policy(term, x, kk)   # grad 가 필요한 경로
                lp = (
                    -0.5 * ((action - mean_g) / std_g) ** 2
                    - torch.log(std_g)
                    - 0.5 * math.log(2 * math.pi)
                )
                self.log_probs.append(lp.flatten(1).sum(dim=1))   # (B,)
            if self.record_transitions:
                rec_s[term] = x
                rec_a[term] = action
            out[term] = action
        if self.record_transitions and rec_s:
            self.transitions.append(dict(k=kk, states=rec_s, actions=rec_a))
        return out

    def episode_log_prob(self) -> torch.Tensor:
        """guided step 전체에 걸친 log-prob 합 -> (B,)."""
        assert self.log_probs, "stochastic=True 로 한 번은 생성해야 한다"
        return torch.stack(self.log_probs, dim=0).sum(dim=0)


class RLResidualGuidedOptimizer(HardwareOptimizer):
    """평가용. 학습된 π_φ 를 얹어 DSG 와 같은 자리에서 돌린다 (결정적 = 평균 사용)."""

    def __init__(
        self,
        policy_ckpt: str,
        guidance_terms: Sequence[str] = ("dyna_joint", "fixed_joint", "link", "actuator"),
        num_steps: int = 100,
        guided_steps: int = 10,
        output_cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(output_cache_dir=output_cache_dir, **kwargs)
        blob = torch.load(policy_ckpt, map_location=self.device, weights_only=False)
        policy: ResidualGuidancePolicy = blob["policy"]
        policy.to(self.device).eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        self.guidance = RLResidualGuidance(
            policy=policy,
            reward_fns=self.reward_fns,
            guidance_terms=guidance_terms,
            num_steps=num_steps,
            guided_steps=guided_steps,
            stochastic=False,
        )
        logging.info(
            f"π_φ 로드: {policy_ckpt} (updates={blob.get('updates')}, "
            f"terms={list(policy.trunks)})"
        )

    def run_hardware_generator(self, batch, seed):
        self.guidance.reset()
        decoded = self.hardware_generator(batch, seed=seed, guidance_fn=self.guidance)
        return {**batch, **decoded}
