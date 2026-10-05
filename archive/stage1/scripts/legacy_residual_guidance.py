"""폐기된 고정 TT + residual guidance 구현. D58 DDPO 전환 전 기록용."""

import pickle
from pathlib import Path
import tempfile

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import t2.eval

# Stage 2 T7처럼 이 모듈을 직접 불러오는 경로에서도 로컬 RL guidance 패치를 찾게 한다.
PATCH_EVAL = Path(__file__).resolve().parents[2] / "tt_patch/t2/eval"
if str(PATCH_EVAL) not in t2.eval.__path__:
    t2.eval.__path__.append(str(PATCH_EVAL))

from t2.eval.hardware_decoding import check_zarr_path, hardware_to_nearest_neighbor_params
from t2.eval.rl_guidance import GuidanceCritic, RLResidualGuidance, ResidualGuidancePolicy
from t2.eval.utils import extract_masked_hardware, traj2batch
from t2.model.t2 import setup_decoder
from t2.train.augment import AddPositionId
from t2.utils.misc import seed_everything


class DesignLearner:
    def __init__(self, tt, checkpoint, bank, batch_size, device, seed,
                 denoise_steps=100, guided_steps=10):
        # 초기화 실패 진단: checkpoint 옆 cfg.pkl, 8비트 이름의 *.zarr bank,
        # CUDA 사용 가능 여부를 먼저 확인한다. TT prior 자체는 T7 동안 갱신하지 않는다.
        seed_everything(seed)
        self.device, self.batch_size = device, batch_size
        self._bank_view = tempfile.TemporaryDirectory(prefix="coevolution_bank_")
        self.bank = self._bank_view.name
        for path in bank.glob("*.zarr"):
            if len(path.stem) == 8 and set(path.stem) <= {"0", "1"}:
                (Path(self.bank) / path.name).symlink_to(path.resolve(), target_is_directory=True)
        with checkpoint.with_name("cfg.pkl").open("rb") as f:
            cfg = pickle.load(f)
        OmegaConf.resolve(cfg.seq_len)
        self.decoder = setup_decoder(
            cfg, str(checkpoint), torch.device(device), "hardware_gen",
            num_inference_steps=denoise_steps, eta=1.0,
            use_flash_attn=False, use_mixed_precision=False, use_torch_compile=False)
        self.policy = ResidualGuidancePolicy(guided_steps=guided_steps).to(device)
        self.guidance = RLResidualGuidance(
            self.policy, {}, num_steps=denoise_steps, guided_steps=guided_steps,
            stochastic=True, record_transitions=True)
        with initialize_config_dir(config_dir=str(tt / "config"), version_base="1.3"):
            inference = compose(config_name="evaluate_hardware_opt_rl")
        inference.timestep_sampler.num_rollout_steps = int(cfg.seq_len.rollout_steps)
        sampler = hydra.utils.instantiate(inference.timestep_sampler)
        processor = hydra.utils.instantiate(cfg.datasets.clean.batch_process_fn)
        add_ids = next(x for x in processor.augmentations if isinstance(x, AddPositionId))
        # Shape-only placeholders: all goal tokens are masked from attention.
        poses = np.tile(np.eye(4, dtype=np.float32),
                        (int(cfg.seq_len.rollout_steps), int(cfg.seq_len.track_link), 1, 1))
        batch = add_ids(traj2batch(poses, cfg.seq_len, sampler, center_traj=False))
        batch["hardware_goal/mask"] = torch.ones_like(batch["target_pose/mask"])
        batch["target_pose/mask"] = torch.ones_like(batch["target_pose/mask"])
        self.batch = {k: v.repeat(batch_size, *([1] * (v.ndim - 1))).to(device)
                      for k, v in batch.items()}
        self.critic = self.actor_opt = self.critic_opt = None
        self.guided_steps = guided_steps

    def generate(self, seed):
        torch.manual_seed(seed)
        self.guidance.reset()
        with torch.no_grad():
            decoded = self.decoder(self.batch, seed=seed, guidance_fn=self.guidance)
        transitions = self.guidance.transitions
        # 개수가 다르면 denoising 설정과 guided_steps가 서로 맞지 않는 것이다.
        # 이 경우 값을 억지로 맞추지 말고 RLResidualGuidance 호출 설정부터 확인한다.
        if len(transitions) != self.guided_steps:
            raise RuntimeError("Denoiser did not produce the expected guidance transitions")
        if self.critic is None:
            dims = {k: v.shape[-1] for k, v in transitions[0]["states"].items()}
            self.critic = GuidanceCritic(dims, guided_steps=self.guided_steps).to(self.device)
            self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        designs = []
        for i in range(self.batch_size):
            item = {k: v[i].detach().cpu().numpy() for k, v in decoded.items()
                    if k.split("/")[0] in ("link", "actuator", "dyna_joint", "fixed_joint")}
            try:
                hardware, *counts = extract_masked_hardware(item)
                choices, hardware_seed = hardware_to_nearest_neighbor_params(
                    hardware, *counts, self.bank, local_only=True)
                match = check_zarr_path(hardware, *counts,
                                       Path(self.bank) / ("".join(map(str, choices)) + ".zarr"))
                if match is None or not np.isfinite(match[1]):
                    designs.append({"invalid": "No finite-distance hardware match"})
                else:
                    designs.append({"choices": choices, "seed": int(hardware_seed)})
            except ValueError as exc:
                # invalid가 많으면 여기의 사유와 self.bank의 archetype 수를 확인한다.
                # No matching archetype은 평가 실패가 아니라 생성 형상이 bank 밖이라는 뜻이다.
                if str(exc) not in ("No matching archetype found", "No best seed found"):
                    raise
                designs.append({"invalid": str(exc)})
        return designs, transitions

    def update(self, transitions, returns, critic_steps=40, actor_steps=5):
        rewards = torch.as_tensor(returns, device=self.device, dtype=torch.float32)
        # NaN/Inf는 πφ 문제가 아니라 앞 단계 rollout 반환값 문제이므로 DesignScorer부터 진단한다.
        if not torch.isfinite(rewards).all():
            raise ValueError("Non-finite rollout return")
        spread = rewards.std(unbiased=False)
        # 모든 return이 같으면 정규화와 정책경사가 정의되지 않는다. 정상적인 안전 중단이며,
        # 형상 다양성·평가 명령·πθ의 형상 민감도를 확인한 뒤 다음 update 여부를 판단한다.
        if spread < 1e-6:
            return {"phi_updated": False, "reason": "equal rollout returns"}
        target = (rewards - rewards.mean()) / spread
        before = torch.cat([p.detach().flatten().clone() for p in self.policy.parameters()])
        # Old rounds use obsolete controllers; fit only this round's observations.
        for _ in range(critic_steps):
            tr = transitions[int(torch.randint(len(transitions), ()).item())]
            kk = torch.full((len(rewards),), tr["k"], device=self.device, dtype=torch.long)
            loss = torch.nn.functional.mse_loss(self.critic(tr["states"], tr["actions"], kk), target)
            self.critic_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.)
            self.critic_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        for _ in range(actor_steps):
            tr = transitions[int(torch.randint(len(transitions), ()).item())]
            kk = torch.full((len(rewards),), tr["k"], device=self.device, dtype=torch.long)
            actions = {t: self.policy(t, s, tr["k"])[0] for t, s in tr["states"].items()}
            actor_loss = -self.critic(tr["states"], actions, kk).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
            self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)
        after = torch.cat([p.detach().flatten() for p in self.policy.parameters()])
        # loss가 폭증하거나 delta가 계속 0이면 critic/actor 학습률, reward 분산,
        # gradient clipping 순서로 확인한다. TT decoder 가중치는 수정 대상이 아니다.
        return {"phi_updated": True, "critic_loss": float(loss.detach()),
                "actor_loss": float(actor_loss.detach()),
                "phi_parameter_delta": float(torch.linalg.vector_norm(after - before))}

    def save(self, path):
        torch.save({"policy": self.policy, "critic": self.critic,
                    "actor_optimizer": self.actor_opt.state_dict(),
                    "critic_optimizer": self.critic_opt.state_dict()}, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.policy = state["policy"].to(self.device)
        self.critic = state["critic"].to(self.device)
        self.guidance.policy = self.policy
        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.actor_opt.load_state_dict(state["actor_optimizer"])
        self.critic_opt.load_state_dict(state["critic_optimizer"])
