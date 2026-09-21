"""π_φ 학습 — critic 방식 (DDPG/SVG 계열). try_1 의 REINFORCE 정체를 푼다.

왜 바꿨나 (try_1 실측): action 차원이 step 당 6,694 · 10 step = **66,940**. REINFORCE 는
그 log-prob 합 하나로 credit 을 배분하므로 표본 16 개로는 신호가 분산에 묻힌다
(30 업데이트 동안 reward 6.52 -> 6.52, loss 부호 무작위).

critic 구조:
    Q_ψ(s_k, a_k) ≈ A            A = 같은 업데이트(같은 궤적) 안에서 표준화한 reward
    정책 손실      = −Q_ψ(s_k, π_φ(s_k))        ← ∇_a Q 가 6,694 차원 각각에 credit 을 준다

에피소드가 10 step 이고 보상이 종단 1 개뿐이라 부트스트랩 없이 **몬테카를로 회귀**로 맞춘다
(§3.4 의 "1 차는 contextual bandit" 과 같은 선택). reward 가 미분 불가능해도 Q 는 학습되므로
**2 단에서 그대로 재사용**된다.

  .venv/bin/python tt_codesign/code/train_pi_phi_critic.py --updates 300 --batch 16
"""

import argparse
import json
import os
import pickle
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

TT = "/mnt/gstore/home/xiangyue/RL/transformer-transformer"
OUT = "/mnt/gstore/home/xiangyue/RL/tt_codesign/runs"
os.chdir(TT)

from t2.eval.rl_guidance import (
    GuidanceCritic,
    RLResidualGuidance,
    ResidualGuidancePolicy,
)
from t2.model.t2 import setup_decoder
from t2.train.augment import AddPositionId
from t2.train.utils import set_torch_configs
from t2.utils.misc import seed_everything

TERMS = ("dyna_joint", "fixed_joint", "link", "actuator")


def build(args, device):
    with initialize_config_dir(config_dir=f"{TT}/config", version_base="1.3"):
        cfg = compose(
            config_name="evaluate_hardware_opt_rl",
            overrides=[
                "hardware_optimizer@eval_fn.hardware_optimizer_fn=zeroth_order",
                "reward_fns@eval_fn.hardware_optimizer_fn.reward_fns=tracking_only",
                f"eval_fn.hardware_optimizer_fn.num_seeds={args.batch}",
                f"eta={args.eta}",
                "use_flash_attn=false",
                "use_mixed_precision=false",
                "use_torch_compile=false",
                f"ckpt_path={args.ckpt}",
                f"pickle_path={args.train_pkl}",
            ],
        )
    policy_cfg = pickle.load(open(os.path.dirname(cfg.ckpt_path) + "/cfg.pkl", "rb"))
    OmegaConf.resolve(policy_cfg.seq_len)
    cfg.timestep_sampler.num_rollout_steps = int(policy_cfg.seq_len["rollout_steps"])
    OmegaConf.resolve(policy_cfg)

    gen = setup_decoder(
        policy_cfg, ckpt_path=cfg.ckpt_path, device=device, task_name=cfg.task_name,
        num_inference_steps=cfg.num_inference_steps,
        num_repeats_per_step=cfg.num_repeats_per_step,
        clip_samples_in_guidance=cfg.clip_samples_in_guidance,
        use_flash_attn=False, eta=cfg.eta,
        use_mixed_precision=False, use_torch_compile=False,
    )
    for p in gen.model.parameters():    # prior 동결 (D11)
        p.requires_grad_(False)

    bpf = hydra.utils.instantiate(policy_cfg.datasets.clean.batch_process_fn)
    add_pos_id = next(a for a in bpf.augmentations if type(a) is AddPositionId)
    ts = hydra.utils.instantiate(cfg.timestep_sampler)
    opt = hydra.utils.instantiate(cfg.eval_fn.hardware_optimizer_fn)(
        hardware_generator=gen, seq_len_cfg=dict(policy_cfg.seq_len),
        timestep_sampler=ts, center_traj=cfg.eval_fn.runner.env.center_traj,
        add_pos_id=add_pos_id, device=device, output_cache_dir=None,
        max_traj_len=policy_cfg.model.pos_embs.time.embedding.num_embeddings - 1,
    )
    return cfg, opt


def generate(opt, guidance, batch, seed):
    """생성 1 회 -> (표본별 reward, transitions). reward 원천 = 모델 예측 (DSG 와 동일)."""
    guidance.reset()
    with torch.no_grad():
        decoded = opt.hardware_generator(batch, seed=seed, guidance_fn=guidance)
        decoded = {**batch, **decoded}
        value = sum(fn(decoded) for fn in opt.reward_fns.values())
    return value.detach().float(), guidance.transitions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--policy-lr", type=float, default=1e-4)
    p.add_argument("--critic-steps", type=int, default=20, help="업데이트당 critic 경사 수")
    p.add_argument("--policy-steps", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=30, help="보관할 업데이트 수")
    p.add_argument("--explore-std", type=float, default=0.005)
    p.add_argument("--action-scale", type=float, default=0.02)
    p.add_argument("--guided-steps", type=int, default=10)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--ckpt", default="checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt")
    p.add_argument("--train-pkl", default="data/july25th2025-huy-20skills-train.pkl")
    p.add_argument("--name", default="pi_phi_try2")
    args = p.parse_args()

    set_torch_configs()
    seed_everything(args.seed)
    device = torch.device("cuda")
    cfg, opt = build(args, device)
    trajs = pickle.load(open(args.train_pkl, "rb"))
    rs = np.random.RandomState(args.seed)
    os.makedirs(f"{OUT}/{args.name}", exist_ok=True)

    policy = ResidualGuidancePolicy(
        guided_steps=args.guided_steps, action_scale=args.action_scale
    ).to(device)
    guidance = RLResidualGuidance(
        policy=policy, reward_fns=opt.reward_fns, guidance_terms=TERMS,
        num_steps=100, guided_steps=args.guided_steps,
        stochastic=True, explore_std=args.explore_std,
        record_log_prob=False, record_transitions=True,
    )

    # 워밍업 1 회로 term 차원을 알아내고 critic 을 만든다
    warm = opt._prepare_batch([trajs[0]])
    warm = {k: v.repeat(args.batch, *([1] * (v.ndim - 1))) for k, v in warm.items()}
    _, tr = generate(opt, guidance, warm, seed=0)
    term_dims = {t: v.shape[-1] for t, v in tr[0]["states"].items()}
    critic = GuidanceCritic(term_dims, guided_steps=args.guided_steps).to(device)
    print(f"학습 궤적 {len(trajs)} | reward {list(opt.reward_fns)} | term_dims {term_dims}")
    print(f"π_φ {sum(x.numel() for x in policy.parameters()):,} 파라미터 | "
          f"Q_ψ {sum(x.numel() for x in critic.parameters()):,} 파라미터", flush=True)

    opt_c = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)

    buf: list[dict] = []          # 항목 = {k, states, actions, target} (CPU 보관)
    hist = []
    t0 = time.time()

    for it in range(1, args.updates + 1):
        ti = int(rs.randint(len(trajs)))
        batch = opt._prepare_batch([trajs[ti]])
        batch = {k: v.repeat(args.batch, *([1] * (v.ndim - 1))) for k, v in batch.items()}
        seed = int(rs.randint(0, np.iinfo(np.int32).max))

        r, trans = generate(opt, guidance, batch, seed=seed)      # (B,)
        # 같은 궤적 안에서 표준화 -> critic 은 "어느 설계가 나은가" 만 배운다
        target = (r - r.mean()) / (r.std() + 1e-6)
        for tr_k in trans:
            buf.append(dict(
                k=tr_k["k"],
                states={t: v.to("cpu") for t, v in tr_k["states"].items()},
                actions={t: v.to("cpu") for t, v in tr_k["actions"].items()},
                target=target.to("cpu"),
            ))
        max_items = args.buffer * args.guided_steps
        if len(buf) > max_items:
            buf = buf[-max_items:]

        def sample_mb():
            item = buf[int(rs.randint(len(buf)))]
            idx = torch.from_numpy(
                rs.choice(args.batch, size=min(args.minibatch, args.batch), replace=False)
            )
            s = {t: v[idx].to(device) for t, v in item["states"].items()}
            a = {t: v[idx].to(device) for t, v in item["actions"].items()}
            y = item["target"][idx].to(device)
            kk = torch.full((len(idx),), item["k"], device=device, dtype=torch.long)
            return s, a, y, kk

        c_loss_v = p_loss_v = 0.0
        for _ in range(args.critic_steps):
            s, a, y, kk = sample_mb()
            q = critic(s, a, kk)
            c_loss = torch.nn.functional.mse_loss(q, y)
            opt_c.zero_grad(set_to_none=True)
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            opt_c.step()
            c_loss_v = float(c_loss.detach())

        for _ in range(args.policy_steps):
            s, _a, _y, kk = sample_mb()
            act = {}
            for t, v in s.items():
                mean, _ = policy(t, v, int(kk[0]))
                act[t] = mean
            q = critic(s, act, kk)
            p_loss = -q.mean()
            opt_p.zero_grad(set_to_none=True)
            p_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_p.step()
            p_loss_v = float(p_loss.detach())

        with torch.no_grad():
            amag = float(torch.stack([v.abs().mean() for v in trans[-1]["actions"].values()]).mean())
        hist.append(dict(it=it, traj=ti, reward_mean=float(r.mean()), reward_max=float(r.max()),
                         critic_loss=c_loss_v, policy_q=-p_loss_v, action_mag=amag))
        if it % 5 == 0 or it == 1:
            print(f"[{it:4d}/{args.updates}] rew {r.mean():.3f} (max {r.max():.3f}) "
                  f"Qloss {c_loss_v:.4f} Q(π) {-p_loss_v:+.4f} |a| {amag:.2e} "
                  f"buf {len(buf)} {time.time() - t0:.0f}s", flush=True)

        if it % args.save_every == 0 or it == args.updates:
            torch.save(dict(policy=policy, critic=critic, updates=it, args=vars(args)),
                       f"{OUT}/{args.name}/policy.pt")
            with open(f"{OUT}/{args.name}/history.json", "w") as f:
                json.dump(hist, f)

    print(f"완료 {time.time() - t0:.0f}s -> {OUT}/{args.name}/policy.pt")


if __name__ == "__main__":
    main()
