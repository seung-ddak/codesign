"""π_φ (residual guidance) 학습 — REINFORCE + 이동평균 baseline, prior 동결 (D11).

1 단 동등성 실험용이다. DSG 와 **같은 reward 함수**(`tracking_only` = regularized_tracking_error)
를 쓰되, DSG 가 그 reward 를 미분해 guidance 를 만드는 자리에서 π_φ 는 표집·보상으로 학습한다.

  학습 궤적 : `data/july25th2025-huy-20skills-train.pkl` (56 개)
  평가 궤적 : test.pkl 20 개 — 학습에 쓰지 않는다 (zero-shot)

  .venv/bin/python tt_codesign/code/train_pi_phi.py --updates 300 --batch 16
"""

import os as _os
_R = _os.environ.get("CODESIGN_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_TT = _os.environ.get("CODESIGN_TT", _os.path.join(_os.path.dirname(_R), "transformer-transformer"))

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

TT = _TT
OUT = _os.path.join(_R, "runs")
os.chdir(TT)

from t2.eval.rl_guidance import RLResidualGuidance, ResidualGuidancePolicy
from t2.model.t2 import setup_decoder
from t2.train.augment import AddPositionId
from t2.train.utils import set_torch_configs
from t2.utils.misc import seed_everything


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


def rollout(opt, guidance, batch, seed, stochastic):
    """한 번 생성하고 표본별 reward 를 돌려준다. reward 원천 = 모델 예측 (DSG 와 동일)."""
    guidance.reset()
    guidance.stochastic = stochastic
    decoded = opt.hardware_generator(batch, seed=seed, guidance_fn=guidance)
    decoded = {**batch, **decoded}
    with torch.no_grad():
        value = sum(fn(decoded) for fn in opt.reward_fns.values())
    return value.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--batch", type=int, default=16, help="업데이트당 표본 설계 수")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--action-scale", type=float, default=0.02)
    p.add_argument("--init-std", type=float, default=0.005)
    p.add_argument("--guided-steps", type=int, default=10)
    p.add_argument("--baseline-momentum", type=float, default=0.9)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--ckpt", default="checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt")
    p.add_argument("--train-pkl", default="data/july25th2025-huy-20skills-train.pkl")
    p.add_argument("--name", default="pi_phi_try1")
    args = p.parse_args()

    set_torch_configs()
    seed_everything(args.seed)
    device = torch.device("cuda")
    cfg, opt = build(args, device)

    trajs = pickle.load(open(args.train_pkl, "rb"))
    print(f"학습 궤적 {len(trajs)} 개 | reward {list(opt.reward_fns)} | batch {args.batch}",
          flush=True)

    policy = ResidualGuidancePolicy(
        guided_steps=args.guided_steps,
        action_scale=args.action_scale,
        init_log_std=float(np.log(args.init_std)),
    ).to(device)
    guidance = RLResidualGuidance(
        policy=policy, reward_fns=opt.reward_fns,
        num_steps=100, guided_steps=args.guided_steps, stochastic=True,
    )

    # 지연 생성이므로 한 번 굴려 파라미터를 만든 뒤 optimizer 를 붙인다
    rs = np.random.RandomState(args.seed)
    warm_batch = opt._prepare_batch([trajs[0]])
    warm_batch = {k: v.repeat(args.batch, *([1] * (v.ndim - 1)))
                  for k, v in warm_batch.items()}
    rollout(opt, guidance, warm_batch, seed=0, stochastic=True)
    n_par = sum(p.numel() for p in policy.parameters())
    print(f"π_φ 파라미터 {n_par:,} | terms {list(policy.trunks)}", flush=True)

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    baseline = None
    hist = []
    os.makedirs(f"{OUT}/{args.name}", exist_ok=True)
    t0 = time.time()

    for it in range(1, args.updates + 1):
        ti = int(rs.randint(len(trajs)))
        batch = opt._prepare_batch([trajs[ti]])
        batch = {k: v.repeat(args.batch, *([1] * (v.ndim - 1)))
                 for k, v in batch.items()}
        seed = int(rs.randint(0, np.iinfo(np.int32).max))

        value = rollout(opt, guidance, batch, seed=seed, stochastic=True)  # (B,)
        logp = guidance.episode_log_prob()                                  # (B,)
        assert logp.shape == value.shape, (logp.shape, value.shape)

        r = value.float()
        # baseline 은 **배치 내 평균**을 쓴다. 한 업데이트의 B 표본은 모두 같은 궤적이므로,
        # 궤적 간 난이도 차이가 advantage 에 섞이지 않는다. 이동평균은 로그용으로만 둔다.
        baseline = r.mean() if baseline is None else (
            args.baseline_momentum * baseline + (1 - args.baseline_momentum) * r.mean()
        )
        adv = r - r.mean()
        adv = adv / (adv.std() + 1e-6)
        loss = -(logp * adv).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()

        hist.append(dict(it=it, traj=ti, reward_mean=float(r.mean()),
                         reward_max=float(r.max()), baseline=float(baseline),
                         loss=float(loss.detach()), grad_norm=float(gn),
                         std=float(list(policy.log_stds.values())[0].exp())))
        if it % 5 == 0 or it == 1:
            print(f"[{it:4d}/{args.updates}] rew {r.mean():.3f} (max {r.max():.3f}) "
                  f"base {baseline:.3f} loss {loss:+.4f} |g| {gn:.3f} "
                  f"std {hist[-1]['std']:.4f} {time.time() - t0:.0f}s", flush=True)

        if it % args.eval_every == 0 or it == args.updates:
            torch.save(dict(policy=policy, updates=it, args=vars(args)),
                       f"{OUT}/{args.name}/policy.pt")
            with open(f"{OUT}/{args.name}/history.json", "w") as f:
                json.dump(hist, f)

    print(f"완료 {time.time() - t0:.0f}s -> {OUT}/{args.name}/policy.pt")


if __name__ == "__main__":
    main()
