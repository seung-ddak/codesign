"""guidance 의 효과 크기를 쌍대(paired) 로 잰다 — "치환" 을 시험할 지점을 찾기 위해.

논문 Table 4 는 quadruped/single 에서 Zeroth 와 DGS 가 동일하고, §3 은 "표본이 많으면
둘은 구분 불가" 라고 명시한다. 효과가 0 인 지점에서 치환을 시험하면 아무것도 말할 수 없다.

같은 궤적·**같은 샘플러 시드**로 초기 노이즈를 고정하고 guidance 만 바꿔 예측 reward 를 비교한다.
쌍대라 궤적·노이즈 분산이 상쇄되어 작은 효과도 잡힌다.

  .venv/bin/python tt_codesign/code/guidance_effect.py --n-traj 20 --n-seed 3 --batch 1
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
OUT = _os.path.join(_R, "results")
os.chdir(TT)

from t2.eval.rl_guidance import RLResidualGuidance
from t2.model.t2 import setup_decoder
from t2.train.augment import AddPositionId
from t2.train.utils import set_torch_configs
from t2.utils.misc import seed_everything

TERMS = ("dyna_joint", "fixed_joint", "link", "actuator")


def build(args, device, optimizer_name, extra=()):
    with initialize_config_dir(config_dir=f"{TT}/config", version_base="1.3"):
        cfg = compose(
            config_name="evaluate_hardware_opt_rl",
            overrides=[
                f"hardware_optimizer@eval_fn.hardware_optimizer_fn={optimizer_name}",
                "reward_fns@eval_fn.hardware_optimizer_fn.reward_fns=tracking_only",
                f"eval_fn.hardware_optimizer_fn.num_seeds={args.batch}",
                f"eta={args.eta}",
                "use_flash_attn=false",
                "use_mixed_precision=false",
                "use_torch_compile=false",
                f"ckpt_path={args.ckpt}",
                f"pickle_path={args.pkl}",
                *extra,
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-traj", type=int, default=20)
    p.add_argument("--n-seed", type=int, default=3)
    p.add_argument("--batch", type=int, default=1, help="n (후보 표본 수). 1 = search 없음")
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--scale", type=float, default=100.0)
    p.add_argument("--pi-phi",
                   default=f"{_R}/runs/pi_phi_try2/policy.pt")
    p.add_argument("--ckpt", default="checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt")
    p.add_argument("--pkl", default="data/july25th2025-huy-20skills-test.pkl")
    p.add_argument("--out-name", default="guidance_effect")
    args = p.parse_args()

    set_torch_configs()
    seed_everything(0)
    device = torch.device("cuda")

    # DSG optimizer (guidance 객체를 쓰기 위해). 모델은 하나만 올리고 guidance 만 갈아 끼운다.
    _, opt = build(args, device, "guided_diffusion",
                   [f"eval_fn.hardware_optimizer_fn.guidance.scale={args.scale}"])
    dsg = opt.guidance

    arms = {"unguided": None, "dsg": dsg}
    if os.path.exists(args.pi_phi):
        blob = torch.load(args.pi_phi, map_location=device, weights_only=False)
        pol = blob["policy"].to(device).eval()
        for q in pol.parameters():
            q.requires_grad_(False)
        arms["pi_phi"] = RLResidualGuidance(
            policy=pol, reward_fns=opt.reward_fns, guidance_terms=TERMS,
            num_steps=100, guided_steps=10, stochastic=False,
        )
        print(f"π_φ 로드 (updates={blob.get('updates')})")

    trajs = pickle.load(open(args.pkl, "rb"))
    rows = []
    t0 = time.time()
    for ti in range(args.n_traj):
        batch = opt._prepare_batch([trajs[ti]])
        batch = {k: v.repeat(args.batch, *([1] * (v.ndim - 1))) for k, v in batch.items()}
        for si in range(args.n_seed):
            seed = 1000 * (ti + 1) + si          # 팔 전체가 공유 = 쌍대 비교
            rec = dict(traj=ti, seed=seed)
            for name, g in arms.items():
                if g is not None and hasattr(g, "reset"):
                    g.reset()
                decoded = opt.hardware_generator(batch, seed=seed, guidance_fn=g)
                decoded = {**batch, **decoded}
                with torch.no_grad():
                    v = sum(fn(decoded) for fn in opt.reward_fns.values())
                rec[name] = float(v.max())        # n 개 중 최고 = optimizer 가 고르는 값
            rows.append(rec)
        print(f"  traj {ti + 1}/{args.n_traj}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{args.out_name}_n{args.batch}.json", "w") as f:
        json.dump(dict(args=vars(args), rows=rows), f, indent=2)

    print(f"\n=== 쌍대 비교 (n={args.batch}, 쌍 {len(rows)} 개) ===")
    base = np.array([r["unguided"] for r in rows])
    print(f"  unguided        {base.mean():.4f} ± {base.std():.4f}")
    for name in arms:
        if name == "unguided":
            continue
        x = np.array([r[name] for r in rows])
        d = x - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {name:15s} {x.mean():.4f} ± {x.std():.4f} | "
              f"쌍대차 {d.mean():+.4f} ± {se:.4f} (SE) | "
              f"t={d.mean() / (se + 1e-12):+.2f} | 이김 {100 * (d > 0).mean():.0f}%")


if __name__ == "__main__":
    main()
