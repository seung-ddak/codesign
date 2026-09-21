"""TT ctrl 정책을 ray 없이 in-process 로 굴린다.

TT 의 평가 파이프라인(`evaluate_ctrl.py`)은 ray actor + policy server 구조라
에피소드당 약 200 초가 걸리고 계단 env 에서는 교착에 빠진다. 물리 자체는 762 step/s
(2000 step 에피소드 = 3 초)이므로 병목은 전적으로 그 파이프라인이다.

여기서는 같은 정책·같은 obs 변환을 쓰되 큐와 actor 를 걷어내고, **베이스 위치를
매 step 기록**한다 — 우리 reward(도달 AND 무낙상)가 필요로 하는 값이다.

  python scripts/ctrl_rollout_direct.py --env <env config> --pkl <궤적> --traj 0
"""

import argparse
import pickle
import time

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from t2.data.dataset import TimestepSampler  # noqa: F401  (hydra 인스턴스화용)
from t2.eval.ctrl import env_obs_to_batch
from t2.model.t2 import setup_decoder
from t2.train.augment import AddPositionId

CFG_DIR = "/mnt/gstore/home/xiangyue/RL/transformer-transformer/config"


def build(args, num_inference_steps=None, half=False):
    """num_inference_steps 를 낮추면 denoising 횟수가 줄어 그만큼 빨라진다."""
    policy_cfg = pickle.load(open(args.ckpt.rsplit("/", 1)[0] + "/cfg.pkl", "rb"))
    OmegaConf.resolve(policy_cfg.ctrl_seq_len)
    seq_len_cfg = OmegaConf.to_container(policy_cfg.ctrl_seq_len, resolve=True)
    group_time_offsets = OmegaConf.to_container(
        policy_cfg.datasets.ctrl_rand.dataset.group_time_offsets, resolve=True
    )

    overrides = [
        "ckpt_path=x",
        f"env@eval_fn.runner.env={args.env}",
        "eval_fn.runner.env.robot_generator.choices=[0,0,0,0,0,0,0,0]",
        "eval_fn.runner.env.robot_generator.num_uniforms=13",
        f"eval_fn.runner.env.pickle_path={args.pkl}",
    ]
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        cfg = compose(config_name="evaluate_ctrl", overrides=overrides)
    env = hydra.utils.instantiate(cfg.eval_fn.runner.env)

    batch_process_fn = hydra.utils.instantiate(
        policy_cfg.datasets.ctrl_rand.batch_process_fn
    )
    batch_process_fn.augmentations = [
        a for a in batch_process_fn.augmentations if type(a) is AddPositionId
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = setup_decoder(
        policy_cfg,
        ckpt_path=args.ckpt,
        device=device,
        task_name="ctrl",
        num_inference_steps=num_inference_steps,
        num_repeats_per_step=1,
        clip_samples_in_guidance=True,
        deterministic=False,
        use_flash_attn=False,      # V100 은 sm_80+ 전용 flash kernel 이 없다
        use_torch_compile=False,
        use_mixed_precision=False,  # V100 은 bfloat16 미지원
    )
    return env, decoder, device, seq_len_cfg, group_time_offsets, batch_process_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/umi_on_legs_plus_plus/d7lra7j0/040.pt")
    p.add_argument("--env", default="umi_on_legs_plus_plus_stairs")
    p.add_argument("--pkl", default="data/stair_walk_r05.pkl")
    p.add_argument("--traj", type=int, default=0)
    p.add_argument("--seed", type=int, default=10000000)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--inference-steps", type=int, default=None,
                   help="denoising 횟수 (기본 5). 낮추면 빨라진다")
    args = p.parse_args()

    env, decoder, device, seq_len_cfg, offsets, batch_process_fn = build(
        args, num_inference_steps=args.inference_steps
    )
    rollout_steps = seq_len_cfg["rollout_steps"]
    env.obs_time_indices = np.arange(rollout_steps) + offsets.get("target_pose", 0)
    env.include_obs_time_indices = "relative"

    env.reset_hardware(args.seed)
    obs, info, hardware = env.reset_sim_and_target_traj(
        seed=args.seed, target_traj_idx=args.traj
    )
    prev_action = [0.0] * env.d.ctrl.shape[0]
    num_past = -offsets.get("ctrl", 0)

    log = []
    t0 = time.time()
    for step in range(args.steps):
        reshaped = {}
        for k, v in obs.items():
            if getattr(v, "ndim", 0) == 2:
                if k.split("/")[0] == "target_pose":
                    v = v.reshape(-1, env.n_end_effectors, v.shape[-1])
                else:
                    v = v[None, :, :]
            reshaped[k] = v

        batch = env_obs_to_batch(
            obs_list=[{**reshaped, **hardware}],
            prev_action_list=[np.array(prev_action)],
            seq_len_cfg=seq_len_cfg,
            device=device,
            group_time_offsets=offsets,
            batch_process_fn=batch_process_fn,
        )
        with torch.inference_mode():
            out = decoder(batch)
        actions = (
            out["ctrl/target_qpos"]
            .cpu()
            .numpy()
            .reshape(1, rollout_steps, seq_len_cfg["actuator"])
        )
        # 정책 출력은 seq_len(=19) 로 패딩돼 있다. 실제 actuator 수로 자른다.
        action = actions[0, num_past][: env.d.ctrl.shape[0]]
        prev_action = action.tolist()

        obs, _, done, info = env.step(action)
        log.append(
            dict(
                step=step,
                x=float(env.d.qpos[0]),
                z=float(env.d.qpos[2]),
                pos_err=float(np.max(env.curr_pos_err)),
                ncon=int(env.d.ncon),
            )
        )
        if done:
            break

    dt = time.time() - t0
    xs = [r["x"] for r in log]
    zs = [r["z"] for r in log]
    errs = [r["pos_err"] for r in log]
    print(
        f"\n=== {args.env} / {args.pkl} traj {args.traj} ===\n"
        f"steps {len(log)} / {args.steps}  ({dt:.1f}s, {len(log)/dt:.1f} step/s)\n"
        f"base x : {xs[0]:+.3f} -> {xs[-1]:+.3f}   (이동 {xs[-1]-xs[0]:+.3f} m, "
        f"최대 {max(xs):+.3f})\n"
        f"base z : {zs[0]:+.3f} -> {zs[-1]:+.3f}   (상승 {zs[-1]-zs[0]:+.3f} m, "
        f"최대 {max(zs):+.3f})\n"
        f"pos_err: 평균 {np.mean(errs):.3f}  최대 {np.max(errs):.3f}\n"
        f"종료   : {'조기 종료' if len(log) < args.steps else '완주'}"
    )


if __name__ == "__main__":
    main()
