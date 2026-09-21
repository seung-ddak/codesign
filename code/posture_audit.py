"""기존 TT ctrl 정책이 "걷는지" 감사한다 — 설계 탓인지 정책 탓인지 가른다.

L0~L3 의 성공 판정은 `distance = max_x - start_x >= goal` 뿐이라 **자세를 보지 않는다**.
낙상 종료도 베이스가 60° 넘게 기울 때만 발동하므로 (umi_on_legs.yaml 0.5),
배를 깔고 미끄러지는 해는 걸러지지 않는다.

그래서 같은 정책·같은 지형에서 설계만 바꿔 가며 **자세**를 직접 잰다:

  center  : uniforms 전부 0.5 (설계공간 중앙)
  prior   : prior 무작위 샘플 (= L0 와 같은 분포)
  best    : L0~L3 각 단의 best_design

  .venv/bin/python tt_codesign/code/posture_audit.py [--n-prior 12]
"""

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import mujoco

TT = "/mnt/gstore/home/xiangyue/RL/transformer-transformer"
OUT = "/mnt/gstore/home/xiangyue/RL/tt_codesign/results"
os.chdir(TT)

from t2.eval.batched_ctrl import BatchedCtrlRollout
from t2.eval.ctrl import env_obs_to_batch

CKPT = "checkpoints/umi_on_legs_plus_plus/d7lra7j0/040.pt"
TRAJ, SEED, STEPS, GOAL = 1, 10000000, 600, 0.8


def posture(env, start_x):
    """(베이스 높이, 기립도, 발 아닌 링크의 바닥 접촉 수)."""
    m, d = env.m, env.d
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    body_ground = 0
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if m.geom_type[g1] == mujoco.mjtGeom.mjGEOM_PLANE:
            g = g2
        elif m.geom_type[g2] == mujoco.mjtGeom.mjGEOM_PLANE:
            g = g1
        else:
            continue
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if "foot" not in bn:
            body_ground += 1
    return (float(d.qpos[2]),
            float(d.xmat[bid].reshape(3, 3)[2, 2]),
            body_ground,
            float(d.qpos[0] - start_x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-prior", type=int, default=12)
    p.add_argument("--designs", default="",
                   help="쉼표 구분 라벨로 직접 지정 (예: prior_2 를 8 번 = 같은 설계 반복 분산 측정)")
    p.add_argument("--out-name", default="posture_audit")
    p.add_argument("--steps", type=int, default=STEPS)
    args = p.parse_args()

    rows = {json.loads(l)["stage"]: json.loads(l)
            for l in open(f"{TT}/design_search_walk.jsonl")}
    rng = np.random.default_rng(SEED)
    pool = {"center": np.full(13, 0.5)}
    for k in range(64):
        pool[f"prior_{k}"] = rng.uniform(0, 1, 13)
    for st in ["L0", "L1", "L2", "L3"]:
        pool[f"{st}_best"] = np.array(rows[st]["best_design"])
    if args.designs:
        names = [x.strip() for x in args.designs.split(",")]
        labels = [f"{n}#{i}" for i, n in enumerate(names)]
        designs = np.array([pool[n] for n in names])
    else:
        labels = ["center"] + [f"prior_{k}" for k in range(args.n_prior)] \
                 + [f"{st}_best" for st in ("L0", "L1", "L2", "L3")]
        designs = np.array([pool[l] for l in labels])
    n = len(designs)

    t0 = time.time()
    R = BatchedCtrlRollout(ckpt=CKPT, env_name="umi_on_legs_plus_plus_from_choice",
                           pkl="data/flat_walk_long.pkl", n_envs=n,
                           inference_steps=2, episode_steps=args.steps)
    print(f"평가기 준비 {time.time() - t0:.1f}s  설계 {n} 개", flush=True)

    envs = R.envs[:n]
    obs_list, hw_list = [], []
    for env, design in zip(envs, designs):
        env.robot_generator = R._design_generator(design)
        env.reset_hardware(SEED)
        obs, _, hw = env.reset_sim_and_target_traj(seed=SEED, target_traj_idx=TRAJ)
        obs_list.append(obs)
        hw_list.append(hw)

    start_x = np.array([float(e.d.qpos[0]) for e in envs])
    # 리셋 직후 값 = 기준선. 이게 1 에 가깝지 않으면 베이스 로컬 z 가 "위"가 아니라는 뜻이라
    # upright 지표 자체를 못 믿는다. 반드시 같이 본다.
    reset0 = [posture(e, sx) for e, sx in zip(envs, start_x)]
    prev_actions = [np.zeros(e.d.ctrl.shape[0]) for e in envs]
    alive = np.ones(n, dtype=bool)
    max_x = start_x.copy()
    fell_at = [None] * n
    diag = [[] for _ in range(n)]
    pos_err = [[] for _ in range(n)]

    t0 = time.time()
    for step in range(args.steps):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        batch = env_obs_to_batch(
            obs_list=[{**R._reshape_obs(envs[i], obs_list[i]), **hw_list[i]} for i in idx],
            prev_action_list=[prev_actions[i] for i in idx],
            seq_len_cfg=R.seq_len_cfg, device=R.device,
            group_time_offsets=R.offsets, batch_process_fn=R.batch_process_fn)
        with torch.inference_mode():
            out = R.decoder(batch)
        actions = (out["ctrl/target_qpos"].float().cpu().numpy()
                   .reshape(len(idx), R.rollout_steps, R.seq_len_cfg["actuator"]))
        for j, i in enumerate(idx):
            env = envs[i]
            a = actions[j, R.num_past][: env.d.ctrl.shape[0]]
            prev_actions[i] = a
            obs_list[i], _, done, _ = env.step(a)
            max_x[i] = max(max_x[i], float(env.d.qpos[0]))
            diag[i].append(posture(env, start_x[i]))
            pos_err[i].append(float(np.max(env.curr_pos_err)))
            if done:
                alive[i] = False
                fell_at[i] = step + 1
        if (step + 1) % 200 == 0:
            print(f"  step {step + 1}/{args.steps}  {time.time() - t0:.0f}s", flush=True)

    res = {}
    for i, lab in enumerate(labels):
        a = np.array(diag[i])
        half = a[len(a) // 2:]
        res[lab] = dict(
            distance=round(float(max_x[i] - start_x[i]), 4),
            success=bool(max_x[i] - start_x[i] >= GOAL),
            fell_at_step=fell_at[i],
            base_z_reset=round(float(reset0[i][0]), 4),
            upright_reset=round(float(reset0[i][1]), 4),
            base_z_mean=round(float(half[:, 0].mean()), 4),
            base_z_min=round(float(a[:, 0].min()), 4),
            upright_mean=round(float(half[:, 1].mean()), 4),
            body_contact_frac=round(float((half[:, 2] > 0).mean()), 4),
            body_contact_mean=round(float(half[:, 2].mean()), 2),
            pos_err_mean=round(float(np.mean(pos_err[i])), 4),
        )

    hdr = (f"{'설계':12s} {'전진 m':>8s} {'성공':>5s} {'높이0':>7s} {'기립0':>7s} "
           f"{'높이':>7s} {'기립도':>7s} {'몸통접지율':>10s} {'pos_err':>8s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for lab, r in res.items():
        print(f"{lab:10s} {r['distance']:8.3f} {'O' if r['success'] else 'X':>5s} "
              f"{r['base_z_reset']:7.3f} {r['upright_reset']:7.3f} "
              f"{r['base_z_mean']:7.3f} {r['upright_mean']:7.3f} "
              f"{r['body_contact_frac']:10.2f} {r['pos_err_mean']:8.4f}")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{args.out_name}.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\nsaved", f"{OUT}/{args.out_name}.json")


if __name__ == "__main__":
    main()
