"""진단 D1: open-loop 민감도. rollout 데이터의 실제 행동 (target_q) 을 MJX 에 그대로 재생한다 (TT 무관).

대상 = 데이터에서 10 s 동안 넘어지지 않은 seq episode (형상별 최대 --episodes 개).
교란 수준마다 시작 state 에 균등 잡음을 더해 재생하고, 낙상 비율·데이터 궤적과의 차이를 잰다.
  none   : 교란 없음 (대조군: 재생이 데이터를 재현하는지 확인)
  small  : 관절 ±0.01 rad, 몸통 위치 ±1 mm
  medium : 관절 ±0.05 rad, 몸통 위치 ±5 mm
판정 (사전): small 에서 10 s 안 낙상 ≥ 50 % → open-loop 실행 자체가 무너진다 (TT 정확도와 무관, 재계획 필요).

usage (codesign/ 에서):
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false ../transformer-transformer/.venv/bin/python \
      code/tt_dynamics/diag_openloop.py --out runs/pipeline_v1/diag/D1_openloop
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

CODESIGN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODESIGN.parent / "transformer-transformer"))
sys.path.insert(0, str(CODESIGN / "code" / "motion_tracking"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pd_replay import build_model  # noqa: E402
from t2.mjx_env.base_env import make_data, step  # noqa: E402
from track_env import TrackMjxEnv  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--designs", nargs="+", default=["e000", "e001", "e002"])
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--samples", type=int, default=2, help="교란 수준당 episode 마다 잡음 표본 수")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
LEVELS = {"none": (0.0, 0.0), "small": (0.01, 0.001), "medium": (0.05, 0.005)}  # (관절 rad, 몸통 위치 m)
CHECK_T = (50, 100, 250, 499)
R = CODESIGN / "runs/pipeline_v1"

expert = R / "experts/walkForwardSlow__r00"
cfg = pickle.load(open(sorted(expert.glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb"))
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
choice = cfg.get("choice")
ecfg["clips"] = ["walkForwardSlow__r00"]


def yaw(q):
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


rng = np.random.default_rng(args.seed)
result = dict(levels={k: dict(joint_rad=v[0], base_m=v[1]) for k, v in LEVELS.items()}, samples=args.samples, designs={})
for d in args.designs:
    z = np.load(R / f"rollouts/{d}/seq_seed0.npz")
    ok = np.where(~z["fell"] & z["valid"].all(1))[0][:args.episodes]
    ecfg["reference_dir"] = str(R / "retargeting" / d / choice)
    env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                     torque_limit=cfg.get("torque_limit"), uniforms=z["E"]), **ecfg)
    model, thr, nsub = env.mjx_model, env.termination_fellover_threshold, env.n_substeps
    off = jnp.asarray(env.action_offset)
    data0 = make_data(env.mj_model, qpos=jnp.asarray(z["qpos"][ok[0], 0]), qvel=jnp.asarray(z["qvel"][ok[0], 0]),
                      impl=env.impl, nconmax=env._nconmax, njmax=env._njmax)
    init = jax.jit(jax.vmap(lambda q, v: mjx.forward(model, data0.replace(qpos=q, qvel=v))))
    stepb = jax.jit(jax.vmap(lambda dt, tq: step(model, dt, off.at[:12].set(tq), nsub)))
    TQ = z["target_q"][ok]  # [n,500,12]
    Qd = z["qpos"][ok]  # [n,501,19] 데이터 궤적
    res = {}
    for lv, (jr, br) in LEVELS.items():
        k = 1 if lv == "none" else args.samples
        q0 = np.repeat(Qd[:, 0], k, 0).copy()
        if jr:
            q0[:, 7:19] += rng.uniform(-jr, jr, q0[:, 7:19].shape)
            q0[:, 0:3] += rng.uniform(-br, br, q0[:, 0:3].shape)
        v0 = np.repeat(z["qvel"][ok, 0], k, 0)
        tq = jnp.asarray(np.repeat(TQ, k, 0))
        ref = np.repeat(Qd, k, 0)
        dat = init(jnp.asarray(q0, jnp.float32), jnp.asarray(v0, jnp.float32))
        n = len(q0)
        fell = np.full(n, -1)
        qs = [np.asarray(dat.qpos)]
        for t in range(500):
            dat = stepb(dat, tq[:, t])
            q = np.asarray(dat.qpos)
            qs.append(q)
            up = 1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)  # 몸통 z 축의 world z 성분
            fell = np.where((fell < 0) & (up < thr), t + 1, fell)
        Q = np.stack(qs, 1)  # [n,501,19]
        alive = lambda t: fell < 0  # noqa: E731
        div = {}
        for t in CHECK_T:
            m = (fell < 0) | (fell > t)
            if m.sum() == 0:
                continue
            dp = np.linalg.norm(Q[m, t, :2] - ref[m, t, :2], axis=-1)
            dy = np.degrees(np.abs(np.angle(np.exp(1j * (yaw(Q[m, t]) - yaw(ref[m, t]))))))
            dj = np.degrees(np.sqrt(((Q[m, t, 7:19] - ref[m, t, 7:19]) ** 2).mean(-1)))
            div[f"t{t}"] = dict(n=int(m.sum()), pos_cm=round(float(np.median(dp) * 100), 2), yaw_deg=round(float(np.median(dy)), 2),
                                joint_deg=round(float(np.median(dj)), 2))
        fs = fell[fell >= 0]
        res[lv] = dict(n=n, fall_rate=round(float((fell >= 0).mean()), 3),
                       fall_step_median=int(np.median(fs)) if len(fs) else None, divergence_median=div)
        print(d, lv, json.dumps(res[lv], ensure_ascii=False), flush=True)
    result["designs"][d] = dict(episodes=len(ok), **res)

small = np.mean([result["designs"][d]["small"]["fall_rate"] for d in args.designs])
result["verdict"] = dict(small_fall_rate=round(float(small), 3), open_loop_breaks=bool(small >= 0.5))
args.out.mkdir(parents=True, exist_ok=True)
json.dump(result, open(args.out / "d1_openloop.json", "w"), indent=1, ensure_ascii=False)
print("->", args.out, result["verdict"])
