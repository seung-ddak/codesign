"""E 반사실 데이터: 같은 행동 A 를 여러 형상 E 에서 open-loop 로 실행 (remind 8-32).

원본 형상 k 의 expert episode target_q 앞 T step 을, 대상 형상 j 의 home 안착 자세 S0 = g(E_j) 에서 MJX 로 실행한다.
qpos 는 형상 사이에 공유하지 않는다 (Foot-Anchored v2 실패 원인). 같은 A 의 결과 차이는 E 에서만 온다.
출력 = <out>/<target>/primitive_seed0.npz (wide16 primitive 형식). source_episode = 원본 row 라서
rollout_data.source_group_keys 가 원본 episode 와 같은 그룹으로 묶는다 (train/val 누수 없음).

usage (codesign/ 에서):
  ../transformer-transformer/.venv/bin/python code/tt_dynamics/collect_e_swap.py --out runs/pipeline_v1/e_swap/smoke \
      --sources e000 e008 --targets e000 e008 --episodes 8
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT.parent / "transformer-transformer"), str(ROOT / "code/motion_tracking")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pd_replay import build_model  # noqa: E402
from t2.mjx_env.base_env import make_data, step  # noqa: E402
from track_env import TrackMjxEnv  # noqa: E402

DATA = ROOT / "runs/pipeline_v1/wide16_action/rollouts"
TRAIN16 = sorted(p.name for p in DATA.glob("e*"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--sources", nargs="+", default=TRAIN16, help="A 를 가져올 형상")
parser.add_argument("--targets", nargs="+", default=TRAIN16, help="A 를 실행할 형상")
parser.add_argument("--episodes", type=int, default=32, help="원본 형상당 episode 수")
parser.add_argument("--steps", type=int, default=100, help="실행 길이 (state 수)")
parser.add_argument("--settle", type=float, default=1.0, help="home 자세 안착 시간 (s), generate_task.rest_s0_of 와 같음")
parser.add_argument("--expert-config", type=Path, default=ROOT / "runs/pipeline_v1/experts_wide16/walkForwardSlow__r00")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--designs", type=Path, default=ROOT / "runs/pipeline_v1/retargeting/designs.json", help="대상 형상 E 목록")
args = parser.parse_args()

cfg = pickle.load(open(sorted(args.expert_config.glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb"))
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
choice = cfg.get("choice")
ecfg["reference_dir"] = str(ROOT / "runs/pipeline_v1/retargeting/e000" / choice)
ecfg["clips"] = ["walkForwardSlow__r00"]
designs = json.loads(args.designs.read_text())

# 원본 A: 형상마다 T step 이상 유효한 episode 중 무작위 (seed 고정)
rng = np.random.default_rng(args.seed)
src = {k: [] for k in ("target_q", "source_episode", "source_design", "clip")}
for k in args.sources:
    with np.load(DATA / k / "primitive_seed0.npz") as z:
        ok = np.flatnonzero(z["valid"].sum(1) >= args.steps)
        rows = np.sort(rng.choice(ok, min(args.episodes, len(ok)), replace=False))
        src["target_q"].append(z["target_q"][rows, :args.steps])
        src["clip"].append(z["clip"][rows, :args.steps])
        src["source_episode"].append(rows)
        src["source_design"] += [k] * len(rows)
        motions = z["motions"]
src = {k: np.concatenate(v) if k != "source_design" else np.array(v) for k, v in src.items()}
B = len(src["source_episode"])
print(f"sources {args.sources}: {B} action sequences x {args.steps} steps", flush=True)


def rest_state(env):
    """generate_task.rest_s0_of 와 같은 안착: home keyframe, 발 최저점을 바닥에, home PD 목표로 settle 초."""
    m = env.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = m.keyframe("home").qpos
    d.qpos[2] = 1.0
    mujoco.mj_forward(m, d)
    feet = np.asarray(env._foot_ids)
    d.qpos[2] -= float((d.geom_xpos[feet, 2] - m.geom_size[feet, 0]).min())
    d.ctrl[:] = np.asarray(env.action_offset)
    for _ in range(round(args.settle / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d.qpos.copy(), d.qvel.copy()


args.out.mkdir(parents=True, exist_ok=True)
for j in args.targets:
    path = args.out / j / "primitive_seed0.npz"
    if path.exists():
        print(f"{j}: exists, skip", flush=True)
        continue
    E = np.asarray(designs[j], np.float32)
    env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                     torque_limit=cfg.get("torque_limit"), uniforms=E), **ecfg)
    q0, v0 = rest_state(env)
    off, sc, (lo, hi) = np.asarray(env.action_offset), float(env.action_scale), env.clip_actions
    tq = np.clip(src["target_q"], off[:12] + lo * sc, off[:12] + hi * sc).astype(np.float32)
    d0 = mjx.forward(env.mjx_model, make_data(env.mj_model, qpos=jnp.asarray(q0), qvel=jnp.asarray(v0), impl=env.impl,
                                              nconmax=env._nconmax, njmax=env._njmax))
    data = jax.tree.map(lambda x: jnp.broadcast_to(x, (B, *x.shape)), d0)
    vstep = jax.jit(jax.vmap(lambda s, c: step(env.mjx_model, s, c, env.n_substeps)))
    feet, floor = np.asarray(env._foot_ids), mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    rec = {k: [] for k in ("qpos", "qvel", "contact", "valid")}
    alive = np.ones(B, bool)
    for t in range(args.steps):
        q, v = np.asarray(data.qpos), np.asarray(data.qvel)
        geom, dist = np.asarray(data._impl.contact.geom), np.asarray(data._impl.contact.dist)
        contact = np.stack([((((geom[..., 0] == f) & (geom[..., 1] == floor)) | ((geom[..., 1] == f) & (geom[..., 0] == floor)))
                             & (dist <= 1e-3)).any(-1) for f in feet], -1)
        alive &= np.isfinite(q).all(1) & np.isfinite(v).all(1) & (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2) >= env.termination_fellover_threshold)
        for k, x in (("qpos", q), ("qvel", v), ("contact", contact), ("valid", alive.copy())):
            rec[k].append(x)
        if t < args.steps - 1:
            ctrl = np.broadcast_to(off, (B, len(off))).copy()
            ctrl[:, :12] = tq[:, t]
            data = vstep(data, jnp.asarray(ctrl))
    out = {k: np.stack(v, 1) for k, v in rec.items()}
    out.update(target_q=tq, E=E, design=j, source_design=src["source_design"], source_episode=src["source_episode"],
               clip=src["clip"], motions=motions, fell=~out["valid"][:, -1], dt=env.dt, seed=args.seed,
               mode="e_swap", rest_settle_s=args.settle)
    for k in ("qpos", "qvel", "target_q"):
        assert np.isfinite(out[k][out["valid"]] if k != "target_q" else out[k]).all(), f"nonfinite {k}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    disp = out["qpos"][:, -1, :2] - out["qpos"][:, 0, :2]
    print(f"{j}: fell {out['fell'].sum()}/{B} | displacement median {np.median(np.linalg.norm(disp, axis=1)):.2f} m -> {path}", flush=True)
