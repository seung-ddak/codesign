"""Rollouts without a policy: replay each retargeted reference through the PD actuators in MJX (remind 8-35).

Each episode starts on a reference frame; the next reference joint angles are the PD target at every step.
There is no feedback on balance, so an episode ends when it falls or the reference ends.
Output = <out>/<design>/primitive_seed0.npz in the collect_rollouts format. Row order is the same for
every design, so one (motion, start) shares a split group across designs.

usage (codesign/ 에서):
  ../transformer-transformer/.venv/bin/python code/motion_tracking/collect_pd_rollouts.py --out runs/pipeline_v1/pd_rollouts
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "transformer-transformer"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pd_replay import build_model  # noqa: E402
from track_env import TrackMjxEnv  # noqa: E402

RETARGET = ROOT / "runs/pipeline_v1/retargeting"
ALL_DESIGNS = [k for k in json.loads((RETARGET / "designs.json").read_text()) if k.startswith("e") and k != "e003"]
with open(ROOT / "runs/pipeline_v1/experts/selection.csv") as f:
    ALL_MOTIONS = [row["motion"] for row in csv.DictReader(f)]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--designs", nargs="+", default=ALL_DESIGNS)
parser.add_argument("--motions", nargs="+", default=ALL_MOTIONS)
parser.add_argument("--starts", type=int, default=8, help="모션당 시작 프레임 수 (전체 구간 균등 간격)")
parser.add_argument("--min-steps", type=int, default=32, help="시작 후 남아야 하는 reference 길이")
parser.add_argument("--expert-config", type=Path, default=ROOT / "runs/pipeline_v1/experts_wide16/walkForwardSlow__r00")
args = parser.parse_args()

cfg = pickle.load(open(sorted(args.expert_config.glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb"))
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
choice = cfg.get("choice")
ecfg.update(clips=list(args.motions), clip_boost=None, switch_prob=0.0, max_episode_length=10**6, termination_pose_consecutive_steps=10**6)


def collect(design):
    ecfg["reference_dir"] = str(RETARGET / design / choice)
    with np.load(RETARGET / design / choice / f"{args.motions[0]}.npz") as z:
        uniforms = np.asarray(z["uniforms"], np.float32)
    env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                     torque_limit=cfg.get("torque_limit"), uniforms=uniforms), **ecfg)
    n, starts, ends = env.n_ctrl, np.asarray(env.clip_start), np.asarray(env.clip_end)
    clips, frames = [], []
    for c, (lo, hi) in enumerate(zip(starts, ends)):
        last = hi - 1 - args.min_steps
        assert last >= lo, f"{args.motions[c]}: shorter than --min-steps"
        for f in np.unique(np.linspace(lo, last, args.starts).astype(int)):
            clips.append(c)
            frames.append(f)
    clips, frames = np.array(clips), np.array(frames)
    B, T = len(clips), int((ends[clips] - 1 - frames).max())
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    state = jax.jit(jax.vmap(env.reset_with_traj))(keys, jnp.asarray(clips), jnp.asarray(frames))
    feet, floor = env._foot_ids, int(env.mj_model.geom("floor").id)

    def contact(data):
        geom, dist = data._impl.contact.geom, data._impl.contact.dist
        return jax.vmap(lambda f: ((((geom[:, 0] == f) & (geom[:, 1] == floor)) | ((geom[:, 1] == f) & (geom[:, 0] == floor)))
                                   & (dist <= 1e-3)).any())(feet)

    @jax.jit
    def step(s, target):
        action = (target - env.action_offset[:n]) / env.action_scale
        s = jax.vmap(env.step)(s, action)
        return s, action, jax.vmap(contact)(s.data)

    rec = {k: [] for k in ("qpos", "qvel", "contact", "action", "clip", "ref_frame", "valid")}
    rec["qpos"].append(np.asarray(state.data.qpos))
    rec["qvel"].append(np.asarray(state.data.qvel))
    rec["contact"].append(np.asarray(jax.vmap(contact)(state.data)))
    alive, fell = frames < ends[clips] - 1, np.zeros(B, bool)
    ref_q = np.asarray(env.ref_qpos[:, 7:7 + n])
    for t in range(T):
        frame = np.minimum(frames + t, ends[clips] - 2)
        state, action, touch = step(state, jnp.asarray(ref_q[frame + 1]))
        bad = alive & (np.asarray(state.info["state_is_nan"]) != 0)
        assert not bad.any(), f"{design}: nonfinite physics at step {t}"
        for k, v in (("qpos", state.data.qpos), ("qvel", state.data.qvel), ("contact", touch), ("action", action),
                     ("clip", clips), ("ref_frame", frame), ("valid", alive)):
            rec[k].append(np.asarray(v))
        fell |= alive & (np.asarray(state.info["done/fellover"]) != 0)
        alive = alive & ~fell & (frames + t + 1 < ends[clips] - 1)
        if not alive.any():
            break
    out = {k: np.stack(v, 1) for k, v in rec.items()}
    out["target_q"] = np.asarray(env.action_offset[:n]) + np.clip(out["action"], *env.clip_actions) * env.action_scale
    out.update(fell=fell, E=uniforms, motions=np.array(args.motions), dt=env.dt, design=design, mode="pd_reference",
               contact_definition="foot_floor", clip_start=starts, clip_end=ends, action_offset=np.asarray(env.action_offset[:n]),
               action_scale=env.action_scale, clip_actions=np.asarray(env.clip_actions), seed=0)
    path = args.out / design / "primitive_seed0.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    steps = out["valid"].sum(1)
    print(f"{design}: fell {fell.sum()}/{B} | steps median {np.median(steps):.0f} / reference {np.median(ends[clips] - 1 - frames):.0f} -> {path}", flush=True)


args.out.mkdir(parents=True, exist_ok=True)
for design in args.designs:
    if (args.out / design / "primitive_seed0.npz").exists():
        print(f"{design}: exists, skip", flush=True)
    else:
        collect(design)
