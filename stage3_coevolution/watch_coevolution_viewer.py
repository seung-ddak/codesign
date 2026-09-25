"""Watch coevolution round checkpoints in a live MuJoCo viewer.

Run this in a graphical session while train_coevolution.py remains headless.
The viewer waits for round_XXXX/theta.pkl, restores the PPO policy, and
replays the best design from that round until the next checkpoint appears.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import jax
import jax.numpy as jp
import mujoco
from mujoco import viewer
import numpy as np
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics

ROOT = Path(__file__).resolve().parents[1]
TT = Path(os.environ.get("CODESIGN_TT", ROOT.parent / "transformer-transformer")).resolve()

# Keep run-directory resolution anchored to the codesign repo even though the
# import path needs TT to be the working directory for the training stack.
WORKDIR = ROOT

os.chdir(TT)
import sys
sys.path.insert(0, str(ROOT))
from coevolution_env import WalkEnv


def load_policy(env, checkpoint):
    """Rebuild the exact PPO network used by train_coevolution.py."""
    abstract = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
    ppo_network = networks.make_ppo_networks(
        observation_size=abstract.obs.shape[-1],
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(128, 128),
        value_hidden_layer_sizes=(128, 128),
    )
    make_policy = networks.make_inference_fn(ppo_network)
    with checkpoint.open("rb") as stream:
        params = pickle.load(stream)
    return make_policy((params[0], params[1]), deterministic=True)


def best_design(round_dir):
    records = json.loads((round_dir / "designs.json").read_text())
    valid = [item for item in records if "invalid" not in item]
    if not valid:
        raise RuntimeError(f"No valid designs in {round_dir}")
    return max(valid, key=lambda item: np.mean([episode[0] for episode in item["episodes"]]))


def sync_viewer_data(model, view_data, frame):
    view_data.qpos[:], view_data.qvel[:], view_data.act[:], view_data.ctrl[:] = frame
    mujoco.mj_forward(model, view_data)


def run_round(round_dir, horizon, seed, action_scale, upright):
    design = best_design(round_dir)
    env = WalkEnv(
        choices=design["choices"],
        seed=int(design["seed"]),
        goal=1.0,
        threshold=upright,
        action_scale=action_scale,
    )
    policy = load_policy(env, round_dir / "theta.pkl")
    key = jax.random.PRNGKey(seed)
    state = env.reset(key)
    view_data = mujoco.MjData(env.mj_model)
    initial = tuple(np.array(getattr(state.data, name), copy=True)
                    for name in ("qpos", "qvel", "act", "ctrl"))
    sync_viewer_data(env.mj_model, view_data, initial)
    print(
        f"Viewing {round_dir.name}: choices={design['choices']} "
        f"seed={design['seed']}", flush=True
    )
    with viewer.launch_passive(model=env.mj_model, data=view_data) as handle:
        handle.opt.geomgroup[3] = 1  # TT body and leg capsules are in group 3.
        handle.cam.distance = 1.8
        handle.cam.azimuth = 125
        handle.cam.elevation = -18
        handle.cam.lookat[:] = [0.4, 0.0, 0.25]
        handle.sync()
        print("MuJoCo window opened; preparing episode...", flush=True)
        frames = []
        # Compute MJX/GPU steps after opening the window; playback stays even.
        for step_index in range(min(horizon, 1000)):
            key, action_key = jax.random.split(key)
            action, _ = policy(state.obs, action_key)
            state = env.step(state, action)
            frames.append(tuple(np.array(getattr(state.data, name), copy=True)
                                for name in ("qpos", "qvel", "act", "ctrl")))
            if (step_index + 1) % 10 == 0:
                print(f"Preparing episode: {step_index + 1} steps", flush=True)
            if bool(np.asarray(state.done)):
                break
        print(f"episode: {len(frames)} steps, "
              f"progress={float(np.asarray(state.info['progress'])):.3f} m, "
              f"fell={bool(np.asarray(state.metrics['fell']))}, "
              f"success={bool(np.asarray(state.metrics['success']))}", flush=True)
        next_frame = time.monotonic()
        for step_index in range(horizon):
            if not handle.is_running():
                return False
            frame = frames[step_index % len(frames)]
            sync_viewer_data(env.mj_model, view_data, frame)
            handle.sync()
            next_frame += env.ctrl_dt
            time.sleep(max(0., next_frame - time.monotonic()))
            if step_index % len(frames) == len(frames) - 1:
                next_frame = time.monotonic()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "coevolution_full")
    parser.add_argument("--round", type=int, help="Replay one completed round, then exit")
    parser.add_argument("--poll", type=float, default=3.0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100001)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--upright", type=float, default=0.1)
    args = parser.parse_args()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = (WORKDIR / run_dir).resolve()
    else:
        run_dir = run_dir.resolve()
    if args.round is not None:
        round_dir = run_dir / f"round_{args.round:04d}"
        if not (round_dir / "theta.pkl").is_file() or not (round_dir / "designs.json").is_file():
            parser.error(f"Completed checkpoint not found: {round_dir}")
        run_round(round_dir, args.horizon, args.seed, args.action_scale, args.upright)
        return
    seen = set()
    print(f"Watching {run_dir} for round checkpoints", flush=True)
    while True:
        rounds = sorted(
            path for path in run_dir.glob("round_*")
            if (path / "theta.pkl").is_file() and (path / "designs.json").is_file()
        )
        pending = [path for path in rounds if path.name not in seen]
        if pending:
            round_dir = pending[-1]
            seen.add(round_dir.name)
            if not run_round(round_dir, args.horizon, args.seed, args.action_scale, args.upright):
                return
        elif (run_dir / "progress.json").is_file():
            status = json.loads((run_dir / "progress.json").read_text()).get("status", "")
            if "completed" in status.lower() and rounds:
                print(f"Training finished; last viewed round is {rounds[-1].name}", flush=True)
                return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
