"""Record a fixed jump-trigger rollout and its per-step trace."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DIR = Path.cwd()
TT = ROOT / "transformer-transformer"
CODE = Path(__file__).resolve().parent
STAGE2 = ROOT / "codesign/stage2/code"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path[:0] = [str(CODE), str(STAGE2), str(TT)]

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402

from task_env import FlatJumpEnv  # noqa: E402
from eval_locomotion_policy import load_policy  # noqa: E402
from record_eval_video import record  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ckpt", help="checkpoint step directory (default: latest)")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    run = (LAUNCH_DIR / args.run).resolve()
    ckpt_name = args.ckpt or sorted(p.name for p in (run / "checkpoints").iterdir())[-1]
    ckpt = run / "checkpoints" / ckpt_name
    designs = [design for group in json.loads((run / "designs.json").read_text()) for design in group]
    choices, seed = designs[args.rank]
    env = FlatJumpEnv(choices, seed, hold_command=True,
                      domain_rand=False, add_noise=False)
    policy = load_policy(ckpt)
    steps = round(args.seconds / env.dt)
    jump_command = jp.array([0., 0., 0., 0., 1.])

    @jax.jit
    def rollout(rng):
        state = env.reset(rng)
        state = state.replace(
            info={**state.info, "command": jump_command},
            obs=env.observation(state.data, jump_command, jp.zeros(env.action_size), rng),
        )

        def body(state, key):
            command = state.info["command"]
            action, _ = policy(state.obs, key)
            state = env.step(state, action)
            return state, (state.data.qpos, command, action, state.info["last_contacts"], state.done)

        return jax.lax.scan(body, state, jax.random.split(rng, steps))[1]

    qpos, command, action, contacts, done = map(
        np.asarray, jax.device_get(rollout(jax.random.PRNGKey(args.seed))))
    n = int(np.argmax(done > 0)) + 1 if done.any() else steps
    qpos, command, action, contacts = qpos[:n], command[:n], action[:n], contacts[:n]
    out = run / "videos"
    out.mkdir(exist_ok=True)
    video = out / f"jump_rank_{args.rank}.mp4"
    record(env, qpos, command, n, video)

    airborne = ~contacts.any(axis=1)
    first_air = int(np.argmax(airborne)) if airborne.any() else None
    landed = np.flatnonzero(~airborne & (np.arange(n) > (first_air if first_air is not None else n)))
    first_landing = int(landed[0]) if landed.size else None
    trace = {
        "checkpoint": str(ckpt), "rank": args.rank, "choices": choices,
        "seed": seed, "dt": env.dt, "video": video.name,
        "trigger_steps": np.flatnonzero(command[:, 4] > .5).tolist(),
        "first_airborne_step": first_air,
        "first_landing_step": first_landing,
        "landed_after_airborne": first_landing is not None,
        "max_base_rise_m": float(qpos[:, 2].max() - qpos[0, 2]),
        "steps": [{"step": i, "jump_trigger": float(command[i, 4]),
                   "base_z": float(qpos[i, 2]), "all_feet_airborne": bool(airborne[i]),
                   "action_max_abs": float(np.abs(action[i]).max())} for i in range(n)],
    }
    trace_path = out / f"jump_rank_{args.rank}_trace.json"
    trace_path.write_text(json.dumps(trace, indent=1))
    print(f"trigger steps: {trace['trigger_steps']}")
    print(f"first airborne: {first_air}, first landing: {first_landing}, "
          f"max rise: {trace['max_base_rise_m']:.3f} m")
    print(f"wrote {video}\nwrote {trace_path}")


if __name__ == "__main__":
    main()
