"""Why do episodes end in ~60 steps? Episode length without learning (2026-09-27).

Conditions (same env, reward and termination as training, no policy):
  home     zero action from the exact home pose, zero base velocity
  reset    zero action from the training reset (dof x U(0.5, 1.5), base vel U(-0.5, 0.5))
  rand_s   training reset + actions ~ N(0, s^2); s = policy std (1.0 init, 2.2 seen in
           t5_screen_200it). rand_0.5 at ACTION_SCALE 0.5 equals rand_1.0 at ACTION_SCALE 0.25.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TT = Path(os.environ.get("CODESIGN_TT", ROOT.parent / "transformer-transformer")).resolve()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(TT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(TT)

import jax
import jax.numpy as jp
import numpy as np

from locomotion_env import EPISODE_STEPS
from multi_design_env import DesignGroupEnv

CONDITIONS = {"home": (1., 0.), "reset": (0., 0.), "rand_0.5": (0., .5),
              "rand_1.0": (0., 1.), "rand_2.2": (0., 2.2)}


def make_rollout(env, n):
    home_qpos = env.env.home

    @jax.jit
    def rollout(rng, home, std):
        rng, key = jax.random.split(rng)
        state = env.reset(jax.random.split(key, n))
        data = state.data
        state = state.replace(data=data.replace(
            qpos=jp.where(home > 0, jp.broadcast_to(home_qpos, data.qpos.shape), data.qpos),
            qvel=jp.where(home > 0, 0., data.qvel)))

        def body(carry, key):
            state, alive, length, sums = carry
            state = env.step(state, std * jax.random.normal(key, (n, env.action_size)))
            sums = {k: sums[k] + v * alive for k, v in state.metrics.items()}
            return (state, alive * (1. - state.done), length + alive, sums), None

        zeros = jp.zeros(n)
        carry = (state, jp.ones(n), zeros, {k: zeros for k in state.metrics})
        (_, _, length, sums), _ = jax.lax.scan(body, carry, jax.random.split(rng, EPISODE_STEPS))
        return length, sums

    return rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", type=Path, default=ROOT / "stage2/runs/t5_smoke_contact/designs.json")
    p.add_argument("--envs-per-group", type=int, default=64)
    p.add_argument("--out", type=Path, default=ROOT / "stage2/runs/diag_fall_modes.json")
    args = p.parse_args()

    groups = json.loads(args.designs.read_text())
    res = {c: {"length": [], "sums": {}, "design": []} for c in CONDITIONS}
    for g, designs in enumerate(groups):
        env = DesignGroupEnv(designs)
        rollout = make_rollout(env, args.envs_per_group)
        for c, (home, std) in CONDITIONS.items():
            length, sums = jax.device_get(rollout(jax.random.PRNGKey(g), home, std))
            res[c]["length"] += length.tolist()
            res[c]["design"] += [f"{g}:{i % len(designs)}" for i in range(args.envs_per_group)]
            for k, v in sums.items():
                res[c]["sums"].setdefault(k, []).extend(v.tolist())
            print(f"group {g} {c}: mean length {length.mean():.1f}", flush=True)

    summary = {}
    for c, r in res.items():
        length = np.array(r["length"])
        steps = length.sum()
        calf_designs = sorted({d for d, v in zip(r["design"], r["sums"]["collision"]) if v < 0})
        summary[c] = {"mean_length": float(length.mean()), "median_length": float(np.median(length)),
                      "survived_frac": float((length >= EPISODE_STEPS).mean()),
                      "per_step": {k: float(np.sum(v) / steps) for k, v in r["sums"].items()},
                      "designs_with_collision": f"{len(calf_designs)}/{len(set(r['design']))}"}
    args.out.write_text(json.dumps(summary, indent=2))

    terms = list(summary["home"]["per_step"])
    print(f"\n{'':22}" + "".join(f"{c:>10}" for c in CONDITIONS))
    for key in ("mean_length", "median_length", "survived_frac"):
        print(f"{key:22}" + "".join(f"{summary[c][key]:10.3g}" for c in CONDITIONS))
    for k in terms:
        print(f"{k:22}" + "".join(f"{summary[c]['per_step'][k]:10.4f}" for c in CONDITIONS))
    print(f"{'designs_w_collision':22}" + "".join(f"{summary[c]['designs_with_collision']:>10}" for c in CONDITIONS))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
