"""T8 R1: frozen pi_theta rollouts as TT training data (D63).

Each batch draws one new continuous morphology (seed -> 13 uniforms, as LocomotionEnv) for every
one of the 128 choices (Q12) and rolls all of them under the 7 fixed EVAL_COMMANDS with DR off
(D51, D52), recording every step. Batches repeat until the next one would overrun --hours, so the
run ends inside the budget whatever the throughput (Q13). Full episodes are kept; the chunk length
H is picked when R2 cuts them.

  python stage2/code/collect_rollouts.py --run stage2/runs/T5/try1 --out stage2/runs/T8/try1 --hours 4.5
  smoke: add --steps 50 --max-batches 2 (second batch must reuse the compiled rollout)

<out>/batch_XXXX.npz, T = steps, N = 128 designs x 7 commands:
  obs (T, N, 69)        policy input before step t: base state, command, joints, last action,
                        morphology (choices + uniforms). At t = 0 the command field still holds the
                        reset-sampled command (same as eval_locomotion_policy); drop t = 0 in R2.
  action (T, N, 12)     deterministic pi_theta action at step t
  reward (T, N)         legged_gym step reward (T4)
  alive (T, N)          1 while the episode has not ended; mask everything else
  base_qpos (T, N, 7)   world base position + quaternion after step t (height for jump/stairs)
  choices (N, 8), seed (N,), uniforms (N, 13), command (N, 3), command_index (N,), return (N,), length (N,)
"""

import argparse
import json
import time

from eval_locomotion_policy import EVAL_COMMANDS, LAUNCH_DIR, load_policy  # chdirs into TT

import jax
import jax.numpy as jp
import numpy as np

from locomotion_env import EPISODE_STEPS
from multi_design_env import MultiDesignEnv, structure_key
from train_locomotion_policy import TT


def sample_groups(rng):
    """One fresh seed per choice, grouped by MJX structure in a fixed order."""
    choices = sorted(json.loads((TT / "choice_to_ckpt_path.json").read_text()))
    groups = {}
    for key in choices:
        c = [int(x) for x in key]
        groups.setdefault(structure_key(c), []).append((c, int(rng.randint(2 ** 31 - 1))))
    return list(groups.values())


def build(groups):
    env = MultiDesignEnv(groups, EPISODE_STEPS, hold_command=True, domain_rand=False)
    # groups are BraxAutoResetWrapper(EpisodeWrapper(DesignGroupEnv)); swap designs as DesignScorer.
    return env, tuple(g.env.env.designs for g in env.groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="frozen T5 run")
    p.add_argument("--out", required=True)
    p.add_argument("--hours", type=float, default=4.5)
    p.add_argument("--steps", type=int, default=EPISODE_STEPS, help="smoke tests only")
    p.add_argument("--max-batches", type=int, default=10 ** 6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run, out = (LAUNCH_DIR / args.run).resolve(), (LAUNCH_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ckpt = sorted((run / "checkpoints").iterdir())[-1]
    (out / "config.json").write_text(json.dumps({
        "args": vars(args), "checkpoint": str(ckpt), "commands": EVAL_COMMANDS,
        "episode_steps": args.steps, "domain_rand": False}, indent=1))

    rng = np.random.RandomState(args.seed)
    groups = sample_groups(rng)
    keys = [structure_key(g[0][0]) for g in groups]
    env, designs = build(groups)
    policy = load_policy(ckpt)
    K, episodes = len(groups[0]), len(EVAL_COMMANDS)
    if any(len(g) != K for g in groups):
        raise ValueError("Every structure group must hold the same number of choices")
    per_group = K * episodes
    # Group slot i runs design i % K with command i // K (as eval_locomotion_policy).
    command_index = np.tile(np.arange(per_group) // K, len(groups))
    commands = jp.array([[vx, vy, wz, 0.] for _, vx, vy, wz in EVAL_COMMANDS])[np.arange(per_group) // K]
    inner = [g.env.env for g in env.groups]

    def rollout(designs, key):
        saved = [g.designs for g in inner]
        try:
            for g, d in zip(inner, designs):
                g.designs = d
            return trace(key)
        finally:  # never leave tracers on the env
            for g, d in zip(inner, saved):
                g.designs = d

    def trace(key):
        state = env.reset(jax.random.split(key, per_group * len(groups)))
        state = env._merge([s.replace(info={**s.info, "command": commands})
                            for s in state.info["groups"]])

        def body(carry, k):
            state, alive = carry
            obs = state.obs
            action, _ = policy(obs, k)
            state = env.step(state, action)
            qpos = jp.concatenate([s.data.qpos[:, :7] for s in state.info["groups"]])
            row = (obs, action, state.reward, alive, qpos)
            return (state, alive * (1. - state.done)), row

        _, rows = jax.lax.scan(body, (state, jp.ones(per_group * len(groups))),
                               jax.random.split(key, args.steps))
        return rows

    rollout = jax.jit(rollout)
    start, last, b = time.time(), 0., 0
    budget = args.hours * 3600
    while b < args.max_batches and (b == 0 or time.time() - start + 1.1 * last < budget):
        t = time.time()
        if b:
            groups = sample_groups(rng)
            if [structure_key(g[0][0]) for g in groups] != keys:
                raise ValueError("Structure group order changed between batches")
            designs = build(groups)[1]
        obs, action, reward, alive, qpos = jax.device_get(
            rollout(designs, jax.random.PRNGKey(args.seed * 100003 + b)))
        flat = [d for g in groups for d in g]
        design = np.array([g * K + i % K for g in range(len(groups)) for i in range(per_group)])
        np.savez(out / f"batch_{b:04d}.npz", obs=obs, action=action, reward=reward, alive=alive,
                 base_qpos=qpos,
                 choices=np.array([flat[d][0] for d in design], np.int8),
                 seed=np.array([flat[d][1] for d in design]),
                 uniforms=np.array([np.random.RandomState(flat[d][1]).uniform(size=13) for d in design]),
                 command=np.asarray(commands)[np.arange(len(design)) % per_group, :3],
                 command_index=command_index,
                 **{"return": (reward * alive).sum(0), "length": alive.sum(0)})
        last = time.time() - t
        b += 1
        print(f"batch {b - 1}: {last:.0f}s, return {(reward * alive).sum(0).mean():.2f}, "
              f"survived {(alive.sum(0) >= args.steps).mean():.2f}, "
              f"elapsed {(time.time() - start) / 3600:.2f} h", flush=True)
    print(f"done: {b} batches = {b} morphologies per choice, {b * len(design)} episodes", flush=True)


if __name__ == "__main__":
    main()
