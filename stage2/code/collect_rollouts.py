"""Frozen locomotion/jump expert rollouts as TT dynamics training data.

Each batch draws one new continuous morphology (seed -> 13 uniforms, as LocomotionEnv) for every
one of the 128 choices (Q12) and rolls all of them under the 7 fixed EVAL_COMMANDS with DR off
(D51, D52), recording every step. Batches repeat until the next one would overrun --hours, so the
run ends inside the budget whatever the throughput (Q13). Full episodes are kept; the chunk length
H is picked when R2 cuts them.

  python stage2/code/collect_rollouts.py --skill locomotion --run stage2/runs/T5/try1 --out stage2/runs/T8/try1 --hours 4.5
  python stage2/code/collect_rollouts.py --skill jump --run runs/expert/jump/train5 --ckpt 000012288000 --out runs/dynamics/jump --hours 4.5
  smoke: add --steps 50 --max-batches 2 (second batch must reuse the compiled rollout)

<out>/batch_XXXX.npz, T = steps, N = 128 designs x 7 commands:
  qpos/qvel              physical state before action t, (T, N, 19)/(T, N, 18)
  action                 deterministic expert action at step t, (T, N, 12)
  next_qpos/next_qvel    physical state returned by that action
  contacts/next_contacts foot contacts before/after that action, (T, N, 4)
  command                expert command before action t, padded to (T, N, 5)
  reward/done/alive      transition result and pre-step validity mask, (T, N)
  choices (N, 8), seed (N,), uniforms (N, 13), skill_id (), command_index (N,), return (N,), length (N,)
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

from eval_locomotion_policy import EVAL_COMMANDS, LAUNCH_DIR, load_policy  # chdirs into TT

import jax
import jax.numpy as jp
import numpy as np

from locomotion_env import EPISODE_STEPS, LocomotionEnv
from multi_design_env import MultiDesignEnv, structure_key
def jump_env_class():
    """Load the 257-observation jump env without shadowing stage2/task_env.py."""
    path = Path(__file__).resolve().parents[2] / "code" / "task_env.py"
    spec = importlib.util.spec_from_file_location("_rollout_jump_task_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlatJumpEnv


def sample_groups(rng, design_path, max_groups=None):
    """One fresh seed per choice, grouped by MJX structure in a fixed order."""
    saved = json.loads(design_path.read_text())
    choices = sorted({tuple(design[0]) for group in saved for design in group})
    groups = {}
    for choice in choices:
        c = list(choice)
        groups.setdefault(structure_key(c), []).append((c, int(rng.randint(2 ** 31 - 1))))
    return list(groups.values())[:max_groups]


def build(groups, env_cls):
    env = MultiDesignEnv(groups, EPISODE_STEPS, env_cls=env_cls,
                         hold_command=True, domain_rand=False, add_noise=False)
    # groups are BraxAutoResetWrapper(EpisodeWrapper(DesignGroupEnv)); swap designs as DesignScorer.
    return env, tuple(g.env.env.designs for g in env.groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skill", choices=("locomotion", "jump"), required=True)
    p.add_argument("--run", required=True, help="frozen T5 run")
    p.add_argument("--ckpt", help="checkpoint directory name (default: latest)")
    p.add_argument("--out", required=True)
    p.add_argument("--hours", type=float, default=4.5)
    p.add_argument("--steps", type=int, default=EPISODE_STEPS, help="smoke tests only")
    p.add_argument("--max-batches", type=int, default=10 ** 6)
    p.add_argument("--max-groups", type=int, help="smoke tests only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run, out = (LAUNCH_DIR / args.run).resolve(), (LAUNCH_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    design_path = run / "designs.json"
    if not design_path.is_file():
        p.error(f"designs not found: {design_path}")
    ckpt = run / "checkpoints" / args.ckpt if args.ckpt else sorted((run / "checkpoints").iterdir())[-1]
    if not ckpt.is_dir():
        p.error(f"checkpoint not found: {ckpt}")
    env_cls = LocomotionEnv if args.skill == "locomotion" else jump_env_class()
    (out / "config.json").write_text(json.dumps({
        "args": vars(args), "checkpoint": str(ckpt),
        "commands": EVAL_COMMANDS if args.skill == "locomotion" else None,
        "episode_steps": args.steps, "domain_rand": False}, indent=1))

    rng = np.random.RandomState(args.seed)
    groups = sample_groups(rng, design_path, args.max_groups)
    keys = [structure_key(g[0][0]) for g in groups]
    env, designs = build(groups, env_cls)
    policy = load_policy(ckpt)
    K = len(groups[0])
    episodes = len(EVAL_COMMANDS) if args.skill == "locomotion" else 1
    if any(len(g) != K for g in groups):
        raise ValueError("Every structure group must hold the same number of choices")
    per_group = K * episodes
    # Group slot i runs design i % K with command i // K (as eval_locomotion_policy).
    command_index = np.tile(np.arange(per_group) // K, len(groups))
    commands = (jp.array([[vx, vy, wz, 0.] for _, vx, vy, wz in EVAL_COMMANDS])
                [np.arange(per_group) // K] if args.skill == "locomotion" else None)
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
        if commands is not None:
            state = env._merge([s.replace(info={**s.info, "command": commands})
                                for s in state.info["groups"]])

        def physical(s):
            gs = s.info["groups"]
            # Raw MuJoCo nq/nv varies with optional morphology components.  The floating base and
            # 12 actuated leg joints are the shared physical state used by both expert policies.
            qpos = jp.concatenate([
                jp.concatenate((s.data.qpos[:, e.env.qadr:e.env.qadr + 7],
                                s.data.qpos[:, e.env.qids]), axis=1)
                for s, e in zip(gs, inner)])
            qvel = jp.concatenate([
                jp.concatenate((s.data.qvel[:, e.env.vadr:e.env.vadr + 6],
                                s.data.qvel[:, e.env.vids]), axis=1)
                for s, e in zip(gs, inner)])
            contacts = jp.concatenate([g.info["last_contacts"] for g in gs])
            command = jp.concatenate([g.info["command"] for g in gs])
            if args.skill == "locomotion":
                command = jp.pad(command, ((0, 0), (0, 1)))
            return qpos, qvel, contacts, command

        def body(carry, k):
            state, alive = carry
            obs = state.obs
            qpos, qvel, contacts, command = physical(state)
            action, _ = policy(obs, k)
            next_state = env.step(state, action)
            next_qpos, next_qvel, next_contacts, _ = physical(next_state)
            row = (obs, qpos, qvel, contacts, command, action,
                   next_qpos, next_qvel, next_contacts,
                   next_state.reward, next_state.done, alive)
            return (next_state, alive * (1. - next_state.done)), row

        _, rows = jax.lax.scan(body, (state, jp.ones(per_group * len(groups))),
                               jax.random.split(key, args.steps))
        return rows

    rollout = jax.jit(rollout)
    start, last, b = time.time(), 0., 0
    budget = args.hours * 3600
    while b < args.max_batches and (b == 0 or time.time() - start + 1.1 * last < budget):
        t = time.time()
        if b:
            groups = sample_groups(rng, design_path, args.max_groups)
            if [structure_key(g[0][0]) for g in groups] != keys:
                raise ValueError("Structure group order changed between batches")
            designs = build(groups, env_cls)[1]
        (obs, qpos, qvel, contacts, command, action, next_qpos, next_qvel,
         next_contacts, reward, done, alive) = jax.device_get(
             rollout(designs, jax.random.PRNGKey(args.seed * 100003 + b)))
        assert qpos.shape[-1] == next_qpos.shape[-1] == 19
        assert qvel.shape[-1] == next_qvel.shape[-1] == 18
        assert action.shape[-1] == 12 and contacts.shape[-1] == next_contacts.shape[-1] == 4
        assert all(np.isfinite(x).all() for x in (qpos, qvel, action, next_qpos, next_qvel))
        flat = [d for g in groups for d in g]
        design = np.array([g * K + i % K for g in range(len(groups)) for i in range(per_group)])
        np.savez(out / f"batch_{b:04d}.npz", obs=obs, qpos=qpos, qvel=qvel,
                 contacts=contacts, command=command, action=action,
                 next_qpos=next_qpos, next_qvel=next_qvel, next_contacts=next_contacts,
                 reward=reward, done=done, alive=alive,
                 choices=np.array([flat[d][0] for d in design], np.int8),
                 seed=np.array([flat[d][1] for d in design]),
                 uniforms=np.array([np.random.RandomState(flat[d][1]).uniform(size=13) for d in design]),
                 command_index=command_index,
                 skill_id=np.int8(0 if args.skill == "locomotion" else 1),
                 **{"return": (reward * alive).sum(0), "length": alive.sum(0)})
        last = time.time() - t
        b += 1
        print(f"batch {b - 1}: {last:.0f}s, return {(reward * alive).sum(0).mean():.2f}, "
              f"survived {(alive.sum(0) >= args.steps).mean():.2f}, "
              f"elapsed {(time.time() - start) / 3600:.2f} h", flush=True)
    print(f"done: {b} batches = {b} morphologies per choice, {b * len(design)} episodes", flush=True)


if __name__ == "__main__":
    main()
