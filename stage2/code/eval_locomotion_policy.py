"""T5 pi_theta rollout evaluation and MuJoCo viewer.

  eval: every design in <run>/designs.json x EVAL_COMMANDS (one episode per command, the same set
        for every design so designs are compared on equal tasks), deterministic policy, no domain
        randomisation (D52, as legged_gym play.py: no observation noise, pushes or friction DR).
        Commands are held for the whole episode.
        Writes <run>/eval_<step>.json with per-design and per-command summaries.
        Besides the reward terms it measures real motion: cmd_progress_m is the base displacement
        projected on the commanded (vx, vy) direction in the base frame, cmd_distance_m the
        distance the command asked for, cmd_progress_ratio their ratio. cmd_vel_progress_m integrates
        the sampled base velocity (what tracking_lin_vel sees) along the same direction; a large gap
        to cmd_progress_m means the reward is read from vibration, not travel. cmd_turn_rad is the
        base yaw change along the commanded yaw-rate sign, cmd_turn_asked_rad what it asked for.
  view: replays one design (--rank into the eval ranking, 0 = best, -1 = worst) in a
        passive MuJoCo viewer. Needs a display.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TT = Path(os.environ.get("CODESIGN_TT", ROOT.parent / "transformer-transformer")).resolve()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(TT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
LAUNCH_DIR = Path.cwd()
os.chdir(TT)

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import checkpoint

from locomotion_env import CTRL_DT, EPISODE_STEPS, LocomotionEnv, REWARD_SCALES, quat_rotate
from multi_design_env import MultiDesignEnv


class PaddedLocomotionEnv(LocomotionEnv):
    """Flat T5 evaluation with the universal policy's appended inputs disabled."""

    def observation(self, data, command, action, rng):
        return jp.pad(super().observation(data, command, action, rng), (0, 188))


def checkpoint_observation_size(ckpt_dir):
    value = json.loads((ckpt_dir / "ppo_network_config.json").read_text())["observation_size"]
    return value if isinstance(value, int) else value["shape"][0]


def load_policy(ckpt_dir):
    # T6 진단: checkpoint·정규화·shape 오류는 여기서 발생한다. 저장된 running statistics는
    # T5에서 사용한 것과 정확히 같은 NETWORK로 불러와야 한다.
    # checkpoint.load_policy fails on the saved config (kernel_init_kwargs None -> KeyError),
    # so the network is rebuilt from the training NETWORK.
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    from train_locomotion_policy import NETWORK
    cfg = json.loads((ckpt_dir / "ppo_network_config.json").read_text())
    net = networks.make_ppo_networks(checkpoint_observation_size(ckpt_dir), cfg["action_size"],
                                     preprocess_observations_fn=running_statistics.normalize,
                                     **NETWORK)
    return networks.make_inference_fn(net)(checkpoint.load(str(ckpt_dir)), deterministic=True)


# (name, vx, vy, yaw rate) in the base frame. Training keeps the original random command range (Q9);
# evaluation and DDPO scoring use this symmetric fixed set (2026-09-27 user decision, D51).
EVAL_COMMANDS = (("forward_0.5", .5, 0., 0.), ("backward_0.5", -.5, 0., 0.),
                 ("left_0.5", 0., .5, 0.), ("right_0.5", 0., -.5, 0.), ("forward_1.0", 1., 0., 0.),
                 ("turn_left_0.5", 0., 0., .5), ("turn_right_0.5", 0., 0., -.5))


def evaluate(run, ckpt, seed):
    groups = [[(c, s) for c, s in g] for g in json.loads((run / "designs.json").read_text())]
    designs = [d for g in groups for d in g]
    episodes, K = len(EVAL_COMMANDS), len(groups[0])
    per_group = K * episodes
    # Group g owns env slots [g * per_group, (g+1) * per_group); slot i runs design i % K with
    # command i // K.
    commands = jp.array([[vx, vy, wz, 0.] for _, vx, vy, wz in EVAL_COMMANDS])[np.arange(per_group) // K]
    t = time.time()
    # T6 진단: "built ... designs" 전 실패는 MultiDesignEnv의 형상 생성·그룹화 문제다.
    # 그 이후 실패는 대개 정책 로딩, JAX 컴파일 또는 rollout 문제다.
    env_cls = PaddedLocomotionEnv if checkpoint_observation_size(ckpt) == 257 else LocomotionEnv
    env = MultiDesignEnv(groups, EPISODE_STEPS, hold_command=True, domain_rand=False,
                         env_cls=env_cls)
    print(f"built {len(designs)} designs: {time.time() - t:.0f}s", flush=True)
    policy = load_policy(ckpt)

    def base_pose(state):
        """Free-joint qpos (the base is each design's root body) and (vx, vy) command per env."""
        gs = state.info["groups"]
        return (jp.concatenate([g.data.qpos[:, :7] for g in gs]),
                jp.concatenate([g.info["command"][:, :3] for g in gs]),
                jp.concatenate([g.data.qvel[:, :3] for g in gs]))

    to_base = jax.vmap(lambda q, v: quat_rotate(q, v, inverse=True))
    yaw = lambda q: jp.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                               1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    @jax.jit
    def rollout(rng):
        state = env.reset(jax.random.split(rng, per_group * len(groups)))
        # The reset observation still carries the sampled command; from the first step on the
        # held command is observed.
        state = env._merge([g.replace(info={**g.info, "command": commands})
                            for g in state.info["groups"]])

        def body(carry, key):
            state, alive, ret, length, terms, progress, asked, vel_progress, turn, turn_asked = carry
            q0, command3, _ = base_pose(state)
            command, wz = command3[:, :2], command3[:, 2]
            action, _ = policy(state.obs, key)
            state = env.step(state, action)
            ret += state.reward * alive
            terms = {k: terms[k] + state.metrics[k] * alive for k in terms}
            length += alive
            # The done step already holds the auto-reset pose, so its displacement is dropped.
            moved = alive * (1. - state.done)
            q1, _, v1 = base_pose(state)
            step = (q1[:, :3] - q0[:, :3]).at[:, 2].set(0.)
            speed = jp.linalg.norm(command, axis=-1)
            unit = command / jp.maximum(speed, 1e-6)[:, None]
            progress += jp.sum(to_base(q0[:, 3:7], step)[:, :2] * unit, -1) * moved
            vel_progress += jp.sum(to_base(q1[:, 3:7], v1)[:, :2] * unit, -1) * CTRL_DT * moved
            asked += speed * CTRL_DT * moved
            dyaw = (yaw(q1[:, 3:7]) - yaw(q0[:, 3:7]) + jp.pi) % (2 * jp.pi) - jp.pi
            turn += dyaw * jp.sign(wz) * moved
            turn_asked += jp.abs(wz) * CTRL_DT * moved
            alive *= 1. - state.done  # auto-reset follows a done; mask the rest
            return (state, alive, ret, length, terms, progress, asked, vel_progress, turn, turn_asked), None

        n = per_group * len(groups)
        zero = jp.zeros(n)
        carry = (state, jp.ones(n), zero, zero, {k: zero for k in REWARD_SCALES}, zero, zero, zero, zero, zero)
        (_, alive, ret, length, terms, progress, asked, vel_progress, turn, turn_asked), _ = jax.lax.scan(
            body, carry, jax.random.split(rng, EPISODE_STEPS))
        return ret, length, alive, terms, progress, asked, vel_progress, turn, turn_asked

    t = time.time()
    ret, length, alive, terms, progress, asked, vel_progress, turn, turn_asked = jax.device_get(rollout(jax.random.PRNGKey(seed)))
    print(f"rollout: {time.time() - t:.0f}s", flush=True)
    idx = np.array([g * K + i % K for g in range(len(groups)) for i in range(per_group)])
    cmd_idx = np.tile(np.arange(per_group) // K, len(groups))
    rows = []
    for d, (c, s) in enumerate(designs):
        m = idx == d
        rows.append({"choices": c, "seed": s, "return": float(ret[m].mean()),
                     # EpisodeWrapper also sets done at the time-out, so `alive` ends at 0 for
                     # every env; only trunk contact ends an episode before EPISODE_STEPS.
                     "length": float(length[m].mean()),
                     "survived": float((length[m] >= EPISODE_STEPS).mean()),
                     "cmd_progress_m": float(progress[m].mean()),
                     "cmd_distance_m": float(asked[m].mean()),
                     "cmd_vel_progress_m": float(vel_progress[m].mean()),
                     "cmd_turn_rad": float(turn[m].mean()),
                     "cmd_turn_ratio": float(turn[m].sum() / max(turn_asked[m].sum(), 1e-6)),
                     "cmd_progress_ratio": float(progress[m].sum() / max(asked[m].sum(), 1e-6)),
                     **{k: float(v[m].mean()) for k, v in terms.items()}})
    rows.sort(key=lambda r: -r["return"])
    keys = ("return", "length", "survived", "cmd_progress_m", "cmd_distance_m", "cmd_vel_progress_m",
            *REWARD_SCALES)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["cmd_progress_ratio"] = float(progress.sum() / max(asked.sum(), 1e-6))
    summary["cmd_turn_ratio"] = float(turn.sum() / max(turn_asked.sum(), 1e-6))
    per_command = {}
    for e, (name, *_) in enumerate(EVAL_COMMANDS):
        m = cmd_idx == e
        per_command[name] = {"survived": float((length[m] >= EPISODE_STEPS).mean()),
                             "length": float(length[m].mean()),
                             "cmd_progress_m": float(progress[m].mean()),
                             "cmd_progress_ratio": float(progress[m].sum() / max(asked[m].sum(), 1e-6)),
                             "cmd_turn_ratio": float(turn[m].sum() / max(turn_asked[m].sum(), 1e-6)),
                             **{k: float(v[m].mean()) for k, v in terms.items()}}
    out = run / f"eval_{ckpt.name}.json"
    out.write_text(json.dumps({"checkpoint": str(ckpt), "commands": EVAL_COMMANDS, "seed": seed,
                               "summary": summary, "per_command": per_command, "designs": rows},
                              indent=1))
    print(json.dumps(summary, indent=1))
    for name, s in per_command.items():
        print(f"{name:13s} survived={s['survived']:.2f} len={s['length']:.0f} "
              f"progress={s['cmd_progress_m']:+.2f} m ({s['cmd_progress_ratio'] * 100:.0f}%) "
              f"turn={s['cmd_turn_ratio'] * 100:.0f}%")
    for r in rows[:5] + rows[-5:]:
        print(f"{''.join(map(str, r['choices']))} s{r['seed']:<5} return={r['return']:.3f} "
              f"len={r['length']:.0f} survived={r['survived']:.2f} "
              f"progress={r['cmd_progress_m']:+.2f} (vel {r['cmd_vel_progress_m']:+.2f})/{r['cmd_distance_m']:.2f} m")
    print(f"wrote {out}")


def replay(run, ckpt, rank, seed):
    """One episode of eval-ranked design `rank`; returns (env, design row, qpos, qvel, command, n)."""
    rows = json.loads((run / f"eval_{ckpt.name}.json").read_text())["designs"]
    d = rows[rank]
    print(f"rank {rank}: {''.join(map(str, d['choices']))} seed={d['seed']} "
          f"eval return={d['return']:.3f} len={d['length']:.0f}", flush=True)
    env = LocomotionEnv(d["choices"], d["seed"], domain_rand=False)
    policy = load_policy(ckpt)

    @jax.jit
    def rollout(rng):
        def body(state, key):
            state = env.step(state, policy(state.obs, key)[0])
            return state, (state.data.qpos, state.data.qvel, state.reward, state.done,
                           state.info["command"])
        return jax.lax.scan(body, env.reset(rng), jax.random.split(rng, EPISODE_STEPS))[1]

    qpos, qvel, reward, done, command = jax.device_get(rollout(jax.random.PRNGKey(seed)))
    n = int(np.argmax(done > 0)) + 1 if done.any() else EPISODE_STEPS
    print(f"episode: {n} steps, return={reward[:n].sum():.3f}, "
          f"x displacement={qpos[n - 1, 0] - qpos[0, 0]:+.2f} m, command[0]={command[0, :3]}",
          flush=True)
    return env, d, qpos, qvel, command, n, float(reward[:n].sum())


def view(run, ckpt, rank, seed):
    import mujoco
    from mujoco import viewer
    env, _, qpos, qvel, _, n, _ = replay(run, ckpt, rank, seed)
    data = mujoco.MjData(env.mj_model)
    with viewer.launch_passive(env.mj_model, data) as handle:
        handle.opt.geomgroup[3] = 1  # TT body and leg capsules are in group 3.
        while handle.is_running():
            for i in range(n):
                if not handle.is_running():
                    return
                data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
                mujoco.mj_forward(env.mj_model, data)
                handle.sync()
                time.sleep(env.dt)
            time.sleep(1.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("eval", "view"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ckpt", help="checkpoint step dir name (default: latest)")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    run = (LAUNCH_DIR / args.run).resolve()
    ckpt = run / "checkpoints" / (args.ckpt or sorted(os.listdir(run / "checkpoints"))[-1])
    if args.mode == "eval":
        evaluate(run, ckpt, args.seed)
    else:
        view(run, ckpt, args.rank, args.seed)


if __name__ == "__main__":
    main()
