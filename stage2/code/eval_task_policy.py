"""T8 TaskEnv policy evaluation + videos (DR off, held commands, D52).

  python stage2/code/eval_task_policy.py --run stage2/runs/T8/pi_smoke [--level 3] [--videos]

Every design in <run>/designs.json runs each SCENARIO once for 20 s on a fixed terrain tile (the
first step moves the base there, as the curriculum does). Writes <run>/eval_task_<ckpt>.json with
per-scenario summaries and per-design rows; --videos also renders the median design of each
scenario to <run>/videos/<scenario>.mp4 (+ videos/task_videos.json).

Metrics: survived (no trunk contact for 20 s), progress_m (base displacement along the commanded
direction, base frame), turn_rad (yaw change along the commanded sign), climb_m (ground height under
the base, end - start: stairs up > 0, stairs down < 0), jump_rise_m (max base rise above standing
height over the ground), air_frac (share of alive steps with all four feet off the ground).
"""

import argparse
import gc
import json
import os
from pathlib import Path

from eval_locomotion_policy import LAUNCH_DIR, load_policy  # noqa: E402  (chdirs into TT)

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402

import terrain as T  # noqa: E402
from locomotion_env import CTRL_DT, EPISODE_STEPS  # noqa: E402
from multi_design_env import MultiDesignEnv  # noqa: E402
from task_env import PRESERVE, TaskEnv  # noqa: E402

EDGES = np.cumsum(T.TERRAIN_PROPORTIONS)
KIND = [int(np.searchsorted(EDGES, j / T.NUM_COLS + 0.001, side="right")) for j in range(T.NUM_COLS)]
FLAT, STAIRS_UP, STAIRS_DOWN = KIND.index(0), KIND.index(2), KIND.index(3)
# (name, command (vx, vy, yaw, heading, jump), terrain column, use --level (else row 0 = flat))
SCENARIOS = (("flat_forward_0.5", (.5, 0., 0., 0., 0.), FLAT, False),
             ("flat_turn_left_0.5", (0., 0., .5, 0., 0.), FLAT, False),
             ("jump", (0., 0., 0., 0., 1.), FLAT, False),
             ("stairs_up_forward_0.5", (.5, 0., 0., 0., 0.), STAIRS_UP, True),
             ("stairs_down_forward_0.5", (.5, 0., 0., 0., 0.), STAIRS_DOWN, True))


def design_batches(group, size):
    if size < 1:
        raise ValueError("--design-batch-size must be positive")
    if len(group) % size:
        raise ValueError("--design-batch-size must divide designs per structure group")
    return [group[i:i + size] for i in range(0, len(group), size)]


def quat_yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", help="checkpoint step dir name (default: latest)")
    p.add_argument("--level", type=int, default=3, help="stairs difficulty row (step 0.05 + 0.018 * level m)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--design-batch-size", type=int, default=1,
                   help="designs per structure compiled at once (default: 1; lower uses less RAM)")
    p.add_argument("--videos", action="store_true")
    p.add_argument("--reuse", action="store_true", help="recompute from saved rollout batch files")
    p.add_argument("--resume", action="store_true", help="reuse completed batches and continue missing ones")
    args = p.parse_args()
    run = (LAUNCH_DIR / args.run).resolve()
    ckpt = run / "checkpoints" / (args.ckpt or sorted(os.listdir(run / "checkpoints"))[-1])
    groups = [[(c, s) for c, s in g] for g in json.loads((run / "designs.json").read_text())]
    raw = run / f"rollout_{ckpt.name}"
    raw.mkdir(exist_ok=True)
    policy = None if args.reuse else load_policy(ckpt)
    rows, S = [], len(SCENARIOS)
    for g_idx, group in enumerate(groups):
        batches = design_batches(group, args.design_batch_size)
        first = batches[0]
        K, per_group = len(first), len(first) * S
        env = MultiDesignEnv([first], EPISODE_STEPS, env_cls=TaskEnv,
                             hold_command=True, domain_rand=False)
        inner = env.groups[0].env.env
        probe = inner.env
        scen = np.arange(per_group) // K
        commands = jp.array([c for _, c, _, _ in SCENARIOS])[scen]
        level = jp.array([args.level if lv else 0 for *_, lv in SCENARIOS], jp.int32)[scen]
        col = jp.array([c for _, _, c, _ in SCENARIOS], jp.int32)[scen]

        @jax.jit
        def rollout(designs, key):
            saved, inner.designs = inner.designs, designs
            state = env.reset(jax.random.split(key, per_group))
            group_state = state.info["groups"][0]
            state = env._merge([group_state.replace(info={**group_state.info, "command": commands,
                                                           PRESERVE: {"level": level, "col": col}})])

            def body(carry, key):
                state, alive = carry
                state = env.step(state, policy(state.obs, key)[0])
                alive = alive * (1. - state.done)
                group_state = state.info["groups"][0]
                return (state, alive), (group_state.data.qpos,
                                        group_state.info["last_contacts"], alive)

            result = jax.lax.scan(body, (state, jp.ones(per_group)),
                                  jax.random.split(key, EPISODE_STEPS))[1]
            inner.designs = saved
            return result

        for b, batch in enumerate(batches):
            path = raw / f"group_{g_idx:02d}_batch_{b:03d}.npz"
            cached = args.reuse or (args.resume and path.exists())
            if cached:
                if not path.exists():
                    raise FileNotFoundError(path)
                with np.load(path) as z:
                    qpos, contacts, alive, batch_stand = (z[k] for k in ("qpos", "contacts", "alive", "stand"))
            else:
                other = env if b == 0 else MultiDesignEnv([batch], EPISODE_STEPS, env_cls=TaskEnv,
                                                           hold_command=True, domain_rand=False)
                designs = other.groups[0].env.env.designs
                batch_stand = other.groups[0].env.env.stand_heights
                qpos, contacts, alive = jax.device_get(rollout(
                    designs, jax.random.fold_in(jax.random.PRNGKey(args.seed), g_idx * len(batches) + b)))
                np.savez(path, qpos=qpos, contacts=contacts, alive=alive, stand=batch_stand)
                if other is not env:
                    del other
            base = qpos[..., :7]
            ground = np.asarray(probe.terrain_height(jp.asarray(base[..., :2])))
            n_alive = alive.sum(0)
            last = np.maximum(n_alive.astype(int) - 1, 0)
            step = np.diff(base[..., :2], axis=0)
            yaw = quat_yaw(base[..., 3:7])
            c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
            body = np.stack([c * step[..., 0] + s * step[..., 1],
                             -s * step[..., 0] + c * step[..., 1]], -1)
            cmd = np.asarray(commands)
            unit = cmd[:, :2] / np.maximum(np.linalg.norm(cmd[:, :2], axis=-1, keepdims=True), 1e-6)
            moved = alive[1:] * alive[:-1]
            progress = ((body * unit).sum(-1) * moved).sum(0)
            turn = (np.diff(np.unwrap(yaw, axis=0), axis=0) * moved).sum(0) * np.sign(cmd[:, 2])
            rise = base[..., 2] - ground - np.tile(batch_stand, S)
            for i in range(per_group):
                d, a = batch[i % K], alive[:, i] > 0
                rows.append({"choices": d[0], "seed": d[1], "scenario": SCENARIOS[i // K][0],
                             "survived": float(n_alive[i] >= EPISODE_STEPS - 1),
                             "steps": float(n_alive[i]), "progress_m": float(progress[i]),
                             "turn_rad": float(turn[i]),
                             "climb_m": float(ground[last[i], i] - ground[0, i]),
                             "jump_rise_m": float(rise[a, i].max()) if a.any() else 0.,
                             "air_frac": float((~contacts[a, i].any(-1)).mean()) if a.any() else 0.,
                             "_batch": b, "_group": g_idx, "_local": i})
            print(f"group {g_idx + 1}/{len(groups)} batch {b + 1}/{len(batches)} complete", flush=True)
            del qpos, contacts, alive, batch_stand, base, ground, n_alive, last
            del step, yaw, c, s, body, cmd, unit, moved, progress, turn, rise
            if not cached:
                del designs
                jax.clear_caches()
                gc.collect()
        del rollout, env, inner, probe
        jax.clear_caches()
        gc.collect()
    per = {}
    for name, command, column, use_level in SCENARIOS:
        r = [x for x in rows if x["scenario"] == name]
        mean = lambda k: float(np.mean([x[k] for x in r]))
        per[name] = {"command": command, "column": column, "level": args.level if use_level else 0,
                     **{k: mean(k) for k in ("survived", "steps", "progress_m", "turn_rad", "climb_m",
                                             "jump_rise_m", "air_frac")},
                     "asked_m": float(np.hypot(*command[:2]) * EPISODE_STEPS * CTRL_DT),
                     "asked_turn_rad": float(abs(command[2]) * EPISODE_STEPS * CTRL_DT)}
    out = run / f"eval_task_{ckpt.name}.json"
    out.write_text(json.dumps({"checkpoint": str(ckpt), "level": args.level, "seed": args.seed,
                               "scenarios": per, "designs": [{k: v for k, v in x.items() if not k.startswith("_")}
                                                             for x in rows]}, indent=1))
    for name, v in per.items():
        print(f"{name:24s} survived={v['survived']:.2f} steps={v['steps']:.0f} progress={v['progress_m']:+.2f}/"
              f"{v['asked_m']:.0f} m turn={v['turn_rad']:+.2f}/{v['asked_turn_rad']:.0f} climb={v['climb_m']:+.3f} "
              f"rise={v['jump_rise_m']:.3f} air={v['air_frac']:.2f}", flush=True)
    print(f"wrote {out}", flush=True)
    if not args.videos:
        return

    from record_eval_video import record
    vids = run / "videos"
    vids.mkdir(exist_ok=True)
    meta = []
    for name, *_ in SCENARIOS:
        r = sorted((x for x in rows if x["scenario"] == name), key=lambda x: x["steps"] + x["progress_m"])
        x = r[len(r) // 2]  # median design of this scenario
        with np.load(raw / f"group_{x['_group']:02d}_batch_{x['_batch']:03d}.npz") as z:
            q = z["qpos"][:, x["_local"]]
        one = TaskEnv(list(x["choices"]), x["seed"], hold_command=True, domain_rand=False)
        n = max(int(x["steps"]), 1)
        command = np.asarray(next(c for scenario, c, *_ in SCENARIOS if scenario == name))
        record(one, q, np.repeat(command[None], n, 0), n, vids / f"{name}.mp4")
        meta.append({"file": f"{name}.mp4", "scenario": name, "choices": "".join(map(str, x["choices"])),
                     "seed": x["seed"], **{k: x[k] for k in ("survived", "steps", "progress_m", "turn_rad",
                                                             "climb_m", "jump_rise_m", "air_frac")}})
        print(f"wrote {vids / name}.mp4", flush=True)
    (vids / "task_videos.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
