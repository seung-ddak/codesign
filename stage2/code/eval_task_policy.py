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


def quat_yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", help="checkpoint step dir name (default: latest)")
    p.add_argument("--level", type=int, default=3, help="stairs difficulty row (step 0.05 + 0.018 * level m)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--videos", action="store_true")
    p.add_argument("--reuse", action="store_true", help="recompute from the saved rollout_<ckpt>.npz")
    args = p.parse_args()
    run = (LAUNCH_DIR / args.run).resolve()
    ckpt = run / "checkpoints" / (args.ckpt or sorted(os.listdir(run / "checkpoints"))[-1])
    groups = [[(c, s) for c, s in g] for g in json.loads((run / "designs.json").read_text())]
    K, S = len(groups[0]), len(SCENARIOS)
    per_group = K * S
    env = MultiDesignEnv(groups, EPISODE_STEPS, env_cls=TaskEnv, hold_command=True, domain_rand=False)
    probe = env.groups[0].env.env.env  # TaskEnv for the shared terrain lookups
    policy = load_policy(ckpt)
    scen = np.arange(per_group) // K  # group slot i: design i % K, scenario i // K
    commands = jp.array([c for _, c, _, _ in SCENARIOS])[scen]
    level = jp.array([args.level if lv else 0 for *_, lv in SCENARIOS], jp.int32)[scen]
    col = jp.array([c for _, _, c, _ in SCENARIOS], jp.int32)[scen]

    @jax.jit
    def rollout(key):
        state = env.reset(jax.random.split(key, per_group * len(groups)))
        state = env._merge([g.replace(info={**g.info, "command": commands,
                                             PRESERVE: {"level": level, "col": col}})
                            for g in state.info["groups"]])

        def body(carry, k):
            state, alive = carry
            state = env.step(state, policy(state.obs, k)[0])
            # On a done step the auto-reset already replaced data and info, so a recorded step is
            # valid only while the episode is still running after it.
            alive = alive * (1. - state.done)
            gs = state.info["groups"]
            row = (tuple(g.data.qpos for g in gs), jp.concatenate([g.info["last_contacts"] for g in gs]),
                   alive)
            return (state, alive), row

        return jax.lax.scan(body, (state, jp.ones(per_group * len(groups))),
                            jax.random.split(key, EPISODE_STEPS))[1]

    raw = run / f"rollout_{ckpt.name}.npz"
    if args.reuse:
        z = np.load(raw)
        qpos_groups = [z[f"qpos_{g}"] for g in range(len(groups))]
        contacts, alive = z["contacts"], z["alive"]
    else:
        qpos_groups, contacts, alive = jax.device_get(rollout(jax.random.PRNGKey(args.seed)))
        np.savez(raw, contacts=contacts, alive=alive, **{f"qpos_{g}": q for g, q in enumerate(qpos_groups)})
    base = np.concatenate([q[:, :, :7] for q in qpos_groups], axis=1)  # (T, N, 7)
    ground = np.asarray(probe.terrain_height(jp.asarray(base[..., :2])))
    n_alive = alive.sum(0)  # valid recorded steps; survivors have EPISODE_STEPS - 1 (the time-out step is done)
    last = np.maximum(n_alive.astype(int) - 1, 0)
    N = base.shape[1]
    # Per-step base-frame displacement (yaw only) projected on the commanded direction.
    step = np.diff(base[..., :2], axis=0)
    yaw = quat_yaw(base[..., 3:7])
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    body = np.stack([c * step[..., 0] + s * step[..., 1], -s * step[..., 0] + c * step[..., 1]], -1)
    cmd = np.asarray(commands)[np.arange(N) % per_group]
    unit = cmd[:, :2] / np.maximum(np.linalg.norm(cmd[:, :2], axis=-1, keepdims=True), 1e-6)
    moved = (alive[1:] * alive[:-1])
    progress = ((body * unit).sum(-1) * moved).sum(0)
    turn = ((np.diff(np.unwrap(yaw, axis=0), axis=0) * moved).sum(0)) * np.sign(cmd[:, 2])
    stand = np.repeat([g.env.env.env.stand_height for g in env.groups], per_group)  # home z per structure
    rise = base[..., 2] - ground - stand
    rows = []
    for g, group in enumerate(groups):
        for i in range(per_group):
            n = g * per_group + i
            d = group[i % K]
            a = alive[:, n] > 0
            rows.append({"choices": d[0], "seed": d[1], "scenario": SCENARIOS[i // K][0],
                         "survived": float(n_alive[n] >= EPISODE_STEPS - 1), "steps": float(n_alive[n]),
                         "progress_m": float(progress[n]), "turn_rad": float(turn[n]),
                         "climb_m": float(ground[last[n], n] - ground[0, n]),
                         "jump_rise_m": float(rise[a, n].max()) if a.any() else 0.,
                         "air_frac": float((~contacts[a, n].any(-1)).mean()) if a.any() else 0.,
                         "index": n})
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
                               "scenarios": per, "designs": [{k: v for k, v in x.items() if k != "index"}
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
    offsets = np.cumsum([0] + [q.shape[1] for q in qpos_groups])
    for name, *_ in SCENARIOS:
        r = sorted((x for x in rows if x["scenario"] == name), key=lambda x: x["steps"] + x["progress_m"])
        x = r[len(r) // 2]  # median design of this scenario
        g = int(np.searchsorted(offsets, x["index"], side="right") - 1)
        q = qpos_groups[g][:, x["index"] - offsets[g]]
        one = TaskEnv(list(x["choices"]), x["seed"], hold_command=True, domain_rand=False)
        n = max(int(x["steps"]), 1)
        record(one, q, np.repeat(cmd[x["index"]][None], n, 0), n, vids / f"{name}.mp4")
        meta.append({"file": f"{name}.mp4", "scenario": name, "choices": "".join(map(str, x["choices"])),
                     "seed": x["seed"], **{k: x[k] for k in ("survived", "steps", "progress_m", "turn_rad",
                                                             "climb_m", "jump_rise_m", "air_frac")}})
        print(f"wrote {vids / name}.mp4", flush=True)
    (vids / "task_videos.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
