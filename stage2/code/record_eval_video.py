"""Offscreen mp4 of eval-ranked designs (headless counterpart of `eval_locomotion_policy.py view`).

  python stage2/code/record_eval_video.py --run stage2/runs/<run> --out stage2/runs/<run>/videos --ranks 0 63 -1

Same single-design rollout as `view`: random commands (seed-fixed), no domain randomisation (D52). Writes <out>/rank_<r>.mp4 and
<out>/videos.json (design, return, episode length, commands).

--fixed (2026-09-27 user request) instead plays the evaluation's EVAL_COMMANDS back to back,
SEGMENT_S seconds each, held (no resampling / heading control), and writes fixed_rank_<r>.mp4 and
fixed_videos.json with the progress along each command.

Render-only markers (2026-09-27 user request): front legs orange, rear legs grey, and a green
arrow above the base pointing along the commanded (vx, vy), length proportional to its speed, and a green curved arrow around it for the commanded yaw rate (direction of turn, arc length ~ rate).
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_locomotion_policy import EVAL_COMMANDS, LAUNCH_DIR, load_policy, replay  # noqa: E402  (chdirs into TT)
from locomotion_env import CTRL_DT, LocomotionEnv  # noqa: E402

import imageio  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


FRONT_RGBA, REAR_RGBA, ARROW_RGBA = (.92, .42, .16, 1.), (.55, .55, .55, 1.), (.1, .8, .3, 1.)
ARROW_HEIGHT, ARROW_M_PER_MPS, ARROW_WIDTH = .35, .6, .012  # above the base; length scale


def marked_model(m):
    """Copy of m with front legs (left_leg, right_leg) and rear legs (*_leg_1) recoloured."""
    m = copy.deepcopy(m)
    for g in range(m.ngeom):
        leg = m.body(m.geom_bodyid[g]).name.split("/")[-2] if "/" in m.body(m.geom_bodyid[g]).name else ""
        if leg in ("left_leg", "right_leg", "left_leg_1", "right_leg_1"):
            m.geom_matid[g] = -1  # a material colour would override geom_rgba
            m.geom_rgba[g] = REAR_RGBA if leg.endswith("_1") else FRONT_RGBA
    return m


TURN_RADIUS, TURN_RAD_PER_RADPS, TURN_SEGMENTS = .22, 1.5 * np.pi, 12  # 1 rad/s -> 270 deg arc


def add_connector(scene, kind, start, end):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array(ARROW_RGBA, np.float32))
    mujoco.mjv_connector(geom, kind, ARROW_WIDTH, start, end)
    scene.ngeom += 1


def add_command_arrow(scene, data, command):
    """Green markers above the base for the body-frame command: a straight arrow along (vx, vy)
    (length ~ speed) and a curved arrow around the base for the yaw rate (counter-clockwise seen
    from above = positive = turn left; arc length ~ |yaw rate|)."""
    top = data.qpos[:3] + np.array([0., 0., ARROW_HEIGHT])
    speed = float(np.linalg.norm(command[:2]))
    if speed > 1e-3:
        world = np.zeros(3)
        mujoco.mju_rotVecQuat(world, np.array([command[0], command[1], 0.]), data.qpos[3:7])
        world[2] = 0.
        end = top + world / max(np.linalg.norm(world), 1e-6) * speed * ARROW_M_PER_MPS
        add_connector(scene, mujoco.mjtGeom.mjGEOM_ARROW, top, end)
    yaw_rate = float(command[2])
    if abs(yaw_rate) > 1e-3:
        w, x, y, z = data.qpos[3:7]
        heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        span = np.sign(yaw_rate) * min(abs(yaw_rate) * TURN_RAD_PER_RADPS, 1.9 * np.pi)
        angles = heading + np.linspace(0., span, TURN_SEGMENTS + 1)
        pts = top + TURN_RADIUS * np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], 1)
        for k in range(TURN_SEGMENTS):
            kind = mujoco.mjtGeom.mjGEOM_ARROW if k == TURN_SEGMENTS - 1 else mujoco.mjtGeom.mjGEOM_CAPSULE
            add_connector(scene, kind, pts[k], pts[k + 1])


def record(env, qpos, command, n, path, width=640, height=480):
    m = marked_model(env.mj_model)
    # Near/far clipping and fog scale with the model extent; the TaskEnv heightfield (130 x 210 m)
    # would clip the robot at the 2.2 m tracking distance.
    m.stat.extent = min(m.stat.extent, 3.)
    data = mujoco.MjData(m)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1  # TT body and leg capsules are in group 3.
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, 1  # body 1 = free base
    cam.distance, cam.elevation, cam.azimuth = 2.2, -20., 135.
    with mujoco.Renderer(m, height, width) as r, \
            imageio.get_writer(path, fps=round(1 / env.dt), macro_block_size=1) as w:
        for i in range(n):
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(m, data)
            r.update_scene(data, camera=cam, scene_option=opt)
            add_command_arrow(r.scene, data, command[i])
            w.append_data(r.render())


SEGMENT_S = 5.


def fixed_replay(run, ckpt, rank, seed):
    """Eval-ranked design `rank` driven through EVAL_COMMANDS, SEGMENT_S seconds each."""
    rows = json.loads((run / f"eval_{ckpt.name}.json").read_text())["designs"]
    d = rows[rank]
    env = LocomotionEnv(d["choices"], d["seed"], hold_command=True, domain_rand=False)
    policy = load_policy(ckpt)
    seg = int(SEGMENT_S / CTRL_DT)
    commands = jp.repeat(jp.array([[vx, vy, wz, 0.] for _, vx, vy, wz in EVAL_COMMANDS]), seg, axis=0)

    @jax.jit
    def rollout(rng):
        def body(state, xs):
            key, command = xs
            state = state.replace(info={**state.info, "command": command})
            state = env.step(state, policy(state.obs, key)[0])
            return state, (state.data.qpos, state.done)
        steps = len(commands)
        return jax.lax.scan(body, env.reset(rng), (jax.random.split(rng, steps), commands))[1]

    qpos, done = jax.device_get(rollout(jax.random.PRNGKey(seed)))
    commands = np.asarray(commands)
    n = int(np.argmax(done > 0)) + 1 if done.any() else len(commands)
    segments = []
    for k, (name, vx, vy, wz) in enumerate(EVAL_COMMANDS):
        lo, hi = k * seg, min((k + 1) * seg, n)
        if lo >= n:
            segments.append({"command": name, "steps": 0})
            continue
        prev = qpos[lo - 1] if lo else qpos[0]
        step = np.diff(np.vstack([prev[None], qpos[lo:hi]]), axis=0)[:, :3]
        along, unit = 0., np.array([vx, vy, 0.]) / max(np.hypot(vx, vy), 1e-6)
        for q, s in zip(np.vstack([prev[None], qpos[lo:hi - 1]]), step):
            body_step = np.zeros(3)
            mujoco.mju_rotVecQuat(body_step, s * np.array([1., 1., 0.]),
                                  q[3:7] * np.array([1., -1., -1., -1.]))  # world -> base
            along += float(body_step @ unit)
        w, x, y, z = np.vstack([prev[None], qpos[lo:hi]])[:, 3:7].T
        yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        segments.append({"command": name, "steps": hi - lo,
                         "progress_m": along if (vx or vy) else None,
                         "asked_m": float(np.hypot(vx, vy) * (hi - lo) * CTRL_DT),
                         "turn_rad": float((yaw[-1] - yaw[0]) * np.sign(wz)) if wz else None,
                         "turn_asked_rad": float(abs(wz) * (hi - lo) * CTRL_DT)})
    print(f"rank {rank}: {''.join(map(str, d['choices']))} seed={d['seed']} {n} steps "
          f"{[(s['command'], s.get('progress_m')) for s in segments]}", flush=True)
    return env, d, qpos, commands, n, segments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ckpt", help="checkpoint step dir name (default: latest)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ranks", type=int, nargs="+", default=[0, 63, -1])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fixed", action="store_true", help="EVAL_COMMANDS back to back instead")
    args = p.parse_args()
    run = (LAUNCH_DIR / args.run).resolve()
    out = (LAUNCH_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = run / "checkpoints" / (args.ckpt or sorted(os.listdir(run / "checkpoints"))[-1])
    meta = []
    if args.fixed:
        for rank in args.ranks:
            env, d, qpos, command, n, segments = fixed_replay(run, ckpt, rank, args.seed)
            name = f"fixed_rank_{rank}.mp4"
            record(env, qpos, command, n, out / name)
            meta.append({"file": name, "rank": rank, "choices": "".join(map(str, d["choices"])),
                         "seed": d["seed"], "eval_return": d["return"],
                         "eval_survived": d["survived"], "video_steps": n, "segments": segments})
            print(f"wrote {out / name}", flush=True)
        (out / "fixed_videos.json").write_text(json.dumps(meta, indent=1))
        return
    for rank in args.ranks:
        env, d, qpos, _, command, n, ret = replay(run, ckpt, rank, args.seed)
        name = f"rank_{rank}.mp4"
        record(env, qpos, command, n, out / name)
        # Commands are resampled every 10 s; keep the distinct ones for the caption.
        cmds = np.unique(np.round(command[:n, :2], 2), axis=0).tolist()  # vx, vy
        meta.append({"file": name, "rank": rank, "choices": "".join(map(str, d["choices"])),
                     "seed": d["seed"], "eval_return": d["return"], "eval_survived": d["survived"],
                     "video_return": ret, "video_steps": n,
                     "x_displacement_m": float(qpos[n - 1, 0] - qpos[0, 0]), "commands": cmds[:4]})
        print(f"wrote {out / name}", flush=True)
    (out / "videos.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
