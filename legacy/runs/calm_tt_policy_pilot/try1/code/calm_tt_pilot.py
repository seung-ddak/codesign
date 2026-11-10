"""Pilot: frozen A1 CALM (HLC+LLC) driving TT morphologies, compared with PD data.

No training. For each (TT shape E, CALM motion) the original A1 command sequence
is replayed through CALM in closed loop on the TT body (MuJoCo CPU). Sit
segments feed the stored A1 LLC latent, as in the original collection.

Start state and fall rule match collect_calm_mujoco.rollout_one (PD data):
reference start pose lifted to foot contact, no settle; fall = upright < 0.5
or trunk z < 0.07.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
WHEEL = ROOT / "runs/menagerie_calm_check/try4/runtime/rl_games-1.1.4-py3-none-any.whl"
MOTIONS = ROOT / "runs/calm_motion_10,000/dataset/accepted"
PD_RUN = WORKSPACE / "codesign/runs/local_mujoco_10k_20261109"
SIT = 10
CONTROL_DT, SUBSTEPS = 1 / 30, 20      # CALM control rate; physics 600 Hz
KP, KD = 40.0, 1.5                     # CALM's trained PD gains
FEET = ("FR", "FL", "RR", "RL")


def choose_shapes(designs, count):
    """Farthest-point sampling over varying E coordinates, starting at e0000."""
    names = sorted(designs)
    E = np.asarray([designs[n] for n in names])
    active = E.std(0) > 1e-9
    X = E[:, active]
    picked = [names.index("e0000")]
    dist = np.linalg.norm(X - X[picked[0]], axis=1)
    while len(picked) < count:
        i = int(dist.argmax())
        picked.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i], axis=1))
    return [names[i] for i in picked]


def choose_motions(shapes, per_shape, seed):
    """Per shape, motions from its own PD set: 10 s long and starting upright."""
    rng = np.random.default_rng(seed)
    pairs, used = [], set()
    for shape in shapes:
        with np.load(PD_RUN / "rollouts" / shape / "variant_0.npz") as z:
            ids = [int(m.split("_")[1]) for m in z["motions"]]
        ok = []
        for i in rng.permutation(ids):
            if i in used:
                continue
            with np.load(MOTIONS / f"{i:05d}.npz") as m:
                if len(m["q"]) == 301 and int(m["motion_index"][0]) != 11:
                    ok.append(int(i))
            if len(ok) == per_shape:
                break
        used.update(ok)
        pairs += [(shape, i) for i in ok]
    return pairs


_W = {}


def worker_init():
    import torch
    torch.set_num_threads(1)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_menagerie_mjx_policy import load_policy
    _W["policy"], joints = load_policy(WORKSPACE, WHEEL.resolve())
    assert joints == [f"{l}_{p}_joint" for l in ("FL", "FR", "RL", "RR") for p in ("hip", "thigh", "calf")]


def make_model(E):
    import mujoco
    from pd_replay import build_model
    from check_calm_tt_direct import check_body_mapping, joint_addresses
    model = build_model("00000011", uniforms=np.asarray(E, float), torque_limit=33.5, flat_ground=True)
    model.opt.timestep = CONTROL_DT / SUBSTEPS
    model.actuator_gainprm[:] = 0       # torques come from CALM's PD below only
    model.actuator_biasprm[:] = 0
    mapping = check_body_mapping(model)
    names, qa, va = joint_addresses(model)
    return model, mapping, qa, va


def run(job):
    import mujoco
    from check_calm_tt_direct import physical_body_states, CALM_BODIES
    shape, E, motion, torque = job["shape"], job["E"], job["motion"], job["torque"]
    model, mapping, qa, va = make_model(E)
    ref = dict(np.load(MOTIONS / f"{motion:05d}.npz"))
    T = len(ref["command"])
    sit = np.zeros(T, bool)
    for k, a, b in zip(ref["motion_index"], ref["segment_start"], ref["segment_end"]):
        if int(k) == SIT:
            sit[a:b] = True
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    d.qpos[:] = model.keyframe("home").qpos
    d.qpos[:2] = 0
    d.qpos[3:7] = ref["root"][0, [6, 3, 4, 5]] / np.linalg.norm(ref["root"][0, 3:7])
    d.qpos[qa] = ref["q"][0]
    mujoco.mj_forward(model, d)
    feet = np.array([model.geom(f + "_foot").id for f in FEET])
    floor = model.geom("floor").id
    d.qpos[2] -= float((d.geom_xpos[feet, 2] - model.geom_size[feet, 0]).min())
    mujoco.mj_forward(model, d)

    def contacts():
        f = np.zeros(4, bool)
        for c in d.contact[:d.ncon]:
            if c.dist <= 1e-3 and floor in (c.geom1, c.geom2):
                f |= feet == (c.geom2 if c.geom1 == floor else c.geom1)
        return f

    qpos, qvel, con, act = [d.qpos.copy()], [d.qvel.copy()], [contacts()], []
    fell = False
    for t in range(T):
        up = 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)
        if up < .5 or d.qpos[2] < .07:
            fell = True
            break
        phys = physical_body_states(model, d, mapping)
        target = _W["policy"](phys, CALM_BODIES, ref["command"][t],
                              latent=ref["latent"][t] if sit[t] else None)
        for _ in range(SUBSTEPS):
            tau = np.clip(KP * (target - d.qpos[qa]) - KD * d.qvel[va], -torque, torque)
            d.qfrc_applied[:] = 0
            d.qfrc_applied[va] = tau
            mujoco.mj_step(model, d)
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            fell = True
            break
        act.append(target)
        qpos.append(d.qpos.copy()); qvel.append(d.qvel.copy()); con.append(contacts())
    up = 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)
    fell = fell or up < .5 or d.qpos[2] < .07
    out = Path(job["out"]) / "episodes" / f"{shape}_{motion:05d}.npz"
    np.savez_compressed(out, qpos=np.asarray(qpos, np.float32), qvel=np.asarray(qvel, np.float32),
                        contact=np.asarray(con), target_q=np.asarray(act, np.float32).reshape(-1, 12),
                        E=np.asarray(E, np.float32), motion=motion, shape=shape, fell=fell,
                        dt=CONTROL_DT, steps=len(act), torque_limit=torque)
    return dict(shape=shape, motion=motion, steps=len(act), fell=bool(fell))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shapes", type=int, default=16)
    p.add_argument("--per-shape", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--torque", type=float, default=33.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="run only the first N jobs (smoke test)")
    a = p.parse_args()
    if a.out.exists():
        p.error(f"Output exists, not overwriting: {a.out}")
    designs = json.load(open(PD_RUN / "designs.json"))
    train = {k: v for k, v in designs.items() if (PD_RUN / "rollouts" / k).is_dir()}
    shapes = choose_shapes(train, a.shapes)
    pairs = choose_motions(shapes, a.per_shape, a.seed)
    motions = [m for _, m in pairs]
    (a.out / "episodes").mkdir(parents=True)
    json.dump(dict(shapes={s: train[s] for s in shapes}, motions=motions,
                   pd_pairs=[[s, m] for s, m in pairs], torque_limit=a.torque,
                   control_hz=30, physics_hz=600, kp=KP, kd=KD,
                   policy="frozen A1 CALM HLC+LLC; sit segments use stored A1 latent",
                   start="A1 reference frame-0 joints and base orientation, lifted to foot contact",
                   fall="upright < 0.5 or trunk z < 0.07 (same as PD data)"),
              open(a.out / "config.json", "w"), indent=1)
    jobs = [dict(shape=s, E=train[s], motion=m, torque=a.torque, out=str(a.out))
            for s in shapes for m in motions]
    if a.limit:
        jobs = jobs[:a.limit]
    import multiprocessing as mp
    import time
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=worker_init) as pool, \
            open(a.out / "episodes.jsonl", "w") as log:
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=2)):
            log.write(json.dumps(r) + "\n"); log.flush()
            if (i + 1) % 32 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
