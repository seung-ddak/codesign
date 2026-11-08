"""Fast resumable 10k CALM -> multi-morphology MuJoCo CPU dynamics -> TT training.

No MJX physics or PPO training is used in this alternative collection path.
The action is the actual clipped reference-angle PD command, not a learned policy.
Run from codesign root; see docs/V100_MUJOCO_10K.md.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "transformer-transformer"))
from shared_calm_common import create_plan, read, save
from collect_pd_rollouts import save_rollout

DT = 0.02
HORIZON = 500
LEGS = ("FR", "FL", "RR", "RL")


def make_plan(source, out, *, seed=20261108):
    if (out / "plan.json").exists():
        raise FileExistsError("Existing plan; use status or run (never overwrite a dataset)")
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(seed=seed, source_motions=10000, train_designs=200,
               eval_designs=20, morphologies_per_motion=2, rollouts_per_pair=2)
    plan = create_plan(source, cfg)
    save(out / "plan.json", plan)
    save(out / "designs.json", {r["design"]: r["E"] for r in plan["designs"] + plan["eval_designs"]})
    save(out / "status.json", dict(stage="initialized", expected_motions=10000,
                                    training_designs=200, heldout_designs=20,
                                    expected_training_episodes=40000,
                                    backend="MuJoCo CPU", control_hz=50, physics_hz=200))
    return plan


def contact_flags(model, data, feet, floor):
    flags = np.zeros(4, dtype=np.float32)
    for c in data.contact[:data.ncon]:
        if c.dist > 1e-3:
            continue
        if c.geom1 == floor:
            flags |= np.asarray([c.geom2 == f for f in feet], dtype=bool)
        elif c.geom2 == floor:
            flags |= np.asarray([c.geom1 == f for f in feet], dtype=bool)
    return flags


def reference_targets(ref, start, count):
    """Linear joint interpolation from retargeted 30Hz reference at 50Hz."""
    q = np.asarray(ref["fix_q"], dtype=np.float64)
    t = start + np.arange(count) * (1.0 / 50) / (1.0 / 30)
    t = np.minimum(t, len(q) - 1)
    j = np.minimum(np.floor(t).astype(np.int64), len(q) - 2)
    a = (t - j)[:, None]
    return q[j] * (1 - a) + q[j + 1] * a


def rollout_one(model, ref, variant, seed):
    import mujoco
    qref = np.asarray(ref["fix_q"], dtype=np.float64)
    root = np.asarray(ref["fix_base_pos"], dtype=np.float64)
    quat = np.asarray(ref["fix_base_quat"], dtype=np.float64)
    if len(qref) < 4 or root.shape != (len(qref), 3) or quat.shape != (len(qref), 4):
        raise ValueError("Invalid retargeted reference dimensions")
    # The second start samples a different physical motion window.
    rng = np.random.default_rng(seed)
    start = 0 if variant == 0 else int(rng.integers(max(1, len(qref)//5), max(2, len(qref)//2)))
    L = min(HORIZON, int((len(qref) - start - 1) * (50.0 / 30.0)))
    if L < 2:
        raise ValueError("Reference too short")
    target = reference_targets(ref, start, L)
    lo = model.actuator_ctrlrange[:12, 0].copy()
    hi = model.actuator_ctrlrange[:12, 1].copy()
    if not np.all(model.actuator_ctrllimited[:12]):
        lo[~model.actuator_ctrllimited[:12].astype(bool)] = -np.inf
        hi[~model.actuator_ctrllimited[:12].astype(bool)] = np.inf
    target = np.clip(target, lo, hi)
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    d.qpos[:] = model.keyframe("home").qpos
    d.qpos[:2] = 0
    d.qpos[2] = max(.1, root[start, 2])
    d.qpos[3:7] = quat[start] / np.linalg.norm(quat[start])
    d.qpos[7:19] = qref[start]
    d.ctrl[:12] = target[0]
    mujoco.mj_forward(model, d)
    feet = np.array([model.geom(leg + "_foot").id for leg in LEGS])
    floor = model.geom("floor").id
    # Restore foot-floor support; never reset the simulation during an episode.
    clearance = d.geom_xpos[feet, 2] - model.geom_size[feet, 0]
    d.qpos[2] -= float(clearance.min())
    mujoco.mj_forward(model, d)
    qpos = np.zeros((HORIZON + 1, model.nq), np.float32)
    qvel = np.zeros((HORIZON + 1, model.nv), np.float32)
    contact = np.zeros((HORIZON + 1, 4), np.float32)
    actions = np.zeros((HORIZON, 12), np.float32)
    valid = np.zeros(HORIZON, bool)
    for t in range(L):
        qpos[t], qvel[t] = d.qpos, d.qvel
        contact[t] = contact_flags(model, d, feet, floor)
        # Do not write a transition after falling or nonfinite state.
        w, x, y, z = d.qpos[3:7]
        upright = 1 - 2 * (x*x + y*y)
        if upright < .5 or d.qpos[2] < .07 or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            break
        d.ctrl[:12] = target[t]
        actions[t] = d.ctrl[:12]
        mujoco.mj_step(model, d, nstep=4)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            # Last physics step failed; keep the previous finite state only.
            d.qpos[:] = qpos[t]
            d.qvel[:] = qvel[t]
            mujoco.mj_forward(model, d)
            break
        valid[t] = True
    length = int(valid.sum())
    qpos[length], qvel[length] = d.qpos, d.qvel
    contact[length] = contact_flags(model, d, feet, floor)
    fell = length < L
    return dict(qpos=qpos, qvel=qvel, contact=contact,
                action=actions, target_q=actions.copy(), valid=valid,
                fell=fell, start_frame=start, steps=length)


def one_design(job):
    out_str, row, ids, motions, is_eval, variants, seed = job
    import mujoco
    from pd_replay import build_model
    from run_calm_dynamics import retarget
    out = Path(out_str)
    name = row["design"]
    folder = out / ("heldout" if is_eval else "rollouts") / name
    folder.mkdir(parents=True, exist_ok=True)
    if all((folder / f"variant_{v}.npz").is_file() and
           (folder / f"variant_{v}.json").is_file() for v in range(variants)):
        return dict(design=name, skipped=True)
    manifest = dict(topology="00000011", motions=motions)
    design = {**row, "common": [], "extra": ids}
    retarget(manifest, design, out)
    model = build_model("00000011", uniforms=row["E"], kp=30, torque_limit=33.5)
    model.opt.timestep = .005
    if model.nq != 19 or model.nv != 18 or model.nu < 12:
        raise ValueError(f"Unexpected A1 model dimensions: {model.nq} {model.nv} {model.nu}")
    refs = out / "references" / name / "00000011"
    source_hash = {m["id"]: m["sha256"] for m in motions}
    stats = {}
    for variant in range(variants):
        dest = folder / f"variant_{variant}.npz"
        summary = dest.with_suffix(".json")
        if dest.exists() and summary.exists():
            continue
        records = []
        names = []
        for motion_id in ids:
            path = refs / f"calm_{motion_id:05d}.npz"
            with np.load(path, allow_pickle=False) as ref:
                if not np.allclose(ref["uniforms"], row["E"], atol=1e-7, rtol=0):
                    raise ValueError("Morphology mismatch in reference")
                rec = rollout_one(model, ref, variant, seed + motion_id * 7 + variant * 1000003)
            records.append(rec)
            names.append(f"calm_{motion_id:05d}")
        n = len(records)
        values = {k: np.stack([r[k] for r in records]) for k in
                  ("qpos", "qvel", "contact", "action", "target_q", "valid")}
        values.update(
            E=np.asarray(row["E"], np.float32),
            motions=np.asarray(names),
            clip=np.broadcast_to(np.arange(n)[:, None], (n, HORIZON)).copy(),
            group_id=np.asarray([source_hash[i] for i in ids]),
            source_motion_id=np.asarray(ids, np.int32),
            start_frame=np.asarray([r["start_frame"] for r in records], np.int32),
            start_variant=np.int32(variant),
            fell=np.asarray([r["fell"] for r in records]),
            dt=np.float32(DT), design=name, mode="mujoco_cpu_pd_reference",
            topology="00000011", controller="reference_target_position_pd",
            kp=np.float32(30), torque_limit=np.float32(33.5),
        )
        save_rollout(dest, values)
        stats = dict(design=name, variant=variant, episodes=n,
                     tt_usable_episodes=int(sum(r["steps"] >= 2 for r in records)),
                     valid_transitions=int(values["valid"].sum()),
                     falls=int(values["fell"].sum()),
                     backend="mujoco_cpu", file=str(dest))
        save(summary, stats)
    return dict(design=name, skipped=False, **{k:v for k,v in stats.items() if k!="design"})


def collect(out, *, workers, limit_designs=0, include_heldout=False, variants=2):
    plan = read(out / "plan.json")
    if not plan:
        raise FileNotFoundError("Run init first")
    designs = list(plan["designs"])
    if limit_designs:
        designs = designs[:limit_designs]
    if include_heldout:
        designs += plan["eval_designs"]
    names = {r["design"] for r in plan["eval_designs"]}
    jobs = [(str(out), row, list(plan["assignments"][row["design"]]), plan["motions"],
             row["design"] in names, variants, plan["config"]["seed"])
            for row in designs]
    print(f"MuJoCo CPU: {len(jobs)} morphologies, {workers} workers", flush=True)
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one_design, job): job[1]["design"] for job in jobs}
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            name = futures[f]
            try:
                print(f"[{i}/{len(jobs)}] {f.result()}", flush=True)
            except Exception as exc:
                failures.append(dict(design=name, error=str(exc)))
                print(f"[{i}/{len(jobs)}] FAILED {name}: {exc}", flush=True)
    report = audit(out)
    report["failures"] = failures
    save(out / "audit.json", report)
    if failures:
        raise RuntimeError(f"{len(failures)} designs failed; safely rerun collect")


def audit(out):
    plan = read(out / "plan.json")
    if not plan:
        raise FileNotFoundError("plan.json")
    source = set()
    episodes = steps = falls = usable = 0
    designs = set()
    invalid = []
    for path in sorted((out / "rollouts").glob("*/variant_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as d:
                m = np.asarray(d["valid"], bool)
                b, t = m.shape
                if t != HORIZON or d["qpos"].shape != (b, t+1, 19) or d["qvel"].shape != (b, t+1, 18):
                    raise ValueError("Wrong state/transition shapes")
                if d["target_q"].shape != (b, t, 12) or d["contact"].shape != (b, t+1, 4):
                    raise ValueError("Wrong actions or contact shape")
                if (m[:, 1:] & ~m[:, :-1]).any() or not np.isclose(d["dt"], DT):
                    raise ValueError("Non-prefix valid mask or wrong time step")
                if len(d["group_id"]) != b or len(d["source_motion_id"]) != b:
                    raise ValueError("Missing source grouping")
                if not np.isfinite(d["qpos"]).all() or not np.isfinite(d["target_q"]).all():
                    raise ValueError("Nonfinite physics")
                episodes += b
                usable += int((m.sum(1) >= 2).sum())
                steps += int(m.sum())
                falls += int(d["fell"].sum())
                source.update(int(i) for i, n in zip(d["source_motion_id"], m.sum(1)) if n >= 2)
                designs.add(str(d["design"]))
        except Exception as exc:
            invalid.append(dict(path=str(path), error=str(exc)))
    result = dict(backend="MuJoCo CPU", original_motion_target=10000,
                  source_coverage=len(source), missing_source_ids=sorted(
                      {m["id"] for m in plan["motions"]} - source),
                  training_designs=len(designs), target_designs=len(plan["designs"]),
                  episodes=episodes, target_episodes=plan["target_episodes"],
                  tt_usable_episodes=usable, valid_transitions=steps,
                  falls=falls, invalid_files=invalid,
                  complete=(not invalid and len(source)==10000 and usable >= 10000 and
                            len(designs)==len(plan["designs"]) and
                            episodes==plan["target_episodes"]))
    return result


def train(out, *, steps, batch, python):
    report = audit(out)
    save(out / "audit.json", report)
    if not report["complete"]:
        raise RuntimeError("Dynamics incomplete; inspect audit.json before TT training")
    data = sorted((out / "rollouts").glob("*/variant_*.npz"))
    dest = out / "tt" / "h32_fast"
    if (dest / "checkpoint.pt").exists():
        raise FileExistsError(f"Preserve existing checkpoint: {dest}")
    cmd = [python, "-u", str(ROOT / "code/tt_dynamics/train_tt.py"), "--data",
           *map(str, data), "--out", str(dest), "--horizon", "32", "--steps", str(steps),
           "--batch", str(batch), "--lr", "0.0002", "--d-model", "256",
           "--layers", "6", "--diff-steps", "5", "--norm", "minmax",
           "--clip", "1", "--sampler", "ddim", "--val-frac", ".2",
           "--seed", "0", "--modes", "1", "2", "4", "--mode-weights", ".2", ".4", ".4",
           "--film", "--eval-episodes", "8"]
    save(out / "tt_train_command.json", dict(command=cmd, backend="PyTorch CUDA",
                                             source_audit=report, train_starts=time.time()))
    with (out / "tt_train.log").open("a") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (dest / "checkpoint.pt").is_file():
        raise RuntimeError("TT exited without checkpoint.pt")
    print(f"TT checkpoint saved: {dest / 'checkpoint.pt'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init", "run", "audit", "status", "train"))
    p.add_argument("--out", type=Path, default=ROOT / "runs/v100_mujoco_10k")
    p.add_argument("--source", type=Path, default=ROOT / "runs/calm_motion_10,000/dataset")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-designs", type=int, default=0, help="Pilot only; 0 means all 200 training designs")
    p.add_argument("--include-heldout", action="store_true")
    p.add_argument("--variants", type=int, default=2, choices=(1, 2))
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    out = a.out.resolve()
    if a.command == "init":
        print(json.dumps(make_plan(a.source, out), ensure_ascii=False, default=str)[:300], flush=True)
    elif a.command == "run":
        if a.workers < 1:
            p.error("--workers must be positive")
        collect(out, workers=a.workers, limit_designs=a.limit_designs,
                include_heldout=a.include_heldout, variants=a.variants)
    elif a.command == "audit":
        result = audit(out)
        save(out / "audit.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    elif a.command == "status":
        for name in ("status.json", "audit.json"):
            print(name, json.dumps(read(out / name, {}), ensure_ascii=False)[:2000])
        print("TT checkpoint:", (out / "tt/h32_fast/checkpoint.pt").exists())
    elif a.command == "train":
        train(out, steps=a.steps, batch=a.batch, python=a.python)


if __name__ == "__main__":
    main()
