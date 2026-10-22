"""Fail-fast origin audit for generated E/S0 ground penetration (CPU MuJoCo).

Stage 0: validate paths/config; stage 1: audit original MJX rollouts;
stage 2: real qpos -> TT feature S0 -> shared S0 decoder -> MJX;
stage 3: audit actual generated (E,S0) for scale 0 and 1.
At first failing stage, save diagnosis.json + diagnosis.md and stop.
No automatic ground-height adjustment, training, guidance, or MJX stepping.
"""
import argparse
from collections import Counter
import json
import pickle
import sys
import traceback
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "motion_tracking"))
sys.path.insert(0, str(ROOT.parent / "transformer-transformer"))
from pd_replay import LEGS, build_model
from rollout_data import S_DIM, features
from s0_geometry import s0_to_q


class StageFailed(Exception):
    pass


def audit(model, q, v, tolerance=0.002, min_support=2):
    q, v = np.asarray(q), np.asarray(v)
    if q.shape != (model.nq,) or v.shape != (model.nv,) or not (
            np.isfinite(q).all() and np.isfinite(v).all()):
        return dict(valid=False, reasons=["nonfinite_or_invalid_qpos_qvel"])
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = q, v
    mujoco.mj_forward(model, d)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")
            for leg in LEGS]
    if floor < 0 or min(feet) < 0:
        raise ValueError("model lacks floor or named feet geometries")
    heights = [float(d.geom_xpos[f, 2] - model.geom_size[f, 0]) for f in feet]
    contacts = [any(floor in (c.geom1, c.geom2)
                    and f in (c.geom1, c.geom2) and c.dist <= 1e-3
                    for c in d.contact[:d.ncon]) for f in feet]
    nonfoot = max([max(0.0, -float(c.dist)) for c in d.contact[:d.ncon]
                   if floor in (c.geom1, c.geom2)
                   and (c.geom2 if c.geom1 == floor else c.geom1) not in feet] or [0.0])
    selfpen = max([max(0.0, -float(c.dist)) for c in d.contact[:d.ncon]
                   if floor not in (c.geom1, c.geom2)] or [0.0])
    joints_bad = any(bool(model.jnt_limited[j])
                     and not (model.jnt_range[j, 0] - 1e-3 <= q[int(model.jnt_qposadr[j])]
                              <= model.jnt_range[j, 1] + 1e-3)
                     for j in range(model.njnt))
    reasons = []
    if min(heights) < -tolerance:
        reasons.append("foot_ground_penetration")
    if nonfoot > tolerance:
        reasons.append("nonfoot_ground_penetration")
    if selfpen > tolerance:
        reasons.append("self_penetration")
    if sum(contacts) < min_support:
        reasons.append("too_few_floor_contacts")
    if abs(np.linalg.norm(q[3:7]) - 1) > 1e-3:
        reasons.append("invalid_body_quaternion")
    if 1 - 2 * (q[4] ** 2 + q[5] ** 2) <= 0.5:
        reasons.append("body_fallen")
    if joints_bad:
        reasons.append("joint_limit_violation")
    return dict(valid=not reasons, reasons=reasons,
                foot_clearance_mm=[round(h * 1000, 3) for h in heights],
                max_foot_penetration_mm=round(max(0.0, -min(heights)) * 1000, 3),
                nonfoot_penetration_mm=round(nonfoot * 1000, 3),
                self_penetration_mm=round(selfpen * 1000, 3),
                foot_contacts=contacts, floor_contact_count=sum(contacts),
                body_height_m=float(q[2]),
                body_speed_m_s=float(np.linalg.norm(v[:3])))


class Recorder:
    def __init__(self, output, config):
        self.output = output
        self.data = dict(status="RUNNING", failed_stage=None, stages=[],
                         config=config, next_action=None)
        self.save()

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "diagnosis.json").write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False))
        text = ["# Generated E/S0 penetration diagnosis", "",
                "Result: " + self.data["status"], ""]
        for s in self.data["stages"]:
            text += ["## " + s["name"] + " : " + s["status"], s["reason"], ""]
        if self.data["next_action"]:
            text += ["## Next action", self.data["next_action"]]
        (self.output / "diagnosis.md").write_text("\n".join(text))

    def stage(self, name, ok, reason, measurements, next_action=""):
        self.data["stages"].append(dict(name=name, status="PASS" if ok else "FAIL",
                                        reason=reason, measurements=measurements))
        if not ok:
            self.data["status"] = "FAIL"
            self.data["failed_stage"] = name
            self.data["next_action"] = next_action
        self.save()
        print(("PASS: " if ok else "FAIL: ") + name + " — " + reason, flush=True)
        if not ok:
            raise StageFailed(reason)


def expert_params(directory):
    candidates = sorted(directory.glob("wandb/offline-run-*/files/config.pkl"))
    if not candidates:
        raise ValueError("Expert config.pkl missing in " + str(directory))
    with candidates[-1].open("rb") as f:
        cfg = pickle.load(f)
    choice = cfg.get("choice")
    if choice is None:
        raise ValueError("Expert configuration has no choice")
    return choice, cfg.get("kp"), cfg.get("tau_scale"), cfg.get("torque_limit")


def run(args):
    recorder = Recorder(args.out, dict(
        real_sources=args.sources, data_root=str(args.data_root),
        smoke_dir=str(args.smoke_dir), expert_config=str(args.expert_config),
        tolerance_mm=args.tolerance_mm, conversion_tolerance_mm=args.conversion_tolerance_mm,
        probes_per_shape=args.probes_per_shape, seed=args.seed))
    active = "0-prerequisites"
    try:
        choice, kp, tau_scale, torque_limit = expert_params(args.expert_config)
        paths = [args.data_root / name / "primitive_seed0.npz" for name in args.sources]
        expected = paths + [args.smoke_dir / name
                            for name in ("summary.json", "scale_0.npz", "scale_1.npz")]
        missing = [str(p) for p in expected if not p.is_file()]
        recorder.stage(active, not missing, "Input artifacts and expert model located"
                       if not missing else "Required files missing",
                       dict(missing=missing, choice=str(choice), paths=[str(p) for p in expected]),
                       "Use the wide16-experts branch with original rollouts and smoke NPZ files.")

        # Stage 1: find physically valid real MJX frames; never apply height correction.
        active = "1-original-MJX-physics"
        real = []
        table = []
        for path in paths:
            with np.load(path) as z:
                e = np.asarray(z["E"], np.float64).reshape(-1)
                if e.shape != (13,) or not np.isfinite(e).all():
                    raise ValueError("Invalid 13D E in " + str(path))
                m = build_model(choice, kp=kp, tau_scale=tau_scale,
                                torque_limit=torque_limit, uniforms=e)
                poses, vels, contacts, masks = z["qpos"], z["qvel"], z["contact"], z["valid"]
                rng = np.random.default_rng(args.seed)
                accepted, attempted = 0, 0
                rejects = Counter()
                examples = []
                for b in rng.permutation(len(masks)):
                    if accepted >= args.probes_per_shape or attempted >= args.max_probes_per_shape:
                        break
                    length = int(np.asarray(masks[b]).sum())
                    if length < 2:
                        continue
                    for t in dict.fromkeys((0, length // 2, max(0, length - 2))):
                        if accepted >= args.probes_per_shape or attempted >= args.max_probes_per_shape:
                            break
                        attempted += 1
                        q, v, contact = (poses[b, t].copy(), vels[b, t].copy(),
                                         contacts[b, t].copy())
                        a = audit(m, q, v, args.tolerance_mm / 1000, args.min_support)
                        if a["valid"]:
                            accepted += 1
                            real.append((path.parent.name, int(b), int(t), m, q, v,
                                         contact, a))
                        else:
                            rejects.update(a["reasons"])
                            if len(examples) < 3:
                                examples.append(dict(episode=int(b), frame=int(t), audit=a))
                table.append(dict(source=path.parent.name, attempted=attempted,
                                  accepted=accepted, rejected_reasons=dict(rejects),
                                  rejected_examples=examples))
        underfilled = [x["source"] for x in table if x["accepted"] < args.probes_per_shape]
        recorder.stage(active, not underfilled,
                       "Original physical MJX states pass the current collision gate"
                       if not underfilled else "No adequate physically valid original frames: " + ", ".join(underfilled),
                       dict(sources=table, total_accepted=len(real)),
                       "Audit model choice, input rollouts, and 2mm contact convention FIRST. "
                       "Original physics failure does not establish a TT generation fault.")

        # Stage 2: compare the very same original physical frames before/after TT roundtrip.
        active = "2-real-S0-feature-roundtrip"
        roundtrips = []
        for source, b, t, model, q, v, contact, original in real:
            s = features(q[None], v[None], contact[None])[0]
            q2, v2 = s0_to_q(s)  # identical implementation to generate_task.py
            recon = audit(model, q2, v2, args.tolerance_mm / 1000, args.min_support)
            s_again = features(q2[None], v2[None], contact[None])[0]
            orig_h = np.asarray(original["foot_clearance_mm"])
            new_h = np.asarray(recon.get("foot_clearance_mm", [np.nan] * 4))
            h_error = float(np.max(np.abs(orig_h - new_h)))
            token_error = float(np.max(np.abs(s[2:38] - s_again[2:38])))
            ok = (recon["valid"] and np.isfinite(h_error)
                  and h_error <= args.conversion_tolerance_mm
                  and token_error <= args.token_tolerance)
            roundtrips.append(dict(source=source, episode=b, frame=t, pass_check=bool(ok),
                                   foot_height_error_mm=h_error, token_error=token_error,
                                   original=original, reconstructed=recon))
        bad = [x for x in roundtrips if not x["pass_check"]]
        recorder.stage(active, not bad,
                       "Real E/S0 preserved across TT features and execution decoder"
                       if not bad else str(len(bad)) + " original frames fail state conversion/geometry",
                       dict(probes=roundtrips, failed=len(bad),
                            conversion_tolerance_mm=args.conversion_tolerance_mm),
                       "Fix geometry/coordinate representation in rollout_data.features "
                       "or s0_geometry.s0_to_q before examining TT generation.")

        # Stage 3: ONLY after real physics and actual roundtrip passed.
        active = "3-generated-E-S0-physics"
        generated = []
        with (args.smoke_dir / "summary.json").open() as f:
            smoke_summary = json.load(f)
        for scale in (0, 1):
            tag = "scale_" + str(scale)
            with np.load(args.smoke_dir / (tag + ".npz")) as z:
                es, preds = np.asarray(z["E"]), np.asarray(z["pred"])
            if es.ndim != 2 or es.shape[1] != 13 or preds.ndim != 3 or preds.shape[2] != 54 or len(es) != len(preds):
                raise ValueError(tag + " NPZ shape mismatch")
            for i, (e, pred) in enumerate(zip(es, preds)):
                s = pred[0, :S_DIM]
                try:
                    model = build_model(choice, kp=kp, tau_scale=tau_scale,
                                        torque_limit=torque_limit, uniforms=e)
                    q2, v2 = s0_to_q(s)
                    pa = audit(model, q2, v2, args.tolerance_mm / 1000, args.min_support)
                except (ValueError, FloatingPointError) as err:
                    pa = dict(valid=False, reasons=["invalid_generated_morphology_or_S0"],
                              error=str(err))
                prior = smoke_summary.get("runs", {}).get(tag, [])
                generated.append(dict(scale=scale, candidate=i, E=e.tolist(),
                                      S0_height_m=float(s[2]), physics=pa,
                                      prior_smoke_physics_valid=(prior[i].get("physically_valid_s0")
                                                                 if i < len(prior) else None)))
        failures = [x for x in generated if not x["physics"]["valid"]]
        recorder.stage(active, not failures,
                       "Generated morphology/S0 pairs pass the same geometry audit"
                       if not failures else str(len(failures)) + "/" + str(len(generated))
                       + " generated pairs fail geometry; MJX execution is blocked",
                       dict(candidates=generated, failed=len(failures)),
                       "The original real states and encoder-decoder passed; the SAVED "
                       "generated E/S0 pairs are inconsistent. Next compare many seeds "
                       "and analyze E-dependent body height/joints. Do not shift height "
                       "or relax collision tolerance to force a pass.")

        recorder.data["status"] = "PASS"
        recorder.data["next_action"] = ("No static penetration found for these samples. "
                                        "Proceed to separate 2s MJX open-loop validation.")
        recorder.save()
        print("PASS: all diagnostic stages; static checks do not prove dynamic stability.")
        return 0

    except StageFailed:
        print("STOP: see " + str(args.out / "diagnosis.md"), file=sys.stderr)
        return 1
    except Exception as err:
        recorder.stage(active, False, type(err).__name__ + ": " + str(err),
                       dict(traceback=traceback.format_exc()),
                       "Resolve this stage error before continuing to later tests.")
        return 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/pipeline_v1/s0_root_cause_audit"))
    p.add_argument("--smoke-dir", type=Path,
                   default=Path("runs/pipeline_v1/fast_mode24_v4_smoke_audit/smoke_zigzag_joint"))
    p.add_argument("--data-root", type=Path,
                   default=Path("runs/pipeline_v1/wide16_action/rollouts"))
    p.add_argument("--expert-config", type=Path,
                   default=Path("runs/pipeline_v1/experts_wide16/walkForwardSlow__r00"))
    p.add_argument("--sources", nargs="+", default=["e000", "e001", "e002"])
    p.add_argument("--probes-per-shape", type=int, default=3)
    p.add_argument("--max-probes-per-shape", type=int, default=90)
    p.add_argument("--tolerance-mm", type=float, default=2.0)
    p.add_argument("--conversion-tolerance-mm", type=float, default=1.0)
    p.add_argument("--token-tolerance", type=float, default=1e-4)
    p.add_argument("--min-support", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if (args.probes_per_shape < 1 or args.max_probes_per_shape < args.probes_per_shape
            or args.tolerance_mm <= 0 or args.conversion_tolerance_mm <= 0
            or args.token_tolerance <= 0 or not 1 <= args.min_support <= 4):
        p.error("invalid thresholds or sample counts")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
