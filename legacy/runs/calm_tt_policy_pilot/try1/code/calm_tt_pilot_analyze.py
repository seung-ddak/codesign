"""Compare CALM-on-TT pilot episodes with PD-reference rollouts of the same shapes.

Reference motion = the original A1 CALM rollout (world frame, same start yaw).
Per 1 s window: displacement direction cosine vs reference (windows where the
reference moves >= 5 cm), plus the mean velocity error in the reference's
heading frame. Only the part before a fall is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calm_tt_pilot import MOTIONS, PD_RUN


def windows(pos, dt, ref, ref_dt=1 / 30, seconds=10):
    """Return (cos list, (forward, lateral) velocity errors) over 1 s windows."""
    cos, err = [], []
    n_alive = (len(pos) - 1) * dt
    for k in range(int(min(seconds, n_alive))):
        a, b = round(k / dt), round((k + 1) / dt)
        ra, rb = round(k / ref_dt), round((k + 1) / ref_dt)
        d = pos[b, :2] - pos[a, :2]
        r = ref[rb, :2] - ref[ra, :2]
        # heading frame of the reference at the window start
        x, y, z, w = ref[ra, 3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, s], [-s, c]])
        err.append(rot @ (d - r))
        if np.linalg.norm(r) >= .05:
            cos.append(float(d @ r / (np.linalg.norm(d) * np.linalg.norm(r) + 1e-9)))
    return cos, err


def summarize(rows):
    cos = [c for r in rows for c in r["cos"]]
    err = np.array([e for r in rows for e in r["err"]]) if any(r["err"] for r in rows) else np.zeros((0, 2))
    return dict(episodes=len(rows), survived_10s=int(sum(not r["fell"] for r in rows)),
                survival_rate=float(np.mean([not r["fell"] for r in rows])),
                direction_cos_mean=float(np.mean(cos)) if cos else None,
                direction_agree_frac=float(np.mean(np.array(cos) > .5)) if cos else None,
                forward_bias_mps=float(err[:, 0].mean()) if len(err) else None,
                lateral_bias_mps=float(err[:, 1].mean()) if len(err) else None,
                velocity_rmse_mps=float(np.sqrt((err ** 2).sum(1).mean())) if len(err) else None)


def ref_root(motion):
    with np.load(MOTIONS / f"{motion:05d}.npz") as z:
        return z["root"].astype(np.float64)


def calm_rows(run):
    rows = []
    for f in sorted((run / "episodes").glob("*.npz")):
        z = np.load(f)
        m, shape = int(z["motion"]), str(z["shape"])
        cos, err = windows(z["qpos"].astype(np.float64), float(z["dt"]), ref_root(m))
        q = z["qpos"]
        rows.append(dict(shape=shape, motion=m, fell=bool(z["fell"]), cos=cos, err=err,
                         disp=(q[-1, :2] - q[0, :2]).tolist(), seconds=float(z["steps"]) * float(z["dt"])))
    return rows


def pd_rows(shapes, only=None):
    rows = []
    for s in shapes:
        z = np.load(PD_RUN / "rollouts" / s / "variant_0.npz")
        for i, name in enumerate(z["motions"]):
            m = int(str(name).split("_")[1])
            if only is not None and (s, m) not in only:
                continue
            with np.load(MOTIONS / f"{m:05d}.npz") as mm:
                if len(mm["q"]) != 301:      # same 10 s motions as the pilot
                    continue
            L = int(z["valid"][i].sum())
            q = z["qpos"][i, :L + 1].astype(np.float64)
            cos, err = windows(q, float(z["dt"]), ref_root(m))
            rows.append(dict(shape=s, motion=m, fell=bool(z["fell"][i]), cos=cos, err=err,
                             disp=(q[-1, :2] - q[0, :2]).tolist(), seconds=L * float(z["dt"])))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    cfg = json.load(open(a.run / "config.json"))
    shapes = list(cfg["shapes"])
    pairs = {(s, m) for s, m in cfg["pd_pairs"]}
    calm = calm_rows(a.run)
    calm_paired = [r for r in calm if (r["shape"], r["motion"]) in pairs]
    pd_paired = pd_rows(shapes, only=pairs)
    pd_all = pd_rows(shapes)
    by_shape = {s: summarize([r for r in calm if r["shape"] == s]) for s in shapes}
    # Shape effect: spread of 10 s displacement across shapes for the same motion (survivors).
    spread = []
    for m in cfg["motions"]:
        d = np.array([r["disp"] for r in calm if r["motion"] == m and not r["fell"]])
        if len(d) >= 4:
            spread.append(float(np.linalg.norm(d - d.mean(0), axis=1).mean()))
    out = dict(calm_all=summarize(calm), calm_paired=summarize(calm_paired),
               pd_paired=summarize(pd_paired), pd_all_same_shapes=summarize(pd_all),
               calm_by_shape=by_shape,
               calm_shape_spread_m=dict(motions=len(spread), mean=float(np.mean(spread)) if spread else None),
               calm_mean_disp_m=np.mean([r["disp"] for r in calm], 0).tolist(),
               reference_mean_disp_m=np.mean([(ref_root(m)[-1, :2] - ref_root(m)[0, :2]).tolist()
                                              for m in cfg["motions"]], 0).tolist())
    json.dump(out, open(a.run / "analysis.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "calm_by_shape"}, indent=1))


if __name__ == "__main__":
    main()
