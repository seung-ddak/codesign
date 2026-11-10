"""Per-1 s tracking error of PD rollouts vs the original A1 CALM motion.

For window k (50 PD steps at 50 Hz, 30 reference frames at 30 Hz from start_frame):
  yaw_err = |dyaw_pd - dyaw_ref|
  pos_err = |d_pd - d_ref|, each displacement in its own heading frame at the window start.
"""
import numpy as np

MOTIONS = "/home/cgr-ugrad-2026/quad_ws/codesign/runs/calm_motion_10,000/dataset/accepted/"
_cache = {}


def yaw_of(w, x, y, z):
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def ref_root(m):
    if m not in _cache:
        if len(_cache) > 4000:
            _cache.clear()
        _cache[m] = np.load(MOTIONS + f"{m:05d}.npz")["root"].astype(np.float64)
    return _cache[m]


def window_errors(qpos, L, motion, start, dt=0.02):
    """Return arrays (yaw_err, pos_err) for each full 1 s window inside the valid length L."""
    R = ref_root(motion)
    q = qpos[:L + 1].astype(np.float64)
    yp = np.unwrap(yaw_of(*q[:, 3:7].T))
    yr = np.unwrap(yaw_of(R[:, 6], R[:, 3], R[:, 4], R[:, 5]))
    per = int(round(1 / dt))
    ye, pe = [], []
    for k in range(L // per):
        a, b, ra, rb = k * per, (k + 1) * per, start + k * 30, start + (k + 1) * 30
        if rb >= len(R):
            break
        def body(P, Y, A, B):
            d = P[B, :2] - P[A, :2]; c, s = np.cos(Y[A]), np.sin(Y[A])
            return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        ye.append(abs((yp[b] - yp[a]) - (yr[rb] - yr[ra])))
        pe.append(float(np.linalg.norm(body(q, yp, a, b) - body(R, yr, ra, rb))))
    return np.asarray(ye), np.asarray(pe)
