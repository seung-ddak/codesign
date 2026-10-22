"""Exact S0-token to MuJoCo state conversion shared by generation and physics audits.

The TT state is yaw-zero and XY-zero at the first window frame.
Keep this conversion in one place so diagnosis audits the actual execution code.
"""
import numpy as np

from rollout_data import S_DIM


def s0_to_q(s):
    """S[42] -> MuJoCo qpos[19], qvel[18] at canonical XY=0, yaw=0."""
    s = np.asarray(s, dtype=np.float64)
    if s.shape != (S_DIM,) or not np.isfinite(s).all():
        raise ValueError(f"S0 must be finite shape ({S_DIM},), got {s.shape}")
    gn = float(np.linalg.norm(s[5:8]))
    if not np.isfinite(gn) or gn < 1e-8:
        raise ValueError(f"S0 has invalid body-frame gravity norm: {gn}")
    g = s[5:8] / gn
    th, ph = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
    ch, sh, cp, sp = np.cos(th / 2), np.sin(th / 2), np.cos(ph / 2), np.sin(ph / 2)
    q = np.concatenate([[0.0, 0.0, s[2]],
                        [ch * cp, ch * sp, sh * cp, -sh * sp],
                        s[14:26]])
    v = np.concatenate([s[8:14], s[26:38]])
    return q, v
