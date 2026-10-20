"""rollout npz -> TT 토큰용 특징. train_tt.py 와 diag_mlp.py 가 같이 쓴다."""

from pathlib import Path

import numpy as np

S_DIM, A_DIM, E_DIM = 42, 12, 13
X_DIM = S_DIM + A_DIM


def yaw_of(quat):
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def features(qpos, qvel, contact):
    """한 episode 의 MJX state [T,nq]·[T,nv]·[T,4] -> S [T,42]."""
    quat = qpos[:, 3:7]
    w, x, y, z = quat.T
    grav = -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)  # R^T (0,0,-1)
    yaw = yaw_of(quat)
    c0, s0 = np.cos(yaw[0]), np.sin(yaw[0])
    dxy = qpos[:, :2] - qpos[0, :2]
    pos = np.stack([c0 * dxy[:, 0] + s0 * dxy[:, 1], -s0 * dxy[:, 0] + c0 * dxy[:, 1]], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    v = qvel[:, :3]
    vh = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
    dyaw = yaw - yaw[0]
    return np.concatenate([pos, qpos[:, 2:3], np.cos(dyaw)[:, None], np.sin(dyaw)[:, None], grav, vh, qvel[:, 3:6],
                           qpos[:, 7:19], qvel[:, 6:18], contact.astype(np.float32)], 1).astype(np.float32)


def load(paths, H):
    X, E, M, src, CL = [], [], [], [], []
    for p in paths:
        d = np.load(p)
        for b in range(len(d["valid"])):
            L = min(int(d["valid"][b].sum()), H)
            if L < 2:
                continue
            x = np.zeros((H, X_DIM), np.float32)
            x[:L, :S_DIM] = features(d["qpos"][b, :L], d["qvel"][b, :L], d["contact"][b, :L])
            x[:L, S_DIM:] = d["target_q"][b, :L]
            m = np.zeros(H, bool)
            m[:L] = True
            X.append(x), M.append(m), E.append(d["E"].astype(np.float32)), src.append(f"{Path(p).parent.name}/{Path(p).stem}")
            cl = np.full(H, "", dtype=object)  # step 별 모션 이름 (진단용)
            cl[:L] = d["motions"][d["clip"][b, :L]] if "motions" in d else d["variant"][b]
            CL.append(cl)
    return np.stack(X), np.stack(E), np.stack(M), np.array(src), np.stack(CL)
