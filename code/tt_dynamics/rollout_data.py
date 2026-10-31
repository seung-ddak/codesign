"""rollout npz -> TT 토큰용 특징. train_tt.py 가 쓴다."""

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


def source_group_keys(path, archive, *, length):
    """Split physically linked episodes together across morphology files.

    Explicit intervention `group_id` wins. Otherwise `source_episode`
    (or row index) is qualified by NPZ stem, NOT morphology directory.
    This prevents e000/e001/... replicas of the same seeded source
    episode from landing in train and val separately.
    """
    stem = Path(path).stem
    if "group_id" in archive:
        return [f"group:{int(x)}" for x in archive["group_id"][:length]]
    if "source_episode" in archive:
        return [f"source:{stem}:{int(x)}" for x in archive["source_episode"][:length]]
    return [f"source:{stem}:{i}" for i in range(length)]


def load(paths, H):
    """Load each compressed NPZ array only once per file, not per episode."""
    X, E, M, src, CL = [], [], [], [], []
    for p in paths:
        with np.load(p) as d:
            valid = d["valid"]
            qpos, qvel = d["qpos"], d["qvel"]
            contact, target_q, physical_e = d["contact"], d["target_q"], d["E"]
            motions = d["motions"] if "motions" in d else None
            clips = d["clip"] if motions is not None else None
            variants = d["variant"] if motions is None else None
            for b in range(len(valid)):
                L = min(int(valid[b].sum()), H)
                if L < 2:
                    continue
                x = np.zeros((H, X_DIM), np.float32)
                x[:L, :S_DIM] = features(qpos[b, :L], qvel[b, :L], contact[b, :L])
                x[:L, S_DIM:] = target_q[b, :L]
                m = np.zeros(H, bool)
                m[:L] = True
                X.append(x)
                M.append(m)
                e = physical_e if physical_e.ndim == 1 else physical_e[b]
                E.append(np.asarray(e, dtype=np.float32))
                src.append(f"{Path(p).parent.name}/{Path(p).stem}")
                cl = np.full(H, "", dtype=object)
                if motions is not None:
                    cl[:L] = motions[clips[b, :L]]
                else:
                    cl[:L] = variants[b]
                CL.append(cl)
    return np.stack(X), np.stack(E), np.stack(M), np.array(src), np.stack(CL)
