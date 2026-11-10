"""s0 보정: 한 형상의 시작 상태를 다른 형상에 옮길 때 몸통 z 만 옮긴다.

관절각·자세(quat)·x/y·qvel 은 그대로 두고, 가장 낮은 발의 바닥 높이(clearance)를
원래 형상과 같게 맞춘다. 원래 형상에 적용하면 shift = 0 (항등).
pd_replay.init_state 의 "가장 낮은 발" 방식과 같은 계산이다.
"""

from __future__ import annotations

import mujoco
import numpy as np


def foot_clearance(m: mujoco.MjModel, qpos: np.ndarray, foot_ids) -> float:
    """가장 낮은 발 sphere 바닥의 z (음수 = 관통)."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    return float(min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in foot_ids))


def align_height(m: mujoco.MjModel, qpos: np.ndarray, foot_ids, clearance: float) -> np.ndarray:
    """qpos 의 몸통 z 만 옮겨 가장 낮은 발의 clearance 를 주어진 값으로 맞춘다."""
    out = np.array(qpos, np.float64, copy=True)
    out[2] += clearance - foot_clearance(m, out, foot_ids)
    return out
