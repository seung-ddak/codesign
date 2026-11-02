"""measure_e_signal / regression_gate 공용: episode 별 몸통 결과량."""
import numpy as np


def yaw_of(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def outcomes(z, Tg):
    """episode 별 [vx, vy, yaw_rate, z_mean, tilt_mean] — group 별 지평 Tg[B] step (0..Tg) 평균 속도, 생존 여부."""
    B = len(Tg); i = np.arange(B)
    q = z["qpos"]
    ok = np.array([z["valid"][b, :Tg[b]].all() for b in range(B)]) & ~z["fell"]
    q0, qT = q[i, 0], q[i, Tg]
    yaw0, yawT = yaw_of(q0[:, 3:7]), yaw_of(qT[:, 3:7])
    dxy = qT[:, :2] - q0[:, :2]
    c, s = np.cos(yaw0), np.sin(yaw0)
    dx, dy = c * dxy[:, 0] + s * dxy[:, 1], -s * dxy[:, 0] + c * dxy[:, 1]
    dyaw = np.arctan2(np.sin(yawT - yaw0), np.cos(yawT - yaw0))
    sec = Tg * 0.02
    zm, tl = np.zeros(B), np.zeros(B)
    for b in range(B):
        qq = q[b, :Tg[b] + 1]
        up = 1 - 2 * (qq[:, 4] ** 2 + qq[:, 5] ** 2)
        zm[b], tl[b] = qq[:, 2].mean(), np.arccos(np.clip(up, -1, 1)).mean()
    return np.stack([dx / sec, dy / sec, dyaw / sec, zm, tl], 1), ok


NAMES = ["vx [m/s]", "vy [m/s]", "yaw_rate [rad/s]", "z_mean [m]", "tilt_mean [rad]"]
