"""S0 feature-to-qpos roundtrip tests (no GPU/MJX rollout required)."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from rollout_data import features
from s0_geometry import s0_to_q


def test_realistic_tilt_yaw_feature_roundtrip():
    roll, pitch, yaw = .13, -.19, .6
    cr, sr, cp, sp = [f(v / 2) for v, f in
                        [(roll, np.cos), (roll, np.sin),
                         (pitch, np.cos), (pitch, np.sin)]]
    # q_y(pitch) q_x(roll) then a world yaw rotation.
    tilted = np.array([cp * cr, cp * sr, sp * cr, -sp * sr])
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    # quaternion qz(yaw)*tilted, with vector components x,y,z
    w, x, y, z = tilted
    quat = np.array([cy * w - sy * z, cy * x - sy * y,
                     cy * y + sy * x, cy * z + sy * w])
    q = np.zeros((1, 19), dtype=np.float64)
    q[0, 2] = .28
    q[0, 3:7] = quat
    q[0, 7:] = np.linspace(-.35, .55, 12)
    v = np.zeros((1, 18), dtype=np.float64)
    v[0, :3] = [.4, -.1, .08]
    v[0, 3:] = np.linspace(-.2, .3, 15)
    contact = np.array([[True, False, True, False]])
    s = features(q, v, contact)[0]
    q2, v2 = s0_to_q(s)
    recovered = features(q2[None], v2[None], contact)[0]
    np.testing.assert_allclose(recovered[2:38], s[2:38], atol=1e-5, rtol=0)
    assert abs(q2[2] - .28) < 1e-6
    assert q2.shape == (19,) and v2.shape == (18,)


@pytest.mark.parametrize("invalid", [
    np.zeros(42),
    np.full(42, np.nan),
    np.zeros(41),
])
def test_reject_invalid_initial_state(invalid):
    with pytest.raises(ValueError):
        s0_to_q(invalid)
