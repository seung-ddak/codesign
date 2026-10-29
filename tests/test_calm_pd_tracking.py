import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'code/motion_tracking'))
from eval_calm_pd_tracking import foot_contacts, tracking_errors


def test_errors_measure_world_translation_and_quaternion_sign():
    ref = np.zeros((2, 19)); ref[:, 3] = 1
    sim = ref.copy(); sim[:, 0] = .1; sim[:, 3] = -1
    feet = np.zeros((2, 4, 3)); moved = feet.copy(); moved[:, :, 0] = .2
    errors = tracking_errors(sim, ref, moved, feet)
    assert errors['joint_rmse_deg'] == 0
    assert errors['orientation_mean_deg'] == 0
    assert np.isclose(errors['body_position_rmse_m'], .1)
    assert np.isclose(errors['foot_world_mean_error_m'], .2)


def test_slip_only_counts_loaded_floor_contacts():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body pos="0 0 .09"><freejoint/><geom name="foot" type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model); data.qvel[0] = .5
    mujoco.mj_forward(model, data)
    feet = [model.geom('foot').id] * 4; planes = {model.geom('floor').id}
    contact, slip = foot_contacts(model, data, feet, planes)
    assert contact.all() and np.allclose(slip, .5)
    data.qpos[2] = 1; mujoco.mj_forward(model, data)
    contact, slip = foot_contacts(model, data, feet, planes)
    assert not contact.any() and np.isnan(slip).all()
