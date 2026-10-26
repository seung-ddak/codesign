import sys
from pathlib import Path
import numpy as np
import torch
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code/tt_dynamics'))
from obstacle_task import ObstacleReward, add_pole, envelope_radius, pole_contact


def test_obstacle_penalty_and_goal_progress():
    reward = ObstacleReward()
    reward.robot_radius = [.3, .3]
    x = torch.zeros(2, 3, 54)
    x[:, :, 2], x[:, :, 7] = .3, -1
    x[:, :, 0] = torch.tensor([0., 1., 2.])
    x[1, :, 1] = .7
    x.requires_grad_()
    mask = torch.ones(2, 3, dtype=torch.bool)
    score, terms = reward(x, mask)
    assert terms['p_obstacle'][0] > 0 and terms['p_obstacle'][1] == 0
    assert terms['r_progress'][0] == 2
    score.sum().backward()
    assert torch.isfinite(x.grad).all()
    mask[:, 1:] = False
    assert torch.all(reward.p_obstacle(x, mask) == 0)


def test_pole_geometry_and_envelope_cover_hinge_motion():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="0 0 .3"><freejoint/>
      <geom type="sphere" size=".1"/><body pos=".2 0 0"><joint axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".03"/></body></body></worldbody></mujoco>''')
    radius = envelope_radius(model)
    for angle in np.linspace(-3, 3, 11):
        d = mujoco.MjData(model); d.qpos[7] = angle; mujoco.mj_forward(model, d)
        distances = np.linalg.norm(d.geom_xpos - d.qpos[:3], axis=1) + model.geom_rbound
        assert np.all(distances <= radius + 1e-8)
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    add_pole(spec, ObstacleReward())
    pole_model = spec.compile()
    assert pole_model.ngeom == 11
    assert pole_model.nq == 0


def test_real_pole_contact_excludes_floor_and_padding():
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
        <body pos="1 0 .4"><freejoint/><geom name="robot" type="sphere" size=".1"/></body>
        </worldbody></mujoco>''')
    add_pole(spec, ObstacleReward())
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pairs = np.array([[c.geom1, c.geom2] for c in data.contact[:data.ncon]])
    distances = np.array([c.dist for c in data.contact[:data.ncon]])
    assert pole_contact(model, pairs, distances)
    pole = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_pole_0")
    assert not pole_contact(model, np.array([[0, pole], [-1, -1]]), np.array([-.01, -1.]))
    assert not pole_contact(model, pairs, np.ones(len(pairs)))


def test_five_poles_are_all_scored_and_built():
    reward = ObstacleReward(count=5)
    assert reward.poles == [(float(i), 0.) for i in range(1, 6)]
    assert reward.goal == (6., 0.)
    reward.robot_radius = [.3]
    x = torch.zeros(1, 5, 54)
    x[0, :, 0] = torch.arange(1., 6.)
    mask = torch.ones(1, 5, dtype=torch.bool)
    for i in range(5):
        single = torch.zeros_like(mask); single[0, i] = True
        assert reward.p_obstacle(x, single).item() > 0
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody/></mujoco>')
    add_pole(spec, reward)
    model = spec.compile()
    assert model.ngeom == 50
    assert np.allclose(model.geom_pos[:, 0], np.repeat(np.arange(1., 6.), 10))
