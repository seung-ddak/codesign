"""Upright poles: conservative guidance envelope, actual MuJoCo contacts."""
import numpy as np
import torch
from task_reward import StraightReward


class ObstacleReward(StraightReward):
    def __init__(self, count=1):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("pole count must be a positive integer")
        super().__init__(half_width=1.5, penalties={"lane": 20., "fall": 20., "action_rate": .05, "obstacle": 100.})
        self.poles = [(float(i + 1), 0.0) for i in range(count)]
        self.pole = self.poles[0]
        self.radius = .06
        self.goal = (float(count + 1), 0.0)
        self.robot_radius = None

    def r_progress(self, x, m):
        distance = (x[..., :2] - x.new_tensor(self.goal)).norm(dim=-1)
        return ((distance[:, :-1] - distance[:, 1:]) * (m[:, :-1] & m[:, 1:])).sum(1)

    def p_obstacle(self, x, m):
        if self.robot_radius is None:
            raise RuntimeError("morphology envelope must be set before scoring")
        # shortcut: all-pose envelope is conservative; use differentiable FK for tight-clearance tasks.
        radius = torch.as_tensor(self.robot_radius, device=x.device, dtype=x.dtype).reshape(-1, 1)
        distance = (x[..., :2].unsqueeze(-2) - x.new_tensor(self.poles)).norm(dim=-1).amin(dim=-1)
        violation = torch.relu(radius + self.radius + .05 - distance).square()
        return (violation * m).sum(1) / m.sum(1).clamp_min(1)

    def describe(self):
        return dict(task="obstacle", pole=self.pole, poles=self.poles, pole_radius=self.radius, pole_height=.82,
                    goal=self.goal, goal_tolerance_m=.2, rewards=self.rewards, penalties=self.penalties,
                    envelope="all-pose morphology bounding sphere; radius gradient stopped", unit=self.unit)

    def guide_lines(self):
        theta = np.linspace(0, 2*np.pi, 80)
        circles = [(np.array(pole) + self.radius * np.stack([np.cos(theta), np.sin(theta)], 1), False)
                   for pole in self.poles]
        return circles + [(np.array([[0., 0.], self.goal]), True)]


def envelope_radius(model):
    """Bound every robot geom for any hinge angles, relative to the free root."""
    reach = np.zeros(model.nbody)
    for b in range(2, model.nbody):
        reach[b] = reach[model.body_parentid[b]] + np.linalg.norm(model.body_pos[b])
    robot = model.geom_bodyid > 0
    return float(np.max(reach[model.geom_bodyid[robot]] + np.linalg.norm(model.geom_pos[robot], axis=1)
                        + model.geom_rbound[robot]))


def add_pole(spec, reward):
    import mujoco
    # Overlapping spheres avoid unsupported MJX cylinder-cylinder pairs.
    for j, pole in enumerate(reward.poles):
        for i, z in enumerate(np.linspace(.04, .76, 10)):
            spec.worldbody.add_geom(name=f"obstacle_pole_{10*j+i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    pos=[*pole, z], size=[reward.radius, 0, 0], rgba=[.9, .2, .1, 1])


def pole_contact(model, geom, dist):
    poles = np.array([i for i in range(model.ngeom) if model.geom(i).name.startswith("obstacle_pole_")])
    hits = np.isin(geom, poles)
    robot = model.geom_bodyid[np.maximum(geom[:, ::-1], 0)] > 0
    return bool(np.any(np.any(hits & robot, axis=1) & (dist <= 0)))
