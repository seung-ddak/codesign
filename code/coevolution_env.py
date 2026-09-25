"""Reference-free MJX walking: WalkEnv.step defines the shared design/control reward."""

import copy

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from dm_control import mjcf
from mujoco import mjx

from t2.env.mj_utils import set_up_default_scene
from t2.mjx_env.base_env import MjxEnv, State, make_data, step
from t2.robogen.umi_on_legs_plus_plus import umi_on_legs_plus_plus_from_params

MAX_ACTUATORS = 19


def walking_outcome(previous, x, start_x, upright, bad_contact, goal, threshold):
    fell = bad_contact | (upright < threshold) | ~jp.isfinite(x + upright)
    progress = jp.where(fell, previous, jp.clip(x - start_x, -goal, goal))
    success = (progress >= goal) & ~fell
    return (progress - previous) / goal, progress, fell, success


class WalkEnv(MjxEnv):
    action_size = MAX_ACTUATORS

    def __init__(self, choices, seed, goal=1.0, threshold=0.1, action_scale=0.25,
                 reset_noise=0.02):
        super().__init__(ctrl_dt=0.02, sim_dt=0.005)
        if len(choices) != 8 or any(c not in (0, 1) for c in choices):
            raise ValueError("Expected an eight-bit TT quadruped choice")
        if not 0 < goal or not 0 < threshold < 1 or action_scale <= 0:
            raise ValueError("Invalid goal, upright threshold or action scale")
        self.goal, self.threshold = goal, threshold
        self.action_scale, self.reset_noise = action_scale, reset_noise
        uniforms = np.random.RandomState(seed).uniform(size=13)
        robot = umi_on_legs_plus_plus_from_params(
            choices=copy.copy(choices), uniforms=uniforms.tolist())
        self.mj_model = mjcf.Physics.from_mjcf_model(
            set_up_default_scene(robot, add_plane=True, plane_z_pos=0.0)).model.ptr
        m = self.mj_model
        m.opt.timestep, m.opt.iterations, m.opt.ls_iterations = self.sim_dt, 5, 7
        geom_names = [m.geom(i).name for i in range(m.ngeom)]
        bad = np.array(["calf_motor" in n or "gripper_collider" in n for n in geom_names])
        plane = m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        if not bad.any() or not plane.any():
            raise ValueError("Design lacks the specified termination geoms or floor")
        # Match TT contact geometry while retaining the D22 termination links.
        enabled = (m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) | plane | bad
        m.geom_contype[:] = enabled
        m.geom_conaffinity[:] = enabled
        self.bad_geoms, self.floor_geoms = jp.array(bad), jp.array(plane)
        free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Expected one floating base")
        self.base = int(m.jnt_bodyid[free[0]])
        self.qadr, self.vadr = int(m.jnt_qposadr[free[0]]), int(m.jnt_dofadr[free[0]])
        self.leg = np.array([i for i in range(m.nu)
                             if "/arx5/" not in m.actuator(i).name
                             and "/rail/" not in m.actuator(i).name])
        if not 0 < len(self.leg) <= MAX_ACTUATORS:
            raise ValueError("Actuator count exceeds the shared policy interface")
        joints = m.actuator_trnid[self.leg, 0]
        if np.any(joints < 0):
            raise ValueError("Expected joint-transmission leg actuators")
        self.qids, self.vids = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
        self.home = jp.array(m.key("home").qpos)
        self.ctrl = jp.array(m.key("home").ctrl)
        self.morphology = jp.array(np.r_[choices, uniforms], dtype=jp.float32)
        self.mask = jp.arange(MAX_ACTUATORS) < len(self.leg)
        self.mjx_model = mjx.put_model(m, impl="jax")

    def pad(self, x):
        return jp.pad(x, (0, MAX_ACTUATORS - x.size))

    def observation(self, data, last_action, start_x):
        rotation = data.xmat[self.base]
        position = data.qpos[self.qadr:self.qadr + 3]
        goal_vector = rotation.T @ jp.array([start_x + self.goal - position[0],
                                             -position[1], 0.0])
        return jp.concatenate([
            rotation.T @ jp.array([0., 0., -1.]),
            rotation.T @ data.qvel[self.vadr:self.vadr + 3],
            rotation.T @ data.qvel[self.vadr + 3:self.vadr + 6],
            self.pad(data.qpos[self.qids] - self.home[self.qids]),
            self.pad(data.qvel[self.vids]), last_action * self.mask, self.mask,
            goal_vector[:2], position[2:3], self.morphology,
        ]).astype(jp.float32)

    def reset(self, rng):
        qpos = self.home.at[self.qids].add(
            jax.random.uniform(rng, (len(self.leg),), minval=-self.reset_noise,
                               maxval=self.reset_noise))
        data = make_data(self.mj_model, qpos=qpos, ctrl=self.ctrl, impl="jax")
        data = mjx.forward(self.mjx_model, data)
        start = data.qpos[self.qadr]
        zero = jp.zeros(())
        return State(data=data, obs=self.observation(data, jp.zeros(self.action_size), start),
                     reward=zero, done=zero,
                     metrics={"progress": zero, "fell": zero, "success": zero},
                     info={"start_x": start, "progress": zero,
                           "pos_err": jp.array(self.goal), "orn_err": zero})

    def step(self, state, action):
        action = jp.clip(action, -1., 1.) * self.mask
        targets = self.ctrl.at[self.leg].add(action[:len(self.leg)] * self.action_scale)
        data = step(self.mjx_model, state.data, targets, self.n_substeps)
        contact = data.contact
        g0, g1 = contact.geom[:, 0], contact.geom[:, 1]
        bad = ((self.bad_geoms[g0] & self.floor_geoms[g1]) |
               (self.bad_geoms[g1] & self.floor_geoms[g0]))
        hit = jp.any(bad & (contact.dist < 0) & (g0 >= 0) & (g1 >= 0))
        reward, progress, fell, success = walking_outcome(
            state.info["progress"], data.qpos[self.qadr], state.info["start_x"],
            data.xmat[self.base, 2, 2], hit, self.goal, self.threshold)
        info = {**state.info, "progress": progress, "pos_err": self.goal - progress,
                "orn_err": 1. - data.xmat[self.base, 2, 2]}
        return State(data=data, obs=self.observation(data, action, info["start_x"]),
                     reward=reward, done=(fell | success).astype(jp.float32), info=info,
                     metrics={"progress": reward * self.goal,
                              "fell": fell.astype(jp.float32),
                              "success": success.astype(jp.float32)})


def evaluate(env, policy, seeds, horizon):
    """Score the current shared controller; never add rewards after termination."""
    def rollout(seed):
        key = jax.random.PRNGKey(seed)
        state = env.reset(key)

        def advance(carry, _):
            state, key, alive, total, fell, success, steps = carry
            key, action_key = jax.random.split(key)
            action, _ = policy(state.obs, action_key)
            next_state = jax.lax.cond(alive, lambda s: env.step(s, action), lambda s: s, state)
            total += jp.where(alive, next_state.reward, 0.)
            fell |= alive & (next_state.metrics["fell"] > 0)
            success |= alive & (next_state.metrics["success"] > 0)
            steps += alive.astype(jp.int32)
            alive &= next_state.done == 0
            return (next_state, key, alive, total, fell, success, steps), None

        result, _ = jax.lax.scan(advance, (state, key, jp.array(True), jp.zeros(()),
                                          jp.array(False), jp.array(False), jp.int32(0)),
                                 None, length=horizon)
        return jp.array([result[3], result[4], result[5], result[6]])

    return np.asarray(jax.jit(jax.vmap(rollout))(jp.array(seeds)))
