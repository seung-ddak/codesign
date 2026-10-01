"""Stage 2 flat locomotion on a TT morphology (T3 env, T4 reward).

Observation, command, reset, push and reward follow legged_gym `LeggedRobotCfg` (base, no
A1 override; upstream leggedrobotics/legged_gym `envs/base/legged_robot*.py`). The robot,
actuators and contact geometry stay TT `umi_on_legs_plus_plus` + MJX (D35, D37); only the
contact link names are mapped to TT bodies (D43). Time-outs come from the brax
EpisodeWrapper (episode_length = EPISODE_STEPS). Friction DR is per env slot in
multi_design_env.DesignGroupEnv.
"""

import copy

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from dm_control import mjcf
from mujoco import mjx
from mujoco.mjx._src.support import contact_force_dim

from t2.env.mj_utils import set_up_default_scene
from t2.mjx_env.base_env import MjxEnv, State, make_data, step
from t2.robogen.umi_on_legs_plus_plus import umi_on_legs_plus_plus_from_params

# TT config/env/mjx_umi_on_legs.yaml
CTRL_DT, SIM_DT, IMPRATIO, ITERATIONS, LS_ITERATIONS = 0.02, 0.005, 1, 5, 7
NUM_LEG_ACTUATORS = 12  # every one of the 128 bank choices has 12 leg actuators

# LeggedRobotCfg
EPISODE_STEPS = int(np.ceil(20. / CTRL_DT))             # env.episode_length_s
RESAMPLE_STEPS = int(10. / CTRL_DT)                     # commands.resampling_time
PUSH_STEPS = int(np.ceil(15. / CTRL_DT))                # domain_rand.push_interval_s
MAX_PUSH_VEL_XY = 1.
COMMAND_RANGES = {"lin_vel_x": (-1., 1.), "lin_vel_y": (-1., 1.),
                  "heading": (-3.14, 3.14)}             # heading_command = True
# A1RoughCfg overrides (2026-09-27 user decision, replaces D43 "A1 override 미사용"): action_scale,
# torques, dof_pos_limits, soft_dof_pos_limit. Robot-specific A1 values (init pos, default angles,
# kp 20 / kd 0.5) are not ported; the TT robot keeps its own home pose and actuators.
ACTION_SCALE, CLIP_ACTIONS, CLIP_OBSERVATIONS = 0.25, 100., 100.
SOFT_DOF_POS_LIMIT = 0.9
LIN_VEL, ANG_VEL, DOF_POS, DOF_VEL = 2.0, 0.25, 1.0, 0.05   # normalization.obs_scales
NOISE = {"lin_vel": 0.1, "ang_vel": 0.2, "gravity": 0.05,
         "dof_pos": 0.01, "dof_vel": 1.5}               # noise_scales, noise_level 1.0
# rewards.scales with zero scales removed and the rest multiplied by dt (_prepare_reward_function);
# termination is -0.0 so it is dropped. tracking_lin_vel 1.0 -> 5.0 (D45) -> 3.0 (D47, 2026-09-27 user
# decisions): t5_a1_nopos_1000it tracked ~19% of max, the same share standing still earns at sigma 0.25;
# x5 without the arm tracked 36% but fell more (survival 22 -> 13%). tracking_ang_vel 0.5 -> 1.5 (D50):
# t5_norail_x3_1000it turned only 20-36% of the commanded yaw, same idea as the linear x3.
REWARD_SCALES = {k: v * CTRL_DT for k, v in {
    "tracking_lin_vel": 3.0, "tracking_ang_vel": 1.5, "lin_vel_z": -2.0, "ang_vel_xy": -0.05,
    "torques": -0.0002, "dof_acc": -2.5e-7, "feet_air_time": 1.0, "collision": -1.,
    "action_rate": -0.01, "dof_pos_limits": -10.}.items()}
# only_positive_rewards: True clipped nearly every step to 0 and std ran away (t5_a1_1000it, std 1.04 -> 1.92,
# ep_len 131 -> 55); False at action_scale 0.5 died early (t5_smoke_nopos, ep_len 64 -> 14). Retrying
# False at action_scale 0.25 + adaptive KL (2026-09-27 user decision).
TRACKING_SIGMA, ONLY_POSITIVE_REWARDS = 0.25, False
# asset.* body-name substrings mapped to TT bodies. The trunk geoms live on
# `quadruped_body` (TT `base` is the geom-less free-joint body). Lists follow the legged_gym
# quadruped cfgs (A1 / ANYmal: terminate on base, penalize thigh + calf), 2026-09-26 user
# decision. Every calf geom (rod and knee `calf_motor`) counts; the foot is its own body.
FOOT_NAME = "foot"
PENALIZE_CONTACTS_ON = ("thigh", "calf")
TERMINATE_AFTER_CONTACTS_ON = ("quadruped_body",)
# D46/D47 (2026-09-27 user decisions, D35/D43 exception): drop the arx5 arm + gripper (4.05 kg, ~22 % of
# the robot) and the rail (bit 5, 1.18 kg) so pi_theta first learns to walk without the payload.
# Bit 5 then no longer changes the robot.
REMOVE_PARTS = ("/arx5/", "/rail/")


def compile_scene(scene):
    """MjModel of the TT scene without the REMOVE_PARTS subtrees, their actuators and key entries."""
    m = mjcf.Physics.from_mjcf_model(scene).model.ptr
    if not REMOVE_PARTS:
        return m
    arm = lambda name: any(p in name for p in REMOVE_PARTS)
    arm_q = [m.jnt_qposadr[j] for j in range(m.njnt) if arm(m.joint(j).name)]
    arm_u = [i for i in range(m.nu) if arm(m.actuator(i).name)]
    top = [m.body(b).name for b in range(m.nbody)
           if arm(m.body(b).name) and not arm(m.body(m.body_parentid[b]).name)]
    spec = mujoco.MjSpec.from_string(scene.to_xml_string(), assets=scene.get_assets())
    # MjSpec checks key sizes on every delete, so the keys are removed and re-added trimmed.
    keys = [dict(name=k.name, time=k.time, qpos=np.delete(k.qpos, arm_q) if len(k.qpos) else [],
                 ctrl=np.delete(k.ctrl, arm_u) if len(k.ctrl) else []) for k in spec.keys]
    for k in list(spec.keys):
        spec.delete(k)
    for a in [a for a in spec.actuators if arm(a.name)]:
        spec.delete(a)
    for name in top:
        spec.delete(spec.body(name))
    for k in keys:
        spec.add_key(**{f: v for f, v in k.items() if len(np.atleast_1d(v)) or f == "time"})
    return spec.compile()


def quat_rotate(q, v, inverse=False):
    """legged_gym quat_apply / quat_rotate_inverse with MuJoCo (w, x, y, z) order."""
    w, u = q[0], q[1:]
    b = jp.cross(u, v) * w * 2.
    return v * (2. * w ** 2 - 1.) + (-b if inverse else b) + u * jp.dot(u, v) * 2.


def wrap_to_pi(angle):
    return (angle + jp.pi) % (2 * jp.pi) - jp.pi


class LocomotionEnv(MjxEnv):
    action_size = NUM_LEG_ACTUATORS

    def __init__(self, choices, seed=None, command_ranges=COMMAND_RANGES, add_noise=True,
                 hold_command=False, domain_rand=True, uniforms=None):
        super().__init__(ctrl_dt=CTRL_DT, sim_dt=SIM_DT)
        if len(choices) != 8 or any(c not in (0, 1) for c in choices):
            raise ValueError("Expected an eight-bit TT quadruped choice")
        self.command_ranges, self.add_noise = command_ranges, add_noise
        # Evaluation only: keep info["command"] (vx, vy, yaw rate) as set by the caller, with no
        # resampling and no heading-to-yaw-rate control.
        self.hold_command = hold_command
        # Evaluation only (D52, legged_gym play.py): domain_rand=False drops observation noise,
        # pushes and (in DesignGroupEnv) friction randomisation.
        self.domain_rand = domain_rand
        self.add_noise = add_noise and domain_rand
        if uniforms is None:
            if seed is None:
                raise ValueError("Expected seed or 13 direct morphology uniforms")
            uniforms = np.random.RandomState(seed).uniform(size=13)
        else:
            uniforms = np.asarray(uniforms, dtype=float)
            if seed is not None:
                raise ValueError("Pass either seed or uniforms, not both")
            if (uniforms.shape != (13,) or not np.isfinite(uniforms).all()
                    or np.any((uniforms < 0.) | (uniforms > 1.))):
                raise ValueError("Expected 13 finite morphology uniforms in [0, 1]")
        robot = umi_on_legs_plus_plus_from_params(
            choices=copy.copy(choices), uniforms=uniforms.tolist())
        scene = set_up_default_scene(robot, add_plane=True, plane_z_pos=0.0)
        self.mj_model = m = compile_scene(scene)
        # TT MjxTrackEnv.preprocess_mj_model: only spheres (feet) and the floor collide.
        m.opt.timestep, m.opt.impratio = SIM_DT, IMPRATIO
        m.opt.iterations, m.opt.ls_iterations = ITERATIONS, LS_ITERATIONS
        collide = ((m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) |
                   (m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
        m.geom_contype[:] = m.geom_conaffinity[:] = collide
        bodies = [m.body(i).name for i in range(m.nbody)]
        find = lambda names: np.array([i for i, b in enumerate(bodies)
                                       if any(n in b for n in names)], dtype=int)
        self.feet = find([FOOT_NAME])
        self.penalised = find(PENALIZE_CONTACTS_ON)
        self.terminate = find(TERMINATE_AFTER_CONTACTS_ON)
        if len(self.feet) != 4 or len(self.terminate) != 1 or len(self.penalised) != 8:
            raise ValueError(f"Expected 4 feet, 1 trunk, 4 thighs + 4 calves, got {len(self.feet)}, "
                             f"{len(self.terminate)}, {len(self.penalised)}")
        # These geoms must touch the floor to report contact; bit 2 makes them collide with
        # the floor only, leaving TT's sphere contacts unchanged.
        contact_only = np.isin(m.geom_bodyid, np.r_[self.terminate, self.penalised])
        m.geom_contype[contact_only], m.geom_conaffinity[contact_only] = 2, 0
        m.geom_conaffinity[m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE] = 3
        free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Expected one floating base")
        self.qadr, self.vadr = int(m.jnt_qposadr[free[0]]), int(m.jnt_dofadr[free[0]])
        # Arm (arx5) and rail stay at home ctrl (D10).
        self.leg = np.array([i for i in range(m.nu) if "/arx5/" not in m.actuator(i).name
                             and "/rail/" not in m.actuator(i).name])
        if len(self.leg) != NUM_LEG_ACTUATORS:
            raise ValueError(f"Expected {NUM_LEG_ACTUATORS} leg actuators, got {len(self.leg)}")
        joints = m.actuator_trnid[self.leg, 0]
        self.qids, self.vids = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
        # legged_gym leaves the P target unclamped (only torques are limited); TT ctrlrange equals
        # the joint range, so joint limits still hold.
        m.actuator_ctrllimited[self.leg] = 0
        # _process_dof_props soft limits
        lo, hi = m.jnt_range[joints, 0], m.jnt_range[joints, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_DOF_POS_LIMIT
        self.dof_pos_limits = jp.array(np.stack([mid - half, mid + half]))
        self.home = jp.array(m.key("home").qpos)
        self.ctrl = jp.array(m.key("home").ctrl)
        self.default_dof_pos = self.home[self.qids]
        self.morphology = jp.array(np.r_[choices, uniforms], dtype=jp.float32)
        self.commands_scale = jp.array([LIN_VEL, LIN_VEL, ANG_VEL])
        n = NUM_LEG_ACTUATORS
        self.noise_vec = jp.concatenate([
            jp.full(3, NOISE["lin_vel"] * LIN_VEL), jp.full(3, NOISE["ang_vel"] * ANG_VEL),
            jp.full(3, NOISE["gravity"]), jp.zeros(3),
            jp.full(n, NOISE["dof_pos"] * DOF_POS), jp.full(n, NOISE["dof_vel"] * DOF_VEL),
            jp.zeros(n), jp.zeros(self.morphology.size)])
        self.contact_bodies = jp.array(np.r_[self.feet, self.penalised, self.terminate])
        self.geom_body = jp.array(m.geom_bodyid)
        self.mjx_model = mjx.put_model(m, impl="jax")

    def sample_command(self, rng):
        """_resample_commands: [vx, vy, yaw rate (from heading), heading]."""
        keys = jax.random.split(rng, 3)
        vx, vy, heading = (jax.random.uniform(k, minval=lo, maxval=hi)
                           for k, (lo, hi) in zip(keys, (self.command_ranges["lin_vel_x"],
                                                         self.command_ranges["lin_vel_y"],
                                                         self.command_ranges["heading"])))
        xy = jp.array([vx, vy])
        xy = xy * (jp.linalg.norm(xy) > 0.2)  # set small commands to zero
        return jp.array([xy[0], xy[1], 0., heading])

    def heading_to_yaw_rate(self, data, command):
        forward = quat_rotate(data.qpos[self.qadr + 3:self.qadr + 7], jp.array([1., 0., 0.]))
        heading = jp.arctan2(forward[1], forward[0])
        return command.at[2].set(jp.clip(0.5 * wrap_to_pi(command[3] - heading), -1., 1.))

    def base_state(self, data):
        """base_lin_vel, base_ang_vel, projected_gravity in the base frame."""
        quat = data.qpos[self.qadr + 3:self.qadr + 7]
        qvel = data.qvel[self.vadr:self.vadr + 6]
        return (quat_rotate(quat, qvel[:3], inverse=True),  # world -> base frame
                qvel[3:],  # MuJoCo free-joint angular velocity is already in the base frame
                quat_rotate(quat, jp.array([0., 0., -1.]), inverse=True))

    def contact_forces(self, data, bodies):
        """Isaac net_contact_force: summed world-frame contact force on each body, (len(bodies), 3)."""
        contact = data._impl.contact
        net = jp.zeros((len(bodies), 3))
        for dim in np.unique(contact.dim):  # static per model
            force, idx = contact_force_dim(self.mjx_model, data, dim)
            world = jp.einsum("ci,cij->cj", force[:, :3], contact.frame[idx])
            geom = contact.geom[idx]
            # Force acts on geom 1 along the normal from geom 0.
            sign = ((self.geom_body[geom[:, 1], None] == bodies).astype(float) -
                    (self.geom_body[geom[:, 0], None] == bodies))
            net += sign.T @ world
        return net

    def reward(self, data, action, command, info):
        """compute_reward + check_termination; returns (reward, done, scaled terms, info)."""
        lin_vel, ang_vel, _ = self.base_state(data)
        dof_pos, dof_vel = data.qpos[self.qids], data.qvel[self.vids]
        forces = self.contact_forces(data, self.contact_bodies)
        nf, np_ = len(self.feet), len(self.penalised)
        feet, penalised, terminate = forces[:nf], forces[nf:nf + np_], forces[nf + np_:]
        # _reward_feet_air_time
        contact = feet[:, 2] > 1.
        contact_filt = contact | info["last_contacts"]
        first_contact = (info["feet_air_time"] > 0.) & contact_filt
        air_time = info["feet_air_time"] + self.dt
        terms = {
            "tracking_lin_vel": jp.exp(-jp.sum(jp.square(command[:2] - lin_vel[:2])) / TRACKING_SIGMA),
            "tracking_ang_vel": jp.exp(-jp.square(command[2] - ang_vel[2]) / TRACKING_SIGMA),
            "lin_vel_z": jp.square(lin_vel[2]),
            "ang_vel_xy": jp.sum(jp.square(ang_vel[:2])),
            "torques": jp.sum(jp.square(data.actuator_force[self.leg])),
            "dof_acc": jp.sum(jp.square((info["last_dof_vel"] - dof_vel) / self.dt)),
            "feet_air_time": jp.sum((air_time - 0.5) * first_contact)
                             * (jp.linalg.norm(command[:2]) > 0.1),
            # An empty-array reduction here slowed the whole step 14x on XLA, hence the guard.
            "collision": jp.sum(jp.linalg.norm(penalised, axis=-1) > 0.1) if np_ else jp.zeros(()),
            "action_rate": jp.sum(jp.square(info["last_actions"] - action)),
            "dof_pos_limits": jp.sum(jp.clip(self.dof_pos_limits[0] - dof_pos, 0.) +
                                     jp.clip(dof_pos - self.dof_pos_limits[1], 0.)),
        }
        terms = {k: v * REWARD_SCALES[k] for k, v in terms.items()}
        reward = sum(terms.values())
        if ONLY_POSITIVE_REWARDS:
            reward = jp.maximum(reward, 0.)
        done = jp.any(jp.linalg.norm(terminate, axis=-1) > 1.)
        info = {**info, "feet_air_time": air_time * ~contact_filt, "last_contacts": contact,
                "last_dof_vel": dof_vel, "last_actions": action}
        return reward, done.astype(jp.float32), terms, info

    def observation(self, data, command, action, rng):
        """compute_observations (blind, plane) + morphology E for the conditioned pi_theta."""
        lin_vel, ang_vel, gravity = self.base_state(data)
        obs = jp.concatenate([
            lin_vel * LIN_VEL,
            ang_vel * ANG_VEL,
            gravity,
            command[:3] * self.commands_scale,
            (data.qpos[self.qids] - self.default_dof_pos) * DOF_POS,
            data.qvel[self.vids] * DOF_VEL,
            action,
            self.morphology,
        ])
        if self.add_noise:
            obs += (2 * jax.random.uniform(rng, obs.shape) - 1) * self.noise_vec
        return jp.clip(obs, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS)

    def reset(self, rng):
        rng, dof_key, vel_key, cmd_key, noise_key = jax.random.split(rng, 5)
        # _reset_dofs: default * U(0.5, 1.5), zero velocity; _reset_root_states: base vel U(-0.5, 0.5)
        qpos = self.home.at[self.qids].set(self.default_dof_pos * jax.random.uniform(
            dof_key, (NUM_LEG_ACTUATORS,), minval=0.5, maxval=1.5))
        qvel = jp.zeros(self.mj_model.nv).at[self.vadr:self.vadr + 6].set(
            jax.random.uniform(vel_key, (6,), minval=-0.5, maxval=0.5))
        data = mjx.forward(self.mjx_model, make_data(self.mj_model, qpos=qpos, qvel=qvel,
                                                     ctrl=self.ctrl, impl="jax"))
        command = self.heading_to_yaw_rate(data, self.sample_command(cmd_key))
        action = jp.zeros(self.action_size)
        zero = jp.zeros(())
        return State(data=data, obs=self.observation(data, command, action, noise_key),
                     reward=zero, done=zero, metrics={k: zero for k in REWARD_SCALES},
                     info={"rng": rng, "command": command, "episode_step": jp.int32(0),
                           "last_actions": action, "last_dof_vel": jp.zeros(NUM_LEG_ACTUATORS),
                           "feet_air_time": jp.zeros(len(self.feet)),
                           "last_contacts": jp.zeros(len(self.feet), dtype=bool)})

    def step(self, state, action):
        action = jp.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
        targets = self.ctrl.at[self.leg].add(action * ACTION_SCALE)
        data = step(self.mjx_model, state.data, targets, self.n_substeps)
        t = state.info["episode_step"] + 1
        rng, cmd_key, noise_key, push_key = jax.random.split(state.info["rng"], 4)
        command = state.info["command"]
        if not self.hold_command:
            command = jp.where(t % RESAMPLE_STEPS == 0, self.sample_command(cmd_key), command)
            command = self.heading_to_yaw_rate(data, command)
        reward, done, terms, info = self.reward(data, action, command, state.info)
        obs = self.observation(data, command, action, noise_key)
        # _push_robots after the observation, as in post_physics_step.
        base_xy = slice(self.vadr, self.vadr + 2)
        push = jax.random.uniform(push_key, (2,), minval=-MAX_PUSH_VEL_XY, maxval=MAX_PUSH_VEL_XY)
        if self.domain_rand:
            data = data.replace(qvel=data.qvel.at[base_xy].set(
                jp.where(t % PUSH_STEPS == 0, push, data.qvel[base_xy])))
        return state.replace(data=data, obs=obs, reward=reward, done=done, metrics=terms,
                             info={**info, "rng": rng, "command": command, "episode_step": t})
