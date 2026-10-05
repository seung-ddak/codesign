"""T8 multi-task env (D63, Q16 a, Q17 b): LocomotionEnv on the legged_gym rough terrain + jump.

Terrain (terrain.py), height scan (17 x 11 points rotated by base yaw, obs = clip(z - 0.5 - h, -1, 1)
x 5, noise 0.1), spawn (+-1 m around the tile origin) and terrain curriculum follow legged_gym
LeggedRobotCfg / legged_robot.py. All designs share one heightfield (DesignGroupEnv keeps a single
copy).

Jump is ours (legged_gym has none; D36 exception, Q17 b): a resampled command becomes a jump with
JUMP_PROB (vx = vy = yaw = 0, flag 1). jump_air rewards all four feet off the ground, jump_height
the base rise above standing height, and lin_vel_z is not penalised while jumping.

Command: (vx, vy, yaw rate, heading, jump). Observation (257): lin vel 3, ang vel 3, gravity 3,
command (vx, vy, yaw, jump) 4, dof pos 12, dof vel 12, last action 12, heights 187, morphology 21.

Curriculum under brax auto-reset: the reset cannot see the finished episode, so the level and
terrain column live in AutoResetWrapper_preserve_info, are updated in step() when the episode ends
(legged_gym _update_terrain_curriculum), and the first step of the next episode moves the base from
the reset's random tile to the curriculum tile. That first step therefore acts on a height scan of
the reset tile.
"""

import functools

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

import terrain as T
from locomotion_env import (ANG_VEL, CTRL_DT, DOF_POS, DOF_VEL, EPISODE_STEPS, LIN_VEL, NOISE,
                            NUM_LEG_ACTUATORS, REWARD_SCALES, LocomotionEnv, State, make_data)

HEIGHT_MEASUREMENTS, HEIGHT_NOISE, HEIGHT_OFFSET = 5.0, 0.1, 0.5  # obs_scales, noise_scales, legged_robot.py
JUMP_PROB = 0.2
TASK_REWARD_SCALES = {**REWARD_SCALES, "jump_air": 1.0 * CTRL_DT, "jump_height": 5.0 * CTRL_DT}
PRESERVE = "AutoResetWrapper_preserve_info"


@functools.cache
def shared_terrain(seed=0):
    hf, origins = T.generate(seed)
    lo, hi = int(hf.min()), int(hf.max())
    # MuJoCo hfield rows run along y, columns along x; legged_gym height_samples is [x, y].
    data = jp.asarray(((hf.T - lo) / (hi - lo)).ravel(), jp.float32)
    return hf, lo, hi, data, jp.asarray(hf, jp.float32) * T.VERTICAL_SCALE, jp.asarray(origins, jp.float32)


class TaskEnv(LocomotionEnv):
    add_plane = False

    def __init__(self, choices, seed=None, **kwargs):
        self.hf, self.hf_lo, self.hf_hi, self.hfield, self.heights, self.origins = shared_terrain()
        super().__init__(choices, seed, **kwargs)
        self.mj_model.hfield_data[:] = np.asarray(self.hfield)  # CPU copy for rendering only
        self.mjx_model = self.mjx_model.replace(hfield_data=self.hfield)
        self.stand_height = float(self.home[self.qadr + 2])
        px, py = np.meshgrid(T.MEASURED_POINTS_X, T.MEASURED_POINTS_Y, indexing="ij")
        self.height_points = jp.array(np.stack([px.ravel(), py.ravel()], -1))
        n, nh = NUM_LEG_ACTUATORS, px.size
        self.commands_scale = jp.array([LIN_VEL, LIN_VEL, ANG_VEL, 1.])
        self.noise_vec = jp.concatenate([
            jp.full(3, NOISE["lin_vel"] * LIN_VEL), jp.full(3, NOISE["ang_vel"] * ANG_VEL),
            jp.full(3, NOISE["gravity"]), jp.zeros(4),
            jp.full(n, NOISE["dof_pos"] * DOF_POS), jp.full(n, NOISE["dof_vel"] * DOF_VEL),
            jp.zeros(n), jp.full(nh, HEIGHT_NOISE * HEIGHT_MEASUREMENTS), jp.zeros(self.morphology.size)])

    def add_terrain(self, scene):
        nrow, ncol = self.hf.shape[1], self.hf.shape[0]
        sx, sy = (ncol - 1) * T.HORIZONTAL_SCALE / 2, (nrow - 1) * T.HORIZONTAL_SCALE / 2
        scene.asset.add("hfield", name="terrain", nrow=nrow, ncol=ncol,
                        size=[sx, sy, (self.hf_hi - self.hf_lo) * T.VERTICAL_SCALE, 1.])
        scene.worldbody.add("geom", name="terrain", type="hfield", hfield="terrain",
                            pos=[sx - T.BORDER_SIZE, sy - T.BORDER_SIZE, self.hf_lo * T.VERTICAL_SCALE])

    def finish_model(self, m):
        # MJX has no cylinder-heightfield collision. Thigh / calf cylinders only sense contact
        # (collision penalty), so they become capsules of the same radius and half-length after
        # compile (mass and inertia are already fixed).
        cyl = (m.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER) & (m.geom_contype > 0)
        m.geom_type[cyl] = mujoco.mjtGeom.mjGEOM_CAPSULE
        m.geom_rbound[cyl] = m.geom_size[cyl, 0] + m.geom_size[cyl, 1]

    def terrain_height(self, xy):
        """legged_robot._get_heights lookup: min of three neighbouring samples, world xy (..., 2)."""
        p = ((xy + T.BORDER_SIZE) / T.HORIZONTAL_SCALE).astype(jp.int32)
        px = jp.clip(p[..., 0], 0, self.heights.shape[0] - 2)
        py = jp.clip(p[..., 1], 0, self.heights.shape[1] - 2)
        h = self.heights
        return jp.minimum(jp.minimum(h[px, py], h[px + 1, py]), h[px, py + 1])

    def measured_heights(self, data):
        q = data.qpos[self.qadr + 3:self.qadr + 7]
        yaw = jp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        c, s = jp.cos(yaw), jp.sin(yaw)
        pts = self.height_points @ jp.array([[c, s], [-s, c]])  # quat_apply_yaw
        return self.terrain_height(pts + data.qpos[self.qadr:self.qadr + 2])

    def sample_command(self, rng):
        k1, k2 = jax.random.split(rng)
        c = super().sample_command(k1)
        return jp.where(jax.random.uniform(k2) < JUMP_PROB,
                        jp.array([0., 0., 0., c[3], 1.]), jp.append(c, 0.))

    def heading_to_yaw_rate(self, data, command):
        command = super().heading_to_yaw_rate(data, command)
        return command.at[2].multiply(1. - command[4])

    def observation(self, data, command, action, rng):
        lin_vel, ang_vel, gravity = self.base_state(data)
        heights = jp.clip(data.qpos[self.qadr + 2] - HEIGHT_OFFSET - self.measured_heights(data), -1., 1.)
        obs = jp.concatenate([
            lin_vel * LIN_VEL, ang_vel * ANG_VEL, gravity,
            command[jp.array([0, 1, 2, 4])] * self.commands_scale,
            (data.qpos[self.qids] - self.default_dof_pos) * DOF_POS,
            data.qvel[self.vids] * DOF_VEL, action,
            heights * HEIGHT_MEASUREMENTS, self.morphology])
        if self.add_noise:
            obs += (2 * jax.random.uniform(rng, obs.shape) - 1) * self.noise_vec
        return jp.clip(obs, -100., 100.)

    def reward(self, data, action, command, info):
        reward, done, terms, info = super().reward(data, action, command, info)
        jump = command[4]
        reward -= terms["lin_vel_z"] * jump
        base = data.qpos[self.qadr:self.qadr + 3]
        rise = jp.clip(base[2] - self.terrain_height(base[:2]) - self.stand_height, 0.)
        extra = {"jump_air": jp.all(~info["last_contacts"]) * jump * TASK_REWARD_SCALES["jump_air"],
                 "jump_height": rise * jump * TASK_REWARD_SCALES["jump_height"]}
        terms = {**terms, "lin_vel_z": terms["lin_vel_z"] * (1. - jump), **extra}
        return reward + sum(extra.values()), done, terms, info

    def reset(self, rng):
        # LocomotionEnv.reset + legged_gym _get_env_origins / _reset_root_states.
        rng, dof_key, vel_key, cmd_key, noise_key, level_key, col_key, xy_key = jax.random.split(rng, 8)
        level = jax.random.randint(level_key, (), 0, T.MAX_INIT_TERRAIN_LEVEL + 1)
        col = jax.random.randint(col_key, (), 0, T.NUM_COLS)
        origin = self.origins[level, col]
        offset = origin + jp.append(jax.random.uniform(xy_key, (2,), minval=-1., maxval=1.), 0.)
        qpos = self.home.at[self.qids].set(self.default_dof_pos * jax.random.uniform(
            dof_key, (NUM_LEG_ACTUATORS,), minval=0.5, maxval=1.5))
        qpos = qpos.at[self.qadr:self.qadr + 3].add(offset)
        qvel = jp.zeros(self.mj_model.nv).at[self.vadr:self.vadr + 6].set(
            jax.random.uniform(vel_key, (6,), minval=-0.5, maxval=0.5))
        data = mjx.forward(self.mjx_model, make_data(self.mj_model, qpos=qpos, qvel=qvel,
                                                     ctrl=self.ctrl, impl="jax"))
        command = self.heading_to_yaw_rate(data, self.sample_command(cmd_key))
        action = jp.zeros(self.action_size)
        zero = jp.zeros(())
        return State(data=data, obs=self.observation(data, command, action, noise_key), reward=zero,
                     done=zero, metrics={k: zero for k in TASK_REWARD_SCALES},
                     info={"rng": rng, "command": command, "episode_step": jp.int32(0),
                            "last_actions": action, "last_dof_vel": jp.zeros(NUM_LEG_ACTUATORS),
                            "feet_air_time": jp.zeros(len(self.feet)),
                            "last_contacts": jp.zeros(len(self.feet), dtype=bool),
                            "spawn_origin": origin, PRESERVE: {"level": level, "col": col}})

    def step(self, state, action):
        info, keep = state.info, state.info[PRESERVE]
        target = self.origins[keep["level"], keep["col"]]
        delta = (target - info["spawn_origin"]) * (info["episode_step"] == 0)
        data = state.data.replace(qpos=state.data.qpos.at[self.qadr:self.qadr + 3].add(delta))
        state = super().step(state.replace(data=data, info={**info, "spawn_origin": info["spawn_origin"] + delta}),
                             action)
        # _update_terrain_curriculum when this episode ends (termination or time-out).
        end = (state.done > 0) | (state.info["episode_step"] >= EPISODE_STEPS)
        dist = jp.linalg.norm(state.data.qpos[self.qadr:self.qadr + 2] - target[:2])
        up = dist > T.TERRAIN_LENGTH / 2
        down = (dist < jp.linalg.norm(state.info["command"][:2]) * EPISODE_STEPS * CTRL_DT * 0.5) & ~up
        level = keep["level"] + up.astype(jp.int32) - down.astype(jp.int32)
        level = jp.where(level >= T.NUM_ROWS,
                         jax.random.randint(jax.random.fold_in(state.info["rng"], 1), (), 0, T.NUM_ROWS),
                         jp.maximum(level, 0))
        keep = {**keep, "level": jp.where(end, level, keep["level"])}
        return state.replace(info={**state.info, PRESERVE: keep})

