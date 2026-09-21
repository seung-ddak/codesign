"""Stair-climbing MJX environment for the quadruped design space.

Outcome-based task (remind.md D9/D10): an episode succeeds when the base reaches
the top of the staircase within the time limit without falling over. The dense
terms below exist only so PPO can learn a controller at all -- the binary
success flag in `metrics` is what the design search (L0~L3) reads.
"""

from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from dm_control import mjcf
from mujoco import mjx

from t2.mjx_env.base_env import MjxEnv, State, make_data
from t2.mjx_env.base_env import step as mjx_step

ARM_TOKEN = "/arx5/"
STAIR_PREFIX = "stair"

# remind.md §4.2. The zarr of morphology attributes and the training scene must
# be built from the same geometry, otherwise the stored model fields no longer
# line up with the compiled model.
DEFAULT_STAIRS = dict(
    num_stairs=10,
    rise=0.10,
    run=0.28,
    start_x=0.5,
    width=2.0,
    landing_length=1.0,
)

# terrain curriculum: env 마다 단높이를 이 범위에서 뽑는다. brax 의 domain
# randomization 은 학습 중 바뀌지 않으므로 시간축 curriculum 대신 난이도를
# env 에 분산시킨다. 단높이는 zarr 의 `geom_size` 를 타고 정책 obs 에 들어가므로
# 정책은 자기가 마주한 계단 높이를 관측할 수 있다.
STAIR_RISE_RANGE = (0.02, 0.10)


def add_stairs(
    root: mjcf.RootElement,
    num_stairs: int,
    rise: float,
    run: float,
    start_x: float,
    width: float,
    landing_length: float,
) -> mjcf.RootElement:
    """Solid boxes rising along +x, plus a flat landing at the top."""
    for i in range(num_stairs):
        height = rise * (i + 1)
        root.worldbody.add(
            "geom",
            name=f"{STAIR_PREFIX}_{i}",
            type="box",
            pos=[start_x + run * i + run / 2, 0.0, height / 2],
            size=[run / 2, width / 2, height / 2],
            rgba=[0.5, 0.5, 0.55, 1],
        )
    top = rise * num_stairs
    root.worldbody.add(
        "geom",
        name=f"{STAIR_PREFIX}_landing",
        type="box",
        pos=[start_x + run * num_stairs + landing_length / 2, 0.0, top / 2],
        size=[landing_length / 2, width / 2, top / 2],
        rgba=[0.45, 0.45, 0.5, 1],
    )
    return root


def build_stair_scene(
    mjcf_model: mjcf.RootElement, rise: Optional[float] = None
) -> mjcf.RootElement:
    """The one scene definition shared by morphology dumping and training."""
    from t2.env.mj_utils import add_mocap_body_with_site, set_up_default_scene

    mjcf_model = add_mocap_body_with_site(
        set_up_default_scene(mjcf_model, add_plane=True, plane_z_pos=0.0),
        name="target",
    )
    stairs = dict(DEFAULT_STAIRS)
    if rise is not None:
        stairs["rise"] = rise
    return add_stairs(mjcf_model, **stairs)


class StairMjxEnv(MjxEnv):
    def __init__(
        self,
        mj_model_fn: Callable[[], mujoco.MjModel],
        max_episode_length: int,
        ctrl_dt: float,
        sim_dt: float,
        action_scale: float,
        solver_iterations: int,
        solver_ls_iterations: int,
        impratio: int,
        impl: Literal["jax", "warp"],
        num_stairs: int,
        stair_rise: float,
        stair_run: float,
        stair_start_x: float,
        goal_margin: float,
        max_progress_speed: float,
        reward_progress: float,
        reward_alive: float,
        reward_success: float,
        penalty_orientation: float,
        penalty_action_rate: float,
        penalty_torque: float,
        penalty_joint_vel: float,
        penalty_joint_limit: float,
        penalty_termination: float,
        soft_joint_pos_limit_factor: float,
        termination_upright: float,
        termination_height: float,
        reset_xy_noise: float,
        reset_yaw_noise: float,
        reset_qpos_noise: float,
        reset_qvel_noise: float,
        obs_noise: dict[str, float],
        clip_actions: tuple[float, float] = (-10.0, 10.0),
        goal_x: Optional[float] = None,
    ):
        super().__init__(ctrl_dt, sim_dt)
        self.max_episode_length = max_episode_length
        self.action_scale = action_scale
        self.clip_actions = clip_actions
        self.solver_iterations = solver_iterations
        self.solver_ls_iterations = solver_ls_iterations
        self.impratio = impratio
        self.impl = impl

        self.num_stairs = num_stairs
        self.stair_rise = stair_rise
        self.stair_run = stair_run
        self.stair_start_x = stair_start_x
        # goal_x 를 직접 주면 코스 길이를 로봇 능력에 맞춰 줄일 수 있다 (§4.2).
        self.goal_x = (
            goal_x
            if goal_x is not None
            else stair_start_x + stair_run * num_stairs + goal_margin
        )
        self.max_progress_speed = max_progress_speed

        self.reward_progress = reward_progress
        self.reward_alive = reward_alive
        self.reward_success = reward_success
        self.penalty_orientation = penalty_orientation
        self.penalty_action_rate = penalty_action_rate
        self.penalty_torque = penalty_torque
        self.penalty_joint_vel = penalty_joint_vel
        self.penalty_joint_limit = penalty_joint_limit
        self.penalty_termination = penalty_termination
        self.termination_upright = termination_upright
        self.termination_height = termination_height

        self.reset_xy_noise = reset_xy_noise
        self.reset_yaw_noise = reset_yaw_noise
        self.reset_qpos_noise = reset_qpos_noise
        self.reset_qvel_noise = reset_qvel_noise
        self.obs_noise = obs_noise

        self.mj_model = mj_model_fn()
        self.preprocess_mj_model(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model, impl=impl)

        soft = soft_joint_pos_limit_factor
        mid = self.mj_model.jnt_range[self._leg_jnt_ids].mean(axis=-1)
        half = np.diff(self.mj_model.jnt_range[self._leg_jnt_ids], axis=-1)[:, 0] / 2
        self._soft_jnt_range = jnp.array(
            np.stack([mid - half * soft, mid + half * soft], axis=-1)
        )

    # -- setup ---------------------------------------------------------------

    def preprocess_mj_model(self, mj_model: mujoco.MjModel):
        mj_model.opt.timestep = self.sim_dt
        mj_model.opt.iterations = self.solver_iterations
        mj_model.opt.ls_iterations = self.solver_ls_iterations
        mj_model.opt.impratio = self.impratio

        # MJX has no cylinder-box collision pair, and TT's robots are full of
        # cylinders: keep contact on feet (spheres) + terrain only.
        geom_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            for i in range(mj_model.ngeom)
        ]
        is_sphere = mj_model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
        is_plane = mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        is_stair = np.array([n.startswith(STAIR_PREFIX) for n in geom_names])
        self._stair_geom_ids = jnp.array(np.flatnonzero(is_stair))
        keep = is_sphere | is_plane | is_stair
        mj_model.geom_contype[~keep] = 0
        mj_model.geom_conaffinity[~keep] = 0
        mj_model.geom_contype[keep] = 1
        mj_model.geom_conaffinity[keep] = 1

        free_jnts = np.flatnonzero(mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        assert len(free_jnts) == 1, "expected exactly one floating base"
        self._base_body_id = int(mj_model.jnt_bodyid[free_jnts[0]])
        self._base_qpos_adr = int(mj_model.jnt_qposadr[free_jnts[0]])
        self._base_qvel_adr = int(mj_model.jnt_dofadr[free_jnts[0]])

        act_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for i in range(mj_model.nu)
        ]
        leg_act = [i for i, n in enumerate(act_names) if ARM_TOKEN not in n]
        arm_act = [i for i, n in enumerate(act_names) if ARM_TOKEN in n]
        self._leg_act_ids = np.array(leg_act)
        self._arm_act_ids = np.array(arm_act)
        self._leg_jnt_ids = mj_model.actuator_trnid[self._leg_act_ids, 0]
        self._leg_qpos_ids = jnp.array(mj_model.jnt_qposadr[self._leg_jnt_ids])
        self._leg_qvel_ids = jnp.array(mj_model.jnt_dofadr[self._leg_jnt_ids])

        key = mj_model.key(("home"))
        self._init_qpos = jnp.array(key.qpos)
        self._default_ctrl = jnp.array(key.qpos[7:])
        self._default_leg_ctrl = self._default_ctrl[self._leg_act_ids]

    @property
    def action_size(self) -> int:  # legs only, the arm is held at its home pose
        return len(self._leg_act_ids)

    # -- helpers -------------------------------------------------------------

    def _terrain_z(self, x: jax.Array) -> jax.Array:
        """지형 높이를 상수가 아니라 **현재 모델에서** 읽는다.

        env 마다 단높이가 다르게 randomize 되므로 (terrain curriculum), 상수로
        계산하면 다른 높이의 env 에서 지형 높이를 오판해 즉시 종료된다.
        """
        pos = self.mjx_model.geom_pos[self._stair_geom_ids]
        size = self.mjx_model.geom_size[self._stair_geom_ids]
        on_step = jnp.abs(x - pos[:, 0]) <= size[:, 0]
        return jnp.max(jnp.where(on_step, pos[:, 2] + size[:, 2], 0.0))

    def _base_frame(self, data: mjx.Data):
        quat = data.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7]
        rot = mjx._src.math.quat_to_mat(quat)
        return quat, rot

    def _get_obs(self, data: mjx.Data, info: dict[str, jax.Array]) -> jax.Array:
        _, rot = self._base_frame(data)
        proj_gravity = rot.T @ jnp.array([0.0, 0.0, -1.0])
        lin_vel = rot.T @ data.qvel[self._base_qvel_adr : self._base_qvel_adr + 3]
        ang_vel = rot.T @ data.qvel[self._base_qvel_adr + 3 : self._base_qvel_adr + 6]
        q = data.qpos[self._leg_qpos_ids] - self._default_leg_ctrl
        qd = data.qvel[self._leg_qvel_ids]

        pos = data.qpos[self._base_qpos_adr : self._base_qpos_adr + 3]
        goal_world = jnp.array([self.goal_x - pos[0], -pos[1], 0.0])
        goal_body = (rot.T @ goal_world)[:2]
        height = pos[2] - self._terrain_z(pos[0])

        obs = jnp.concatenate(
            [
                proj_gravity,
                ang_vel,
                lin_vel,
                q,
                qd,
                info["last_action"],
                goal_body,
                height[None],
            ]
        )
        noise = jnp.concatenate(
            [
                jnp.full(3, self.obs_noise.get("gravity", 0.0)),
                jnp.full(3, self.obs_noise.get("ang_vel", 0.0)),
                jnp.full(3, self.obs_noise.get("lin_vel", 0.0)),
                jnp.full(q.size, self.obs_noise.get("joint_pos", 0.0)),
                jnp.full(qd.size, self.obs_noise.get("joint_vel", 0.0)),
                jnp.zeros(info["last_action"].size),
                jnp.full(3, self.obs_noise.get("goal", 0.0)),
            ]
        )
        rng, key = jax.random.split(info["rng"])
        info["rng"] = rng
        return obs + jax.random.uniform(key, obs.shape, minval=-1.0, maxval=1.0) * noise

    def _init_metrics(self) -> dict[str, jax.Array]:
        zero = jnp.zeros(())
        return {
            "success": zero,
            "fell": zero,
            "progress_x": zero,
            "max_x": zero,
            "reward/progress": zero,
            "reward/alive": zero,
            "reward/orientation": zero,
            "reward/action_rate": zero,
            "reward/torque": zero,
            "reward/joint_vel": zero,
            "reward/joint_limit": zero,
        }

    # -- api -----------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        rng, xy_key, yaw_key, q_key, qd_key, obs_key = jax.random.split(rng, 6)
        qpos = self._init_qpos
        xy = jax.random.uniform(xy_key, (2,), minval=-1.0, maxval=1.0)
        qpos = qpos.at[self._base_qpos_adr : self._base_qpos_adr + 2].add(
            xy * self.reset_xy_noise
        )
        yaw = (
            jax.random.uniform(yaw_key, (), minval=-1.0, maxval=1.0)
            * self.reset_yaw_noise
        )
        quat = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        qpos = qpos.at[self._base_qpos_adr + 3 : self._base_qpos_adr + 7].set(quat)
        qpos = qpos.at[self._leg_qpos_ids].add(
            jax.random.uniform(q_key, (self.action_size,), minval=-1.0, maxval=1.0)
            * self.reset_qpos_noise
        )
        qvel = (
            jax.random.uniform(qd_key, (self.mj_model.nv,), minval=-1.0, maxval=1.0)
            * self.reset_qvel_noise
        )
        data = make_data(
            self.mj_model, qpos=qpos, qvel=qvel, ctrl=self._default_ctrl, impl=self.impl
        )
        data = mjx.forward(self.mjx_model, data)

        pos_x = data.qpos[self._base_qpos_adr]
        info = {
            "rng": obs_key,
            "last_action": jnp.zeros(self.action_size),
            "prev_dist": jnp.maximum(self.goal_x - pos_x, 0.0),
            "max_x": pos_x,
            # brax's evaluator pulls these two names out of info (evaluator.py:75);
            # for this task they mean distance-to-goal and tilt.
            "pos_err": jnp.maximum(self.goal_x - pos_x, 0.0),
            "orn_err": jnp.zeros(()),
        }
        obs = self._get_obs(data, info)
        return State(
            data=data,
            obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics=self._init_metrics(),
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, *self.clip_actions)
        leg_target = self._default_leg_ctrl + action * self.action_scale
        ctrl = self._default_ctrl.at[self._leg_act_ids].set(leg_target)
        data = mjx_step(self.mjx_model, state.data, ctrl, self.n_substeps)

        pos = data.qpos[self._base_qpos_adr : self._base_qpos_adr + 3]
        _, rot = self._base_frame(data)
        proj_gravity = rot.T @ jnp.array([0.0, 0.0, -1.0])
        height = pos[2] - self._terrain_z(pos[0])

        dist = jnp.maximum(self.goal_x - pos[0], 0.0)
        progress = jnp.clip(
            (state.info["prev_dist"] - dist) / self.dt,
            -self.max_progress_speed,
            self.max_progress_speed,
        )

        fell = jnp.logical_or(
            proj_gravity[2] > -self.termination_upright,
            height < self.termination_height,
        ).astype(jnp.float32)
        success = (pos[0] >= self.goal_x).astype(jnp.float32)
        done = jnp.clip(fell + success, 0.0, 1.0)

        q = data.qpos[self._leg_qpos_ids]
        qd = data.qvel[self._leg_qvel_ids]
        joint_limit = jnp.sum(
            jnp.clip(self._soft_jnt_range[:, 0] - q, 0.0, None)
            + jnp.clip(q - self._soft_jnt_range[:, 1], 0.0, None)
        )
        terms = {
            "reward/progress": self.reward_progress * progress,
            "reward/alive": self.reward_alive * (1.0 - done),
            "reward/orientation": self.penalty_orientation * (1.0 + proj_gravity[2]),
            "reward/action_rate": self.penalty_action_rate
            * jnp.sum(jnp.square(action - state.info["last_action"])),
            "reward/torque": self.penalty_torque
            * jnp.sum(jnp.square(data.actuator_force[self._leg_act_ids])),
            "reward/joint_vel": self.penalty_joint_vel * jnp.sum(jnp.square(qd)),
            "reward/joint_limit": self.penalty_joint_limit * joint_limit,
        }
        reward = (
            sum(terms.values())
            + self.reward_success * success
            + self.penalty_termination * fell
        )
        reward = jnp.nan_to_num(reward)

        info = dict(state.info)
        info["last_action"] = action
        info["prev_dist"] = dist
        info["max_x"] = jnp.maximum(info["max_x"], pos[0])
        info["pos_err"] = dist
        info["orn_err"] = 1.0 + proj_gravity[2]
        obs = self._get_obs(data, info)

        metrics = dict(state.metrics)
        metrics.update(terms)
        metrics["success"] = success
        metrics["fell"] = fell
        metrics["progress_x"] = pos[0]
        metrics["max_x"] = info["max_x"]
        return State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )
