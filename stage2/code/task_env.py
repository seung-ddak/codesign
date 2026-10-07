"""T8 multi-task 환경 (D63, Q16 a, Q17 b): legged_gym rough terrain 위 LocomotionEnv + jump.

지형 (terrain.py), height scan (17 x 11 점, 몸통 yaw 로 회전, 관측 = clip(z - 0.5 - h, -1, 1) x 5,
noise 0.1), 스폰 (타일 원점 ±1 m), 지형 curriculum 은 legged_gym LeggedRobotCfg /
legged_robot.py 를 따른다. 모든 디자인이 heightfield 하나를 공유한다 (DesignGroupEnv 가 한 벌만 보관).

Jump 는 자체 설계 (legged_gym 에 없음; D36 예외, Q17 b): 재샘플링된 명령이 JUMP_PROB 확률로
jump 가 된다 (vx = vy = yaw = 0, flag 1). jump_air = 네 발 모두 비접촉, jump_height = 서 있는
높이 위로 오른 몸통 높이, jump 중에는 lin_vel_z 페널티 없음.

명령: (vx, vy, yaw rate, heading, jump). 관측 (257): 선속도 3, 각속도 3, 중력 3,
명령 (vx, vy, yaw, jump) 4, 관절 위치 12, 관절 속도 12, 직전 행동 12, 지형 높이 187, 형상 21.

brax auto-reset 아래 curriculum: reset 은 끝난 에피소드를 볼 수 없으므로 level·지형 열은
AutoResetWrapper_preserve_info 에 두고 에피소드가 끝날 때 step() 에서 갱신한다
(legged_gym _update_terrain_curriculum). 다음 에피소드 첫 step 이 몸통을 reset 의 임의 타일에서
curriculum 타일로 옮기므로, 첫 step 은 reset 타일의 height scan 으로 행동한다.
"""

import functools

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

import locomotion_env as L
import multi_design_env as M
import terrain as T
from locomotion_env import LocomotionEnv, State, make_data
from multi_design_env import MultiDesignEnv, structure_key

# ── T8 환경 설정: 로봇 제어·관측·보상·지형은 이 블록만 고친다 ──────────────────
CTRL_DT, SIM_DT = 0.02, 0.005
EPISODE_SECONDS, COMMAND_RESAMPLE_SECONDS, PUSH_INTERVAL_SECONDS = 20., 10., 15.
EPISODE_STEPS = int(np.ceil(EPISODE_SECONDS / CTRL_DT))
RESAMPLE_STEPS = int(COMMAND_RESAMPLE_SECONDS / CTRL_DT)
PUSH_STEPS = int(np.ceil(PUSH_INTERVAL_SECONDS / CTRL_DT))
NUM_LEG_ACTUATORS, ACTION_SCALE = 12, 0.25
CLIP_ACTIONS, CLIP_OBSERVATIONS, SOFT_DOF_POS_LIMIT = 100., 100., 0.9
MAX_PUSH_VEL_XY = 1.
COMMAND_RANGES = {"lin_vel_x": (-1., 1.), "lin_vel_y": (-1., 1.), "heading": (-3.14, 3.14)}
LIN_VEL, ANG_VEL, DOF_POS, DOF_VEL = 2.0, 0.25, 1.0, 0.05
NOISE = {"lin_vel": 0.1, "ang_vel": 0.2, "gravity": 0.05,
         "dof_pos": 0.01, "dof_vel": 1.5}
REWARD_SCALES = {k: v * CTRL_DT for k, v in {
    "tracking_lin_vel": 3.0, "tracking_ang_vel": 1.5, "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05, "torques": -0.0002, "dof_acc": -2.5e-7,
    "feet_air_time": 1.0, "collision": -1., "action_rate": -0.01,
    "dof_pos_limits": -10.}.items()}
ONLY_POSITIVE_REWARDS = False
TRACKING_SIGMA = 0.25
FOOT_NAME = "foot"
PENALIZE_CONTACTS_ON = ("thigh", "calf")
TERMINATE_AFTER_CONTACTS_ON = ("quadruped_body",)
REMOVE_PARTS = ("/arx5/", "/rail/")

TERRAIN_ROWS, TERRAIN_COLS = 10, 20
TERRAIN_LENGTH = TERRAIN_WIDTH = 8.
HORIZONTAL_SCALE, VERTICAL_SCALE, BORDER_SIZE = 0.1, 0.005, 25.
TERRAIN_PROPORTIONS = (0.1, 0.1, 0.35, 0.25, 0.2)
MAX_INIT_TERRAIN_LEVEL = 5
MEASURED_POINTS_X = np.round(np.arange(-0.8, 0.81, 0.1), 2)
MEASURED_POINTS_Y = np.round(np.arange(-0.5, 0.51, 0.1), 2)

HEIGHT_MEASUREMENTS, HEIGHT_NOISE, HEIGHT_OFFSET = 5.0, 0.1, 0.5  # obs_scales, noise_scales, legged_robot.py
JUMP_PROB = 0.2
TASK_REWARD_SCALES = {**REWARD_SCALES, "jump_air": 1.0 * CTRL_DT, "jump_height": 5.0 * CTRL_DT}
PRESERVE = "AutoResetWrapper_preserve_info"

# ── 위 설정을 엔진 모듈에 주입 ───────────────────────────────────────────────
# LocomotionEnv·terrain.py 는 호환 엔진이다. 실행 시 읽는 전역값을 위 설정으로 덮어써서
# T8 튜닝을 이 파일 하나로 한다.
for _name in ("CTRL_DT", "SIM_DT", "EPISODE_STEPS", "RESAMPLE_STEPS", "PUSH_STEPS",
              "NUM_LEG_ACTUATORS", "ACTION_SCALE", "CLIP_ACTIONS", "CLIP_OBSERVATIONS",
              "SOFT_DOF_POS_LIMIT", "MAX_PUSH_VEL_XY", "COMMAND_RANGES", "LIN_VEL",
              "ANG_VEL", "DOF_POS", "DOF_VEL", "NOISE", "REWARD_SCALES",
              "ONLY_POSITIVE_REWARDS", "TRACKING_SIGMA", "FOOT_NAME", "PENALIZE_CONTACTS_ON",
              "TERMINATE_AFTER_CONTACTS_ON", "REMOVE_PARTS"):
    setattr(L, _name, globals()[_name])
M.REMOVE_PARTS = REMOVE_PARTS
T.NUM_ROWS, T.NUM_COLS = TERRAIN_ROWS, TERRAIN_COLS
T.TERRAIN_LENGTH, T.TERRAIN_WIDTH = TERRAIN_LENGTH, TERRAIN_WIDTH
T.HORIZONTAL_SCALE, T.VERTICAL_SCALE, T.BORDER_SIZE = HORIZONTAL_SCALE, VERTICAL_SCALE, BORDER_SIZE
T.TERRAIN_PROPORTIONS = TERRAIN_PROPORTIONS
T.MAX_INIT_TERRAIN_LEVEL = MAX_INIT_TERRAIN_LEVEL
T.MEASURED_POINTS_X, T.MEASURED_POINTS_Y = MEASURED_POINTS_X, MEASURED_POINTS_Y


def make_training_env(groups):
    """train_locomotion_policy.py 가 쓰는 유일한 T8 학습 환경 생성자."""
    return MultiDesignEnv(groups, EPISODE_STEPS, env_cls=TaskEnv)


@functools.cache
def shared_terrain(seed=0):
    hf, origins = T.generate(seed)
    lo, hi = int(hf.min()), int(hf.max())
    # MuJoCo hfield 행 = y, 열 = x. legged_gym height_samples 는 [x, y].
    data = jp.asarray(((hf.T - lo) / (hi - lo)).ravel(), jp.float32)
    return hf, lo, hi, data, jp.asarray(hf, jp.float32) * T.VERTICAL_SCALE, jp.asarray(origins, jp.float32)


class TaskEnv(LocomotionEnv):
    add_plane = False

    def __init__(self, choices, seed=None, **kwargs):
        # 1. 공유 지형 → 부모 생성자 (add_terrain·finish_model 호출)
        kwargs.setdefault("command_ranges", COMMAND_RANGES)
        self.hf, self.hf_lo, self.hf_hi, self.hfield, self.heights, self.origins = shared_terrain()
        super().__init__(choices, seed, **kwargs)
        self.mj_model.hfield_data[:] = np.asarray(self.hfield)  # 렌더링용 CPU 사본
        self.mjx_model = self.mjx_model.replace(hfield_data=self.hfield)

        # 2. jump 높이 기준·height scan 점
        self.stand_height = float(self.home[self.qadr + 2])
        px, py = np.meshgrid(T.MEASURED_POINTS_X, T.MEASURED_POINTS_Y, indexing="ij")
        self.height_points = jp.array(np.stack([px.ravel(), py.ravel()], -1))

        # 3. 관측 스케일·noise (jump flag·지형 높이 포함)
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
        # MJX 에는 cylinder-heightfield 충돌이 없다. 허벅지·종아리 cylinder 는 접촉 감지(충돌 페널티)
        # 용이라 compile 후 같은 반지름·반길이의 capsule 로 바꾼다 (질량·관성은 이미 확정).
        cyl = (m.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER) & (m.geom_contype > 0)
        m.geom_type[cyl] = mujoco.mjtGeom.mjGEOM_CAPSULE
        m.geom_rbound[cyl] = m.geom_size[cyl, 0] + m.geom_size[cyl, 1]

    # ── 에피소드 진행 ────────────────────────────────────────────────────────
    def reset(self, rng):
        # LocomotionEnv.reset + legged_gym _get_env_origins / _reset_root_states
        rng, dof_key, vel_key, cmd_key, noise_key, level_key, col_key, xy_key = jax.random.split(rng, 8)
        # 1. 스폰 타일: 초기 level 0~MAX_INIT, 임의 열, 원점 ±1 m
        level = jax.random.randint(level_key, (), 0, T.MAX_INIT_TERRAIN_LEVEL + 1)
        col = jax.random.randint(col_key, (), 0, T.NUM_COLS)
        origin = self.origins[level, col]
        offset = origin + jp.append(jax.random.uniform(xy_key, (2,), minval=-1., maxval=1.), 0.)
        # 2. 관절·몸통 초기 상태
        qpos = self.home.at[self.qids].set(self.default_dof_pos * jax.random.uniform(
            dof_key, (NUM_LEG_ACTUATORS,), minval=0.5, maxval=1.5))
        qpos = qpos.at[self.qadr:self.qadr + 3].add(offset)
        qvel = jp.zeros(self.mj_model.nv).at[self.vadr:self.vadr + 6].set(
            jax.random.uniform(vel_key, (6,), minval=-0.5, maxval=0.5))
        data = mjx.forward(self.mjx_model, make_data(self.mj_model, qpos=qpos, qvel=qvel,
                                                     ctrl=self.ctrl, impl="jax"))
        # 3. 명령·관측
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
        # 1. 에피소드 첫 step: 몸통을 reset 타일에서 curriculum 타일로 이동
        info, keep = state.info, state.info[PRESERVE]
        target = self.origins[keep["level"], keep["col"]]
        delta = (target - info["spawn_origin"]) * (info["episode_step"] == 0)
        data = state.data.replace(qpos=state.data.qpos.at[self.qadr:self.qadr + 3].add(delta))

        # 2. 부모 step
        state = super().step(state.replace(data=data, info={**info, "spawn_origin": info["spawn_origin"] + delta}),
                             action)

        # 3. 에피소드 종료(종료·time-out) 시 _update_terrain_curriculum
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

    # ── 명령 ─────────────────────────────────────────────────────────────────
    def sample_command(self, rng):
        k1, k2 = jax.random.split(rng)
        c = super().sample_command(k1)
        return jp.where(jax.random.uniform(k2) < JUMP_PROB,
                        jp.array([0., 0., 0., c[3], 1.]), jp.append(c, 0.))

    def heading_to_yaw_rate(self, data, command):
        command = super().heading_to_yaw_rate(data, command)
        return command.at[2].multiply(1. - command[4])

    # ── 보상 ─────────────────────────────────────────────────────────────────
    def reward(self, data, action, command, info):
        reward, done, terms, info = super().reward(data, action, command, info)
        # jump 중에는 lin_vel_z 페널티 제외
        jump = command[4]
        reward -= terms["lin_vel_z"] * jump
        # jump 보상: 네 발 비접촉, 서 있는 높이 위로 오른 몸통 높이
        base = data.qpos[self.qadr:self.qadr + 3]
        rise = jp.clip(base[2] - self.terrain_height(base[:2]) - self.stand_height, 0.)
        extra = {"jump_air": jp.all(~info["last_contacts"]) * jump * TASK_REWARD_SCALES["jump_air"],
                 "jump_height": rise * jump * TASK_REWARD_SCALES["jump_height"]}
        terms = {**terms, "lin_vel_z": terms["lin_vel_z"] * (1. - jump), **extra}
        return reward + sum(extra.values()), done, terms, info

    # ── 관측 ─────────────────────────────────────────────────────────────────
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

    # ── 지형 높이 ────────────────────────────────────────────────────────────
    def measured_heights(self, data):
        """몸통 yaw 로 회전한 height scan 점의 지형 높이."""
        q = data.qpos[self.qadr + 3:self.qadr + 7]
        yaw = jp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        c, s = jp.cos(yaw), jp.sin(yaw)
        pts = self.height_points @ jp.array([[c, s], [-s, c]])  # quat_apply_yaw
        return self.terrain_height(pts + data.qpos[self.qadr:self.qadr + 2])

    def terrain_height(self, xy):
        """legged_robot._get_heights: 이웃 3 샘플의 최솟값, world xy (..., 2)."""
        p = ((xy + T.BORDER_SIZE) / T.HORIZONTAL_SCALE).astype(jp.int32)
        px = jp.clip(p[..., 0], 0, self.heights.shape[0] - 2)
        py = jp.clip(p[..., 1], 0, self.heights.shape[1] - 2)
        h = self.heights
        return jp.minimum(jp.minimum(h[px, py], h[px + 1, py]), h[px, py + 1])
