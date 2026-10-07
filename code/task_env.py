"""Expert 정책용 환경. 관측 앞 69 개는 T5 와 정확히 같고 뒤에 새 입력을 붙인다:
jump flag 1 + 지형 height scan 187 = 257.

phase: flat (FlatTaskEnv) · jump (FlatJumpEnv) · universal (TaskEnv)
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

# ── stage2 환경 불러오기 (import 시 locomotion_env 설정값을 T8 설정으로 덮어씀) ──
_LEGACY = Path(__file__).resolve().parents[1] / "stage2" / "code"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

_spec = importlib.util.spec_from_file_location("_codesign_stage2_task_env", _LEGACY / "task_env.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

ACTION_SCALE = _base.ACTION_SCALE
COMMAND_RANGES = _base.COMMAND_RANGES
EPISODE_STEPS = _base.EPISODE_STEPS
ONLY_POSITIVE_REWARDS = _base.ONLY_POSITIVE_REWARDS
TASK_REWARD_SCALES = _base.TASK_REWARD_SCALES
TERMINATE_AFTER_CONTACTS_ON = _base.TERMINATE_AFTER_CONTACTS_ON
NUM_LEG_ACTUATORS = _base.NUM_LEG_ACTUATORS
MultiDesignEnv = _base.MultiDesignEnv
structure_key = _base.structure_key

# ── 관측 크기 ────────────────────────────────────────────────────────────────
T5_OBSERVATION_SIZE = 69
OBSERVATION_SIZE = 257

# ── jump 설정 ────────────────────────────────────────────────────────────────
# 스케줄 (50 Hz): 안정 45 → [점프 윈도우 80 → 대기 10] 반복. 속도 명령은 항상 0.
JUMP_SETTLE_STEPS = 45
JUMP_WINDOW_STEPS = 80
JUMP_REST_STEPS = 10
# 점프 인정 = 몸통 z >= TARGET_Z(절대좌표) 이면서 네 발 모두 지면에서 MIN_FOOT_CLEARANCE 이상
# 성공 = 점프 인정 → 발 착지 → 네 발 접지 + 자세 유지 연속 BALANCE_STEPS
JUMP_TARGET_Z = 0.6  # m, 절대좌표. 체공 높이 보상의 중심도 이 값
JUMP_MIN_FOOT_CLEARANCE = 0.1  # m, 발 구 아래끝 ~ 지면(z=0)
JUMP_MIN_AIR_STEPS = 6  # 네 발 비접촉 연속 step 이 이 이상일 때만 air·height 보상 (noise 들썩임 ≤5 step 제외)
JUMP_BALANCE_STEPS = 10  # 0.2 s
JUMP_MAX_TILT = 0.26  # |중력 xy| 상한 = sin(15°)
JUMP_LANDING_REWARD = 20.  # train3 후 1 → 20 (높이 보상 대비 착지 가치 상향)
JUMP_TOUCHDOWN_REWARD = 1.0  # 착지 순간 1회: exp(-|중력 xy|^2 / sigma)
JUMP_TOUCHDOWN_SIGMA = 0.1  # 15° 에서 약 0.5
JUMP_REJUMP_PENALTY = 5.0 * _base.CTRL_DT  # 착지 후 같은 윈도우에서 다시 체공(flight)하면 step 당 벌점
JUMP_UP_VEL_SCALE = 1.0  # 윈도우 최고 위쪽 속도(world vz) 1회 보상 (1회성이라 dt 안 곱함). 기록 갱신분만 지급 → 윈도우 합 = 최고 vz x scale
# Atanassov et al. 2024 (go1_upwards) 이식. Go1 값 → 우리 로봇 비율로 환산
JUMP_SQUAT_Z = 0.19  # m, 도약 전 웅크림 목표 = 서 있는 높이 0.30 x 0.625 (Go1 0.20 / 0.32)
JUMP_SQUAT_SIGMA = 0.001  # exp(-(z - SQUAT_Z)^2 / sigma)
JUMP_SQUAT_SCALE = 3.0 * _base.CTRL_DT  # Go1: base_height_stance 5 x 0.6
JUMP_FLIGHT_SIGMA = 0.1  # 체공 높이 보상 exp(-(z - TARGET_Z)^2 / sigma)
JUMP_FLIGHT_SCALE = 80. * _base.CTRL_DT  # Go1: base_height_flight 80
JUMP_LAUNCH_PROB = 0.3  # 학습 시(domain_rand) 윈도우 시작에 몸통을 위로 띄워 착지를 미리 경험시킬 확률 (Go1 0.7, smoke1 후 0.3)
JUMP_LAUNCH_DZ = (0., 0.3)  # m, 몸통 높이 추가 (Go1 0~0.7)
JUMP_LAUNCH_VZ = (0., 3.)  # m/s, 위쪽 속도 (Go1 0~3)
# 에피소드 중 flag = 1 비율. 학습 시작 시 flag 관측의 정규화 통계로 쓴다 (train_policy.py).
_k = np.arange(1, EPISODE_STEPS + 1) - JUMP_SETTLE_STEPS
JUMP_FLAG_MEAN = float(np.mean((_k >= 0) & (_k % (JUMP_WINDOW_STEPS + JUMP_REST_STEPS) < JUMP_WINDOW_STEPS)))
JUMP_DIAGNOSTICS = (
    "jump_trigger_fraction_per_step", "jump_episode_fraction",
    "raw_jump_height_reward_per_step", "scaled_jump_height_reward_per_step",
    "raw_jump_air_reward_per_step", "scaled_jump_air_reward_per_step",
    "jump_takeoff_success", "jump_landing_success", "jump_launched",
    "tracking_lin_vel_reward_per_step", "tracking_ang_vel_reward_per_step",
)


def make_training_env(groups, phase="flat"):
    env_cls = {"flat": FlatTaskEnv, "jump": FlatJumpEnv, "universal": TaskEnv}[phase]
    return MultiDesignEnv(groups, EPISODE_STEPS, env_cls=env_cls)


class FlatTaskEnv(_base.LocomotionEnv):
    """T5 환경. 새 관측 자리는 0 으로 채운다."""

    def observation(self, data, command, action, rng):
        old = super().observation(data, command, action, rng)
        return jp.pad(old, (0, OBSERVATION_SIZE - T5_OBSERVATION_SIZE))


class FlatJumpEnv(_base.LocomotionEnv):
    """Jump expert: 고정 스케줄로 제자리 점프. 속도 명령은 항상 0."""

    def __init__(self, choices, seed=None, **kwargs):
        # 스케줄이 명령을 정한다: 재샘플링·heading→yaw rate 제어 없음
        super().__init__(choices, seed, **{**kwargs, "hold_command": True})
        # 발 구 geom (발 body 당 1개). 반지름은 디자인마다 달라 reward 에서 mjx_model 로 읽는다.
        m = self.mj_model
        self.foot_geoms = np.flatnonzero(np.isin(m.geom_bodyid, self.feet)
                                         & (m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE))
        if len(self.foot_geoms) != len(self.feet):
            raise ValueError(f"Expected one sphere per foot, got {len(self.foot_geoms)}")
        self.leg_bodies = jp.array(self.penalised)  # 허벅지·종아리: 바닥 접촉 시 체공 아님

    # ── 에피소드 진행 ────────────────────────────────────────────────────────
    def reset(self, rng):
        state = super().reset(rng)
        zero = jp.zeros(())
        metrics = {**state.metrics, "jump_air": zero, "jump_height": zero,
                   "jump_up_vel": zero, "jump_squat": zero, "jump_landing": zero, "jump_touchdown": zero, "jump_rejump": zero,
                   **{name: zero for name in JUMP_DIAGNOSTICS}}
        no = jp.array(False)
        return state.replace(metrics=metrics, info={
            **state.info, "jump_seen": no,
            "air_steps": jp.int32(0), "takeoff": no, "landed": no,
            "upright_steps": jp.int32(0), "success": no, "best_vz": jp.zeros(()),
            "launch_now": no, "launched": no})

    def step(self, state, action):
        # 1. 스케줄상 점프 윈도우면 jump flag = 1
        k = state.info["episode_step"] + 1 - JUMP_SETTLE_STEPS
        window = (k >= 0) & (k % (JUMP_WINDOW_STEPS + JUMP_REST_STEPS) < JUMP_WINDOW_STEPS)
        command = state.info["command"].at[4].set(window.astype(jp.float32))

        # 2. 공중 스폰 (학습 시만): 윈도우 시작에 확률 LAUNCH_PROB 로 몸통을 위로 띄움
        data, launch = state.data, jp.array(False)
        if self.domain_rand:
            start = (k >= 0) & (k % (JUMP_WINDOW_STEPS + JUMP_REST_STEPS) == 0)
            k_p, k_dz, k_vz = jax.random.split(jax.random.fold_in(state.info["rng"], 2), 3)
            launch = start & (jax.random.uniform(k_p) < JUMP_LAUNCH_PROB)
            dz = jax.random.uniform(k_dz, minval=JUMP_LAUNCH_DZ[0], maxval=JUMP_LAUNCH_DZ[1])
            vz = jax.random.uniform(k_vz, minval=JUMP_LAUNCH_VZ[0], maxval=JUMP_LAUNCH_VZ[1])
            data = data.replace(qpos=data.qpos.at[self.qadr + 2].add(dz * launch),
                                qvel=data.qvel.at[self.vadr + 2].set(
                                    jp.where(launch, vz, data.qvel[self.vadr + 2])))

        info = {**state.info, "command": command, "launch_now": launch}
        return super().step(state.replace(data=data, info=info), action)

    def sample_command(self, rng):
        return jp.zeros(5)  # [vx, vy, yaw rate, heading, jump]

    # ── 보상 ─────────────────────────────────────────────────────────────────
    def reward(self, data, action, command, info):
        # 1. 윈도우 시작이면 점프 상태 초기화
        k = info["episode_step"] + 1 - JUMP_SETTLE_STEPS
        start = (k >= 0) & (k % (JUMP_WINDOW_STEPS + JUMP_REST_STEPS) == 0)
        z = data.qpos[self.qadr + 2]
        launched = jp.where(start, info["launch_now"], info["launched"])  # 이번 윈도우가 공중 스폰인지
        air_steps, upright_steps = (jp.where(start, 0, info[n]) for n in ("air_steps", "upright_steps"))
        best_vz = jp.where(start, 0., info["best_vz"])
        takeoff, landed, success = (info[n] & ~start for n in ("takeoff", "landed", "success"))

        # 2. 기본 보상 (last_contacts·done 갱신). 점프 중엔 수직속도 페널티 제외
        reward, done, terms, info = super().reward(data, action, command, info)
        jump = command[4]
        reward -= terms["lin_vel_z"] * jump

        # 3. 현재 상태. 체공 = 발·허벅지·종아리 모두 바닥과 비접촉 (무릎 짚기 제외)
        legs_touch = jp.any(jp.linalg.norm(self.contact_forces(data, self.leg_bodies), axis=-1) > 1.)
        airborne = jp.all(~info["last_contacts"]) & ~legs_touch
        stance = jp.all(info["last_contacts"])
        vz = data.qvel[self.vadr + 2]  # world 좌표계 수직 속도
        clearance = data.geom_xpos[self.foot_geoms, 2] - self.mjx_model.geom_size[self.foot_geoms, 0]
        tilt = jp.linalg.norm(self.base_state(data)[2][:2])  # |중력 xy| = sin(roll·pitch 기울기)
        upright = tilt < JUMP_MAX_TILT

        # 4. 단계 판정: 이륙 → 착지 → 자세 유지 → 성공 (윈도우당 1회)
        air_steps = jp.where(airborne, air_steps + 1, 0)
        flight = air_steps >= JUMP_MIN_AIR_STEPS
        new_takeoff = ((jump > 0) & ~takeoff & airborne & (z >= JUMP_TARGET_Z)
                       & jp.all(clearance >= JUMP_MIN_FOOT_CLEARANCE))
        takeoff = takeoff | new_takeoff
        touchdown = takeoff & ~landed & ~airborne  # 점프 인정 후 첫 접지 순간
        landed = landed | touchdown
        upright_steps = jp.where(landed & stance & upright, upright_steps + 1, 0)
        new_success = landed & ~success & (upright_steps >= JUMP_BALANCE_STEPS)
        success = success | new_success

        # 5. 점프 보상 (점프 인정 후 착지하면 그 윈도우는 끝)
        #    웅크림: 도약 전 접지 중 / 위쪽 속도: 윈도우 최고 vz 기록 갱신분 (공중 스폰 윈도우 제외)
        #    체공·높이: 연속 체공(flight) 중에만. 높이는 공중 스폰 윈도우 제외 (스폰 궤적은 정책이 못 바꿈)
        #    착지 자세: 첫 접지 순간 roll·pitch 기울기로 1회 / 재점프: 착지 후 다시 연속 체공하면 벌점
        raw_squat = jp.exp(-jp.square(z - JUMP_SQUAT_Z) / JUMP_SQUAT_SIGMA) * jump * ~airborne * ~takeoff
        new_best_vz = jp.where((jump > 0) & ~landed & ~launched, jp.maximum(best_vz, vz), best_vz)
        raw_up_vel = new_best_vz - best_vz
        active = jump * flight * ~landed
        raw_air = active
        raw_height = jp.exp(-jp.square(z - JUMP_TARGET_Z) / JUMP_FLIGHT_SIGMA) * active * ~launched
        extra = {
            "jump_squat": raw_squat * JUMP_SQUAT_SCALE,
            "jump_up_vel": raw_up_vel * JUMP_UP_VEL_SCALE,
            "jump_air": raw_air * TASK_REWARD_SCALES["jump_air"],
            "jump_height": raw_height * JUMP_FLIGHT_SCALE,
            "jump_touchdown": touchdown * jp.exp(-jp.square(tilt) / JUMP_TOUCHDOWN_SIGMA) * JUMP_TOUCHDOWN_REWARD,
            "jump_landing": new_success * JUMP_LANDING_REWARD,
            "jump_rejump": -(jump * flight * landed) * JUMP_REJUMP_PENALTY,
        }
        task_reward = sum(extra.values())

        # 6. 로그 항목: 점프 중 속도 추종 항 제외 + 진단값
        terms = {**terms, "lin_vel_z": terms["lin_vel_z"] * (1. - jump),
                 "tracking_lin_vel": terms["tracking_lin_vel"] * (1. - jump),
                 "tracking_ang_vel": terms["tracking_ang_vel"] * (1. - jump), **extra,
                 "jump_trigger_fraction_per_step": jump,
                 "jump_episode_fraction": jump * ~info["jump_seen"],
                 "raw_jump_height_reward_per_step": raw_height,
                 "scaled_jump_height_reward_per_step": extra["jump_height"],
                 "raw_jump_air_reward_per_step": raw_air,
                 "scaled_jump_air_reward_per_step": extra["jump_air"],
                 "jump_takeoff_success": (new_takeoff & ~launched).astype(jp.float32),  # 공중 스폰 제외
                 "jump_landing_success": (new_success & ~launched).astype(jp.float32),
                 "jump_launched": (start & launched).astype(jp.float32),
                 "tracking_lin_vel_reward_per_step": terms["tracking_lin_vel"],
                 "tracking_ang_vel_reward_per_step": terms["tracking_ang_vel"]}

        # 7. 다음 step 으로 상태 전달
        info = {**info, "jump_seen": info["jump_seen"] | (jump > 0),
                "air_steps": air_steps, "takeoff": takeoff, "landed": landed,
                "upright_steps": upright_steps, "success": success, "best_vz": new_best_vz,
                "launched": launched}
        return reward + task_reward, done, terms, info

    # ── 관측 ─────────────────────────────────────────────────────────────────
    def observation(self, data, command, action, rng):
        old = super().observation(data, command, action, rng)
        return jp.concatenate([old, command[4:5], jp.zeros(187)])


class TaskEnv(_base.TaskEnv):
    """jump·계단 환경. 관측 앞부분이 T5 벡터 전체."""

    def __init__(self, choices, seed=None, **kwargs):
        super().__init__(choices, seed, **kwargs)
        n, nh = NUM_LEG_ACTUATORS, self.height_points.shape[0]
        self.noise_vec = jp.concatenate([
            jp.full(3, _base.NOISE["lin_vel"] * _base.LIN_VEL),
            jp.full(3, _base.NOISE["ang_vel"] * _base.ANG_VEL),
            jp.full(3, _base.NOISE["gravity"]), jp.zeros(3),
            jp.full(n, _base.NOISE["dof_pos"] * _base.DOF_POS),
            jp.full(n, _base.NOISE["dof_vel"] * _base.DOF_VEL),
            jp.zeros(n), jp.zeros(self.morphology.size), jp.zeros(1),
            jp.full(nh, _base.HEIGHT_NOISE * _base.HEIGHT_MEASUREMENTS),
        ])
        if self.noise_vec.size != OBSERVATION_SIZE:
            raise ValueError(f"Expected {OBSERVATION_SIZE} observations, got {self.noise_vec.size}")

    def observation(self, data, command, action, rng):
        lin_vel, ang_vel, gravity = self.base_state(data)
        heights = jp.clip(
            data.qpos[self.qadr + 2] - _base.HEIGHT_OFFSET - self.measured_heights(data), -1., 1.)
        obs = jp.concatenate([
            # T5 순서 그대로. 이 앞부분 사이에 새 값을 끼워 넣지 않는다.
            lin_vel * _base.LIN_VEL, ang_vel * _base.ANG_VEL, gravity,
            command[:3] * jp.array([_base.LIN_VEL, _base.LIN_VEL, _base.ANG_VEL]),
            (data.qpos[self.qids] - self.default_dof_pos) * _base.DOF_POS,
            data.qvel[self.vids] * _base.DOF_VEL, action, self.morphology,
            # 새 입력
            command[4:5], heights * _base.HEIGHT_MEASUREMENTS,
        ])
        if self.add_noise:
            obs += (2 * jax.random.uniform(rng, obs.shape) - 1) * self.noise_vec
        return jp.clip(obs, -_base.CLIP_OBSERVATIONS, _base.CLIP_OBSERVATIONS)
