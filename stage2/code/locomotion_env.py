"""Stage 2 평지 보행 환경 (T3 env, T4 reward).

관측·명령·리셋·push·보상은 legged_gym `LeggedRobotCfg` (base, A1 override 없음; upstream
`envs/base/legged_robot*.py`) 를 따른다. 로봇·액추에이터·접촉 형상은 TT `umi_on_legs_plus_plus`
+ MJX 그대로 (D35, D37), 접촉 link 이름만 TT body 에 매핑한다 (D43).
time-out 은 brax EpisodeWrapper (episode_length = EPISODE_STEPS), 마찰 DR 은
multi_design_env.DesignGroupEnv 가 env slot 마다 건다.

주의: stage2 task_env.py 가 import 시 아래 설정 전역값을 덮어쓴다. 메서드는 실행 시점에
전역값을 읽으므로 이름을 바꾸거나 import 시점에 값을 묶어 두지 않는다.
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

# ── 설정 ─────────────────────────────────────────────────────────────────────
# 시뮬레이션 (TT config/env/mjx_umi_on_legs.yaml)
CTRL_DT, SIM_DT, IMPRATIO, ITERATIONS, LS_ITERATIONS = 0.02, 0.005, 1, 5, 7
NUM_LEG_ACTUATORS = 12  # 128 개 bank choice 모두 다리 액추에이터 12 개

# 에피소드·명령·push (LeggedRobotCfg)
EPISODE_STEPS = int(np.ceil(20. / CTRL_DT))             # env.episode_length_s
RESAMPLE_STEPS = int(10. / CTRL_DT)                     # commands.resampling_time
PUSH_STEPS = int(np.ceil(15. / CTRL_DT))                # domain_rand.push_interval_s
MAX_PUSH_VEL_XY = 1.
COMMAND_RANGES = {"lin_vel_x": (-1., 1.), "lin_vel_y": (-1., 1.),
                  "heading": (-3.14, 3.14)}             # heading_command = True

# 행동·관측 (A1RoughCfg override: action_scale, torques, dof_pos_limits, soft_dof_pos_limit —
# 2026-09-27 사용자 결정, D43 대체. A1 고유값(초기 자세, 기본 각도, kp/kd)은 가져오지 않는다)
ACTION_SCALE, CLIP_ACTIONS, CLIP_OBSERVATIONS = 0.25, 100., 100.
SOFT_DOF_POS_LIMIT = 0.9
LIN_VEL, ANG_VEL, DOF_POS, DOF_VEL = 2.0, 0.25, 1.0, 0.05   # normalization.obs_scales
NOISE = {"lin_vel": 0.1, "ang_vel": 0.2, "gravity": 0.05,
         "dof_pos": 0.01, "dof_vel": 1.5}               # noise_scales, noise_level 1.0

# 보상: scale 0 항 제거 후 dt 곱 (_prepare_reward_function). termination(-0.0) 은 제외.
# tracking_lin_vel 1.0 → 5.0 (D45) → 3.0 (D47): 정지 상태와 같은 추종률(~19%) → x5 는 넘어짐 증가.
# tracking_ang_vel 0.5 → 1.5 (D50): yaw 추종 20-36% 에 그침.
REWARD_SCALES = {k: v * CTRL_DT for k, v in {
    "tracking_lin_vel": 3.0, "tracking_ang_vel": 1.5, "lin_vel_z": -2.0, "ang_vel_xy": -0.05,
    "torques": -0.0002, "dof_acc": -2.5e-7, "feet_air_time": 1.0, "collision": -1.,
    "action_rate": -0.01, "dof_pos_limits": -10.}.items()}
# only_positive_rewards: True 는 거의 매 step 0 으로 잘려 std 발산 (t5_a1_1000it).
# False + action_scale 0.25 + adaptive KL 로 재시도 (2026-09-27 사용자 결정).
TRACKING_SIGMA, ONLY_POSITIVE_REWARDS = 0.25, False

# 접촉 body (asset.* 이름 일부 → TT body). 몸통 geom 은 `quadruped_body` 에 있다
# (TT `base` 는 geom 없는 free-joint body). A1/ANYmal 처럼 몸통 접촉 = 종료, 허벅지·종아리 = 페널티
# (2026-09-26 사용자 결정). 종아리 geom 전부(rod, 무릎 calf_motor) 포함, 발은 별도 body.
FOOT_NAME = "foot"
PENALIZE_CONTACTS_ON = ("thigh", "calf")
TERMINATE_AFTER_CONTACTS_ON = ("quadruped_body",)

# 제거 부품 (D46/D47): arx5 팔+그리퍼(4.05 kg, 로봇의 ~22%)와 rail(bit 5, 1.18 kg).
# 페이로드 없이 먼저 보행을 배운다. 이로써 bit 5 는 로봇을 바꾸지 않는다.
REMOVE_PARTS = ("/arx5/", "/rail/")


# ── 모델 생성 ─────────────────────────────────────────────────────────────────
def compile_scene(scene):
    """TT scene 을 MjModel 로 컴파일하되 REMOVE_PARTS 하위 트리·액추에이터·key 항목은 뺀다."""
    m = mjcf.Physics.from_mjcf_model(scene).model.ptr
    if not REMOVE_PARTS:
        return m
    arm = lambda name: any(p in name for p in REMOVE_PARTS)
    arm_q = [m.jnt_qposadr[j] for j in range(m.njnt) if arm(m.joint(j).name)]
    arm_u = [i for i in range(m.nu) if arm(m.actuator(i).name)]
    top = [m.body(b).name for b in range(m.nbody)
           if arm(m.body(b).name) and not arm(m.body(m.body_parentid[b]).name)]
    spec = mujoco.MjSpec.from_string(scene.to_xml_string(), assets=scene.get_assets())
    # MjSpec 은 delete 마다 key 크기를 검사하므로 key 를 지웠다가 잘라서 다시 넣는다.
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


# ── 환경 ─────────────────────────────────────────────────────────────────────
class LocomotionEnv(MjxEnv):
    action_size = NUM_LEG_ACTUATORS
    add_plane = True  # TaskEnv 는 평면 대신 heightfield 를 쓴다

    def __init__(self, choices, seed=None, command_ranges=COMMAND_RANGES, add_noise=True,
                 hold_command=False, domain_rand=True, uniforms=None):
        super().__init__(ctrl_dt=CTRL_DT, sim_dt=SIM_DT)

        # 1. 입력 검사·옵션
        if len(choices) != 8 or any(c not in (0, 1) for c in choices):
            raise ValueError("Expected an eight-bit TT quadruped choice")
        self.command_ranges = command_ranges
        # 평가 전용: 호출자가 넣은 command 유지 (재샘플링·heading→yaw rate 제어 없음)
        self.hold_command = hold_command
        # 평가 전용 (D52, legged_gym play.py): False 면 관측 noise·push·마찰 DR 끔
        self.domain_rand = domain_rand
        self.add_noise = add_noise and domain_rand

        # 2. 형상 파라미터 (13 개 uniform: seed 로 뽑거나 직접 지정)
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

        # 3. scene → MjModel
        robot = umi_on_legs_plus_plus_from_params(
            choices=copy.copy(choices), uniforms=uniforms.tolist())
        scene = set_up_default_scene(robot, add_plane=self.add_plane, plane_z_pos=0.0)
        self.add_terrain(scene)
        self.mj_model = m = compile_scene(scene)
        m.opt.timestep, m.opt.impratio = SIM_DT, IMPRATIO
        m.opt.iterations, m.opt.ls_iterations = ITERATIONS, LS_ITERATIONS

        # 4. 접촉 body 찾기
        bodies = [m.body(i).name for i in range(m.nbody)]
        find = lambda names: np.array([i for i, b in enumerate(bodies)
                                       if any(n in b for n in names)], dtype=int)
        self.feet = find([FOOT_NAME])
        self.penalised = find(PENALIZE_CONTACTS_ON)
        self.terminate = find(TERMINATE_AFTER_CONTACTS_ON)
        if len(self.feet) != 4 or len(self.terminate) != 1 or len(self.penalised) != 8:
            raise ValueError(f"Expected 4 feet, 1 trunk, 4 thighs + 4 calves, got {len(self.feet)}, "
                             f"{len(self.terminate)}, {len(self.penalised)}")
        self.contact_bodies = jp.array(np.r_[self.feet, self.penalised, self.terminate])
        self.geom_body = jp.array(m.geom_bodyid)

        # 5. 충돌 설정
        # TT preprocess_mj_model: 구(발)와 바닥만 충돌 (bit 1)
        ground = np.isin(m.geom_type, (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD))
        collide = (m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) | ground
        m.geom_contype[:] = m.geom_conaffinity[:] = collide
        # 몸통·허벅지·종아리는 접촉 감지용으로 바닥과만 충돌 (bit 2). 구끼리의 접촉은 그대로.
        contact_only = np.isin(m.geom_bodyid, np.r_[self.terminate, self.penalised])
        m.geom_contype[contact_only], m.geom_conaffinity[contact_only] = 2, 0
        m.geom_conaffinity[ground] = 3

        # 6. 몸통 free joint·다리 관절 주소
        free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Expected one floating base")
        self.qadr, self.vadr = int(m.jnt_qposadr[free[0]]), int(m.jnt_dofadr[free[0]])
        # 팔(arx5)·rail 액추에이터는 home ctrl 고정 (D10)
        self.leg = np.array([i for i in range(m.nu) if "/arx5/" not in m.actuator(i).name
                             and "/rail/" not in m.actuator(i).name])
        if len(self.leg) != NUM_LEG_ACTUATORS:
            raise ValueError(f"Expected {NUM_LEG_ACTUATORS} leg actuators, got {len(self.leg)}")
        joints = m.actuator_trnid[self.leg, 0]
        self.qids, self.vids = m.jnt_qposadr[joints], m.jnt_dofadr[joints]

        # 7. 관절 한계·기본 자세
        # legged_gym 은 P 목표를 자르지 않는다 (토크만 제한). TT ctrlrange = 관절 범위라 관절 한계는 유지.
        m.actuator_ctrllimited[self.leg] = 0
        lo, hi = m.jnt_range[joints, 0], m.jnt_range[joints, 1]  # _process_dof_props soft limits
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_DOF_POS_LIMIT
        self.dof_pos_limits = jp.array(np.stack([mid - half, mid + half]))
        self.home = jp.array(m.key("home").qpos)
        self.ctrl = jp.array(m.key("home").ctrl)
        self.default_dof_pos = self.home[self.qids]

        # 8. 관측 스케일·noise
        self.morphology = jp.array(np.r_[choices, uniforms], dtype=jp.float32)
        self.commands_scale = jp.array([LIN_VEL, LIN_VEL, ANG_VEL])
        n = NUM_LEG_ACTUATORS
        self.noise_vec = jp.concatenate([
            jp.full(3, NOISE["lin_vel"] * LIN_VEL), jp.full(3, NOISE["ang_vel"] * ANG_VEL),
            jp.full(3, NOISE["gravity"]), jp.zeros(3),
            jp.full(n, NOISE["dof_pos"] * DOF_POS), jp.full(n, NOISE["dof_vel"] * DOF_VEL),
            jp.zeros(n), jp.zeros(self.morphology.size)])

        # 9. 하위 클래스 후처리 → MJX 모델
        self.finish_model(m)
        self.mjx_model = mjx.put_model(m, impl="jax")

    def add_terrain(self, scene):
        """하위 클래스용: compile 전에 지형 추가."""

    def finish_model(self, m):
        """하위 클래스용: MJX 변환 직전 MjModel 수정."""

    # ── 에피소드 진행 ────────────────────────────────────────────────────────
    def reset(self, rng):
        rng, dof_key, vel_key, cmd_key, noise_key = jax.random.split(rng, 5)
        # _reset_dofs: 기본 각도 x U(0.5, 1.5), 속도 0. _reset_root_states: 몸통 속도 U(-0.5, 0.5)
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
        # 1. 행동 → 물리
        action = jp.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
        targets = self.ctrl.at[self.leg].add(action * ACTION_SCALE)
        data = step(self.mjx_model, state.data, targets, self.n_substeps)
        t = state.info["episode_step"] + 1
        rng, cmd_key, noise_key, push_key = jax.random.split(state.info["rng"], 4)

        # 2. 명령 갱신
        command = state.info["command"]
        if not self.hold_command:
            command = jp.where(t % RESAMPLE_STEPS == 0, self.sample_command(cmd_key), command)
            command = self.heading_to_yaw_rate(data, command)

        # 3. 보상·종료 → 관측
        reward, done, terms, info = self.reward(data, action, command, state.info)
        obs = self.observation(data, command, action, noise_key)

        # 4. push (관측 뒤, post_physics_step 순서)
        base_xy = slice(self.vadr, self.vadr + 2)
        push = jax.random.uniform(push_key, (2,), minval=-MAX_PUSH_VEL_XY, maxval=MAX_PUSH_VEL_XY)
        if self.domain_rand:
            data = data.replace(qvel=data.qvel.at[base_xy].set(
                jp.where(t % PUSH_STEPS == 0, push, data.qvel[base_xy])))
        return state.replace(data=data, obs=obs, reward=reward, done=done, metrics=terms,
                             info={**info, "rng": rng, "command": command, "episode_step": t})

    # ── 명령 ─────────────────────────────────────────────────────────────────
    def sample_command(self, rng):
        """_resample_commands: [vx, vy, yaw rate (heading 에서 계산), heading]."""
        keys = jax.random.split(rng, 3)
        vx, vy, heading = (jax.random.uniform(k, minval=lo, maxval=hi)
                           for k, (lo, hi) in zip(keys, (self.command_ranges["lin_vel_x"],
                                                         self.command_ranges["lin_vel_y"],
                                                         self.command_ranges["heading"])))
        xy = jp.array([vx, vy])
        xy = xy * (jp.linalg.norm(xy) > 0.2)  # 작은 명령은 0
        return jp.array([xy[0], xy[1], 0., heading])

    def heading_to_yaw_rate(self, data, command):
        forward = quat_rotate(data.qpos[self.qadr + 3:self.qadr + 7], jp.array([1., 0., 0.]))
        heading = jp.arctan2(forward[1], forward[0])
        return command.at[2].set(jp.clip(0.5 * wrap_to_pi(command[3] - heading), -1., 1.))

    # ── 보상·종료 ─────────────────────────────────────────────────────────────
    def reward(self, data, action, command, info):
        """compute_reward + check_termination → (reward, done, scale 곱한 항, info)."""
        # 1. 현재 상태
        lin_vel, ang_vel, _ = self.base_state(data)
        dof_pos, dof_vel = data.qpos[self.qids], data.qvel[self.vids]
        forces = self.contact_forces(data, self.contact_bodies)
        nf, np_ = len(self.feet), len(self.penalised)
        feet, penalised, terminate = forces[:nf], forces[nf:nf + np_], forces[nf + np_:]

        # 2. 발 접촉 (수직력 > 1 N) · 체공 시간 (_reward_feet_air_time)
        contact = feet[:, 2] > 1.
        contact_filt = contact | info["last_contacts"]
        first_contact = (info["feet_air_time"] > 0.) & contact_filt
        air_time = info["feet_air_time"] + self.dt

        # 3. 보상 항
        terms = {
            "tracking_lin_vel": jp.exp(-jp.sum(jp.square(command[:2] - lin_vel[:2])) / TRACKING_SIGMA),
            "tracking_ang_vel": jp.exp(-jp.square(command[2] - ang_vel[2]) / TRACKING_SIGMA),
            "lin_vel_z": jp.square(lin_vel[2]),
            "ang_vel_xy": jp.sum(jp.square(ang_vel[:2])),
            "torques": jp.sum(jp.square(data.actuator_force[self.leg])),
            "dof_acc": jp.sum(jp.square((info["last_dof_vel"] - dof_vel) / self.dt)),
            "feet_air_time": jp.sum((air_time - 0.5) * first_contact)
                             * (jp.linalg.norm(command[:2]) > 0.1),
            # 빈 배열 reduction 이 XLA 에서 step 전체를 14 배 느리게 해서 guard
            "collision": jp.sum(jp.linalg.norm(penalised, axis=-1) > 0.1) if np_ else jp.zeros(()),
            "action_rate": jp.sum(jp.square(info["last_actions"] - action)),
            "dof_pos_limits": jp.sum(jp.clip(self.dof_pos_limits[0] - dof_pos, 0.) +
                                     jp.clip(dof_pos - self.dof_pos_limits[1], 0.)),
        }
        terms = {k: v * REWARD_SCALES[k] for k, v in terms.items()}
        reward = sum(terms.values())
        if ONLY_POSITIVE_REWARDS:
            reward = jp.maximum(reward, 0.)

        # 4. 종료: 몸통 접촉력 > 1 N
        done = jp.any(jp.linalg.norm(terminate, axis=-1) > 1.)

        # 5. 다음 step 으로 상태 전달
        info = {**info, "feet_air_time": air_time * ~contact_filt, "last_contacts": contact,
                "last_dof_vel": dof_vel, "last_actions": action}
        return reward, done.astype(jp.float32), terms, info

    # ── 관측 ─────────────────────────────────────────────────────────────────
    def observation(self, data, command, action, rng):
        """compute_observations (blind, 평지) + 조건부 pi_theta 용 형상 벡터."""
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

    # ── 상태 읽기 ────────────────────────────────────────────────────────────
    def base_state(self, data):
        """몸통 좌표계의 선속도, 각속도, 투영 중력."""
        quat = data.qpos[self.qadr + 3:self.qadr + 7]
        qvel = data.qvel[self.vadr:self.vadr + 6]
        return (quat_rotate(quat, qvel[:3], inverse=True),  # world → 몸통 좌표계
                qvel[3:],  # MuJoCo free joint 각속도는 이미 몸통 좌표계
                quat_rotate(quat, jp.array([0., 0., -1.]), inverse=True))

    def contact_forces(self, data, bodies):
        """Isaac net_contact_force: body 별 world 좌표계 접촉력 합, (len(bodies), 3)."""
        contact = data._impl.contact
        net = jp.zeros((len(bodies), 3))
        for dim in np.unique(contact.dim):  # 모델마다 고정
            force, idx = contact_force_dim(self.mjx_model, data, dim)
            world = jp.einsum("ci,cij->cj", force[:, :3], contact.frame[idx])
            geom = contact.geom[idx]
            # 힘은 geom 0 → geom 1 법선 방향으로 geom 1 에 작용
            sign = ((self.geom_body[geom[:, 1], None] == bodies).astype(float) -
                    (self.geom_body[geom[:, 0], None] == bodies))
            net += sign.T @ world
        return net


# ── 수학 도구 ────────────────────────────────────────────────────────────────
def quat_rotate(q, v, inverse=False):
    """legged_gym quat_apply / quat_rotate_inverse, MuJoCo (w, x, y, z) 순서."""
    w, u = q[0], q[1:]
    b = jp.cross(u, v) * w * 2.
    return v * (2. * w ** 2 - 1.) + (-b if inverse else b) + u * jp.dot(u, v) * 2.


def wrap_to_pi(angle):
    return (angle + jp.pi) % (2 * jp.pi) - jp.pi
