"""A1 raw 클립 open-loop PD replay (MuJoCo CPU, menagerie A1 MJCF).

raw/*.csv 의 관절각을 position actuator 에 그대로 넣고 base 궤적을 sim 으로 복원한다.
reference 는 q + IMU quat 운동학(kin_ref)으로 만들고, foot_lock 이 접지 발을 고정하도록 mink IK 로 q 를 보정한다
(Kovar 2002 footskate cleanup). inverse_check 가 역동역학으로 물리 가능성(유령 힘·토크)을 판정한다.
PD replay(open-loop)는 피드백이 없어 yaw 가 누적 이탈하므로 가능성 판정 기준으로 쓰지 않는다.
출력: artifacts/pd_replay/summary.csv, <clip>.npz, videos/<clip>.mp4

  MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python code/motion_tracking/pd_replay.py [--video] [--clips trotForwardSlow ...]
TT 형상 리타겟 (무릎 방향 4 조합, IK): pd_replay.py --retarget → artifacts/retarget_tt/<choice>/<motion>.npz
전환 모션 (A1, 이어붙이기): pd_replay.py --transitions → artifacts/pd_replay/<이름>.npz, 이후 --retarget --only <이름들>
"""
import argparse
import csv
import glob
import os

import mink
import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENE = os.path.join(ROOT, "..", "transformer-transformer", "assets", "mjcf", "unitree_a1", "scene.xml")
FRICTION = 0.8  # MJCF foot friction
LEGS = ["FR", "FL", "RR", "RL"]  # CSV 순서 = MJCF actuator 순서
JOINTS = [f"q_{l}_{j}" for l in LEGS for j in ("hip", "thigh", "calf")]
STANCE_EPS = 0.015  # 가장 낮은 발 대비 높이(m) 이내면 딛은 발. trot swing 높이(수 cm)보다 작게
STANCE_VMAX = 0.3  # m/s. 딛은 발 월드 수평 속도 상한. kin_ref slip 평균(0.05~0.19)보다 크게
RADIUS = 0.02  # A1 foot sphere
BASE_SG_WIN = 9  # reference base 위치 평활 창(프레임, 180ms)
SG_WIN = 7  # 속도 평활 Savitzky-Golay 창(프레임, 140ms)
GHOST_FRAC = 0.1  # base 잔차 힘이 체중의 10% 를 넘으면 불가능 프레임
MIN_RUN = 3  # 이보다 짧은 접지/공중 구간은 깜빡임으로 보고 앞 구간에 합친다
IK_ITERS = 30
FALL_UP = 0.5  # body z 축의 world z 성분. 높이 기준은 sitDown/standUp 의 정상 저자세를 넘어짐으로 오판해 제거
UNUSED = {"backFlip", "howAreYou", "rollBack"}  # 사용자 결정: 미사용 모션
UNLOCKED = "Raw"  # 이름이 <모션>Raw 인 reference 는 발 고정 보정 없음 (kin_ref 결과 그대로, 실제 발 미끄러짐 포함)
TT = os.path.abspath(os.path.join(ROOT, "..", "transformer-transformer"))
# TT 4 족 design space (umi_on_legs_plus_plus) choice 8 자리 = 몸체 0, 다리 종류 FL FR RL RR (0 일반), 팔 장착 (0 직접),
# 앞다리 무릎 반전, 뒷다리 무릎 반전. 사용자 결정: 무릎 방향 4 조합, 연속 치수 13 개는 중간값 0.5 고정, 팔·레일 제거
TT_CHOICES = ["00000000", "00000010", "00000001", "00000011"]
TT_LEG = {"FR": "right_leg", "FL": "left_leg", "RR": "right_leg_1", "RL": "left_leg_1"}
N_LEG = 12  # 다리 관절 수 (qpos·actuator 앞쪽 12)


def build_model(choice=None, kp=None, tau_scale=None, torque_limit=None, uniforms=None, scene_fn=None):
    """choice 가 없으면 A1 (menagerie MJCF, kp 100, ±33.5 N·m), 있으면 TT design space 형상 (kp 30, ±23.7 N·m, 팔 제거).
    두 경우 모두 발 geom 이름을 "<LEG>_foot", 렌더링 카메라 body_cam 을 몸통에 둔다."""
    if choice is None:
        # foot geom 은 이름이 없어 class "foot" 의 반지름 0.02 sphere 에 이름을 붙인다 (body 순서 FR,FL,RR,RL). mink 가 이름으로 찾음
        spec = mujoco.MjSpec.from_file(SCENE)
        spec_feet = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and np.isclose(g.size[0], RADIUS)]
        assert len(spec_feet) == 4, len(spec_feet)
        for g, l in zip(spec_feet, LEGS):
            g.name = f"{l}_foot"
        base = "trunk"
    else:
        import sys

        uniforms = [0.5] * 13 if uniforms is None else list(uniforms)
        if len(uniforms) != 13 or not np.isfinite(uniforms).all() or np.any(np.asarray(uniforms) < 0) or np.any(np.asarray(uniforms) > 1):
            raise ValueError("uniforms must contain 13 finite values in [0, 1]")

        sys.path.insert(0, TT)
        from t2.env.mj_utils import set_up_default_scene
        from t2.robogen.umi_on_legs_plus_plus import umi_on_legs_plus_plus_from_params

        import contextlib

        with contextlib.chdir(TT):  # robogen 컴포넌트 경로가 TT 저장소 기준 상대경로
            root = umi_on_legs_plus_plus_from_params(choices=[int(c) for c in str(choice).zfill(8)], uniforms=uniforms)
            root = set_up_default_scene(root, add_plane=True, plane_z_pos=0.0)
            spec = mujoco.MjSpec.from_string(root.to_xml_string(), assets=root.get_assets())
        # 사용자 결정 (2026-10-11): 팔(ARX5)·레일 제거 → 다리 4 족만. keyframe 은 크기가 바뀌므로 지우고 다리 부분만 다시 만든다
        home = spec.key("home")
        hq, hc = list(home.qpos)[: 7 + N_LEG], list(home.ctrl)[:N_LEG]
        for k in list(spec.keys):
            spec.delete(k)
        drop = set()

        def subtree(bd):
            drop.add(bd.name)
            for c in bd.bodies:
                subtree(c)

        roots = [bd for bd in spec.bodies if bd.name.endswith(("/arx5/", "/rail/"))]  # 컴포넌트 루트 body
        for bd in roots:
            subtree(bd)
        for e in [e for e in spec.excludes if e.bodyname1 in drop or e.bodyname2 in drop]:
            spec.delete(e)
        for act in [act for act in spec.actuators if "arx5" in act.target or "rail" in act.target]:
            spec.delete(act)
        for bd in roots:
            if not any(bd.name != r.name and bd.name.startswith(r.name) for r in roots):  # rail 안의 arx5 는 rail 과 함께 지워짐
                spec.delete(bd)
        spec.add_key(name="home", qpos=hq, ctrl=hc)
        for l in LEGS:
            g = spec.geom(f"quadruped_manipulator/{TT_LEG[l]}/foot")
            assert np.isclose(g.size[0], RADIUS)
            g.name = f"{l}_foot"
    if scene_fn is not None:
        scene_fn(spec)
    if choice is not None:  # TT 형상에는 body_cam 이 이미 있다
        add_terrain(spec)
        m = spec.compile()
        if kp is not None:  # position actuator 비례 이득만 교체 (kd 는 원본 0.5 유지). 사용자 실험 2026-10-11
            m.actuator_gainprm[:, 0] = kp
            m.actuator_biasprm[:, 1] = -kp
        if tau_scale is not None:  # 토크 한계 배율. jumpInPlace 토크 부족 확인용 (사용자 실험 2026-10-12)
            m.actuator_forcerange *= tau_scale
        if torque_limit is not None:
            if not np.isfinite(torque_limit) or torque_limit <= 0:
                raise ValueError("torque_limit must be positive and finite")
            m.actuator_forcerange[:N_LEG] = (-torque_limit, torque_limit)
        return m
    # track_env 렌더링용 (TT a1_umi_on_legs.xml 의 body_cam 과 동일)
    cam = spec.body(base).add_camera(name="body_cam", mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, pos=[1.2, 1.2, 0.5])
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [-0.766, 0.643, 0.0, -0.127, -0.151, 0.980]
    return spec.compile()


def add_terrain(spec, ref=os.path.join(ROOT, "artifacts", "pd_replay", "stairsUpDown.npz")):
    """계단 모션 reference 의 상자들을 정적 box geom "stair_k" 로 장면에 넣는다 (없으면 아무것도 안 함)."""
    if not os.path.exists(ref):
        return
    for k, (x0, x1, y0, y1, top) in enumerate(np.load(ref)["terrain_boxes"]):
        spec.worldbody.add_geom(name=f"stair_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, (y1 - y0) / 2, top / 2],
                                pos=[(x0 + x1) / 2, (y0 + y1) / 2, top / 2], rgba=[0.55, 0.5, 0.45, 1], friction=[FRICTION, 0.005, 0.0001])


def load_clip(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return dict(
        name=os.path.basename(path).replace("_raw_state.csv", ""),
        q=np.stack([d[c] for c in JOINTS], 1),
        rpy=np.stack([d["roll"], d["pitch"], d["yaw"]], 1),
        # 열 이름은 quat_w,x,y,z 지만 실제 값은 (x,y,z,w) 순서다 (rpy 열과 대조해 오차 0 확인). MuJoCo (w,x,y,z) 로 재배열
        quat=np.stack([d["quat_z"], d["quat_w"], d["quat_x"], d["quat_y"]], 1),
        force=np.stack([d[f"force{i}"] for i in range(4)], 1),
        dt=float(np.median(np.diff(d["time"]))),
    )


def rp_quat(roll, pitch):
    """yaw=0, ZYX 오일러 → (w,x,y,z)."""
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])


def init_state(m, d, clip, foot_ids):
    mujoco.mj_resetData(m, d)
    d.qpos[3:7] = rp_quat(*clip["rpy"][0, :2])
    d.qpos[7:19] = clip["q"][0]
    d.ctrl[:] = clip["q"][0]
    mujoco.mj_kinematics(m, d)
    lowest = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in foot_ids)
    d.qpos[2] -= lowest  # 가장 낮은 발이 바닥에 닿게
    mujoco.mj_forward(m, d)


def replay(m, clip, foot_ids, renderer=None):
    d = mujoco.MjData(m)
    init_state(m, d, clip, foot_ids)
    sub = round(clip["dt"] / m.opt.timestep)
    q = clip["q"]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    log = {k: [] for k in ("base_pos", "base_quat", "base_linvel", "qpos", "contact")}
    frames = []
    for t in range(len(q)):
        nxt = q[min(t + 1, len(q) - 1)]
        for s in range(sub):
            d.ctrl[:] = q[t] + (nxt - q[t]) * (s + 1) / sub
            mujoco.mj_step(m, d)
        touching = {c.geom1 if c.geom2 == floor else c.geom2 for c in d.contact[: d.ncon] if floor in (c.geom1, c.geom2)}
        log["base_pos"].append(d.qpos[0:3].copy())
        log["base_quat"].append(d.qpos[3:7].copy())
        log["base_linvel"].append(d.qvel[0:3].copy())
        log["qpos"].append(d.qpos[7:19].copy())
        log["contact"].append([g in touching for g in foot_ids])
        if renderer is not None:
            renderer.update_scene(d, camera=-1)
            frames.append(renderer.render())
    return {k: np.array(v) for k, v in log.items()}, frames


def kin_ref(m, clip, foot_ids):
    """q + IMU quat 만으로 reference 를 계산한다 (force/접촉/속도 센서값은 쓰지 않음).

    접촉 = 가장 낮은 발에서 STANCE_EPS 이내. base xy = 연속 두 프레임 모두 딛은 발이 월드에서 고정이라는 조건,
    base z = 딛은 발 바닥이 z=0. 딛은 발이 없는 프레임(공중)은 직전 수평 속도를 유지한다.
    """
    d = mujoco.MjData(m)
    T, dt = len(clip["q"]), clip["dt"]
    y0 = np.unwrap(clip["rpy"][:, 2])[0]
    qz = np.array([np.cos(-y0 / 2), 0, 0, np.sin(-y0 / 2)])  # 0번 프레임 yaw 를 0 으로
    quat = np.zeros((T, 4))
    rel = np.zeros((T, 4, 3))  # base 원점 기준 월드 방향 발바닥 위치
    for t in range(T):
        mujoco.mju_mulQuat(quat[t], qz, clip["quat"][t])
        d.qpos[:] = 0
        d.qpos[3:7] = quat[t]
        d.qpos[7:19] = clip["q"][t]
        mujoco.mj_kinematics(m, d)
        rel[t] = d.geom_xpos[foot_ids]
        rel[t, :, 2] -= m.geom_size[foot_ids, 0]
    h = rel[:, :, 2]
    contact = h - h.min(1, keepdims=True) < STANCE_EPS

    def integrate(contact):
        pos = np.zeros((T, 3))
        pos[0, 2] = -h[0, contact[0]].mean() if contact[0].any() else -h[0].min()
        v = np.zeros(2)
        for t in range(1, T):
            # ponytail: 접지 발 없는 프레임은 높이·수평속도 유지. 점프 클립에선 포물선 base 로 바꿀 것
            pos[t, 2] = -h[t, contact[t]].mean() if contact[t].any() else pos[t - 1, 2]
            both = contact[t] & contact[t - 1]
            if both.any():
                v = -(rel[t, both, :2] - rel[t - 1, both, :2]).mean(0) / dt
            pos[t, :2] = pos[t - 1, :2] + v * dt
        return pos

    # 2 차: 높이로는 바닥이지만 월드에서 빨리 움직이는 발(낮게 스치는 swing)은 접지에서 뺀다
    foot = integrate(contact)[:, None] + rel
    speed = np.r_[np.zeros((1, 4)), np.linalg.norm(np.diff(foot[:, :, :2], axis=0), axis=-1) / dt]
    fast = speed > STANCE_VMAX
    fast[:, :] &= ~(fast.all(1, keepdims=True))  # 전부 빠르면(base 추정 자체 오류) 높이 판정 유지
    contact &= ~fast
    pos = integrate(contact)
    # 딛는 발 쌍이 바뀔 때 FK 높이 차(~1cm)로 base 가 계단식으로 튄다 → 가속도 1.6g 잡음. 저역 평활
    from scipy.signal import savgol_filter

    pos = savgol_filter(pos, BASE_SG_WIN, 2, axis=0)
    foot = pos[:, None] + rel
    angvel = np.zeros((T, 3))  # body frame
    for t in range(T - 1):
        mujoco.mju_subQuat(angvel[t], quat[t + 1], quat[t])
    angvel[:-1] /= dt
    angvel[-1] = angvel[-2]
    # 품질 지표: 딛은 발이 월드에서 미끄러지는 속도. 0 에 가까울수록 q·quat 가 서로 일관
    slip = np.linalg.norm(np.diff(foot[:, :, :2], axis=0), axis=-1) / dt
    stance2 = contact[1:] & contact[:-1]
    return dict(
        ref_base_pos=pos, ref_base_quat=quat, ref_base_linvel=np.gradient(pos, dt, axis=0), ref_base_angvel=angvel,
        ref_q=clip["q"], ref_qvel=np.gradient(clip["q"], dt, axis=0), ref_foot_pos=foot, ref_contact=contact,
        ref_slip=float(slip[stance2].mean()) if stance2.any() else 0.0,
    )


def clean_runs(c, min_run=MIN_RUN):
    """발별 bool 시퀀스에서 min_run 프레임보다 짧은 구간을 직전 상태로 채운다."""
    c = c.copy()
    for f in range(c.shape[1]):
        t = 0
        while t < len(c):
            e = t
            while e < len(c) and c[e, f] == c[t, f]:
                e += 1
            if e - t < min_run and t > 0 and e < len(c):  # 클립 처음/끝에서 잘린 구간은 깜빡임이 아니다
                c[t:e, f] = c[t - 1, f]
            t = e
    return c


def foot_lock(m, clip, ref, posture_q=None, first_iters=IK_ITERS, ground_z=False, lock=True, base_position_cost=0.1):
    """접지 구간마다 발바닥을 구간 평균 xy, z=0 에 고정하고 mink IK 로 q 를 보정한다.

    공중 구간의 발 목표 = 원래 궤적 + (앞뒤 접지 구간 보정량의 선형 보간) → 경계에서 관절이 튀지 않음.
    base 는 reference 궤적을 따른다. TT 리타겟은 높이 변화를 보존하도록 위치 비용을 높인다.
    """
    posture_q = clip["q"] if posture_q is None else posture_q  # (T, nq-7) 자세 정규화 목표
    T = len(posture_q)
    contact = clean_runs(ref["ref_contact"])
    foot = ref["ref_foot_pos"]  # 발바닥
    target = foot.copy()
    for f in range(4):
        off = np.zeros((T, 3))
        t = 0
        while t < T:
            e = t
            while e < T and contact[e, f] == contact[t, f]:
                e += 1
            if contact[t, f] and lock:  # lock=False: 발 고정 없이 원래 발 궤적 그대로 (미끄러짐 포함) 비교용
                anchor = np.r_[foot[t:e, f].mean(0)] if ground_z else np.r_[foot[t:e, f, :2].mean(0), 0.0]  # 지형 모션은 딛은 높이 유지
                off[t:e] = anchor - foot[t:e, f]
            t = e
        st = np.where(contact[:, f])[0]
        if len(st):
            for k in range(3):
                target[:, f, k] = foot[:, f, k] + np.interp(np.arange(T), st, off[st, k])
    target[:, :, 2] += RADIUS  # mink 는 geom 중심을 맞춘다

    cfg = mink.Configuration(m)
    feet = [mink.FrameTask(f"{l}_foot", "geom", position_cost=1.0, orientation_cost=0.0, lm_damping=1.0) for l in LEGS]
    base = mink.FrameTask(m.body(m.jnt_bodyid[0]).name, "body", position_cost=base_position_cost, orientation_cost=1.0, lm_damping=1.0)
    posture = mink.PostureTask(m, cost=1e-2)
    limits = [mink.ConfigurationLimit(m)]
    q_fix = np.zeros_like(posture_q)
    base_fix = np.zeros((T, 7))
    qpos = np.r_[ref["ref_base_pos"][0], ref["ref_base_quat"][0], posture_q[0]]
    for t in range(T):
        ref_qpos = np.r_[ref["ref_base_pos"][t], ref["ref_base_quat"][t], posture_q[t]]
        cfg.update(qpos)  # 직전 해에서 warm start
        posture.set_target(ref_qpos)
        base.set_target(mink.SE3(np.r_[ref["ref_base_quat"][t], ref["ref_base_pos"][t]]))
        for i, task in enumerate(feet):
            task.set_target(mink.SE3.from_translation(target[t, i]))
        for _ in range(first_iters if t == 0 else IK_ITERS):
            vel = mink.solve_ik(cfg, [*feet, base, posture], clip["dt"], "daqp", damping=1e-5, limits=limits)
            cfg.integrate_inplace(vel, clip["dt"])
        qpos = cfg.q.copy()
        q_fix[t], base_fix[t] = qpos[7:], qpos[:7]

    # 보정 결과의 발 위치로 미끄러짐 재측정
    d = mujoco.MjData(m)
    fpos = np.zeros((T, 4, 3))
    for t in range(T):
        d.qpos[:7], d.qpos[7:] = base_fix[t], q_fix[t]
        mujoco.mj_kinematics(m, d)
        fpos[t] = d.geom_xpos[[m.geom(f"{l}_foot").id for l in LEGS]]
    slip = np.linalg.norm(np.diff(fpos[:, :, :2], axis=0), axis=-1) / clip["dt"]
    stance2 = contact[1:] & contact[:-1]
    dq = np.degrees(np.abs(q_fix - posture_q)[:, :N_LEG])
    return dict(
        fix_q=q_fix, fix_base_pos=base_fix[:, :3], fix_base_quat=base_fix[:, 3:], fix_foot_pos=fpos - [0, 0, RADIUS],
        fix_contact=contact, fix_foot_target=target - [0, 0, RADIUS],
        fix_slip=float(slip[stance2].mean()) if stance2.any() else 0.0,
        fix_track_err=float(np.linalg.norm(fpos - target, axis=-1).mean()),
        fix_dq_mean_deg=float(dq.mean()), fix_dq_max_deg=float(dq.max()),
    )


def inverse_check(m, fix, dt):
    """보정된 reference 가 A1 몸으로 물리적으로 가능한지 프레임별로 판정한다.

    접촉 없는 역동역학으로 필요한 일반화 힘 τ 를 구하고, base 6 축 τ_b 를 접지 발의 접지력
    (마찰 피라미드 안, NNLS) 으로 설명한다. 남는 잔차 = 유령 힘, 관절 토크 = τ_j − J_jᵀ f.
    """
    from scipy.optimize import nnls
    from scipy.signal import savgol_filter

    T = len(fix["fix_q"])
    qpos = np.c_[fix["fix_base_pos"], fix["fix_base_quat"], fix["fix_q"]]
    qvel = np.zeros((T, m.nv))
    for t in range(T - 1):
        mujoco.mj_differentiatePos(m, qvel[t], dt, qpos[t], qpos[t + 1])
    qvel[-1] = qvel[-2]
    # 50Hz 차분 노이즈가 가속도에서 증폭되므로 속도를 평활한 뒤 미분
    qvel = savgol_filter(qvel, SG_WIN, 2, axis=0)
    qacc = np.gradient(qvel, dt, axis=0)

    mu = FRICTION
    edges = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]]).T  # 마찰 피라미드 모서리 (3,4)
    m2 = m  # 접촉만 잠시 끄고 끝에서 복원
    flags = m2.opt.disableflags
    m2.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m2)
    foot_body = [m2.geom_bodyid[m2.geom(f"{l}_foot").id] for l in LEGS]
    weight = mujoco.mj_getTotalmass(m2) * 9.81
    res_f, res_m, tau, grf = np.zeros(T), np.zeros(T), np.zeros((T, m2.nv - 6)), np.zeros((T, 4, 3))
    jacp = np.zeros((3, m2.nv))
    for t in range(T):
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos[t], qvel[t], qacc[t]
        mujoco.mj_inverse(m2, d)
        need = d.qfrc_inverse.copy()
        cols, Js = [], []
        for i in np.where(fix["fix_contact"][t])[0]:
            p = d.geom_xpos[m2.geom(f"{LEGS[i]}_foot").id] - [0, 0, RADIUS]
            mujoco.mj_jac(m2, d, jacp, None, p, foot_body[i])
            Js.append((i, jacp.copy()))
            cols.append(jacp[:, :6].T @ edges)  # 각 모서리 단위힘이 base 6 축에 주는 일반화 힘
        if cols:
            A = np.hstack(cols)
            lam, _ = nnls(A, need[:6])
            r = need[:6] - A @ lam
            f_all = need[6:].copy()
            for k, (i, J) in enumerate(Js):
                f = edges @ lam[4 * k: 4 * k + 4]
                grf[t, i] = f
                f_all -= J[:, 6:].T @ f
        else:
            r, f_all = need[:6], need[6:]
        res_f[t], res_m[t] = np.linalg.norm(r[:3]), np.linalg.norm(r[3:])
        tau[t] = f_all
    m2.opt.disableflags = flags
    bad = (res_f > GHOST_FRAC * weight) | (np.abs(tau) > m.actuator_forcerange[:, 1]).any(1)
    return dict(
        inv_res_force=res_f, inv_res_moment=res_m, inv_tau=tau, inv_grf=grf, inv_qvel=qvel[:, 6:], inv_bad=bad,
        inv_bad_frac=float(bad.mean()), inv_res_force_p95_pct=float(np.percentile(res_f, 95) / weight * 100),
        inv_tau_p99=float(np.percentile(np.abs(tau), 99)), inv_tau_max=float(np.abs(tau).max()),
        inv_qvel_max=float(np.abs(qvel[:, 6:]).max()),
    )


def quat_rpy(qs):
    w, x, y, z = qs.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], 1)


def metrics(clip, log, ref):
    rpy = quat_rpy(log["base_quat"])
    up_z = 1 - 2 * (log["base_quat"][:, 1] ** 2 + log["base_quat"][:, 2] ** 2)
    fallen = up_z < FALL_UP
    fall_t = int(np.argmax(fallen)) if fallen.any() else -1
    ok = slice(0, fall_t if fall_t > 0 else len(rpy))  # 넘어진 뒤는 오차 계산에서 제외
    yaw_real = np.unwrap(clip["rpy"][:, 2])
    yaw_sim = np.unwrap(rpy[:, 2])
    rp, sp = ref["ref_base_pos"], log["base_pos"]
    return dict(
        clip=clip["name"],
        frames=len(rpy),
        survived=int(fall_t < 0),
        fall_frame=fall_t,
        joint_rmse_deg=np.degrees(np.sqrt(np.mean((log["qpos"][ok] - clip["q"][ok]) ** 2))),
        rp_rmse_deg=np.degrees(np.sqrt(np.mean((rpy[ok, :2] - clip["rpy"][ok, :2]) ** 2))),
        dyaw_sim_deg=np.degrees(yaw_sim[ok][-1] - yaw_sim[0]),
        dyaw_real_deg=np.degrees(yaw_real[ok][-1] - yaw_real[0]),
        dx_sim_m=sp[ok][-1, 0] - sp[0, 0],
        dy_sim_m=sp[ok][-1, 1] - sp[0, 1],
        dx_ref_m=rp[-1, 0],
        dy_ref_m=rp[-1, 1],
        ref_feet_down=float(ref["ref_contact"].sum(1).mean()),
        ref_slip_mps=ref["ref_slip"],
        contact_agree=float(np.mean(log["contact"][ok] == ref["ref_contact"][ok])),
    )


# sim 정책으로 만든 모션 (gen_jump.py 등): 이름 → (raw npz, 사용할 프레임 구간). 위치·접지가 sim 정답이라 kin_ref/foot_lock 불필요
SIM_MOTIONS = {
    # MetalHead 제자리 점프 2 회 중 2 번째 (앞서 4 발로 정지한 상태에서 시작, 착지 후 안정까지 2 s)
    "jumpInPlace": ("artifacts/sim_motions/jumpInPlace_raw.npz", (150, 216)),
    # legged_gym A1 rough 정책 (1500 it), 10 cm × 4 단 오르기 → 1 m 평판 → 내리기 (gen_stairs.py), 0.5 m/s 11.5 s
    "stairsUpDown": ("artifacts/sim_motions/stairs_raw.npz", (25, None)),  # 생성 직후 내려앉는 0.5 s 제외
}
STAIRS_ORIGIN = (0.0, 30.0)  # 계단 모션 시작 xy. 평지 모션(원점 출발)과 tracking 장면에서 겹치지 않게 y 로 떨어뜨림


def import_sim(name, raw, window, dt=0.02):
    """sim 녹화(정책 주기)를 50 Hz reference(fix_*)로 리샘플하고 역동역학 검사를 붙인다. 출력 형식은 pd_replay 와 같다."""
    from scipy.spatial.transform import Rotation, Slerp

    z = np.load(os.path.join(ROOT, raw))
    a, b = window
    b = len(z["q"]) if b is None else b
    src_dt = float(z["dt"])
    ts = np.arange(b - a) * src_dt
    td = np.arange(0, ts[-1] + 1e-9, dt)
    pos = np.stack([np.interp(td, ts, z["base_pos"][a:b, k]) for k in range(3)], 1)
    q = np.stack([np.interp(td, ts, z["q"][a:b, k]) for k in range(12)], 1)
    quat_xyzw = Slerp(ts, Rotation.from_quat(z["base_quat"][a:b][:, [1, 2, 3, 0]]))(td).as_quat()
    contact = z["contact"][a:b][np.minimum(np.round(td / src_dt).astype(int), b - a - 1)]
    extra = {}
    if "boxes" in z.files:
        # 지형 모션: 회전하지 않고 시작 xy 를 STAIRS_ORIGIN 으로 평행이동 (계단 상자도 같이). 상자 = (x0, x1, y0, y1, 윗면 z)
        shift = np.r_[STAIRS_ORIGIN] - pos[0, :2]
        quat = quat_xyzw[:, [3, 0, 1, 2]]
        pos[:, :2] += shift
        bx = z["boxes"]
        extra["terrain_boxes"] = np.c_[bx[:, :2] + shift[0], np.tile([-3.0, 3.0], (len(bx), 1)) + STAIRS_ORIGIN[1], bx[:, 2]]
    else:
        # kin_ref 와 같은 원점 규칙: 0 번 프레임 yaw = 0, xy = 0
        yaw0 = Rotation.from_quat(quat_xyzw[0]).as_euler("ZYX")[0]
        rz = Rotation.from_euler("z", -yaw0)
        quat = (rz * Rotation.from_quat(quat_xyzw)).as_quat()[:, [3, 0, 1, 2]]
        pos[:, :2] = rz.apply(pos - [pos[0, 0], pos[0, 1], 0])[:, :2]
    m = build_model()
    d = mujoco.MjData(m)
    feet = [m.geom(f"{l}_foot").id for l in LEGS]
    fpos = np.zeros((len(td), 4, 3))
    for t in range(len(td)):
        d.qpos[:] = np.r_[pos[t], quat[t], q[t]]
        mujoco.mj_kinematics(m, d)
        fpos[t] = d.geom_xpos[feet] - [0, 0, RADIUS]
    fix = dict(fix_q=q, fix_base_pos=pos, fix_base_quat=quat, fix_foot_pos=fpos, fix_contact=contact)
    inv = inverse_check(m, fix, dt)
    out = os.path.join(ROOT, "artifacts", "pd_replay", f"{name}.npz")
    np.savez(out, **fix, **inv, **extra, dt=dt)
    print(f"{name}: {len(td)} frames, base z {pos[:, 2].min():.3f}~{pos[:, 2].max():.3f}, flight frames {(contact.sum(1) == 0).sum()}, "
          f"inv bad={inv['inv_bad_frac'] * 100:.1f}% ghost p95={inv['inv_res_force_p95_pct']:.1f}%W tau p99/max={inv['inv_tau_p99']:.1f}/{inv['inv_tau_max']:.1f} -> {out}")


def export_unlocked(name):
    """<name>.npz 의 kin_ref 결과(ref_*, 발 고정 전)를 fix_* 형식의 <name>Raw.npz 로 저장 + 역동역학 검사."""
    z = np.load(os.path.join(ROOT, "artifacts", "pd_replay", f"{name}.npz"))
    fix = dict(fix_q=z["ref_q"], fix_base_pos=z["ref_base_pos"], fix_base_quat=z["ref_base_quat"],
               fix_foot_pos=z["ref_foot_pos"], fix_contact=clean_runs(z["ref_contact"]))
    inv = inverse_check(build_model(), fix, float(z["dt"]))
    out = os.path.join(ROOT, "artifacts", "pd_replay", f"{name}{UNLOCKED}.npz")
    np.savez(out, **fix, **inv, dt=z["dt"])
    print(f"{name}{UNLOCKED}: slip {float(z['ref_slip']):.3f} m/s, inv bad={inv['inv_bad_frac'] * 100:.1f}% -> {out}")


# 전환 모션 (사용자 결정 2026-10-12): 이름 → ((A 모션, 시작, 끝), (B 모션, 시작, 끝), B 마지막 프레임 추가 유지 프레임 수).
# 계단 전환은 stairsUpDown 안에 이미 있고, 조합 3·jump→stair_up·회전 보행 중 점프는 held-out
TRANSITIONS = {
    "walkToJump": (("walkForwardSlow", 0, 200), ("jumpInPlace", 0, None), 0),
    "jumpToWalk": (("jumpInPlace", 0, 80), ("walkForwardSlow", 30, None), 0),
    "walkToStand": (("walkForwardSlow", 0, 200), ("walkForwardSlow", 0, 1), 74),  # 0 번 프레임 = 서 있는 자세
}
BLEND = 25  # 프레임 (0.5 s). A 끝과 B 앞을 겹쳐 섞는 길이


def stitch(name, A, B, hold):
    """A 끝 BLEND 프레임과 B 앞 BLEND 프레임을 섞어 잇는다 (motion graph 전환, Kovar 2002).

    B 는 겹침 시작에서 A 와 xy·yaw 가 같도록 평행이동·회전. 관절·위치는 smoothstep 선형 보간, 자세는 slerp.
    접지는 섞은 자세의 발 높이·속도로 다시 판정하고 foot_lock 으로 발 미끄러짐을 보정한다.
    """
    from scipy.spatial.transform import Rotation, Slerp

    def seg(clip, a, b):
        z = np.load(os.path.join(ROOT, "artifacts", "pd_replay", f"{clip}.npz"))
        return z["fix_base_pos"][a:b].copy(), z["fix_base_quat"][a:b].copy(), z["fix_q"][a:b].copy(), float(z["dt"])

    pa, qa, ja, dt = seg(*A)
    pb, qb, jb, _ = seg(*B)
    if hold:
        pb, qb, jb = (np.r_[x, np.repeat(x[-1:], hold, 0)] for x in (pb, qb, jb))
    W = min(BLEND, len(pa), len(pb))
    k = len(pa) - W
    ra, rb = Rotation.from_quat(qa[:, [1, 2, 3, 0]]), Rotation.from_quat(qb[:, [1, 2, 3, 0]])
    rz = Rotation.from_euler("z", ra[k].as_euler("ZYX")[0] - rb[0].as_euler("ZYX")[0])
    pb[:, :2] = rz.apply(pb - [pb[0, 0], pb[0, 1], 0])[:, :2] + pa[k, :2]
    rb = rz * rb
    s = (np.arange(W) + 1) / (W + 1)
    w = s * s * (3 - 2 * s)
    pos = np.r_[pa[:k], (1 - w)[:, None] * pa[k:] + w[:, None] * pb[:W], pb[W:]]
    q = np.r_[ja[:k], (1 - w)[:, None] * ja[k:] + w[:, None] * jb[:W], jb[W:]]
    mid = Rotation.concatenate([Slerp([0, 1], Rotation.concatenate([ra[k + i], rb[i]]))(w[i]) for i in range(W)])
    quat = Rotation.concatenate([ra[:k], mid, rb[W:]]).as_quat()[:, [3, 0, 1, 2]]

    m = build_model()
    d = mujoco.MjData(m)
    feet = [m.geom(f"{l}_foot").id for l in LEGS]
    T = len(q)
    foot = np.zeros((T, 4, 3))
    for t in range(T):
        d.qpos[:] = np.r_[pos[t], quat[t], q[t]]
        mujoco.mj_kinematics(m, d)
        foot[t] = d.geom_xpos[feet] - [0, 0, RADIUS]
    speed = np.r_[np.zeros((1, 4)), np.linalg.norm(np.diff(foot[:, :, :2], axis=0), axis=-1) / dt]
    contact = (foot[:, :, 2] < STANCE_EPS) & (speed < STANCE_VMAX)  # 평지 전용
    ref = dict(ref_base_pos=pos, ref_base_quat=quat, ref_foot_pos=foot, ref_contact=contact)
    fix = foot_lock(m, dict(dt=dt), ref, posture_q=q)
    inv = inverse_check(m, fix, dt)
    out = os.path.join(ROOT, "artifacts", "pd_replay", f"{name}.npz")
    np.savez(out, **fix, **inv, dt=dt)
    print(f"{name}: {T} frames (blend {k}~{k + W}), slip {fix['fix_slip']:.3f} m/s, foot_err={fix['fix_track_err'] * 1000:.1f}mm, "
          f"dq max={fix['fix_dq_max_deg']:.1f}deg, inv bad={inv['inv_bad_frac'] * 100:.1f}% ghost p95={inv['inv_res_force_p95_pct']:.1f}%W -> {out}")


def retarget_tt(src, dst, only=None, height_trace_only=False, torque_limit=None, choices=None, uniforms=None):
    """A1 reference(fix_*) → TT 형상 (TT_CHOICES). 치수가 달라 IK 로 리타겟한다 (Peng 2020 keypoint IK 방식).

    발 목표 (몸통 기준) = TT 엉덩이 + s·(A1 발 − A1 엉덩이), s = 다리 길이 비 (TT/A1). base xy 도 s 배 (보폭과 일치),
    base 자세는 A1 그대로, base z 는 딛은 발이 바닥에 닿게. 이후 foot_lock 으로 딛은 발 고정 + mink IK.
    출력: <dst>/<choice>/<motion>.npz (fix_*, inv_*, dt, a1_qpos), <dst>/summary.csv
    height_trace_only=True: 중간 높이만 반환하고 IK·역동역학·파일 쓰기를 건너뛴다.
    """
    from scipy.signal import savgol_filter

    mA = build_model()
    dA = mujoco.MjData(mA)
    footA = [mA.geom(f"{l}_foot").id for l in LEGS]
    thighA = [mA.body(f"{l}_thigh").id for l in LEGS]
    rows = []
    height_traces = {}
    for choice in TT_CHOICES if choices is None else choices:
        m = build_model(choice, torque_limit=torque_limit, uniforms=uniforms)
        d = mujoco.MjData(m)
        base = m.jnt_bodyid[0]
        home = m.key("home").qpos.copy()
        d.qpos[:] = home
        mujoco.mj_kinematics(m, d)
        R = d.xmat[base].reshape(3, 3)
        hipT = np.array([(d.xpos[m.body(f"quadruped_manipulator/{TT_LEG[l]}/thigh").id] - d.xpos[base]) @ R for l in LEGS])
        # 다리 길이 = 허벅지 관절 ~ 무릎 ~ 발 중심 (TT 는 무릎 반전과 무관)
        legT = sum(np.linalg.norm(m.body_pos[m.body(f"quadruped_manipulator/left_leg/{b}").id]) for b in ("calf", "foot"))
        legA = sum(np.linalg.norm(mA.body_pos[mA.body(b).id]) for b in ("FL_calf",)) + np.linalg.norm(mA.geom_pos[footA[1]])
        s = legT / legA
        if not height_trace_only:
            os.makedirs(os.path.join(dst, choice), exist_ok=True)
        for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
            name = os.path.basename(f)[:-4]
            if name in UNUSED or (only and name not in only):
                continue
            z = np.load(f)
            dt = float(z["dt"])
            a1_qpos = np.c_[z["fix_base_pos"], z["fix_base_quat"], z["fix_q"]]
            T = len(a1_qpos)
            tgt_b = np.zeros((T, 4, 3))
            for t in range(T):
                dA.qpos[:] = a1_qpos[t]
                mujoco.mj_kinematics(mA, dA)
                RA = dA.xmat[mA.jnt_bodyid[0]].reshape(3, 3)
                pA = dA.xpos[mA.jnt_bodyid[0]]
                tgt_b[t] = hipT + s * ((dA.geom_xpos[footA] - pA) @ RA - (dA.xpos[thighA] - pA) @ RA)
            quat = z["fix_base_quat"]
            contact = z["fix_contact"].astype(bool)
            rel = np.zeros((T, 4, 3))  # 몸통 원점 기준 월드 방향 발 중심
            for t in range(T):
                Rq = np.zeros(9)
                mujoco.mju_quat2Mat(Rq, quat[t])
                rel[t] = tgt_b[t] @ Rq.reshape(3, 3).T
            terrain = "terrain_boxes" in z.files
            # 딛은 발 바닥 높이: 평지 0, 지형 모션은 A1 발이 딛은 높이 (계단 윗면)
            ground = z["fix_foot_pos"][:, :, 2] if terrain else np.zeros((T, 4))
            zb = np.full(T, np.nan)
            for t in range(T):
                if contact[t].any():
                    zb[t] = (ground[t, contact[t]] + RADIUS - rel[t, contact[t], 2]).mean()
            ok = ~np.isnan(zb)
            contact_z = zb.copy()
            # 딛은 발이 없는 프레임(점프 공중)은 A1 높이의 s 배 + 앞뒤 접지 구간 오프셋 보간 → 포물선 유지
            off = zb - s * z["fix_base_pos"][:, 2]
            zb = s * z["fix_base_pos"][:, 2] + np.interp(np.arange(T), np.where(ok)[0], off[ok])
            interpolated_z = zb.copy()
            zb = savgol_filter(zb, BASE_SG_WIN, 2)
            if height_trace_only:
                height_traces[(choice, name)] = dict(dt=dt, scale=s, a1_z=z["fix_base_pos"][:, 2].copy(),
                    contact=contact.copy(), interp_ok=ok.copy(), off=off.copy(), contact_z=contact_z,
                    interpolated_z=interpolated_z, smoothed_z=zb.copy())
                continue
            sxy = 1.0 if terrain else s  # 계단은 고정 위치라 보폭을 줄이지 않는다
            pos = np.c_[sxy * z["fix_base_pos"][:, :2], zb]
            ref = dict(ref_base_pos=pos, ref_base_quat=quat, ref_foot_pos=pos[:, None] + rel - [0, 0, RADIUS], ref_contact=contact)
            posture = np.tile(home[7:], (T, 1))
            fix = foot_lock(m, dict(dt=dt), ref, posture_q=posture, first_iters=300, ground_z=terrain,
                            lock=not name.endswith(UNLOCKED), base_position_cost=1.0)
            fix["fix_q"][:, N_LEG:] = home[7 + N_LEG:]  # 팔은 home 자세 고정
            inv = inverse_check(m, fix, dt)
            extra = {k: z[k] for k in ("terrain_boxes", "source_motion", "seed", "speed", "stride", "heading_rad") if k in z.files}
            np.savez(os.path.join(dst, choice, f"{name}.npz"), **fix, **inv, **extra, dt=dt, a1_qpos=a1_qpos, scale=s,
                     uniforms=np.asarray([0.5] * 13 if uniforms is None else uniforms))
            lo, hi = m.jnt_range[1:].T
            r = dict(choice=choice, motion=name, scale=s, foot_err_mm=fix["fix_track_err"] * 1000, slip=fix["fix_slip"],
                     q_at_limit=float(((fix["fix_q"] <= lo + 1e-3) | (fix["fix_q"] >= hi - 1e-3)).any(1).mean()),
                     **{k: v for k, v in inv.items() if isinstance(v, float)})
            rows.append(r)
            print(f"{choice} {name:18s} s={s:.3f} foot_err={r['foot_err_mm']:5.1f}mm slip={r['slip']:.3f} "
                  f"q_at_limit={r['q_at_limit']:.2f} inv bad={inv['inv_bad_frac'] * 100:4.1f}% "
                  f"ghost p95={inv['inv_res_force_p95_pct']:5.1f}%W tau p99/max={inv['inv_tau_p99']:5.1f}/{inv['inv_tau_max']:5.1f}", flush=True)
    if height_trace_only:
        return height_traces
    if not rows:
        raise ValueError(f"no reference clips found in {src}")
    summary = os.path.join(dst, "summary.csv" if not only else f"summary_{'_'.join(only)}.csv")
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("->", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlocked", nargs="*", help="발 고정 없는 <모션>Raw reference 를 만든다")
    ap.add_argument("--only", nargs="*", help="--retarget 할 모션 이름 (기본 전부)")
    ap.add_argument("--import_sim", action="store_true", help="SIM_MOTIONS 의 sim 녹화 → artifacts/pd_replay/<name>.npz")
    ap.add_argument("--transitions", action="store_true", help="TRANSITIONS 이어붙이기 → artifacts/pd_replay/<name>.npz")
    ap.add_argument("--retarget", action="store_true", help="artifacts/pd_replay → artifacts/retarget_tt/<choice>/")
    ap.add_argument("--source-dir", default=os.path.join(ROOT, "artifacts", "pd_replay"))
    ap.add_argument("--choices", nargs="*", choices=TT_CHOICES, help="리타겟할 TT topology (기본 4개)")
    ap.add_argument("--uniforms", nargs=13, type=float, help="TT 연속 치수 13개 (기본 모두 0.5)")
    ap.add_argument("--torque-limit", type=float, help="TT retarget inverse check의 관절별 대칭 토크 한계 (N·m)")
    ap.add_argument("--clips", nargs="*", help="기본: raw/ 전체")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "pd_replay"))
    a = ap.parse_args()
    if a.unlocked:
        for name in a.unlocked:
            export_unlocked(name)
        return
    if a.import_sim:
        for name, (raw, window) in SIM_MOTIONS.items():
            import_sim(name, raw, window)
        return
    if a.transitions:
        for name, (A, B, hold) in TRANSITIONS.items():
            stitch(name, A, B, hold)
        return
    if a.retarget:
        dst = a.out if a.out != os.path.join(ROOT, "artifacts", "pd_replay") else os.path.join(ROOT, "artifacts", "retarget_tt")
        return retarget_tt(a.source_dir, dst, a.only, torque_limit=a.torque_limit, choices=a.choices,
                           uniforms=a.uniforms)

    clips = [load_clip(p) for p in sorted(glob.glob(os.path.join(ROOT, "raw", "*_raw_state.csv")))]
    if a.clips:
        clips = [c for c in clips if c["name"] in a.clips]
    m = build_model()
    foot_ids = [m.geom(f"{l}_foot").id for l in LEGS]
    renderer = mujoco.Renderer(m, 360, 480) if a.video else None
    os.makedirs(os.path.join(a.out, "videos"), exist_ok=True)

    rows = []
    for c in clips:
        ref = kin_ref(m, c, foot_ids)
        fix = foot_lock(m, c, ref)
        inv = inverse_check(m, fix, c["dt"])
        log, frames = replay(m, dict(c, q=fix["fix_q"]), foot_ids, renderer)  # 보정된 q 를 재생
        np.savez(os.path.join(a.out, f"{c['name']}.npz"), **log, **ref, **fix, **inv, rpy_real=c["rpy"], force=c["force"], dt=c["dt"])
        if frames:
            import imageio

            imageio.mimsave(os.path.join(a.out, "videos", f"{c['name']}.mp4"), frames, fps=round(1 / c["dt"]))
        rows.append(metrics(dict(c, q=fix["fix_q"]), log, ref))
        r = rows[-1]
        r.update({k: v for k, v in {**fix, **inv}.items() if isinstance(v, float)})
        print(f"{r['clip']:18s} slip ref->fix {r['ref_slip_mps']:.3f}->{r['fix_slip']:.3f} foot_err={r['fix_track_err']*1000:4.1f}mm "
              f"dq mean/max={r['fix_dq_mean_deg']:4.2f}/{r['fix_dq_max_deg']:5.2f}deg | inv bad={r['inv_bad_frac']*100:4.1f}% "
              f"ghost p95={r['inv_res_force_p95_pct']:5.1f}%W tau p99/max={r['inv_tau_p99']:5.1f}/{r['inv_tau_max']:5.1f} qvel max={r['inv_qvel_max']:5.1f}")
        print(f"{r['clip']:18s} surv={r['survived']} fall={r['fall_frame']:4d} jrmse={r['joint_rmse_deg']:5.2f} "
              f"rp={r['rp_rmse_deg']:5.2f} dyaw sim/real={r['dyaw_sim_deg']:7.1f}/{r['dyaw_real_deg']:7.1f} "
              f"dxy sim=({r['dx_sim_m']:6.2f},{r['dy_sim_m']:6.2f}) ref=({r['dx_ref_m']:6.2f},{r['dy_ref_m']:6.2f}) "
              f"feet={r['ref_feet_down']:.2f} slip={r['ref_slip_mps']:.3f} contact={r['contact_agree']:.2f}")

    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("->", a.out)


if __name__ == "__main__":
    main()
