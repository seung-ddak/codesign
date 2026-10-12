"""Full-body motion tracking MJX 환경 (A1).

reference = pd_replay.py 가 만든 artifacts/pd_replay/<clip>.npz 의 fix_* (50Hz).
보상은 Peng et al. 2020 "Learning Agile Robotic Locomotion Skills by Imitating Animals" 식 (4)~(9).
"""

import os
from typing import Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math
from scipy.signal import savgol_filter

from t2.mjx_env.base_env import MjxEnv, State, make_data, step

LEGS = ["FR", "FL", "RR", "RL"]
FOOT_RADIUS = 0.02  # fix_foot_pos 는 발바닥, sim 은 foot geom 중심을 비교하므로 보정
SG_WIN = 7  # 속도 평활 Savitzky-Golay 창 (pd_replay.SG_WIN 과 동일)


def quat_angle(q: jax.Array) -> jax.Array:
    """단위 quaternion 의 회전각 (rad)."""
    return 2.0 * jnp.arctan2(jnp.linalg.norm(q[1:]), jnp.abs(q[0]))


class TrackMjxEnv(MjxEnv):
    def __init__(
        self,
        reference_dir: str,
        clips: list[str],
        max_episode_length: int,
        mj_model_fn: Callable[[], mujoco.MjModel],
        n_ctrl: int,
        action_scale: float,
        obs_time_indices: list[int],
        w_pose: float,
        w_vel: float,
        w_ee: float,
        w_root_pose: float,
        w_root_vel: float,
        k_pose: float,
        k_vel: float,
        k_ee: float,
        k_root_pos: float,
        k_root_orn: float,
        k_root_linvel: float,
        k_root_angvel: float,
        penalty_action_rate: float,
        penalty_torque: float,
        penalty_joint_vel: float,
        penalty_joint_acc: float,
        penalty_joint_pos_limit: float,
        penalty_termination: float,
        reward_clip_positive: bool,
        reward_nan_to_num: Optional[float],
        ctrl_dt: float,
        sim_dt: float,
        solver_iterations: int,
        solver_ls_iterations: int,
        impratio: int,
        impl: Literal["jax", "warp"],
        kick_wait_times: tuple[float, float],
        kick_durations: tuple[float, float],
        velocity_kick: tuple[float, float],
        soft_joint_pos_limit_factor: float,
        reset_qvel: float,
        reset_qpos: float,
        min_start_margin: int,
        termination_fellover_threshold: float,
        termination_pos_err_threshold: float,
        termination_orn_err_threshold: float,
        termination_pose_consecutive_steps: int,
        obs_noise: dict[str, float],
        clip_actions: tuple[float, float] = (-10, 10),
        clip_boost: Optional[dict[str, float]] = None,
        switch_prob: float = 0.0,
        switch_wait: tuple[float, float] = (1.0, 3.0),
        switch_window: int = 25,
        switch_min_remain: int = 50,
        switch_exclude: tuple[str, ...] = (),
        switch_forbid: tuple[tuple[str, str], ...] = (),
        switch_entry: Literal["match", "start"] = "match",
        switch_at: Literal["time", "end"] = "time",
        switch_ramp: bool = True,
    ):
        super().__init__(ctrl_dt, sim_dt)
        self.max_episode_length = max_episode_length
        self.action_scale = action_scale
        self.obs_time_indices = sorted(obs_time_indices)
        self.clip_actions = clip_actions
        self.w = dict(pose=w_pose, vel=w_vel, ee=w_ee, root_pose=w_root_pose, root_vel=w_root_vel)
        self.k_pose, self.k_vel, self.k_ee = k_pose, k_vel, k_ee
        self.k_root_pos, self.k_root_orn = k_root_pos, k_root_orn
        self.k_root_linvel, self.k_root_angvel = k_root_linvel, k_root_angvel
        self.penalty_action_rate = penalty_action_rate
        self.penalty_torque = penalty_torque
        self.penalty_joint_vel = penalty_joint_vel
        self.penalty_joint_acc = penalty_joint_acc
        self.penalty_joint_pos_limit = penalty_joint_pos_limit
        self.penalty_termination = penalty_termination
        self.reward_clip_positive = reward_clip_positive
        self.reward_nan_to_num = reward_nan_to_num
        self.impratio = impratio
        self.solver_iterations = solver_iterations
        self.solver_ls_iterations = solver_ls_iterations
        self.impl = impl
        self.kick_wait_times = kick_wait_times
        self.kick_durations = kick_durations
        self.velocity_kick = velocity_kick
        self.soft_joint_pos_limit_factor = soft_joint_pos_limit_factor
        self.reset_qvel = reset_qvel
        self.reset_qpos = reset_qpos
        self.min_start_margin = min_start_margin
        self.termination_fellover_threshold = termination_fellover_threshold
        self.termination_pos_err_threshold = termination_pos_err_threshold
        self.termination_orn_err_threshold = termination_orn_err_threshold
        self.termination_pose_consecutive_steps = termination_pose_consecutive_steps
        self.obs_noise = obs_noise

        self.mj_model = mj_model_fn()
        self.n_ctrl = n_ctrl  # 정책이 제어하는 앞쪽 actuator 수 (다리 12). 나머지(TT 팔)는 home 자세 PD 고정
        self.preprocess_mj_model(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model, impl=impl)
        self.load_references([os.path.join(reference_dir, f"{c}.npz") for c in clips])
        # 클립 샘플링 가중치 (이름 → 배수, 나머지 1). jumpInPlace 등 비율 상향 실험 (사용자 결정 2026-10-12)
        boost = clip_boost or {}
        assert set(boost) <= set(clips), set(boost) - set(clips)
        w = np.array([float(boost.get(c, 1.0)) for c in clips])
        self.clip_p = jnp.array(w / w.sum())
        # 목표 바꾸기 (전환 학습, 사용자 결정 2026-10-12): episode 중 목표 모션을 A → B 로 바꾸고 B 를 로봇 현재 xy·yaw 에 맞춘다.
        # switch_prob = 바꾸기가 있는 episode 비율 (0 = 기존과 동일). 바꾼 뒤 switch_window step 은 tracking 보상을 0→1 로 올리고
        # 위치·자세 오차 종료를 끈다. 계단 등 지형이 고정된 모션은 switch_exclude (바꾸기의 출발·도착 모두 금지)
        self.switch_prob, self.switch_wait, self.switch_window = switch_prob, tuple(switch_wait), switch_window
        # B 진입 프레임: match = A 현재 자세와 가장 비슷한 프레임, start = B 의 처음 (사용자 결정 2026-10-12: start).
        # start 는 바꾼 뒤 switch_window 동안 B 를 첫 프레임에 멈춰 두고 (그 자세로 옮겨 가는 구간), 이후 재생한다
        self.switch_entry = switch_entry
        # 바꾸는 시점: time = switch_wait 뒤 아무 접지 프레임, end = A 의 마지막 프레임 (DeepMimic skill selector 처럼 동작 경계에서,
        # 끝나는 대신 B 로 이어 간다). switch_ramp = 바꾼 뒤 tracking 보상을 0→1 로 올림 (DeepMimic 은 없음)
        self.switch_at, self.switch_ramp = switch_at, switch_ramp
        self._build_switch_tables(list(clips), switch_min_remain, set(switch_exclude), {tuple(x) for x in switch_forbid})

    def _build_switch_tables(self, clips, min_remain, exclude, forbid):
        assert exclude <= set(clips) and all(a in clips and b in clips for a, b in forbid), (exclude, forbid)
        starts, ends = np.asarray(self.clip_start), np.asarray(self.clip_end)
        n_clip, N = len(clips), int(ends[-1])
        frame_clip = np.repeat(np.arange(n_clip), ends - starts)
        excl = np.array([c in exclude for c in clips])
        grounded = np.asarray(self.ref_contact).sum(1) >= 2  # 공중·한 발 프레임에서는 바꾸지 않는다 (점프 중 등)
        src_ok = grounded & ~excl[frame_clip]
        dst_ok = src_ok & (np.arange(N) <= ends[frame_clip] - min_remain)  # B 가 최소 min_remain 프레임 남는 곳으로만
        q = np.asarray(self.ref_qpos)[:, 7:7 + self.n_ctrl]
        c = np.asarray(self.ref_contact)
        match = np.zeros((N, n_clip), np.int32)  # A 의 각 프레임에서 B 로 갈 때 들어갈 프레임 = 관절각·접지가 가장 비슷한 프레임
        for b in range(n_clip):
            cand = np.where(dst_ok & (frame_clip == b))[0]
            if len(cand) == 0:
                match[:, b] = starts[b]
                continue
            dist = ((q[:, None] - q[cand][None]) ** 2).sum(-1) + 0.5 * (c[:, None] != c[cand][None]).sum(-1)
            match[:, b] = cand[dist.argmin(1)]
        has_dst = np.array([(dst_ok & (frame_clip == b)).any() for b in range(n_clip)])
        pair_ok = (~excl[:, None]) & (~excl[None]) & has_dst[None] & ~np.eye(n_clip, dtype=bool)
        for a, b in forbid:  # held-out 조합은 학습에서 뺀다
            pair_ok[clips.index(a), clips.index(b)] = False
        self.src_ok, self.match, self.pair_ok = jnp.array(src_ok), jnp.array(match), jnp.array(pair_ok)

    def _ref(self, i: jax.Array, off: jax.Array) -> tuple[jax.Array, jax.Array]:
        """프레임 i 의 reference qpos·qvel 을 (dx, dy, dyaw) 만큼 옮긴다. off = 0 이면 원본 그대로."""
        q, v = self.ref_qpos[i], self.ref_qvel[i]
        c, s = jnp.cos(off[2]), jnp.sin(off[2])
        R = jnp.array([[c, -s], [s, c]])
        q = q.at[:2].set(R @ q[:2] + off[:2]).at[3:7].set(math.quat_mul(jnp.array([jnp.cos(off[2] / 2), 0.0, 0.0, jnp.sin(off[2] / 2)]), q[3:7]))
        v = v.at[:2].set(R @ v[:2])  # 선속도는 world frame, 각속도는 body frame 이라 그대로
        return q, v

    @staticmethod
    def _yaw(quat: jax.Array) -> jax.Array:
        w, x, y, z = quat
        return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _switch(self, data: mjx.Data, info: dict, due: jax.Array, clip_b: jax.Array) -> dict:
        """due 이면 목표를 clip_b 로 바꾼다: 들어갈 프레임 = match, B 를 로봇 현재 xy·yaw 에 맞춘다."""
        nf = self.match[info["frame"], clip_b] if self.switch_entry == "match" else self.clip_start[clip_b]
        dyaw = self._yaw(data.qpos[3:7]) - self._yaw(self.ref_qpos[nf, 3:7])
        c, s = jnp.cos(dyaw), jnp.sin(dyaw)
        dxy = data.qpos[:2] - jnp.array([[c, -s], [s, c]]) @ self.ref_qpos[nf, :2]
        info["rng"], k = jax.random.split(info["rng"])
        wait = jnp.round(jax.random.uniform(k, minval=self.switch_wait[0], maxval=self.switch_wait[1]) / self.dt).astype(jnp.int32)
        info["clip"] = jnp.where(due, clip_b, info["clip"])
        info["frame"] = jnp.where(due, nf, info["frame"])
        info["clip_end"] = jnp.where(due, self.clip_end[clip_b], info["clip_end"])
        info["ref_off"] = jnp.where(due, jnp.array([dxy[0], dxy[1], dyaw]), info["ref_off"])
        info["sw_left"] = jnp.where(due, self.switch_window, info["sw_left"])
        info["sw_step"] = jnp.where(due, (info["steps"] + wait).astype(jnp.int32), info["sw_step"])  # steps 는 wrapper 에서 float
        info["pose_steps_violated"] = jnp.where(due, 0.0, info["pose_steps_violated"])
        info["switches"] = info["switches"] + due.astype(jnp.int32)
        return info

    def _maybe_switch(self, data: mjx.Data, info: dict) -> dict:
        info["rng"], k = jax.random.split(info["rng"])
        p = jnp.where(self.pair_ok[info["clip"]], self.clip_p, 0.0)
        clip_b = jax.random.choice(k, len(self.clip_start), p=p / jnp.maximum(p.sum(), 1e-9))
        when = (info["frame"] >= info["clip_end"] - 1) & (info["sw_step"] < 10**6) if self.switch_at == "end" else (info["steps"] >= info["sw_step"])
        due = (when & self.src_ok[info["frame"]] & (p.sum() > 0) & (info["sw_left"] == 0)
               & (data.qpos[1] < 20.0))  # 계단 장면 (y 27~33 m) 근처에서는 바꾸지 않는다
        return self._switch(data, info, due, clip_b)

    def switch_now(self, state: State, clip_b: jax.Array) -> State:
        """평가용: 지금 프레임에서 바로 clip_b 로 바꾼다."""
        info = self._switch(state.data, dict(state.info), jnp.bool_(True), clip_b)
        info, _ = self.tracking_terms(state.data, info)
        return state.replace(info=info, obs=self._get_obs(state.data, info))

    def preprocess_mj_model(self, mj_model: mujoco.MjModel):
        mj_model.opt.impratio = self.impratio
        mj_model.opt.timestep = self.sim_dt
        mj_model.opt.iterations = self.solver_iterations
        mj_model.opt.ls_iterations = self.solver_ls_iterations
        # 발(sphere)과 바닥(plane)만 충돌. 넘어짐은 up 벡터로 종료
        geom_is_sphere = mj_model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
        geom_is_ground = (mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) | np.array(
            [mj_model.geom(i).name.startswith("stair_") for i in range(mj_model.ngeom)]
        )  # 바닥 + 계단 상자 (pd_replay.add_terrain)
        mj_model.geom_conaffinity[:] = geom_is_sphere | geom_is_ground
        mj_model.geom_contype[:] = geom_is_sphere | geom_is_ground

        assert mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and (mj_model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE).all()
        home = mj_model.keyframe("home")
        self.init_q = jnp.array(home.qpos)
        self.action_offset = jnp.array(home.ctrl)
        self._lowers, self._uppers = mj_model.jnt_range[1:].T
        self._soft_lowers = self._lowers * self.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self.soft_joint_pos_limit_factor
        self.action_size = self.n_ctrl
        self._base_idx = mj_model.jnt_bodyid[0]  # free joint 의 body = 몸통
        self.base_mass = mj_model.body_subtreemass[self._base_idx]
        self._foot_ids = jnp.array([mj_model.geom(f"{l}_foot").id for l in LEGS])

    def load_references(self, reference_files: list[str]):
        """클립 npz 를 한 배열로 이어 붙인다. 속도·발 위치는 q + base pose 에서 계산 (동료 조언: 그 외 센서값 불신)."""
        m, d = self.mj_model, mujoco.MjData(self.mj_model)
        feet = np.array(self._foot_ids)
        qpos_all, qvel_all, foot_all, contact_all, starts, ends = [], [], [], [], [], []
        n = 0
        for f in reference_files:
            z = np.load(f)
            assert np.isclose(float(z["dt"]), self.dt, rtol=1e-3), f"{f}: reference dt {float(z['dt'])} != ctrl_dt {self.dt}"
            qpos = np.c_[z["fix_base_pos"], z["fix_base_quat"], z["fix_q"]]
            T = len(qpos)
            qvel = np.zeros((T, m.nv))
            for t in range(T - 1):
                mujoco.mj_differentiatePos(m, qvel[t], self.dt, qpos[t], qpos[t + 1])
            qvel[-1] = qvel[-2]
            qvel = savgol_filter(qvel, SG_WIN, 2, axis=0)
            foot_b = np.zeros((T, 4, 3))  # base frame 의 foot geom 중심 (Peng 식 7: root 기준 상대 위치)
            for t in range(T):
                d.qpos[:] = qpos[t]
                mujoco.mj_kinematics(m, d)
                foot_b[t] = (d.geom_xpos[feet] - d.xpos[self._base_idx]) @ d.xmat[self._base_idx].reshape(3, 3)
            qpos_all.append(qpos)
            qvel_all.append(qvel)
            foot_all.append(foot_b)
            contact_all.append(z["fix_contact"].astype(np.float32))
            starts.append(n)
            ends.append(n + T)
            n += T
        self.ref_qpos = jnp.array(np.concatenate(qpos_all))
        self.ref_qvel = jnp.array(np.concatenate(qvel_all))
        self.ref_foot = jnp.array(np.concatenate(foot_all))
        self.ref_contact = jnp.array(np.concatenate(contact_all))
        self.clip_start = jnp.array(starts)
        self.clip_end = jnp.array(ends)
        assert (self.clip_end - self.clip_start > self.min_start_margin).all()
        self.reference_files = reference_files

    def reset(self, rng: jax.Array) -> State:
        # Reference State Initialization: 클립·시작 프레임 무작위, 로봇을 그 프레임 reference 상태로 둔다
        rng, k1, k2 = jax.random.split(rng, 3)
        clip = jax.random.choice(k1, len(self.clip_start), p=self.clip_p)
        start, end = self.clip_start[clip], self.clip_end[clip]
        frame = jax.random.randint(k2, (), start, end - self.min_start_margin)
        return self.reset_with_traj(rng, clip, frame)

    def reset_with_traj(self, rng: jax.Array, clip: jax.Array, frame: jax.Array) -> State:
        qpos = self.ref_qpos[frame]
        qvel = self.ref_qvel[frame]
        rng, k1, k2 = jax.random.split(rng, 3)
        n = self.n_ctrl
        qpos = qpos.at[7:7 + n].add(jax.random.uniform(k1, (n,), minval=-self.reset_qpos, maxval=self.reset_qpos))
        qvel = qvel.at[6:6 + n].add(jax.random.uniform(k2, (n,), minval=-self.reset_qvel, maxval=self.reset_qvel))
        data = make_data(self.mj_model, qpos=qpos, qvel=qvel, impl=self.impl, nconmax=self._nconmax, njmax=self._njmax)
        data = mjx.forward(self.mjx_model, data)

        rng, key1, key2, key3 = jax.random.split(rng, 4)
        rng, ks1, ks2 = jax.random.split(rng, 3)
        sw_step = jnp.where(
            jax.random.uniform(ks1) < self.switch_prob,
            jnp.round(jax.random.uniform(ks2, minval=self.switch_wait[0], maxval=self.switch_wait[1]) / self.dt),
            10**6,
        ).astype(jnp.int32)
        steps_until_next_kick = jnp.round(
            jax.random.uniform(key1, minval=self.kick_wait_times[0], maxval=self.kick_wait_times[1]) / self.dt
        ).astype(jnp.int32)
        kick_duration_seconds = jax.random.uniform(key2, minval=self.kick_durations[0], maxval=self.kick_durations[1])
        kick_mag = jax.random.uniform(key3, minval=self.velocity_kick[0], maxval=self.velocity_kick[1])

        info: dict[str, Union[jax.Array, float]] = {
            "rng": rng,
            "clip": clip,
            "frame": frame,
            "clip_end": self.clip_end[clip],
            "steps": 0,
            "state_is_nan": jnp.zeros(()),
            "prev_action": jnp.zeros(self.action_size),
            "pos_err": jnp.zeros(()),
            "orn_err": jnp.zeros(()),
            "up_vec": math.rotate(jnp.array([0.0, 0.0, 1.0]), data.xquat[self._base_idx]),
            "done/fellover": jnp.zeros(()),
            "done/pos_err": jnp.zeros(()),
            "done/orn_err": jnp.zeros(()),
            "done/clip_end": jnp.zeros(()),
            "pose_steps_violated": jnp.zeros(()),
            "steps_until_next_kick": steps_until_next_kick,
            "kick_duration_seconds": kick_duration_seconds,
            "kick_duration": jnp.round(kick_duration_seconds / self.dt).astype(jnp.int32),
            "steps_since_last_kick": 0,
            "kick_steps": 0,
            "kick_dir": jnp.zeros(3),
            "kick_mag": kick_mag,
            "ref_off": jnp.zeros(3),
            "sw_step": sw_step,
            "sw_left": jnp.zeros((), jnp.int32),
            "switches": jnp.zeros((), jnp.int32),
        }
        info, terms = self.tracking_terms(data, info)
        obs = self._get_obs(data, info)
        rewards = self.compute_reward(data=data, action=jnp.zeros(self.action_size), info=info, terms=terms)
        reward = self._total_reward(rewards)
        metrics = {**self.get_task_metrics(info, terms), **rewards, "reward": reward}
        return State(data, obs, reward, jnp.zeros(()), metrics, info)

    def tracking_terms(self, data: mjx.Data, info: dict) -> tuple[dict, dict]:
        """현재 프레임 reference 와의 오차 (Peng 식 5~9 의 제곱합들)."""
        i = info["frame"]
        ref_qpos, ref_qvel = self._ref(i, info["ref_off"])
        base_quat = data.qpos[3:7]
        rot = data.xmat[self._base_idx]
        foot_b = (data.geom_xpos[self._foot_ids] - data.xpos[self._base_idx]) @ rot
        dq = math.quat_mul(math.quat_inv(ref_qpos[3:7]), base_quat)
        terms = dict(
            pose=jnp.sum(jnp.square(ref_qpos[7:7 + self.n_ctrl] - data.qpos[7:7 + self.n_ctrl])),
            vel=jnp.sum(jnp.square(ref_qvel[6:6 + self.n_ctrl] - data.qvel[6:6 + self.n_ctrl])),
            ee=jnp.sum(jnp.square(self.ref_foot[i] - foot_b)),
            root_pos=jnp.sum(jnp.square(ref_qpos[:3] - data.qpos[:3])),
            root_orn=jnp.square(quat_angle(dq)),
            root_linvel=jnp.sum(jnp.square(ref_qvel[:3] - data.qvel[:3])),  # world frame (MuJoCo free joint)
            root_angvel=jnp.sum(jnp.square(ref_qvel[3:6] - data.qvel[3:6])),  # body frame
        )
        info["pos_err"] = jnp.sqrt(terms["root_pos"])
        info["orn_err"] = jnp.sqrt(terms["root_orn"])
        info["up_vec"] = math.rotate(jnp.array([0.0, 0.0, 1.0]), data.xquat[self._base_idx])
        return info, terms

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, self.clip_actions[0], self.clip_actions[1])
        state = self._maybe_apply_kick(state)
        data = step(
            model=self.mjx_model,
            data=state.data,
            action=self.action_offset.at[: self.n_ctrl].add(action * self.action_scale),
            n_substeps=self.n_substeps,
        )
        data_is_nan = jnp.any(jnp.array([jnp.isnan(x).any() for x in jax.tree_util.tree_leaves(data)]))
        data = jax.tree.map(lambda x, y: jnp.where(jnp.isnan(x), y, x), data, state.data)
        info = state.info
        info["state_is_nan"] = data_is_nan.astype(jnp.float32)
        info["steps"] += 1
        hold = (info["sw_left"] > 0) & (self.switch_entry == "start")  # start: 전환 구간에는 B 첫 프레임 유지
        info["frame"] = jnp.where(hold, info["frame"], jnp.minimum(info["frame"] + 1, info["clip_end"] - 1))
        info["sw_left"] = jnp.maximum(info["sw_left"] - 1, 0)
        info = self._maybe_switch(data, info)
        info, terms = self.tracking_terms(data, info)
        state = self.get_done(state.replace(data=data, info=info))
        state = state.replace(done=(state.done.astype(jnp.bool_) | data_is_nan).astype(jnp.float32))

        rewards = self.compute_reward(data=data, action=action, info=state.info, terms=terms)
        reward = self._total_reward(rewards)
        state.info["prev_action"] = action
        return state.replace(
            obs=self._get_obs(data, state.info),
            reward=reward,
            metrics={**self.get_task_metrics(state.info, terms), **rewards, "reward": reward},
        )

    def _total_reward(self, rewards: dict) -> jax.Array:
        reward = sum(rewards.values()) * self.dt
        if self.reward_nan_to_num is not None:
            reward = jnp.nan_to_num(
                reward, nan=self.reward_nan_to_num, posinf=self.reward_nan_to_num, neginf=self.reward_nan_to_num
            )
        if self.reward_clip_positive:
            reward = jnp.maximum(reward, 0.0)
        return reward

    def compute_reward(self, data: mjx.Data, action: jax.Array, info: dict, terms: dict) -> dict[str, jax.Array]:
        ramp = 1.0 - info["sw_left"] / self.switch_window if self.switch_ramp else 1.0  # 목표를 바꾼 직후 tracking 보상 0 → 1
        rewards = {
            # Peng 2020 식 (4)~(9)
            "track_pose": self.w["pose"] * jnp.exp(-self.k_pose * terms["pose"]),
            "track_vel": self.w["vel"] * jnp.exp(-self.k_vel * terms["vel"]),
            "track_ee": self.w["ee"] * jnp.exp(-self.k_ee * terms["ee"]),
            "track_root_pose": self.w["root_pose"]
            * jnp.exp(-self.k_root_pos * terms["root_pos"] - self.k_root_orn * terms["root_orn"]),
            "track_root_vel": self.w["root_vel"]
            * jnp.exp(-self.k_root_linvel * terms["root_linvel"] - self.k_root_angvel * terms["root_angvel"]),
        }
        rewards = {k: v * ramp for k, v in rewards.items()}
        rewards |= {
            "action_rate": jnp.sum((action - info["prev_action"]) ** 2) * self.penalty_action_rate,
            "torque": jnp.sum(data.actuator_force[: self.n_ctrl] ** 2) * self.penalty_torque,
            "joint_vel": jnp.sum(data.qvel[6:6 + self.n_ctrl] ** 2) * self.penalty_joint_vel,
            "joint_acc": jnp.sum(data.qacc[6:6 + self.n_ctrl] ** 2) * self.penalty_joint_acc,
        }
        joint_pos = data.qpos[7:]
        out_of_limits = -jnp.clip(joint_pos - self._soft_lowers, None, 0.0) + jnp.clip(joint_pos - self._soft_uppers, 0.0, None)
        rewards["joint_pos_limit"] = jnp.sum(out_of_limits) * self.penalty_joint_pos_limit
        rewards["termination"] = (
            info["done/fellover"] + info["done/pos_err"] + info["done/orn_err"] + info["state_is_nan"]
        ) * self.penalty_termination
        return rewards

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        rng = info["rng"]
        rot = data.xmat[self._base_idx]
        base_pos, base_quat = data.qpos[:3], data.qpos[3:7]

        def noisy(x, name):
            nonlocal rng
            key, rng = jax.random.split(rng)
            return x + jax.random.normal(key, x.shape) * self.obs_noise[name]

        obs_list = [
            noisy(rot.T @ jnp.array([0.0, 0.0, -1.0]), "gravity"),
            noisy(data.qpos[7:] - self.init_q[7:], "qpos"),
            noisy(data.qvel[6:], "qvel"),
            info["prev_action"],
            noisy(rot.T @ data.qvel[:3], "lin_vel"),
            noisy(data.qvel[3:6], "ang_vel"),  # free joint 각속도는 이미 body frame
        ]
        # 미래 reference 프레임 (클립 끝에서 멈춤): 관절각, 현재 base 기준 base 위치·자세, 발 위치, 접지
        for k in self.obs_time_indices:
            i = jnp.minimum(info["frame"] + k, info["clip_end"] - 1)
            ref, _ = self._ref(i, info["ref_off"])
            rel_quat = math.quat_mul(math.quat_inv(base_quat), ref[3:7])
            obs_list += [
                ref[7:] - self.init_q[7:],
                rot.T @ (ref[:3] - base_pos),
                math.quat_to_mat(rel_quat).reshape(-1),
                self.ref_foot[i].reshape(-1),
                self.ref_contact[i],
            ]
        return jnp.concatenate(obs_list)

    def get_task_metrics(self, info: dict, terms: dict) -> dict[str, jax.Array]:
        return {
            **{f"err/{k}": v for k, v in terms.items()},
            "done/fellover": info["done/fellover"],
            "done/pos_err": info["done/pos_err"],
            "done/orn_err": info["done/orn_err"],
            "done/clip_end": info["done/clip_end"],
            "switches": info["switches"].astype(jnp.float32),
        }

    def get_done(self, state: State) -> State:
        info = state.info
        end_of_episode = info["steps"] >= self.max_episode_length
        clip_end = info["frame"] >= info["clip_end"] - 1
        fell_over = info["up_vec"][2] < self.termination_fellover_threshold
        in_window = info["sw_left"] > 0  # 목표를 바꾼 직후에는 위치·자세 오차로 종료하지 않는다
        pos_v = (info["pos_err"] > self.termination_pos_err_threshold) & ~in_window
        orn_v = (info["orn_err"] > self.termination_orn_err_threshold) & ~in_window
        info["pose_steps_violated"] = jnp.where(pos_v | orn_v, info["pose_steps_violated"] + 1, jnp.zeros(()))
        held = info["pose_steps_violated"] >= self.termination_pose_consecutive_steps
        info["done/fellover"] = fell_over.astype(jnp.float32)
        info["done/pos_err"] = (pos_v & held).astype(jnp.float32)
        info["done/orn_err"] = (orn_v & held).astype(jnp.float32)
        info["done/clip_end"] = clip_end.astype(jnp.float32)
        done = end_of_episode | clip_end | fell_over | (pos_v & held) | (orn_v & held)
        return state.replace(done=done.astype(jnp.float32), info=info)

    def _maybe_apply_kick(self, state: State) -> State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jnp.pi * 2)
            return jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])

        def apply_kick(state: State) -> State:
            t = state.info["kick_steps"] * self.dt
            u_t = 0.5 * jnp.sin(jnp.pi * t / state.info["kick_duration_seconds"])
            force = u_t * self.base_mass * state.info["kick_mag"] / state.info["kick_duration_seconds"]
            xfrc_applied = jnp.zeros((self.mjx_model.nbody, 6))
            xfrc_applied = xfrc_applied.at[self._base_idx, :3].set(force * state.info["kick_dir"])
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc_applied))
            state.info["steps_since_last_kick"] = jnp.where(
                state.info["kick_steps"] >= state.info["kick_duration"], 0, state.info["steps_since_last_kick"]
            )
            state.info["kick_steps"] += 1
            return state

        def wait(state: State) -> State:
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_last_kick"] += 1
            trigger = state.info["steps_since_last_kick"] >= state.info["steps_until_next_kick"]
            state.info["kick_steps"] = jnp.where(trigger, 0, state.info["kick_steps"])
            state.info["kick_dir"] = jnp.where(trigger, gen_dir(rng), state.info["kick_dir"])
            return state.replace(data=state.data.replace(xfrc_applied=jnp.zeros((self.mjx_model.nbody, 6))))

        return jax.lax.cond(
            state.info["steps_since_last_kick"] >= state.info["steps_until_next_kick"], apply_kick, wait, state
        )
