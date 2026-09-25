"""D22 reward — TT 의 RL 학습 reward 를 **평가 rollout(CPU TrackEnv)** 위에서 계산한다.

원본은 `t2/mjx_env/track_env.py:616` + `config/env/mjx_umi_on_legs.yaml` 이고 MJX 학습용이다.
평가는 CPU `TrackEnv` 로 도는데 여기엔 reward 가 없으므로, **같은 항·같은 계수**로 다시 쓴다.

DSG 가 이 reward 를 guide 할 수 없는 이유가 곧 2 단의 근거다 — `bad_contact` 와 `termination`
은 접촉·종료 **사건**이라 모델이 토큰에서 해석적으로 예측할 수 없고, 따라서 ∇R̂ 이 없다.

계수 (mjx_umi_on_legs.yaml 그대로):
    exp_pose_err +4.0 (pos_sigma 0.01, orn_sigma 0.5)   유일한 양의 항
    upright −1.0 · bad_contact −1.0 · default_qpos −1.0 · joint_pos_limit −1.0
    action_rate −0.002 · joint_vel −0.002 · joint_acc −5e-5 · torque −5e-5
    termination −10.0      종료: fellover 0.1 / pos_err 0.2 / orn_err 1.0
"""

from __future__ import annotations

import numpy as np

W = dict(
    exp_pose_err=4.0,
    orientation=-1.0,
    bad_contact=-1.0,
    default_qpos=-1.0,
    joint_pos_limit=-1.0,
    action_rate=-0.002,
    joint_vel=-0.002,
    joint_acc=-5e-5,
    torque=-5e-5,
    termination=-10.0,
)
POS_SIGMA, ORN_SIGMA = 0.01, 0.5
FELLOVER_THRESHOLD = 0.1          # mjx 쪽 값. 평가 env 기본 0.5 보다 훨씬 엄격하다
POS_ERR_THRESHOLD, ORN_ERR_THRESHOLD = 0.2, 1.0
# 추종 오차 종료는 **연속 50 step** 위반해야 발동한다 (mjx track_env.py:798).
# 이걸 빼면 초기 오차 한 번에 바로 종료돼 reward 가 전부 −10 으로 깔린다.
POSE_CONSECUTIVE_STEPS = 50
SOFT_LIMIT_FACTOR = 0.95
# 바닥에 닿으면 안 되는 링크 (mjx_umi_on_legs.yaml 의 penalty_bad_contact_pairs 와 같은 취지):
# 발이 아닌 모든 링크.
FOOT_TOKEN = "foot"


class D22Reward:
    """한 에피소드를 step 마다 먹이고 마지막에 합을 받는다."""

    def __init__(self, env, runner=None, base_body_name: str = "base"):
        """runner 를 주면 동역학 항(qacc·qvel·actuator_force)을 **MJX 쪽에서** 읽는다.

        평가는 CPU MuJoCo 로 돌지만 D22 는 MJX 학습 환경에서 정의된 식이다. 접촉 솔버가
        달라 CPU 의 qacc 는 학습 때 값과 크기가 다르고, 그대로 쓰면 `joint_acc` 항이
        점수의 80 % 를 먹는다 (step 당 −13.9 vs 추종 +3.5 — 이 비율이 학습 때도 같았다면
        expert 는 움직이기를 포기했어야 한다). `RLEnvRunner` 는 매 step
        `mjx_forward(mjx_env.mjx_model, ...)` 를 돌리므로 **학습과 같은 솔버로 계산된**
        양이 `runner.state.data` 에 들어 있다. 그걸 읽어 정합을 맞춘다.
        """
        import mujoco

        self.env = env
        self.runner = runner
        m = env.m
        self.m = m
        self._mj = mujoco
        self.base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        self.qpos_is_free = np.zeros(m.nq, dtype=bool)
        self.qvel_is_free = np.zeros(m.nv, dtype=bool)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.qpos_is_free[m.jnt_qposadr[j] : m.jnt_qposadr[j] + 7] = True
                self.qvel_is_free[m.jnt_dofadr[j] : m.jnt_dofadr[j] + 6] = True
        self.init_q = env.d.qpos.copy()
        # default_qpos 가중치는 **모델의 `penalty_default_qpos_weight` 키프레임**에서 온다
        # (mjx track_env.py:257). 부품마다 다른 값이고 대부분 0 이다. 키프레임이 없을 때만
        # free joint 7 개 0 + 나머지 1.0 로 폴백한다 (같은 파일 264).
        self.default_qpos_w = None
        for k in range(m.nkey):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or "").endswith(
                "penalty_default_qpos_weight"
            ):
                self.default_qpos_w = np.abs(np.array(m.key_qpos[k], dtype=float))
                break
        if self.default_qpos_w is None:
            self.default_qpos_w = np.zeros(m.nq)
            self.default_qpos_w[~self.qpos_is_free] = 1.0
        # 관절 한계는 **soft** 한계를 쓴다 (soft_joint_pos_limit_factor 0.95)
        rng = m.jnt_range[~(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)]
        self.lo, self.hi = rng[:, 0] * SOFT_LIMIT_FACTOR, rng[:, 1] * SOFT_LIMIT_FACTOR
        self.prev_action = None
        self.terms = {k: 0.0 for k in W}
        self.steps = 0
        self.terminated = False
        self.pose_steps_violated = 0
        self.term_cause = None

    def _bad_contact(self) -> int:
        d, m = self.env.d, self.m
        n = 0
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if m.geom_type[g1] == self._mj.mjtGeom.mjGEOM_PLANE:
                g = g2
            elif m.geom_type[g2] == self._mj.mjtGeom.mjGEOM_PLANE:
                g = g1
            else:
                continue
            bn = self._mj.mj_id2name(m, self._mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if FOOT_TOKEN not in bn:
                n += 1
        return n

    def step(self, action: np.ndarray) -> None:
        env, m, d = self.env, self.m, self.env.d
        self.steps += 1

        pos_err = np.asarray(env.curr_pos_err, dtype=float)
        orn_err = np.asarray(env.curr_orn_err, dtype=float)
        self.terms["exp_pose_err"] += float(
            np.exp(-(orn_err ** 2) / ORN_SIGMA - (pos_err ** 2) / POS_SIGMA).sum()
        )

        up = d.xmat[self.base_bid].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        self.terms["orientation"] += float(np.sum(up[:2] ** 2))
        self.terms["bad_contact"] += float(self._bad_contact())

        qpos = d.qpos[~self.qpos_is_free]
        self.terms["default_qpos"] += float(
            np.sum((d.qpos - self.init_q) ** 2 * self.default_qpos_w)
        )
        oob = -np.clip(qpos - self.lo, None, 0.0) + np.clip(qpos - self.hi, 0.0, None)
        self.terms["joint_pos_limit"] += float(np.sum(oob))

        if self.prev_action is not None:
            self.terms["action_rate"] += float(np.sum((action - self.prev_action) ** 2))
        self.prev_action = np.array(action, dtype=float)

        # 동역학 항은 MJX 쪽 값으로 (위 docstring 참조). runner 가 없으면 CPU 폴백.
        dyn = getattr(getattr(self.runner, "state", None), "data", None)
        if dyn is not None:
            af = np.asarray(dyn.actuator_force)
            qv = np.asarray(dyn.qvel)[~self.qvel_is_free]
            qa = np.asarray(dyn.qacc)[~self.qvel_is_free]
        else:
            af, qv, qa = d.actuator_force, d.qvel[~self.qvel_is_free], d.qacc[~self.qvel_is_free]
        self.terms["torque"] += float(np.sum(af ** 2))
        self.terms["joint_vel"] += float(np.sum(qv ** 2))
        self.terms["joint_acc"] += float(np.sum(qa ** 2))

        fell = float(up[2]) < FELLOVER_THRESHOLD
        pos_v = bool(np.any(pos_err > POS_ERR_THRESHOLD))
        orn_v = bool(np.any(orn_err > ORN_ERR_THRESHOLD))
        self.pose_steps_violated = self.pose_steps_violated + 1 if (pos_v or orn_v) else 0
        pose_done = self.pose_steps_violated >= POSE_CONSECUTIVE_STEPS
        if fell or pose_done:
            self.terminated = True
            self.term_cause = "fellover" if fell else "pose"
            self.terms["termination"] += 1.0

    def total(self) -> float:
        return float(sum(W[k] * v for k, v in self.terms.items()))

    def summary(self) -> dict:
        return dict(
            reward=self.total(),
            steps=self.steps,
            terminated=self.terminated,
            term_cause=self.term_cause,
            **{k: round(W[k] * v, 3) for k, v in self.terms.items()},
        )
