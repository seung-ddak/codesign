"""2 단 reward — **결과 기반 하나** (D22 개정, 2026-09-21).

    낙상 = TT 가 지목한 링크가 바닥에 접촉  또는  베이스가 약 84° 기울어짐  → 즉시 종료
    점수 = 종료 전까지의 전진 진척도 (목표 1.0 m 로 정규화)
    성공 = 종료 없이 목표 거리 도달

근거:
- **접촉 기반 즉시 종료**가 이 분야 관행이다. legged_gym 은 지정 링크 접촉력 > 1 N 이면 reset
  (`legged_robot.py:168`), TT 는 `penalty_bad_contact_pairs` 로 calf_motor ×4 + gripper 를 지목하고
  `terminate_on_bad_contact: true` 로 종료시킨다. 둘 다 "에피소드 평균" 같은 누적 지표를 쓰지 않는다.
- 링크 목록은 **TT 의 것을 그대로** 쓴다. 우리가 임의로 정하면 근거가 없고, 실측상 정상 보행
  중에도 calf·팔 링크가 2~10 % 스친다 (`contact_audit.py`) — "발 아닌 전부" 로 하면 멀쩡한 보행도 죽는다.
- 기울기 임계 0.1 (≈84°) 도 TT 값 그대로 (`termination_fellover_threshold`). 백스톱 역할.
- 이 구조가 §5.21·§5.22 의 병리를 원천 차단한다 — 포복하면 첫 접촉에서 끝나 거리를 쌓을 수 없다.

미분 불가능하다 (접촉·종료는 이산 사건) → DSG 로는 guide 할 수 없다 = 2 단의 성립 근거.
"""

from __future__ import annotations

import numpy as np

GOAL_M = 1.0                 # 목표 전진 거리 (사용자 확정 2026-09-21)
FELLOVER_THRESHOLD = 0.1     # up_vec·z. TT `termination_fellover_threshold`
# TT `penalty_bad_contact_pairs` 가 지목한 링크 (floor 와의 접촉만 본다)
TERMINATION_BODY_TOKENS = ("calf_motor", "gripper_collider")


class WalkReward:
    """step 마다 먹이고 끝에 요약을 받는다."""

    def __init__(self, env, base_body_name: str = "base", goal: float = GOAL_M):
        import mujoco

        self._mj = mujoco
        self.env = env
        self.m = env.m
        self.goal = goal
        self.base_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        self.start_x = float(env.d.qpos[0])
        self.max_x = self.start_x
        self.steps = 0
        self.terminated = False
        self.term_cause = None
        self.bad_contact_steps = 0
        self.upright_sum = 0.0

    def _bad_contact_body(self):
        d, m, mj = self.env.d, self.m, self._mj
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if m.geom_type[g1] == mj.mjtGeom.mjGEOM_PLANE:
                g = g2
            elif m.geom_type[g2] == mj.mjtGeom.mjGEOM_PLANE:
                g = g1
            else:
                continue
            gname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
            bname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            tag = f"{gname} {bname}"
            for tok in TERMINATION_BODY_TOKENS:
                if tok in tag:
                    return tag
        return None

    def step(self) -> bool:
        """종료됐으면 True."""
        d = self.env.d
        self.steps += 1
        self.max_x = max(self.max_x, float(d.qpos[0]))

        up_z = float(d.xmat[self.base_bid].reshape(3, 3)[2, 2])
        self.upright_sum += up_z

        hit = self._bad_contact_body()
        if hit is not None:
            self.bad_contact_steps += 1
        if not self.terminated:
            if hit is not None:
                self.terminated, self.term_cause = True, f"contact:{hit}"
            elif up_z < FELLOVER_THRESHOLD:
                self.terminated, self.term_cause = True, "fellover"
        return self.terminated

    @property
    def distance(self) -> float:
        return self.max_x - self.start_x

    def summary(self) -> dict:
        dist = self.distance
        return dict(
            score=round(min(dist, self.goal) / self.goal, 4),   # π_φ 가 보는 값 [0,1]
            success=bool(dist >= self.goal and not self.terminated),
            distance=round(dist, 4),
            steps=self.steps,
            terminated=self.terminated,
            term_cause=self.term_cause,
            upright_mean=round(self.upright_sum / max(self.steps, 1), 4),
            bad_contact_steps=self.bad_contact_steps,
        )
