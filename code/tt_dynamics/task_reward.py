"""복합 task 보상. TT 생성 guidance (미분) 와 MJX 실행 결과 채점에 같은 class 를 쓴다.

설계 방법: rewards / penalties 사전에 {항목: 가중치} 를 넣는다. 항목 x 는 r_x (보상) 또는 p_x (벌점) 메서드로 정의한다.
  total = Σ rewards[x] · r_x(traj) - Σ penalties[x] · p_x(traj)
traj = 토큰 [B,T,54] (episode 시작 기준, rollout_data 의 S 42 + A 12) 와 mask [B,T]. 모든 항목은 sample 당 값 [B].
"""

import math

import numpy as np
import torch

POS, Z, COS, SIN, GRAV, Q, A = slice(0, 2), 2, 3, 4, slice(5, 8), slice(14, 26), slice(42, 54)


class TrackReward:
    """원형 트랙을 가장 빨리 돈다. 출발 = 원점, +x 방향. 트랙 중심 = (0, radius) 이므로 반시계 (좌회전) 로 돈다."""

    def __init__(self, radius=1.0, half_width=0.3, min_height_frac=0.6, rewards=None, penalties=None):
        self.radius, self.half_width, self.min_height_frac = radius, half_width, min_height_frac
        self.center = (0.0, radius)
        self.rewards = rewards if rewards is not None else {"progress": 1.0}
        self.penalties = penalties if penalties is not None else {"lane": 20.0, "fall": 20.0, "action_rate": 0.05}

    # ---- 보상 ----
    def r_progress(self, x, m):
        """트랙 중심 기준 반시계 회전각 합 (rad). 2π = 1 바퀴."""
        rel = x[..., POS] - x.new_tensor(self.center)
        a, b = rel[:, :-1], rel[:, 1:]
        dth = torch.atan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1))
        return (dth * (m[:, 1:] & m[:, :-1])).sum(1)

    # ---- 벌점 ----
    def p_lane(self, x, m):
        """트랙 폭 (radius ± half_width) 을 벗어난 거리² 의 평균 (m²)."""
        r = (x[..., POS] - x.new_tensor(self.center)).norm(dim=-1)
        out = torch.relu((r - self.radius).abs() - self.half_width) ** 2
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

    def p_fall(self, x, m):
        """몸통 기울기 (1 + body 중력 z, 수직 = 0) + 출발 높이의 min_height_frac 아래로 내려간 높이² 의 평균."""
        # Diffusion predicts three unconstrained gravity channels. Interpret
        # their DIRECTION as body orientation and penalize invalid magnitude.
        # A z component < -1 must not create a negative fall penalty / reward.
        gravity = x[..., GRAV]
        magnitude = torch.linalg.vector_norm(gravity, dim=-1)
        tilt = (1 + gravity[..., 2] / magnitude.clamp_min(1e-6)).clamp_min(0)
        gravity_invalid = (magnitude - 1).square()
        low = torch.relu(self.min_height_frac * x[:, :1, Z] - x[..., Z]) ** 2
        return ((tilt + gravity_invalid + low) * m).sum(1) / m.sum(1).clamp(min=1)

    def p_action_rate(self, x, m):
        """연속 step 목표 관절각 변화² 의 평균 (rad²)."""
        d = ((x[:, 1:, A] - x[:, :-1, A]) ** 2).sum(-1) * (m[:, 1:] & m[:, :-1])
        return d.sum(1) / m.sum(1).clamp(min=1)

    # ---- 합산 ----
    def terms(self, x, m):
        out = {f"r_{k}": getattr(self, f"r_{k}")(x, m) for k in self.rewards}
        out.update({f"p_{k}": getattr(self, f"p_{k}")(x, m) for k in self.penalties})
        return out

    def __call__(self, x, m):
        t = self.terms(x, m)
        total = sum(w * t[f"r_{k}"] for k, w in self.rewards.items()) - sum(w * t[f"p_{k}"] for k, w in self.penalties.items())
        return total, t

    def describe(self):
        return dict(task="circle_track", radius=self.radius, half_width=self.half_width, center=self.center,
                    rewards=self.rewards, penalties=self.penalties, min_height_frac=self.min_height_frac,
                    laps_per_rad=1 / (2 * math.pi))

    # 진행량 표시 단위: r_progress · unit_scale = unit
    unit, unit_scale = "바퀴", 1 / (2 * math.pi)

    def guide_lines(self):
        """그림·영상용 기준선 [(N,2) s0 frame, 실선 여부]: 중심선 + 폭 경계 2 개."""
        th = np.linspace(0, 2 * np.pi, 120)
        return [(np.stack([r * np.sin(th), self.radius - r * np.cos(th)], 1), r == self.radius)
                for r in (self.radius, self.radius - self.half_width, self.radius + self.half_width)]


class ZigzagReward(TrackReward):
    """+x 로 빠르게 전진하면서 y 는 삼각파 중심선 (진폭 amp, 주기 period m) 을 따른다. 출발 = 원점, +x 방향.

    중심선 y_ref(x) = amp · (2/π) · asin(sin(2π x / period)) — x = 0 에서 0, period/4 마다 꺾임. 꺾임에서도 미분 가능.
    progress = +x 이동 거리 (m). lane = 중심선에서 y 방향 거리가 half_width 를 넘은 거리² 평균. fall·action_rate 는 원형 트랙과 같다.
    """

    unit, unit_scale = "m", 1.0

    def __init__(self, amp=0.3, period=2.0, half_width=0.15, **kw):
        super().__init__(half_width=half_width, **kw)
        self.amp, self.period = amp, period

    def y_ref(self, px):
        return self.amp * (2 / math.pi) * torch.asin(torch.sin(2 * math.pi * px / self.period).clamp(-1 + 1e-6, 1 - 1e-6))

    def r_progress(self, x, m):
        dx = x[:, 1:, 0] - x[:, :-1, 0]
        return (dx * (m[:, 1:] & m[:, :-1])).sum(1)

    def p_lane(self, x, m):
        out = torch.relu((x[..., 1] - self.y_ref(x[..., 0])).abs() - self.half_width) ** 2
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

    def describe(self):
        return dict(task="zigzag", amp=self.amp, period=self.period, half_width=self.half_width,
                    rewards=self.rewards, penalties=self.penalties, min_height_frac=self.min_height_frac, unit=self.unit)

    def guide_lines(self, length=6.0):
        px = torch.linspace(0, length, 300)
        y = self.y_ref(px).numpy()
        px = px.numpy()
        return [(np.stack([px, y + o], 1), o == 0) for o in (0.0, -self.half_width, self.half_width)]


class StraightReward(TrackReward):
    """Forward progress in a straight corridor, with the same stability and action regularizers."""

    unit, unit_scale = "m", 1.0

    def __init__(self, half_width=0.15, **kw):
        super().__init__(half_width=half_width, **kw)

    r_progress = ZigzagReward.r_progress

    def p_lane(self, x, m):
        outside = torch.relu(x[..., 1].abs() - self.half_width) ** 2
        return (outside * m).sum(1) / m.sum(1).clamp(min=1)

    def describe(self):
        return dict(task="straight", half_width=self.half_width,
                    rewards=self.rewards, penalties=self.penalties,
                    min_height_frac=self.min_height_frac, unit=self.unit)

    def guide_lines(self, length=6.0):
        return [(np.array([[0.0, y], [length, y]]), y == 0.0)
                for y in (0.0, -self.half_width, self.half_width)]


if __name__ == "__main__":
    # 점검: 반지름 1 m 원을 반시계로 반 바퀴 돌면 progress = π, lane = 0
    th = torch.linspace(0, math.pi, 100)
    x = torch.zeros(1, 100, 54)
    x[0, :, 0], x[0, :, 1] = torch.sin(th), 1 - torch.cos(th)
    x[0, :, 2], x[0, :, 7] = 0.3, -1.0
    m = torch.ones(1, 100, dtype=torch.bool)
    total, t = TrackReward()(x, m)
    assert abs(t["r_progress"].item() - math.pi) < 1e-4 and t["p_lane"].item() < 1e-9 and t["p_fall"].item() < 1e-9, t
    x[0, :, 0], x[0, :, 1] = 1.5 * torch.sin(th), 1 - 1.5 * torch.cos(th)  # 반지름 1.5 m: 폭 0.3 m 초과분 0.2 m
    assert abs(TrackReward()(x, m)[1]["p_lane"].item() - 0.04) < 1e-4
    print("ok", {k: round(v.item(), 4) for k, v in t.items()}, "total", round(total.item(), 4))
    # 점검: 중심선을 따라 4 m 전진하면 progress = 4, lane = 0. 중심선 +0.25 m 평행이동 = 폭 0.15 초과분 0.1 m
    zz = ZigzagReward()
    px = torch.linspace(0, 4, 400)
    x = torch.zeros(1, 400, 54)
    x[0, :, 0], x[0, :, 1], x[0, :, 2], x[0, :, 7] = px, zz.y_ref(px), 0.3, -1.0
    m = torch.ones(1, 400, dtype=torch.bool)
    t = zz.terms(x, m)
    assert abs(t["r_progress"].item() - 4) < 1e-4 and t["p_lane"].item() < 1e-9, t
    x[0, :, 1] += 0.25
    assert abs(zz.terms(x, m)["p_lane"].item() - 0.01) < 1e-4
    print("zigzag ok")
