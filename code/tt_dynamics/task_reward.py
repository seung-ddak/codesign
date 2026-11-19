"""복합 task 보상. TT 생성 guidance (미분) 와 MJX 실행 결과 채점에 같은 class 를 쓴다.

설계 방법: rewards / penalties 사전에 {항목: 가중치} 를 넣는다. 항목 x 는 r_x (보상) 또는 p_x (벌점) 메서드로 정의한다.
  total = Σ rewards[x] · r_x(traj) - Σ penalties[x] · p_x(traj)
traj = 토큰 [B,T,54] (episode 시작 기준, rollout_data 의 S 42 + A 12) 와 mask [B,T]. 모든 항목은 sample 당 값 [B].
"""

import math
import os

import numpy as np
import torch

POS, Z, COS, SIN, GRAV, Q, A = slice(0, 2), 2, 3, 4, slice(5, 8), slice(14, 26), slice(42, 54)


class TrackReward:
    """원형 트랙을 가장 빨리 돈다. 출발 = 원점, +x 방향. 트랙 중심 = (0, radius) 이므로 반시계 (좌회전) 로 돈다."""

    def __init__(self, radius=1.0, half_width=0.3, min_height_frac=0.6, rewards=None, penalties=None):
        self.radius, self.half_width, self.min_height_frac = radius, half_width, min_height_frac
        self.center = (0.0, radius)
        # TT_REWARD_PROFILE=v2 (2026-11-10): 진행 ×10 (1 m 당 10, 원형은 반지름 1 m 라 1 rad = 1 m),
        # 기울기·높이 감점 ×2 (실제 넘어짐은 생존 감점 ×20 이 맡는다). 기본 (v1) 은 이전과 같다.
        v2 = os.environ.get("TT_REWARD_PROFILE", "v1") == "v2"
        self.rewards = rewards if rewards is not None else {"progress": 10.0 if v2 else 1.0}
        self.penalties = penalties if penalties is not None else {"lane": 20.0, "fall": 2.0 if v2 else 20.0, "action_rate": 0.05, "survival": 20.0,
                                                                  "heading": self.heading_weight}

    # 2026-11-19 사용자 결정 (쓸데없는 회전 제거): 몸 방향이 과제 진행 방향에서 벗어난 정도 (1 - cos) 의 평균에 감점.
    # 0 = 늘 진행 방향, 1 = 옆, 2 = 반대. 가중치 5 = 10 s 내내 반대를 보면 -10, 제자리 한 바퀴 (1 s) 면 약 -0.5.
    heading_weight = 5.0

    def heading_ref(self, x):
        """[B,T,2] 진행 방향 단위벡터 (출발 기준 xy). 원형 트랙 = 반시계 접선."""
        rel = x[..., POS] - x.new_tensor(self.center)
        t = torch.stack([-rel[..., 1], rel[..., 0]], -1)
        return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def p_heading(self, x, m):
        """몸 방향 (특징 cos·sin = 출발 기준 yaw) 과 heading_ref 사이 1 - cos 의 평균 (0~2)."""
        n = torch.sqrt(x[..., COS] ** 2 + x[..., SIN] ** 2).clamp_min(1e-6)
        ref = self.heading_ref(x)
        out = 1 - (x[..., COS] * ref[..., 0] + x[..., SIN] * ref[..., 1]) / n
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

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

    def p_survival(self, x, m):
        """10 초 중 넘어진 뒤 남은 시간 비율 (0 = 끝까지 생존, 1 = 바로 넘어짐).

        넘어짐 판정은 실제 MuJoCo 채점과 같다: 직립도 (몸통 위쪽 축의 수직 성분) <= 0.5 또는 몸통 높이 <= 0.07 m.
        TT 유도에서 미분되도록 경계를 sigmoid 로 부드럽게 하고, 한 번 넘어지면 이후는 모두 넘어진 것으로 센다.
        실제 실행은 넘어지면 기록이 끝나므로 기록이 없는 step (m = False) 도 넘어진 것으로 센다.
        """
        gravity = x[..., GRAV]
        upright = -gravity[..., 2] / torch.linalg.vector_norm(gravity, dim=-1).clamp_min(1e-6)
        fallen = torch.maximum(torch.sigmoid((0.5 - upright) / 0.02), torch.sigmoid((0.07 - x[..., Z]) / 0.005))
        alive = torch.cumprod((1 - fallen) * m.to(x.dtype), dim=1)
        return 1 - alive.mean(1)

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

    def heading_ref(self, x):
        """중심선 y_ref 의 기울기 방향 (+x 쪽)."""
        px = x[..., 0].detach()
        slope = (self.y_ref(px + 0.01) - self.y_ref(px - 0.01)) / 0.02
        t = torch.stack([torch.ones_like(slope), slope], -1)
        return t / t.norm(dim=-1, keepdim=True)

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
    """최대 속도 직진: 통로 안에서 10 초 동안 +x 로 최대한 빨리 간다. 진행 점수 = 전진 거리 (m), 표시 = 10 초 평균 속도 (m/s)."""

    unit, unit_scale = "m/s", 0.1

    def __init__(self, half_width=0.15, **kw):
        super().__init__(half_width=half_width, **kw)

    r_progress = ZigzagReward.r_progress

    def heading_ref(self, x):
        """+x."""
        ref = torch.zeros_like(x[..., POS])
        ref[..., 0] = 1
        return ref

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


class SlalomReward(ZigzagReward):
    """Pole slalom. Differentiable guidance uses the same route as physical scoring."""

    pole_radius = 0.06
    pole_height = 0.82
    collision_penalty = 20.0

    # 2026-11-19 사용자 요청 "기둥에 닿으면 패널티": 몸통 중심 원 (반경 0.15) 대신 몸통 방향 직사각형 (다리 포함) 으로 판정.
    # s11z MPC 실제 MuJoCo 폴 접촉 7 건은 모두 몸통 중심에서 폴 중심까지 0.36~0.45 m (이전 판정 0.21 m 밖 → 7 건 모두 감점 0).
    # 반길이 0.35 · 반폭 0.20 이면 160 개 실제 궤적에서 접촉 7/7 판정 (접촉 없는 폴 통과 15 건도 판정: 보수적).
    foot_half_length = 0.35
    foot_half_width = 0.20

    def __init__(self):
        super().__init__(half_width=0.22)
        self.penalties = dict(self.penalties, pole=self.collision_penalty)
        self.poles = [(x, 0.0) for x in (1.0, 2.3, 3.6, 4.9)]
        self.gates = []
        for index, (x, _) in enumerate(self.poles):
            y = 0.42 * (-1)**index
            self.gates.extend([(x - 0.30, y), (x + 0.20, y)])
        self.gates.append((6.0, 0.0))
        self.route = [(0.0, 0.0), *self.gates]

    def y_ref(self, px):
        route = px.new_tensor(self.route)
        x, y = route[:, 0].contiguous(), route[:, 1]
        i = torch.searchsorted(x, px.contiguous()).clamp(1, len(x) - 1)
        fraction = ((px - x[i - 1]) / (x[i] - x[i - 1])).clamp(0, 1)
        return y[i - 1] + fraction * (y[i] - y[i - 1])

    def r_progress(self, x, m):
        # Positive forward progress loses credit away from the route. Reverse
        # motion always costs progress, even when it happens off the route.
        px = x[..., 0].clamp(max=self.gates[-1][0])
        dx = px[:, 1:] - px[:, :-1]
        xy = (x[:, 1:, :2] + x[:, :-1, :2]) / 2
        lateral = xy[..., 1] - self.y_ref(xy[..., 0])
        weight = torch.exp(-(lateral / self.half_width).square())
        return ((dx.clamp_min(0) * weight + dx.clamp_max(0)) *
                (m[:, 1:] & m[:, :-1])).sum(1)

    def p_pole(self, x, m):
        """폴마다 가장 깊이 닿은 순간의 충돌 정도 (0~1) 의 합. 몸통 방향 직사각형 (반길이 foot_half_length, 반폭 foot_half_width) 과
        폴 원기둥 사이 거리가 0 이하면 1 에 가깝다. 방향 = 특징 cos·sin (출발 기준 yaw), 위치 = 특징 xy (출발 기준)."""
        poles = x.new_tensor(self.poles)
        rel = poles - x[..., None, POS]
        n = torch.sqrt(x[..., COS] ** 2 + x[..., SIN] ** 2).clamp_min(1e-6)
        c, s = (x[..., COS] / n)[..., None], (x[..., SIN] / n)[..., None]
        lon = (c * rel[..., 0] + s * rel[..., 1]).abs() - self.foot_half_length
        lat = (-s * rel[..., 0] + c * rel[..., 1]).abs() - self.foot_half_width
        outside = torch.sqrt(lon.clamp_min(0) ** 2 + lat.clamp_min(0) ** 2 + 1e-8)
        dist = outside + torch.maximum(lon, lat).clamp_max(0) - self.pole_radius
        hit = torch.sigmoid(-dist / 0.02) * m[..., None].to(x.dtype)
        return hit.amax(1).sum(-1)

    def describe(self):
        return dict(task="pole_slalom", poles=self.poles, gates=self.gates,
                    route=self.route, half_width=self.half_width,
                    pole_radius=self.pole_radius, pole_height=self.pole_height,
                    pole_footprint=(self.foot_half_length, self.foot_half_width), heading_weight=self.heading_weight,
                    collision_penalty_per_pole=self.collision_penalty,
                    rewards=self.rewards, penalties=self.penalties, unit=self.unit)

    def gate_crossings(self, xy):
        """Return the fractional frame indices of ordered forward gate crossings."""
        crossings = []
        for index, (start, end) in enumerate(zip(xy[:-1], xy[1:])):
            while len(crossings) < len(self.gates):
                gx, gy = self.gates[len(crossings)]
                if not start[0] < gx <= end[0]:
                    break
                fraction = (gx - start[0]) / (end[0] - start[0])
                y = start[1] + fraction * (end[1] - start[1])
                if abs(y - gy) > self.half_width:
                    break
                crossings.append(float(index + fraction))
        return crossings


class Slalom3Reward(SlalomReward):
    """2026-11-19 사용자 결정: 기둥 3 개 (x = 1.0, 2.3, 3.6) 를 피해 끝 게이트 (x = 4.7, y = 0) 까지 가장 빨리.
    게이트·경로·감점은 SlalomReward 와 같고 4 번째 기둥만 뺀다. 끝 = 게이트 7 개를 순서대로 통과."""

    def __init__(self):
        super().__init__()
        self.poles = self.poles[:3]
        self.gates = self.gates[:6] + [(self.poles[-1][0] + 1.1, 0.0)]
        self.route = [(0.0, 0.0), *self.gates]

    def guide_lines(self, length=None):
        return super().guide_lines(self.gates[-1][0] if length is None else length)



class Poles3ForwardReward(SlalomReward):
    """2026-11-19 사용자 결정: 지그재그 경로를 따르지 않고, 기둥 3 개 (x = 1.0, 2.3, 3.6, y = 0) 에 닿지 않으면서 끝선 x = 4.7 m 까지 가장 빨리.
    경로·통로 (lane) 없음. 진행 = +x 전진 (끝선에서 자름, 뒤로 가면 감점). 몸 방향 기준 = +x. 폴 감점 = SlalomReward 직사각형 판정 (가중치 20)."""

    unit, unit_scale = "m", 1.0
    finish_x = 4.7

    def __init__(self):
        super().__init__()
        self.poles = self.poles[:3]
        self.gates, self.route = [], [(0.0, 0.0), (self.finish_x, 0.0)]
        self.penalties = {k: v for k, v in self.penalties.items() if k != "lane"}

    def r_progress(self, x, m):
        px = x[..., 0].clamp(max=self.finish_x)
        return ((px[:, 1:] - px[:, :-1]) * (m[:, 1:] & m[:, :-1])).sum(1)

    heading_ref = StraightReward.heading_ref

    def describe(self):
        return dict(task="poles3_forward", poles=self.poles, finish_x=self.finish_x,
                    pole_radius=self.pole_radius, pole_height=self.pole_height,
                    pole_footprint=(self.foot_half_length, self.foot_half_width), heading_weight=self.heading_weight,
                    collision_penalty_per_pole=self.collision_penalty,
                    rewards=self.rewards, penalties=self.penalties, unit=self.unit)

    def guide_lines(self, length=None):
        """출발선 · 끝선 (y ±1 m)."""
        return [(np.array([[0.0, -1.0], [0.0, 1.0]]), False), (np.array([[self.finish_x, -1.0], [self.finish_x, 1.0]]), True)]


class CorridorPolesReward(Poles3ForwardReward):
    """2026-11-19 사용자 결정 (장애물 피하기 검증): 직선 통로 (몸통 중심 |y| <= 0.40 m) 안에 좌우로 엇갈린 기둥 3 개
    (x 1.0 / 2.3 / 3.6, y +0.15 / -0.15 / +0.15). 기둥과 같은 쪽 틈 (0.40 - 0.15 = 0.25 < 몸 반폭 0.20 + 기둥 반지름 0.06) 은 막혀
    기둥마다 반대편으로 비켜 가야 한다. 끝선 x 4.7 m 까지 가장 빨리.
    통로 감점 = 통로 밖에 있던 시간 비율 (sigmoid 판정) × 20, 기둥 감점 20, 진행 = +x 전진 (끝선에서 자름), 몸 방향 기준 +x."""

    corridor_half_width = 0.40

    def __init__(self):
        super().__init__()
        self.poles = [(1.0, 0.15), (2.3, -0.15), (3.6, 0.15)]
        self.half_width = self.corridor_half_width
        self.penalties = dict(self.penalties, lane=20.0)

    def p_lane(self, x, m):
        out = torch.sigmoid((x[..., 1].abs() - self.half_width) / 0.02)
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

    def describe(self):
        return dict(super().describe(), task="corridor_poles3", corridor_half_width=self.half_width)

    def guide_lines(self, length=None):
        L = self.finish_x + 0.3 if length is None else length
        return [(np.array([[0.0, y], [L, y]]), True) for y in (-self.half_width, self.half_width)] + \
               [(np.array([[self.finish_x, -self.half_width], [self.finish_x, self.half_width]]), False)]


class CorridorPolesFastReward(CorridorPolesReward):
    """2026-11-19 사용자 결정: corridor3 와 같은 통로 · 엇갈린 기둥 3 개, 끝선까지 최소 시간. 몸이 +x 를 볼 필요 없음.
    - 진행 가중치 1 → 5 (최소 시간 쪽으로).
    - 몸 방향 감점 기준 = +x 대신 실제 움직이는 방향 (xy 속도). 몸을 틀어 가는 쪽을 보며 가면 감점 0, 옆걸음 · 뒷걸음에 감점.
      속도 0.2 m/s 미만은 비례해서 줄임 (거의 멈춘 순간의 방향은 의미 없음). 가중치 5 그대로."""

    progress_weight = 5.0
    heading_min_speed = 0.2

    def __init__(self):
        super().__init__()
        self.rewards = {"progress": self.progress_weight}

    def p_heading(self, x, m):
        hz = float(os.environ.get("TT_CONTROL_HZ", "100"))
        v = (x[:, 1:, POS] - x[:, :-1, POS]) * hz
        v = torch.cat([v, v[:, -1:]], 1)
        speed = v.norm(dim=-1)
        d = v / speed.clamp_min(1e-6)[..., None]
        n = torch.sqrt(x[..., COS] ** 2 + x[..., SIN] ** 2).clamp_min(1e-6)
        out = (1 - (x[..., COS] * d[..., 0] + x[..., SIN] * d[..., 1]) / n) * (speed / self.heading_min_speed).clamp(max=1)
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

    def describe(self):
        return dict(super().describe(), task="corridor_poles3_fast", heading_ref="velocity_direction", heading_min_speed=self.heading_min_speed)


class _TerrainForward(StraightReward):
    """2026-11-19 사용자 결정 (지금 TT 로 역동적 과제가 되는지 확인): 지형 과제 공통. 끝선 finish_x 까지 최소 시간.
    진행 = +x 전진 (끝선에서 자름) × 5, 통로 = 몸통 중심 |y| <= 0.40 밖 시간 비율 × 20, 몸 방향 기준 +x (× 5), 넘어짐 · 생존 20.
    지형 상자는 add_scene 으로 MuJoCo 에 넣고 (이름 terrain_k = 발 접촉을 바닥으로 셈), 장애물은 obstacle_k (닿으면 접촉으로 셈)."""

    finish_x = 3.0
    corridor_half_width = 0.40
    boxes = ()      # 바닥 지형 (x0, x1, 윗면 z) — 발이 딛는 곳
    obstacles = ()  # 장애물 (x0, x1, 윗면 z) — 닿으면 안 되는 것
    box_half_y = 1.0

    def __init__(self):
        super().__init__()
        self.half_width = self.corridor_half_width
        self.rewards = {"progress": 5.0}
        self.penalties = dict(self.penalties, lane=20.0)

    def r_progress(self, x, m):
        px = x[..., 0].clamp(max=self.finish_x)
        return ((px[:, 1:] - px[:, :-1]) * (m[:, 1:] & m[:, :-1])).sum(1)

    def p_lane(self, x, m):
        out = torch.sigmoid((x[..., 1].abs() - self.half_width) / 0.02)
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

    def ground_h(self, px):
        """x 위치의 바닥 높이 (m)."""
        h = torch.zeros_like(px)
        for x0, x1, top in self.boxes:
            h = torch.where((px >= x0) & (px < x1), torch.full_like(px, top), h)
        return h

    def add_scene(self, spec):
        import mujoco
        for kind, items, rgba in (("terrain", self.boxes, [0.55, 0.55, 0.6, 1]), ("obstacle", self.obstacles, [0.85, 0.18, 0.12, 1])):
            for k, (x0, x1, top) in enumerate(items):
                spec.worldbody.add_geom(name=f"{kind}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, self.box_half_y, top / 2],
                                        pos=[(x0 + x1) / 2, 0, top / 2], rgba=rgba, contype=1, conaffinity=1)

    def guide_lines(self, length=None):
        L = self.finish_x + 0.3 if length is None else length
        return [(np.array([[0.0, y], [L, y]]), True) for y in (-self.half_width, self.half_width)] + \
               [(np.array([[self.finish_x, -self.half_width], [self.finish_x, self.half_width]]), False)]

    def describe(self):
        return dict(task=type(self).__name__, finish_x=self.finish_x, corridor_half_width=self.half_width, boxes=list(self.boxes),
                    obstacles=list(self.obstacles), rewards=self.rewards, penalties=self.penalties, min_height_frac=self.min_height_frac)


class Stairs5Reward(_TerrainForward):
    """계단 5 칸 오르고 내리기: 1 칸 높이 0.10 m · 디딤 0.30 m (legacy stairsUpDown 과 같은 치수), x 1.0 부터 오르기 → 윗면 (0.5 m) 1.0 m → 5 칸 내리기 → 끝선 x 5.5.
    추가 감점 height (× 100): 몸통 높이 - 발 아래 바닥 높이 가 출발 때 몸통 높이에서 벗어난 거리² 평균 (TT 계획이 계단 높이를 따라가게)."""

    rise, tread, n_steps, x_start, top_len = 0.10, 0.30, 5, 1.0, 1.0
    height_weight = 100.0

    def __init__(self):
        r, d, n, x0 = self.rise, self.tread, self.n_steps, self.x_start
        up = [(x0 + i * d, x0 + (i + 1) * d, r * (i + 1)) for i in range(n - 1)]
        xt = x0 + (n - 1) * d
        top = [(xt, xt + d + self.top_len + d * 0, r * n)]
        xd = xt + d + self.top_len
        down = [(xd + i * d, xd + (i + 1) * d, r * (n - 1 - i)) for i in range(n - 1)]
        self.boxes = tuple(up + top + down)
        self.finish_x = xd + (n - 1) * d + 0.5
        super().__init__()
        self.penalties = dict(self.penalties, height=self.height_weight)

    def p_height(self, x, m):
        rel = x[..., Z] - self.ground_h(x[..., 0])
        out = (rel - x[:, :1, Z]) ** 2
        return (out * m).sum(1) / m.sum(1).clamp(min=1)


class HurdleReward(_TerrainForward):
    """허들 넘기: x 1.5 에 높이 0.10 m · 두께 0.04 m 허들 (폭 y ±1.0), 끝선 x 2.5.
    추가 감점 clear (× 100): 몸통 중심이 허들 ± 0.35 m 안에 있을 때 몸통 높이가 (출발 높이 + 허들 높이) 보다 낮은 만큼² 평균 (뛰어 넘게)."""

    hurdle_x, hurdle_h, hurdle_t = 1.5, 0.10, 0.04
    clear_weight = 100.0

    def __init__(self):
        self.obstacles = ((self.hurdle_x - self.hurdle_t / 2, self.hurdle_x + self.hurdle_t / 2, self.hurdle_h),)
        self.finish_x = self.hurdle_x + 1.0
        super().__init__()
        self.penalties = dict(self.penalties, clear=self.clear_weight)

    def p_clear(self, x, m):
        near = torch.sigmoid((0.35 - (x[..., 0] - self.hurdle_x).abs()) / 0.02)
        low = torch.relu(x[:, :1, Z] + self.hurdle_h - x[..., Z]) ** 2
        return (near * low * m).sum(1) / m.sum(1).clamp(min=1)
