"""목표 도달형 과제 MPC (2026-11-19 사용자 결정).
과제: circle3 = 반지름 1 m 원형 트랙 3 바퀴 가장 빨리, zigzag3 = 기둥 3 개 피해 끝 게이트 (x 4.7 m) 까지 가장 빨리, straight = 10 s 동안 가장 멀리.
- 최대 에피소드: circle3·zigzag3 60 s, straight 10 s. 넘어지거나 목표 도달하면 그 후보는 끝.
- 예측 구간 고정: 창 10 개 (311 상태, 3.1 s). 31 step 실행 → MuJoCo 실제 상태로 다음 3.1 s 다시 생성 (형상 E 고정).
- warm start: 이전 계획 창을 한 창 앞으로 밀고 (끝 창은 새 noise) noise 단계 warm_k 부터 denoise.
- MuJoCo 는 후보마다 상태를 유지하며 31 step 씩 이어서 실행 (처음부터 다시 재생하지 않음; generate_task_mujoco.replay 와 같은 판정).
출력: <out>/result.npz (후보별 실제 특징·예측 특징·valid), summary.json"""
import argparse, json, math, os, sys, time
from pathlib import Path
os.environ.setdefault('TT_CONTROL_HZ', '100'); os.environ.setdefault('TT_PHYSICS_HZ', '200')  # s8z: ctrl 0.01, sim 0.005
ROOT = Path(__file__).resolve().parents[3]  # codesign
sys.path.insert(0, str(ROOT / 'code/tt_dynamics'))
import mujoco
import numpy as np, torch
import generate_task_mujoco as gtm
from joint_diffusion_guidance import joint_diffusion_full_reward
from rollout_data import features
from s0_geometry import s0_to_q
from task_reward import TrackReward, Slalom3Reward, StraightReward, SlalomReward
from tt_model import make_net, schedule, check_rest_checkpoint

TASKS = dict(circle3=TrackReward, zigzag3=Slalom3Reward, straight=StraightReward)
MAX_S = dict(circle3=60.0, zigzag3=60.0, straight=10.0)
CIRCLE_LAPS = 3.0


def load(ckdir, dev):
    ck = torch.load(Path(ckdir) / 'checkpoint.pt', map_location=dev, weights_only=False); ta = ck['args']; check_rest_checkpoint(ta)
    net = make_net(ta).to(dev); net.load_state_dict(ck['ema_net'] if ck.get('ema_net') is not None else ck['net']); net.eval().requires_grad_(False)
    mu = torch.as_tensor(ck['mu'], device=dev); sd = torch.as_tensor(ck['sd'], device=dev)
    beta, abar = schedule(ta['diff_steps'], dev); emu, esd = ((.5, .5) if ta['norm'] == 'minmax' else (.5, .25))
    return net, ta, dict(mu=mu, sd=sd, beta=beta, abar=abar, emu=emu, esd=esd, clip=ta['clip'], horizon=ta['horizon'], state_dim=42, action_dim=12, e_dim=13)


class Sim:
    """후보 하나의 MuJoCo 상태를 유지하며 행동을 이어서 실행. 판정 (넘어짐·physics 오류·폴 접촉) 은 gtm.replay 와 같다."""

    def __init__(self, e, s0, task):
        self.task = task; self.m = gtm.make_physics(e, task); self.d = mujoco.MjData(self.m); m, d = self.m, self.d
        self.feet = [m.geom(leg + "_foot").id for leg in gtm.FEET]; self.floor = m.geom("floor").id
        d.qpos[:], d.qvel[:] = s0_to_q(np.asarray(s0, np.float32)); mujoco.mj_forward(m, d)
        self.pole_ids = {m.geom(f"slalom_pole_{i}").id: i for i in range(len(task.poles))} if isinstance(task, SlalomReward) else {}
        self.pole_contacts = set(); self._poles()
        lim = m.actuator_ctrllimited[:12].astype(bool)
        self.lo = np.where(lim, m.actuator_ctrlrange[:12, 0], -np.inf); self.hi = np.where(lim, m.actuator_ctrlrange[:12, 1], np.inf)
        self.qpos, self.qvel, self.con, self.applied = [d.qpos.copy()], [d.qvel.copy()], [gtm.floor_contacts(m, d, self.feet, self.floor)], []
        self.end = None  # None = 계속, 'fall' / 'physics_error' / 'goal' / 'time'

    def _poles(self):
        if not self.pole_ids: return
        for c in self.d.contact[:self.d.ncon]:
            if c.dist > 0: continue
            for pole, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if pole in self.pole_ids and other >= 0 and self.m.geom_bodyid[other] > 0: self.pole_contacts.add(self.pole_ids[pole])

    def _failed(self):
        d = self.d; bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
        return any(d.warning[w].number for w in bad) or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()

    def step(self, actions):
        m, d = self.m, self.d
        for a in np.clip(actions, self.lo, self.hi):
            d.ctrl[:12] = a
            for _ in range(gtm.SUBSTEPS):
                mujoco.mj_step(m, d); self._poles()
            if self._failed(): self.end = 'physics_error'; return
            mujoco.mj_forward(m, d)
            if self._failed(): self.end = 'physics_error'; return
            self._poles()
            self.applied.append(d.ctrl[:12].copy()); self.qpos.append(d.qpos.copy()); self.qvel.append(d.qvel.copy())
            self.con.append(gtm.floor_contacts(m, d, self.feet, self.floor))
            if 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2) <= .5 or d.qpos[2] <= .07: self.end = 'fall'; return

    def traj(self):
        """[n+1, 54] 출발 기준 특징 + 행동 (마지막 행 행동 = 직전 행동 반복, gtm.replay 와 같음)."""
        n = len(self.applied); out = np.zeros((n + 1, 54), np.float32)
        out[:, :42] = features(np.asarray(self.qpos), np.asarray(self.qvel), np.asarray(self.con))
        if n: out[:n, 42:] = self.applied; out[n, 42:] = self.applied[-1]
        return out

    def truncate(self, n):
        """목표 도달 step n 까지만 남김."""
        self.qpos, self.qvel, self.con, self.applied = self.qpos[:n + 1], self.qvel[:n + 1], self.con[:n + 1], self.applied[:n]


def goal_step(name, task, x):
    """실제 궤적 x [n+1,54] 에서 목표에 처음 도달한 step (없으면 None)."""
    if name == 'circle3':
        rel = x[:, :2] - np.asarray(task.center); a, b = rel[:-1], rel[1:]
        th = np.cumsum(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)))
        hit = np.flatnonzero(th >= CIRCLE_LAPS * 2 * np.pi)
        return int(hit[0]) + 1 if len(hit) else None
    if name == 'zigzag3':
        c = task.gate_crossings(x[:, :2])
        return int(math.ceil(c[-1])) if len(c) == len(task.gates) else None
    return None


def progress(name, task, x):
    if name == 'circle3':
        rel = x[:, :2] - np.asarray(task.center); a, b = rel[:-1], rel[1:]
        return float(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)).sum() / (2 * np.pi))  # 바퀴
    if name == 'zigzag3': return float(len(task.gate_crossings(x[:, :2])))  # 통과 게이트 수
    return float(x[-1, 0] - x[0, 0])  # x 전진 (m)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--task', required=True, choices=list(TASKS)); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scale', type=float, default=50); p.add_argument('--candidates', type=int, default=8); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--windows', type=int, default=10, help='예측 구간 창 수 (창 = 31 step)'); p.add_argument('--warm-k', type=int, default=60)
    p.add_argument('--max-seconds', type=float, default=None, help='check 용 최대 시간 덮어쓰기'); p.add_argument('--circle-laps', type=float, default=None, help='check 용 목표 바퀴 덮어쓰기')
    a = p.parse_args()
    global CIRCLE_LAPS
    if a.circle_laps is not None: CIRCLE_LAPS = a.circle_laps
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; net, ta, kw = load(a.ckpt, dev); task = TASKS[a.task](); a.out.mkdir(parents=True, exist_ok=True)
    H = kw['horizon']; every = H - 1; L = a.windows * every + 1; N = a.candidates
    T = int(round((a.max_seconds or MAX_S[a.task]) * gtm.CONTROL_HZ))

    def s0_fn(eb):
        return torch.as_tensor(np.stack([gtm.settled_start(e) for e in eb.detach().cpu().numpy()]), device=dev, dtype=kw['mu'].dtype)
    t0 = time.time()
    plan, E, aud = joint_diffusion_full_reward(net, task, n=N, episode_states=L, guidance_scale=a.scale, seed=a.seed, s0_fn=s0_fn, return_latents=True, **kw)
    tok = plan.detach().cpu().numpy(); E = E.detach().cpu().numpy(); lat = aud['latents']
    sims = [Sim(E[i], tok[i, 0, :42], task) for i in range(N)]
    pred = [[tok[i, 0, :42]] for i in range(N)]  # 실행한 step 의 TT 예측 상태 (31 step 마다 실제 상태에서 다시 시작)
    t, replans = 0, 0
    while True:
        for i, s in enumerate(sims):
            if s.end: continue
            k = min(every, T - t); s.step(tok[i, :k, 42:54])
            got = len(s.applied) - t; pred[i].extend(tok[i, 1:got + 1, :42])
            if s.end: continue
            g = goal_step(a.task, task, s.traj())
            if g is not None: s.truncate(g); pred[i] = pred[i][:g + 1]; s.end = 'goal'
            elif len(s.applied) >= T: s.end = 'time'
        t += every
        alive = np.array([s.end is None for s in sims])
        if not alive.any(): break
        cur = np.stack([s.traj()[-1, :42] for s in sims])
        new, _, aud = joint_diffusion_full_reward(net, task, n=N, episode_states=L, guidance_scale=a.scale, seed=a.seed + t, fixed_e=E,
                                                  init_state=torch.as_tensor(cur, device=dev), warm_x=list(lat[1:]) + [None], warm_k=a.warm_k, return_latents=True, **kw)
        tok = new.detach().cpu().numpy(); lat = aud['latents']; replans += 1
        if replans % 20 == 0: print(f'{a.task} t={t / gtm.CONTROL_HZ:.1f}s replans {replans} alive {int(alive.sum())}/{N} {time.time() - t0:.0f}s', flush=True)
    rows, acts, preds = [], [], []
    for i, s in enumerate(sims):
        x = s.traj(); pr = np.asarray(pred[i])[:len(x)]; acts.append(x); preds.append(pr)
        with torch.no_grad():
            tot, terms = task(torch.as_tensor(x[None]), torch.ones(1, len(x), dtype=torch.bool))
        n = len(x) - 1; yaw = np.unwrap(np.arctan2(x[:, 4], x[:, 3]))
        rows.append(dict(candidate=i, end=s.end, seconds=n / gtm.CONTROL_HZ, finish_s=n / gtm.CONTROL_HZ if s.end == 'goal' else None,
                         progress=progress(a.task, task, x), reward=float(tot[0]), reward_terms={k: float(v[0]) for k, v in terms.items()},
                         pole_contacts=sorted(s.pole_contacts), fwd_x=float(x[-1, 0]), rot_deg=float(np.degrees(np.abs(np.diff(yaw)).sum())),
                         joint_err_deg=float(np.degrees(np.sqrt(np.mean((pr[:, 14:26] - x[:len(pr), 14:26]) ** 2)))),
                         pos_err_cm=float(np.linalg.norm(pr[:, :2] - x[:len(pr), :2], axis=1).mean() * 100), morphology=E[i].tolist()))
    Tm = max(len(x) for x in acts)
    pad = lambda xs, d: np.stack([np.concatenate([x, np.zeros((Tm - len(x), d), np.float32)]) for x in xs])
    np.savez_compressed(a.out / 'result.npz', E=E, actual=pad(acts, 54), pred=pad(preds, 42), valid=np.stack([np.arange(Tm) < len(x) for x in acts]))
    (a.out / 'summary.json').write_text(json.dumps(dict(task=a.task, ckpt=a.ckpt, scale=a.scale, seed=a.seed, windows=a.windows, lookahead_s=(L - 1) / gtm.CONTROL_HZ,
        every=every, warm_k=a.warm_k, max_s=T / gtm.CONTROL_HZ, circle_laps=CIRCLE_LAPS if a.task == 'circle3' else None, hz=gtm.CONTROL_HZ, replans=replans,
        seconds=time.time() - t0, reward=task.describe(), results=rows), indent=1, ensure_ascii=False, default=float))
    f = lambda r: (r['candidate'], r['end'], round(r['seconds'], 2), round(r['progress'], 2), len(r['pole_contacts']))
    print(f'{a.task} done {time.time() - t0:.0f}s replans {replans}:', [f(r) for r in rows], flush=True)


if __name__ == '__main__':
    main()
