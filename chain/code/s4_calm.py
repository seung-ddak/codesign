"""4 단계: 조이스틱 CALM 모션 재구성. trot 레퍼런스 10 개에 할당량 (기본 1,000) 씩 고르게 채운다.
라운드마다: 부족분 가중 조이스틱 계획 → Isaac Gym CALM 수집 (collect_calm_joystick.py, rlgpu) → 구간별 최근접 레퍼런스 분류
(ref_ppo/_classification/classify.py 와 같은 특징·규칙: 모션 = 가장 긴 시간을 차지한 레퍼런스) → 할당량 집계.
출력: runs/s4_calm/round_XX/ (수집), selected.json {레퍼런스: [npz 경로]}, report.html, 그림."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from common import ROOT, RUNS, CHAIN, CODE_DIR, log, run, report, table, img
sys.path.insert(0, str(ROOT / 'ref_ppo/_classification'))
from calm_joystick import TROT, make_plan, commands_for
RLGPU = '/home/cgr-ugrad-2026/miniconda3/envs/rlgpu'
OUT = RUNS / 's4_calm'
REF = ROOT / 'legacy/artifacts/randomized_reference'
WHEEL = ROOT / 'legacy/runs/menagerie_calm_check/try4/runtime/rl_games-1.1.4-py3-none-any.whl'


def feat(pos, quat_xyzw, q, dt):
    from scipy.spatial.transform import Rotation
    yaw = Rotation.from_quat(quat_xyzw).as_euler('zyx')[:, 0]
    v = np.diff(pos[:, :2], axis=0) / dt; c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    vx, vy = c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]
    wz = np.diff(np.unwrap(yaw)) / dt
    return np.r_[vx.mean(), vy.mean(), wz.mean(), pos[:, 2].mean(), vx.std(), vy.std(), wz.std(), pos[:, 2].std(), q.mean(0), q.std(0)]


def ref_features():
    names, f = [], []
    for p in sorted(REF.glob('*.npz')):
        if p.stem not in TROT: continue  # 6 개 묶음 레퍼런스에만 분류 (빠른 구간이 Fast 레퍼런스로 빠지지 않게)
        z = np.load(p); names.append(p.stem)
        f.append(feat(z['fix_base_pos'], z['fix_base_quat'][:, [1, 2, 3, 0]], z['fix_q'], float(z['dt'])))
    return names, np.array(f)


def classify(files):
    """반환: {파일: (레퍼런스, 구간 특징 목록)}. 스케일 = 레퍼런스 + 이번 구간 특징 전체 표준편차 (classify.py 와 같은 방식)."""
    names, rf = ref_features(); segs = []
    for fi, path in enumerate(files):
        z = np.load(path); jn = z['joint_names'].tolist()
        order = [jn.index(l + '_' + j + '_joint') for l in ('FR', 'FL', 'RR', 'RL') for j in ('hip', 'thigh', 'calf')]
        dt = float(np.diff(z['time']).mean()); root, q = z['root'], z['q'][:, order]
        for a0, b0 in zip(z['segment_start'], z['segment_end']):
            a0, b0 = int(a0), int(b0) + 1
            if b0 - a0 >= 6: segs.append((fi, b0 - a0, feat(root[a0:b0, :3], root[a0:b0, 3:7], q[a0:b0], dt)[:3]))
    # 2026-11-12 사용자 결정: 실제 속도 (몸통 기준 vx, vy, wz 평균) 만으로 분류. 관절 특징 24 개가 속도를 이겨 Left 구간이 Right 로 붙었다.
    rf = rf[:, :3]; sf = np.array([s[2] for s in segs]); scale = np.r_[rf, sf].std(0) + 1e-9
    best = np.linalg.norm((sf[:, None] - rf[None]) / scale, axis=-1).argmin(1)
    votes = np.zeros((len(files), len(names)))
    for (fi, n, _), b in zip(segs, best): votes[fi, b] += n
    return {str(files[i]): names[int(votes[i].argmax())] for i in range(len(files))}, sf, best, names


def tally(quota):
    files = sorted(OUT.glob('round_*/accepted/*.npz'))
    if not files: return {r: [] for r in TROT}, {}
    label, *_ = classify(files)
    by = {r: [] for r in TROT}; other = {}
    for f, r in label.items():
        (by[r].append(f) if r in by else other.__setitem__(r, other.get(r, 0) + 1))
    return by, other


def main(quota=1000, per_round=2000, max_rounds=8):
    OUT.mkdir(parents=True, exist_ok=True)
    for rnd in range(max_rounds):
        by, other = tally(quota)
        deficit = np.array([max(0, quota - len(by[r])) for r in TROT])
        log(f's4 round {rnd}: ' + ', '.join(f'{r[:-5]} {len(by[r])}' for r in TROT) + f' | 기타 {sum(other.values())}')
        if deficit.sum() == 0: break
        d = OUT / f'round_{rnd:02d}'
        if not (d / 'status.json').exists() or json.loads((d / 'status.json').read_text()).get('status') != 'complete':
            d.mkdir(exist_ok=True)
            if not (d / 'plan.json').exists():
                n = int(min(per_round, max(200, deficit.sum() * 1.5)))
                (d / 'plan.json').write_text(json.dumps(dict(motions=make_plan(n, 1000 + rnd, weights=deficit / deficit.sum()))))
            run([f'{RLGPU}/bin/python', '-u', str(CODE_DIR / 'collect_calm_joystick.py'), '--out', str(d), '--batch', '16',
                 '--max-attempts', str(3 * len(json.loads((d / 'plan.json').read_text())['motions'])), '--rl-games-wheel', str(WHEEL)],
                d / 'collect.log', env=dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLCONFIGDIR='/tmp/calm-mpl', LD_LIBRARY_PATH=f'{RLGPU}/lib'))
    by, other = tally(quota)
    sel = {r: by[r][:quota] for r in TROT}
    json.dump(sel, open(OUT / 'selected.json', 'w'), indent=1)
    make_report(by, other, quota)
    return sel


def make_report(by, other, quota):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    rep = OUT / 'report'; rep.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([r.replace('__r00', '') for r in TROT], [len(by[r]) for r in TROT], color='#2a78d6'); ax.axhline(quota, color='k', ls='--', label=f'할당량 {quota}')
    ax.set_ylabel('모션 수'); ax.legend(); plt.xticks(rotation=30, ha='right'); fig.tight_layout(); fig.savefig(rep / 'counts.png', dpi=110); plt.close(fig)
    # 명령 대 실제 속도 (모션 일부 표본)
    files = [f for r in TROT for f in by[r][:40]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for f in files:
        z = np.load(f); c = z['command']; root = z['root']; dt = float(np.diff(z['time']).mean())
        from scipy.spatial.transform import Rotation
        yaw = np.unwrap(Rotation.from_quat(root[:, 3:7]).as_euler('zyx')[:, 0]); v = np.diff(root[:, :2], axis=0) / dt
        vx = np.cos(yaw[:-1]) * v[:, 0] + np.sin(yaw[:-1]) * v[:, 1]; vy = -np.sin(yaw[:-1]) * v[:, 0] + np.cos(yaw[:-1]) * v[:, 1]; wz = np.diff(yaw) / dt
        ax[0].scatter(vx[::10], vy[::10], s=2, alpha=.4); ax[1].hist(np.hypot(vx, vy), bins=40, alpha=.15); ax[2].hist(wz, bins=40, alpha=.15)
    ax[0].set_xlabel('실제 vx (m/s)'); ax[0].set_ylabel('실제 vy'); ax[1].set_xlabel('실제 이동 속력 (m/s)'); ax[2].set_xlabel('실제 회전 wz (rad/s)')
    fig.tight_layout(); fig.savefig(rep / 'speeds.png', dpi=110); plt.close(fig)
    rows = [(r.replace('__r00', ''), len(by[r]), quota, '충족' if len(by[r]) >= quota else f'부족 {quota - len(by[r])}') for r in TROT]
    st = [json.loads(p.read_text()) for p in sorted(OUT.glob('round_*/status.json'))]
    report(rep / 'report.html', '4 단계: 조이스틱 CALM 모션 재구성', [
        ('레퍼런스별 모션 수', img(rep / 'counts.png', '분류 = 구간별 최근접 레퍼런스, 모션 = 가장 긴 시간') + table(['레퍼런스', '모션', '할당량', '상태'], rows,
                                                                                                         lambda i: 'bad' if rows[i][1] < quota else 'good')),
        ('trot 이 아닌 레퍼런스로 분류된 모션', table(['레퍼런스', '모션 수'], sorted(other.items(), key=lambda x: -x[1])) if other else '<p>없음</p>'),
        ('실제 속도 분포 (레퍼런스별 40 개 표본)', img(rep / 'speeds.png')),
        ('조이스틱 명령 예', img(CODE_DIR / 'joystick_example.png', 'vx, vy, wz (몸통 기준), seed 1~3, 20 초')),
        ('수집 라운드', table(['라운드', '목표', '통과', '시도'], [(i, s.get('target'), s.get('accepted'), s.get('attempts')) for i, s in enumerate(st)]))],
        [('CALM', 'A1 CALM HLC+LLC (external/a1-calm-main), Isaac Gym PhysX, 30 Hz 제어, kp 40 kd 1.5'),
         ('명령', '조이스틱: 모션마다 주 방향 1 개 (부족분 가중), 1~3 초마다 목표, 구간의 75 % 주 방향 · 나머지 고르게, 0.3~1.0 초 smoothstep 전환, 속도 배율 0.5~2.5, 10 % 다른 축 섞기, 3 % 정지, OU 잡음 3 %'),
         ('길이', '10 초 / 20 초 반반'), ('통과 검사', '넘어짐 없음, 활동 비율 ≥ 60 %, 연속 정지 ≤ 3 초, 근접 중복 제외 (0.03)'),
         ('분류', '구간별 실제 속도 (몸통 기준 vx, vy, wz 평균) 의 최근접 trot 레퍼런스 6 개, 모션 = 가장 긴 시간')])
    return rep


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--quota', type=int, default=1000); p.add_argument('--per-round', type=int, default=2000); p.add_argument('--max-rounds', type=int, default=8)
    a = p.parse_args(); main(a.quota, a.per_round, a.max_rounds)
