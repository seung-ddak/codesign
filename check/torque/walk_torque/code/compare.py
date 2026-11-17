"""walk 대 trot 레퍼런스: 학습 형상 9 개 리타겟 (s5_retarget/references) 의 역동역학 토크 한계 (33.5 N·m) 초과 비교 (2026-11-17 사용자 요청).
CALM trot 모션 (check/torque_exceed/rows.json, 12,000 쌍) 과도 같이 본다. 출력: rows.json, compare.png, report.html"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관; R = ROOT / 'chain/runs/s5_retarget/references'; LIM = 33.5
sys.path.insert(0, str(ROOT / 'chain/code'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus'] = False
SH = sorted(os.listdir(R)); JN = ['hip', 'thigh', 'calf']


def main():
    rows = []
    for s in SH:
        for f in sorted(glob.glob(str(R / s / '00000011' / '*.npz'))):
            m = Path(f).stem.replace('__r00', '')
            if not m.startswith(('walk', 'trot')): continue
            z = np.load(f); tau = np.abs(z['inv_tau'][:, :12]); over = tau > LIM
            rows.append(dict(shape=s, motion=m, gait=m[:4], speed='Fast' if m.endswith('Fast') else 'Slow', dir=m[4:].replace('Slow', '').replace('Fast', ''),
                             tau_max=float(tau.max()), tau_p99=float(np.percentile(tau, 99)), over_frac=float(over.any(1).mean()),
                             joint_over=[float(over[:, j::3].any(1).mean()) for j in range(3)], joint_max=[float(tau[:, j::3].max()) for j in range(3)],
                             res_p95=float(z['inv_res_force_p95_pct'])))
    json.dump(rows, open(HERE / 'rows.json', 'w'), indent=1)
    calm = json.load(open(ROOT / 'check/torque_exceed/rows.json'))
    cats = [('walk', 'Slow'), ('trot', 'Slow'), ('walk', 'Fast'), ('trot', 'Fast')]; col = {'walk': '#1baa6b', 'trot': '#e8590c'}
    dirs = ['Forward', 'Backward', 'Left', 'Right', 'CW', 'CCW']
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    # (a) 형상 × 모션 히트맵: 초과 프레임 비율 (Slow)
    for k, g in enumerate(['walk', 'trot']):
        a = ax[0, k]; M = np.full((len(SH), len(dirs)), np.nan)
        for r in rows:
            if r['gait'] == g and r['speed'] == 'Slow': M[SH.index(r['shape']), dirs.index(r['dir'])] = r['over_frac'] * 100
        im = a.imshow(M, cmap='Reds', vmin=0, vmax=30, aspect='auto'); a.set_xticks(range(len(dirs)), dirs); a.set_yticks(range(len(SH)), SH)
        for i in range(len(SH)):
            for j in range(len(dirs)):
                if not np.isnan(M[i, j]): a.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=8, color='k' if M[i, j] < 18 else 'w')
        a.set_title(f'{g} Slow: 토크 초과 프레임 비율 (%)'); fig.colorbar(im, ax=a, fraction=.04)
    # (b) 최대 토크 분포: walk/trot Slow·Fast + CALM trot
    a = ax[1, 0]; data, labs, cs = [], [], []
    for g, sp in cats:
        v = [r['tau_max'] for r in rows if r['gait'] == g and r['speed'] == sp]; data.append(v); labs.append(f'{g} {sp}\n({len(v)})'); cs.append(col[g])
    data.append([r['tau_max'] for r in calm]); labs.append(f'CALM trot\n({len(calm)})'); cs.append('#888')
    bp = a.boxplot(data, tick_labels=labs, patch_artist=True, showfliers=False)
    for p, c in zip(bp['boxes'], cs): p.set_facecolor(c); p.set_alpha(.5)
    a.axhline(LIM, color='r', ls='--', label='한계 33.5 N·m'); a.set_ylabel('최대 |τ| (N·m)'); a.set_title('최대 토크 (형상 × 모션 쌍)'); a.legend(); a.grid(alpha=.3)
    # (c) 관절별 초과 비율 (Slow)
    a = ax[1, 1]; w = .38
    for k, g in enumerate(['walk', 'trot']):
        v = [np.mean([r['joint_over'][j] for r in rows if r['gait'] == g and r['speed'] == 'Slow']) * 100 for j in range(3)]
        a.bar(np.arange(3) + (k - .5) * w, v, w, color=col[g], label=f'{g} Slow')
    a.set_xticks(range(3), JN); a.set_ylabel('초과 프레임 비율 평균 (%)'); a.set_title('관절별 토크 초과 (Slow, 형상 9 × 방향 6 평균)'); a.legend(); a.grid(alpha=.3, axis='y')
    fig.tight_layout(); fig.savefig(HERE / 'compare.png', dpi=100); plt.close(fig)
    from common import report, table, img
    def summ(sel, name):
        o = np.array([r['over_frac'] for r in sel]); t = np.array([r['tau_max'] for r in sel])
        return (name, len(sel), f'{(o > 0).mean() * 100:.0f} %', f'{(o > .01).mean() * 100:.0f} %', f'{o.mean() * 100:.1f} %', f'{np.median(t):.1f}', f'{t.max():.1f}')
    S = [summ([r for r in rows if r['gait'] == g and r['speed'] == sp], f'{g} {sp}') for g, sp in cats] + [summ(calm, 'CALM trot (s5, 형상 2 개씩)')]
    H = ['모션', '쌍 수', '초과 있는 쌍', '초과 프레임 > 1 % 쌍', '초과 프레임 평균', '최대 토크 중앙값 (N·m)', '최대 토크 최대']
    det = table(['형상', '모션', '최대 토크', '초과 프레임 %', 'hip / thigh / calf 최대', '루트 잔여힘 p95 %'],
                [(r['shape'], r['motion'], f"{r['tau_max']:.1f}", f"{r['over_frac'] * 100:.1f}", ' / '.join(f'{x:.0f}' for x in r['joint_max']), f"{r['res_p95']:.0f}") for r in rows], filter_col=1)
    report(HERE / 'report.html', 'walk 대 trot 토크 초과', [('요약', table(H, S)), ('그림', img(HERE / 'compare.png')), ('형상 × 모션 표', det)],
           [('데이터', 'chain/runs/s5_retarget/references/<형상>/00000011/*.npz 의 inv_tau (리타겟 모션의 역동역학 관절 토크, 30 Hz)'),
            ('형상', '학습 형상 9 개 (다리 −20·0·+20 %, 몸통 −20·0·+20 %)'), ('기준', '다리 관절 12 개 중 하나라도 |τ| > 33.5 N·m 인 프레임'),
            ('CALM trot', 'check/torque_exceed/rows.json (CALM 6,000 개 × 배정 형상 2 개)'), ('코드', 'check/walk_torque/code/compare.py')])
    for s in S: print(s)


if __name__ == '__main__':
    main()
