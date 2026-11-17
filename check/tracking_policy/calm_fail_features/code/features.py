"""s6c50 (CALM 단독 50M) 실패 모션 대 통과 모션의 레퍼런스 특징 비교 (2026-11-17 사용자 요청).
실패 = 통과 형상 ≤ 1/9, 통과 = ≥ 8/9. 특징은 A1 레퍼런스 (s5_retarget/a1_reference, 30 Hz) 에서 계산.
출력: features.json, features.png, report.html, videos/ (s6c50 영상 복사)"""
import json, shutil, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관; RUNS = ROOT / 'chain/runs'
sys.path.insert(0, str(ROOT / 'chain/code'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus'] = False
A1_LO = np.array([-0.802, -1.047, -2.697] * 4); A1_HI = np.array([0.802, 4.189, -0.916] * 4)  # A1 관절 한계 (hip, thigh, calf)


def feats(f):
    z = np.load(f); dt = float(z['dt']); L = len(z['fix_q']) * dt; k = int(round(10 / dt)) + 1  # 평가는 처음 10 초만 → 특징도 처음 10 초
    p = z['fix_base_pos'][:k]; q = z['fix_q'][:k]; c = z['fix_contact'][:k]; fp = z['fix_foot_pos'][:k]
    R = Rotation.from_quat(z['fix_base_quat'][:k, [1, 2, 3, 0]]); e = R.as_euler('zyx'); yaw = np.unwrap(e[:, 0])
    v = np.diff(p[:, :2], axis=0) / dt; wz = np.diff(yaw) / dt
    c_, s_ = np.cos(yaw[:-1]), np.sin(yaw[:-1]); vb = np.c_[c_ * v[:, 0] + s_ * v[:, 1], -s_ * v[:, 0] + c_ * v[:, 1]]
    qd = np.diff(q, axis=0) / dt; qdd = np.diff(qd, axis=0) / dt
    fv = np.linalg.norm(np.diff(fp[:, :, :2], axis=0), axis=2) / dt; inc = c[1:] & c[:-1]
    margin = np.minimum(q - A1_LO, A1_HI - q).min(1)
    return {
        '클립 전체 길이 (s)': L,
        '평균 속력 (m/s)': float(np.linalg.norm(vb, axis=1).mean()),
        '평균 |vx| (m/s)': float(np.abs(vb[:, 0]).mean()), '평균 |vy| (m/s)': float(np.abs(vb[:, 1]).mean()),
        '평균 |wz| (rad/s)': float(np.abs(wz).mean()), '최대 |wz| (rad/s)': float(np.percentile(np.abs(wz), 99)),
        '수평 가속 RMS (m/s²)': float(np.sqrt((np.diff(v, axis=0) / dt) ** 2).mean() ** 1 if False else np.sqrt(((np.diff(v, axis=0) / dt) ** 2).sum(1).mean())),
        '몸통 높이 평균 (m)': float(p[:, 2].mean()), '몸통 높이 표준편차 (cm)': float(p[:, 2].std() * 100),
        '최대 |roll| (°)': float(np.degrees(np.abs(e[:, 2]).max())), '최대 |pitch| (°)': float(np.degrees(np.abs(e[:, 1]).max())),
        '관절 속도 RMS (rad/s)': float(np.sqrt((qd ** 2).mean())), '관절 가속 RMS (rad/s²)': float(np.sqrt((qdd ** 2).mean())),
        '관절 한계 여유 최소 (rad)': float(margin.min()),
        '접촉 비율': float(c.mean()), '접촉 발 2 개 미만 비율': float((c.sum(1) < 2).mean()),
        '접촉 전환 (회/s)': float(np.abs(np.diff(c.astype(int), axis=0)).sum() / (len(c) * dt)),
        '레퍼런스 발 미끄러짐 (m/s)': float(fv[inc].mean()) if inc.any() else 0.,
    }


def main():
    res = json.load(open(RUNS / 's6c50_calm/results.json')); rows = []
    for key, r in res.items():
        R = r['results']; n = sum(x['passed'] for x in R)
        if 2 <= n <= 7: continue
        f = RUNS / 's5_retarget/a1_reference' / f"{r['clip']}.npz"
        d = dict(key=key, group=r['group'].replace('__r00', ''), clip=r['clip'], passed=n, label='실패' if n <= 1 else '통과',
                 surv=float(np.mean([x['survived'] for x in R])), jrmse=float(np.mean([x['joint_rmse_deg'] or np.nan for x in R])),
                 rrmse=float(np.mean([x['root_rmse_cm'] for x in R])), feats=feats(f)); rows.append(d)
    json.dump(rows, open(HERE / 'features.json', 'w'), ensure_ascii=False, indent=1)
    names = list(rows[0]['feats']); F = {k: np.array([[r['feats'][n] for n in names] for r in rows if r['label'] == k]) for k in ('실패', '통과')}
    # 효과 크기 (Cohen d) 로 정렬
    def d_of(i):
        a, b = F['실패'][:, i], F['통과'][:, i]; s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12; return (a.mean() - b.mean()) / s
    order = sorted(range(len(names)), key=lambda i: -abs(d_of(i)))
    nc = 6; nr = int(np.ceil(len(names) / nc)); fig, ax = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.9 * nr)); ax = ax.ravel()
    col = {'실패': '#d6336c', '통과': '#2a78d6'}
    for a, i in zip(ax, order):
        for j, k in enumerate(('실패', '통과')):
            y = F[k][:, i]; a.scatter(np.full(len(y), j) + np.random.default_rng(i).uniform(-.12, .12, len(y)), y, color=col[k], s=22)
            a.hlines(np.median(y), j - .25, j + .25, color='k', lw=1.5)
        a.set_xticks([0, 1], ['실패', '통과']); a.set_title(f'{names[i]}\nd = {d_of(i):+.2f}', fontsize=9); a.grid(alpha=.3); a.set_xlim(-.6, 1.6)
    for a in ax[len(names):]: a.axis('off')
    fig.suptitle('실패 (단독 50M 에서 통과 ≤ 1/9) 대 통과 (≥ 8/9) 모션의 레퍼런스 특징. 효과 크기 |d| 큰 순. 검은 선 = 중앙값', fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / 'features.png', dpi=100); plt.close(fig)
    # 실패 원인 (평가 지표)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for a, (k, t, th) in zip(ax, [('jrmse', '관절 RMSE 평균 (°)', 10), ('rrmse', '몸통 위치 RMSE 평균 (cm)', 10), ('surv', '생존 비율', None)]):
        for j, lab in enumerate(('실패', '통과')):
            y = [r[k] for r in rows if r['label'] == lab]; a.scatter(np.full(len(y), j) + np.random.default_rng(j).uniform(-.12, .12, len(y)), y, color=col[lab], s=26)
        if th: a.axhline(th, color='k', ls='--', lw=1)
        a.set_xticks([0, 1], ['실패', '통과']); a.set_title(t); a.grid(alpha=.3); a.set_xlim(-.6, 1.6)
    fig.suptitle('s6c50 평가 지표 (형상 9 개 평균). 점선 = 통과 기준'); fig.tight_layout(); fig.savefig(HERE / 'metrics.png', dpi=100); plt.close(fig)
    # 영상 복사
    vd = HERE / 'videos'; vd.mkdir(exist_ok=True); vids = []
    for r in rows:
        src = list((RUNS / 's6c50_calm/videos' / r['group']).glob(f"*_{r['clip']}.mp4"))
        if src: shutil.copy(src[0], vd / src[0].name); vids.append((r, f'videos/{src[0].name}'))
    from common import report, table, img, video
    tab = table(['결과', '묶음', '모션', '통과 형상', '생존', '관절 RMSE (°)', '몸통 RMSE (cm)'] + names,
                [(r['label'], r['group'], r['clip'], f"{r['passed']}/9", f"{r['surv']:.2f}", f"{r['jrmse']:.1f}", f"{r['rrmse']:.1f}") + tuple(f"{r['feats'][n]:.3g}" for n in names)
                 for r in sorted(rows, key=lambda r: (r['label'], r['group']))], lambda i: '', filter_col=0)
    eff = table(['특징', '실패 중앙값', '통과 중앙값', '효과 크기 d'], [(names[i], f"{np.median(F['실패'][:, i]):.3g}", f"{np.median(F['통과'][:, i]):.3g}", f'{d_of(i):+.2f}') for i in order])
    vh = ''.join(video(p, f"{r['label']} · {r['group']} {r['clip']} ({r['passed']}/9)", tag=r['label']) for r, p in sorted(vids, key=lambda x: x[0]['label']))
    report(HERE / 'report.html', 'CALM 실패 모션 대 통과 모션 특징 비교', [
        ('평가 지표: 무엇 때문에 실패했나', img(HERE / 'metrics.png', 's6c50 결과 (형상 9 개 평균)')),
        ('레퍼런스 특징 비교', img(HERE / 'features.png') + eff),
        ('모션별 표', tab),
        ('영상 (s6c50 정책, 단독 50M)', f'<div class="grid">{vh}</div>')],
        [('대상', f"s6c50 모션 24 개 중 실패 (통과 ≤ 1/9) {len(F['실패'])} 개, 통과 (≥ 8/9) {len(F['통과'])} 개. 중간 (2~7/9) 제외"),
         ('특징 계산', 'A1 레퍼런스 (chain/runs/s5_retarget/a1_reference, 30 Hz, 리타겟 전), 평가 구간과 같은 처음 10 초. 속도는 몸통 기준. 최대 |wz| = 99 백분위'),
         ('효과 크기', 'Cohen d = (실패 평균 − 통과 평균) / 합동 표준편차. 표본이 작아 (9 대 14) 참고용'),
         ('코드', 'check/calm_fail_features/code/features.py')])
    print(HERE / 'report.html')


if __name__ == '__main__':
    main()
