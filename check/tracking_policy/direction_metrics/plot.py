import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'axes.unicode_minus': False, 'font.size': 10})
D = json.load(open('data.json')); R = D['rows']; INK, MUT, GRID = '#2b2b29', '#6b6a65', '#e6e5e1'
def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
# ---- 2: 수행 가능성별 통과율
BK = [('0 %', lambda f: f == 0), ('≤1 %', lambda f: 0 < f <= .01), ('1–5 %', lambda f: .01 < f <= .05), ('>5 %', lambda f: f > .05)]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, key, title in [(axs[0], 'bad_frac', '불가능 프레임 비율 (토크 > 33.5 N·m 또는 유령 힘 > 체중 10 %)'), (axs[1], 'tau_frac', '토크 초과 프레임 비율만')]:
    for j, (sub, col, lab) in enumerate([([r for r in R if r['calm']], '#2a78d6', 'CALM'), ([r for r in R if not r['calm']], '#eb6834', 'A1 레퍼런스')]):
        for i, (b, fn) in enumerate(BK):
            z = [r for r in sub if fn(r[key])]
            if not z: continue
            p = sum(r['passed'] for r in z) / len(z); x = i + (j - .5) * .38
            ax.bar(x, p * 100, .36, color=col, label=lab if i == next(k for k, (bb, ff) in enumerate(BK) if any(ff(r[key]) for r in sub)) else None)
            ax.text(x, p * 100 + 1.5, f'{sum(r["passed"] for r in z)}/{len(z)}', ha='center', fontsize=8.5, color=INK)
    ax.set_xticks(range(4), [b for b, _ in BK]); ax.set_ylim(0, 112); ax.set_ylabel('S7 통과율 (%)'); ax.set_title(title, fontsize=10.5); clean(ax)
axs[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.0), ncol=2)
fig.suptitle('지표 2: 수행 가능성별 모션 추종 통과율 (S7, 묶음별 공유 정책 100M, 324 개)', fontsize=12.5); fig.tight_layout(); fig.savefig('m2_feasible_pass.png', dpi=130)
# ---- 3: 통과 rollout 의 속도 공간 범위
cell = np.array([.1, .1, .2])
def cells(ws): return {tuple(np.floor(np.asarray(w) / cell).astype(int)) for w in ws}
refp = [w for r in R if not r['calm'] and r['passed'] for w in r['win']]
calp = [w for r in R if r['calm'] and r['passed'] for w in r['win']]
calf = [w for r in R if r['calm'] and not r['passed'] for w in r['win']]
pool = [w for ws in D['pool'].values() for w in ws]
feas = [w for r in R if r['calm'] and r['bad_frac'] <= .01 for w in r['win']]
fig = plt.figure(figsize=(14, 4.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, .8])
for k, (a, b, xl, yl) in enumerate([(0, 1, '전진 속도 vx (m/s)', '옆 속도 vy (m/s)'), (0, 2, '전진 속도 vx (m/s)', '회전 속도 wz (rad/s)')]):
    ax = fig.add_subplot(gs[k])
    for pts, c, s, lab, al in [(pool, '#d6d5d0', 4, 'CALM 전체 (10 % 표본)', .5), (calf, '#f2a679', 9, 'CALM 실패', .8),
                                (calp, '#2a78d6', 9, 'CALM 통과', .9), (refp, INK, 40, 'A1 레퍼런스 통과', 1)]:
        P = np.asarray(pts); ax.scatter(P[:, a], P[:, b], s=s, c=c, alpha=al, lw=0, label=lab, marker='*' if c == INK else 'o')
    ax.set_xlabel(xl); ax.set_ylabel(yl); clean(ax); ax.grid(axis='x', color=GRID)
    if k == 0: ax.legend(frameon=False, fontsize=8.5, loc='upper left', markerscale=1.5)
ax = fig.add_subplot(gs[2]); B = [('A1 레퍼런스\n통과', len(cells(refp)), INK), ('A1 + CALM\n통과', len(cells(refp) | cells(calp)), '#2a78d6'),
                                  ('평가한 CALM\n전체', len(cells(calp) | cells(calf)), '#86b6ef')]
ax.bar(range(3), [b[1] for b in B], .6, color=[b[2] for b in B])
for i, b in enumerate(B): ax.text(i, b[1] * 1.02 + 1, str(b[1]), ha='center', color=INK)
ax.set_xticks(range(3), [b[0] for b in B]); ax.set_ylabel('덮은 칸 수 (0.1 m/s × 0.1 m/s × 0.2 rad/s)'); clean(ax); ax.set_title('속도 공간에서 덮은 칸', fontsize=10.5)
fig.suptitle('지표 3: 통과한 rollout 이 덮는 속도 공간 (1 초 창 평균, 몸통 기준)', fontsize=12.5); fig.tight_layout(); fig.savefig('m3_coverage.png', dpi=130)
cal = [r for r in R if r['calm']]
f = [r for r in cal if r['bad_frac'] <= .01]
print('calm all %d/%d, feasible(<=1%%) %d/%d, infeasible %d/%d' % (sum(r['passed'] for r in cal), len(cal), sum(r['passed'] for r in f), len(f),
      sum(r['passed'] for r in cal if r['bad_frac'] > .01), len(cal) - len(f)))
print('cells ref', len(cells(refp)), 'ref+calm', len(cells(refp) | cells(calp)), 'calm eval', len(cells(calp) | cells(calf)), 'calm feasible', len(cells(feas)))
print('speed range |v| ref max %.2f calm passed max %.2f' % (np.abs(np.asarray(refp)[:, :2]).max(), np.abs(np.asarray(calp)[:, :2]).max()))
