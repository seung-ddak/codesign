"""rows.json → 형상별·묶음별 토크 초과 모션 수 (프레임 비율 단계) + 최대 토크 분포."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'axes.unicode_minus': False, 'font.size': 10})
R = json.load(open('rows.json')); LIM = 33.5
BK = [('초과 없음', lambda f: f == 0, '#d6d5d0'), ('프레임 ≤1 %', lambda f: 0 < f <= .01, '#86b6ef'),
      ('1–5 %', lambda f: .01 < f <= .05, '#2a78d6'), ('>5 %', lambda f: f > .05, '#104281')]
def stack(ax, keys, key):
    left = np.zeros(len(keys))
    for name, fn, c in BK:
        v = np.array([sum(fn(r['over_frac']) for r in R if r[key] == k) for k in keys])
        ax.barh(range(len(keys)), v, left=left, color=c, label=name, height=.62, edgecolor='white', linewidth=2)
        left += v
    for i, k in enumerate(keys):
        n = sum(r[key] == k for r in R); o = sum(r['over_frac'] > 0 for r in R if r[key] == k)
        ax.text(left[i] + n * .01, i, f'{o}/{n} ({o / n:.0%})', va='center', color='#3a3a38', fontsize=9)
    ax.set_yticks(range(len(keys)), keys); ax.invert_yaxis(); ax.set_xlim(0, left.max() * 1.22)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#e6e5e1'); ax.set_axisbelow(True)
shapes = sorted({r['shape'] for r in R}); groups = sorted({r['group'] for r in R})
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
stack(ax[0], shapes, 'shape'); ax[0].set_title('형상별 (모션 수, 오른쪽 = 초과 모션 / 전체)'); ax[0].set_xlabel('모션 수')
stack(ax[1], groups, 'group'); ax[1].set_title('레퍼런스 묶음별 (9 형상 합)'); ax[1].set_xlabel('모션 × 형상 수')
ax[0].legend(title=f'|토크| > {LIM} N·m 인 프레임 비율', loc='lower center', bbox_to_anchor=(1.1, -0.28), ncol=4, frameon=False)
fig.suptitle(f's5 리타겟 역동역학 다리 토크 한계 {LIM} N·m 초과 (CALM 모션, 학습 형상 9 개)', fontsize=13); fig.tight_layout()
fig.savefig('exceed_counts.png', dpi=130, bbox_inches='tight')
fig, ax = plt.subplots(figsize=(8, 3.6)); t = np.array([r['tau_max'] for r in R])
ax.hist(t, bins=np.arange(0, min(t.max(), 120) + 2, 2), color='#2a78d6', edgecolor='white', linewidth=1)
ax.axvline(LIM, color='#3a3a38', ls='--', lw=1.5); ax.text(LIM + 1, ax.get_ylim()[1] * .9, f'한계 {LIM}', color='#3a3a38')
ax.set_xlabel('모션별 최대 |토크| (N·m)'); ax.set_ylabel('모션 × 형상 수'); ax.set_title('최대 토크 분포 (120 이상은 잘림)')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('tau_max_hist.png', dpi=130)
f = np.array([r['over_frac'] for r in R])
print('pairs', len(R), 'any', (f > 0).sum(), '>1%', (f > .01).sum(), '>5%', (f > .05).sum(), 'tau_max median', np.median(t), 'max', t.max())
