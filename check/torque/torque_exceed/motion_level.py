"""모션 기준 6,000 개: 모션마다 배정된 형상 2 개 중 더 나쁜 값 (초과 프레임 비율·최대 토크 = 최댓값)."""
import json, collections, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'axes.unicode_minus': False, 'font.size': 10})
INK, GRID, LIM = '#2b2b29', '#e6e5e1', 33.5
by = collections.defaultdict(list)
for r in json.load(open('rows.json')): by[r['motion']].append(r)
M = [dict(motion=k, group=v[0]['group'], speed=v[0]['speed'], over=max(x['over_frac'] for x in v), tau=max(x['tau_max'] for x in v),
          nshape=sum(x['over_frac'] > 0 for x in v)) for k, v in by.items()]
def clean(ax, axis='x'):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID); ax.set_axisbelow(True)
BK = [('초과 없음', lambda f: f == 0, '#d6d5d0'), ('프레임 ≤1 %', lambda f: 0 < f <= .01, '#86b6ef'),
      ('1–5 %', lambda f: .01 < f <= .05, '#2a78d6'), ('>5 %', lambda f: f > .05, '#104281')]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
G = sorted({m['group'] for m in M}) + ['전체']; left = np.zeros(len(G))
for name, fn, c in BK:
    v = np.array([sum(fn(m['over']) for m in M if g == '전체' or m['group'] == g) for g in G])
    ax[0].barh(range(len(G)), v, left=left, color=c, label=name, height=.62, edgecolor='white', linewidth=2); left += v
for i, g in enumerate(G):
    z = [m for m in M if g == '전체' or m['group'] == g]; o = sum(m['over'] > 0 for m in z); o1 = sum(m['over'] > .01 for m in z)
    ax[0].text(left[i] * 1.01 + 20, i, f'초과 {o}/{len(z)} ({o / len(z):.0%}) · >1 % {o1 / len(z):.0%}', va='center', fontsize=9, color=INK)
ax[0].set_yticks(range(len(G)), [g.replace('__r00', '') for g in G]); ax[0].invert_yaxis(); ax[0].set_xlim(0, left.max() * 1.55)
ax[0].set_xlabel('모션 수'); ax[0].set_title('레퍼런스 묶음별 (형상 2 개 중 더 나쁜 쪽)', fontsize=10.5); clean(ax[0])
ax[0].legend(title=f'|토크| > {LIM} N·m 인 프레임 비율', frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(.45, -.4))
cnt = [sum(m['nshape'] == k for m in M) for k in (0, 1, 2)]
ax[1].bar(range(3), cnt, .6, color=['#d6d5d0', '#86b6ef', '#104281'])
for i, c in enumerate(cnt): ax[1].text(i, c + 40, f'{c} ({c / len(M):.0%})', ha='center', color=INK)
ax[1].set_xticks(range(3), ['둘 다 초과 없음', '한 형상만 초과', '두 형상 모두 초과']); ax[1].set_ylabel('모션 수')
ax[1].set_title('배정된 형상 2 개 중 초과한 형상 수', fontsize=10.5); clean(ax[1], 'y')
fig.suptitle(f'모션 기준 {len(M)} 개: 토크 한계 {LIM} N·m 초과', fontsize=12.5); fig.tight_layout(); fig.savefig('motion_counts.png', dpi=130)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2)); t = np.array([m['tau'] for m in M])
ax[0].hist(np.minimum(t, 120), bins=np.arange(0, 122, 2), color='#2a78d6', edgecolor='white', linewidth=1)
ax[0].axvline(LIM, color=INK, ls='--', lw=1.5); ax[0].text(LIM + 1, ax[0].get_ylim()[1] * .9, f'한계 {LIM}', color=INK)
ax[0].set_xlabel('모션별 최대 |토크| (N·m, 형상 2 개 중 큰 값, 120 이상은 마지막 칸)'); ax[0].set_ylabel('모션 수'); ax[0].set_title('최대 토크 분포', fontsize=10.5); clean(ax[0], 'y')
bins = [0, .1, .2, .3, .4, .5, .6, .8, 1.2]; sp = np.array([m['speed'] for m in M]); fr = np.array([m['over'] for m in M]); x = np.arange(len(bins) - 1)
n = [((sp >= bins[i]) & (sp < bins[i + 1])).sum() for i in x]
for th, c, name in [(0, '#86b6ef', '한 프레임이라도 초과'), (.01, '#2a78d6', '초과 프레임 >1 %'), (.05, '#104281', '>5 %')]:
    ax[1].plot(x, [((fr > th) & (sp >= bins[i]) & (sp < bins[i + 1])).sum() / max(1, n[i]) * 100 for i in x], '-o', color=c, lw=2, ms=6, label=name)
ax[1].set_xticks(x, [f'{bins[i]:.1f}–{bins[i + 1]:.1f}\n(n={n[i]})' for i in x], fontsize=8.5); ax[1].set_xlabel('모션 평균 이동 속도 (m/s)')
ax[1].set_ylabel('토크 초과 모션 비율 (%)'); ax[1].set_ylim(0, 105); ax[1].legend(frameon=False); ax[1].set_title('속도별 토크 초과 비율', fontsize=10.5); clean(ax[1], 'both')
fig.tight_layout(); fig.savefig('motion_tau_speed.png', dpi=130)
f = np.array([m['over'] for m in M]); print(len(M), 'any', (f > 0).sum(), '>1%', (f > .01).sum(), '>5%', (f > .05).sum(), 'nshape', cnt, 'median tau', np.median(t))
for g in G[:-1]:
    z = [m for m in M if m['group'] == g]; print(g, len(z), sum(m['over'] > 0 for m in z), sum(m['over'] > .01 for m in z))
print('n', n)
