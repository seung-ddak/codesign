"""관절별 토크 초과 + 모션 속도와 토크 초과의 관계. rows.json 의 12,000 쌍 (s5 리타겟, 학습 형상 9)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'axes.unicode_minus': False, 'font.size': 10})
R0 = '/home/cgr-ugrad-2026/quad_ws/codesign/chain/runs/s5_retarget'; LIM = 33.5; INK, GRID = '#2b2b29', '#e6e5e1'
rows = json.load(open('rows.json'))
LEG = ['앞 왼 FL', '앞 오른 FR', '뒤 왼 RL', '뒤 오른 RR']; JT = ['hip', 'thigh', 'calf']
over = np.zeros(12); frames = 0; speed = {}
for r in rows:
    z = np.load(f"{R0}/references/{r['shape']}/00000011/{r['motion']}.npz"); t = np.abs(z['inv_tau'][:, :12])
    over += (t > LIM).sum(0); frames += len(t)
    if r['motion'] not in speed:
        a = np.load(f"{R0}/a1_reference/{r['motion']}.npz"); v = np.gradient(a['fix_base_pos'][:, :2], float(a['dt']), axis=0)
        speed[r['motion']] = float(np.linalg.norm(v, axis=1).mean())
    r['speed'] = speed[r['motion']]
json.dump(rows, open('rows.json', 'w'))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
M = (over / frames * 100).reshape(4, 3)
im = ax[0].imshow(M, cmap='Blues', vmin=0); ax[0].set_xticks(range(3), JT); ax[0].set_yticks(range(4), LEG)
for i in range(4):
    for j in range(3): ax[0].text(j, i, f'{M[i, j]:.2f} %', ha='center', va='center', color='white' if M[i, j] > M.max() * .55 else INK)
ax[0].set_title(f'관절별: |토크| > {LIM} N·m 인 프레임 비율 (전체 프레임 대비)', fontsize=10.5); fig.colorbar(im, ax=ax[0], shrink=.8)
# 속도 구간별 초과 모션 비율
bins = [0, .1, .2, .3, .4, .5, .6, .8, 1.2]; sp = np.array([r['speed'] for r in rows]); fr = np.array([r['over_frac'] for r in rows])
x = np.arange(len(bins) - 1); lab = [f'{bins[i]:.1f}–{bins[i + 1]:.1f}' for i in x]
for k, (th, c, name) in enumerate([(0, '#86b6ef', '한 프레임이라도 초과'), (.01, '#2a78d6', '초과 프레임 >1 %'), (.05, '#104281', '>5 %')]):
    y = [((fr > th) & (sp >= bins[i]) & (sp < bins[i + 1])).sum() / max(1, ((sp >= bins[i]) & (sp < bins[i + 1])).sum()) * 100 for i in x]
    ax[1].plot(x, y, '-o', color=c, lw=2, ms=6, label=name)
n = [((sp >= bins[i]) & (sp < bins[i + 1])).sum() for i in x]
ax[1].set_xticks(x, [f'{l}\n(n={m})' for l, m in zip(lab, n)], fontsize=8.5); ax[1].set_xlabel('모션 평균 이동 속도 (m/s, A1 레퍼런스 기준)')
ax[1].set_ylabel('토크 초과 모션 비율 (%)'); ax[1].set_ylim(0, 105); ax[1].legend(frameon=False)
ax[1].set_title('모션 속도별 토크 초과 비율', fontsize=10.5); ax[1].grid(color=GRID); ax[1].set_axisbelow(True)
for s in ('top', 'right'): ax[1].spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('joints_speed.png', dpi=130)
print(np.round(M, 2)); print('n by speed', n)
