import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE = Path(__file__).resolve().parent
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False, 'axes.spines.right': False})
S1, S2, INK, MUTED, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#fcfcfb'
z = np.load(HERE / 'tracking_errors.npz'); dt = float(z['dt'])
J, P, S, SR = z['joint_deg'], z['root_m'], z['slip_m_s'], z['ref_slip_m_s']
T = 500; t = np.arange(T) * dt
def band(ax, a, color, label):
    a = a[:, :T]; ok = (~np.isnan(a)).sum(0) >= 10
    lo, md, hi = np.nanpercentile(a, [25, 50, 75], axis=0)
    ax.fill_between(t[ok], lo[ok], hi[ok], color=color, alpha=.18, linewidth=0)
    ax.plot(t[ok], md[ok], color=color, linewidth=2, label=label)
    return md[ok][-1]
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURF)
for ax in axs.flat: ax.set_facecolor(SURF); ax.grid(axis='y', color='#e6e5e0', linewidth=.8)
ax = axs[0, 0]; band(ax, J, S1, 'TT 실제 실행'); ax.set_title('관절 추종 오차 (12 관절 RMSE)', color=INK, loc='left'); ax.set_ylabel('deg'); ax.set_xlabel('시간 (s)')
ax = axs[0, 1]; band(ax, P, S1, 'TT 실제 실행'); ax.set_title('몸통 위치 추종 오차 (레퍼런스 몸통 위치와 거리)', color=INK, loc='left'); ax.set_ylabel('m'); ax.set_xlabel('시간 (s)')
ax = axs[1, 0]; band(ax, S, S1, 'TT 실제 실행'); band(ax, SR, S2, '리타겟 레퍼런스'); ax.set_title('접지한 발의 수평 미끄러짐 속도', color=INK, loc='left')
ax.set_ylabel('m/s'); ax.set_xlabel('시간 (s)'); ax.legend(frameon=False, loc='upper right')
ax = axs[1, 1]
ms, pe = np.nanmean(S, 1), np.array([row[~np.isnan(row)][-1] for row in P])
f = z['fell'].astype(bool)
ax.scatter(ms[~f], pe[~f], s=36, color=S1, edgecolor=SURF, linewidth=1.5, label='끝까지 실행')
ax.scatter(ms[f], pe[f], s=36, color=S2, edgecolor=SURF, linewidth=1.5, label='넘어짐')
ax.set_title('에피소드별: 평균 미끄러짐 vs 마지막 몸통 위치 오차', color=INK, loc='left'); ax.set_xlabel('평균 미끄러짐 (m/s)'); ax.set_ylabel('마지막 위치 오차 (m)')
ax.legend(frameon=False, loc='upper right')
fig.suptitle('e0000 PPO 트래킹 데이터 (모션 100개, 50 Hz) · 선 = 중앙값, 띠 = 25–75 %, 넘어진 뒤 구간 제외', x=.01, ha='left', color=MUTED, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(HERE / 'tracking_errors.png', dpi=130, facecolor=SURF)
