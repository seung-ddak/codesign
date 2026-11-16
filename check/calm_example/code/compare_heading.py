"""예시 0 을 같은 조이스틱 명령으로 회전 명령 변환 방식 3 가지로 만든 결과 비교: lead (수정 전), integrate (적분, 지금), step (계단형 목표, 1 초마다).
그림: compare_heading.png (wz · 방향각 · vx · vy, 방식별 색). 영상: videos/calm_example_0_step.mp4 (step 방식)."""
import json, sys
from pathlib import Path
import numpy as np
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
import render as R
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus'] = False
RUNS = [('lead (수정 전: 현재+wz×0.5s)', 'collect_lead', '#999999'), ('integrate (적분, 지금)', 'collect', '#2a78d6'), ('step (계단형, 1 초마다)', 'collect_step', '#1baa6b')]
pid = 0; plan = R.plan[pid]; cmd = R.commands_for(plan); dt = 1 / 30; t = np.arange(len(cmd)) * dt
fig, ax = plt.subplots(4, 1, figsize=(14, 13), sharex=True); stats = []
ax[0].plot(t, cmd[:, 2], color='#e8590c', lw=2, label='명령')
ax[1].plot(t, np.cumsum(cmd[:, 2]) * dt, color='#e8590c', lw=2, label='명령 적분 (목표 방향)')
for j in (0, 1): ax[2 + j].plot(t, cmd[:, j], color='#e8590c', lw=2, label='명령')
for lab, d, col in RUNS:
    z = np.load(HERE / d / 'accepted' / f'{pid:05d}.npz'); v = R.body_vel(z['root'], float(z['dt'])); n = min(len(v), len(cmd))
    yaw = np.unwrap(Rotation.from_quat(z['root'][:, 3:7]).as_euler('zyx')[:, 0]); yaw -= yaw[0]
    ax[0].plot(t[:n], v[:n, 2], color=col, label=lab); ax[1].plot(np.arange(len(yaw)) * dt, yaw, color=col, label=lab)
    for j in (0, 1): ax[2 + j].plot(t[:n], v[:n, j], color=col, label=lab)
    e = v[:n] - cmd[:n]; head_err = yaw[:n] - np.cumsum(cmd[:n, 2]) * dt
    stats.append((lab, f'{np.sqrt(np.mean(e[:, 2] ** 2)):.2f}', f'{np.abs(e[:, 2]).max():.2f}', f'{np.abs(head_err).max():.2f}', f'{abs(head_err[-1]):.2f}',
                  f'{np.sqrt(np.mean(e[:, 0] ** 2)):.3f}', f'{np.sqrt(np.mean(e[:, 1] ** 2)):.3f}'))
for a, tt in zip(ax, ['회전 wz (rad/s)', '방향각 (rad, 시작 = 0)', '전후 vx (m/s)', '좌우 vy (m/s)']):
    a.set_ylabel(tt); a.grid(alpha=.3)
    for sg in plan['segments']: a.axvline(sg['start'] * dt, color='#bbb', lw=.7)
ax[0].legend(fontsize=9, ncol=4); ax[1].legend(fontsize=9); ax[-1].set_xlabel('시간 (s)')
fig.tight_layout(); fig.savefig(HERE / 'compare_heading.png', dpi=95); plt.close(fig)
json.dump(stats, open(HERE / 'compare_heading.json', 'w'), ensure_ascii=False, indent=1)
for s in stats: print(s)
if __name__ == '__main__' and '--video' in sys.argv:
    print(R.render(HERE / 'collect_step/accepted/00000.npz', pid, tag='_step', note=' [step target]')[2])
