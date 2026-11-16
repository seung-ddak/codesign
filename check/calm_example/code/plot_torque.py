"""전환 순간 토크 대 회전: 예시 3 개 (collect/accepted) 마다 시간축 공유 4 줄.
(1) 회전 wz 명령·실제 (2) 이동 vx·vy 명령·실제 (3) 관절 토크 (hip 4 개 + 다리 12 개 |τ| 합) (4) CALM 상위 정책 latent 변화량 (1 - 이전 step 과의 코사인).
회전 튐 = |실제 wz - 명령 wz| > 0.5 rad/s 구간 (빨간 띠). 세로선 = 구간 경계 (전환 시작). 출력: torque_vs_turn.png"""
import json, sys
from pathlib import Path
import numpy as np
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_DIR)); sys.path.insert(0, str(ROOT / 'chain/code'))
from render import body_vel, plan
from calm_joystick import commands_for
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus'] = False

files = sorted((HERE / 'collect/accepted').glob('*.npz'))
fig, ax = plt.subplots(4, len(files), figsize=(7 * len(files), 11), sharex='col', squeeze=False)
for c, f in enumerate(files):
    pid = int(f.stem); z = np.load(f); dt = float(z['dt']); cmd = commands_for(plan[pid]); v = body_vel(z['root'], dt)
    tau = z['joint_force_sensor']; jn = list(z['joint_names']); lat = z['latent']; n = min(len(cmd), len(v), len(tau)); t = np.arange(n) * dt
    bad = np.abs(v[:n, 2] - cmd[:n, 2]) > 0.5
    lchg = np.r_[0, 1 - np.sum(lat[1:] * lat[:-1], 1) / (np.linalg.norm(lat[1:], axis=1) * np.linalg.norm(lat[:-1], axis=1) + 1e-9)][:n]
    a = ax[0, c]; a.plot(t, cmd[:n, 2], color='#e8590c', label='wz 명령'); a.plot(t, v[:n, 2], color='#2a78d6', label='wz 실제'); a.set_ylabel('회전 (rad/s)')
    a.set_title(f'예시 {pid}'); a.legend(fontsize=8)
    a = ax[1, c]
    for j, (nm, col) in enumerate([('vx', '#1baa6b'), ('vy', '#8e44ad')]):
        a.plot(t, cmd[:n, j], color=col, ls='--', lw=1, label=f'{nm} 명령'); a.plot(t, v[:n, j], color=col, label=f'{nm} 실제')
    a.set_ylabel('이동 (m/s)'); a.legend(fontsize=8, ncol=2)
    a = ax[2, c]
    for leg, col in zip(['FL', 'FR', 'RL', 'RR'], ['#2a78d6', '#e8590c', '#1baa6b', '#8e44ad']):
        a.plot(t, tau[:n, jn.index(f'{leg}_hip_joint')], color=col, lw=1, label=f'{leg} hip')
    a2 = a.twinx(); a2.plot(t, np.abs(tau[:n]).sum(1), color='k', lw=.8, alpha=.5, label='|τ| 합 (12 관절)'); a2.set_ylabel('|τ| 합 (N·m)')
    a.set_ylabel('hip 토크 (N·m)'); a.legend(fontsize=8, ncol=4, loc='upper left'); a2.legend(fontsize=8, loc='upper right')
    a = ax[3, c]; a.plot(t, lchg, color='#555'); a.set_ylabel('latent 변화 (1-cos)'); a.set_xlabel('시간 (s)')
    for r in range(4):
        aa = ax[r, c]; aa.grid(alpha=.3)
        for s in plan[pid]['segments']: aa.axvline(s['start'] * dt, color='#999', lw=.7)
        for i0 in np.flatnonzero(np.diff(np.r_[0, bad.astype(int)]) == 1):
            i1 = i0 + np.argmax(~bad[i0:]) if (~bad[i0:]).any() else n; aa.axvspan(i0 * dt, i1 * dt, color='red', alpha=.12)
    print(pid, 'spikes at', [round(i * dt, 2) for i in np.flatnonzero(np.diff(np.r_[0, bad.astype(int)]) == 1)],
          'segment starts', [round(s['start'] * dt, 2) for s in plan[pid]['segments']], flush=True)
fig.tight_layout(); fig.savefig(HERE / 'torque_vs_turn.png', dpi=95); plt.close(fig); print(HERE / 'torque_vs_turn.png')
