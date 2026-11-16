"""heading_sweep 방식별 지표 (12 모션 평균): wz RMSE, 회전 떨림 (wz 시간미분 RMS), 명령 없는 회전 (|명령 wz|<0.2 인데 |실제|>0.8 인 시간 비율),
방향각 오차 (명령 적분 대비, 1 초 창 드리프트 제거 없이 최대), vx·vy RMSE. 출력: heading_sweep/metrics.json"""
import json, sys
from pathlib import Path
import numpy as np
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; sys.path.insert(0, str(CODE_DIR))
from render import body_vel
from calm_joystick import commands_for
from scipy.spatial.transform import Rotation
S = HERE / 'heading_sweep'; plans = {m['id']: m for m in json.load(open(S / 'plan.json'))['motions']}; out = {}
for d in ['int03', 'int02', 'int05', 'step10', 'step05', 'sabs10', 'sabs05', 'sabs033']:
    acc = []
    for f in sorted((S / d / 'accepted').glob('*.npz')):
        z = np.load(f); dt = float(z['dt']); c = commands_for(plans[int(f.stem)]); v = body_vel(z['root'], dt); n = min(len(v), len(c)); v, c = v[:n], c[:n]
        yaw = np.unwrap(Rotation.from_quat(z['root'][:, 3:7]).as_euler('zyx')[:, 0])[:n]; he = (yaw - yaw[0]) - np.cumsum(c[:, 2]) * dt
        acc.append([np.sqrt(np.mean((v[:, 2] - c[:, 2]) ** 2)), np.sqrt(np.mean((np.diff(v[:, 2]) / dt) ** 2)),
                    np.mean((np.abs(c[:, 2]) < .2) & (np.abs(v[:, 2]) > .8)), np.abs(he - np.median(he[:30])).max(),
                    np.sqrt(np.mean((v[:, 0] - c[:, 0]) ** 2)), np.sqrt(np.mean((v[:, 1] - c[:, 1]) ** 2))])
    a = np.array(acc); out[d] = dict(n=len(a), mean=a.mean(0).round(3).tolist(), worst=a.max(0).round(3).tolist())
    print(d, len(a), 'mean', a.mean(0).round(3), 'worst', a.max(0).round(3))
json.dump(dict(cols=['wz RMSE', 'wz 떨림 (rad/s²)', '명령 없는 회전 비율', '방향각 오차 최대 (rad)', 'vx RMSE', 'vy RMSE'], res=out), open(S / 'metrics.json', 'w'), ensure_ascii=False, indent=1)
