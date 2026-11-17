"""S7 결과에 (2) 수행 가능성 판정, (3) 통과 rollout 의 속도 공간 범위 를 붙인다.
수행 가능성 = s5/s7 리타겟 역동역학: 토크 > 33.5 N·m 또는 유령 힘 > 기준 (inv_bad) 인 프레임 비율.
범위 = A1 레퍼런스 (형상 무관) 를 1 초 창으로 나눠 몸통 기준 (vx, vy, wz) 평균, 격자 0.1 m/s × 0.1 m/s × 0.2 rad/s 칸 수."""
import json, glob, os, numpy as np
C = '/home/cgr-ugrad-2026/quad_ws/codesign/chain/runs'; LIM = 33.5
def ok(r): return bool(r['survived'] and r['joint_rmse_deg'] is not None and r['joint_rmse_deg'] <= 10 and r['root_rmse_cm'] <= 10)
def ref_file(shape, clip):
    for d in (f'{C}/s5_retarget/references/{shape}/00000011', f'{C}/s7_eval/unpaired_refs/{shape}/00000011'):
        if os.path.exists(f'{d}/{clip}.npz'): return f'{d}/{clip}.npz'
def yaw(q): return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
def windows(clip):
    z = np.load(f'{C}/s5_retarget/a1_reference/{clip}.npz'); dt = float(z['dt']); p = z['fix_base_pos']; y = np.unwrap(yaw(z['fix_base_quat']))
    v = np.gradient(p[:, :2], dt, axis=0); c, s = np.cos(y), np.sin(y)
    vb = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], np.gradient(y, dt)], 1)
    n = max(1, int(round(1 / dt)))
    return [vb[i:i + n].mean(0).tolist() for i in range(0, len(vb) - n + 1, n)]
rows = []
for g, v in json.load(open(f'{C}/s7_eval/results.json')).items():
    for r in v['results']:
        f = ref_file(r['shape'], r['clip']); d = dict(group=g, shape=r['shape'], clip=r['clip'], pair=r['pair'], passed=ok(r),
                                                      joint=r['joint_rmse_deg'], calm=r['clip'].startswith('calm'))
        if f:
            z = np.load(f); tau = np.abs(z['inv_tau'][:, :12])
            d.update(tau_frac=float((tau > LIM).any(1).mean()), bad_frac=float(z['inv_bad'].mean()))
        d['win'] = windows(r['clip']); rows.append(d)
allcalm = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{C}/s5_retarget/a1_reference/calm_*.npz'))
pool = {c: windows(c) for c in allcalm[::10]}  # 전체 CALM 10 개 중 1 개 (분포 배경)
json.dump(dict(rows=rows, pool=pool), open('data.json', 'w'))
print(len(rows), 'rows', sum('tau_frac' in r for r in rows), 'with feasibility')
