"""CALM 모션 10,000 개를 A1 레퍼런스 모션 25 개 중 가장 가까운 것에 하나도 빠짐없이 배정한다.

CALM 모션은 기본 동작 구간 (motion_index, segment_start/end) 을 이어 붙인 것이라, 구간마다 가장 가까운 레퍼런스를 찾고
모션은 가장 긴 시간을 차지하는 레퍼런스에 배정한다 (모션 단위 배정, 빠짐없음). 구간 단위 개수도 함께 기록.
특징 (구간 또는 레퍼런스 전체, A1 기준): 몸통 heading 기준 앞뒤·옆 속도, yaw 회전 속도, 몸통 높이의 평균·표준편차 (8)
+ 관절 12 개 각도의 평균·표준편차 (24). 각 특징을 전체 (10,025 개) 표준편차로 나눈 뒤 유클리드 거리 최근접.
관절 순서 FR, FL, RR, RL × hip, thigh, calf (레퍼런스 fix_q 와 같음). 출력: assignment.json, counts.csv
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
CALM = ROOT / 'legacy/runs/calm_motion_10,000/dataset/accepted'
REF = ROOT / 'legacy/artifacts/randomized_reference'


def feat(pos, quat_xyzw, q, dt):
    yaw = Rotation.from_quat(quat_xyzw).as_euler('zyx')[:, 0]
    v = np.diff(pos[:, :2], axis=0) / dt; c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    vx, vy = c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]
    wz = np.diff(np.unwrap(yaw)) / dt
    base = np.array([vx.mean(), vy.mean(), wz.mean(), pos[:, 2].mean(), vx.std(), vy.std(), wz.std(), pos[:, 2].std()])
    return np.r_[base, q.mean(0), q.std(0)]


refs, rf = [], []
for f in sorted(REF.glob('*.npz')):
    z = np.load(f); refs.append(f.stem)
    rf.append(feat(z['fix_base_pos'], z['fix_base_quat'][:, [1, 2, 3, 0]], z['fix_q'], float(z['dt'])))
ids, segs = [], []   # segs: (motion index, skill id, frames, feature)
for f in sorted(CALM.glob('*.npz')):
    z = np.load(f); names = z['joint_names'].tolist()
    order = [names.index(l + '_' + j + '_joint') for l in ('FR', 'FL', 'RR', 'RL') for j in ('hip', 'thigh', 'calf')]
    dt = float(np.diff(z['time']).mean()); root, q = z['root'], z['q'][:, order]
    for k, a0, b0 in zip(z['motion_index'], z['segment_start'], z['segment_end']):
        a0, b0 = int(a0), int(b0) + 1
        if b0 - a0 < 6: continue
        segs.append((len(ids), int(k), b0 - a0, feat(root[a0:b0, :3], root[a0:b0, 3:7], q[a0:b0], dt)))
    ids.append(int(f.stem))
rf = np.array(rf); sf = np.array([s[3] for s in segs])
scale = np.r_[rf, sf].std(0) + 1e-9
seg_best = np.linalg.norm((sf[:, None] - rf[None]) / scale, axis=-1).argmin(1)
votes = np.zeros((len(ids), len(refs)))
for (mi, _, n, _), b in zip(segs, seg_best): votes[mi, b] += n
assert (votes.sum(1) > 0).all(), 'every motion needs at least one segment'
best = votes.argmax(1)
assign = {refs[k]: sorted(int(ids[i]) for i in np.flatnonzero(best == k)) for k in range(len(refs))}
seg_counts = {refs[k]: int((seg_best == k).sum()) for k in range(len(refs))}
skill_to_ref = {}
for (mi, sk, n, _), b in zip(segs, seg_best): skill_to_ref.setdefault(sk, np.zeros(len(refs)))[b] += n
d = np.zeros((1, 1))
assert sum(len(v) for v in assign.values()) == len(ids) == 10000, 'every motion must be assigned'
json.dump(dict(rule=__doc__.strip().splitlines()[0], features='vx,vy,wz,z mean/std + 12 joint mean/std, scaled by std',
               references=refs, assignment=assign, segment_counts=seg_counts,
               skill_to_reference_frames={int(k): {refs[i]: int(v[i]) for i in np.flatnonzero(v)} for k, v in sorted(skill_to_ref.items())}), open(HERE / 'assignment.json', 'w'), indent=1)
with open(HERE / 'counts.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['reference', 'motions', 'segments', 'ref_vx', 'ref_vy', 'ref_wz'])
    for k, r in enumerate(refs): w.writerow([r, len(assign[r]), seg_counts[r], *np.round(rf[k, :3], 3)])
for k, r in enumerate(refs): print(f'{r:24s} motions {len(assign[r]):5d}  segments {seg_counts[r]:5d}   ref vx {rf[k,0]:+.2f} vy {rf[k,1]:+.2f} wz {rf[k,2]:+.2f}')
print('total', sum(len(v) for v in assign.values()))

NAMES = ['forward','backward','left','right','cw_turn','ccw_turn','left_then_forward','right_then_forward','left_then_backward','right_then_backward','sit','stand_up']
for k, v in sorted(skill_to_ref.items()):
    top = np.argsort(v)[::-1][:3]; print(f'skill {NAMES[k]:20s} → ' + ', '.join(f'{refs[i]} {v[i] / v.sum() * 100:.0f}%' for i in top))
