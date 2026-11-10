"""v7 (wide16_action, 모션별 PPO 전문가) 데이터의 발 미끄러짐·관절·몸통 위치 오차. e0000 분석과 같은 방식."""
import json, sys
from pathlib import Path
import numpy as np, mujoco
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'code/motion_tracking'))
from pd_replay import build_model
V7 = ROOT / 'runs/pipeline_v1'
out = {}; allS, allSR, allJ, allP5, per_motion = [], [], [], [], {}
for f in sorted((V7 / 'wide16_action/rollouts').glob('e*/primitive_seed0.npz')):
    with np.load(f, allow_pickle=True) as z: r = {k: z[k] for k in z.files}
    design = str(r['design']); dt = float(r['dt'])
    m = build_model('00000011', uniforms=list(r['E']), torque_limit=33.5); d = mujoco.MjData(m)
    feet = [m.geom(f'{l}_foot').id for l in ('FR', 'FL', 'RR', 'RL')]
    refs = {i: np.load(V7 / f'retargeting/{design}/00000011/{name}.npz') for i, name in enumerate(r['motions'])}
    for e in range(len(r['valid'])):
        n = int(r['valid'][e].sum())
        if n < 10 or r['switched'][e][:n].any():
            continue
        c0 = int(r['clip'][e, 0]); ref = refs[c0]
        f_loc = np.clip(r['ref_frame'][e][:n] - int(r['clip_start'][c0]), 0, len(ref['fix_q']) - 1)
        q = r['qpos'][e, 1:n + 1]
        J = np.degrees(np.sqrt(((q[:, 7:] - ref['fix_q'][f_loc]) ** 2).mean(1)))
        # 몸통 위치: 시작 위치 차이를 빼고 이동량을 비교 (리셋 위치가 레퍼런스와 다를 수 있음)
        p0 = r['qpos'][e, 0, :3] - ref['fix_base_pos'][max(f_loc[0] - 1, 0)]
        P = np.linalg.norm(q[:, :3] - p0 - ref['fix_base_pos'][f_loc], axis=1)
        fp = []
        for t in range(n + 1):
            d.qpos[:] = r['qpos'][e, t]; mujoco.mj_kinematics(m, d); fp.append(d.geom_xpos[feet].copy())
        fp = np.array(fp); v = np.linalg.norm(np.diff(fp[:, :, :2], axis=0), axis=-1) / dt
        c = r['contact'][e, 1:n + 1].astype(bool)
        S = np.where(c.any(1), (v * c).sum(1) / np.maximum(c.sum(1), 1), np.nan)
        rv = np.linalg.norm(np.diff(ref['fix_foot_pos'][:, :, :2], axis=0), axis=-1) / dt
        rc = ref['fix_contact'][1:].astype(bool); ff = np.minimum(f_loc, len(rv) - 1)
        SR = np.where(rc[ff].any(1), (rv[ff] * rc[ff]).sum(1) / np.maximum(rc[ff].sum(1), 1), np.nan)
        allS.append(np.nanmean(S)); allSR.append(np.nanmean(SR)); allJ.append(np.nanmean(J)); allP5.append(P[min(n, 250) - 1])
        per_motion.setdefault(str(r['motions'][c0]), []).append((np.nanmean(S), np.nanmean(SR), np.nanmean(J), P[-1]))
S, SR, J, P5 = map(np.array, (allS, allSR, allJ, allP5))
summary = dict(episodes=int(len(S)), slip_m_s_median=float(np.nanmedian(S)), slip_m_s_q25_q75=[float(x) for x in np.nanpercentile(S, [25, 75])],
               ref_slip_m_s_median=float(np.nanmedian(SR)), joint_deg_median=float(np.median(J)), root_m_at_5s_median=float(np.median(P5)),
               per_motion={k: dict(n=len(v), slip=float(np.nanmedian([x[0] for x in v])), joint=float(np.median([x[2] for x in v])), root_end=float(np.median([x[3] for x in v]))) for k, v in sorted(per_motion.items())})
(HERE / 'v7_tracking_errors.json').write_text(json.dumps(summary, indent=1)); np.savez(HERE / 'v7_tracking_errors.npz', slip=S, ref_slip=SR, joint=J, root5=P5)
print(json.dumps({k: v for k, v in summary.items() if k != 'per_motion'}))
for k, v in summary['per_motion'].items(): print(f"{k:24s} n{v['n']:4d} slip {v['slip']:.3f}  joint {v['joint']:.1f}  root_end {v['root_end']:.2f}")
