"""e0000 PPO 트래킹 데이터의 시간별 오차: 관절 추종, 몸통 위치 추종, 접지한 발 미끄러짐 (실제 vs 리타겟 레퍼런스)."""
import json, sys
from pathlib import Path
import numpy as np, mujoco
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'code/motion_tracking'))
from pd_replay import build_model
D = ROOT / 'runs/calm_dynamics_1000x100'
with np.load(D / 'rollouts/e0000/primitive_seed0.npz', allow_pickle=True) as z:
    r = {k: z[k] for k in z.files}  # npz 는 접근할 때마다 다시 읽으므로 한 번만 읽어 둔다
m = build_model('00000011', uniforms=list(r['E']), torque_limit=33.5); d = mujoco.MjData(m)
feet = [m.geom(f'{l}_foot').id for l in ('FR', 'FL', 'RR', 'RL')]
dt = float(r['dt']); N, T = r['valid'].shape
J = np.full((N, T), np.nan); P = np.full((N, T), np.nan); S = np.full((N, T), np.nan); SR = np.full((N, T), np.nan)
for e in range(N):
    ref = np.load(D / f"references/e0000/00000011/{r['motions'][e]}.npz")
    n = int(r['valid'][e].sum())
    # ref_frame 은 모든 모션을 이어 붙인 전체 번호다. 이 모션 안의 번호 = 전체 번호 - 첫 번호 + 시작 프레임
    f = np.minimum(r['ref_frame'][e][:n] - r['ref_frame'][e][0] + int(r['start_frame'][e]), len(ref['fix_q']) - 1)
    q = r['qpos'][e, 1:n + 1]
    J[e, :n] = np.degrees(np.sqrt(((q[:, 7:] - ref['fix_q'][f]) ** 2).mean(1)))
    P[e, :n] = np.linalg.norm(q[:, :3] - ref['fix_base_pos'][f], axis=1)
    fp = []
    for t in range(n + 1):
        d.qpos[:] = r['qpos'][e, t]; mujoco.mj_kinematics(m, d); fp.append(d.geom_xpos[feet].copy())
    fp = np.array(fp); v = np.linalg.norm(np.diff(fp[:, :, :2], axis=0), axis=-1) / dt
    c = r['contact'][e, 1:n + 1].astype(bool)
    S[e, :n] = np.where(c.any(1), (v * c).sum(1) / np.maximum(c.sum(1), 1), np.nan)
    rv = np.linalg.norm(np.diff(ref['fix_foot_pos'][:, :, :2], axis=0), axis=-1) / dt
    rc = ref['fix_contact'][1:].astype(bool); ff = np.minimum(f, len(rv) - 1)
    SR[e, :n] = np.where(rc[ff].any(1), (rv[ff] * rc[ff]).sum(1) / np.maximum(rc[ff].sum(1), 1), np.nan)
np.savez(HERE / 'tracking_errors.npz', joint_deg=J, root_m=P, slip_m_s=S, ref_slip_m_s=SR, fell=r['fell'], dt=dt)
def q(a): return np.nanpercentile(a, [25, 50, 75], axis=0)
summary = dict(episodes=int(N), fell=int(r['fell'].sum()),
               joint_deg_median=float(np.nanmedian(J)), root_m_median_at_end=float(np.nanmedian(P[:, 499] if T > 499 else P[:, -1])),
               slip_m_s_median=float(np.nanmedian(S)), ref_slip_m_s_median=float(np.nanmedian(SR)),
               corr_episode_slip_vs_root=float(np.corrcoef(np.nanmean(S, 1), np.nanmean(P, 1))[0, 1]))
(HERE / 'tracking_errors.json').write_text(json.dumps(summary, indent=1)); print(summary)
