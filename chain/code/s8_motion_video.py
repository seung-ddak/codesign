"""legacy CALM 모션 하나 (기본 calm_06117) 를 s6 공용 정책으로 학습 형상 1 개 + 학습에 쓰지 않은 형상 2 개에 적용한 4 칸 영상.
칸: A1 원본 | 학습 형상 G_l+00_b+00 | 미사용 G_l+20_b+10 | 미사용 G_l-20_b-10. 묶음 = 몸통 속도 (vx, vy, wz) 평균이 가장 가까운 trot 레퍼런스 (s4 분류와 같은 방식).
출력: runs/s8_collect/videos/calm_<번호>/ (영상, refs/, result.json)"""
import json, os, sys
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
from common import ROOT, RUNS, log
import s4_calm, s6_train as s6, evalpol
from s7_eval import ckpts
sys.path.insert(0, str(ROOT / 'legacy/runs/v100_mujoco_10k/mesh_render'))
NUM = sys.argv[1] if len(sys.argv) > 1 else '06117'; M = f'calm_{NUM}'
OUT = RUNS / 's8_collect/videos' / M; A1 = OUT / 'refs/a1'
TRAIN = json.load(open(RUNS / 's5_retarget/assignment.json'))['shapes']; UNSEEN = json.load(open(RUNS / 's8_collect/shapes.json'))
SHAPES = [('G_l+00_b+00', TRAIN['G_l+00_b+00']['E'], '학습 형상'), ('G_l+20_b+10', UNSEEN['G_l+20_b+10']['E'], '미사용 형상'), ('G_l-20_b-10', UNSEEN['G_l-20_b-10']['E'], '미사용 형상')]


def group(z):
    names, rf = s4_calm.ref_features()
    f = s4_calm.feat(z['fix_base_pos'], z['fix_base_quat'][:, [1, 2, 3, 0]], z['fix_q'], float(z['dt']))[:3]
    scale = np.r_[rf[:, :3], f[None]].std(0) + 1e-9
    return names[int(np.linalg.norm((rf[:, :3] - f) / scale, axis=1).argmin())], f


def main():
    import pd_replay as pd
    from plan_calm_dynamics import prepare_reference
    A1.mkdir(parents=True, exist_ok=True); a1 = A1 / f'{M}.npz'
    if not a1.exists(): prepare_reference(ROOT / f'legacy/runs/calm_motion_10,000/dataset/accepted/{NUM}.npz', a1, hz=30)
    z = np.load(a1); g, f = group(z); ck = ckpts(s6.OUT / (g if g.endswith('__r00') else f'{g}__r00'))[-1]
    log(f'{M} 묶음 {g} (vx {f[0]:.2f} vy {f[1]:.2f} wz {f[2]:.2f}), 정책 {ck}')
    res, qs = [], []
    for s, E, kind in SHAPES:
        d = OUT / 'refs' / s
        if not (d / '00000011' / f'{M}.npz').exists():
            pd.retarget_tt(str(A1), str(d), only=[M], choices=['00000011'], uniforms=E, torque_limit=33.5)
        r = evalpol.evaluate(ck, E, d / '00000011', M, multi=True, keep_qpos=True, max_seconds=10.0)
        qs.append(np.asarray(r.pop('qpos'), np.float64)); res.append(dict({k: v for k, v in r.items() if np.isscalar(v) or v is None}, shape=s, kind=kind, passed=evalpol.passes(r)))
        log(f"{M} {s} 생존 {r['survived']} 관절 {r['joint_rmse_deg']}")
    json.dump(dict(motion=M, group=g, velocity=f.tolist(), checkpoint=str(ck), results=res), open(OUT / 'result.json', 'w'), indent=1, default=float)
    render(z, qs, res, g)


def render(z, qs, res, g):
    import render_three as r3
    rm = r3.rm; dt = float(z['dt'])
    a1m = r3.a1_model(); idx = [a1m.jnt_qposadr[a1m.joint(j).id] for j in r3.A1_JOINTS]
    aq = np.tile(a1m.qpos0, (len(z['fix_q']), 1)); aq[:, :3] = z['fix_base_pos']; aq[:, 3:7] = z['fix_base_quat']; aq[:, idx] = z['fix_q']; aq[:, :2] -= aq[0, :2]
    for q in qs: q[:, :2] -= q[0, :2]
    panels = [r3.Panel(a1m)] + [r3.Panel(rm.gtm.make_physics(np.asarray(E, np.float64))) for _, E, _ in SHAPES]
    T = len(aq) * dt; head = f"{M} · policy {g.replace('trot', '').replace('Slow__r00', '').replace('Slow', '')}"

    def frames():
        for t in np.arange(0, T, 1 / r3.FPS):
            k = min(int(round(t / dt)), len(aq) - 1); row = [rm.label(panels[0].frame(aq[k], aq[:k + 1, :2], aq[k, :2]), head, 'A1: original CALM motion', f't = {t:4.1f} s   moved {np.linalg.norm(aq[k, :2]):.2f} m', False)]
            for p, q, r, (s, _, kind) in zip(panels[1:], qs, res, SHAPES):
                j = min(int(round(t / dt)), len(q) - 1); fell = not r['survived'] and j == len(q) - 1
                tag = 'TRAIN shape' if kind == '학습 형상' else 'UNSEEN shape'
                row.append(rm.label(p.frame(q[j], q[:j + 1, :2], q[j, :2]), f'{tag} · {s}', f"shared PPO (joint {r['joint_rmse_deg'] or 0:.1f} deg, root {r['root_rmse_cm'] or 0:.0f} cm)",
                                    f'FELL at {len(q) * dt:.1f} s' if fell else f't = {t:4.1f} s   moved {np.linalg.norm(q[j, :2]):.2f} m', fell))
            yield np.concatenate(row, 1)
    out = OUT / f'{M}_A1_train_vs_unseen.mp4'
    try: rm.encode_frames(out, frames(), width=len(panels) * r3.W, height=r3.H, fps=r3.FPS)
    finally:
        for p in panels: p.r.close()
    log(f'영상 {out}')


if __name__ == '__main__':
    main()
