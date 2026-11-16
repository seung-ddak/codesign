"""레퍼런스 6 종 (trot Forward/Backward/Left/Right/CW/CCW) 구간을 모두 0.5 초 이상 포함한 새 CALM 모션 6 개를
A1 과 TT (학습 가운데 형상 G_l+00_b+00, 가장 긴 구간 묶음의 s6 공용 정책) 에 적용. 모션 끝까지 실행.
출력: runs/s7_eval/videos/mixed_calm/ (<모션>_same_scene.mp4, a1_only/<모션>.mp4, refs/, mixed_calm.json)"""
import json, os, sys
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
from common import ROOT, RUNS, log
import evalpol
from same_scene import render
from s7_eval import ckpts
sys.path.insert(0, str(ROOT / 'legacy/runs/v100_mujoco_10k/mesh_render'))
S5 = RUNS / 's5_retarget'; OUT = RUNS / 's7_eval/videos/mixed_calm'; A1 = OUT / 'refs/a1'
SHAPE = 'G_l+00_b+00'; E = json.load(open(S5 / 'assignment.json'))['shapes'][SHAPE]['E']
PICK = [('round_02', '01669', 'trotBackwardSlow__r00'), ('round_02', '00571', 'trotForwardSlow__r00'), ('round_00', '00541', 'trotForwardSlow__r00'),
        ('round_00', '01109', 'trotRightSlow__r00'), ('round_02', '01795', 'trotCCWSlow__r00'), ('round_00', '00237', 'trotRightSlow__r00')]


def a1_only(z, path, title):
    import render_three as r3
    rm = r3.rm; dt = float(z['dt']); a = z['a1_qpos'].astype(float).copy(); a[:, :2] -= a[0, :2]; p = r3.Panel(r3.a1_model())

    def frames():
        for t in np.arange(0, len(a) * dt, 1 / r3.FPS):
            k = min(int(round(t / dt)), len(a) - 1)
            yield rm.label(p.frame(a[k], a[:k + 1, :2], a[k, :2]), title, 'A1: original CALM motion', f't = {t:4.1f} s   moved {np.linalg.norm(a[k, :2]):.2f} m', False)
    try: rm.encode_frames(path, frames(), width=r3.W, height=r3.H, fps=r3.FPS)
    finally: p.r.close()


def main():
    import pd_replay as pd
    from plan_calm_dynamics import prepare_reference
    A1.mkdir(parents=True, exist_ok=True); (OUT / 'a1_only').mkdir(exist_ok=True); rows = []
    for rnd, num, g in PICK:
        M = f'calm_{rnd}_{num}'; a1 = A1 / f'{M}.npz'
        if not a1.exists():
            s5 = S5 / 'a1_reference' / f'{M}.npz'
            if s5.exists(): a1.symlink_to(s5)
            else: prepare_reference(RUNS / f's4_calm/{rnd}/accepted/{num}.npz', a1, hz=30)
        d = OUT / 'refs' / SHAPE
        if not (d / '00000011' / f'{M}.npz').exists(): pd.retarget_tt(str(A1), str(d), only=[M], choices=['00000011'], uniforms=E, torque_limit=33.5)
        z = np.load(d / '00000011' / f'{M}.npz'); dt = float(z['dt']); ck = ckpts(RUNS / 's6_train' / g)[-1]
        r = evalpol.evaluate(ck, E, d / '00000011', M, multi=True, keep_qpos=True, max_seconds=len(z['fix_q']) * dt)
        gn = g.replace('trot', '').replace('Slow__r00', '')
        render(np.asarray(r['qpos']), z['a1_qpos'], dt, E, OUT / f'{M}_same_scene.mp4', f'{M} · {gn} policy · TT {SHAPE}   (same scene: A1 left, TT right)',
               f"all 6 reference types · joint RMSE {r['joint_rmse_deg']:.1f} deg   root RMSE {r['root_rmse_cm']:.1f} cm", not r['survived'])
        a1_only(z, OUT / 'a1_only' / f'{M}.mp4', M)
        rows.append(dict(motion=M, policy=g, shape=SHAPE, motion_seconds=len(z['fix_q']) * dt, survived=r['survived'], seconds=r['seconds'],
                         joint_rmse_deg=r['joint_rmse_deg'], root_rmse_cm=r['root_rmse_cm'], passed=evalpol.passes(r)))
        log(f"mixed {M} {gn} 생존 {r['survived']} {r['seconds']:.1f}/{rows[-1]['motion_seconds']:.1f}s 관절 {r['joint_rmse_deg']:.1f}°")
    json.dump(rows, open(OUT / 'mixed_calm.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
