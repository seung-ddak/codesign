"""역동적인 (방향이 많이 섞인) 새 CALM 모션을 A1 과 TT 에 같은 장면으로 적용. 묶음마다 1 개:
s6 학습에 실제로 쓰인 CALM 중 0.5 초 이상 나오는 방향 수가 가장 많은 모션 (같으면 주 방향 비중이 작은 것).
TT = 그 모션을 학습한 형상 + 그 묶음의 s6 공용 정책 (100M), 레퍼런스 = s5 리타겟. 모션 끝까지 실행.
출력: runs/s7_eval/videos/dynamic_calm/ (<모션>_same_scene.mp4, a1_only/<모션>.mp4, dynamic_calm.json)"""
import json, os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
from common import RUNS, log
import evalpol
from same_scene import render
from ref_calm import a1_only
from s7_eval import ckpts
S5 = RUNS / 's5_retarget'; OUT = RUNS / 's7_eval/videos/dynamic_calm'
SH = json.load(open(S5 / 'assignment.json'))['shapes']
PICK = [('calm_round_02_01669', 'trotBackwardSlow__r00', 'G_l-20_b+00', 6, 0.56), ('calm_round_02_01795', 'trotCCWSlow__r00', 'G_l-20_b-20', 6, 0.41), ('calm_round_00_01039', 'trotCWSlow__r00', 'G_l+00_b+00', 5, 0.31), ('calm_round_02_00571', 'trotForwardSlow__r00', 'G_l-20_b+20', 6, 0.35), ('calm_round_01_00042', 'trotLeftSlow__r00', 'G_l+00_b+20', 5, 0.33), ('calm_round_00_01109', 'trotRightSlow__r00', 'G_l+00_b+00', 6, 0.26)]


def main():
    (OUT / 'a1_only').mkdir(parents=True, exist_ok=True); rows = []
    for M, g, s, ndir, share in PICK:
        d = S5 / 'references' / s / '00000011'; z = np.load(d / f'{M}.npz'); dt = float(z['dt']); E = SH[s]['E']
        r = evalpol.evaluate(ckpts(RUNS / 's6_train' / g)[-1], E, d, M, multi=True, keep_qpos=True, max_seconds=len(z['fix_q']) * dt)
        gn = g.replace('trot', '').replace('Slow__r00', '')
        render(np.asarray(r['qpos']), z['a1_qpos'], dt, E, OUT / f'{M}_same_scene.mp4', f'{M} · {ndir} directions · {gn} policy (100M) · TT {s} (trained)',
               f"same scene: A1 left, TT right · joint RMSE {r['joint_rmse_deg']:.1f} deg   root RMSE {r['root_rmse_cm']:.1f} cm", not r['survived'])
        a1_only(z, OUT / 'a1_only' / f'{M}.mp4', f'{M} · {ndir} directions')
        rows.append(dict(motion=M, policy=g, shape=s, directions=ndir, main_share=share, motion_seconds=len(z['fix_q']) * dt, survived=r['survived'],
                         seconds=r['seconds'], joint_rmse_deg=r['joint_rmse_deg'], root_rmse_cm=r['root_rmse_cm'], passed=evalpol.passes(r)))
        log(f"dynamic {M} {gn} {s} 생존 {r['survived']} {r['seconds']:.1f}s 관절 {r['joint_rmse_deg']:.1f}°")
    json.dump(rows, open(OUT / 'dynamic_calm.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
