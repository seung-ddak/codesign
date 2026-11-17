"""CALM 모션 100 Hz 리타겟 확인 (2026-11-17 사용자 요청: 54 개 정책 학습 전 2 개 모션으로 확인). 방식 A = s5 A1 레퍼런스 (30 Hz) 를 100 Hz 로 보간 (resample_reference) → 형상 9 개 리타겟 (pd_replay.retarget_tt, foot_lock + IK).
모션 = 아직 안 쓴 모션 세트 2 개의 첫 CALM (trotForwardSlow, trotLeftSlow). 형상 9 개를 프로세스 9 개로 병렬 → 병렬 속도 측정.
지표: 저장 지표 (fix_slip, fix_track_err, fix_dq_max_deg, inv_tau_max, inv_bad_frac) + 디딘 발 수평 속도 (cm/s), s5 30 Hz 리타겟 (있는 형상만) 과 비교.
출력: a1/, refs/<형상>/00000011/, metrics.json, timing.json
2026-11-17 v2: pd_replay·track_env 평활 창을 시간 기준으로 바꾼 뒤 다시 실행 (100 Hz: 속도 23 프레임, 몸통 31 프레임). 이전 결과 (7·9 프레임 고정) = v1_win7/."""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())
sys.path.insert(0, str(ROOT / 'chain/code')); sys.path.insert(0, str(ROOT / 'code/motion_tracking'))
S5 = ROOT / 'chain/runs/s5_retarget'; HZ = 100
CLIPS = {'calm_round_00_00021': 'trotForwardSlow__r00', 'calm_round_00_00002': 'trotLeftSlow__r00'}


def one(args):
    s, E = args
    import pd_replay as pd
    t0 = time.time(); pd.retarget_tt(str(HERE / 'a1'), str(HERE / 'refs' / s), only=list(CLIPS), choices=['00000011'], uniforms=E, torque_limit=33.5)
    return s, time.time() - t0


def metrics(f):
    z = np.load(f); c = z['fix_contact'].astype(bool); dt = float(z['dt'])
    v = np.linalg.norm(np.diff(z['fix_foot_pos'][:, :, :2], axis=0), axis=2) / dt * 100
    return dict(hz=round(1 / dt), slip=float(z['fix_slip']), track_err=float(z['fix_track_err']), dq_max_deg=float(z['fix_dq_max_deg']),
                tau_max=float(z['inv_tau_max']), bad_frac=float(z['inv_bad_frac']), contact_foot_speed_cm_s=float(v[c[1:] & c[:-1]].mean()))


def main():
    from plan_calm_dynamics import resample_reference
    shapes = json.load(open(S5 / 'assignment.json'))['shapes']; (HERE / 'a1').mkdir(exist_ok=True)
    for c in CLIPS:
        if not (HERE / 'a1' / f'{c}.npz').exists(): resample_reference(S5 / 'a1_reference' / f'{c}.npz', HERE / 'a1' / f'{c}.npz', hz=HZ)
    t0 = time.time()
    with Pool(9) as p: per = dict(p.map(one, [(s, v['E']) for s, v in shapes.items()]))
    wall = time.time() - t0; n = len(shapes) * len(CLIPS)
    (HERE / 'timing.json').write_text(json.dumps(dict(pairs=n, workers=9, wall_s=wall, s_per_pair_effective=wall / n, per_shape_s=per), indent=1))
    print(f'리타겟 {n} 쌍, 병렬 9, {wall:.0f} s → 쌍당 실효 {wall / n:.1f} s', flush=True)
    out = []
    for c, g in CLIPS.items():
        for s in shapes:
            r = dict(clip=c, motion_set=g, shape=s, hz100=metrics(HERE / 'refs' / s / '00000011' / f'{c}.npz'))
            f30 = S5 / 'references' / s / '00000011' / f'{c}.npz'
            if f30.exists(): r['hz30'] = metrics(f30)
            out.append(r)
    (HERE / 'metrics.json').write_text(json.dumps(out, indent=1))
    for r in out: print(r['clip'], r['shape'], '100Hz', {k: round(v, 3) for k, v in r['hz100'].items()}, '| 30Hz', {k: round(v, 3) for k, v in r.get('hz30', {}).items()}, flush=True)


if __name__ == '__main__':
    main()
