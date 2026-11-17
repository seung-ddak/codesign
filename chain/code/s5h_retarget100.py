"""5h 단계: 100 Hz 리타겟, CALM 하나를 형상 9 개 전부에 (2026-11-17 사용자 결정, check/tracking_policy/PLAN_100HZ.md).
모션 세트 = trot 레퍼런스 1 + 그 세트 CALM 1000 (s5 assignment by_group 의 모든 형상 목록 합집합). 레퍼런스는 trot 6 개만.
방식: s5 A1 레퍼런스 (30 Hz) → resample_reference 100 Hz → pd_replay.retarget_tt (평활 창 시간 기준, 2026-11-17 변경).
모션 세트 순서로 처리하고, 세트 × 형상이 다 차면 ready/<세트>/<형상>.json 을 쓴다 (6h 학습이 세트별로 바로 시작).
실행: python s5h_retarget100.py [--workers 16] [--sets trotCWSlow__r00 ...]
출력: runs/s5h_retarget100/ (a1_100/, references/<형상>/00000011/, ready/<세트>/<형상>.json, assignment.json)"""
import argparse, json, os
from multiprocessing import Pool
from common import RUNS, log
OUT = RUNS / 's5h_retarget100'; S5 = RUNS / 's5_retarget'; A1 = OUT / 'a1_100'; HZ = 100


def resample(name):
    from plan_calm_dynamics import resample_reference
    dst = A1 / f'{name}.npz'
    if not dst.exists(): resample_reference(S5 / 'a1_reference' / f'{name}.npz', dst, hz=HZ)
    return name


def retarget(job):
    os.environ['OMP_NUM_THREADS'] = '1'
    import pd_replay as pd
    shape, E, clips = job; dst = OUT / 'references' / shape
    todo = [c for c in clips if not (dst / '00000011' / f'{c}.npz').exists()]
    if todo: pd.retarget_tt(str(A1), str(dst), only=todo, choices=['00000011'], uniforms=E, torque_limit=33.5)
    return shape, len(todo)


def plan():
    a = json.load(open(S5 / 'assignment.json'))
    sets = {g: [g] + sorted({c for v in per.values() for c in v}) for g, per in a['by_group'].items()}
    return a['shapes'], sets


def mark_ready(shapes, sets):
    done = []
    for g, clips in sets.items():
        for s in shapes:
            have = {f.stem for f in (OUT / 'references' / s / '00000011').glob('*.npz')}
            if set(clips) <= have:
                p = OUT / 'ready' / g / f'{s}.json'; p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists(): p.write_text(json.dumps(dict(clips=len(clips))))
                done.append((g, s))
    return done


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--sets', nargs='*'); ap.add_argument('--chunk', type=int, default=64); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); A1.mkdir(exist_ok=True)
    shapes, sets = plan()
    if a.sets: sets = {g: sets[g] for g in a.sets}
    json.dump(dict(shapes=shapes, sets=sets, hz=HZ), open(OUT / 'assignment.json', 'w'), indent=1)
    with Pool(a.workers) as p:
        for g, clips in sets.items():
            p.map(resample, clips, chunksize=32)
            jobs = [(s, shapes[s]['E'], clips[i:i + a.chunk]) for i in range(0, len(clips), a.chunk) for s in shapes]
            n = 0
            for s, k in p.imap_unordered(retarget, jobs):
                n += k
            ready = [s for gg, s in mark_ready(shapes, {g: clips}) if gg == g]
            log(f's5h {g} retarget +{n}, ready {len(ready)}/{len(shapes)}')
    log('s5h END')


if __name__ == '__main__':
    main()
