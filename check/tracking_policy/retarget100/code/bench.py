"""100 Hz 리타겟 병렬 속도 측정 (2026-11-17, CPU 가 빈 상태). trotCWSlow CALM 64 개 × 형상 G_l+00_b+00, 프로세스 N 개 (청크 분할). 출력: bench/<N>/, bench.json"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())
sys.path.insert(0, str(ROOT / 'code/motion_tracking')); S5 = ROOT / 'chain/runs/s5_retarget'


def job(a):
    os.environ['OMP_NUM_THREADS'] = '1'
    import pd_replay as pd
    src, dst, clips, E = a; pd.retarget_tt(src, dst, only=clips, choices=['00000011'], uniforms=E, torque_limit=33.5); return len(clips)


def main():
    from plan_calm_dynamics import resample_reference
    A = json.load(open(S5 / 'assignment.json')); E = A['shapes']['G_l+00_b+00']['E']
    clips = sorted({c for v in A['by_group']['trotCWSlow__r00'].values() for c in v})[:64]
    src = HERE / 'bench' / 'a1'; src.mkdir(parents=True, exist_ok=True)
    for c in clips:
        if not (src / f'{c}.npz').exists(): resample_reference(S5 / 'a1_reference' / f'{c}.npz', src / f'{c}.npz', hz=100)
    out = {}
    for n in [16, 8]:
        t0 = time.time()
        with Pool(n) as p: p.map(job, [(str(src), str(HERE / 'bench' / str(n) / f'w{i}'), clips[i::n], E) for i in range(n)])
        w = time.time() - t0; out[n] = dict(pairs=len(clips), wall_s=w, s_per_pair=w / len(clips)); print(n, out[n], flush=True)
    (HERE / 'bench.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
