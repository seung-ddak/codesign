"""본 학습 (s6h) 규모 GPU 메모리·시작 시간·속도 확인 (2026-11-17). 정책 하나 = 형상 9 × 클립 1001 × 100 Hz (20 s 클립, 최악 길이).
리타겟을 기다리지 않도록 클립 1001 개를 hz100_sg 레퍼런스 한 개 (calm_round_00_00939, 20 s) 의 심볼릭 링크로 채운다 (형상마다 자기 형상 파일 → E 검사 통과). 메모리는 실제와 같음.
학습 2M transition, 설정 = s6h_train100.build (100/200 Hz, 평활 70 ms). GPU 메모리는 nvidia-smi 로 2 초마다 기록.
출력: references/, ready/, run/ (plan.json, train.log), gpu.log, probe.json"""
import json, os, subprocess, sys, threading, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())
sys.path.insert(0, str(ROOT / 'chain/code'))
os.environ['TRACK_SG_SEC'] = '0.07'
import s6_train as s6
SRC = HERE.parent / 'hz100_sg' / 'refs'; CLIP = 'calm_round_00_00939'; N = int(os.environ.get('PROBE_N', 1001)); STEPS = 2_000_000


def main():
    shapes = json.load(open(s6.S5 / 'assignment.json'))['shapes']; names = [f'probe_{i:04d}' for i in range(N)]
    for s in shapes:
        d = HERE / 'references' / s / '00000011'; d.mkdir(parents=True, exist_ok=True)
        for n in names:
            if not (d / f'{n}.npz').exists(): (d / f'{n}.npz').symlink_to(SRC / s / '00000011' / f'{CLIP}.npz')
        (HERE / 'ready').mkdir(exist_ok=True); (HERE / 'ready' / f'{s}.json').write_text('{}')
    f = HERE / f'run_{N}'; f.mkdir(exist_ok=True)
    for n in ('references', 'ready'):
        if not (f / n).exists(): (f / n).symlink_to(HERE / n)
    c = s6.base_config(); c.update(control_hz=100, physics_hz=200, max_episode_length=2000, resident_designs=9, train_transitions=STEPS, checkpoint_steps=1_000_000)
    json.dump(dict(version=1, kind='ref_ppo', reference=CLIP, motion_count=N, config=c, topology='00000011', designs=[dict(design=k, E=v['E']) for k, v in shapes.items()],
                   eval_designs=[], assignments={k: names for k in shapes}, eval_assignments={k: [] for k in shapes}), open(f / 'plan.json', 'w'), indent=1)
    stop = [False]; peak = [0]
    def poll():
        with open(HERE / f'gpu_{N}.log', 'w') as g:
            while not stop[0]:
                m = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
                peak[0] = max(peak[0], m); g.write(f'{time.time():.0f} {m}\n'); g.flush(); time.sleep(2)
    th = threading.Thread(target=poll, daemon=True); th.start(); t0 = time.time()
    ok = s6.train(f); stop[0] = True
    m = [json.loads(l) for l in open(f / 'metrics.jsonl')] if (f / 'metrics.jsonl').exists() else []
    first = min((x['time'] for x in m), default=None); sps = [x['metrics']['training/sps'] for x in m if 'training/sps' in x['metrics']]
    out = dict(ok=ok, wall_s=time.time() - t0, startup_s=(first - t0) if first else None, peak_gpu_mib=peak[0], train_sps=sps[-1] if sps else None)
    (HERE / f'probe_{N}.json').write_text(json.dumps(out, indent=1)); print(out, flush=True)


if __name__ == '__main__':
    main()
