"""hz100_sg 에서 calm_round_02_00546 학습 실패 (0/9) 원인 확인 (2026-11-17 사용자 승인). 레퍼런스 = hz100_sg/refs (평활 창 시간 기준 리타겟), 20M, 100 Hz / 200 Hz, 그 외 hz100_sg 와 같음.
  a: track_env 레퍼런스 속도 평활만 7 프레임 (TRACK_SG_SEC=0.07) 으로 되돌림 → 학습되면 평활 변경이 원인
  b: 설정 그대로, seed 1 → 학습되면 초기 운
실행: python run.py a|b. 출력: runs_<a|b>/, results_<a|b>.json"""
import json, os, sys
from pathlib import Path
V = sys.argv[1]
if V == 'a': os.environ['TRACK_SG_SEC'] = '0.07'
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())
sys.path.insert(0, str(ROOT / 'chain/code')); sys.path.insert(0, str(ROOT / 'code/motion_tracking'))
import s6_train as s6, s6c_calm as s6c
G, C, STEPS = 'trotCWSlow__r00', 'calm_round_02_00546', 20_000_000


def main():
    shapes = json.load(open(s6.S5 / 'assignment.json'))['shapes']; s6c.REFS = HERE.parent / 'hz100_sg' / 'refs'
    f = s6c.build(G, C, shapes, out=HERE / f'runs_{V}', steps=STEPS); p = json.load(open(f / 'plan.json'))
    p['config'].update(train_transitions=STEPS, checkpoint_steps=5_000_000, control_hz=100, physics_hz=200, max_episode_length=2000, seed=1 if V == 'b' else p['config']['seed'])
    json.dump(p, open(f / 'plan.json', 'w'), indent=1)
    ok = (f / 'done').exists() or s6.train(f); rr = s6c.evaluate(f, C, shapes) if ok else []
    (HERE / f'results_{V}.json').write_text(json.dumps(dict(variant=V, failed=not ok, results=rr), indent=1, default=float))
    print(V, '통과', sum(r['passed'] for r in rr), '/', len(rr), flush=True)


if __name__ == '__main__':
    main()
