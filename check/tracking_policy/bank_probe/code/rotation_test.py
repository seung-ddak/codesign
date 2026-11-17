"""s6h_train100.train_rotating 확인 (2026-11-17): 묶음 2 개 × 클립 251 개 (probe 링크), 총 2M, 방문 1M → 묶음이 바뀌어도 step · 학습 상태가 이어지는지.
출력: rot/ (plan.json, metrics.jsonl, train.log), rotation.json"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())
sys.path.insert(0, str(ROOT / 'chain/code'))
import s6h_train100 as h, s6_train as s6


def main():
    shapes = json.load(open(s6.S5 / 'assignment.json'))['shapes']; f = HERE / 'rot'; f.mkdir(exist_ok=True)
    for n in ('references', 'ready'):
        if not (f / n).exists(): (f / n).symlink_to(HERE / n)
    clips = [f'probe_{i:04d}' for i in range(501)]; parts = h.chunks(clips, 2)
    c = s6.base_config(); c.update(control_hz=100, physics_hz=200, max_episode_length=2000, resident_designs=9, train_transitions=2_000_000, checkpoint_steps=1_000_000)
    json.dump(dict(version=1, kind='ref_ppo', reference='probe', motion_count=len(clips), config=c, topology='00000011', designs=[dict(design=k, E=v['E']) for k, v in shapes.items()],
                   eval_designs=[], assignments={k: clips for k in shapes}, eval_assignments={k: [] for k in shapes}), open(f / 'plan.json', 'w'), indent=1)
    ok = h.train_rotating(f, parts, 1_000_000)
    m = [json.loads(l) for l in open(f / 'metrics.jsonl')]
    out = dict(ok=ok, step=s6.step_of(f), visits=len(list((f / 'stages').glob('[0-9]*'))), chunk_sizes=[len(p) for p in parts],
               sum_reward=[(x['step'], round(x['metrics']['episode/sum_reward'], 2)) for x in m if 'episode/sum_reward' in x['metrics']])
    (HERE / 'rotation.json').write_text(json.dumps(out, indent=1)); print(out, flush=True)


if __name__ == '__main__':
    main()
