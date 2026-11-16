"""레퍼런스 묶음별 학습 폴더와 plan.json 을 만든다.
A: ref_ppo/<레퍼런스>/        형상 E0~E4 를 함께 학습하는 정책 1 개 (레퍼런스 25 개 → 정책 25 개)
B: E_ppo/<E>/<레퍼런스>/      형상 1 개 정책 (5 × 25 = 125 개)
묶음 = 레퍼런스 모션 1 개 + 분류 (ref_ppo/_classification/assignment.json) 에서 그 레퍼런스에 배정된 CALM 모션 전부.
references / ready 는 ref_ppo/_data 를 가리키는 링크다 (리타겟 결과 공유)."""
import json, sys
from pathlib import Path
TOOLS = Path(__file__).resolve().parent; REFPPO = TOOLS.parent; ROOT = REFPPO.parent
sys.path.insert(0, str(ROOT / 'check/slide/code'))
from shapes import SHAPES

DATA = REFPPO / '_data'
assign = json.load(open(REFPPO / '_classification/assignment.json'))['assignment']
base = json.load(open(ROOT / 'legacy/runs/ppo_tracking/try2/plan.json'))
CONFIG = dict(base['config'])
CONFIG.update(num_envs=1024, batch_size=1024, unroll_length=24, num_minibatches=4, num_updates_per_batch=8, learning_rate=1e-3,
              entropy_cost=0.01, hidden=[256, 128, 64], control_hz=50, physics_hz=200, max_episode_length=1000,
              penalty_termination=0.0, reward_clip_positive=True, penalty_foot_slip=0.0, checkpoint_steps=10_000_000,
              metrics_steps=2_000_000, seed=0, rollout_batch=256)


def budget(n_motions):
    """학습량: 5M + 모션당 1 만 step, 최대 50M."""
    return int(min(50_000_000, 5_000_000 + 10_000 * n_motions))


def make(folder, shapes, ref, motions):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('references', 'ready'):
        link = folder / name
        if not link.exists(): link.symlink_to(DATA / name)
    clips = [ref] + list(motions)
    c = dict(CONFIG, resident_designs=len(shapes), train_transitions=budget(len(motions)))
    plan = dict(version=1, kind='ref_ppo' if len(shapes) > 1 else 'E_ppo', reference=ref, motion_count=len(motions), config=c,
                topology='00000011', designs=[dict(design=k, E=SHAPES[k]['E']) for k in shapes], eval_designs=[],
                assignments={k: clips for k in shapes}, eval_assignments={k: [] for k in shapes})
    json.dump(plan, open(folder / 'plan.json', 'w'), indent=1)
    return c['train_transitions']


if __name__ == '__main__':
    total = dict(A=0, B=0)
    for ref, motions in assign.items():
        total['A'] += make(REFPPO / ref, list(SHAPES), ref, motions)
        for k in SHAPES:
            total['B'] += make(ROOT / 'E_ppo' / k / ref, [k], ref, motions)
    print({k: f'{v / 1e6:.0f}M steps' for k, v in total.items()})
