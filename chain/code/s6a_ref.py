"""6a 단계 (2026-11-12 사용자 결정): 레퍼런스 하나를 학습 형상 전부로 학습하는 정책. trot Slow 레퍼런스 6 개 → 정책 6 개.
설정은 6 단계 (s6_train.base_config) 와 같고 모션만 레퍼런스 1 개. 학습량 = s6_train.budget(1) (약 5M).
출력: runs/s6a_ref/<레퍼런스>/ (plan.json, stages/*, metrics.jsonl, done), report.html"""
import json
import s6_train as s6
from common import RUNS
OUT = RUNS / 's6a_ref'


def build():
    a = json.load(open(s6.S5 / 'assignment.json')); shapes = a['shapes']; folders = []
    for ref in a['by_group']:
        f = OUT / ref; f.mkdir(parents=True, exist_ok=True)
        for n in ('references', 'ready'):
            if not (f / n).exists(): (f / n).symlink_to(s6.S5 / n)
        c = s6.base_config(); c.update(resident_designs=len(shapes), train_transitions=s6.budget(1))
        c['checkpoint_steps'] = max(1_000_000, c['train_transitions'] // 10)
        plan = dict(version=1, kind='ref_ppo', reference=ref, motion_count=1, config=c, topology='00000011',
                    designs=[dict(design=k, E=v['E']) for k, v in shapes.items()], eval_designs=[],
                    assignments={k: [ref] for k in shapes}, eval_assignments={k: [] for k in shapes})
        if not (f / 'plan.json').exists(): json.dump(plan, open(f / 'plan.json', 'w'), indent=1)
        folders.append(f)
    return sorted(folders)


def main():
    OUT.mkdir(parents=True, exist_ok=True); folders = build(); status = {}
    for f in folders:
        status[f.name] = True if (f / 'done').exists() else s6.train(f)
    keep = s6.OUT; s6.OUT = OUT  # 보고서·그림을 6a 폴더에
    try: s6.make_report(folders, status, title='6a 단계: 레퍼런스 1 개 × 학습 형상 전부 정책')
    finally: s6.OUT = keep  # 2026-11-12 버그 수정: 복원 안 해 6 단계가 6a 폴더를 완료로 보고 건너뜀
    return status
