"""6h 단계: 모션 세트마다 정책 1 개 (형상 9 개 함께, 형상 E 입력), 100 Hz / 200 Hz (2026-11-17 사용자 결정, check/tracking_policy/PLAN_100HZ.md).
정책 하나 = trot 레퍼런스 1 + CALM 1000 × 형상 9 (s5h_retarget100). 학습 환경 레퍼런스 속도 평활 70 ms (TRACK_SG_SEC=0.07, 사용자 확정).
나머지 PPO · 보상 = s6_train.base_config (= hz100 확인 실험). 체크포인트 = 학습량의 1/10 마다. 세트의 리타겟이 다 끝나야 (ready 9/9) 학습 시작.
실행: python s6h_train100.py --steps <정책당 transition> [--sets ...]
출력: runs/s6h_train100/<세트>/ (plan.json, stages/*/files/checkpoints/*, metrics.jsonl, done)"""
import argparse, json, os, time
from common import RUNS, log
import s6_train as s6
OUT = RUNS / 's6h_train100'; S5H = RUNS / 's5h_retarget100'
os.environ['TRACK_SG_SEC'] = '0.07'  # 학습 환경 평활 70 ms (s6_train.train 이 os.environ 을 넘김)


def build(g, shapes, clips, steps):
    f = OUT / g; f.mkdir(parents=True, exist_ok=True)
    if not (f / 'references').exists(): (f / 'references').symlink_to(S5H / 'references')
    if not (f / 'ready').exists(): (f / 'ready').symlink_to(S5H / 'ready' / g)
    c = s6.base_config()
    c.update(control_hz=100, physics_hz=200, max_episode_length=2000, resident_designs=len(shapes), train_transitions=steps,
             checkpoint_steps=max(1_000_000, steps // 10), track_sg_sec=0.07)
    plan = dict(version=1, kind='ref_ppo', reference=g, motion_count=len(clips), config=c, topology='00000011',
                designs=[dict(design=k, E=v['E']) for k, v in shapes.items()], eval_designs=[],
                assignments={k: list(clips) for k in shapes}, eval_assignments={k: [] for k in shapes})
    if not (f / 'plan.json').exists(): json.dump(plan, open(f / 'plan.json', 'w'), indent=1)
    return f


def chunks(clips, n):
    """레퍼런스 (clips[0]) 는 모든 묶음에, CALM 은 n 개 묶음에 고르게 (GPU 12 GB 에 형상 9 × 약 250 클립까지, check/tracking_policy/bank_probe)."""
    ref, calm = clips[0], clips[1:]
    return [[ref] + calm[i::n] for i in range(n)]


def train_rotating(f, parts, visit_steps):
    """묶음을 번갈아 학습 (2026-11-17 사용자 결정 a). 방문마다 plan 의 assignments 를 다음 묶음으로 바꾸고 shared_calm_train 을 visit_steps 만큼 실행.
    학습 상태 (파라미터 · Adam · 관측 정규화) 는 learner.msgpack 로 이어진다."""
    import subprocess
    from common import CODE, PY
    p = json.load(open(f / 'plan.json')); target = p['config']['train_transitions']; fails = 0
    while s6.step_of(f) < target and fails < 3:
        visit = len(list((f / 'stages').glob('[0-9]*'))) if (f / 'stages').exists() else 0; k = visit % len(parts)
        p['assignments'] = {s: parts[k] for s in p['assignments']}; p['chunk'] = dict(index=k, count=len(parts), clips=len(parts[k]))
        json.dump(p, open(f / 'plan.json', 'w'), indent=1)
        n = min(visit_steps, target - s6.step_of(f)); log(f's6h START {f.name} visit {visit} chunk {k}/{len(parts)} step {s6.step_of(f)}/{target}')
        with open(f / 'train.log', 'a') as fh:
            r = subprocess.run([PY, '-u', 'shared_calm_train.py', '--out', str(f), '--stage', '0', '--visit', str(visit), '--steps', str(n), '--seconds', '1e12'],
                               cwd=CODE, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1'))
        log(f's6h EXIT {f.name} visit {visit} code {r.returncode} step {s6.step_of(f)}')
        fails = fails + 1 if r.returncode else 0
    ok = s6.step_of(f) >= target * 0.98
    if ok: (f / 'done').write_text(json.dumps(dict(step=s6.step_of(f), time=time.time())))
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, required=True); ap.add_argument('--sets', nargs='*')
    ap.add_argument('--chunks', type=int, default=4); ap.add_argument('--visit-steps', type=int, default=25_000_000); a = ap.parse_args()
    A = json.load(open(S5H / 'assignment.json')); shapes, sets = A['shapes'], A['sets']
    order = a.sets or list(sets)
    for g in order:
        while len(list((S5H / 'ready' / g).glob('*.json'))) < len(shapes):
            time.sleep(60)
        f = build(g, shapes, sets[g], a.steps)
        ok = (f / 'done').exists() or train_rotating(f, chunks(sets[g], a.chunks), min(a.visit_steps, a.steps // a.chunks))
        log(f's6h {g} {"완료" if ok else "실패"} step {s6.step_of(f)}')
    log('s6h END')


if __name__ == '__main__':
    main()
