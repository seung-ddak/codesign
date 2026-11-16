"""6 단계: trot 레퍼런스 묶음마다 공유 PPO 정책 1 개 (학습 형상 전부를 함께, 형상 토큰 입력).
기존 ref_ppo 설정 (ref_ppo/_tools/build_runs.py: 동시 환경 1,024, lr 1e-3, 256-128-64, 30 Hz / 240 Hz (11-12 변경, 원래 50/200), 미끄러짐 감점 없음,
넘어지면 감점 없이 조기 종료, step 보상 ≥ 0, 학습량 5M + 모션당 1 만 step) 을 그대로 쓰고, 체크포인트를 학습량의 1/10 마다 저장 (7 단계 선택용).
학습기: code/motion_tracking/shared_calm_train.py (ref_ppo/_tools/train_queue.py 와 같은 호출). 한 번에 하나 (GPU).
출력: runs/s6_train/<레퍼런스>/ (plan.json, stages/*/files/checkpoints/*, metrics.jsonl, done), report.html"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from common import ROOT, RUNS, CODE, PY, log, report, table, img
sys.path.append(str(ROOT / 'ref_ppo/_tools'))  # 2026-11-12: insert(0) 이면 _tools/queue.py 가 표준 queue 를 가려 s7 실패 (11-12 train_queue.py 로 이름 변경)
OUT = RUNS / 's6_train'; S5 = RUNS / 's5_retarget'


def base_config():
    base = json.load(open(ROOT / 'legacy/runs/ppo_tracking/try2/plan.json'))
    c = dict(base['config'])
    c.update(num_envs=1024, batch_size=1024, unroll_length=24, num_minibatches=4, num_updates_per_batch=8, learning_rate=1e-3,
             entropy_cost=0.01, hidden=[256, 128, 64], control_hz=30, physics_hz=240,  # 2026-11-12 사용자 결정: CALM 과 같은 30 Hz (shared_calm_common 은 30/240 만 허용)
             max_episode_length=1000,
             penalty_termination=0.0, reward_clip_positive=True, penalty_foot_slip=0.0, metrics_steps=2_000_000, seed=0, rollout_batch=256)
    return c


def budget(n):
    return int(min(50_000_000, 5_000_000 + 10_000 * n))


def build():
    a = json.load(open(S5 / 'assignment.json')); shapes = a['shapes']; folders = []
    for ref, per in a['by_group'].items():
        f = OUT / ref; f.mkdir(parents=True, exist_ok=True)
        for n in ('references', 'ready'):
            if not (f / n).exists(): (f / n).symlink_to(S5 / n)
        m = min(len(v) for v in per.values()); per = {k: v[:m] for k, v in per.items()}  # 2026-11-12 수정: 상주 형상은 클립 수가 같아야 함 → 형상별 모션 수를 최솟값으로 맞춤
        n_motion = len({c for v in per.values() for c in v}); c = base_config()
        c.update(resident_designs=len(shapes), train_transitions=budget(n_motion)); c['checkpoint_steps'] = max(1_000_000, c['train_transitions'] // 10)
        plan = dict(version=1, kind='ref_ppo', reference=ref, motion_count=n_motion, config=c, topology='00000011',
                    designs=[dict(design=k, E=v['E']) for k, v in shapes.items()], eval_designs=[],
                    assignments={k: [ref] + per[k] for k in shapes}, eval_assignments={k: [] for k in shapes})
        if not (f / 'plan.json').exists(): json.dump(plan, open(f / 'plan.json', 'w'), indent=1)
        folders.append(f)
    return sorted(folders, key=lambda f: -json.load(open(f / 'plan.json'))['motion_count'])


def step_of(f):
    p = f / 'learner.json'
    return json.load(open(p))['step'] if p.exists() else 0


def train(f):
    target = json.load(open(f / 'plan.json'))['config']['train_transitions']
    for attempt in range(3):
        if step_of(f) >= target: break
        visit = len(list((f / 'stages').glob('[0-9]*'))) if (f / 'stages').exists() else 0
        log(f's6 START {f.name} visit {visit} step {step_of(f)}/{target}')
        with open(f / 'train.log', 'a') as fh:
            r = subprocess.run([PY, '-u', 'shared_calm_train.py', '--out', str(f), '--stage', '0', '--visit', str(visit), '--steps', str(target - step_of(f)),
                                '--seconds', '1e12'], cwd=CODE, stdout=fh, stderr=subprocess.STDOUT,
                               env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1'))
        log(f's6 EXIT {f.name} code {r.returncode} step {step_of(f)}')
        if r.returncode != 0: return False
    ok = step_of(f) >= target * 0.98
    if ok: (f / 'done').write_text(json.dumps(dict(step=step_of(f), time=time.time())))
    return ok


def main():
    folders = build(); status = {}
    for f in folders:
        status[f.name] = True if (f / 'done').exists() else train(f)
    make_report(folders, status)
    return status


def make_report(folders, status, title='6 단계: 레퍼런스 묶음별 공유 정책 학습'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5)); rows = []
    for f in folders:
        m = [json.loads(l) for l in open(f / 'metrics.jsonl')] if (f / 'metrics.jsonl').exists() else []
        m = [x for x in m if 'episode/sum_reward' in x['metrics']]
        if m:
            ax[0].plot([x['step'] / 1e6 for x in m], [x['metrics']['episode/sum_reward'] for x in m], label=f.name.replace('__r00', ''))
            ax[1].plot([x['step'] / 1e6 for x in m], [x['metrics'].get('episode/length', np.nan) for x in m])
        p = json.load(open(f / 'plan.json'))
        rows.append((f.name.replace('__r00', ''), p['motion_count'], f"{p['config']['train_transitions'] / 1e6:.1f}M", f'{step_of(f) / 1e6:.1f}M', '완료' if status.get(f.name) else '실패'))
    ax[0].set_title('학습 보상 (에피소드 합)'); ax[1].set_title('에피소드 길이'); ax[0].legend(fontsize=8)
    for a in ax: a.set_xlabel('학습 step (M)'); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'curves.png', dpi=105); plt.close(fig)
    p0 = json.load(open(folders[0] / 'plan.json'))
    report(OUT / 'report.html', title, [
        ('학습 곡선', img(OUT / 'curves.png')), ('정책별 상태', table(['묶음', 'CALM 모션', '목표 step', '학습 step', '상태'], rows, lambda i: 'bad' if rows[i][4] != '완료' else ''))],
        [('정책', '묶음마다 1 개, 형상 토큰 (13) 입력, 학습 형상 전부 함께 (shared_calm_train.py)'),
         ('학습 형상', ', '.join(d['design'] for d in p0['designs'])),
         ('PPO', f"동시 환경 {p0['config']['num_envs']}, lr {p0['config']['learning_rate']}, 망 256-128-64, {p0['config']['control_hz']} Hz 제어 / {p0['config']['physics_hz']} Hz 물리"),
         ('보상', '트래킹 보상 Peng 2020, 미끄러짐 감점 없음, 넘어지면 감점 없이 조기 종료, step 보상 ≥ 0'),
         ('학습량', '5M + 모션당 1 만 step (최대 50M), 체크포인트 = 학습량의 1/10 마다')])
