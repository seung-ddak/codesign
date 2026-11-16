"""학습 큐: A (ref_ppo/<레퍼런스>) 를 모두 학습한 뒤 B (E_ppo/<E>/<레퍼런스>) 를 학습한다. 한 번에 정책 하나 (GPU).
각 폴더의 learner.json step 이 plan 의 train_transitions 이상이면 done 으로 보고 건너뛴다. 멈추면 다시 실행해 이어 간다.
  tmux new-session -d -s ref_queue "python ref_ppo/_tools/train_queue.py"
"""
import json, os, subprocess, sys, time
from pathlib import Path
TOOLS = Path(__file__).resolve().parent; REFPPO = TOOLS.parent; ROOT = REFPPO.parent
PY = str(ROOT.parent / 'transformer-transformer/.venv/bin/python')
CODE = ROOT / 'code/motion_tracking'
LOG = REFPPO / '_tools/queue.log'


def log(msg):
    with LOG.open('a') as f: f.write(f"{time.strftime('%F %T')} {msg}\n")


def runs():
    a = sorted((p for p in REFPPO.iterdir() if (p / 'plan.json').exists()), key=lambda p: -json.load(open(p / 'plan.json'))['motion_count'])
    b = sorted(p.parent for p in (ROOT / 'E_ppo').glob('*/*/plan.json'))
    return a + b


def step_of(folder):
    f = folder / 'learner.json'
    return json.load(open(f))['step'] if f.exists() else 0


def train(folder):
    plan = json.load(open(folder / 'plan.json')); target = plan['config']['train_transitions']
    while step_of(folder) < target:
        visit = len(list((folder / 'stages').glob('[0-9]*'))) if (folder / 'stages').exists() else 0
        remaining = target - step_of(folder)
        log(f'START {folder.relative_to(ROOT)} visit {visit} from {step_of(folder)} remaining {remaining}')
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
        with open(folder / 'train.log', 'a') as fh:
            r = subprocess.run([PY, '-u', 'shared_calm_train.py', '--out', str(folder), '--stage', '0', '--visit', str(visit),
                                '--steps', str(remaining), '--seconds', '1e12'], cwd=CODE, stdout=fh, stderr=subprocess.STDOUT, env=env)
        log(f'EXIT {folder.relative_to(ROOT)} code {r.returncode} step {step_of(folder)}')
        if r.returncode != 0:
            return False
        if step_of(folder) < target:  # 학습 루프가 eval 단위로 끝나 목표에 조금 못 미치면 한 번 더 (최대 1 회)
            if remaining - (target - step_of(folder)) <= 0:
                return False
    (folder / 'done').write_text(json.dumps(dict(step=step_of(folder), time=time.time())))
    return True


if __name__ == '__main__':
    for folder in runs():
        if (folder / 'done').exists():
            continue
        if not train(folder):
            log(f'FAIL {folder.relative_to(ROOT)} (다음으로 넘어감)')
    log('QUEUE DONE')
