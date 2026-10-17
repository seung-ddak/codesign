"""Run the CGR experiment in tmux; publish metrics and notify its dedicated Issue."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/motion_tracking'))
from notify_selected_experts import post, TOKEN_FILE

OUT = Path('runs/pipeline_v1/tt/h32_try4_cgr')
BRANCH = 'experiment/h32_try4_cgr'
FLAGS = ['--horizon', '32', '--steps', '100000', '--batch', '32', '--lr', '0.0002',
         '--d-model', '256', '--layers', '6', '--diff-steps', '5', '--norm', 'minmax',
         '--clip', '1', '--modes', '0', '1', '2', '--sampler', 'ddim', '--seed', '0']

def finite(value):
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, list):
        for v in value: finite(v)
    elif isinstance(value, (float, int)):
        if not math.isfinite(value): raise ValueError('Non-finite result')


def main():
    os.chdir(ROOT)
    token = TOKEN_FILE.read_text().strip()
    issue = json.loads((OUT / 'issue.json').read_text())['number']
    def notify(body):
        post(f'/issues/{issue}/comments', {'body': '@seung-ddak\n' + body}, token)
    try:
        if (OUT / 'progress.jsonl').exists() or (OUT / 'checkpoint.pt').exists():
            raise FileExistsError('Refusing to overwrite an existing training run')
        cmd = [sys.executable, 'code/tt_dynamics/train_tt.py', '--out', str(OUT), *FLAGS]
        for name, extra in [('train.log', []), ('diag.log', ['--eval-only', '--diag'])]:
            with (OUT / name).open('x') as log:
                subprocess.run(cmd + extra, stdout=log, stderr=subprocess.STDOUT, check=True)
        progress = [json.loads(line) for line in (OUT / 'progress.jsonl').read_text().splitlines()]
        assert progress[-1]['step'] == 100000
        finite(progress)
        metrics = {}
        for name in ('eval_dynamics_ddim.json', 'diag_window.json'):
            metrics[name] = json.loads((OUT / name).read_text())
            finite(metrics[name])
        assert (OUT / 'checkpoint.pt').stat().st_size > 0
        summary = '# h32_try4_cgr complete\n\n100,000 steps; horizon 32; modes 0/1/2; diffusion 5; minmax; DDIM.\n\n'
        summary += 'Checkpoint and generated sample arrays remain on the CGR PC.\n\n```json\n'
        summary += json.dumps({'last_progress': progress[-1], 'window': metrics['diag_window.json']['전체'],
                               'episode': metrics['eval_dynamics_ddim.json']['all']}, ensure_ascii=False, indent=2) + '\n```\n'
        (OUT / 'RESULTS.md').write_text(summary)
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        assert branch == BRANCH, branch
        subprocess.run(['git', 'diff', '--cached', '--quiet'], check=True)
        files = [OUT / name for name in ('config.json', 'progress.jsonl', 'eval_dynamics_ddim.json',
                 'diag_window.json', 'train.log', 'diag.log', 'RESULTS.md')]
        subprocess.run(['git', 'add', '--', *map(str, files)], check=True)
        subprocess.run(['git', 'commit', '-q', '-m', 'Record h32_try4_cgr training and dynamics evaluation'], check=True)
        subprocess.run(['git', 'push', 'origin', f'HEAD:refs/heads/{BRANCH}'], check=True)
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        notify(f'학습·평가 및 결과 업로드 완료: https://github.com/seung-ddak/codesign/commit/{sha}\n\n' + summary)
        (OUT / 'completed.txt').write_text(sha + '\n')
    except Exception as error:
        (OUT / 'failed.txt').write_text(f'{type(error).__name__}: {error}\n')
        try: notify(f'h32_try4_cgr 작업 실패: {type(error).__name__}: {error}. 로컬 로그를 확인하세요.')
        except Exception as notify_error: print('Failure notification also failed:', type(notify_error).__name__, file=sys.stderr)
        raise

if __name__ == '__main__':
    main()
