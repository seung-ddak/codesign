"""9 단계: 생성 모델 (TT) 학습. code/tt_dynamics/train_tt.py 에 8 단계 rollout 을 넘긴다. 인자는 결정 댓글 (tt_args) 로 받는다.
출력: runs/s9_tt/ (train_tt 출력 + report.html)"""
import glob, json, shlex
from pathlib import Path
from common import ROOT, RUNS, PY, log, run, report, table, img
OUT = RUNS / 's9_tt'
DEFAULT_ARGS = '--horizon 64 --steps 20000 --batch 32 --norm minmax --clip 1 --sampler ddim'


def main(tt_args=DEFAULT_ARGS):
    OUT.mkdir(parents=True, exist_ok=True)
    data = sorted(glob.glob(str(RUNS / 's8_collect/G_*/*.npz')))
    cmd = [PY, '-u', str(ROOT / 'code/tt_dynamics/train_tt.py'), '--data', *data, '--out', str(OUT / 'tt'), *shlex.split(tt_args)]
    (OUT / 'command.txt').write_text(' '.join(cmd[:4]) + f' <{len(data)} files> ' + ' '.join(cmd[4 + len(data):]))
    log(f's9 TT 학습 시작 ({len(data)} 파일)'); run(cmd, OUT / 'train.log')
    pngs = sorted((OUT / 'tt').glob('*.png')); tail = (OUT / 'train.log').read_text().splitlines()[-40:]
    report(OUT / 'report.html', '9 단계: 생성 모델 (TT) 학습', [('그림', ''.join(img(p, p.name) for p in pngs) or '<p>그림 없음</p>'),
                                                              ('학습 로그 마지막 40 줄', '<pre>' + '\n'.join(tail) + '</pre>')],
           [('학습기', 'code/tt_dynamics/train_tt.py'), ('데이터', f'8 단계 rollout {len(data)} 파일'), ('인자', tt_args)])
    return OUT / 'tt'
