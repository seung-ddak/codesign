"""10 단계 (2026-11-16 사용자 결정): TT 로 task 형상·행동 생성. 지그재그 (zigzag), 원형 트랙 (circle), 빨리 달리기 (straight).
TT = s9 인자 + --modes 0 1 2 4 (생성기의 형상 생성에 mode 4 필요), 200k step → runs/s9_tt_m4/.
생성 = code/tt_dynamics/generate_task_mujoco.py (30 Hz / 240 Hz, 후보 8, 유도 0·1) → runs/s10_generate/<task>/.
영상 = 과제·유도마다 보상 최고 후보, 왼쪽 MuJoCo 재생 · 오른쪽 TT 예측."""
import json, os, subprocess, sys
import numpy as np
import s9_tt, s9_report
from common import ROOT, RUNS, PY, log, run, report, table, video
TT = RUNS / 's9_tt_m4'; OUT = RUNS / 's10_generate'
TASKS = {'zigzag': '지그재그', 'circle': '원형 트랙', 'straight': '빨리 달리기 (직진)'}
HZ = 30


def train():
    if (TT / 'tt/checkpoint.pt').exists(): return
    s9_tt.OUT = s9_report.OUT = TT; s9_report.TT = TT / 'tt'; s9_report.V = TT / 'videos'
    s9_tt.main('--horizon 64 --steps 200000 --batch 32 --norm minmax --clip 1 --sampler ddim --modes 0 1 2 4')


def generate(task):
    d = OUT / task
    if (d / 'summary.json').exists(): return
    env = dict(os.environ, TT_CONTROL_HZ=str(HZ), TT_PHYSICS_HZ='240')
    log(f's10 생성 {task}')
    run([PY, '-u', str(ROOT / 'code/tt_dynamics/generate_task_mujoco.py'), '--ckpt', str(TT / 'tt'), '--out', str(d),
         '--task', task, '--candidates', '8', '--scales', '0', '1'], OUT / f'{task}.log', env=env)


def to50(x, m):
    x = x[m]; t = np.arange(len(x)) / HZ; t2 = np.arange(0, t[-1], 1 / 50)
    return np.stack([np.interp(t2, t, x[:, j]) for j in range(x.shape[1])], 1)


def videos(task, rows):
    """과제·유도마다 보상 최고 후보 1 개: 과제 환경 한 장면에 MuJoCo 실제 (초록) · TT 예측 (주황 반투명) 겹침 (11-17 사용자 요청)."""
    sys.path.insert(0, str(ROOT / 'code/tt_dynamics'))
    from render_tt import render_task_overlay
    from generate_task_mujoco import TASKS as TK
    out = []
    for sc, r in rows.items():
        ok = [x for x in r['rows'] if x['initial_valid'] and x['reward'] is not None]
        if not ok: continue
        b = max(ok, key=lambda x: x['reward']); i = b['candidate']
        dst = OUT / 'videos' / f'{task}_{sc}_c{i}.mp4'; out.append((dst, b, sc))
        if dst.exists(): continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        z = np.load(OUT / task / f'{sc}.npz'); E = z['E'][i]
        render_task_overlay(s9_report.to_qpos(to50(z['mujoco'][i], z['valid'][i])),
                            s9_report.to_qpos(to50(z['pred'][i], np.ones(len(z['pred'][i]), bool))), E, TK[task](), dst,
                            title=f'{task} · {sc} · candidate {i} · reward {b["reward"]:.2f}', fell_t=b['valid_seconds'] if b['fell'] else None)
        log(f'영상 {dst.name}')
    return out


def main():
    train(); OUT.mkdir(parents=True, exist_ok=True); rows, vh = [], ''
    for task in TASKS:
        generate(task); S = json.load(open(OUT / task / 'summary.json'))['results']
        for sc, r in S.items():
            for x in r['rows']:
                rows.append((TASKS[task], sc, x['candidate'], '생성 실패' if not x['initial_valid'] else ('넘어짐' if x['fell'] else '생존'),
                             f"{x['valid_seconds']:.1f}", f"{x['reward']:.3f}" if x['reward'] is not None else '-',
                             f"{x['progress']:.2f} {x['progress_unit']}" if x.get('progress') is not None else '-',
                             '성공' if x['task_success'] else '', ' '.join(f'{v:.2f}' for v in x['morphology'])))
        for dst, b, sc in videos(task, S):
            vh += video(f'videos/{dst.name}', f"{TASKS[task]} · {sc} · 후보 {b['candidate']} · 보상 {b['reward']:.3f} · 한 장면: 초록 MuJoCo 실제 / 주황 TT 예측", tag=task)
    report(OUT / 'report.html', '10 단계: TT 로 task 형상·행동 생성', [
        ('영상 (과제·유도마다 보상 최고 후보)', f'<div class="grid">{vh}</div>'),
        ('후보별 결과', table(['과제', '유도', '후보', 'MuJoCo 재생', '생존 (s)', '보상', '진행', '과제 성공', '형상 E (13)'], rows, filter_col=0))],
        [('TT', f'runs/s9_tt_m4 (s9 인자 + --modes 0 1 2 4, 200k step)'), ('생성기', 'code/tt_dynamics/generate_task_mujoco.py, 후보 8, 유도 0 (보상 없음) · 1'),
         ('물리 확인', 'TT 가 만든 형상·관절 목표를 CPU MuJoCo 에서 그대로 재생 (30 Hz 제어 / 240 Hz 물리, kp 30, 10 초)'),
         ('과제', '지그재그 = 슬라럼 폴 통과, 원형 트랙 = 통로 안 원 한 바퀴, 빨리 달리기 = 직진 통로 안 10 초 전진 거리')])


if __name__ == '__main__':
    main()
