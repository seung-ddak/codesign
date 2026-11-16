"""11 단계 (2026-11-17 사용자 요청): s10 의 reward-gradient 유도를 RL 로 대체.
TT = runs/s9_tt_m4 (고정, 학습 안 함). RL = code/tt_dynamics/benchmark_task_inference.py --methods rl
(episodic PPO 가 DDIM 잡음 잔차를 조정, 보상 = 실제 MuJoCo 재생 보상, TT 예측 보상 안 씀).
30 Hz / 240 Hz, PPO 20 update × 8 rollout, 평가 후보 8 (s10 과 같음) → runs/s11_rl/<task>/.
영상 = 과제마다 RL 보상 최고 후보 (왼쪽 MuJoCo 재생 · 오른쪽 TT 예측) + 비교용 s10 영상 복사."""
import json, os, shutil, subprocess
import s10_generate as s10
from common import ROOT, RUNS, PY, log, report, table, video
OUT = RUNS / 's11_rl'; S10 = RUNS / 's10_generate'
ENV = dict(os.environ, TT_CONTROL_HZ='30', TT_PHYSICS_HZ='240')
UPDATES, ROLLOUTS, CAND = 20, 8, 8


def train_all():
    procs = []
    for task in s10.TASKS:
        d = OUT / task
        if (d / task / 'rl_ppo/seed_20000.json').exists(): continue
        cmd = [PY, '-u', str(ROOT / 'code/tt_dynamics/benchmark_task_inference.py'), '--ckpt', str(s10.TT / 'tt'),
               '--out', str(d), '--tasks', task, '--methods', 'rl', '--candidates', str(CAND), '--eval-seeds', '1',
               '--rl-updates', str(UPDATES), '--rl-rollouts', str(ROLLOUTS), '--cpu-threads', '5']
        if d.exists(): cmd.append('--resume')
        log(f's11 RL 학습 {task}')
        procs.append((task, subprocess.Popen(cmd, cwd=ROOT / 'code/tt_dynamics', env=ENV,
                                              stdout=open(OUT / f'{task}.log', 'a'), stderr=subprocess.STDOUT)))
    for task, p in procs:
        if p.wait(): raise RuntimeError(f's11 {task} 실패 (log {OUT / task}.log)')


def main():
    OUT.mkdir(parents=True, exist_ok=True); train_all()
    s10.OUT = OUT  # 영상 함수가 OUT/videos, OUT/<task>/<sc>.npz 를 쓰도록
    rows, hist_rows, vh = [], [], ''
    for task, name in s10.TASKS.items():
        res = json.load(open(OUT / task / task / 'rl_ppo/seed_20000.json'))
        link = OUT / task / 'rl.npz'
        if not link.exists(): os.symlink(res['npz'], link)
        S = {'rl': {'rows': res['candidates']}}
        for x in res['candidates']:
            rows.append((name, 'RL', x['candidate'], '생성 실패' if not x['initial_valid'] else ('넘어짐' if x['fell'] else '생존'),
                         f"{x['valid_seconds']:.1f}", f"{x['reward']:.3f}" if x['reward'] is not None else '-',
                         f"{x['progress']:.2f} {x['progress_unit']}" if x.get('progress') is not None else '-',
                         '성공' if x['task_success'] else '', ' '.join(f'{v:.2f}' for v in x['morphology'])))
        hist = [json.loads(l) for l in open(OUT / task / task / 'rl_training/training.jsonl')]
        hist_rows += [(name, h['iteration'], f"{h.get('reward_mean', float('nan')):.3f}" if 'reward_mean' in h else '-',
                   h['completed'], h['task_successes']) for h in hist]
        for dst, b, sc in s10.videos(task, S):
            vh += video(f'videos/{dst.name}', f"{name} · RL · 후보 {b['candidate']} · 보상 {b['reward']:.3f} · 초록 MuJoCo 실제 / 주황 TT 예측", tag=task)
        for f in sorted((S10 / 'videos').glob(f'{task}_scale_*.mp4')):
            dst = OUT / 'videos' / f's10_{f.name}'
            if not dst.exists(): shutil.copy(f, dst)
            vh += video(f'videos/{dst.name}', f"{name} · 비교: s10 gradient {f.stem.split('_')[1]}_{f.stem.split('_')[2]} 최고 후보", tag=task)
    report(OUT / 'report.html', '11 단계: reward-gradient 유도 대신 RL (PPO) 로 task 형상·행동 생성', [
        ('영상 (과제마다 RL 최고 후보 + s10 gradient 비교)', f'<div class="grid">{vh}</div>'),
        ('RL 후보별 결과', table(['과제', '방법', '후보', 'MuJoCo 재생', '생존 (s)', '보상', '진행', '과제 성공', '형상 E (13)'], rows, filter_col=0)),
        ('PPO 학습 경과', table(['과제', 'update', '보상 평균', '10 초 완주', '과제 성공'], hist_rows, filter_col=0))],
        [('TT', 'runs/s9_tt_m4 (고정, s10 과 같음)'), ('방법', 'episodic PPO: DDIM 각 단계 잡음에 저차원 잔차 (E · 상태 · 관절) 를 더함, 행동 1 개 = 10 초 에피소드 1 개'),
         ('RL 학습량', f'{UPDATES} update × {ROLLOUTS} rollout = {UPDATES * ROLLOUTS} 회 MuJoCo 재생, lr 1e-3, PPO epoch 4'),
         ('평가', f'정책 평균, 후보 {CAND}, seed 20000 (s10 과 후보 수 같음, 잡음 seed 다름)'),
         ('물리 확인', '30 Hz 제어 / 240 Hz 물리, CPU MuJoCo, 10 초'), ('과제', '지그재그 · 원형 트랙 · 빨리 달리기 (s10 과 같음)')])
    log('s11 완료')


if __name__ == '__main__':
    main()
