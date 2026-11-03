"""Report the requested 5s / 8 candidates / 100-iteration diffusion PPO runs."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
rows=[]
video_sections=[]
aggregates={}
fig,axes=plt.subplots(2,2,figsize=(12,8))
for col,task in enumerate(('zigzag','circle')):
    run=ROOT/f'{task}_seed0'
    result=json.loads((run/'summary.json').read_text())
    protocol=json.loads((run/'protocol.json').read_text())
    logs=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
    assert len(logs)==100 and protocol['episode_states']==250 and protocol['batch']==8
    assert protocol['diffusion_steps']==5 and logs[-1]['denoising_transitions']==4000
    aggregates[task]={}
    for method in ('unguided','reward_gradient','rl_before','rl_after'):
        candidates=result[method]['rows']
        median=lambda k:float(np.median([r[k] for r in candidates]))
        original=[r['original_reward'] for r in candidates if r['original_reward'] is not None]
        a=dict(terminal_reward_median=median('terminal_reward'),progress_median=median('progress'),
               survived=sum(r['survived'] for r in candidates),
               invalid_initial=sum(r['invalid_initial'] for r in candidates),
               original_reward_median=float(np.median(original)) if original else None,
               prediction_position_rmse_median=median('prediction_position_rmse'))
        aggregates[task][method]=a
        original_text=f"{a['original_reward_median']:.4f}" if original else 'N/A'
        rows.append(f'<tr><td>{task}</td><td>{method}</td><td>{a["terminal_reward_median"]:.4f}</td>'
                    f'<td>{original_text}</td><td>{a["progress_median"]:.4f}</td>'
                    f'<td>{a["survived"]}/8</td><td>{a["invalid_initial"]}</td>'
                    f'<td>{a["prediction_position_rmse_median"]:.4f}</td></tr>')
    ax=axes[0,col]
    values=[r['terminal_reward_mean'] for r in logs]
    ax.plot(range(1,101),values,alpha=.35,label='batch mean')
    ax.plot(range(10,101),np.convolve(values,np.ones(10)/10,mode='valid'),label='10-iteration mean')
    ax.set_title(task+' training');ax.set_xlabel('PPO iteration');ax.set_ylabel('physical terminal reward');ax.legend();ax.grid(alpha=.2)
    ax=axes[1,col]
    for method,color in [('rl_before','#999999'),('rl_after','#1679c4')]:
        with np.load(run/f'{method}.npz') as data:
            for i in range(8):
                xy=data[f'tokens_{i}'][:,:2]
                ax.plot(xy[:,0],xy[:,1],color=color,alpha=.6,label=method if i==0 else None)
    if task=='circle':
        t=np.linspace(0,2*np.pi,200);ax.plot(np.sin(t),1-np.cos(t),'k--',alpha=.5,label='track center')
    else:
        x=np.linspace(0,3,200);ax.plot(x,.3*2/np.pi*np.arcsin(np.sin(np.pi*x)),'k--',alpha=.5,label='track center')
    ax.set_title(task+' held-out noise evaluation');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_aspect('equal',adjustable='datalim');ax.legend();ax.grid(alpha=.2)
    for i in range(2):
        p=html.escape(str(ROOT/f'videos/{task}_candidate{i}_rl_track.mp4'),quote=True)
        video_sections.append(f'<h3>{task} · 후보 {i}</h3><video controls preload="metadata" src="{p}"></video><p><a href="{p}">{p}</a></p>')
fig.tight_layout();fig.savefig(ROOT/'results.png',dpi=150);plt.close(fig)
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Diffusion PPO · 5초 · 후보8 · 100iter</title>
<style>body{font:16px/1.8 system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#20303c}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd3dd;padding:7px}img,video{width:100%}a{color:#1565c0}</style>
<h1>Diffusion PPO: 5초 · 후보 8개 · 100 iteration</h1>
<p><strong>두 task의 학습 및 별도 평가 잡음에서 최종 평가 완료.</strong> 후보는 고정 형상 8개가 아니라 iteration마다 잡음에서 새로 생성한 E·S·A다. E를 포함한 공동 생성 과정에 PPO가 개입한다.</p>
<p>디노이징 5단계, 100 iteration, iteration당 후보 8개, PPO 내부 optimizer epoch 4회. task당 학습 물리 episode 800개, PPO 전이 4,000개. 각 episode는 250 상태 프레임/249 물리 제어 전이(4.98초)이며 낙상 시 조기 종료한다. 영상은 5초 실제 속도로 표시한다.</p>
<p>TT는 v16 twin으로 동결했다. S0=rest(E), PPO 보정은 epsilon의 E 및 구간별 S/A 채널에 적용한다. RL 경로에서 reward gradient를 계산하지 않는다. 학습 보상은 실제 dense return − 5×생성/물리 위치 MSE다. 학습과 평가 잡음은 분리하며 최종 정책을 평가했다.</p>
<table><tr><th>Task</th><th>방법</th><th>학습 목적 보상 중앙값</th><th>기존 평가 보상 중앙값</th><th>진행 중앙값</th><th>생존</th><th>시작 무효</th><th>예측/물리 위치 RMSE 중앙값(m)</th></tr>'''+''.join(rows)+'''</table>
<p>지그재그 진행 단위 m, 원형 바퀴. 시작 무효 후보는 학습 목적 보상 -20으로 포함하고 실제 실행이 없는 기존 보상은 제외한다. 보상과 생존만으로 task 성공·수렴을 선언하지 않는다.</p>
<h2>학습과 실제 이동</h2><img src="results.png" alt="Training reward and physical evaluation paths">
<p>학습 곡선은 매 iteration 서로 다른 잡음의 batch 평균이다. 고정 평가 후보의 성능 곡선으로 해석하지 않는다. 아래 경로는 별도 평가 잡음 후보 8개 전체다.</p>
<h2>학습 후 RL 영상</h2><p>첫 평가 후보 0·1을 사전에 정해 task별로 표시한다. 노란색은 목표 중심선, 흰색은 경계다. 각 후보의 생성 E와 저장된 물리 qpos를 렌더링했다. 낙상 뒤에는 마지막 프레임을 유지한다. 전체 8후보 기록은 원자료에 있다.</p>'''+''.join(video_sections)+'''
<p>단일 seed·별도 평가 잡음 한 batch의 결과다. 더 긴 task나 다른 seed의 일반화는 검증하지 않았다. 이전 4초·2iter 결과와는 생성 길이와 후보 수가 달라 직접적인 인과 비교가 아니다.</p>
<p><a href="README.md">조건·실행 명령</a> · <a href="aggregates.json">요약 수치</a> · <a href="zigzag_seed0/summary.json">지그재그 전체 결과</a> · <a href="circle_seed0/summary.json">원형 전체 결과</a></p></html>'''
(ROOT/'report.html').write_text(page)
(ROOT/'aggregates.json').write_text(json.dumps(aggregates,indent=2))
print(json.dumps(aggregates,indent=2))
print(ROOT/'report.html')
