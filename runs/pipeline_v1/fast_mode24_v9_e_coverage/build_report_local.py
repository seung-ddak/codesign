"""Report the frozen v9 comparison and native MJX results."""
import os
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/'evaluation_local';BASE=P.parent/'fast_mode24_v8_film/A/check_w1'
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:]);os.nice(10)
os.environ.setdefault('MPLCONFIGDIR',str(P.parents[2]/'.runtime/mpl'))
import json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
read=lambda p:json.loads(p.read_text())
held=read(O/'given_a_held.json');seen=read(O/'given_a_seen.json')
assert held['v8']['start30']['pairs']==held['v9']['start30']['pairs']>0
labels={'height':'높이','tilt':'기울기','linvel':'이동 속도','yaw':'방향','pos':'몸통 위치','joint':'관절각'}
comparison=[]
for g,label in labels.items():
 a,b=[held[v]['start30'][g] for v in ['v8','v9']]
 comparison.append(dict(group=g,label=label,v8=a['e_agree_moving'],v9=b['e_agree_moving'],change=b['e_agree_moving']-a['e_agree_moving'],v8_wrongE_penalty=a['wrongE_moving']-a['pin_moving'],v9_wrongE_penalty=b['wrongE_moving']-b['pin_moving'],v8_rmse=a['pin_moving'],v9_rmse=b['pin_moving']))
main_pass=all(r['change']>=.1 for r in comparison if r['group'] in ['height','tilt','linvel'])
# Keep the original comparison's design order and aggregate seeds within morphology.
rng=np.random.default_rng(0)
def eta(y,g):
 den=((y-y.mean())**2).sum()
 return float(sum((g==k).sum()*(y[g==k].mean()-y.mean())**2 for k in np.unique(g))/den) if den else 0.
def task_result(folder,task):
 summary=read(folder/f'fixedE_{task}_4s/summary.json');r=summary['runs']['scale_10']
 n=16;assert len(r)==n*4
 assert all(all(row['E']==r[i*4]['E'] for row in r[i*4:(i+1)*4]) for i in range(n))
 g=np.repeat(np.arange(n),len(r)//n);pred=np.array([x['pred_total'] for x in r]);actual=np.array([np.nan if x['mjx_total'] is None else x['mjx_total'] for x in r]);valid=np.isfinite(actual)
 predmeans=np.array([pred[g==i].mean() for i in range(n)]);actmeans=np.array([np.nanmean(actual[g==i]) for i in range(n)])
 effect=eta(pred,g);p=float(np.mean([eta(pred,rng.permutation(g))>=effect for _ in range(10000)]));rank=float(spearmanr(predmeans,actmeans).statistic)
 joint=read(folder/f'joint_{task}_4s/summary.json')['runs']['scale_10'];surv=sum(x['mjx_fell_step']==-1 for x in joint)
 smoke=read(folder/'smoke_zigzag_joint_c16/summary.json')['runs']['scale_0']
 path=float(np.median([x['path_err_m'] for x in smoke if x['path_err_m'] is not None]))
 return dict(pred_effect_p=p,pred_effect_eta2=effect,rank_correlation=rank,smoke_path_error_m=path,smoke_survived=sum(x['mjx_fell_step']==-1 for x in smoke),smoke_total=len(smoke),gate=p<.05 and rank>.6 and path<.15,fixed_valid=int(valid.sum()),fixed_total=len(r),joint_survived=surv,joint_total=len(joint),joint_reward_median=float(np.median([x['mjx_total'] for x in joint if x['mjx_total'] is not None])),joint_progress_median=float(np.median([x['mjx_progress'] for x in joint if x['mjx_progress'] is not None])),predicted_design_reward=predmeans.tolist(),actual_design_reward=actmeans.tolist())
tasks={v:{task:task_result(folder,task) for task in ['zigzag','circle']} for v,folder in [('v8',BASE),('v9',O/'check_w1')]}
resources=[json.loads(x) for x in (O/'resources.jsonl').read_text().splitlines()]
limits=dict(min_ram_available_gib=min(x['ram_available_gib'] for x in resources),max_gpu_used_mib=max(float(x['gpu'].split(',')[0]) for x in resources if x['gpu']))
result=dict(comparison=comparison,main_sensitivity_pass=main_pass,tasks=tasks,resources=limits,paired_cases=held['v9']['start30']['pairs'])
(O/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(len(comparison));ax.bar(x-.18,[r['v8'] for r in comparison],.36,label='v8');ax.bar(x+.18,[r['v9'] for r in comparison],.36,label='v9');ax.set(xticks=x,xticklabels=[r['group'] for r in comparison],ylabel='Morphology-change agreement');ax.axhline(0,color='#666',linewidth=.6);ax.legend();fig.savefig(O/'agreement.png',dpi=160)
progress=[json.loads(x) for x in (P/'tt_joint/progress.jsonl').read_text().splitlines()];fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.plot([x['step'] for x in progress],[x['loss'] for x in progress],alpha=.45,label='train minibatch');ax.plot([x['step'] for x in progress],[x['val_loss'] for x in progress],label='validation');ax.set(xlabel='Training steps',ylabel='Loss',yscale='log');ax.legend();fig.savefig(O/'training.png',dpi=160)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,task in zip(axes,['zigzag','circle']):
 for v in tasks:
  t=tasks[v][task];ax.scatter(t['predicted_design_reward'],t['actual_design_reward'],label=v)
 ax.set(title=task,xlabel='Predicted mean reward per morphology',ylabel='MJX mean reward per morphology');ax.legend()
fig.savefig(O/'reward_scatter.png',dpi=160)
rows=''.join(f'<tr><td>{r["label"]}</td><td>{r["v8"]:.3f}</td><td>{r["v9"]:.3f}</td><td>{r["change"]:+.3f}</td><td>{r["v8_wrongE_penalty"]:.4f}</td><td>{r["v9_wrongE_penalty"]:.4f}</td></tr>' for r in comparison)
taskrows=''
for task in ['zigzag','circle']:
 for v in tasks:
  t=tasks[v][task];taskrows+=f'<tr><td>{v} · {task}</td><td>{t["pred_effect_p"]:.4f}</td><td>{t["rank_correlation"]:.3f}</td><td>{t["smoke_path_error_m"]*100:.1f}</td><td>{t["smoke_survived"]}/{t["smoke_total"]}</td><td>{"통과" if t["gate"] else "미달"}</td><td>{t["joint_survived"]}/{t["joint_total"]}</td><td>{t["joint_reward_median"]:.3f}</td><td>{t["joint_progress_median"]:.3f}</td></tr>'
videos=''
for task in ['zigzag','circle']:
 files=sorted((O/f'check_w1/joint_{task}_4s/videos').glob('*.mp4'));assert len(files)==16
 videos+=f'<h3>{task} · 공동 생성 후보 16개</h3>'
 for f in files:videos+=f'<details><summary>{html.escape(f.stem)}</summary><video controls preload="none" src="{f.resolve().as_uri()}"></video></details>'
conclusion='미학습 형상에서의 움직임 일치도 개선 기준을 통과하지 못했다.' if not main_pass else '미학습 형상에서의 움직임 일치도 개선 기준을 통과했다.'
conclusion += f' 실제 MJX Co-Design Gate도 두 Task 모두 미달이며, 4초 공동 생성 생존은 지그재그 {tasks["v9"]["zigzag"]["joint_survived"]}/16, 원형 {tasks["v9"]["circle"]["joint_survived"]}/16이다.'
body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v9 형상 범위 확대 평가</title><style>body{{font-family:system-ui;line-height:1.8;max-width:1150px;margin:36px auto;padding:20px;color:#253945}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #ccd5db;text-align:left}}th{{background:#eef4f7}}img,video{{width:100%}}h2{{margin-top:35px}}details{{margin:10px 0}}</style>
<h1>TT v9: 형상 범위를 늘리면 Dynamics 예측이 개선되는가?</h1><p>2026년 11월 1일 · v8 대비 학습 형상 추가 · 재계획 없는 MJX 검증</p>
<h2>판정</h2><p><strong>{conclusion}</strong> 정해둔 핵심 지표는 높이·기울기·이동 속도의 움직임 일치도가 각각 v8보다 0.1 이상 높아지는지다. 형상 수 확대만으로 문제가 해결됐다고 판단하지 않는다. 아래 MJX 판정은 별도 기준으로 제시한다.</p>
<h2>실험 과정</h2><p>기존 16개 형상에 새 학습 형상 32개를 더한 TT를 60,000스텝 학습했다. 배치 32, 50Hz 제어의 길이 32 window, Mode 2·4, 형상 전 층 주입(FiLM)을 유지했다. 새 시험 형상 8개는 학습에서 제외했다. 학습 완료와 별도로 실제 물리 기록을 대상으로 평가했다.</p><p>v8·v9에는 같은 물리 사례 {result['paired_cases']}쌍, 같은 seed 8개, 같은 오차 단위(v8 정규화 기준)를 사용했다. 두 형상에 적용된 행동이 동일한지도 검사했다. 같은 행동 원본에서 형상쌍을 반복 뽑았으므로 모든 쌍이 독립 표본인 것은 아니다. 주 판정은 시작 프레임 30이며, 정적인 형상 차이와 시간에 따른 움직임을 분리했다. 잘못된 E를 넣을 때의 추가 오차가 커지는지도 함께 기록했다.</p>
<h2>미학습 형상의 움직임 예측</h2><p>일치도는 실제 형상 변화와 예측 형상 변화의 상관이다. 1에 가까울수록 좋고, 0은 일치가 약함을 뜻한다. 잘못된 E 벌점은 잘못된 E의 움직임 RMSE에서 올바른 E의 RMSE를 뺀 값이다.</p><table><tr><th>항목</th><th>v8 일치도</th><th>v9 일치도</th><th>변화</th><th>v8 E 벌점</th><th>v9 E 벌점</th></tr>{rows}</table><img src="evaluation_local/agreement.png" alt="동일 미학습 형상 사례에서 v8 v9 비교">
<h2>실제 MJX Co-Design 판정</h2><p>기존 16개 형상에서 형상당 후보 4개를 실제 MJX로 실행하고, 공동 생성 후보 16개도 실행했다. 통과 기준은 예측 보상의 형상 효과 p&lt;0.05, 형상 평균 보상의 예측·실제 순위 상관&gt;0.6, 2초 무 guidance 경로 오차 중앙값&lt;15cm다. 4초 실행 결과이며 최종 10초 성공을 뜻하지 않는다. 경로 차이는 유효 실행 구간에서 계산하므로 조기 낙상과 생존 수를 함께 본다.</p><table><tr><th>모델·Task</th><th>예측 형상 효과 p</th><th>보상 순위 상관</th><th>2초 경로 차(cm)</th><th>2초 생존</th><th>Gate</th><th>4초 생존</th><th>실제 보상 중앙값</th><th>실제 진행(m)</th></tr>{taskrows}</table><img src="evaluation_local/reward_scatter.png" alt="형상 평균 예측 보상과 실제 MJX 보상">
<h2>학습 진행과 자원</h2><p>최종 검증 loss {progress[-1]['val_loss']:.5f}. 마지막 학습률은 0으로 내려가므로 말미 loss 안정만으로 충분한 수렴을 선언하지 않는다. 최종 성능 판정에는 위의 외부 평가를 사용한다. 평가 중 시스템 GPU 사용 최대 {limits['max_gpu_used_mib']/1024:.2f}GiB, RAM 여유 최소 {limits['min_ram_available_gib']:.2f}GiB였다. CPU 2코어, PyTorch GPU 할당 25%, JAX 풀 비율 40%로 제한하고 단계를 순차 실행했다.</p><img src="evaluation_local/training.png" alt="실제 학습과 검증 loss 그래프">
<h2>영상</h2><p>공동 생성한 형상과 행동을 실제 MJX에서 실행한 기록이다. 흰 점은 Task 경계, 파란 점은 TT 예측 경로이며 로봇은 실제 MJX 기록이다. 각 후보의 절대 경로 영상을 제공한다.</p>{videos}
<h2>해석 범위와 다음 판단</h2><p>이번에는 형상 수와 데이터 양이 함께 늘었고 학습 스텝은 같았다. 따라서 형상 범위만의 효과나 학습량 부족을 각각 확정할 수는 없다. 형상별 초기 상태 차이까지 완전히 제거한 인과 실험은 아니며, 새로운 모든 행동이나 10초 복합 Task의 성공을 보장하지 않는다. 판정 미달이라면 형상 수나 학습 시간만 계속 늘리기보다 E·행동 조건을 실제 상태 변화와 연결하는 학습 목표와 데이터 구성을 먼저 점검해야 한다.</p><p><a href="evaluation_local/comparison.json">종합 수치</a> · <a href="evaluation_local/given_a_held.json">미학습 형상 원자료</a> · <a href="evaluation_local/given_a_seen.json">학습 형상 원자료</a> · <a href="evaluation_local/status.json">실행 상태</a></p></html>'''
(P/'report.html').write_text(body);print(json.dumps(result,ensure_ascii=False,indent=2))
