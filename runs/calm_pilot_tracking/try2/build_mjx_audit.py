"""Compare native MJX checkpoints and report recorded convergence evidence."""
import json,os,html
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-calm')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
folders=['evaluation','mjx_midpoint_evaluation','mjx_evaluation'];groups={f:{r['motion']:r for r in json.loads((p/f/'metrics.json').read_text())} for f in folders}
names=sorted(groups['evaluation']);assert all(set(g)==set(names) for g in groups.values())
for name in names:assert len({groups[f][name]['reference_sha256'] for f in folders})==1
keys=['joint_rmse_deg','body_position_rmse_m','foot_body_mean_error_m','orientation_mean_deg','slip_mean_mps']
means={f:{k:float(np.mean([g[k] for g in groups[f].values()])) for k in keys} for f in folders}
(p/'mjx_comparison.json').write_text(json.dumps(dict(averages=means,by_motion=groups),ensure_ascii=False,indent=2))
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');x=np.arange(5)
for ax,key,title,scale,unit in zip(axes,['joint_rmse_deg','body_position_rmse_m','slip_mean_mps'],['Final checkpoint: joint RMSE','Final checkpoint: body RMSE','Final checkpoint: supported-foot slip'],[1,100,1],['deg','cm','m/s']):
 for j,(folder,label,color) in enumerate([('evaluation','CPU','#888'),('mjx_evaluation','MJX','#247db5')]):ax.bar(x+(j-.5)*.35,[groups[folder][n][key]*scale for n in names],width=.35,label=label,color=color)
 ax.set(title=title,ylabel=unit,xticks=x,xticklabels=[n[-3:] for n in names]);ax.legend()
fig.savefig(p/'mjx_cpu_comparison.png',dpi=180)
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,key,title,scale,unit in zip(axes,['joint_rmse_deg','body_position_rmse_m','slip_mean_mps'],['Same-start MJX: joint RMSE','Same-start MJX: body RMSE','Same-start MJX: supported-foot slip'],[1,100,1],['deg','cm','m/s']):
 for n in names:ax.plot([2.623488,5.246976],[groups[f][n][key]*scale for f in ['mjx_midpoint_evaluation','mjx_evaluation']],'o--',alpha=.65,label=n[-3:])
 ax.plot([2.623488,5.246976],[means[f][key]*scale for f in ['mjx_midpoint_evaluation','mjx_evaluation']],'ko-',lw=2,label='mean')
 ax.set(title=title,xlabel='Training steps (millions)',ylabel=unit);ax.margins(x=.15);ax.legend(fontsize=8)
fig.savefig(p/'mjx_checkpoint_trends.png',dpi=180)
summary=''
for f,label in [('evaluation','최종 CPU'),('mjx_midpoint_evaluation','중간 MJX (262만)'),('mjx_evaluation','최종 MJX (525만)')]:
 g=list(groups[f].values());m=means[f]
 summary+=f'<tr><td>{label}</td><td>{sum(r["survived_10s"] for r in g)}/5</td><td>{m["joint_rmse_deg"]:.2f}</td><td>{m["body_position_rmse_m"]*100:.2f}</td><td>{m["foot_body_mean_error_m"]*100:.2f}</td><td>{m["orientation_mean_deg"]:.2f}</td><td>{m["slip_mean_mps"]:.3f}</td><td>{sum(r["tracking_error_screen_pass"] for r in g)}/5</td></tr>'
cond=json.loads((p/'mjx_evaluation/conditions.json').read_text())
video_meta=p/'mjx_evaluation/shared_scene_videos.json';videos=''
if video_meta.exists():
 for r in json.loads(video_meta.read_text())['rows']:
  path=Path(r['shared_scene_video_absolute_path']);assert path.exists()
  videos+=f'<h3>{r["motion"]} · 최종 MJX 실제 기록</h3><video controls preload="metadata" src="{html.escape(path.as_uri())}"></video>'
text=f'''<section id="mjx-audit"><h2>동일 MJX 환경 재평가 및 학습 수렴 점검</h2>
<p><strong>학습했던 MJX에서도 생존 5/5, 오차 선별 0/5다. CPU로 옮겨 평가한 것만으로 현재 추종 오차가 생긴 것은 아니다.</strong> 중간보다 최종 모델의 평균 관절·위치 오차가 증가했고, 미끄러짐은 감소했다. 현재 자료로 수렴 여부를 확정할 수 없으며, 단순히 학습량만 늘리면 좋아진다는 근거도 부족하다.</p>
<h3>비교 조건</h3><p>기존 중간·최종 체크포인트를 재학습 없이 평가했다. 저장된 MJX 환경과 모델·보상 설정을 불러와 30Hz 제어, 300Hz 물리 계산, solver 반복 {cond['solver_iterations']}회, 동일한 5개 모션·seed 0을 사용했다. CPU 평가와 같은 첫 레퍼런스 자세·차분 초기 속도·행동 이력 및 관측 잡음 키 순서를 맞췄다. 학습 때의 임의 프레임 시작과 달리, 모든 모션을 첫 프레임부터 10초 평가했다. 자동 초기화는 없으며 전도 시 지표를 잘랐다.</p><p>접지 미끄러짐은 MJX가 계산한 실제 접촉 힘과 접촉점 속도로 측정했다. 측정을 위해 실제 실행 중인 로봇의 자세를 강제로 보정하지 않았다. 기존 CPU 기본 solver와 달리 MJX의 학습 설정을 유지했으므로, CPU–MJX 차이는 물리 엔진과 설정 차이를 함께 포함한다.</p>
<table><tr><th>평가</th><th>10초 생존</th><th>관절 RMSE(°)</th><th>몸통 위치 RMSE(cm)</th><th>몸체 기준 발 오차(cm)</th><th>자세 오차(°)</th><th>접지 미끄러짐(m/s)</th><th>오차 선별</th></tr>{summary}</table><p>수치는 5개 모션별 지표의 단순 평균이다. CPU와 MJX의 오차 수준은 비슷하지만 접촉·미끄러짐 수치에는 차이가 있다. MJX에서 평가해도 관절 5°·발 3cm·자세 5°의 세 조건을 함께 만족한 모션은 없다.</p>
<img src="mjx_cpu_comparison.png" alt="같은 최종 체크포인트의 CPU와 MJX 모션별 지표 비교">
<h3>학습이 수렴했는가?</h3><p><strong>현재는 수렴 판정에 필요한 기록이 부족하다.</strong> loss는 개별 업데이트의 연속 기록이 아니라 약 262만 스텝씩 묶은 구간 평균 두 개뿐이고, 내부 평가는 초기·중간·최종의 세 점이다. 점 사이의 점선은 위치를 연결한 안내선이며 실제 중간 변동을 측정한 곡선이 아니다. 측정되지 않은 구간을 보간하거나 부드럽게 만들지 않았다.</p>
<img src="learning_curves.png" alt="두 구간 평균 loss와 세 내부 평가점, 기록 부족을 명시한 PPO 학습 그래프">
<ul><li>전체 loss: −0.1091 → −0.0576. 감소한 것이 아니다.</li><li>Value loss: 0.0883 → 0.1162. 가치 예측 오차 항이 증가했다.</li><li>Policy loss: −0.06083 → −0.06112. 구간 평균은 비슷하지만 두 점만으로 안정화 또는 개선을 판정할 수 없다.</li><li>내부 평가 보상: 7.26 → 19.72 → 19.07. 초반 개선 후 마지막 구간에서 소폭 낮아졌다.</li><li>내부 평균 위치 오차: 73.54 → 5.91 → 7.41cm. 후반에도 계속 줄었다는 증거는 아니다.</li></ul>
<p>PPO의 전체 loss는 정책·가치 예측·탐색을 위한 엔트로피 항의 합이다. 정책이 바뀌면 학습에 쓰는 경험과 목표값도 바뀌므로, 전체 loss의 부호나 단조 감소만으로 제어 성능의 수렴을 판단하지 않는다. 보상·실제 추종 오차·생존·미끄러짐을 함께 보아야 한다.</p>
<h3>같은 시작 상태에서 본 중간 → 최종 변화</h3><img src="mjx_checkpoint_trends.png" alt="동일 시작 상태의 MJX에서 중간과 최종 모델의 실제 추종 오차 변화"><p>학습량이 약 두 배가 되는 동안 평균 관절 오차는 6.94→7.49°, 위치 오차는 7.34→8.07cm로 증가했고, 미끄러짐은 0.160→0.141m/s로 감소했다. 일부 지표의 개선과 다른 지표의 악화가 함께 있어, 모든 목표가 계속 개선 중이라고 할 수 없다. 한 seed와 두 체크포인트이므로 최적점을 지났는지, 학습 변동인지, 보상의 절충인지도 확정하지 않는다.</p>
<h3>추가 학습의 가치에 대한 판단</h3><p><strong>추가 학습을 시험할 가치는 있지만, 현재 설정 그대로 오래 연장하는 것보다 짧은 검증 구간에서 변화 과정을 더 자주 기록하는 것이 먼저다.</strong> 다음 연장 시험에서는 설정을 유지한 채 약 10만~25만 스텝 간격의 loss 구간 평균과 고정된 5개 전체 모션 MJX 평가를 기록한다. 같은 구간에서 실제 오차와 보상이 함께 개선되는지 확인하고, 보상 상승에도 추종 오차가 줄지 않으면 보상·레퍼런스의 절충을 별도 실험으로 검토한다. 이번에는 추가 학습을 실행하지 않았으며 기존 체크포인트의 평가와 기록 분석만 수행했다.</p>
<h3>최종 MJX 실행 영상</h3><p>주황 A1 전체 외형, 보라 TT 목표, 파랑 MJX 실제 기록을 공통 장면에서 보여준다. 가려짐 방지용 일정한 표시 간격만 적용했으며 영상 주소는 절대 경로다.</p>{videos}
<p><a href="mjx_comparison.json">MJX·CPU 및 체크포인트 비교 수치</a> · <a href="convergence_data.json">기록된 학습 수치</a> · <a href="mjx_evaluation/conditions.json">MJX 평가 조건</a></p></section>'''
(p/'mjx_report_fragment.html').write_text(text)
print('MJX audit fragment and plots saved.')
