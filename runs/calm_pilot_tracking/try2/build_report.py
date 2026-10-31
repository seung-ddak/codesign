"""Compare fixed-budget PPO with the recorded PD baseline; never runs simulation."""
import json,os,re,sys,html
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-calm')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
p=Path(__file__).resolve().parent;root=p.parents[2];sys.path.insert(0,str(root/'code/motion_tracking'))
from eval_calm_pd_tracking import tracking_errors
baseline=p.parent/'try1';evaluation=p/'evaluation'
old={r['motion']:r for r in json.loads((baseline/'metrics.json').read_text())};rows=json.loads((evaluation/'metrics.json').read_text())
assert len(rows)==5 and {r['motion'] for r in rows}==set(old)
conditions=json.loads((evaluation/'conditions.json').read_text());checkpoint=Path(conditions['checkpoint']);actual_steps=int(checkpoint.name)
assert actual_steps>=5242880
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
comparison=[]
for ax,r in zip(axes.flat,rows):
 name=r['motion'];b=old[name];d=np.load(evaluation/(name+'.npz'));pd=np.load(baseline/(name+'.npz'));n=len(d['time'])
 assert r['reference_sha256']==b['reference_sha256'] and np.isclose(d['time'][-1],n/30)
 q=d['qpos'];ref=d['reference_qpos'];a=Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix();c=Rotation.from_quat(ref[:,[4,5,6,3]]).as_matrix()
 feet=np.einsum('tfi,tij->tfj',d['foot_pos']-q[:,None,:3],a);ref_feet=np.einsum('tfi,tij->tfj',d['reference_foot_pos']-ref[:,None,:3],c)
 r['foot_body_mean_error_m']=float(np.linalg.norm(feet-ref_feet,axis=-1).mean())
 r['tracking_error_screen_pass']=bool(r['joint_rmse_deg']<5 and r['foot_body_mean_error_m']<.03 and r['orientation_mean_deg']<5)
 matched=tracking_errors(pd['qpos'][:n],pd['reference_qpos'][:n],pd['foot_pos'][:n],pd['reference_foot_pos'][:n])
 slip=pd['slip_mps'][:n];slip=slip[np.isfinite(slip)];matched['slip_mean_mps']=float(slip.mean()) if len(slip) else None
 comparison.append(dict(motion=name,pd_full=b,ppo=r,pd_matched_duration=matched,matched_seconds=n/30))
 ax.plot(ref[:,0],ref[:,1],label='Reference',c='#d28b2d');ax.plot(pd['qpos'][:,0],pd['qpos'][:,1],label='try1 PD',c='#777');ax.plot(q[:,0],q[:,1],label='try2 PPO',c='#247db5');ax.set(title=name,xlabel='World X (m)',ylabel='World Y (m)',aspect='equal');ax.legend(fontsize=8)
axes.flat[-1].axis('off');fig.savefig(p/'trajectories.png',dpi=170)
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');x=np.arange(5)
for ax,key,title,unit in zip(axes,['joint_rmse_deg','body_position_rmse_m','slip_mean_mps'],['Joint tracking error','Body position error','Actual supported-foot slip'],['deg','m','m/s']):
 for j,(which,label,color) in enumerate([('pd_matched_duration','PD (matched time)','#888'),('ppo','PPO','#247db5')]):
  ax.bar(x+(j-.5)*.35,[v[which][key] if v[which][key] is not None else np.nan for v in comparison],width=.35,label=label,color=color)
 ax.set(xticks=x,xticklabels=[r['motion'][-3:] for r in rows],title=title,ylabel=unit);ax.legend(fontsize=8)
fig.savefig(p/'comparison.png',dpi=170)
(p/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2,allow_nan=False));(evaluation/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False))
num=lambda v:f'{v:.3f}' if v is not None else '접촉 없음'
table=''
for v in comparison:
 b,r=v['pd_full'],v['ppo'];name=r['motion']
 for label,a in [('try1 PD',b),('try2 PPO',r)]:
  table+=f'<tr><td>{name}</td><td>{label}</td><td>{a["evaluated_seconds"]:.2f}</td><td>{"완주" if a["survived_10s"] else "전도"}</td><td>{a["joint_rmse_deg"]:.2f}</td><td>{a["body_position_rmse_m"]:.3f}</td><td>{a["foot_body_mean_error_m"]*100:.2f}</td><td>{a["orientation_mean_deg"]:.2f}</td><td>{num(a["slip_mean_mps"])}</td></tr>'
pd_videos={r['motion']:r['shared_scene_video_absolute_path'] for r in json.loads((baseline/'shared_scene_videos.json').read_text())['rows']}
ppo_videos={r['motion']:r['shared_scene_video_absolute_path'] for r in json.loads((evaluation/'shared_scene_videos.json').read_text())['rows']}
videos=''
for r in rows:
 name=r['motion'];videos+=f'<h3>{name}</h3><div class="videos">'
 for label,path in [('try1 PD',Path(pd_videos[name])),('try2 PPO',Path(ppo_videos[name]))]:
  assert path.is_file();videos+=f'<div><p>{label}</p><video controls preload="metadata" src="{html.escape(path.as_uri())}"></video><small>{html.escape(str(path))}</small></div>'
 videos+='</div>'
mid=json.loads((p/'midpoint_evaluation/metrics.json').read_text()) if (p/'midpoint_evaluation/metrics.json').exists() else []
mid_summary=''
if len(mid)==5:
 mid_summary=f'<h2>중간 물리 검증</h2><p>2,623,488 스텝 체크포인트에서도 동일한 5개 모션을 평가했다. 생존 {sum(r["survived_10s"] for r in mid)}/5, 몸통 위치 RMSE {min(r["body_position_rmse_m"] for r in mid)*100:.2f}~{max(r["body_position_rmse_m"] for r in mid)*100:.2f}cm, 관절 RMSE {min(r["joint_rmse_deg"] for r in mid):.2f}~{max(r["joint_rmse_deg"] for r in mid):.2f}°였다. 이 결과로 최종 체크포인트를 바꾸지 않았으며 아래 비교는 예정한 전체 예산 완료 모델을 사용한다.</p>'

history=[]
blocks=re.split(r'-{5,}(\d+)-{5,}',(p/'train.log').read_text())
for i in range(1,len(blocks),2):
 metrics=dict(re.findall(r'^([A-Za-z0-9_/]+): ([0-9.eE+\-]+)$',blocks[i+1],re.M))
 history.append(dict(step=int(blocks[i]),reward=float(metrics['eval/episode/reward/mean']),mean_episode_steps=float(metrics['eval/avg_episode_length'])))
(p/'training_history.json').write_text(json.dumps(history,indent=2))
history_table=''.join(f'<tr><td>{r["step"]:,}</td><td>{r["reward"]:.2f}</td><td>{r["mean_episode_steps"]:.1f}</td></tr>' for r in history)

survived=sum(r['survived_10s'] for r in rows);passed=sum(r['survived_10s'] and r['tracking_error_screen_pass'] for r in rows)
wins={key:sum(v['ppo'][key] is not None and v['pd_matched_duration'][key] is not None and v['ppo'][key]<v['pd_matched_duration'][key] for v in comparison) for key in ['joint_rmse_deg','body_position_rmse_m','slip_mean_mps']}
aggregate=''
for key,label,scale,unit in [('joint_rmse_deg','관절 RMSE',1,'°'),('body_position_rmse_m','몸통 위치 RMSE',100,'cm'),('foot_body_mean_error_m','몸체 기준 발 평균 오차',100,'cm'),('orientation_mean_deg','자세 평균 오차',1,'°'),('slip_mean_mps','실제 접지 미끄러짐',1,'m/s')]:
 before=np.mean([v['pd_full'][key] for v in comparison])*scale;after=np.mean([v['ppo'][key] for v in comparison])*scale
 aggregate+=f'<tr><td>{label}</td><td>{before:.3f} {unit}</td><td>{after:.3f} {unit}</td><td>{(1-after/before)*100:.1f}%</td></tr>'
aggregate='<h2>전체 경향</h2><p>아래 값은 모션별 지표를 동일 가중치로 평균한 값이다. 두 제어기 모두 모든 모션을 10초 수행했으므로 전체 평가 시간도 같다. 한 seed의 관찰 결과이며 통계적 유의성을 뜻하지 않는다.</p><table><tr><th>지표</th><th>try1 PD</th><th>try2 PPO</th><th>평균값 감소율</th></tr>'+aggregate+'</table><p><strong>해석:</strong> PPO는 목표 경로에서 크게 벗어나던 PD 기준 실험보다 몸통 경로와 방향을 잘 유지했다. 그러나 모든 모션의 관절 RMSE가 5°를 넘고 몸체 기준 발 평균 오차도 3cm를 넘어, 세부 움직임의 정확도는 아직 기준에 미달한다. 미끄러짐은 평균적으로 감소했지만 004에서는 PD보다 증가했다. 경로 개선이 미끄러짐 없는 동작까지 보장하지는 않는다.</p><p>중간 모델에서 최종 모델로 갈 때 모든 모션이 일관되게 개선되지 않았다. 예를 들어 001의 몸통 위치 RMSE는 중간 약 7.93cm에서 최종 약 15.24cm로 증가했다. 따라서 추가 학습량만으로 정확도가 계속 좋아진다고 단정할 수 없다.</p>'

mjx_audit=(p/'mjx_report_fragment.html').read_text() if (p/'mjx_report_fragment.html').exists() else ''
body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TT PPO 추종 try2 · PD 비교</title><style>body{{font-family:system-ui;max-width:1300px;margin:40px auto;padding:0 24px;line-height:1.85;color:#243743}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d3dce0;text-align:left}}th{{background:#eef3f6}}img{{width:100%}}.videos{{display:grid;grid-template-columns:1fr;gap:18px}}video{{width:100%}}small{{overflow-wrap:anywhere}}h2{{margin-top:36px}}@media(max-width:800px){{.videos{{grid-template-columns:1fr}}}}</style>
<h1>TT PPO 추종 실험 · try2</h1><p>2026년 10월 29일 · 30Hz·발 고정 보정 없음 · try1 PD와 비교</p>
{mjx_audit}<h2>기존 CPU 평가 결론</h2><p><strong>PPO의 10초 생존은 {survived}/5, 생존과 세 가지 오차 선별을 함께 만족한 모션은 {passed}/5다.</strong> try1 PD는 생존 5/5, 같은 오차 선별 통과 0/5였다. 같은 실행 길이로 비교하면 PPO의 관절 오차가 낮아진 모션은 {wins['joint_rmse_deg']}/5, 몸통 위치 오차는 {wins['body_position_rmse_m']}/5, 접지 미끄러짐은 {wins['slip_mean_mps']}/5다. 모션별 성공 여부와 생존·정확도·미끄러짐을 함께 보아야 한다.</p>
{aggregate}<h2>목적과 방법</h2><p>계산 단계의 레퍼런스 선별 통과가 실제 추종을 보장하지 않았으므로, 현재 상태와 앞으로의 목표 움직임을 관측하는 PPO 정책으로 추종 성능이 개선되는지 확인했다. try1은 목표 관절각을 그대로 PD 제어기에 넣었고, try2는 PPO가 관절 목표를 선택하며 같은 PD 구동기가 힘을 낸다. 모션마다 정책을 따로 학습하지 않고 5개를 공유하는 정책 하나를 처음부터 학습했다.</p>
<p>학습 요청 예산 5,242,880 환경 스텝, 실제 저장 체크포인트 <strong>{actual_steps:,} 스텝</strong>, seed 0. 병렬 환경 32개, 기준 클립을 임의 시점에서 시작하는 학습 초기화, 외력 교란 없음, 초기 관절 자세·속도 추가 잡음 없음. 관측에는 기존 환경의 작은 잡음을 유지했다. 네트워크와 보상은 기존 추종 PPO 설정을 재사용했다. 마지막 예산 체크포인트를 고정 사용했으며 비교 평가로 체크포인트를 선별하지 않았다.</p>
<p>학습은 MJX GPU, 최종 비교 평가는 try1과 같은 MuJoCo CPU에서 결정론적 정책으로 수행했다. 모델과 레퍼런스는 TT e000/00000011, 치수 중간값 13개, 동일한 30Hz 데이터다. 두 평가 모두 300Hz 물리 계산·30Hz 제어, 자기 충돌 끔·바닥 충돌 유지, KP 30·KD 0.5, 관절 토크 한계 33.5N·m를 사용했다. 첫 레퍼런스 자세와 같은 차분 초기 속도에서 시작하고, 외력·높이 보정·중간 초기화 없이 최대 10초 실행했다. 다만 MJX에서 학습한 정책을 CPU 물리 환경에서 평가하므로 학습 환경 내 평가와 구분해야 한다.</p>
<h2>학습 중 내부 평가</h2><table><tr><th>학습 스텝</th><th>평균 보상</th><th>평균 평가 구간(스텝)</th></tr>{history_table}</table><p>내부 평가는 임의 시작 상태에서 수행하며 평균 평가 길이는 전도뿐 아니라 클립 끝에서도 짧아질 수 있다. 아래 첫 프레임부터의 CPU 비교 평가와 구분한다. 보상만으로 성공을 판정하지 않는다.</p><h2>판정과 지표</h2><p>몸통 위쪽 축의 수직 성분이 0.5 미만이면 전도이며 30Hz 경계에서 검사했다. 관절 전체 RMSE 5° 미만, 몸체 기준 발 평균 오차 3cm 미만, 자세 평균 오차 5° 미만을 참고 오차 선별로 사용했다. 기존 전체 평가와 동일한 판정은 아니며, 미끄러짐은 합격 기준 없이 실제 수치를 보고한다. 접지는 레퍼런스 표시가 아니라 0.001N 이상 힘을 받는 실제 발–바닥 접촉으로 측정했고, 접촉점의 수평 속도를 미끄러짐으로 계산했다. 몸체 기준 발 오차는 몸통의 이동·방향 이탈을 제거하고 몸통에 대한 네 발의 상대 위치를 비교한 값이다.</p>
{mid_summary}<h2>모션별 결과</h2><p>각 행은 전도 또는 10초 종료까지의 평가다. 일찍 전도한 결과의 작은 오차가 완주보다 우수하다는 뜻은 아니다.</p><table><tr><th>모션</th><th>제어</th><th>평가(s)</th><th>생존</th><th>관절 RMSE(°)</th><th>몸통 위치 RMSE(m)</th><th>몸체 기준 발 오차(cm)</th><th>자세 오차(°)</th><th>접지 미끄러짐(m/s)</th></tr>{table}</table>
<h2>같은 시간 구간의 비교</h2><p>PPO가 전도한 경우 PD 데이터도 같은 시점까지만 잘라 비교했다. 아래 그래프와 개선 모션 수는 이 공통 구간을 사용한다. 전체 수치는 각자의 실행 길이로 집계한 위 표와 구분한다.</p><img src="comparison.png" alt="같은 평가 시간 구간에서 PD와 PPO의 관절·몸통 위치·접지 미끄러짐 비교"><p>궤적 그래프에서 주황은 목표, 회색은 try1 PD, 파랑은 try2 PPO다.</p><img src="trajectories.png" alt="목표 궤적과 PD·PPO 실제 궤적 비교">
<h2>실제 실행 영상</h2><p>하나의 바닥과 공통 카메라 안에 주황 A1 원본, 보라 TT 목표, 파랑 TT 실제 기록이 함께 있다. A1은 원본 CALM URDF의 전체 외형 메시(몸통·힙·허벅지·종아리)를 사용하며 충돌 형상은 숨겼다. 세 모델이 겹쳐 가려지지 않도록 인접 모델 사이에 약 1.15m의 일정한 표시용 간격을 두었다. 이 간격 외에 각 모델의 원래 동작·방향·이동 궤적은 변경하지 않았으며 카메라는 세 몸통을 함께 따라간다. 실제 추종 오차는 표시 간격을 적용하기 전 기록으로 계산한 값이다. 이 영상은 저장된 상태를 한 장면에 재생한 것으로 세 로봇이 서로 충돌하는 새 물리 실험은 아니다. 학습·물리 실행·측정값은 변경하지 않았으며 모든 영상 주소는 file:/// 절대 경로다.</p>{videos}
<h2>해석의 범위와 다음 단계</h2><p>이 결과는 한 seed·고정 학습 예산과 학습에 사용한 5개 모션에 대한 평가다. 미학습 모션·다른 형상·초기 상태 변화에 대한 일반화 증거가 아니다. 성능 미달은 현재 예산과 설정에서 기준을 만족하지 못했다는 뜻이며 PPO로 학습 불가능하다는 뜻은 아니다. 학습 보상 증가를 실제 경로 추종 성공으로 대체하지 않는다. 다음 실험에서는 부족한 지표와 모션을 먼저 확인하고 학습량·보상·초기화·물리 엔진 간 차이를 한 번에 하나씩 비교해야 한다.</p>
<p><a href="comparison.json">전체 비교 수치</a> · <a href="evaluation/metrics.json">PPO 평가 수치</a> · <a href="evaluation/conditions.json">실행 조건</a> · <a href="../try1/report_try1.html">try1 보고서</a></p></html>'''
(p/'report_try2.html').write_text(body)
print(json.dumps(dict(survived=survived,tracking_screen=passed,matched_time_improvements=wins,actual_steps=actual_steps),ensure_ascii=False))
