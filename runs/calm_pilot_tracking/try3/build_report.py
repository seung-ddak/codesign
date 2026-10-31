"""Report the fixed-policy MJX friction-only comparison."""
import os,json,html
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-calm')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;base=p.parent/'try2/mjx_evaluation'
before={r['motion']:r for r in json.loads((base/'metrics.json').read_text())};after={r['motion']:r for r in json.loads((p/'metrics.json').read_text())};names=sorted(before);assert set(after)==set(before) and len(names)==5
bcond=json.loads((base/'conditions.json').read_text());acond=json.loads((p/'conditions.json').read_text())
for k in ['checkpoint','reference_hz','physics_hz','seed','solver_iterations','solver_ls_iterations','observation_noise','initialization','observation']:assert bcond[k]==acond[k],k
assert acond['sliding_friction_override']==1.0 and all(v[0]==1. for v in acond['foot_and_floor_friction'].values())
for n in names:assert before[n]['reference_sha256']==after[n]['reference_sha256']
keys=['joint_rmse_deg','body_position_rmse_m','foot_body_mean_error_m','orientation_mean_deg','slip_mean_mps']
means={label:{k:float(np.mean([rows[n][k] for n in names])) for k in keys} for label,rows in [('0.8',before),('1.0',after)]}
(p/'comparison.json').write_text(json.dumps(dict(averages=means,before=before,after=after),ensure_ascii=False,indent=2))
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');x=np.arange(5)
for ax,key,title,scale,unit in zip(axes.flat,keys,['Joint RMSE','Body position RMSE','Body-relative foot error','Mean orientation error','Loaded contact-point slip'],[1,100,100,1,1],['deg','cm','cm','deg','m/s']):
 for j,(rows,label,color) in enumerate([(before,'friction 0.8','#888'),(after,'friction 1.0','#247db5')]):ax.bar(x+(j-.5)*.35,[rows[n][key]*scale for n in names],width=.35,label=label,color=color)
 ax.set(title=title,ylabel=unit,xticks=x,xticklabels=[n[-3:] for n in names]);ax.legend(fontsize=8)
axes.flat[-1].axis('off');fig.savefig(p/'friction_comparison.png',dpi=180)
summary=''
for key,label,scale,unit in zip(keys,['관절 RMSE','몸통 위치 RMSE','몸체 기준 발 평균 오차','자세 평균 오차','접지 미끄러짐'],[1,100,100,1,1],['°','cm','cm','°','m/s']):
 b,a=means['0.8'][key]*scale,means['1.0'][key]*scale
 summary+=f'<tr><td>{label}</td><td>{b:.3f} {unit}</td><td>{a:.3f} {unit}</td><td>{(a/b-1)*100:+.1f}%</td></tr>'
rows=''
for n in names:
 for label,data in [('0.8',before[n]),('1.0',after[n])]:rows+=f'<tr><td>{n}</td><td>{label}</td><td>{data["evaluated_seconds"]:.2f}</td><td>{"완주" if data["survived_10s"] else "전도"}</td><td>{data["joint_rmse_deg"]:.2f}</td><td>{data["body_position_rmse_m"]*100:.2f}</td><td>{data["foot_body_mean_error_m"]*100:.2f}</td><td>{data["orientation_mean_deg"]:.2f}</td><td>{data["slip_mean_mps"]:.3f}</td></tr>'
videos='';bm={r['motion']:r['shared_scene_video_absolute_path'] for r in json.loads((base/'shared_scene_videos.json').read_text())['rows']};am={r['motion']:r['shared_scene_video_absolute_path'] for r in json.loads((p/'shared_scene_videos.json').read_text())['rows']}
for n in names:
 videos+=f'<h3>{n}</h3>'
 for label,paths in [('기존 마찰 0.8',bm),('변경 마찰 1.0',am)]:
  path=Path(paths[n]);assert path.is_file();videos+=f'<p>{label}</p><video controls preload="metadata" src="{html.escape(path.as_uri())}"></video>'
wins={k:sum(after[n][k]<before[n][k] for n in names) for k in keys}
text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TT 추종 try3 · A1 마찰 조건 재평가</title><style>body{{font-family:system-ui;max-width:1250px;margin:40px auto;padding:0 24px;line-height:1.85;color:#243743}}h2{{margin-top:36px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ccd6db;padding:10px;text-align:left}}th{{background:#eef3f6}}img,video{{width:100%}}</style>
<h1>TT 모션 추종 · try3 마찰 변경 재평가</h1><p>2026년 10월 29일 · 같은 PPO 체크포인트 · MJX · 미끄럼 마찰계수 0.8 → 1.0</p>
<h2>판단</h2><p><strong>마찰을 A1의 MJX 기준값에 맞추는 것만으로 현재 PPO의 전체 추종 성능은 개선되지 않았다.</strong> 평균 접지 미끄러짐은 약 4.2% 감소했지만, 몸통 위치 오차는 약 10.6% 증가했고 관절·발·자세 오차도 평균적으로 증가했다. 모두 10초 생존했으나 오차 선별은 0/5로 유지됐다. 이는 마찰을 키우는 것 자체가 항상 나쁘다는 뜻이 아니라, 기존 정책을 그대로 둔 이번 비교의 결과다.</p><h2>실험 목적과 조건</h2><p>A1의 MJX 기준 환경과 동일한 미끄럼 마찰계수를 적용하면 TT의 추종 오차와 접지 미끄러짐이 줄어드는지 확인했다. 5,246,976 스텝 학습한 기존 정책을 그대로 사용했고 재학습하지 않았다. 비교 대상은 이전 CPU 평가가 아니라 같은 MJX의 마찰 0.8 평가다.</p><p>바꾼 것은 네 발과 바닥의 미끄럼 마찰계수뿐이다. 비틀림·굴림 마찰, 다른 부품의 마찰, 형상·질량·관절 범위, KP 30·KD 0.5, 토크 한계 33.5N·m, 자기 충돌 끔·바닥 충돌 유지, 30Hz 제어·300Hz 물리 계산, 초기 자세·속도, 관측 잡음과 seed 0은 유지했다. 동일한 5개 레퍼런스를 첫 프레임부터 최대 10초 실행했다.</p>
<h2>결과</h2><p>10초 생존: 기존 {sum(r['survived_10s'] for r in before.values())}/5 → 변경 {sum(r['survived_10s'] for r in after.values())}/5. 생존과 세 오차 기준을 모두 만족한 모션: 기존 {sum(r['tracking_error_screen_pass'] for r in before.values())}/5 → 변경 {sum(r['tracking_error_screen_pass'] for r in after.values())}/5.</p><table><tr><th>모션별 지표의 평균</th><th>마찰 0.8</th><th>마찰 1.0</th><th>변화율(음수=감소)</th></tr>{summary}</table><p>위 값은 각 모션의 지표를 같은 비중으로 평균했다. 관절 오차가 줄어든 모션은 {wins['joint_rmse_deg']}/5, 위치 오차는 {wins['body_position_rmse_m']}/5, 미끄러짐은 {wins['slip_mean_mps']}/5다. 평균 변화와 모션별 변화를 함께 봐야 한다.</p><img src="friction_comparison.png" alt="마찰만 변경한 MJX의 모션별 생존·추종 오차·미끄러짐 비교">
<h2>모션별 측정값</h2><table><tr><th>모션</th><th>마찰</th><th>시간(s)</th><th>생존</th><th>관절 RMSE(°)</th><th>몸통 위치 RMSE(cm)</th><th>몸체 기준 발 오차(cm)</th><th>자세 오차(°)</th><th>접지 미끄러짐(m/s)</th></tr>{rows}</table>
<p>전도는 몸통 위쪽 축의 수직 성분 0.5 미만이다. 오차 선별은 생존·관절 RMSE 5° 미만·몸체 기준 발 평균 오차 3cm 미만·자세 평균 오차 5° 미만을 모두 요구한다. 미끄러짐은 실제 MJX 접촉 힘이 0.001N 이상인 발–바닥 접촉점의 수평 속도이며 합격 기준은 정하지 않았다.</p>
<h2>해석의 범위</h2><p>이 비교는 마찰 0.8에서 학습한 고정 정책의 마찰 민감도를 보는 실험이다. 마찰 변경만으로 동작이 개선되는지 확인할 수 있지만, 마찰 1.0에서 새로 학습했을 때의 최종 성능은 알 수 없다. 마찰을 키워도 제어 행동과 접촉 타이밍이 함께 바뀌므로 모든 모션에서 오차나 미끄러짐이 반드시 줄지는 않는다. 한 seed·학습에 사용한 5개 모션의 결과이며 실제 하드웨어나 다른 형상으로 일반화하지 않는다.</p>
<h2>실제 기록 비교 영상</h2><p>주황 A1 전체 외형, 보라 TT 목표, 파랑 해당 마찰 조건의 MJX 실제 기록을 공통 바닥에서 재생한다. 가려짐 방지를 위한 일정한 표시 간격만 두었다. 모든 영상 주소는 file:/// 절대 경로다.</p>{videos}
<p><a href="comparison.json">전체 비교 수치</a> · <a href="metrics.json">마찰 1.0 측정값</a> · <a href="conditions.json">실험 조건</a> · <a href="../try2/report_try2.html">이전 실험 보고서</a></p></html>'''
(p/'report_try3.html').write_text(text)
print(json.dumps(dict(averages=means,improved_motion_counts=wins),ensure_ascii=False))
