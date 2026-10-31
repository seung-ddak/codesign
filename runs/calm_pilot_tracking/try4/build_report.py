"""Compare unseen-morphology tracking using one frozen e000 checkpoint."""
import os,json,html
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-calm')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;base=p.parent/'try2/mjx_evaluation'
defs=json.loads((p/'designs.json').read_text())
labels={'near_short':'허벅지 연장·종아리 단축','near_long':'허벅지 단축·종아리 연장','near_body':'몸통·무게중심 변화','e003':'분산 형상 e003','e008':'분산 형상 e008'}
for d in defs:d['label']=labels[d['id']]
base_rows=json.loads((base/'metrics.json').read_text());base_cond=json.loads((base/'conditions.json').read_text());groups=[dict(id='e000',label='학습 형상 e000',E=[.5]*13,rows=base_rows,path=base)]
for definition in defs:
 folder=p/'evaluations'/definition['id'];rows=json.loads((folder/'metrics.json').read_text());cond=json.loads((folder/'conditions.json').read_text())
 assert len(rows)==5 and np.allclose(cond['E'],definition['E'])
 for key in ['checkpoint','reference_hz','physics_hz','solver_iterations','solver_ls_iterations','seed','observation_noise']:assert cond[key]==base_cond[key],key
 diag=json.loads((p/'references'/definition['id']/'diagnostics.json').read_text())
 groups.append(dict(**definition,rows=rows,path=folder,mass_kg=cond['model_mass_kg'],diagnostics=diag))
keys=['joint_rmse_deg','body_position_rmse_m','foot_body_mean_error_m','orientation_mean_deg','slip_mean_mps']
summary=[]
for group in groups:
 r=group['rows'];summary.append(dict(design=group['id'],label=group['label'],E=group['E'],survived=sum(x['survived_10s'] for x in r),tracking_pass=sum(x['tracking_error_screen_pass'] for x in r),mean_seconds=float(np.mean([x['evaluated_seconds'] for x in r])),**{k:float(np.mean([x[k] for x in r if x[k] is not None])) for k in keys}))
(p/'comparison.json').write_text(json.dumps(dict(summary=summary,groups=[{k:str(v) if isinstance(v,Path) else v for k,v in g.items()} for g in groups]),ensure_ascii=False,indent=2,allow_nan=False))
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');x=np.arange(6);ticks=[g['id'] for g in groups]
for ax,key,title,scale,unit in zip(axes.flat,['mean_seconds',*keys],['Survival/evaluation time','Joint RMSE','Body position RMSE','Body-relative foot error','Orientation error','Loaded contact-point slip'],[1,1,100,100,1,1],['s','deg','cm','cm','deg','m/s']):
 values=[s[key]*scale for s in summary];ax.bar(x,values,color=['#666']+['#247db5']*5);ax.set(title=title,ylabel=unit,xticks=x,xticklabels=ticks);ax.tick_params(axis='x',labelrotation=25)
 for i,v in enumerate(values):ax.annotate(f'{v:.3g}',(i,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
 ax.margins(y=.15)
fig.savefig(p/'morphology_comparison.png',dpi=180)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for ax,group in zip(axes.flat,groups):
 for row in group['rows']:
  with np.load(group['path']/(row['motion']+'.npz')) as d:ax.plot(d['time'],np.linalg.norm(d['qpos'][:,:3]-d['reference_qpos'][:,:3],axis=1),label=row['motion'][-3:])
 ax.set(title=group['id'],xlabel='Time (s)',ylabel='Body position error (m)');ax.legend(fontsize=8)
fig.savefig(p/'position_error_traces.png',dpi=180)
rows=''
for s in summary:rows+=f'<tr><td>{s["label"]}</td><td>{s["survived"]}/5</td><td>{s["tracking_pass"]}/5</td><td>{s["mean_seconds"]:.2f}</td><td>{s["joint_rmse_deg"]:.2f}</td><td>{s["body_position_rmse_m"]*100:.2f}</td><td>{s["foot_body_mean_error_m"]*100:.2f}</td><td>{s["orientation_mean_deg"]:.2f}</td><td>{s["slip_mean_mps"]:.3f}</td></tr>'
detail='';videos='';diagnostics=''
for group in groups:
 detail+=f'<h3>{group["label"]}</h3><p>E = '+', '.join(f'{v:.3f}' for v in group['E'])+'</p><table><tr><th>모션</th><th>시간(s)</th><th>전도</th><th>관절 RMSE(°)</th><th>몸통 위치 RMSE(cm)</th><th>몸체 기준 발 오차(cm)</th><th>미끄러짐(m/s)</th></tr>'
 for r in group['rows']:detail+=f'<tr><td>{r["motion"]}</td><td>{r["evaluated_seconds"]:.2f}</td><td>{"발생" if r["fell"] else "없음"}</td><td>{r["joint_rmse_deg"]:.2f}</td><td>{r["body_position_rmse_m"]*100:.2f}</td><td>{r["foot_body_mean_error_m"]*100:.2f}</td><td>{r["slip_mean_mps"]:.3f}</td></tr>'
 detail+='</table>'
 if group['id']=='e000':continue
 dg=group['diagnostics'];diagnostics+=f'<tr><td>{group["label"]}</td><td>{group["mass_kg"]:.3f}</td><td>{max(r["foot_target_error_m"] for r in dg)*1000:.3f}</td><td>{max(r["inverse_bad_fraction"] for r in dg)*100:.2f}%</td><td>{max(r["initial_ground_penetration_m"] for r in dg)*1000:.2f}</td></tr>'
 meta=json.loads((group['path']/'shared_scene_videos.json').read_text())
 videos+=f'<h3>{group["label"]}</h3>'
 for r in meta['rows']:
  path=Path(r['shared_scene_video_absolute_path']);assert path.is_file();videos+=f'<details><summary>{r["motion"]} 영상</summary><video controls preload="none" src="{html.escape(path.as_uri())}"></video></details>'
new_survived=sum(s['survived'] for s in summary[1:]);new_pass=sum(s['tracking_pass'] for s in summary[1:])
body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>기존 e000 PPO의 다른 TT 형상 추종 검증 · try4</title><style>body{{font-family:system-ui;max-width:1300px;margin:40px auto;padding:0 24px;line-height:1.85;color:#243743}}h2{{margin-top:36px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ccd6db;text-align:left}}th{{background:#eef3f6}}img,video{{width:100%}}details{{margin:12px 0}}summary{{cursor:pointer}}</style>
<h1>기존 e000 PPO를 다른 TT 형상에 적용 · try4</h1><p>2026년 10월 29일 · 재학습 없음 · 기존 모션 5개 × 새로운 형상 5개</p>
<h2>실험 목적과 판정</h2><p><strong>다른 형상에서의 10초 생존은 {new_survived}/25, 생존과 추종 오차 선별을 함께 만족한 실행은 {new_pass}/25다.</strong> 학습 형상 e000의 기존 MJX 평가는 생존 5/5, 오차 선별 0/5였다. 따라서 새 형상의 선별 미달을 모두 형상 전이만의 문제로 해석하면 안 되며, e000에서 이미 남은 오차와 추가 악화를 구분해야 한다.</p>
<p><strong>결론: 가까운 형상 변형에서는 기존 정책이 생존을 유지했지만, 넓은 형상 범위에 그대로 적용하기에는 부족하다.</strong> e003은 몸통 위치 오차가 평균 85.2cm로 커졌고 e008은 평균 0.64초에 전도됐다. 현재 정책을 128개 형상 전체의 안정적인 수집기로 가정해서는 안 된다. 재학습이 필요하다면 형상마다 별개 정책을 만드는 대신 여러 형상에서 공유하는 정책의 학습을 우선 검토할 수 있다.</p><h2>동일하게 유지한 조건</h2><p>5,246,976 스텝 e000 PPO 체크포인트 하나를 고정했다. topology 00000011·무릎 방향·팔 제거는 유지하고 E만 바꿨다. 각 E에서 같은 A1 원본 5개를 다시 리타겟팅해 목표를 만들었다. 30Hz·발 고정 보정 없음, MJX GPU 300Hz, KP 30·KD 0.5, 토크 ±33.5N·m, 마찰 0.8, 자기 충돌 끔·바닥 충돌 유지, seed 0과 기존 관측 잡음을 유지했다.</p><p>새 형상에 e000의 초기 자세를 그대로 복사하지 않았다. 각 형상의 목표 첫 자세·차분 속도에서 시작했으며, 10초 종료 또는 몸통 위쪽 축의 수직 성분 0.5 미만 전도까지 실행했다. 중간 초기화·추가 PPO 학습·형상별 정책 튜닝은 없다. 형상을 하나씩 처리해 GPU 메모리 누적을 제한했다.</p>
<h2>형상별 결과</h2><p>숫자는 각 형상에서 모션 5개의 평균이다. 조기 전도가 있는 행은 평가 시간이 짧으므로 작은 오차를 더 좋은 추종으로 해석하지 않는다.</p><table><tr><th>형상</th><th>10초 생존</th><th>오차 선별</th><th>평균 평가(s)</th><th>관절 RMSE(°)</th><th>몸통 위치 RMSE(cm)</th><th>몸체 기준 발 오차(cm)</th><th>자세 오차(°)</th><th>접지 미끄러짐(m/s)</th></tr>{rows}</table><img src="morphology_comparison.png" alt="e000과 새로운 5개 형상의 생존·추종 오차·미끄러짐 비교"><img src="position_error_traces.png" alt="각 형상 모션별 시간에 따른 위치 오차">
<h2>목표와 초기 상태의 품질</h2><p>다리 비율 두 변형은 E의 3·4·9·10 좌표를 각각 0.4 또는 0.6으로 변경했다. 이 좌표의 물리 길이 변화 방향은 동일하지 않아, 총 다리 길이만 늘이거나 줄인 실험이 아니다. 몸통 변형은 E의 1·2를 0.6으로 변경했다. e003·e008은 기존 형상 목록의 분산 샘플이다. 모두 이번 e000 전용 PPO의 학습 형상이 아니다.</p><table><tr><th>형상</th><th>총 질량(kg)</th><th>모션별 평균 발 목표 오차 중 최댓값(mm)</th><th>역동역학 불일치 비율 중 최댓값</th><th>첫 자세 최대 바닥 관통(mm)</th></tr>{diagnostics}</table><p>기하학적 발 오차가 작아도 동적 수행 가능성을 보장하지 않는다. 목표의 역동역학 불일치나 초기 관통이 큰 형상은 제어기 전이와 목표 난도가 함께 영향을 줄 수 있다. 여기서는 초기 상태를 숨겨 보정하거나 불리한 모션을 제외하지 않았다.</p>
<h2>모션별 상세</h2>{detail}
<h2>실제 MJX 기록 영상</h2><p>주황 A1 전체 외형, 보라 해당 형상의 TT 목표, 파랑 같은 형상의 TT 실제 기록이다. 하나의 바닥·카메라에서 일정한 표시 간격만 두었다. TT 외형도 각 E에 맞게 생성했다. 영상은 실제 종료 시점까지며 file:/// 절대 경로다.</p>{videos}
<h2>해석 범위</h2><p>현재 정책이 새로운 형상에서 어느 정도 실행되는지 보는 1차 전이 평가다. 학습에 사용한 5개 모션과 한 seed만 확인했으므로 기본 모션 전체·100개 연속동작·128형상으로의 확장 가능성을 아직 보장하지 않는다. 생존이 높더라도 정확한 추종과 동일하지 않다. 대량 수집은 이번 작업에서 진행하지 않았다.</p><p><a href="comparison.json">전체 비교 수치</a> · <a href="designs.json">형상 E 목록</a> · <a href="../try2/report_try2.html">e000 기준 보고서</a></p></html>'''
(p/'report_try4.html').write_text(body);print(json.dumps(summary,ensure_ascii=False,indent=2))
