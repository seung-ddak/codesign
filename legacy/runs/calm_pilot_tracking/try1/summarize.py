"""Build explanatory plots from the recorded simulation; no resimulation."""
import hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-calm')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
p=Path(__file__).resolve().parent
rows=json.loads((p/'metrics.json').read_text());conditions=json.loads((p/'conditions.json').read_text())
assert hashlib.sha256((p/'evaluation_source.py').read_bytes()).hexdigest()==conditions['source_script_sha256']
fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained');fig2,ax2=plt.subplots(figsize=(11,4),layout='constrained')
for ax,r in zip(axs.flat,rows):
 d=np.load(p/(r['motion']+'.npz'));q=d['qpos'];ref=d['reference_qpos'];time=d['time']
 assert len(time)==300 and np.isclose(time[-1],10) and np.isfinite(q).all()
 a=Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix();b=Rotation.from_quat(ref[:,[4,5,6,3]]).as_matrix()
 simlocal=np.einsum('tfi,tij->tfj',d['foot_pos']-q[:,:3,None].transpose(0,2,1),a)
 reflocal=np.einsum('tfi,tij->tfj',d['reference_foot_pos']-ref[:,:3,None].transpose(0,2,1),b)
 r['foot_body_mean_error_m']=float(np.linalg.norm(simlocal-reflocal,axis=-1).mean())
 r['max_position_error_m']=float(np.linalg.norm(q[:,:3]-ref[:,:3],axis=1).max())
 r['upright_min_z']=float(d['up_z'].min())
 r['joint_5deg_screen_pass']=r['joint_rmse_deg']<5
 r['foot_3cm_screen_pass']=r['foot_body_mean_error_m']<.03
 r['orientation_5deg_screen_pass']=r['orientation_mean_deg']<5
 r['tracking_error_screen_pass']=all(r[k] for k in ['joint_5deg_screen_pass','foot_3cm_screen_pass','orientation_5deg_screen_pass'])
 with imageio.get_reader(r['video_absolute_path']) as reader:
  meta=reader.get_meta_data();assert meta['size']==(960,544) and np.isclose(meta['fps'],30)
 r['video_resolution']=[960,544]
 ax.plot(ref[:,0],ref[:,1],c='#d88426',label='Reference');ax.plot(q[:,0],q[:,1],c='#2377bd',label='Physics TT')
 ax.scatter(q[0,0],q[0,1],c='black',s=15);ax.set(title=r['motion'],xlabel='World X (m)',ylabel='World Y (m)',aspect='equal');ax.legend(fontsize=8)
 ax2.plot(time,np.linalg.norm(q[:,:3]-ref[:,:3],axis=1),label=r['motion'][-3:])
axs.flat[-1].axis('off');fig.savefig(p/'trajectories.png',dpi=170)
ax2.set(xlabel='Time (s)',ylabel='Body position error (m)',title='Same-time body tracking error');ax2.legend(title='Motion ID');fig2.savefig(p/'position_errors.png',dpi=170)
(p/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False))
extra='''<h2>生存と追従の区別</h2>'''.replace('生存と追従の区別','생존과 추종의 구분')
extra+=f'<p><strong>5개 모두 10초 생존했지만, 5개 모두 추종 오차 선별 기준에 미달했다.</strong> 관절 RMSE는 {min(r["joint_rmse_deg"] for r in rows):.2f}~{max(r["joint_rmse_deg"] for r in rows):.2f}°, 몸통 위치 RMSE는 {min(r["body_position_rmse_m"] for r in rows):.2f}~{max(r["body_position_rmse_m"] for r in rows):.2f}m, 실제 접지 미끄러짐 평균은 {min(r["slip_mean_mps"] for r in rows):.3f}~{max(r["slip_mean_mps"] for r in rows):.3f}m/s다.</p>'
extra+='<p>관절 5°, 몸체 기준 발 위치 3cm, 자세 5°라는 기존 추종 기준값을 참고한 별도 오차 선별이다. 관절 지표는 이번 실험에서 전체 상태·관절에 대한 RMSE를 사용하므로 기존 평가기의 프레임별 RMS 평균과 집계 방식이 다르다. 위치 종료 오차·접지 일치율 등을 포함하는 원래 전체 평가와 동일한 판정은 아니다. 미끄러짐은 합격 기준 없이 측정값으로 제시한다.</p>'
extra+='<table><tr><th>모션</th><th>몸체 기준 발 평균 오차(cm)</th><th>접지 미끄러짐 95백분위(m/s)</th><th>실제 최대 토크(N·m)</th><th>초기 바닥 관통(mm)</th><th>오차 선별</th></tr>'
for r in rows:extra+=f'<tr><td>{r["motion"]}</td><td>{r["foot_body_mean_error_m"]*100:.2f}</td><td>{r["slip_p95_mps"]:.3f}</td><td>{r["peak_actual_torque_nm"]:.2f}</td><td>{r["initial_ground_penetration_m"]*1000:.2f}</td><td>{"통과" if r["tracking_error_screen_pass"] else "미통과"}</td></tr>'
extra+='</table><p>몸체 기준 발 오차는 몸통 위치와 방향의 이탈을 제거한 후 네 발의 상대 위치를 비교한다. 초기 바닥 관통 약 1mm는 그대로 둔 조건이며 이번 생존 판정에서 별도로 탈락시키지 않았다. 최대 토크가 한계에 도달하지 않았다는 사실만으로 이득이나 토크가 충분했다고 단정할 수는 없다.</p><h2>궤적과 시간별 이탈</h2><img style="width:100%" src="trajectories.png" alt="각 모션의 목표 궤적과 실제 TT 궤적 비교"><img style="width:100%" src="position_errors.png" alt="시간에 따른 몸통 위치 오차">'
extra+='<h2>이번 실험의 판단</h2><p>계산 단계에서 불일치 0%로 선별된 레퍼런스도, 기존 PD 제어기로 물리 실행하면 목표 자세·경로를 정확히 따라가지 못했다. 발 고정 보정 제거가 계산 지표를 개선했다는 사실과 실제 추종 성능 개선은 구분해야 한다. 다음에는 같은 30Hz 레퍼런스로 상태 피드백을 사용하는 추종 정책을 학습한 뒤 동일한 생존·오차·미끄러짐 지표로 비교하는 것이 적절하다. 이번 결과만으로 실패 원인을 레퍼런스 또는 PD 이득 하나로 확정하지 않는다.</p>'
report=p/'report_try1.html';text=report.read_text();text=text.replace('<h2>실제 실행 영상</h2>',extra+'<h2>실제 실행 영상</h2>').replace('원본 해상도 960×540','저장 해상도 960×544')
report.write_text(text)
assert sum(r['survived_10s'] for r in rows)==5 and not any(r['tracking_error_screen_pass'] for r in rows)
print(json.dumps(rows,ensure_ascii=False,indent=2))
