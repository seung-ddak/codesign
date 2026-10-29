"""Separate tracking errors and compare matched Isaac Gym/MJX recordings."""
import argparse
import base64
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from audit_calm_reference import tracking_hits
from eval_menagerie_calm import balanced_commands


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    project=Path(__file__).resolve().parents[2];workspace=project.parent
    cfg=yaml.safe_load((workspace/'external/a1-calm-main/calm/data/cfg/a1_heading_lat_up_c8_hit2.yaml').read_text())['env']
    grouped={};engine={};fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for source in ['Isaac Gym','MJX']:
        errors=[];hits=[];faces=[];ups=[];biases=[];derivative_bias=[];speed_pass=[]
        for i in range(10):
            isaac=np.load(project/f'runs/calm_balanced_motion/try1/hlc_random_{i:03d}/trace.npz');mjx=np.load(project/f'runs/calm_mjx_closed_loop/try1/episode_{i:03d}.npz')
            np.testing.assert_allclose(isaac['command'],mjx['command'],atol=1e-7)
            if source=='Isaac Gym':
                root=isaac['root'];q=np.zeros((300,19));q[:,:3]=root[1:,:3];q[:,3:7]=root[1:,[6,3,4,5]];v=root[1:,7:10];command=isaac['command'];position=root[:,:2]
            else:q=mjx['qpos'][1:];v=mjx['velocity'];command=mjx['command'];position=mjx['qpos'][:,:2]
            hit,e,f,u=tracking_hits(q,v,command,cfg);indices=np.r_[90:150,240:300]
            speed_ok=e<=np.maximum(cfg['faceLatHitVelAbs'],cfg['faceLatHitVelRel']*np.linalg.norm(command[:,:2],axis=1))
            errors.extend(e[indices]);hits.extend(hit[indices]);speed_pass.extend(speed_ok[indices]);faces.extend(f[indices]);ups.extend(u[indices]);biases.extend((v[:,:2]-command[:,:2])[indices]);derivative_bias.extend((np.diff(position,axis=0)*30-command[:,:2])[indices])
            if i==0:
                axes[0].plot(np.arange(300)/30,v[:,0],label=source);axes[1].plot(np.arange(300)/30,e,label=source)
            if source=='MJX':
                _,segments=balanced_commands(0,i)
                for j,segment in enumerate(segments):
                    idx=np.arange(j*150+90,(j+1)*150);kind=segment['kind'];g=grouped.setdefault(kind,[])
                    g.append(dict(episode=i,segment=j,speed_hit_rate=float(speed_ok[idx].mean()),facing_hit_rate=float((f[idx]<=.5).mean()),up_hit_rate=float((u[idx]<=.2).mean()),velocity_rmse_mps=float(np.sqrt(np.mean(e[idx]**2))),mean_vector_error_mps=float(np.linalg.norm((v[:,:2]-command[:,:2])[idx].mean(0))),mean_actual_speed_mps=float(np.linalg.norm(v[idx,:2],axis=1).mean()),mean_target_speed_mps=float(np.linalg.norm(command[idx,:2],axis=1).mean())))
        engine[source]=dict(tail_velocity_rmse_mps=float(np.sqrt(np.mean(np.array(errors)**2))),tail_joint_hit_rate=float(np.mean(hits)),tail_speed_hit_rate=float(np.mean(speed_pass)),tail_facing_hit_rate=float((np.array(faces)<=.5).mean()),tail_up_hit_rate=float((np.array(ups)<=.2).mean()),world_velocity_bias_mps=np.mean(biases,axis=0).tolist(),position_derived_world_velocity_bias_mps=np.mean(derivative_bias,axis=0).tolist())
    cmd=np.load(project/'runs/calm_mjx_closed_loop/try1/episode_000.npz')['command'];axes[0].plot(np.arange(300)/30,cmd[:,0],'k--',label='command');axes[0].set_ylabel('World X velocity (m/s)');axes[1].set_ylabel('Velocity error (m/s)');axes[1].set_xlabel('Time (s)')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.out/'comparison.png',dpi=150)
    summary=dict(scope='tracking only; floor penetration excluded from this analysis',window='last 60 frames of each five-second segment, 1200 frames total per engine',engines=engine,mjx_by_command_type={k:{field:float(np.mean([r[field] for r in rows])) for field in rows[0] if field not in ['episode','segment']} for k,rows in grouped.items()},segments=grouped,findings=['Velocity error dominates; facing and up criteria pass in all evaluated tail frames.','Matched command sequences differ in actual response across engines.','World +X bias persists in position-derived velocity; not just a saved instantaneous velocity sampling artifact.'],hypotheses=['PhysX implicit position drive versus explicit torque PD and contact solver differences; not isolated.','Observation/frame discrepancies are not fully excluded, although tested joint mapping and quaternion conversion match source conventions.'],not_tested=['One-step equal state/action torque and acceleration parity across engines','Single-variable drive/contact ablations','Exact original training setting recovery'])
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    text='<h1>추종 오차 원인 분석</h1><p>바닥 관통은 이번 분석에서 제외했습니다. 동일한 명령 10개를 Isaac Gym과 MJX에서 비교하고 각 5초 구간 후반 2초를 평가했습니다. 초기 전환 구간을 제외한 비교입니다.</p><table><tr><th>환경</th><th>속도 오차 m/s</th><th>속도 기준 만족률</th><th>방향 기준 만족률</th><th>기울기 기준 만족률</th><th>월드 속도 편향 [X,Y] m/s</th></tr>'
    for name,r in engine.items():text+='<tr><td>'+name+'</td><td>'+f"{r['tail_velocity_rmse_mps']:.3f}"+'</td><td>'+f"{r['tail_speed_hit_rate']:.1%}"+'</td><td>'+f"{r['tail_facing_hit_rate']:.1%}"+'</td><td>'+f"{r['tail_up_hit_rate']:.1%}"+'</td><td>'+str([round(x,3) for x in r['world_velocity_bias_mps']])+'</td></tr>'
    text+='</table><p>주요 직접 현상은 속도 크기·벡터 편향입니다. 전진과 왼쪽 이동에서는 실제 속도가 큰 경향, 후진에서는 작은 경향이 있습니다. 방향·기울기 실패가 이번 선별 실패의 주원인은 아닙니다. 위치 차분으로 계산한 속도에도 편향이 남으므로 순간 속도 저장 방식만의 문제로 보기 어렵습니다.</p><p>유력 가설은 물리 엔진 이전에 따른 동역학 및 관절 구동 차이입니다. 같은 게인·토크 제한이어도 PhysX 위치 드라이브와 MJX 명시적 토크 PD는 동일하지 않습니다. 이 가설의 인과관계는 아직 분리 검증하지 않았습니다. 단순히 HLC 능력 부족이나 명령 변화율 탓으로 단정하지 않습니다.</p><p>코드 대조: 관절 순서, quaternion 순서, 행동 범위 확대식, HLC 갱신 주기 30Hz 및 llc_steps=1은 원본과 대조했습니다. 이것만으로 모든 관측·동역학 차이가 배제되지는 않습니다.</p><p>다음 최소 실험: 두 엔진의 같은 초기 상태에서 같은 관절 목표각을 짧게 적용해 관절 응답·몸통 속도 변화를 비교합니다. 이후 일정 속도 전진·후진·횡이동에서 구동 설정만 하나씩 바꿔 편향 변화를 확인합니다. 보상이나 통과 기준은 바꾸지 않습니다.</p><img style="width:100%" src="data:image/png;base64,'+base64.b64encode((a.out/'comparison.png').read_bytes()).decode()+'">'
    (a.out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><body style="max-width:1100px;margin:auto;font:17px system-ui">'+text+'</body></html>');print(json.dumps(engine,indent=2))


if __name__=='__main__':main()
