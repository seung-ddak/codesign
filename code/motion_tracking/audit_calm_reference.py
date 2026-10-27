"""Audit existing MJX references without changing physics or collecting data."""
import argparse
import base64
import html
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def tracking_hits(qpos,velocity,command,limits):
    rotation=Rotation.from_quat(qpos[:,[4,5,6,3]])
    forward=rotation.apply(np.tile([1.,0,0],(len(qpos),1)))
    up=rotation.apply(np.tile([0.,0,1],(len(qpos),1)))
    facing=np.column_stack([np.cos(command[:,2]),np.sin(command[:,2]),np.zeros(len(command))])
    face_error=np.arccos(np.clip(np.sum(forward*facing,axis=1),-1,1));up_error=np.arccos(np.clip(up[:,2],-1,1))
    speed_error=np.linalg.norm(velocity[:,:2]-command[:,:2],axis=1)
    hit=(speed_error<=np.maximum(limits['faceLatHitVelAbs'],limits['faceLatHitVelRel']*np.linalg.norm(command[:,:2],axis=1)))&(face_error<=limits['faceLatHitYaw'])&(up_error<=limits['faceLatHitUp'])
    return hit,speed_error,face_error,up_error


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import mujoco,yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    workspace=Path(__file__).resolve().parents[3];cfgpath=workspace/'external/a1-calm-main/calm/data/cfg/a1_heading_lat_up_c8_hit2.yaml'
    raw=yaml.safe_load(cfgpath.read_text());cfg=raw['env'];model=mujoco.MjModel.from_xml_path(str(a.source/'model.xml'));data=mujoco.MjData(model)
    floor=model.geom('floor').id;rows=[];cards=[];all_direction=np.zeros(4,int)
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for path in sorted(a.source.glob('episode_*.npz')):
        d=np.load(path);q=d['qpos'];command=d['command'];v=d['velocity'];n=len(command);assert len(q)==n+1 and len(d['target_q'])==n
        hit,se,fe,ue=tracking_hits(q[1:],v,command,cfg);dt=float(d['dt']);time=np.arange(n)*dt
        rates=[float(hit[j:j+150][-cfg['faceLatHitWindow']:].mean()) for j in range(0,n,150)]
        depths=[]
        for pose in q:
            data.qpos[:]=pose;mujoco.mj_forward(model,data)
            depths.append(max([max(0.,-c.dist) for c in data.contact if floor in c.geom]+[0.]))
        rotation=Rotation.from_quat(q[1:,[4,5,6,3]]);world=np.column_stack([v[:,:2],np.zeros(n)]);body=rotation.inv().apply(world);moving=np.linalg.norm(v[:,:2],axis=1)>.05
        sector=np.floor(((np.arctan2(body[moving,1],body[moving,0])+np.pi/4)%(2*np.pi))/(np.pi/2)).astype(int);all_direction+=np.bincount(sector,minlength=4)
        passed=bool(n==300 and np.isfinite(q).all() and np.isfinite(d['target_q']).all() and all(r>=cfg['faceLatHitRate'] for r in rates))
        row=dict(episode=path.stem,tracking_status='PASS' if passed else 'FAIL',seconds=n*dt,segment_tail_hit_rates=rates,all_frame_hit_rate=float(hit.mean()),velocity_rmse_mps=float(np.sqrt(np.mean(se**2))),facing_error_p95_rad=float(np.percentile(fe,95)),up_error_p95_rad=float(np.percentile(ue,95)),initial_floor_penetration_m=float(depths[0]),max_floor_penetration_m=float(max(depths)),penetration_status='NOT RUN: acceptance tolerance not defined',missing=['joint velocity','root angular velocity','foot position/contact','actual substep torque','original termination flags']);rows.append(row)
        for ax,values in zip(axes,[se,fe,ue]):ax.plot(time,values,label=path.stem)
        cards.append('<tr><td>'+path.stem+'</td><td>'+row['tracking_status']+'</td><td>'+', '.join(f'{r:.1%}' for r in rates)+'</td><td>'+f'{max(depths)*1000:.2f}'+'</td></tr>')
    for ax,label in zip(axes,['Velocity error (m/s)','Facing error (rad)','Up error (rad)']):ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(a.out/'tracking.png',dpi=150)
    summary=dict(physics_decision='self collision OFF; ground collision ON',criterion_source=str(cfgpath),criterion_note='Adapted screening: original per-step velocity/facing/up thresholds; last 60 control frames of each 5-second changing-command segment must reach 80%. This is not the original training success evaluation. Stop yaw-rate condition cannot be verified from saved angular velocity.',G0='NOT RUN: full original checklist approval incomplete; modified physics execution demonstrated',G1='NOT RUN: full checklist approval incomplete; actual diversity demonstrated',G2='NOT RUN: data recording incomplete; representative selection not started',tracking_pass=sum(r['tracking_status']=='PASS' for r in rows),tracking_fail=sum(r['tracking_status']=='FAIL' for r in rows),direction_fraction_moving=dict(zip(['forward','left','backward','right'],(all_direction/all_direction.sum()).tolist())),results=rows)
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));payload=base64.b64encode((a.out/'tracking.png').read_bytes()).decode()
    text='<h1>CALM MJX G0·G1 판정 준비 감사</h1><p>자기 충돌 끔·바닥 충돌 켬으로 고정합니다. 10초 완료와 정상 명령 추종은 구분합니다.</p><p>추종 선별 기준: 속도 오차 ≤ max(0.15 m/s, 목표 속도의 30%), 몸통 앞 방향 오차 ≤ 0.5 rad, 몸통 기울기 ≤ 0.2 rad. 각 5초 구간의 마지막 60프레임에서 세 조건을 동시에 만족한 비율이 80% 이상이어야 합니다. 기존 HLC 설정값을 사용한 별도 선별 검사이며 원본 평가 절차와 동일하지 않습니다. 정지 시 각속도 조건은 미검증입니다.</p><p>전체 G0·G1 승인 보류: 초기 관통 허용치, 정지 시 각속도, 실제 토크 및 원래 종료 조건 검사가 남았습니다. G2는 관절속도·루트 각속도·발 위치와 접촉 저장을 보완해야 합니다. 자기 관통은 현재 허용한 물리 조건이며 검사 대상에서 제외합니다.</p><p>추종 선별 '+str(summary['tracking_pass'])+'개 통과, '+str(summary['tracking_fail'])+'개 실패.</p><table><tr><th>구간</th><th>추종 선별</th><th>두 구간 마지막 60프레임 만족 비율</th><th>최대 바닥 관통 mm</th></tr>'+''.join(cards)+'</table><img style="width:100%" src="data:image/png;base64,'+payload+'"><p>실제 영상: <a href="'+str((a.source/'report.html').resolve())+'">기존 실행 보고서</a></p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
    video=a.source/'episode_000.mp4'
    if video.exists():text+='<h2>대표 실행 영상</h2><video style="width:100%" controls muted playsinline src="data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()+'"></video>'
    (a.out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><body style="max-width:1100px;margin:auto;font:17px system-ui">'+text+'</body></html>');print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
