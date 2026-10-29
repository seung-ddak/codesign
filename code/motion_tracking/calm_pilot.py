"""Sequential CALM collection -> TT retarget -> MJX PPO pilot, with issue events."""
import argparse
import csv
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import traceback
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
STAGES=['collect','prepare','retarget','train_one','evaluate_one','train_five','evaluate_five']
LABELS=['Isaac Gym 10개 수집','기록 검증·대표 5개 선정','TT 고정 형상 리타겟팅','단일 모션 PPO 시험','단일 모션 실제 추종 평가','5개 모션 PPO 확대','5개 모션 실제 추종 평가']


def _json_default(value):
    """Preserve NumPy values in audit JSON without allowing NaN/Inf."""
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=_json_default))
    tmp.replace(path)


def validate_trace(d):
    import numpy as np
    n=len(d['target_q'])
    shapes={'root':(n+1,13),'q':(n+1,12),'qvel':(n+1,12),'foot_pos':(n+1,4,3),'foot_floor_contact':(n+1,4),'target_q':(n,12),'llc_action':(n,12),'latent':(n,64),'command':(n,3),'joint_force_sensor':(n,12),'valid_transition':(n,),'time':(n+1,)}
    for k,shape in shapes.items():
        if d[k].shape!=shape or not np.isfinite(d[k]).all():raise ValueError(f'Invalid {k}: {d[k].shape}, expected {shape}')
    if n!=300 or not d['valid_transition'].all() or not np.allclose(np.diff(d['time']),1/30):raise ValueError('Expected 300 valid continuous transitions')
    if not np.allclose(np.linalg.norm(d['root'][:,3:7],axis=1),1,atol=1e-4):raise ValueError('Invalid quaternion')
    if not d['foot_floor_contact'].any():raise ValueError('No recorded foot-floor contacts')
    return n


def select_diverse(features,count):
    import numpy as np
    x=np.asarray(features);x=(x-x.mean(0))/np.maximum(x.std(0),1e-6)
    selected=[int(np.argmin(np.linalg.norm(x,axis=1)))]
    while len(selected)<count:
        distance=np.min(np.linalg.norm(x[:,None]-x[selected][None],axis=2),axis=1);distance[selected]=-1
        candidate=int(np.argmax(distance))
        if distance[candidate]<=1e-6:raise ValueError('Not enough distinct motion features')
        selected.append(candidate)
    return selected


def notify(out,issue,event):
    from notify_selected_experts import TOKEN_FILE,API,post
    if not TOKEN_FILE.is_file() or TOKEN_FILE.stat().st_mode&(stat.S_IRWXG|stat.S_IRWXO):raise RuntimeError('Private GitHub token file is unavailable or permissions are too broad')
    token=TOKEN_FILE.read_text().strip()
    if not token:raise RuntimeError('Empty GitHub token')
    marker=f'<!-- calm-pilot:{out.name}:{event["stage"]}:{event["status"]} -->'
    # Reconcile a posted event after a crash before local acknowledgement.
    page=1
    while True:
        req=Request(f'{API}/issues/{issue}/comments?per_page=100&page={page}',headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json','User-Agent':'codesign-calm-pilot'})
        with urlopen(req,timeout=20) as response:comments=json.load(response)
        found=next((c for c in comments if marker in c.get('body','')),None)
        if found:return found['html_url']
        if len(comments)<100:break
        page+=1
    body=marker+'\n### '+event['label']+' — '+event['status']+'\n\n'+event['summary']+'\n\n실행: `'+out.name+'`\n코드: `'+event['commit']+'`\n결과 경로: `'+str(out.relative_to(ROOT))+'`\n\n자기 충돌 끔·바닥 충돌 켬. 학습 완료와 실제 추종 통과를 구분합니다.'
    return post(f'/issues/{issue}/comments',{'body':body},token)['html_url']


def run_command(cmd,log,env=None):
    with log.open('w') as stream:
        result=subprocess.run([str(x) for x in cmd],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Command exit {result.returncode}; log: {log.relative_to(ROOT)}')


def calm_source_model(output):
    import mujoco
    from check_calm_mjx import build_model
    output.mkdir()
    build_model(ROOT.parent/'external/a1-calm-main/assets/urdf/a1_description/urdf/a1.urdf',output)
    spec=mujoco.MjSpec.from_file(str(output/'model.xml'))
    for leg in ['FR','FL','RR','RL']:
        geoms=list(spec.body(leg+'_foot').geoms)
        if len(geoms)!=1:raise ValueError('Ambiguous foot')
        geoms[0].name=leg+'_foot'
    # Existing retarget FK expects foot geoms on calf bodies, not separate fixed links.
    spec.compiler.fusestatic=True
    model=spec.compile()
    assert abs(np_length(model)-.4)<1e-6
    return model


def np_length(model):
    import numpy as np
    return np.linalg.norm(model.body_pos[model.body('FL_calf').id])+np.linalg.norm(model.geom_pos[model.geom('FL_foot').id])


def worker(stage,out):
    import numpy as np
    python=ROOT.parent/'transformer-transformer/.venv/bin/python'
    if stage=='collect':
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(Path.home()/'miniconda3/envs/rlgpu/lib'))
        cmd=[Path.home()/'miniconda3/envs/rlgpu/bin/python','-u',CODE/'eval_menagerie_calm.py','--out',out/'collection','--rl-games-wheel',ROOT/'runs/menagerie_calm_check_final/runtime/rl_games-1.1.4-py3-none-any.whl','--calm-no-self-collision','--random-episodes','10','--balanced','--seed','100']
        run_command(cmd,out/'collect_execution.log',env)
        rows=json.loads((out/'collection/summary.json').read_text());ok=sum(r['termination']=='duration' and r['resets']==0 for r in rows)
        return dict(status='PASS' if ok==10 else 'FAIL',summary=f'수집 {len(rows)}개, 10초 완료 {ok}개. 이는 기록 완료이며 완벽한 명령 추종 승인은 아닙니다.',count=len(rows),completed=ok)
    if stage=='prepare':
        from scipy.spatial.transform import Rotation,Slerp
        cases=sorted((out/'collection').glob('hlc_random_*/trace.npz'));features=[];records=[]
        for path in cases:
            d=np.load(path);validate_trace(d);summary=json.loads((path.parent/'summary.json').read_text())
            if summary['termination']!='duration' or summary['resets']!=0:raise ValueError('Invalid episode termination')
            rot=Rotation.from_quat(d['root'][:,3:7]);body=rot.inv().apply(d['root'][:,7:10]);yaw=np.unwrap(rot.as_euler('ZYX')[:,0]);acc=np.diff(d['root'][:,7:9],axis=0)*30
            f=[body[:,0].mean(),body[:,1].mean(),np.linalg.norm(body[:,:2],axis=1).mean(),yaw[-1]-yaw[0],np.abs(d['root'][:,12]).mean(),np.linalg.norm(acc,axis=1).mean()];features.append(f);records.append(path)
        if len(records)!=10:raise ValueError('Expected ten episodes')
        chosen=select_diverse(features,5)
        # Start with the selected clip with the lowest measured acceleration, not a claimed pure forward clip.
        chosen.sort(key=lambda i:features[i][-1]);selected=[];(out/'a1_reference').mkdir()
        for i in chosen:
            d=np.load(records[i]);ts=d['time'];td=np.arange(501)/50;root=d['root'];names=d['joint_names'].tolist();order=[names.index(l+'_'+p+'_joint') for l in ['FR','FL','RR','RL'] for p in ['hip','thigh','calf']]
            q=np.stack([np.interp(td,ts,d['q'][:,j]) for j in order],axis=1);pos=np.stack([np.interp(td,ts,root[:,j]) for j in range(3)],axis=1);quat=Slerp(ts,Rotation.from_quat(root[:,3:7]))(td).as_quat()[:,[3,0,1,2]]
            near=np.minimum(np.round(td*30).astype(int),300);contact=d['foot_floor_contact'][near]
            foot=np.stack([np.stack([np.interp(td,ts,d['foot_pos'][:,f,j]) for j in range(3)],axis=1) for f in range(4)],axis=1)-[0,0,.02]
            name='calm_'+records[i].parent.name.removeprefix('hlc_');np.savez_compressed(out/'a1_reference'/f'{name}.npz',fix_q=q,fix_base_pos=pos,fix_base_quat=quat,fix_foot_pos=foot,fix_contact=contact,dt=.02,source_motion=str(records[i].relative_to(ROOT)),seed=100+i)
            selected.append(name)
        save(out/'selection.json',dict(clips=selected,features=features,selected_indices=chosen,feature_columns=['body_vx','body_vy','speed','yaw_delta','abs_yaw_rate','acceleration'],source_model='Original CALM A1',reference_dt=.02))
        return dict(status='PASS',summary='10개 기록의 상태 301개·행동 300개 정합성과 유한값 검증 완료. 실제 움직임 특징으로 대표 5개 선정, 50Hz 상태 501개로 변환했습니다. 힘 센서는 순수 구동 토크로 간주하지 않습니다.',clips=selected)
    if stage=='retarget':
        import mujoco
        import pd_replay as pd
        from check_calm_mjx import build_model
        selection=json.loads((out/'selection.json').read_text());source=calm_source_model(out/'source_model');original=pd.build_model
        def model_fn(choice=None,**kwargs):return source if choice is None else original(choice,**kwargs)
        pd.build_model=model_fn
        try:pd.retarget_tt(str(out/'a1_reference'),str(out/'references/e000'),choices=['00000011'],uniforms=[.5]*13,torque_limit=33.5)
        finally:pd.build_model=original
        target=original('00000011',uniforms=[.5]*13,torque_limit=33.5);rows=[]
        for name in selection['clips']:
            d=np.load(out/'references/e000/00000011'/f'{name}.npz');q=d['fix_q'];lo,hi=target.jnt_range[1:].T
            finite=all(np.isfinite(d[k]).all() for k in ['fix_q','fix_base_pos','fix_base_quat','fix_foot_pos']);limit=float(np.maximum(lo-q,q-hi).max());err=float(d['fix_track_err']);bad=float(d['inv_bad_frac'])
            passed=finite and len(q)==501 and abs(float(d['dt'])-.02)<1e-9 and limit<=.01 and err<=.03 and bad<=.1
            rows.append(dict(clip=name,pass_pilot=bool(passed),frames=len(q),foot_error_m=err,joint_limit_excess_rad=max(0,limit),inverse_bad_fraction=bad))
        save(out/'retarget_metrics.json',rows);passed=sum(r['pass_pilot'] for r in rows)
        return dict(status='PASS' if passed==5 else 'FAIL',summary=f'CALM 원본 운동학으로 TT 00000011/e000 리타겟팅 수행. 선별 {passed}/5 통과. 사전 기준: 발 오차 ≤3cm, 관절 한계 초과 ≤0.01rad, 역동역학 불일치 프레임 ≤10%. 이는 소규모 학습 진입 선별이며 전체 G3 승인은 아닙니다.',metrics=rows)
    selection=json.loads((out/'selection.json').read_text());clips=selection['clips'][:1] if stage.endswith('one') else selection['clips']
    mode='one' if stage.endswith('one') else 'five';stage_dir=out/('ppo_'+mode)
    if stage.startswith('train'):
        stage_dir.mkdir();env=os.environ.copy();env.update(WANDB_MODE='offline',WANDB_DIR=str(stage_dir),MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1')
        ref=str(out/'references/e000/00000011');cliparg='['+','.join(clips)+']'
        cmd=[python,CODE/'train_rl_procedural.py','choice=00000011','multi_design=false',f'env.reference_dir={ref}',f'eval_env.reference_dir={ref}',f'env.clips={cliparg}',f'eval_env.clips={cliparg}','env.max_episode_length=500','eval_env.max_episode_length=500','env.velocity_kick=[0,0]','env.reset_qvel=0','env.reset_qpos=0','train.num_timesteps=5242880','train.num_envs=32','train.batch_size=32','train.num_eval_envs=4','train.num_evals=3','train.num_minibatches=4','train.unroll_length=24','render_final=false','seed=0',f'hydra.run.dir={stage_dir}/hydra']
        run_command(cmd,stage_dir/'train.log',env)
        candidates=list(stage_dir.glob('wandb/offline-run-*/files/checkpoints/*'));candidates=[p for p in candidates if p.name.isdigit()]
        if not candidates:raise RuntimeError('No saved PPO checkpoint')
        ckpt=max(candidates,key=lambda p:int(p.name));save(out/f'checkpoint_{mode}.json',dict(path=str(ckpt),run=str(ckpt.parents[2])))
        return dict(status='PASS',summary=f'{len(clips)}개 모션 PPO 시험 학습 완료. 체크포인트 {ckpt.name}. 고정 예산 5,242,880 환경 스텝 요청. 실제 추종 성공은 다음 평가 단계에서 별도로 판정합니다.',checkpoint=str(ckpt))
    ckpt=json.loads((out/f'checkpoint_{mode}.json').read_text());env=os.environ.copy();env.update(MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    run_command([python,CODE/'eval_motions.py',ckpt['run'],*clips,'--checkpoint',ckpt['path'],'--output',stage_dir/'evaluation'],stage_dir/'eval.log',env)
    rows=list(csv.DictReader((stage_dir/'evaluation/e000/motion_eval.csv').open()));passed=sum(int(r['pass_all']) for r in rows)
    return dict(status='PASS' if len(rows)==len(clips) and passed==len(clips) else 'FAIL',summary=f'시작 프레임부터 전체 10초 실제 추종 평가: {passed}/{len(clips)} 통과. 기존 eval_motions 판정 기준을 그대로 적용했습니다. 실패하면 학습 확대를 멈춥니다. 고정 시험 예산 실패가 학습 불가능을 의미하지는 않습니다.',metrics=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--issue',type=int,required=True);p.add_argument('--worker',choices=STAGES);a=p.parse_args();out=a.out.resolve()
    if not out.is_relative_to(ROOT/'runs') or a.issue<=0:p.error('Use an output under runs and a positive issue number')
    if a.worker:
        result=worker(a.worker,out);save(out/(a.worker+'_result.json'),result);return
    out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();statefile=out/'pipeline.json'
        state=json.loads(statefile.read_text()) if statefile.exists() else dict(issue=a.issue,commit=commit,events=[],criteria=dict(self_collision=False,ground_collision=True,retarget_foot_error_m=.03,joint_limit_excess_rad=.01,inverse_bad_fraction=.1,ppo_requested_steps=5242880,evaluation='existing eval_motions.PASS; full episode from first frame'),source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),CODE/'pd_replay.py',CODE/'track_env.py',CODE/'eval_menagerie_calm.py',CODE/'train_rl_procedural.py',CODE/'eval_motions.py']})
        if state['issue']!=a.issue:raise RuntimeError('Issue differs from saved run')
        save(statefile,state)
        if not (out/'dependency_sources').exists():
            import shutil
            (out/'dependency_sources').mkdir()
            for name in ['pd_replay.py','track_env.py']:shutil.copyfile(CODE/name,out/'dependency_sources'/name)
        def persist():
            save(statefile,state)
            content='<h1>CALM → TT → MJX PPO 연결 시험</h1><p>자기 충돌 끔·바닥 충돌 켬. 바닥 관통은 이번 통과 기준에서 제외합니다. 기존 파일을 덮어쓰지 않으며 실패 시 의존 단계를 중단합니다.</p>'
            for event in state['events']:content+='<h2>'+html.escape(event['label']+' — '+event['status'])+'</h2><p>'+html.escape(event['summary'])+'</p>'
            (out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><body style="max-width:1000px;margin:auto;font:17px system-ui">'+content+'</body></html>')
        for event in state['events']:
            if not event.get('comment_url'):event['comment_url']=notify(out,a.issue,event);persist()
        if any(e['status'] in ['FAIL','BLOCKED'] for e in state['events']):return
        done={e['stage'] for e in state['events']}
        for stage,label in zip(STAGES,LABELS):
            if stage in done:continue
            running=out/(stage+'.started')
            if running.exists():raise RuntimeError(f'Interrupted stage {stage}; inspect outputs before restarting')
            running.write_text(commit);state['running']=stage;persist()
            try:
                result_path=out/(stage+'_result.json')
                run_command([sys.executable,Path(__file__),'--out',out,'--issue',str(a.issue),'--worker',stage],out/(stage+'.log'))
                result=json.loads(result_path.read_text())
            except Exception as error:
                result=dict(status='FAIL',summary=str(error)+'\n\n'+(out/(stage+'.log')).read_text(errors='replace')[-2500:]);(out/(stage+'_error.txt')).write_text(traceback.format_exc())
            event=dict(stage=stage,label=label,commit=commit,**result);state['events'].append(event);state['running']=None;persist()
            try:event['comment_url']=notify(out,a.issue,event);persist()
            except Exception as error:
                state['notification_pending']=str(error);persist();raise
            if event['status']!='PASS':
                for later in STAGES[STAGES.index(stage)+1:]:state.setdefault('not_run',[]).append(later)
                persist();return
        state['finished']=True;persist()


if __name__=='__main__':main()
