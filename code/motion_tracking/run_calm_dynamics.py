"""Resumable per-design retarget -> PPO tracking -> actual dynamics rollout queue."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from plan_calm_dynamics import build,digest,prepare_reference

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).parent


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def command(args,log):
    env=os.environ.copy();env.update(WANDB_MODE='offline',WANDB_SILENT='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl')
    with log.open('a') as f:subprocess.run([str(x) for x in args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def retarget(manifest,design,out):
    import pd_replay as pd
    from calm_pilot import calm_source_model
    from multi_design import check_reference
    originals={m['id']:m for m in manifest['motions']};ids=design['common']+design['extra']
    refs=out/'references'/design['design'];target=refs/manifest['topology'];target.mkdir(parents=True,exist_ok=True)
    source=out/'a1_reference';source.mkdir(exist_ok=True)
    names=[]
    for i in ids:
        motion=originals[i];path=Path(motion['file'])
        if digest(path)!=motion['sha256']:raise ValueError('source motion changed')
        name='calm_%05d'%i;names.append(name);dest=source/(name+'.npz')
        if not dest.exists():prepare_reference(path,dest)
    missing=[]
    for name in names:
        path=target/(name+'.npz')
        if path.exists():check_reference(path,design['E'])
        else:missing.append(name)
    if missing:
        # calm_source_model creates this directory itself. Rebuild from the original URDF on resume.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='calm-source-') as tmp:
            model=calm_source_model(Path(tmp)/'model')
        original=pd.build_model
        pd.build_model=lambda choice=None,**kw:model if choice is None else original(choice,**kw)
        try:pd.retarget_tt(str(source),str(refs),only=missing,choices=[manifest['topology']],uniforms=design['E'],torque_limit=33.5)
        finally:pd.build_model=original
    model=pd.build_model(manifest['topology'],uniforms=design['E'],torque_limit=33.5);low,high=model.jnt_range[1:].T
    metrics=[]
    for name in names:
        path=target/(name+'.npz');check_reference(path,design['E'],low,high)
        with np.load(path) as z:
            metrics.append(dict(motion=name,foot_error_m=float(z['fix_track_err']),inverse_bad_fraction=float(z['inv_bad_frac']),frames=len(z['fix_q']),sha256=digest(path)))
    save(refs/'metrics.json',metrics)
    return target


def train_arguments(design,refs,run,steps):
    clips=['calm_%05d'%i for i in design['common']+design['extra']];array='['+','.join(clips)+']'
    return [sys.executable,'-u',CODE/'train_rl_procedural.py','multi_design=false','choice=00000011',
            'uniforms='+json.dumps(design['E'],separators=(',',':')),f'env.reference_dir={refs.parent}',f'eval_env.reference_dir={refs.parent}',
            'env.clips='+array,'eval_env.clips='+array,'env.max_episode_length=1000','eval_env.max_episode_length=1000',
            'env.velocity_kick=[0,0]','env.reset_qpos=0','env.reset_qvel=0',f'train.num_timesteps={steps}','train.num_envs=256',
            'train.batch_size=256','train.num_eval_envs=8','train.num_evals=2','train.num_minibatches=4','render_final=false','seed=0',f'hydra.run.dir={run}/hydra']


def checkpoints(run):
    return sorted((p for p in run.glob('wandb/offline-run-*/files/checkpoints/*') if p.name.isdigit()),key=lambda p:int(p.name))


def process(manifest,design,out,stage,steps):
    name=design['design'];job=out/'jobs'/name;job.mkdir(parents=True,exist_ok=True);statefile=job/'state.json'
    protocol=dict(manifest_sha256=digest(out/'manifest.json'),E=design['E'],steps=steps,
                  sources={p.name:digest(p) for p in [Path(__file__),CODE/'train_rl_procedural.py',CODE/'collect_tracking_dynamics.py',CODE/'pd_replay.py',CODE/'track_env.py',CODE/'plan_calm_dynamics.py']})
    protocolpath=job/'protocol.json'
    if protocolpath.exists() and json.loads(protocolpath.read_text())!=protocol:raise ValueError('job source/config mismatch')
    save(protocolpath,protocol);state=json.loads(statefile.read_text()) if statefile.exists() else dict(design=name,retarget=False,train=False,rollout=False)
    refs=out/'references'/name/manifest['topology'];run=job/'policy';run.mkdir(exist_ok=True)
    try:
        if not state['retarget']:
            save(statefile,dict(state,status='retargeting'));retarget(manifest,design,out);state['retarget']=True;save(statefile,state)
        if stage=='retarget':return state
        if not state['train']:
            save(statefile,dict(state,status='training'));args=train_arguments(design,refs,run,steps)
            existing=checkpoints(run)
            if existing:args.append('restore_ckpt='+str(existing[-1]))
            env_dir=os.environ.get('WANDB_DIR');os.environ['WANDB_DIR']=str(run.resolve())
            try:command(args,job/'train.log')
            finally:
                if env_dir is None:os.environ.pop('WANDB_DIR',None)
                else:os.environ['WANDB_DIR']=env_dir
            if not checkpoints(run):raise RuntimeError('training returned without checkpoint')
            state['train']=True;save(statefile,state)
        if stage=='train':return state
        if not state['rollout']:
            save(statefile,dict(state,status='collecting_dynamics'));path=out/'rollouts'/name/'primitive_seed0.npz';path.parent.mkdir(parents=True,exist_ok=True)
            command([sys.executable,'-u',CODE/'collect_tracking_dynamics.py','--manifest',out/'manifest.json','--design',name,'--checkpoint',checkpoints(run)[-1],'--references',refs,'--out',path],job/'rollout.log')
            state['rollout']=True
        state['status']='complete';save(statefile,state);return state
    except Exception as e:
        state.update(status='failed',error=str(e));save(statefile,state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['retarget','train','all'],default='all');p.add_argument('--designs',nargs='+');p.add_argument('--train-steps',type=int,default=5242880);p.add_argument('--wait-for-source',action='store_true');a=p.parse_args()
    if a.train_steps<1:p.error('train steps must be positive')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (out/'manifest.json').exists():
        while True:
            source=a.source/'status.json'
            try:s=json.loads(source.read_text())
            except (OSError,json.JSONDecodeError):s={}
            waiting='waiting_for_source'
            if s.get('status')=='complete' and s.get('accepted')==10000:
                try:report=json.loads((a.source/'report_ready.json').read_text())
                except (OSError,json.JSONDecodeError):report={}
                if (report.get('ready') is True and report.get('video_count')==5
                        and report.get('source_config_sha256')==digest(a.source/'collection_config.json')
                        and report.get('report_sha256')==digest(a.source/'report.html')):break
                waiting='waiting_for_final_report'
            if s.get('status')=='incomplete':
                save(out/'status.json',dict(status='blocked_source_incomplete',source_accepted=s.get('accepted',0)))
                raise ValueError('source collection ended before 10,000 accepted motions')
            if not a.wait_for_source:raise ValueError('source collection is not complete')
            save(out/'status.json',dict(status=waiting,source_accepted=s.get('accepted',0),source_target=10000));time.sleep(20)
        build(a.source,out)
    manifest=json.loads((out/'manifest.json').read_text());selected=[e for e in manifest['designs'] if a.designs is None or e['design'] in a.designs]
    if a.designs and len(selected)!=len(set(a.designs)):raise ValueError('unknown design')
    failures=[]
    for index,design in enumerate(selected):
        save(out/'status.json',dict(status='running',stage=a.stage,current_design=design['design'],processed=index,total_designs=len(selected),failed_designs=failures))
        try:
            process(manifest,design,out,a.stage,a.train_steps)
            if index==0 and a.stage=='all':
                metrics=json.loads((out/'rollouts'/design['design']/'primitive_seed0.json').read_text())
                if metrics['tracking_pass_fraction']<.9:
                    save(out/'status.json',dict(status='tracking_quality_gate_not_met',design=design['design'],metrics=metrics,
                                               pending_designs=len(selected)-1,note='Baseline tracking must be improved before scaling to 1,000 shapes.'))
                    return
        except Exception:
            traceback.print_exc();failures.append(design['design'])
            if index==0:
                save(out/'status.json',dict(status='initial_design_failed',failed_designs=failures,pending_designs=len(selected)-1))
                return
    save(out/'status.json',dict(status='incomplete' if failures else 'complete',stage=a.stage,processed=len(selected),total_designs=len(selected),failed_designs=failures))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
