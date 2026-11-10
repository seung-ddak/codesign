"""Manual V100 entry point: prepare -> shared PPO -> rollout -> audit, within 48h."""
import argparse,concurrent.futures,fcntl,json,math,os,shutil,signal,subprocess,sys,time,threading
from pathlib import Path
from shared_calm_common import CODE,ROOT,create_plan,event,load_config,read,save,sha,stage_groups,ready_batch,covered_length

STOP=threading.Event()

def deadline(clock,config,phase):
    hours=0
    for key,value in config['phase_hours'].items():
        hours+=value
        if key==phase:return min(clock['deadline'],clock['started']+hours*3600)
    raise ValueError(phase)

def run_child(args,out,log,until,c,cpu=False):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',WANDB_MODE='offline',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION=str(c['pool_fraction']))
    # Volta uses the CUDA12 backend; avoid forcing Ampere-only Triton kernels.
    if cpu:env['JAX_PLATFORMS']='cpu'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                if shutil.disk_usage(out).free<c['reserve_disk_gb']*1024**3:raise OSError('Free disk fell below reserved capacity')
                if STOP.is_set() or time.time()>=until:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    return False
                time.sleep(2)
            return child.returncode==0
        finally:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()

def preflight(c,allow_other=False):
    import importlib.metadata
    cuda13=False
    try:importlib.metadata.version('jax-cuda13-plugin');cuda13=True
    except importlib.metadata.PackageNotFoundError:pass
    query=['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits']
    if os.environ.get('CUDA_VISIBLE_DEVICES'):query+=['-i',os.environ['CUDA_VISIBLE_DEVICES']]
    rows=subprocess.check_output(query,text=True).strip().splitlines()
    if len(rows)!=1:raise ValueError('Select exactly one GPU with CUDA_VISIBLE_DEVICES')
    name,mem,driver=[x.strip() for x in rows[0].split(',')]
    if 'V100' in name and cuda13:raise ValueError('V100 requires JAX CUDA12, not CUDA13. See README_V100_48H.md')
    if int(driver.split('.')[0])<525:raise ValueError('CUDA12 requires NVIDIA driver >=525')
    if not allow_other and ('V100' not in name or float(mem)<30000):raise ValueError('This preset expects one V100 32GB; use --allow-other-gpu only for a separate pilot')
    subprocess.run([sys.executable,'-c',"import jax,jax.numpy as jnp; assert any(d.platform=='gpu' for d in jax.devices()), 'JAX sees no GPU'; print(jax.devices()); print(jax.jit(lambda x:x+1)(jnp.ones(8)).block_until_ready())"],check=True,timeout=90,env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_PYTHON_CLIENT_MEM_FRACTION':'0.10'})
    return dict(gpu=name,memory_mib=float(mem),driver=driver)

def audit(out,plan):
    import numpy as np
    completed=0;valid=0;falls=0;sources=set();pairs=set();invalid=[];sizes=0;variants={};starts={};duplicate_starts=0;quality={};intervals={};tt_episodes=0;long_episodes=0
    catalog={r['design']:r['E'] for r in plan.get('designs',[])};hashes={r['id']:r.get('sha256') for r in plan['motions']};durations={r['id']:round(r.get('seconds',0)*30) for r in plan['motions']}
    for p in sorted((out/'rollouts').glob('*/variant_*.npz')):
        try:
            with np.load(p) as z:
                b,t=z['valid'].shape
                if z['qpos'].shape[:2]!=(b,t+1) or z['action'].shape[:2]!=(b,t):raise ValueError('transition length mismatch')
                if not np.isclose(float(z['dt']),1/30):raise ValueError('wrong control period')
                mask=z['valid'].astype(bool)
                if np.any(mask[:,1:]&~mask[:,:-1]):raise ValueError('valid must be a prefix')
                for key in ['qpos','qvel','action','target_q']:
                    if not np.isfinite(z[key]).all():raise ValueError('nonfinite saved data')
                name=str(z['design'])
                if catalog and (name not in catalog or not np.allclose(z['E'],catalog[name],rtol=0,atol=1e-7)):raise ValueError('wrong morphology')
                for i,n in enumerate(z['source_motion_id']):
                    n=int(n)
                    if n not in hashes:raise ValueError('unknown source motion')
                    if hashes[n] is not None and str(z['group_id'][i])!=hashes[n]:raise ValueError('wrong source group')
                    if plan.get('assignments') and n not in plan['assignments'][name]:raise ValueError('wrong source/design pair')
                    if mask[i].sum()>=2:
                        pair=(name,n);sources.add(n);pairs.add(pair);variants.setdefault(pair,set()).add(int(z['start_variant'] if 'start_variant' in z else z['variant']))
                        start=int(z['start_frame'][i]) if 'start_frame' in z else 0
                        end=start+int(mask[i].sum())
                        if start<0 or (durations[n] and end>durations[n]):raise ValueError('source reference frame outside motion')
                        intervals.setdefault(n,[]).append((start,end))
                        fingerprint=np.round(np.r_[z['qpos'][i,0],z['qvel'][i,0]],4).tobytes()
                        if fingerprint in starts.setdefault(pair,set()):duplicate_starts+=1
                        starts[pair].add(fingerprint)
                tt_episodes+=int((mask.sum(1)>=2).sum());long_episodes+=int((mask.sum(1)>=30).sum())
                valid+=int(mask.sum());falls+=int(z['fell'].sum());completed+=b;sizes+=p.stat().st_size
                q=quality.setdefault(name,dict(episodes=0,falls=0));q['episodes']+=b;q['falls']+=int(z['fell'].sum())
        except Exception as e:invalid.append(dict(file=str(p),error=str(e)))
    for q in quality.values():q.update(fall_fraction=q['falls']/q['episodes'],collection_pass=q['falls']/q['episodes']<=.5)
    expected={m['id'] for m in plan['motions']};full_pairs=sum(len(v)==2 for v in variants.values())
    time_target=sum(round(m.get('seconds',0)*30) for m in plan['motions']);time_covered=sum(covered_length(v) for v in intervals.values())
    primitive_frames=[0]*12
    for m in plan['motions']:
        for segment in m.get('segments',[]):
            overlaps=[(max(a,segment['start']),min(b,segment['end'])) for a,b in intervals.get(m['id'],[]) if b>segment['start'] and a<segment['end']]
            primitive_frames[segment['index']]+=covered_length(overlaps)
    result=dict(tt_usable_episodes=tt_episodes,episodes_at_least_one_second=long_episodes,reference_frames_covered=time_covered,reference_frames_target=time_target,reference_time_coverage=time_covered/time_target if time_target else None,primitive_reference_frames=primitive_frames,episodes=completed,target_episodes=plan['target_episodes'],valid_transitions=valid,falls=falls,source_coverage=len(sources),source_target=len(expected),unique_pairs=len(pairs),target_pairs=plan['target_pairs'],pairs_with_both_starts=full_pairs,missing_source_ids=sorted(expected-sources),invalid_files=invalid,bytes=sizes,duplicate_initial_states=duplicate_starts,design_quality=quality,
                coverage_complete=not invalid and duplicate_starts==0 and sources==expected and full_pairs==plan['target_pairs'],note='Coverage is not proof of tracking accuracy or independent samples.')
    save(out/'audit.json',result);return result

def execute(out,allow_other):
    plan=read(out/'plan.json');c=plan['config'];lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=[*CODE.glob('shared_calm*.py'),CODE/'multi_design.py',CODE/'track_env.py',CODE/'pd_replay.py',CODE/'plan_calm_dynamics.py',CODE/'run_calm_dynamics.py',ROOT.parent/'transformer-transformer/t2/mjx_env/train.py']
    signature={str(p.relative_to(ROOT.parent)):sha(p) for p in sources};signature['plan']=sha(out/'plan.json')
    previous=read(out/'run_signature.json')
    if previous is not None and previous!=signature:raise ValueError('Code or immutable plan changed; preserve this run and use a new output directory')
    save(out/'run_signature.json',signature)
    gpu=preflight(c,allow_other);save(out/'machine.json',gpu)
    if shutil.disk_usage(out).free<c['reserve_disk_gb']*1024**3:raise ValueError('Insufficient free disk reserve')
    clock=read(out/'clock.json')
    if clock is None:clock=dict(started=time.time(),deadline=time.time()+c['hours']*3600);save(out/'clock.json',clock)
    if time.time()>=clock['deadline']:raise ValueError('48h deadline already expired; inspect audit and create a new output for another budget')
    stop=False
    def handler(*_):
        nonlocal stop
        stop=True;STOP.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    monitor=subprocess.Popen([sys.executable,'-u',CODE/'shared_calm_dashboard.py','monitor','--out',out],cwd=ROOT)
    state=read(out/'state.json',{});state.update(status='running',started=clock['started'],deadline=clock['deadline']);save(out/'state.json',state)
    try:
        ready=out/'ready';ready.mkdir(exist_ok=True)
        rows=plan['designs']+plan['eval_designs']
        def prepare(row):
            name=row['design']
            if (ready/(name+'.json')).exists():return True
            if STOP.is_set() or time.time()>=deadline(clock,c,'prepare'):return False
            okay=run_child([CODE/'shared_calm.py','retarget','--out',out,'--design',name],out,out/'logs'/f'retarget_{name}.log',deadline(clock,c,'prepare'),c,cpu=True)
            if okay:save(ready/(name+'.json'),dict(design=name,time=time.time()))
            event(out,'retarget_result',design=name,okay=okay);return okay
        state['phase']='prepare';save(out/'state.json',state);print('단계: 모션 리타겟팅',flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=c['prepare_workers']) as pool:
            for _ in pool.map(prepare,rows):
                state['ready_designs']=len(list(ready.glob('*.json')));save(out/'state.json',state);print(f'리타겟 준비 형상 {state["ready_designs"]}/{len(rows)}',flush=True)
        groups=stage_groups(plan);order=[]
        import numpy as np
        rng=np.random.default_rng(c['seed'])
        for _ in range(c['train_sweeps']):order.extend(rng.permutation(len(groups)).tolist())
        state['phase']='train';state['total_visits']=len(order);save(out/'state.json',state)
        for visit,index in enumerate(order):
            finished=out/'stages'/f'{visit:03d}'/'complete.json'
            if finished.exists():continue
            if time.time()>=deadline(clock,c,'train')-30:break
            names,_=ready_batch(out,plan,index)
            if not names:
                event(out,'train_batch_skipped_missing_references',stage=index);continue
            current=read(out/'learner.json',{}).get('step',0)
            steps=max(c['checkpoint_steps'],math.ceil((c['train_transitions']-current)/max(1,len(order)-visit)))
            seconds=max(15,(deadline(clock,c,'train')-time.time())/max(1,len(order)-visit)-10)
            state.update(visit=visit+1,current_designs=names,step=current);save(out/'state.json',state);print(f'학습 묶음 {visit+1}/{len(order)} 시작, 누적 전이 {current:,}',flush=True)
            okay=run_child([CODE/'shared_calm_train.py','--out',out,'--stage',index,'--visit',visit,'--steps',steps,'--seconds',seconds],out,out/'logs'/f'train_{visit:03d}.log',min(deadline(clock,c,'train'),time.time()+seconds+10),c)
            event(out,'train_visit_result',visit=visit,stage=index,okay=okay)
            if not okay and not (out/'checkpoint.json').exists():continue
        if not (out/'checkpoint.json').exists():raise RuntimeError('No policy checkpoint saved; inspect logs/train_*.log')
        state['phase']='collect';state['step']=read(out/'learner.json',{}).get('step',0);save(out/'state.json',state)
        for row in plan['designs']:
            name=row['design']
            if not (ready/(name+'.json')).exists():continue
            for variant in range(2):
                summary=out/'rollouts'/name/f'variant_{variant}.json'
                if summary.exists() and summary.with_suffix('.npz').exists():continue
                if time.time()>=deadline(clock,c,'collect'):break
                state.update(current_designs=[name],variant=variant);save(out/'state.json',state);print(f'실제 rollout 수집: {name}, 시작 조건 {variant}',flush=True)
                okay=run_child([CODE/'shared_calm_collect.py','--out',out,'--design',name,'--variant',variant],out,out/'logs'/f'collect_{name}_{variant}.log',deadline(clock,c,'collect'),c)
                event(out,'collect_result',design=name,variant=variant,okay=okay)
        state['phase']='verify';save(out/'state.json',state)
        for row in plan['eval_designs']:
            name=row['design']
            if time.time()>=clock['deadline']-120:break
            if not (ready/(name+'.json')).exists():continue
            state['current_designs']=[name];save(out/'state.json',state)
            summary=out/'evaluation'/name/'variant_0.json'
            if not summary.exists():run_child([CODE/'shared_calm_collect.py','--out',out,'--design',name,'--variant',0,'--eval'],out,out/'logs'/f'eval_{name}.log',clock['deadline']-60,c)
        result=audit(out,plan);coverage=read(out/'training_coverage.json',{'motions':{}})
        learned=read(out/'learner.json',{}).get('step',0)
        reasons=[]
        if not result['coverage_complete']:reasons.append('실제 rollout 커버리지 또는 파일 검사 미달')
        if len(coverage['motions'])!=len(plan['motions']):reasons.append('학습에서 노출되지 않은 원본 모션 있음')
        if learned<c['train_transitions']:reasons.append('목표 학습 전이 수 미달')
        if len(list((out/'evaluation').glob('*/variant_0.json')))<len(plan['eval_designs']):reasons.append('검증 형상 평가 미완료')
        budget_limited=(time.time()>=clock['deadline']-120 or time.time()>=deadline(clock,c,'collect')-60 or (learned<c['train_transitions'] and time.time()>=deadline(clock,c,'train')-60))
        state.update(status='complete' if not reasons else 'budget_finished_incomplete' if budget_limited else 'incomplete',phase='finished',step=learned,finished=time.time(),audit=result,training_source_coverage=len(coverage['motions']),note=' / '.join(reasons))
    except KeyboardInterrupt:state.update(status='paused',note='Resume with the same run command; the original deadline remains.');raise
    except Exception as e:state.update(status='failed',error=str(e));raise
    finally:
        save(out/'state.json',state);monitor.terminate();monitor.wait();event(out,'runner_exit',status=state['status'])
        from shared_calm_dashboard import report
        report(out)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['init','run','retarget','audit','status','preflight']);p.add_argument('--config',type=Path,default=ROOT/'configs/shared_calm_v100_48h.yaml');p.add_argument('--source',type=Path,default=ROOT/'legacy/runs/calm_motion_10,000/dataset');p.add_argument('--out',type=Path,default=ROOT/'legacy/runs/shared_calm_v100_48h');p.add_argument('--design');p.add_argument('--allow-other-gpu',action='store_true');a=p.parse_args();out=a.out.resolve()
    if a.command=='init':
        if (out/'plan.json').exists():raise FileExistsError('Plan exists; use run to resume')
        c=load_config(a.config);out.mkdir(parents=True,exist_ok=True);plan=create_plan(a.source,c);save(out/'plan.json',plan);save(out/'designs.json',{r['design']:r['E'] for r in plan['designs']+plan['eval_designs']});save(out/'state.json',dict(status='prepared',phase='ready'));print(f'준비 완료: 모션 {len(plan["motions"]):,}개, 목표 rollout {plan["target_episodes"]:,}개; {out}')
        from shared_calm_dashboard import report
        report(out)
    elif a.command=='run':execute(out,a.allow_other_gpu)
    elif a.command=='preflight':print(preflight(load_config(a.config),a.allow_other_gpu))
    elif a.command=='status':print(json.dumps(read(out/'state.json',{}),ensure_ascii=False,indent=2))
    elif a.command=='audit':print(json.dumps(audit(out,read(out/'plan.json')),ensure_ascii=False,indent=2))
    elif a.command=='retarget':
        from run_calm_dynamics import retarget
        plan=read(out/'plan.json');row=next(r for r in plan['designs']+plan['eval_designs'] if r['design']==a.design)
        retarget(plan,dict(row,common=[],extra=plan['assignments'][a.design]),out)

if __name__=='__main__':main()
