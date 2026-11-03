"""Matched v12/v13 MJX task validation, with separate status and bounded resources."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
V13=OUT.parent
V12=V13.parent/'fast_mode24_v12_pd_kp200_mode1'
STAGES=['v13_smoke_2s']+[f'v13_{kind}_{task}_{seconds}s' for seconds in (4,10) for task in ('zigzag','circle') for kind in ('fixedE','joint')]+[f'v12_joint_{task}_10s' for task in ('zigzag','circle')]


def spec(stage):
    if stage=='v13_smoke_2s':return dict(model='v13',kind='joint',task='zigzag',seconds=2,scale=0,candidates=16,render=False)
    model,kind,task,duration=stage.split('_')
    return dict(model=model,kind=kind,task=task,seconds=int(duration[:-1]),scale=10,candidates=4 if kind=='fixedE' else 16,render=model=='v13' and kind=='joint')


def command(stage,designs):
    s=spec(stage);run=V13 if s['model']=='v13' else V12
    args=[str(ROOT/'code/tt_dynamics/generate_task.py'),'--ckpt',str(run/'tt_joint'),
          '--expert-config','runs/pipeline_v1/experts_wide16/walkForwardSlow__r00','--anchor-s0','--joint-full-guidance',
          '--e-rest-s0','--no-s0-align','--kp','200','--seed','0','--out',str(OUT/stage),
          '--seconds',str(s['seconds']),'--task',s['task'],'--scales',str(s['scale']),'--candidates',str(s['candidates'])]
    if not s['render']:args+=['--no-render']
    if s['kind']=='fixedE':args+=['--fixed-e',*designs]
    return args


def fingerprint():return hashlib.sha256((OUT/'conditions.json').read_bytes()).hexdigest()


def worker(stage):
    os.chdir(ROOT);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
    import jax
    jax.config.update('jax_compilation_cache_dir',str(ROOT/'.runtime/cache/v13_tasks_jax'))
    jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
    c=json.loads((OUT/'conditions.json').read_text())
    sys.path[:0]=[str(ROOT/'code/tt_dynamics'),str(ROOT/'code/motion_tracking')]
    args=command(stage,c['designs']);sys.argv=args
    print('STAGE',stage,flush=True)
    runpy.run_path(args[0],run_name='__main__')


def prepare():
    designs=sorted(k for k in json.loads((ROOT/'runs/pipeline_v1/retargeting/designs.json').read_text()) if k.startswith('e') and k!='e003')
    assert len(designs)==16
    files=[ROOT/'code/tt_dynamics'/f for f in ('generate_task.py','joint_diffusion_guidance.py','shared_e_guidance.py','task_reward.py','tt_model.py','initial_geometry_gate.py','s0_geometry.py','rollout_data.py')]
    files += [ROOT/'code/motion_tracking'/f for f in ('pd_replay.py','track_env.py')]
    files += [V12/'tt_joint/checkpoint.pt',V13/'tt_joint/checkpoint.pt',ROOT/'runs/pipeline_v1/retargeting/designs.json',Path(__file__)]
    files += [OUT/'summarize_tasks.py']
    baseline={}
    for stage in ['smoke_zigzag_joint_c16']+[f'{kind}_{task}_4s' for task in ('zigzag','circle') for kind in ('fixedE','joint')]:
        folder=V12/'evaluation/check_w1'/stage
        for name in ('summary.json','scale_0.npz' if stage.startswith('smoke') else 'scale_10.npz'):
            p=folder/name;files.append(p);baseline[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    c={'models':{'v12':str(V12.relative_to(ROOT)),'v13':str(V13.relative_to(ROOT))},'designs':designs,'seed':0,'kp':200,
       'stages':{s:spec(s) for s in STAGES},'new_mjx_candidate_count':368,
       'initialization':'E-derived home1s state, --anchor-s0 --joint-full-guidance --e-rest-s0 --no-s0-align; no replanning or geometry projection',
       'reward':'unchanged TrackReward/ZigzagReward defaults and scale10; smoke scale0',
       'historical_gate':'predicted morphology effect permutation p<.05, all16-design predicted/actual mean reward Spearman>.6, unguided 2s median path error<.15m',
       'rank_missing_data':'retain historical NaN->undefined failure when any design lacks finite reward; separate survivor-only rank reported as secondary if useful',
       'long_horizon':'10s evaluated regardless of 4s gate; report full-horizon survival, reward, signed progress and lane adherence; no new task-success threshold invented',
       'baseline_4s_reused':baseline,'baseline_10s':'new v12 joint16 per task with identical generation/execution settings; no extra fixed-E v12 10s grid',
       'resources':'serial CPU2, Torch GPU25%, JAX pool40%, preallocationfalse; RAM>=4GiB/GPUfree>=1GiB/rootfree>=1GiB/homefree>=8GiB; temp/cache on /home',
       'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'conditions.json').write_text(json.dumps(c,indent=2,ensure_ascii=False))


def main():
    os.chdir(ROOT)
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2]);return
    if not (OUT/'conditions.json').exists():prepare()
    c=json.loads((OUT/'conditions.json').read_text())
    for p,h in c['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.40',TMPDIR=str(ROOT/'.runtime/tmp'),
             MPLCONFIGDIR=str(ROOT/'.runtime/mpl'),XDG_CACHE_HOME=str(ROOT/'.runtime/cache'))
    env.pop('JAX_PLATFORMS',None);env.pop('XLA_PYTHON_CLIENT_ALLOCATOR',None)
    path=OUT/'status.json';status=json.loads(path.read_text()) if path.exists() else {}
    for stage in STAGES:
        if status.get(stage,{}).get('status')=='complete':
            assert status[stage]['conditions_sha256']==fingerprint();continue
        if status.get(stage,{}).get('status')=='running' and (Path('/proc')/str(status[stage]['pid'])).exists():raise RuntimeError('Stage already running')
        ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        assert ram>=8*2**30
        with (OUT/(stage+'.log')).open('a') as log,(OUT/'resources.jsonl').open('a') as resources:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',stage],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status[stage]={'status':'running','pid':child.pid,'started':time.time(),'conditions_sha256':fingerprint()};path.write_text(json.dumps(status,indent=2));low=0
            while child.poll() is None:
                ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
                gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                vals=gpu.stdout.strip().split(',');free=float(vals[1]) if len(vals)>=2 else 0
                disk={p:shutil.disk_usage(p).free/2**30 for p in ('/','/home')}
                resources.write(json.dumps({'stage':stage,'time':time.time(),'ram_available_gib':ram/2**30,'gpu':gpu.stdout.strip(),'disk_free_gib':disk})+'\n');resources.flush()
                low=low+1 if ram<4*2**30 or free<1024 or disk['/']<1 or disk['/home']<8 else 0
                if low>=2:
                    status[stage]['stop_reason']='resource threshold';os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(10)
            status[stage].update(status='complete' if child.returncode==0 else 'stopped',returncode=child.returncode,ended=time.time());path.write_text(json.dumps(status,indent=2))
            print(stage,status[stage]['status'],flush=True)
            if child.returncode:raise SystemExit(child.returncode)
        # Only compiled duplicates from this new stage; JSON definitions and trajectories are preserved.
        for model in (OUT/stage/'designs').glob('*.mjb'):
            assert model.with_suffix('.json').is_file();model.unlink()
    subprocess.run([sys.executable,str(OUT/'summarize_tasks.py')],env=env,cwd=ROOT,check=True)
    print('ALL_TASKS_COMPLETE',flush=True)


if __name__=='__main__':main()
