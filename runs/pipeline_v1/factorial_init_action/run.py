"""Serial GPU workers with a resource watchdog; never kills unrelated processes."""
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

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10)
        script=OUT/(sys.argv[2]+'.py');sys.argv=[str(script)]
        runpy.run_path(str(script),run_name='__main__');return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.40',TMPDIR=str(ROOT/'.runtime/tmp'),
             MPLCONFIGDIR=str(ROOT/'.runtime/mpl'),XDG_CACHE_HOME=str(ROOT/'.runtime/cache'))
    env.pop('JAX_PLATFORMS',None);env.pop('XLA_PYTHON_CLIENT_ALLOCATOR',None)
    env['JAX_COMPILATION_CACHE_DIR']=str(ROOT/'.runtime/cache/factorial_jax')
    for p in ('tmp','mpl','cache'):(ROOT/'.runtime'/p).mkdir(parents=True,exist_ok=True)
    if not (OUT/'conditions.json').exists():
        subprocess.run([sys.executable,str(OUT/'prepare.py')],env=env,check=True)
    c=json.loads((OUT/'conditions.json').read_text())
    for p,h in c['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    status_path=OUT/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    for stage in ('collect','evaluate'):
        if status.get(stage,{}).get('status')=='complete':continue
        ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        assert ram>=8*2**30,'Available RAM below 8GiB'
        with (OUT/(stage+'.log')).open('a') as log,(OUT/'resources.jsonl').open('a') as resources:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',stage],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status[stage]={'status':'running','pid':child.pid,'started':time.time()}
            status_path.write_text(json.dumps(status,indent=2));low=0
            while child.poll() is None:
                ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
                gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                vals=gpu.stdout.strip().split(',');free=float(vals[1]) if len(vals)>=2 else 0
                disk={p:shutil.disk_usage(p).free/2**30 for p in ('/','/home')}
                resources.write(json.dumps({'stage':stage,'time':time.time(),'ram_available_gib':ram/2**30,'gpu':gpu.stdout.strip(),'disk_free_gib':disk})+'\n');resources.flush()
                low=low+1 if ram<4*2**30 or free<1024 or disk['/']<1 or disk['/home']<8 else 0
                if low>=2:
                    status[stage]['stop_reason']='resource threshold'
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(10)
            status[stage].update(status='complete' if child.returncode==0 else 'stopped',returncode=child.returncode,ended=time.time())
            status_path.write_text(json.dumps(status,indent=2));print(stage,status[stage]['status'],flush=True)
            if child.returncode:raise SystemExit(child.returncode)
    print('ALL_STAGES_COMPLETE',flush=True)


if __name__=='__main__':main()
