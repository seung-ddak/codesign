"""Run comparison and MJX stages serially while guarding system headroom."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RUN=Path(__file__).resolve().parent;OUT=RUN/'evaluation_local'
OUT.mkdir(exist_ok=True)
env=dict(os.environ,MUJOCO_GL='egl',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.40',TMPDIR=str(ROOT/'.runtime/tmp'),MPLCONFIGDIR=str(ROOT/'.runtime/mpl'))
env.pop('XLA_PYTHON_CLIENT_ALLOCATOR',None)
status_path=OUT/'status.json'
status=json.loads(status_path.read_text()) if status_path.exists() else {}
for stage in ['held','seen','sensitivity','smoke','fixedE_zigzag','joint_zigzag','fixedE_circle','joint_circle']:
 if status.get(stage,{}).get('status')=='completed':continue
 with (OUT/(stage+'.log')).open('a') as log:
  child=subprocess.Popen([sys.executable,str(RUN/'evaluate_worker.py'),stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status[stage]=dict(status='running',pid=child.pid,started=time.time());status_path.write_text(json.dumps(status,indent=2))
  low=0
  with (OUT/'resources.jsonl').open('a') as resources:
   while child.poll() is None:
    ram=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    vals=gpu.stdout.strip().split(',');free=float(vals[1]) if len(vals)>=2 else 0
    resources.write(json.dumps(dict(stage=stage,time=time.time(),ram_available_gib=ram/1024**3,gpu=gpu.stdout.strip()))+'\n');resources.flush()
    low=low+1 if ram<4*1024**3 or free<1024 else 0
    if low>=2:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     break
    time.sleep(10)
  status[stage].update(status='completed' if child.returncode==0 else 'stopped',returncode=child.returncode,ended=time.time());status_path.write_text(json.dumps(status,indent=2))
  if child.returncode:raise SystemExit(child.returncode)
print('EVALUATION_STAGES_COMPLETE',flush=True)
