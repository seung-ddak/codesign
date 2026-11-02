"""Run the approved training once, keeping resource headroom and durable status."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    status_path=OUT/'status.json'
    if status_path.exists():
        old=json.loads(status_path.read_text())
        if old.get('status')=='complete':raise RuntimeError('Training is already complete')
        if old.get('status')=='running' and (Path('/proc')/str(old['pid'])).exists():raise RuntimeError('Training already running')
    ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert ram>=8*2**30,'RAM headroom below 8GiB'
    env=dict(os.environ,TMPDIR=str(ROOT/'.runtime/tmp'),MPLCONFIGDIR=str(ROOT/'.runtime/mpl'),XDG_CACHE_HOME=str(ROOT/'.runtime/cache'),
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0')
    with (OUT/'train.log').open('a') as log,(OUT/'resources.jsonl').open('a') as resources:
        child=subprocess.Popen([sys.executable,str(OUT/'run_train.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status={'status':'running','pid':child.pid,'started':time.time()}
        status_path.write_text(json.dumps(status,indent=2)); low=0
        while child.poll() is None:
            ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            vals=gpu.stdout.strip().split(',');free=float(vals[1]) if len(vals)>=2 else 0
            disk={p:shutil.disk_usage(p).free/2**30 for p in ('/','/home')}
            resources.write(json.dumps({'time':time.time(),'ram_available_gib':ram/2**30,'gpu':gpu.stdout.strip(),'disk_free_gib':disk})+'\n');resources.flush()
            low=low+1 if ram<4*2**30 or free<1024 or disk['/']<1 or disk['/home']<8 else 0
            if low>=2:
                status['stop_reason']='resource headroom threshold';os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            time.sleep(10)
        status.update(status='complete' if child.returncode==0 else 'stopped',returncode=child.returncode,ended=time.time())
        status_path.write_text(json.dumps(status,indent=2))
        print(json.dumps(status),flush=True)
        if child.returncode:raise SystemExit(child.returncode)


if __name__=='__main__':main()
