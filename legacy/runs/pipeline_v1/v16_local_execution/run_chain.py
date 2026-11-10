"""Resume the approved local twin/train/eval -> delta/train/eval chain without duplicate training."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
P1=ROOT/'runs/pipeline_v1'


def train(run):
    source=P1/'fast_mode24_v13_group_loss/supervise_train.py'
    spec=importlib.util.spec_from_file_location('v16_resource_supervisor',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=run
    m.main()


def main():
    os.chdir(ROOT)
    if len(sys.argv)==3 and sys.argv[1]=='--train-worker':train(P1/sys.argv[2]);return
    env=dict(os.environ,TMPDIR=str(ROOT/'.runtime/tmp'),MPLCONFIGDIR=str(ROOT/'.runtime/mpl'),XDG_CACHE_HOME=str(ROOT/'.runtime/cache'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    env.pop('JAX_PLATFORMS',None)
    state_path=OUT/'status.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
    for label in ('twin','delta'):
        run=P1/('fast_mode24_v16_'+label)
        path=run/'status.json'
        while path.exists() and json.loads(path.read_text()).get('status')=='running':
            s=json.loads(path.read_text())
            if not (Path('/proc')/str(s['pid'])).exists():raise RuntimeError('Training status running but PID absent: '+label)
            print(label,'waiting on existing training',s['pid'],flush=True);time.sleep(30)
        if not (path.exists() and json.loads(path.read_text()).get('status')=='complete'):
            with (OUT/(label+'_train_supervisor.log')).open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--train-worker',run.name],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (run/'tt_joint/checkpoint.pt').exists()
        last=json.loads((run/'tt_joint/progress.jsonl').read_text().splitlines()[-1]);assert last['step']==60000
        state[label+'_train']={'status':'complete','completed_at':time.time()};state_path.write_text(json.dumps(state,indent=2))
        state[label+'_eval']={'status':'running','started':time.time()};state_path.write_text(json.dumps(state,indent=2))
        with (OUT/(label+'_evaluation.log')).open('a') as log:
            subprocess.run([sys.executable,str(P1/'eval_v12parity/run_eval.py'),str(run)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (run/'gate.json').exists() and (run/'report.html').exists()
        state[label+'_eval'].update(status='complete',ended=time.time());state_path.write_text(json.dumps(state,indent=2))
    print('V16_TWIN_DELTA_CHAIN_COMPLETE',flush=True)


if __name__=='__main__':main()
