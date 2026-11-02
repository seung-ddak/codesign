"""Train the requested six-group objective with the unchanged v12 data and model settings."""
import hashlib
import json
import os
from pathlib import Path
import sys
import runpy

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    c=json.loads((OUT/'conditions.json').read_text())
    for group in ('sha256','implementation_sha256'):
        for p,h in c[group].items():
            if group=='sha256' and not p.startswith('runs/pipeline_v1/pd_rollouts_kp200/'):continue
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    if (OUT/'tt_joint/checkpoint.pt').exists():raise FileExistsError('Existing checkpoint; use a new output folder to retrain')
    for key,sub in (('TMPDIR','tmp'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache')):
        path=ROOT/'.runtime'/sub;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Run in a terminal with GPU access')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.60)
    script=ROOT/'code/tt_dynamics/train_tt.py'
    sys.path.insert(0,str(script.parent))
    sys.argv=[str(script),*c['training_args']]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
