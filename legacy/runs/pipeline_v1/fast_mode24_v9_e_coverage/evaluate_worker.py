"""Sequential v9 evaluation, one bounded GPU process per stage."""
import json, os, runpy, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RUN=Path(__file__).resolve().parent
os.chdir(ROOT)
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:]);os.nice(10)
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(.25,0)
sys.path[:0]=[str(ROOT/'code/tt_dynamics'),str(ROOT/'code/motion_tracking')]
import jax
jax.config.update('jax_compilation_cache_dir', str(ROOT/'jax_cache'))
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
stage=sys.argv[1]
R=str(RUN.relative_to(ROOT));O=R+'/evaluation_local';V8='runs/pipeline_v1/fast_mode24_v8_film/A/tt_joint'
if stage in ['held','seen']:
 args=[str(RUN/'measure_given_a.py'),f'{O}/given_a_{stage}.json',json.dumps({'v8':[V8,1],'v9':[R+'/tt_joint',1]})]
 if stage=='held':args.append(R+'/data_test')
 args+=['--only-requested','--pair-checkpoint',R+'/tt_joint','--metric-reference',V8]
elif stage=='sensitivity':
 args=['code/tt_dynamics/e_sensitivity.py','--ckpt',R+'/tt_joint','--out',O+'/sensitivity.json']
else:
 base=['code/tt_dynamics/generate_task.py','--ckpt',R+'/tt_joint','--expert-config','runs/pipeline_v1/experts_wide16/walkForwardSlow__r00','--anchor-s0','--joint-full-guidance','--e-rest-s0','--no-s0-align']
 if stage=='smoke':args=base+['--no-render','--out',O+'/check_w1/smoke_zigzag_joint_c16','--seconds','2','--candidates','16','--task','zigzag','--scales','0']
 else:
  kind,task=stage.split('_',1)
  args=base+['--out',O+f'/check_w1/{kind}_{task}_4s','--seconds','4','--scales','10','--task',task]
  if kind=='fixedE':
   designs=json.loads(Path('runs/pipeline_v1/retargeting/designs.json').read_text())
   args+=['--no-render','--candidates','4','--fixed-e',*[k for k in designs if k.startswith('e') and k!='e003']]
  else:args+=['--candidates','16']
sys.argv=args
print('STAGE',stage,flush=True)
runpy.run_path(args[0],run_name='__main__')
