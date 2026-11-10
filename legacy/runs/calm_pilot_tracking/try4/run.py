"""Sequential, bounded-memory transfer evaluation of the frozen e000 PPO."""
from pathlib import Path
import json,sys,subprocess,hashlib,os
import numpy as np
p=Path(__file__).resolve().parent;root=p.parents[2];code=root/'code/motion_tracking'
sys.path.insert(0,str(code))
import calm_pilot as pilot
import pd_replay as pd
checkpoint=root/'runs/calm_pilot_tracking/try2/wandb/offline-run-20261029_130428-utdk7d6i/files/checkpoints/5246976'
legacy=json.loads((root/'runs/pipeline_v1/retargeting/designs.json').read_text())
defs=[]
for name,label,value in [('near_short','허벅지 연장·종아리 단축',.4),('near_long','허벅지 단축·종아리 연장',.6)]:
 e=[.5]*13
 for j in [3,4,9,10]:e[j]=value
 defs.append(dict(id=name,label=label,E=e))
e=[.5]*13;e[1]=.6;e[2]=.6;defs.append(dict(id='near_body',label='몸통·무게중심 변화',E=e))
for name in ['e003','e008']:
 e=legacy[name].copy()
 for j in [0,11,12]:e[j]=.5
 defs.append(dict(id=name,label='기존 분산 샘플 '+name,E=e))
(p/'designs.json').write_text(json.dumps(defs,ensure_ascii=False,indent=2))
source_dir=p/'source_model'
source=pilot.calm_source_model(source_dir)
original_build,original_lock=pd.build_model,pd.foot_lock
source_refs=root/'runs/calm_pilot_retarget/try3/a1_reference'
env=os.environ.copy();env.update(MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/mpl-calm')
progress=[]
for definition in defs:
 name=definition['id'];out=p/'evaluations'/name;refbase=p/'references'/name
 pd.build_model=lambda choice=None,**kw:source if choice is None else original_build(choice,**kw)
 def unlocked(*a,**kw):kw['lock']=False;return original_lock(*a,**kw)
 pd.foot_lock=unlocked
 try:pd.retarget_tt(str(source_refs),str(refbase),choices=['00000011'],uniforms=definition['E'],torque_limit=33.5)
 finally:pd.build_model=original_build;pd.foot_lock=original_lock
 ref=refbase/'00000011';m=original_build('00000011',uniforms=definition['E'],torque_limit=33.5)
 diagnostics=[]
 import mujoco
 for f in sorted(ref.glob('*.npz')):
  with np.load(f) as d:
   first=np.r_[d['fix_base_pos'][0],d['fix_base_quat'][0],d['fix_q'][0]];state=mujoco.MjData(m);state.qpos[:]=first;mujoco.mj_forward(m,state)
   floor=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)
   pen=max([0.]+[-float(c.dist) for c in state.contact if c.geom1 in floor or c.geom2 in floor])
   diagnostics.append(dict(motion=f.stem,foot_target_error_m=float(d['fix_track_err']),inverse_bad_fraction=float(d['inv_bad_frac']),initial_ground_penetration_m=pen,initial_up_z=float(1-2*(first[4]**2+first[5]**2))))
 (refbase/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
 cmd=[sys.executable,str(code/'eval_calm_mjx_tracking.py'),'--checkpoint',str(checkpoint),'--out',str(out),'--reference',str(ref),'--uniforms',*[str(x) for x in definition['E']]]
 with (p/(name+'_evaluation.log')).open('w') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (p/(name+'_render.log')).open('w') as log:subprocess.run([sys.executable,str(code/'render_calm_three_models.py'),'--run',str(out)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 rows=json.loads((out/'metrics.json').read_text());progress.append(dict(design=name,survived=sum(r['survived_10s'] for r in rows),tracking_pass=sum(r['tracking_error_screen_pass'] for r in rows)))
 (p/'progress.json').write_text(json.dumps(progress,indent=2));print('DONE',name,progress[-1],flush=True)
print('All transfer evaluations finished.',flush=True)
