import sys,json
from pathlib import Path
import numpy as np
root=Path('/home/cgr-ugrad-2026/quad_ws/codesign');sys.path.insert(0,str(root/'code/motion_tracking'))
import calm_pilot as pilot
import pd_replay as pd
base=root/'runs/calm_pilot_retarget/try1';selection=json.loads((base/'selection.json').read_text())
for hz in (50,30):
 out=root/f'runs/calm_pilot_no_foot_lock_{hz}hz_try1';out.mkdir(exist_ok=False)
 refs=out/'a1_reference';refs.mkdir()
 for name in selection['clips']:
  if hz==50:
   import shutil
   shutil.copyfile(base/'a1_reference'/f'{name}.npz',refs/f'{name}.npz')
  else:
   src=base/'collection'/('hlc_'+name.removeprefix('calm_'))/'trace.npz'
   with np.load(src) as d:
    pilot.validate_trace(d);names=d['joint_names'].tolist();order=[names.index(l+'_'+j+'_joint') for l in ['FR','FL','RR','RL'] for j in ['hip','thigh','calf']]
    np.savez_compressed(refs/f'{name}.npz',fix_q=d['q'][:,order],fix_base_pos=d['root'][:,:3],fix_base_quat=d['root'][:,[6,3,4,5]],fix_foot_pos=d['foot_pos']-[0,0,.02],fix_contact=d['foot_floor_contact'],dt=1/30)
 source=pilot.calm_source_model(out/'source_model');build=pd.build_model;lock=pd.foot_lock
 def model(choice=None,**kwargs):return source if choice is None else build(choice,**kwargs)
 def unlocked(*args,**kwargs):
  kwargs['lock']=False
  return lock(*args,**kwargs)
 pd.build_model=model;pd.foot_lock=unlocked
 try:pd.retarget_tt(str(refs),str(out/'references/e000'),choices=['00000011'],uniforms=[.5]*13,torque_limit=33.5)
 finally:pd.build_model=build;pd.foot_lock=lock
 target=build('00000011',uniforms=[.5]*13,torque_limit=33.5);lo,hi=target.jnt_range[1:].T;rows=[]
 for name in selection['clips']:
  d=np.load(out/'references/e000/00000011'/f'{name}.npz');q=d['fix_q'];excess=max(0,float(np.maximum(lo-q,q-hi).max()));err=float(d['fix_track_err']);bad=float(d['inv_bad_frac'])
  assert len(q)==hz*10+1 and np.isfinite(q).all() and np.isclose(float(d['dt']),1/hz)
  rows.append(dict(clip=name,frames=len(q),foot_error_m=err,joint_limit_excess_rad=excess,inverse_bad_fraction=bad,pass_pilot=bool(err<=.03 and excess<=.01 and bad<=.1)))
 (out/'retarget_metrics.json').write_text(json.dumps(rows,indent=2));(out/'experiment.json').write_text(json.dumps(dict(hz=hz,foot_lock=False,notes='Contact cleanup, body-height smoothing and geometry retargeting retained. No TT forward simulation.',metrics=rows),indent=2))
 print('COMPLETE',hz,rows,flush=True)
