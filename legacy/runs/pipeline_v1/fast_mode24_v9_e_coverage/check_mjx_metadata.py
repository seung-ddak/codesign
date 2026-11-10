"""Check that sharing MJX compile-time metadata preserves the physical step."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
os.chdir(ROOT);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
sys.path[:0]=[str(ROOT/'code/motion_tracking'),str(ROOT.parent/'transformer-transformer')]
import numpy as np,mujoco,jax
from mujoco import mjx
from pd_replay import build_model
m=build_model('00000011',uniforms=[.5]*13,torque_limit=33.5)
from types import SimpleNamespace
from track_env import TrackMjxEnv
TrackMjxEnv.preprocess_mj_model(SimpleNamespace(impratio=1,sim_dt=.005,solver_iterations=5,solver_ls_iterations=7,n_ctrl=12,soft_joint_pos_limit_factor=.95),m)
d=mujoco.MjData(m)
z=np.load('runs/pipeline_v1/retargeting/e000/00000011/walkForwardSlow__r00.npz')
d.qpos[:]=np.r_[z['fix_base_pos'][0],z['fix_base_quat'][0],z['fix_q'][0]];d.ctrl[:]=z['fix_q'][0];mujoco.mj_forward(m,d)
model=mjx.put_model(m);data=mjx.put_data(m,d);general=model.replace(geom_sameframe=np.zeros_like(model.geom_sameframe))
f=jax.jit(mjx.step);a=f(model,data);b=f(general,data)
for name in ['qpos','qvel','geom_xpos']:
 x,y=np.asarray(getattr(a,name)),np.asarray(getattr(b,name));assert np.allclose(x,y,atol=2e-5,rtol=2e-5),(name,np.max(np.abs(x-y)))
print('MJX metadata parity: passed',flush=True)
