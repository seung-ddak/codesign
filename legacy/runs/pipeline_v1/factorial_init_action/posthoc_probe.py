"""Post-hoc replay validity check, e000 only; does not change the primary factorial gate."""
import json
import os
from pathlib import Path
import pickle
import sys
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def main():
    sys.path[:0]=[str(ROOT/'code/motion_tracking'),str(ROOT.parent/'transformer-transformer')]
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx
    from omegaconf import OmegaConf
    from pd_replay import build_model
    from track_env import TrackMjxEnv
    from t2.mjx_env.base_env import make_data,step
    c=json.loads((OUT/'conditions.json').read_text())
    with (ROOT/c['config']).open('rb') as f:cfg=pickle.load(f)
    bank=dict(np.load(OUT/'action_bank.npz'))
    original=ROOT/'runs/pipeline_v1/wide16_action/rollouts/e000/primitive_seed0.npz'
    with np.load(original) as z:
        trueq=np.stack([z['qpos'][row,t:t+51] for row,t in zip(bank['source_episode'],bank['source_t'])])
        truev=np.stack([z['qvel'][row,t:t+51] for row,t in zip(bank['source_episode'],bank['source_t'])])
        assert np.allclose(z['E'],bank['E'][0],atol=1e-7)
    with np.load(OUT/'rollouts/e000/factorial.npz') as z:
        q0=np.stack([trueq[:,0],z['qpos'][0,1,:,0],z['qpos'][1,1,:,0]])
        v0=np.stack([truev[:,0],z['qvel'][0,1,:,0],z['qvel'][1,1,:,0]])
    protocol={'label':'post-hoc, after primary common survivor count=0', 'scope':'e000, same 76 PPO action sequences; actual source state vs reference vs home1s state; kp30 vs200',
              'diagnostic':'trajectory replay errors and survival; no new regression gate, no claim about all-shape generalization',
              'limitation':'only qpos/qvel are restored; source solver warm-start state is unavailable',
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),OUT/'action_bank.npz',original)}}
    (OUT/'posthoc_probe_conditions.json').write_text(json.dumps(protocol,indent=2))
    ecfg=OmegaConf.to_container(cfg.eval_env,resolve=True);ecfg.pop('_target_')
    ecfg.update(impl='jax',reference_dir=str(ROOT/'runs/pipeline_v1/retargeting/e000'/c['choice']),clips=bank['motions'].tolist(),clip_boost=None,switch_prob=0.)
    G=c['groups'];B=3*G
    forward=jax.jit(jax.vmap(mjx.forward,in_axes=(None,0)))
    nsub=round(ecfg['ctrl_dt']/ecfg['sim_dt'])
    vstep=jax.jit(lambda model,data,u:jax.vmap(lambda d,a:step(model,d,a,nsub))(data,u))
    result={'label':'post-hoc','kp':{}}
    saved={}
    for kp in (30,200):
        env=TrackMjxEnv(mj_model_fn=lambda:build_model(c['choice'],kp=kp,tau_scale=cfg.get('tau_scale'),torque_limit=cfg.get('torque_limit'),uniforms=bank['E'][0]),**ecfg)
        template=make_data(env.mj_model,impl='jax',nconmax=env._nconmax,njmax=env._njmax)
        data=jax.tree.map(lambda x:jnp.broadcast_to(x,(B,*x.shape)),template)
        data=data.replace(qpos=jnp.asarray(q0.reshape(B,19)),qvel=jnp.asarray(v0.reshape(B,18)))
        data=forward(env.mjx_model,data)
        traj=[];alive=np.ones(B,bool);off=np.asarray(env.action_offset)
        for t in range(51):
            q=np.asarray(data.qpos)
            assert np.isfinite(q).all()
            alive &= 1-2*(q[:,4]**2+q[:,5]**2)>=env.termination_fellover_threshold
            traj.append(q)
            if t<50:
                u=np.broadcast_to(off,(B,len(off))).copy();u[:,:12]=np.tile(bank['target_q'][1,:,t],(3,1))
                data=vstep(env.mjx_model,data,jnp.asarray(u))
        q=np.stack(traj,1).reshape(3,G,51,19)
        diff=q[0]-trueq
        result['kp'][str(kp)]={'survived':dict(zip(('actual_PPO_state','reference_state','home1s_state'),map(int,alive.reshape(3,G).sum(1)))),
                              'episodes_per_state':G,'actual_state_replay':{'first_step_joint_rmse_rad':float(np.sqrt((diff[:,1,7:]**2).mean())),
                              'all_step_joint_rmse_rad':float(np.sqrt((diff[:,:,7:]**2).mean())),
                              'end_xy_error_median_m':float(np.median(np.linalg.norm(diff[:,-1,:2],axis=1)))}}
        saved['kp'+str(kp)+'_qpos']=q
        print(kp,result['kp'][str(kp)],flush=True)
    np.savez_compressed(OUT/'posthoc_probe_trajectories.npz',source_qpos=trueq,**saved)
    (OUT/'posthoc_probe_results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
