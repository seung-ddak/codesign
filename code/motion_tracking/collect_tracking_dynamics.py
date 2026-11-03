"""Execute a trained tracking policy once per assigned motion, preserving failures."""
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent/'transformer-transformer'))


def main():
    import functools,jax,jax.numpy as jnp
    from omegaconf import OmegaConf
    from pd_replay import build_model
    from policy_io import load_run
    from track_env import TrackMjxEnv
    from collect_pd_rollouts import save_rollout,episode_group_ids
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--design',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.manifest.read_text());design=next(e for e in m['designs'] if e['design']==a.design)
    ids=design['common']+design['extra'];motions=['calm_%05d'%i for i in ids]
    cfg,policy=load_run(a.checkpoint);ecfg=OmegaConf.to_container(cfg.eval_env,resolve=True);ecfg.pop('_target_')
    ecfg.update(reference_dir=str(a.references.resolve()),clips=motions,switch_prob=0.,max_episode_length=1001,
                velocity_kick=[0.,0.],reset_qpos=0.,reset_qvel=0.,obs_noise={k:0. for k in ecfg.get('obs_noise',{})})
    model_fn=functools.partial(build_model,choice=m['topology'],uniforms=design['E'],kp=cfg.get('kp'),tau_scale=cfg.get('tau_scale'),torque_limit=cfg.get('torque_limit'))
    for name in motions:
        with np.load(a.references/(name+'.npz')) as ref:
            if not np.allclose(ref['uniforms'],design['E'],rtol=0,atol=1e-7):raise ValueError('reference morphology mismatch')
    env=TrackMjxEnv(mj_model_fn=model_fn,**ecfg);b=len(ids);clips=np.arange(b);starts=np.asarray(env.clip_start);ends=np.asarray(env.clip_end);lengths=ends-starts-1;T=int(lengths.max());n=env.n_ctrl
    reset=jax.jit(jax.vmap(env.reset_with_traj));s=reset(jax.random.split(jax.random.PRNGKey(0),b),jnp.asarray(clips),jnp.asarray(starts))
    floor=int(env.mj_model.geom('floor').id);feet=env._foot_ids
    def contacts(data):
        geom,dist=data._impl.contact.geom,data._impl.contact.dist
        def touches(f):
            pair=((geom[:,0]==f)&(geom[:,1]==floor))|((geom[:,1]==f)&(geom[:,0]==floor))
            return (pair&(dist<=1e-3)).any()
        return jax.vmap(touches)(feet)
    contact=jax.jit(jax.vmap(contacts))
    @jax.jit
    def step(state,keys):
        action=jax.vmap(lambda obs,key:policy(obs,key)[0])(state.obs,keys)
        return jax.vmap(env.step)(state,action),action
    rec={k:[] for k in ['qpos','qvel','contact','action','target_q','valid','clip','ref_frame']}
    for k,v in [('qpos',s.data.qpos),('qvel',s.data.qvel),('contact',contact(s.data))]:rec[k].append(np.asarray(v))
    alive=np.ones(b,bool);fell=np.zeros(b,bool);termination=np.full(b,'reference_end',dtype='<U20');joint_error=np.zeros(b);root_error=np.zeros(b);counts=np.zeros(b);rng=jax.random.PRNGKey(1)
    for t in range(T):
        valid=alive&(t<lengths);frame=np.asarray(s.info['frame']);rng,key=jax.random.split(rng)
        s,action=step(s,jax.random.split(key,b));q=np.asarray(s.data.qpos);v=np.asarray(s.data.qvel);action=np.asarray(action)
        if not np.isfinite(q[valid]).all() or not np.isfinite(v[valid]).all() or not np.isfinite(action[valid]).all():raise FloatingPointError('nonfinite valid transition; no rollout saved')
        applied=np.clip(action,*env.clip_actions);target=np.asarray(env.action_offset[:n])+applied*np.asarray(env.action_scale)
        for k,value in [('qpos',q),('qvel',v),('contact',contact(s.data)),('action',applied),('target_q',target),('valid',valid),('clip',clips),('ref_frame',frame)]:rec[k].append(np.nan_to_num(np.asarray(value)))
        joint_error+=np.where(valid,np.asarray(s.metrics['err/pose'])/n,0)
        root_error+=np.where(valid,np.asarray(s.metrics['err/root_pos']),0);counts+=valid
        falling=valid&(np.asarray(s.info['done/fellover'])!=0);fell|=falling;termination[falling]='fell'
        alive=valid&~falling
    data={k:np.stack(v,axis=1) for k,v in rec.items()}
    data.update(E=np.asarray(design['E']),motions=np.array(motions),dt=env.dt,design=a.design,mode='ppo_reference',
                group_id=episode_group_ids([next(item['sha256'] for item in m['motions'] if item['id']==i)+':'+name for i,name in zip(ids,motions)],np.zeros(b,dtype=int)),start_frame=np.zeros(b,dtype=int),source_motion_id=np.array(ids),
                common=np.array([i in design['common'] for i in ids]),tracking_joint_rmse_rad=np.sqrt(joint_error/np.maximum(counts,1)),tracking_root_rmse_m=np.sqrt(root_error/np.maximum(counts,1)),fell=fell,termination=termination,checkpoint=str(a.checkpoint.resolve()))
    assert all(data[k].shape[:2]==(b,T+1) for k in ['qpos','qvel','contact'])
    assert all(data[k].shape[:2]==(b,T) for k in ['action','target_q','valid'])
    complete=(data['valid'].sum(1)==lengths)&~fell
    data['tracking_pass']=complete&(np.rad2deg(data['tracking_joint_rmse_rad'])<=5)&(data['tracking_root_rmse_m']<=.03)
    save_rollout(a.out,data)
    summary=dict(design=a.design,episodes=b,complete_episodes=int(complete.sum()),falls=int(fell.sum()),valid_transitions=int(data['valid'].sum()),tracking_pass_count=int(data['tracking_pass'].sum()),tracking_pass_fraction=float(data['tracking_pass'].mean()),
                 median_joint_rmse_deg=float(np.rad2deg(np.median(data['tracking_joint_rmse_rad']))),median_root_rmse_m=float(np.median(data['tracking_root_rmse_m'])),
                 note='Actual policy execution; completion is not precise motion tracking success. Failed episodes remain for dynamics learning.')
    a.out.with_suffix('.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':main()
