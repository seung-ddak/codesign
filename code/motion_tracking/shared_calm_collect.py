"""Collect actual shared-policy transitions, with E observations and two distinct starts."""
import argparse,functools,sys,time
from pathlib import Path
import numpy as np
from shared_calm_common import ROOT,read,save,event,choose_undercovered_start
sys.path.insert(0,str(ROOT.parent/'transformer-transformer'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--design',required=True);p.add_argument('--variant',type=int,choices=[0,1],required=True);p.add_argument('--eval',action='store_true');a=p.parse_args();out=a.out.resolve()
    import jax,jax.numpy as jnp
    from omegaconf import OmegaConf
    from policy_io import load_run
    from pd_replay import build_model
    from track_env import TrackMjxEnv
    from collect_pd_rollouts import save_rollout
    from collect_tracking_dynamics import merge_batches
    plan=read(out/'plan.json');c=plan['config'];design=next(r for r in plan['designs']+plan['eval_designs'] if r['design']==a.design);ids=plan['assignments'][a.design];names=['calm_%05d'%i for i in ids]
    checkpoint=read(out/'checkpoint.json');cfg,policy=load_run(Path(checkpoint['path']));ecfg=OmegaConf.to_container(cfg.eval_env,resolve=True);ecfg.pop('_target_')
    ecfg.update(reference_dir=str(out/'references'/a.design/plan['topology']),clips=names,max_episode_length=601,reset_qpos=.01 if a.variant else 0.,reset_qvel=.05 if a.variant else 0.)
    env=TrackMjxEnv(mj_model_fn=functools.partial(build_model,choice=plan['topology'],uniforms=design['E'],torque_limit=33.5),**ecfg)
    e=jnp.asarray(design['E'],dtype=jnp.float32);starts=np.asarray(env.clip_start);ends=np.asarray(env.clip_end);T=int((ends-starts-1).max());batch=c['rollout_batch']
    reset=jax.jit(jax.vmap(env.reset_with_traj))
    @jax.jit
    def step(s,keys):
        obs=jnp.concatenate((s.obs,jnp.broadcast_to(e,(batch,13))),axis=1)
        actions=jax.vmap(lambda x,k:policy(x,k)[0])(obs,keys)
        return jax.vmap(env.step)(s,actions),actions
    floor=int(env.mj_model.geom('floor').id)
    def contacts(data):
        geom,dist=data._impl.contact.geom,data._impl.contact.dist
        def one(f):return (((geom[:,0]==f)&(geom[:,1]==floor)|((geom[:,1]==f)&(geom[:,0]==floor)))&(dist<=.001)).any()
        return jax.vmap(one)(env._foot_ids)
    contact=jax.jit(jax.vmap(contacts));rng=np.random.default_rng(c['seed']+int(a.design[1:])*7+a.variant);pieces=[]
    episode_keys={'qpos','qvel','contact','action','target_q','valid','clip','ref_frame','source_motion_id','group_id','motions','start_frame','fell','termination','tracking_joint_rmse_rad','tracking_root_rmse_m','tracking_pass'}
    hashes={r['id']:r['sha256'] for r in plan['motions']};metadata={r['id']:r for r in plan['motions']};previous={}
    for path in (out/'rollouts').glob('*/variant_*.json'):
        for i,start,end in read(path,{}).get('source_intervals',[]):previous.setdefault(int(i),[]).append((int(start),int(end)))
    for offset in range(0,len(ids),batch):
        selected=np.arange(offset,min(offset+batch,len(ids)));count=len(selected);selected=np.pad(selected,(0,batch-count),mode='edge');length=ends[selected]-starts[selected]-1
        rel=np.array([choose_undercovered_start(metadata[ids[int(i)]],previous.get(ids[int(i)],[]),rng) if a.variant else 0 for i in selected]);frames=starts[selected]+rel;length=length-rel
        s=reset(jax.random.split(jax.random.PRNGKey(int(rng.integers(2**30))),batch),jnp.asarray(selected),jnp.asarray(frames))
        rec={k:[] for k in ['qpos','qvel','contact','action','target_q','valid','clip','ref_frame']}
        for k,v in [('qpos',s.data.qpos),('qvel',s.data.qvel),('contact',contact(s.data))]:rec[k].append(np.asarray(v))
        alive=np.ones(batch,bool);fell=np.zeros(batch,bool);termination=np.full(batch,'reference_end',dtype='<U20');joint=np.zeros(batch);root=np.zeros(batch);counts=np.zeros(batch);key=jax.random.PRNGKey(c['seed']+offset+a.variant)
        for t in range(T):
            valid=alive&(t<length)
            frame=np.asarray(s.info['frame']);key,sub=jax.random.split(key)
            if valid.any():
                s,action=step(s,jax.random.split(sub,batch));q=np.asarray(s.data.qpos);v=np.asarray(s.data.qvel);action=np.asarray(action);touch=np.asarray(contact(s.data))
                finite=np.isfinite(q).all(1)&np.isfinite(v).all(1)&np.isfinite(action).all(1)
                termination[valid&~finite]='nonfinite';valid &= finite
                falling=valid&(np.asarray(s.info['done/fellover'])!=0);fell|=falling;termination[falling]='fell'
                joint+=np.where(valid,np.asarray(s.metrics['err/pose'])/env.n_ctrl,0);root+=np.where(valid,np.asarray(s.metrics['err/root_pos']),0);counts+=valid
                alive=valid&~falling
            else:
                q=rec['qpos'][-1];v=rec['qvel'][-1];touch=rec['contact'][-1];action=np.zeros((batch,12));alive[:]=False
            applied=np.clip(action,*env.clip_actions);target=np.asarray(s.data.ctrl)[:,:12]
            for k,value in [('qpos',q),('qvel',v),('contact',touch),('action',applied),('target_q',target),('valid',valid),('clip',selected),('ref_frame',frame)]:rec[k].append(np.nan_to_num(np.asarray(value)))
        data={k:np.stack(values,axis=1) for k,values in rec.items()};batchids=np.array(ids)[selected]
        data.update(source_motion_id=batchids,group_id=np.array([hashes[int(i)] for i in batchids]),motions=np.array(names)[selected],start_frame=rel,fell=fell,termination=termination,
                    tracking_joint_rmse_rad=np.sqrt(joint/np.maximum(counts,1)),tracking_root_rmse_m=np.sqrt(root/np.maximum(counts,1)),E=np.array(design['E']),dt=1/30,design=a.design,mode='shared_ppo_reference',start_variant=a.variant,checkpoint=checkpoint['path'],policy_step=checkpoint['step'],action_reference=env.action_reference)
        data['tracking_pass']=(data['valid'].sum(1)==length)&~fell&(np.rad2deg(data['tracking_joint_rmse_rad'])<=5)&(data['tracking_root_rmse_m']<=.03)
        pieces.append({k:v[:count] if k in episode_keys else v for k,v in data.items()})
        print(f'{a.design} 시작조건 {a.variant}: {min(offset+batch,len(ids))}/{len(ids)} 완료',flush=True)
    data=merge_batches(pieces,episode_keys);folder=out/('evaluation' if a.eval else 'rollouts')/a.design;folder.mkdir(parents=True,exist_ok=True);path=folder/f'variant_{a.variant}.npz';save_rollout(path,data)
    lengths=ends-starts-1-data['start_frame'];valid=data['valid'];finished=(valid.sum(1)==lengths)&~data['fell'];q=data['qpos'][:,:-1];v=data['qvel'][:,:-1]
    def hist(values,edges):
        values=np.asarray(values)[valid];return np.histogram(values,bins=edges)[0].tolist()
    orientation=data['qpos'][:,:,3:7];w,x,y,zz=np.moveaxis(orientation,-1,0)
    yaw=np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz));delta=np.diff(yaw,axis=1);yaw_rate=np.arctan2(np.sin(delta),np.cos(delta))*30
    small_change=(np.linalg.norm(v[:,:,:2],axis=-1)<.03)&(np.abs(yaw_rate)<np.deg2rad(5))&(np.abs(v[:,:,2])<.02)
    primitive_counts=np.zeros(12,dtype=int);intervals=[]
    for i,source in enumerate(ids):
        start=int(data['start_frame'][i]);end=start+int(valid[i].sum());intervals.append([int(source),start,end])
        for segment in metadata[source].get('segments',[]):primitive_counts[segment['index']]+=max(0,min(end,segment['end'])-max(start,segment['start']))
    summary=dict(primitive_counts=primitive_counts.tolist(),source_intervals=intervals,small_change_fraction=float(small_change[valid].mean()) if valid.any() else None,design=a.design,variant=a.variant,episodes=len(ids),falls=int(data['fell'].sum()),fall_fraction=float(data['fell'].mean()),collection_pass=bool(data['fell'].mean()<=.5),complete_episodes=int(finished.sum()),valid_transitions=int(valid.sum()),
                 source_ids=[int(i) for i,yes in zip(ids,valid.sum(1)>=2) if yes],tracking_pass_count=int(data['tracking_pass'].sum()),joint_rmse_deg=float(np.rad2deg(np.median(data['tracking_joint_rmse_rad']))),root_rmse_m=float(np.median(data['tracking_root_rmse_m'])),
                 distributions={'speed':hist(np.linalg.norm(v[:,:,:2],axis=-1),np.linspace(0,3,25)),'height':hist(q[:,:,2],np.linspace(0,.8,25)),'yaw_rate':hist(np.abs(yaw_rate),np.linspace(0,6,25)),'joint_speed':hist(np.abs(v[:,:,6:]),np.linspace(0,60,25)),'action':hist(data['action'],np.linspace(-10,10,25)),'contact_count':hist(data['contact'][:,:-1].sum(-1),np.arange(-.5,5.5,1))},file=str(path),time=time.time())
    save(path.with_suffix('.json'),summary);event(out,'rollout_complete',**summary)

if __name__=='__main__':main()
