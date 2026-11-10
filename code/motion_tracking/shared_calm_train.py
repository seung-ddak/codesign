"""One bounded shared-PPO morphology bank; carries full learner state between banks."""
import argparse,functools,json,math,os,pickle,sys,threading,time
from pathlib import Path
import numpy as np
from shared_calm_common import CODE,ROOT,read,save,event
sys.path.insert(0,str(ROOT.parent/'transformer-transformer'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',type=int,required=True);p.add_argument('--visit',type=int,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--seconds',type=float,required=True);a=p.parse_args();out=a.out.resolve()
    import hydra,jax,jax.numpy as jnp
    from omegaconf import OmegaConf
    from orbax import checkpoint as ocp
    from multi_design import build_bank,EpisodeDesignReset,MultiDesignVmapWrapper,make_fixed_e_ppo_networks
    from pd_replay import build_model
    from t2.mjx_env.train import train
    from shared_calm_common import ready_batch
    plan=read(out/'plan.json');c=plan['config'];names,evalnames=ready_batch(out,plan,a.stage)
    if not names:raise ValueError('No prepared training morphologies in batch')
    designs={r['design']:r['E'] for r in plan['designs']+plan['eval_designs']};root=out/'references';run=out/'stages'/f'{a.visit:03d}'/'files';run.mkdir(parents=True,exist_ok=True)
    envcfg=OmegaConf.load(CODE/'config/env/a1_track.yaml');envcfg.update(dict(reference_dir=str(root),clips=[],ctrl_dt=1/30,sim_dt=1/240,max_episode_length=600,w_vel=1.,action_reference=c.get('action_reference','home'),termination_pose_consecutive_steps=15,obs_time_indices=[0,1,2,4,5],reset_qpos=0.,reset_qvel=0.,velocity_kick=[0.,0.]))
    envcfg.penalty_foot_slip=float(c.get('penalty_foot_slip',0.))  # 0 = 끔 (기존과 동일)
    evalcfg=OmegaConf.create(OmegaConf.to_container(envcfg));evalcfg.update(dict(termination_pos_err_threshold=100.,termination_orn_err_threshold=100.,termination_pose_consecutive_steps=600,obs_noise={k:0. for k in envcfg.obs_noise}))
    clips={k:['calm_%05d'%i for i in ids] for k,ids in plan['assignments'].items()}
    model=functools.partial(build_model,choice=plan['topology'],torque_limit=33.5)
    env,bank=build_bank(names,envcfg,root,plan['topology'],designs,model,[0]*len(names),clips_by_design=clips)
    if evalnames:ev,evalbank=build_bank(evalnames,evalcfg,root,plan['topology'],designs,model,[0],clips_by_design=clips)
    else:ev,evalbank=env,bank
    env.design_bank=bank;ev.design_bank=evalbank
    lock=threading.Lock();coverage=read(out/'training_coverage.json',{'motions':{},'pairs':{}});last_flush=[0.]
    def note(flags,design_ids,local_clips,bank_names):
        with lock:
            for i in np.flatnonzero(flags):
                name=bank_names[int(design_ids[i])];source=plan['assignments'][name][int(local_clips[i])];key=str(source);pair=name+':'+key
                coverage['motions'][key]=coverage['motions'].get(key,0)+1;coverage['pairs'][pair]=coverage['pairs'].get(pair,0)+1
            if time.time()-last_flush[0]>10:
                save(out/'training_coverage.json',coverage);last_flush[0]=time.time()
    class CountStarts(EpisodeDesignReset):
        def reset(self,rng):
            s=super().reset(rng)
            jax.debug.callback(lambda *x:note(*x,names),jnp.ones_like(s.done),s.info['design_id'],s.info['clip'])
            return s
        def step(self,s,action):
            s=super().step(s,action)
            def emit(_):
                jax.debug.callback(lambda *x:note(*x,names),s.done,s.info['design_id'],s.info['clip']);return jnp.int32(0)
            jax.lax.cond(jnp.any(s.done),emit,lambda _:jnp.int32(0),None)
            return s
    def wrap(e,episode_length,action_repeat,randomization_fn,**kw):
        wrapper=CountStarts if e is env else EpisodeDesignReset
        return wrapper(MultiDesignVmapWrapper(e,e.design_bank),episode_length,action_repeat)
    network=functools.partial(make_fixed_e_ppo_networks,policy_hidden_layer_sizes=tuple(c.get('hidden',(256,256,128))),value_hidden_layer_sizes=tuple(c.get('hidden',(256,256,128))),activation=jax.nn.elu,state_dependent_std=False,init_noise_std=c.get('init_noise_std',1.),noise_std_type='scalar',distribution_type='normal')
    if c.get('zero_mean_init',False):
        network=functools.partial(network,mean_kernel_init_fn=functools.partial(jax.nn.initializers.constant,0.))
    cfg=OmegaConf.create(dict(eval_env=OmegaConf.to_container(evalcfg),kp=None,tau_scale=None,torque_limit=33.5,multi_design=True,choice=plan['topology'],train=dict(normalize_observations=True,network_factory={'_target_':'multi_design.make_fixed_e_ppo_networks','_partial_':True,'policy_hidden_layer_sizes':list(c.get('hidden',(256,256,128))),'value_hidden_layer_sizes':list(c.get('hidden',(256,256,128))),'activation':{'_target_':'flax.linen.elu','_partial_':True},'state_dependent_std':False,'init_noise_std':c.get('init_noise_std',1.),'noise_std_type':'scalar','distribution_type':'normal'})))
    if c.get('zero_mean_init',False):
        cfg.train.network_factory.mean_kernel_init_fn={'_target_':'jax.nn.initializers.constant','_partial_':True,'value':0.}
    (run/'config.pkl').write_bytes(pickle.dumps(cfg))
    save(out/'resolved_training_config.json',dict(env=OmegaConf.to_container(envcfg,resolve=True),eval_env=OmegaConf.to_container(evalcfg,resolve=True),policy_hidden=list(c.get('hidden',(256,256,128))),critic_hidden=list(c.get('hidden',(256,256,128))),discounting=.99,gae_lambda=.95,clipping_epsilon=.2,max_grad_norm=1.,torque_limit=33.5))
    initial=read(out/'learner.json',{}).get('step',0);last=[initial]
    def progress(step,metrics):
        # Brax's episode logger reports per-worker steps; epoch metrics report global steps.
        step=initial+int(step) if all(str(k).startswith('episode/') for k in metrics) else int(step)
        scalars={str(k):float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1}
        if not all(np.isfinite(v) for v in scalars.values()):raise FloatingPointError('Nonfinite PPO metrics')
        last[0]=max(last[0],step)
        row=dict(time=time.time(),stage=a.stage,step=step,metrics=scalars)
        with lock:
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        display={'평균 누적 보상':scalars.get('episode/sum_reward'),'초당 전이':scalars.get('training/sps'),'전체 손실':scalars.get('training/total_loss')}
        print(f'학습 전이 {step:,} / {c["train_transitions"]:,} | 묶음 {a.stage+1} | '+str({k:round(v,4) for k,v in display.items() if v is not None}),flush=True)
    def checkpoint(step,make_policy,params):
        path=run/'checkpoints'/str(step);path.parent.mkdir(exist_ok=True)
        ocp.PyTreeCheckpointer().save(str(path),params,force=True)
        save(out/'checkpoint.json',dict(step=int(step),path=str(path),stage=a.stage,time=time.time()))
    train(environment=env,eval_env=ev,num_timesteps=a.steps,num_envs=c['num_envs'],batch_size=c['batch_size'],unroll_length=c['unroll_length'],num_minibatches=c['num_minibatches'],num_updates_per_batch=c['num_updates_per_batch'],
          learning_rate=c['learning_rate'],entropy_cost=c['entropy_cost'],discounting=.99,gae_lambda=.95,clipping_epsilon=.2,max_grad_norm=1.,normalize_observations=True,
          episode_length=600,num_evals=max(2,math.ceil(a.steps/c['checkpoint_steps'])+1),num_eval_envs=1,run_evals=False,network_factory=network,wrap_env_fn=wrap,
          progress_fn=progress,policy_params_fn=checkpoint,full_state_path=str(out/'learner.msgpack'),max_walltime_seconds=a.seconds,
          log_training_metrics=True,training_metrics_steps=c.get('metrics_steps',98304),seed=c['seed']+a.visit)
    jax.effects_barrier();save(out/'training_coverage.json',coverage)
    save(run.parent/'complete.json',dict(step=read(out/'checkpoint.json')['step'],time=time.time()))
    event(out,'train_batch_complete',stage=a.stage,step=last[0])

if __name__=='__main__':main()
