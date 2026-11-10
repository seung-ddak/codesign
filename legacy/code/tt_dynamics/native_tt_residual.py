"""Per-token residual policy plugged into the unchanged original sampler step."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from torch import nn

from native_tt_data import ROOT, atomic_torch, digest, save_json
from native_tt_train import rng_state, restore_rng

sys.path.insert(0,str(ROOT.parent/'quad/rsl_rl'))
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO


def layout_for(task):
    layout=[];offset=0
    for name,adapter in sorted(task.adapters.items()):
        n=adapter.modality.max_seq_len
        if adapter.loss_weight:
            layout.append((name,offset,offset+n,adapter.modality.dim))
        offset+=n
    return layout,offset


class TokenActor(nn.Module):
    def __init__(self,layout,tokens):
        super().__init__();self.layout=layout;self.tokens=tokens
        self.heads=nn.ModuleDict({name:nn.Sequential(nn.Linear(257,64),nn.SiLU(),nn.Linear(64,d))
                                  for name,lo,hi,d in layout})
        for head in self.heads.values():nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def forward(self,x):
        x=x.reshape(len(x),self.tokens,257)
        return torch.cat([self.heads[name](x[:,lo:hi]).flatten(1)
                          for name,lo,hi,d in self.layout],1)


class TokenCritic(nn.Module):
    def __init__(self,tokens):
        super().__init__();self.tokens=tokens;self.net=nn.Sequential(nn.Linear(257,64),nn.Tanh(),nn.Linear(64,1))

    def forward(self,x):return self.net(x.reshape(len(x),self.tokens,257).mean(1))


def policy_for(task,device):
    layout,tokens=layout_for(task)
    actions=sum((hi-lo)*d for _,lo,hi,d in layout)
    policy=ActorCritic(1,1,actions,actor_hidden_dims=[1],critic_hidden_dims=[1],
                       init_noise_std=.1,max_std=.5)
    policy.actor=TokenActor(layout,tokens);policy.critic=TokenCritic(tokens)
    return policy.to(device),layout,tokens,actions


class Controller:
    def __init__(self,task,device,policy=None):
        initial,self.layout,self.tokens,self.actions=policy_for(task,device)
        self.policy=initial if policy is None else policy
        self.algorithm=None;self.steps=0;self.last_obs=None

    def residuals(self,features,t):
        n=features.shape[0]
        time_feature=torch.full((n,self.tokens,1),float(t)/99.,device=features.device)
        obs=torch.cat([features.detach().float(),time_feature],-1).flatten(1)
        self.last_obs=obs
        # PPO rollout log-probabilities and PPO updates must use the same
        # precision, independently of the frozen TT's bf16 autocast context.
        with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
            if self.algorithm is not None:raw=self.algorithm.act(obs,obs)
            else:raw=self.policy.act_inference(obs)
        result={};offset=0
        for name,lo,hi,d in self.layout:
            count=(hi-lo)*d
            result[name]=.1*torch.tanh(raw[:,offset:offset+count].reshape(n,hi-lo,d))
            offset+=count
        return result


@contextmanager
def residual_context(task,controller):
    if controller is None:
        yield;return
    original=task._denoise_samples
    def step(self,*args,**kwargs):
        features=kwargs['backbone_outputs'];masks=kwargs['dont_update_masks']
        residuals=controller.residuals(features,kwargs['t'])
        hooks=[]
        try:
            for name,delta in residuals.items():
                # The original scheduler, sample clipping and known masks remain
                # in the original method. Hook only its epsilon projection.
                def hook(module,args,kw,output,name=name,delta=delta):
                    correction=torch.where(masks[name],torch.zeros_like(delta),delta)
                    return output+correction.to(output.dtype)
                hooks.append(self.adapters[name].out_proj.register_forward_hook(hook,with_kwargs=True))
            original(*args,**kwargs)
        finally:
            for handle in hooks:handle.remove()
        if controller.algorithm is not None:
            n=features.shape[0];device=features.device
            controller.algorithm.process_env_step(torch.zeros(n,device=device),
                                                   torch.zeros(n,dtype=torch.bool,device=device),{})
        controller.steps+=1
    task._denoise_samples=types.MethodType(step,task)
    try:yield
    finally:task._denoise_samples=original


def train_policy(model,cfg,task,seed,base_checkpoint,out,updates=1000,batch=16,stop_after=None):
    from native_tt_eval import sample,physical_replay,evaluate
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device
    torch.manual_seed(seed);np.random.seed(seed)
    control=Controller(model.tasks.hardware_gen,device)
    # A joint Gaussian has thousands of coordinates: scale the update so its
    # aggregate likelihood change does not grow with the action dimension.
    learning_rate=3e-4/np.sqrt(control.actions)
    algo=PPO(control.policy,num_learning_epochs=4,num_mini_batches=16,
             learning_rate=learning_rate,clip_param=.2,gamma=1.,lam=.95,device=str(device))
    algo.init_storage(batch,100,[control.tokens*257],[None],[control.actions])
    control.algorithm=algo
    ckpath=out/'training_state.pt';first=0;best=-float('inf')
    contract=dict(base_sha256=digest(base_checkpoint),task=task,seed=seed,updates=updates,batch=batch,
                  learning_rate=learning_rate,mini_batches=16,action_dim=control.actions)
    if ckpath.exists():
        ck=torch.load(ckpath,map_location=device,weights_only=False)
        if ck['contract']!=contract:raise ValueError('PPO resume contract differs')
        control.policy.load_state_dict(ck['policy']);algo.optimizer.load_state_dict(ck['optimizer'])
        first=ck['update'];best=ck['best'];restore_rng(ck['rng'])
    started=time.time()
    for update in range(first,min(updates,stop_after) if stop_after else updates):
        control.steps=0
        decoded=sample(model,cfg,task,batch,seed*100000+update,controller=control)
        if control.steps!=100 or algo.storage.step!=100:raise AssertionError('Not 100 true denoising transitions')
        rows=[]
        for i in range(batch):
            row,arrays=physical_replay({k:v[i] for k,v in decoded.items()},task);rows.append(row)
            path=out/f'rollout_{update:04d}_{i:02d}.npz'
            np.savez_compressed(path,**arrays,**{'pred_'+k:v[i] for k,v in decoded.items()})
        rewards=torch.tensor([r['return_value'] for r in rows],device=device)
        algo.storage.rewards[-1,:,0]=rewards;algo.storage.dones[-1,:,0]=1
        if torch.any(algo.storage.rewards[:-1]!=0):raise AssertionError('Intermediate reward must be zero')
        with torch.no_grad():
            # Before any update, re-evaluating stored actions must reproduce
            # their rollout probabilities, including the TT autocast boundary.
            obs=algo.storage.observations[0]
            control.policy.update_distribution(obs)
            logp=control.policy.get_actions_log_prob(algo.storage.actions[0])
            torch.testing.assert_close(logp,algo.storage.actions_log_prob[0,:,0],atol=.02,rtol=0.)
        algo.compute_returns(control.last_obs)
        losses=algo.update()
        if not all(torch.isfinite(parameter).all() for parameter in control.policy.parameters()):
            raise FloatingPointError('Nonfinite PPO parameter after update')
        save_json(out/f'rollout_{update:04d}.json',dict(rows=rows,terminal_rewards=rewards.tolist()))
        control.algorithm=None
        if update%50==0 or update+1==updates:
            saved_rng=rng_state()
            dev=evaluate(model,cfg,task,range(1000000+seed*100,1000016+seed*100),out/f'dev_{update+1}',controller=control)
            restore_rng(saved_rng)
            score=float(np.mean([r['return_value'] for r in dev]))
            if score>best:
                best=score;atomic_torch(out/'policy.pt',dict(policy=control.policy.state_dict(),contract=contract,update=update+1))
        control.algorithm=algo
        atomic_torch(ckpath,dict(policy=control.policy.state_dict(),optimizer=algo.optimizer.state_dict(),
            update=update+1,best=best,contract=contract,rng=rng_state()))
        record=dict(update=update+1,mean_return=float(rewards.mean()),seconds=time.time()-started)
        with (out/'progress.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        print(record,flush=True)
    if digest(base_checkpoint)!=contract['base_sha256']:raise AssertionError('Frozen TT checkpoint changed')
    control.algorithm=None
    control.policy.load_state_dict(torch.load(out/'policy.pt',map_location=device,weights_only=False)['policy'])
    return control
