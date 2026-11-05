import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'transformer-transformer'))
import jax,jax.numpy as jnp,numpy as np
from t2.mjx_env.base_env import State
from t2.mjx_env.train import train
from brax.training.agents.ppo.networks import make_ppo_networks
from flax.serialization import msgpack_restore
import functools,json
class Toy:
 action_size=1
 observation_size=2
 @property
 def unwrapped(self):return self
 def reset(self,keys):
  shape=keys.shape[:-1];v=jnp.zeros(shape)
  return State(data=v,obs=jnp.zeros(shape+(2,)),reward=v,done=v,metrics={},info={'truncation':v,'episode_metrics':{'reward':v},'episode_done':v})
 def step(self,s,a):
  r=1.-jnp.square(a[...,0]);return s.replace(data=s.data+1,obs=jnp.stack((s.data,s.data),-1)*.01,reward=r,info={'truncation':jnp.zeros_like(r),'episode_metrics':{'reward':r},'episode_done':jnp.zeros_like(r)})
def test_full_optimizer_state_and_step_continue_across_batches(tmp_path):
 path=tmp_path/'learner.msgpack';steps=[]
 kw=dict(environment=Toy(),wrap_env=False,num_timesteps=8,episode_length=4,num_envs=2,batch_size=2,unroll_length=2,num_minibatches=1,num_updates_per_batch=1,num_evals=2,num_eval_envs=2,run_evals=False,full_state_path=str(path),network_factory=functools.partial(make_ppo_networks,policy_hidden_layer_sizes=(4,),value_hidden_layer_sizes=(4,)),progress_fn=lambda s,m:steps.append(s))
 train(**kw);a=msgpack_restore(path.read_bytes());assert json.loads(path.with_suffix('.json').read_text())['step']==8
 train(**kw);b=msgpack_restore(path.read_bytes());assert json.loads(path.with_suffix('.json').read_text())['step']==16
 def counts(x):
  result=[]
  if isinstance(x,dict):
   for k,v in x.items():
    if k=='count':result.append(int(v))
    else:result.extend(counts(v))
  return result
 assert max(counts(b['optimizer_state']))>max(counts(a['optimizer_state'])),(counts(a['optimizer_state']),counts(b['optimizer_state']))
 print('Full-state resume passed:',steps,'Adam update counts:',counts(a['optimizer_state']),counts(b['optimizer_state']))


def test_nonfinite_update_preserves_last_finite_checkpoint(tmp_path):
 import pytest
 class Broken(Toy):
  def step(self,s,a):return super().step(s,a).replace(reward=jnp.full_like(s.reward,jnp.nan))
 path=tmp_path/'learner.msgpack'
 with pytest.raises(FloatingPointError,match='previous checkpoint preserved'):
  train(environment=Broken(),wrap_env=False,num_timesteps=4,episode_length=4,num_envs=2,batch_size=2,unroll_length=2,num_minibatches=1,num_updates_per_batch=1,num_evals=2,num_eval_envs=2,run_evals=False,full_state_path=str(path),network_factory=functools.partial(make_ppo_networks,policy_hidden_layer_sizes=(4,),value_hidden_layer_sizes=(4,)))
 assert json.loads(path.with_suffix('.json').read_text())['step']==0
