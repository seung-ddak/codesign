import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/tt_dynamics'))
from native_tt_train import rng_state,restore_rng
from native_tt_residual import TokenActor,TokenCritic
from native_tt_eval import task_step,trajectory


def test_per_token_policy_starts_at_zero_and_can_vary_over_time():
    actor=TokenActor([('ctrl',0,2,1)],2)
    x=torch.zeros(3,2,257);x[:,1,0]=1
    torch.testing.assert_close(actor(x.flatten(1)),torch.zeros(3,2))
    with torch.no_grad():
        actor.heads['ctrl'][0].weight.fill_(.1)
        actor.heads['ctrl'][-1].weight.fill_(.1)
    y=actor(x.flatten(1))
    assert torch.all(y[:,1]>y[:,0])
    assert TokenCritic(2)(x.flatten(1)).shape==(3,1)


def test_checkpoint_rng_restores_next_batch_and_noise():
    state=rng_state();expected=(torch.randn(3),np.random.randint(100,size=8))
    restore_rng(state)
    torch.testing.assert_close(torch.randn(3),expected[0])
    np.testing.assert_array_equal(np.random.randint(100,size=8),expected[1])


def test_task_reward_uses_real_progress_and_distinguishes_straight_from_circle():
    poses=trajectory('straight')
    assert poses.shape==(500,4,4)
    np.testing.assert_allclose(poses[-1,:2,3],[4.99,0])
    prev=np.array([0.,0.,.3]);pos=np.array([.01,0.,.3])
    reward,_,lane=task_step(prev,pos,np.eye(3),np.zeros(12),np.zeros(12),'straight',1,.3)
    bad,_,bad_lane=task_step(prev,pos+[0,1,0],np.eye(3),np.zeros(12),np.zeros(12),'straight',1,.3)
    assert lane==0 and bad_lane==1 and reward>bad
    with pytest.raises(ValueError):trajectory('unknown')


def test_installed_ppo_learns_known_residual_and_stores_terminal_reward():
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    torch.manual_seed(0)
    policy=ActorCritic(1,1,2,actor_hidden_dims=[1],critic_hidden_dims=[1],init_noise_std=.1,max_std=.5)
    policy.actor=TokenActor([('ctrl',0,2,1)],2);policy.critic=TokenCritic(2)
    obs=torch.zeros(16,514)
    algo=PPO(policy,num_learning_epochs=4,num_mini_batches=1,learning_rate=.003,gamma=1.)
    algo.init_storage(16,1,[514],[None],[2])
    target=torch.full((16,2),.2)
    initial=float(((policy.act_inference(obs)-target)**2).mean())
    for _ in range(200):
        with torch.no_grad():
            action=algo.act(obs,obs)
            rewards=-((action-target)**2).mean(1)
            algo.process_env_step(rewards,torch.ones(16,dtype=torch.bool),{})
            torch.testing.assert_close(algo.storage.rewards[0,:,0],rewards)
            algo.compute_returns(obs)
        algo.update()
    final=float(((policy.act_inference(obs)-target)**2).mean())
    assert final<initial*.1,(initial,final)


def test_generated_invalid_graph_is_a_failure_and_programming_errors_propagate(monkeypatch):
    import t2.eval.utils as utils
    import t2.robotok.io as robot_io
    import t2.robotok.tokenizer as tokenizer
    from native_tt_eval import physical_replay
    monkeypatch.setattr(utils,'extract_masked_hardware',lambda candidate:({},0,0,0,0))
    monkeypatch.setattr(robot_io,'deserialize',lambda candidate:object())
    def invalid(*args,**kwargs):raise tokenizer.DetokenizeError('bad generated graph')
    monkeypatch.setattr(tokenizer,'detokenize',invalid)
    row,arrays=physical_replay({},'straight')
    assert row['invalid'] and row['return_value']==-20 and arrays=={}
    def bug(*args,**kwargs):raise TypeError('programming error')
    monkeypatch.setattr(tokenizer,'detokenize',bug)
    with pytest.raises(TypeError):physical_replay({},'straight')
