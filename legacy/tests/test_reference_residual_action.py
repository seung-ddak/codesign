"""Motor targets stay consistent between PPO and recorded dynamics."""
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'code/motion_tracking'),str(Path(__file__).resolve().parents[2]/'transformer-transformer')]
from track_env import TrackMjxEnv


def test_reference_target_uses_next_frame_and_clips_residual():
    env=TrackMjxEnv.__new__(TrackMjxEnv)
    env.switch_entry="match"
    env.n_ctrl=12;env.action_scale=.25;env.clip_actions=(-10.,10.)
    env.action_offset=jnp.arange(12,dtype=jnp.float32)/100
    env.ref_qpos=jnp.arange(5*19,dtype=jnp.float32).reshape(5,19)/100
    action=jnp.ones(12)*20
    info={'frame':jnp.int32(2),'clip_end':jnp.int32(4)}
    env.action_reference='home'
    np.testing.assert_allclose(env.action_target(action,info),env.action_offset+2.5)
    env.action_reference='reference'
    got=jax.jit(env.action_target)(action,info)
    np.testing.assert_allclose(got,env.ref_qpos[3,7:19]+2.5)
    env.switch_entry='start'
    info['sw_left']=jnp.int32(2)
    np.testing.assert_allclose(env.action_target(jnp.zeros(12),info),env.ref_qpos[2,7:19])
    info['sw_left']=jnp.int32(0)
    info['frame']=jnp.int32(3)
    np.testing.assert_allclose(env.action_target(jnp.zeros(12),info),env.ref_qpos[3,7:19])


def test_zero_mean_initializer_is_portable_to_checkpoint_loader():
    import hydra
    from omegaconf import OmegaConf
    from multi_design import make_fixed_e_ppo_networks
    cfg=OmegaConf.create({'_target_':'jax.nn.initializers.constant','_partial_':True,'value':0.})
    init=hydra.utils.instantiate(cfg)
    net=make_fixed_e_ppo_networks(30,12,preprocess_observations_fn=lambda x,p:x,
        policy_hidden_layer_sizes=(8,8),value_hidden_layer_sizes=(8,8),distribution_type='normal',
        init_noise_std=.2,mean_kernel_init_fn=init)
    params=net.policy_network.init(jax.random.PRNGKey(0))
    logits=net.policy_network.apply(None,params,jnp.ones((2,30)))
    np.testing.assert_allclose(logits[0],0,atol=0)
    np.testing.assert_allclose(logits[1],.2,atol=1e-6)
