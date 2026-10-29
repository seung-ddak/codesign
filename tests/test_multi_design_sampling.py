import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]/'transformer-transformer'))
sys.path.insert(0, str(Path(__file__).parents[1]/'code/motion_tracking'))
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from t2.mjx_env.base_env import State
from multi_design import (MultiDesignVmapWrapper, EpisodeDesignReset,
                          normalize_design_catalog, validate_split)


class ToySampler(MultiDesignVmapWrapper):
    def __init__(self):
        self.bank = {'unlock_steps': [0, 2], 'uniforms': jnp.array([[.5]*13,[.6]*13])}
    def model_for(self, i):
        return i
    def design_env(self, model, design_id):
        import contextlib
        @contextlib.contextmanager
        def scope():
            self.current = design_id
            yield self
        return scope()
    @property
    def unwrapped(self):
        return self
    def reset(self, rng, elapsed=None):
        if rng.ndim == 1:
            clip, frame = jax.random.randint(rng, (2,), 0, 100)
            return State(jnp.array([self.current]), jnp.array([self.current], dtype=float),
                         jnp.array(0.), jnp.array(0.), {}, {'clip':clip, 'frame':frame})
        return super().reset(rng, elapsed)
    def step(self, state, action):
        return state.replace(reward=jnp.ones_like(state.reward))


def test_episode_sampling_curriculum_and_reset():
    env = EpisodeDesignReset(ToySampler(), episode_length=2, action_repeat=1)
    keys = jax.random.split(jax.random.PRNGKey(13), 64)
    state = jax.jit(env.reset)(keys)
    np.testing.assert_array_equal(state.info['design_id'], 0)
    first = np.array(state.info['frame'])
    advance = jax.jit(env.step)
    state = advance(state, jnp.zeros((64,1)))
    np.testing.assert_array_equal(state.info['design_id'], 0)
    state = advance(state, jnp.zeros((64,1)))
    assert set(np.array(state.info['design_id'])) == {0,1}
    assert np.any(first != np.array(state.info['frame']))
    np.testing.assert_allclose(state.obs[:,-13:], np.where(np.array(state.info['design_id'])[:,None] == 0,.5,.6)*np.ones((64,13)), atol=1e-6)
    np.testing.assert_array_equal(state.data[:,0], state.info['design_id'])
    assert np.all(np.array(state.done)==1)
    state = advance(state, jnp.zeros((64,1)))
    assert np.all(np.array(state.done)==0)


def test_disjoint_splits():
    validate_split(['e000'], ['e008'], ['walk'], ['turn'])
    with pytest.raises(ValueError):validate_split(['e000'], ['e000'], ['walk'], ['turn'])
    with pytest.raises(ValueError):validate_split(['e000'], ['e008'], ['walk'], ['walk'])


def test_normalize_design_catalog():
    catalog = [{'id': 'near_short', 'E': [.5] * 13},
               {'id': 'near_long', 'E': [.6] * 13}]
    normalized = normalize_design_catalog(catalog, '00000011')
    assert normalized['topology'] == '00000011'
    np.testing.assert_array_equal(normalized['near_short'], [.5] * 13)
    assert normalize_design_catalog({'topology': '00000011', 'e000': [.5] * 13},
                                    '00000011')['e000'] == [.5] * 13
    with pytest.raises(ValueError):
        normalize_design_catalog([{'id': 'duplicate', 'E': [.5] * 13},
                                  {'id': 'duplicate', 'E': [.6] * 13}], '00000011')


def test_recorder_keeps_terminal_transition(tmp_path):
    from flax import struct
    @struct.dataclass
    class Data:
        qpos: object
        qvel: object
        ctrl: object
    class RecordedToy(ToySampler):
        @property
        def action_size(self):return 1
        @property
        def dt(self):return .03333333333333333
        def reset(self, rng, elapsed=None):
            s = super().reset(rng, elapsed)
            if rng.ndim == 1:
                s.info["done/fellover"] = jnp.array(0.)
                s = s.replace(data=Data(s.data.astype(float), jnp.zeros_like(s.data, dtype=float), jnp.zeros_like(s.data, dtype=float)))
            return s
        def step(self, s, a):
            info = dict(s.info)
            info['done/fellover'] = jnp.ones_like(s.done)
            return s.replace(data=s.data.replace(qpos=s.data.qpos+1,ctrl=a), done=jnp.ones_like(s.done),info=info)
    toy=RecordedToy();toy._ctrl_dt=1/30;toy.bank['rollout_dir']=str(tmp_path)
    env=EpisodeDesignReset(toy,2,1)
    state=jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(3),2))
    state=jax.jit(env.step)(state,jnp.ones((2,1)))
    jax.effects_barrier()
    files=list((tmp_path/'failed').glob('*.npz'))
    assert len(files)==2
    for f in files:
        with np.load(f) as d:
            assert len(d['qpos'])==1 and d['failed']
            np.testing.assert_allclose(d['next_qpos'],d['qpos']+1)
            np.testing.assert_allclose(d['ctrl'],1)
