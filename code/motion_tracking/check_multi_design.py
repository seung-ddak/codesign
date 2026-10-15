"""Small reset/step check for matched multi-morphology training."""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from multi_design import MultiDesignVmapWrapper, make_bank, randomize_bank, wrap_for_multi_design
from pd_replay import ROOT, build_model
from track_env import TrackMjxEnv


root = Path(ROOT) / "runs/pipeline_v1/retargeting"
designs = json.loads((root / "designs.json").read_text())
config = OmegaConf.to_container(OmegaConf.load(Path(__file__).parent / "config/env/a1_track.yaml"))
config.pop("_target_")
config["clips"] = ["walkBackwardSlow__r00"]
names = ("e000", "e001", "e002")
envs = [TrackMjxEnv(**{**config, "reference_dir": str(root / name / designs["topology"]),
                       "mj_model_fn": functools.partial(build_model, choice=designs["topology"],
                                                         uniforms=designs[name], torque_limit=33.5)})
        for name in names]
bank = make_bank(envs, [designs[name] for name in names])
envs[0].design_bank = bank
keys = jax.random.split(jax.random.PRNGKey(0), len(names))
models, axes, ids = randomize_bank(bank, envs[0].mjx_model, keys)
raw = MultiDesignVmapWrapper(envs[0], models, axes, ids, bank)
raw_state = jax.jit(raw.reset)(keys)
raw_next = jax.jit(raw.step)(raw_state, jnp.zeros((len(names), 12)))
for i, env in enumerate(envs):
    direct = jax.jit(env.reset)(keys[i])
    assert jnp.allclose(raw_state.data.qpos[i], direct.data.qpos, atol=1e-5)
    assert jnp.allclose(raw_state.data.geom_xpos[i], direct.data.geom_xpos, atol=1e-4)
    direct_next = jax.jit(env.step)(direct, jnp.zeros(12))
    assert jnp.allclose(raw_next.data.qpos[i], direct_next.data.qpos, atol=1e-4)
wrapped = wrap_for_multi_design(envs[0], 100, 1,
                                functools.partial(randomize_bank, bank, rng=keys), True)
state = jax.jit(wrapped.reset)(keys)
assert state.obs.shape == (len(names), envs[0].observation_size + 13)
assert jnp.allclose(state.obs[:, -13:], bank["uniforms"])
next_state = jax.jit(wrapped.step)(state, jnp.zeros((len(names), 12)))
assert bool(jnp.isfinite(next_state.obs).all())
print("multi-design reset/step passed", flush=True)
