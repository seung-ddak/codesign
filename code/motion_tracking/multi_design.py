"""Train one tracking policy against matched TT models and references."""

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training as brax_training

from t2.mjx_env.wrapper import BraxAutoResetWrapper, Wrapper


REFERENCE_ATTRS = (
    "ref_qpos", "ref_qvel", "ref_foot", "ref_contact", "init_q", "action_offset",
    "_lowers", "_uppers", "_soft_lowers", "_soft_uppers", "base_mass",
)


def make_bank(envs, uniforms):
    """Stack design-dependent MJX fields and matching reference arrays."""
    if len(envs) != len(uniforms) or len(envs) < 2:
        raise ValueError("Need at least two matched models and morphology vectors")
    models = [env.mjx_model for env in envs]
    base = models[0]
    for env in envs[1:]:
        if (env.mj_model.nq, env.mj_model.nv, env.mj_model.ngeom) != (
            envs[0].mj_model.nq, envs[0].mj_model.nv, envs[0].mj_model.ngeom
        ) or not np.array_equal(env.clip_start, envs[0].clip_start) or not np.array_equal(env.clip_end, envs[0].clip_end):
            raise ValueError("Morphologies must share model topology and reference lengths")
    # MJX treats these fields as compile-time metadata. Zero sameframe is the general kinematic path.
    models = [m.replace(stat=base.stat, geom_sameframe=np.zeros_like(m.geom_sameframe),
                        _impl=m._impl.replace(geom_rbound_hfield=base._impl.geom_rbound_hfield)) for m in models]
    if any(jax.tree.structure(m) != jax.tree.structure(models[0]) for m in models[1:]):
        raise ValueError("MJX model structures still differ")
    changed = [f.name for f in dataclasses.fields(type(base))
               if isinstance(getattr(base, f.name), jax.Array)
               and any(not np.array_equal(getattr(base, f.name), getattr(m, f.name)) for m in models[1:])]
    bank = {
        "base": models[0],
        "model_fields": {name: jnp.stack([getattr(m, name) for m in models]) for name in changed},
        "refs": {name: jnp.stack([jnp.asarray(getattr(env, name)) for env in envs]) for name in REFERENCE_ATTRS},
        "uniforms": jnp.asarray(uniforms, dtype=jnp.float32),
    }
    if bank["uniforms"].shape != (len(envs), 13):
        raise ValueError("Expected 13 morphology values per design")
    return bank


class MultiDesignVmapWrapper(Wrapper):
    def __init__(self, env, model_v, in_axes, design_ids, bank):
        super().__init__(env)
        self.model_v, self.in_axes, self.design_ids, self.bank = model_v, in_axes, design_ids, bank

    @contextlib.contextmanager
    def design_env(self, model, design_id):
        env = self.env.unwrapped
        old = {name: getattr(env, name) for name in ("mjx_model", *REFERENCE_ATTRS)}
        try:
            env.mjx_model = model
            for name in REFERENCE_ATTRS:
                setattr(env, name, self.bank["refs"][name][design_id])
            yield env
        finally:
            for name, value in old.items():
                setattr(env, name, value)

    def reset(self, rng):
        def one(model, design_id, key):
            with self.design_env(model, design_id) as env:
                state = env.reset(key)
                return state.replace(obs=jnp.concatenate((state.obs, self.bank["uniforms"][design_id])))
        return jax.vmap(one, in_axes=(self.in_axes, 0, 0))(self.model_v, self.design_ids, rng)

    def step(self, state, action):
        def one(model, design_id, s, a):
            with self.design_env(model, design_id) as env:
                next_state = env.step(s, a)
                return next_state.replace(obs=jnp.concatenate((next_state.obs, self.bank["uniforms"][design_id])))
        return jax.vmap(one, in_axes=(self.in_axes, 0, 0, 0))(self.model_v, self.design_ids, state, action)


def randomize_bank(bank, model, rng):
    design_ids = jnp.arange(len(rng)) % len(bank["uniforms"])
    model_v = bank["base"].tree_replace({
        name: values[design_ids] for name, values in bank["model_fields"].items()
    })
    in_axes = jax.tree.map(lambda _: None, bank["base"]).tree_replace({
        name: 0 for name in bank["model_fields"]
    })
    return model_v, in_axes, design_ids


def wrap_for_multi_design(env, episode_length, action_repeat, randomization_fn, full_reset):
    model_v, in_axes, design_ids = randomization_fn(env.mjx_model)
    wrapped = MultiDesignVmapWrapper(env, model_v, in_axes, design_ids, env.design_bank)
    wrapped = brax_training.EpisodeWrapper(wrapped, episode_length, action_repeat)
    return BraxAutoResetWrapper(wrapped, full_reset=full_reset)
