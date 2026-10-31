"""Train one tracking policy against matched TT models and references."""

import contextlib
import dataclasses
import functools
import json
from pathlib import Path
from uuid import uuid4

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training as brax_training
from brax.training.agents.ppo import networks as ppo_networks
from omegaconf import OmegaConf

from t2.mjx_env.wrapper import Wrapper


REFERENCE_ATTRS = (
    "ref_qpos", "ref_qvel", "ref_foot", "ref_contact", "init_q", "action_offset",
    "_lowers", "_uppers", "_soft_lowers", "_soft_uppers", "base_mass",
)


ROLLOUT_FIELDS = ("qpos", "qvel", "next_qpos", "next_qvel", "action", "ctrl")
FAILURE_KEYS = ("done/fellover", "done/pos_err", "done/orn_err", "state_is_nan")
E_SIZE = 13


def make_fixed_e_ppo_networks(observation_size, action_size, preprocess_observations_fn, **kwargs):
    """Scale E in [0, 1] to [-1, 1] instead of running statistics.

    A coordinate constant during training has std 1e-6, so any other morphology
    would otherwise enter the networks at about 1e5. Normalizes all other inputs.
    """
    def preprocess(obs, normalizer):
        normalized = preprocess_observations_fn(obs, normalizer)
        return jnp.concatenate((normalized[..., :-E_SIZE], 2 * obs[..., -E_SIZE:] - 1), axis=-1)

    return ppo_networks.make_ppo_networks(
        observation_size, action_size, preprocess_observations_fn=preprocess, **kwargs)


def normalize_design_catalog(catalog, topology):
    """Normalize legacy mappings and pilot lists to {name: E, topology: ...}."""
    if isinstance(catalog, list):
        designs = {}
        for item in catalog:
            if not isinstance(item, dict) or "id" not in item or "E" not in item:
                raise ValueError("Design catalog entries must contain id and E")
            name = item["id"]
            if name in designs:
                raise ValueError(f"Duplicate design in catalog: {name}")
            designs[name] = item["E"]
        designs["topology"] = topology
        return designs
    if isinstance(catalog, dict):
        return catalog
    raise ValueError("Design catalog must be a mapping or a list of design records")


def load_catalog(root, topology):
    """Find designs.json at the reference root or one level above it."""
    for folder in (root, root.parent):
        if (folder / "designs.json").is_file():
            return normalize_design_catalog(json.loads((folder / "designs.json").read_text()), topology)
    raise FileNotFoundError(f"Could not find designs.json under {root} or {root.parent}")


def check_reference(path, E, lowers=None, uppers=None):
    """Reject a reference whose morphology, values or joint range cannot be trained."""
    with np.load(path) as ref:
        if "uniforms" not in ref or not np.allclose(ref["uniforms"], E, rtol=0, atol=1e-7):
            raise ValueError(f"Reference E mismatch: {path}")
        if not all(np.isfinite(ref[k]).all() for k in ["fix_q", "fix_base_pos", "fix_base_quat"]):
            raise ValueError(f"Nonfinite reference: {path}")
        if lowers is not None and (np.any(ref["fix_q"] < lowers - 1e-5) or np.any(ref["fix_q"] > uppers + 1e-5)):
            raise ValueError(f"Reference violates joint limits: {path}")


def build_bank(names, env_cfg, root, choice, designs, model_fn, unlock_steps):
    """Instantiate one validated env per morphology and stack them into a bank."""
    if len(unlock_steps) != len(names) or min(unlock_steps) < 0 or 0 not in unlock_steps:
        raise ValueError("Each design needs a nonnegative unlock step; at least one starts at zero")
    envs = []
    for name in names:
        if name not in designs:
            raise ValueError(f"Unknown design: {name}")
        design_cfg = OmegaConf.create(OmegaConf.to_container(env_cfg, resolve=True))
        design_cfg.reference_dir = str(root / name / choice)
        for clip in design_cfg.clips:
            check_reference(Path(design_cfg.reference_dir) / f"{clip}.npz", designs[name])
        env = hydra.utils.instantiate(design_cfg, mj_model_fn=functools.partial(model_fn, uniforms=designs[name]))
        for path in env.reference_files:
            check_reference(path, designs[name], np.asarray(env._lowers), np.asarray(env._uppers))
        envs.append(env)
    bank = make_bank(envs, [designs[name] for name in names])
    bank["unlock_steps"] = list(unlock_steps)
    return envs[0], bank


def make_bank(envs, uniforms):
    """Stack design-dependent MJX fields and matching reference arrays."""
    if len(envs) != len(uniforms) or len(envs) < 1:
        raise ValueError("Need matched models and morphology vectors")
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
    if bank["uniforms"].shape != (len(envs), 13) or not np.isfinite(uniforms).all() or np.any(np.asarray(uniforms) < 0) or np.any(np.asarray(uniforms) > 1):
        raise ValueError("Expected 13 morphology values per design")
    bank["unlock_steps"] = [0] * len(envs)
    return bank


class MultiDesignVmapWrapper(Wrapper):
    def __init__(self, env, bank):
        super().__init__(env)
        self.bank = bank

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

    def model_for(self, design_id):
        return self.bank["base"].tree_replace({
            name: values[design_id] for name, values in self.bank["model_fields"].items()
        })

    def reset(self, rng, elapsed=None):
        elapsed = jnp.zeros(rng.shape[:-1], dtype=jnp.int32) if elapsed is None else elapsed
        def one(key, age):
            key, design_key = jax.random.split(key)
            thresholds = jnp.asarray(self.bank["unlock_steps"])
            eligible = thresholds <= age
            design_id = jax.random.categorical(design_key, jnp.where(eligible, 0., -jnp.inf))
            with self.design_env(self.model_for(design_id), design_id) as env:
                state = env.reset(key)
                state.info["design_id"] = design_id
                return state.replace(obs=jnp.concatenate((state.obs, self.bank["uniforms"][design_id])))
        return jax.vmap(one)(rng, elapsed)

    def step(self, state, action):
        def one(s, a):
            design_id = s.info["design_id"]
            with self.design_env(self.model_for(design_id), design_id) as env:
                previous_steps = s.info["steps"]
                next_state = env.step(s, a)
                # EpisodeWrapper owns this counter; TrackMjxEnv also increments it.
                next_state.info["steps"] = previous_steps
                return next_state.replace(obs=jnp.concatenate((next_state.obs, self.bank["uniforms"][design_id])))
        return jax.vmap(one)(state, action)


class EpisodeDesignReset(Wrapper):
    """Resample morphology only at episode boundaries; expose pre-reset transitions."""

    def __init__(self, env, episode_length, action_repeat):
        super().__init__(brax_training.EpisodeWrapper(env, episode_length, action_repeat))
        self.design_env = env
        self.action_repeat = action_repeat
        self.episode_length = episode_length
        self.rollout_dir = env.bank.get("rollout_dir")
        self.host_uniforms = np.asarray(env.bank["uniforms"])

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["design_elapsed"] = jnp.zeros(rng.shape[:-1], dtype=jnp.int32)
        state.info["design_rng"] = rng
        if self.rollout_dir:
            self.init_recording(state)
        return state

    def step(self, state, action):
        elapsed = state.info["design_elapsed"] + self.action_repeat
        keys = jax.vmap(jax.random.split)(state.info["design_rng"])
        state = state.replace(done=jnp.zeros_like(state.done))
        if self.rollout_dir:
            previous_qpos, previous_qvel = state.data.qpos, state.data.qvel
        terminal = self.env.step(state, action)
        if self.rollout_dir:
            self.record(terminal, previous_qpos, previous_qvel, action)
        # Most steps do not finish an episode: avoid MJX forward/reset for the whole
        # batch unless at least one environment actually terminates.
        def reset_finished(current):
            fresh = self.design_env.reset(keys[:, 0], elapsed)
            fresh.info["steps"] = jnp.zeros_like(current.info["steps"])
            fresh.info["truncation"] = jnp.zeros_like(current.info["truncation"])

            def select(x, y):
                mask = current.done.reshape(current.done.shape + (1,) * (x.ndim - current.done.ndim))
                return jnp.where(mask, x, y)

            info = {k: jax.tree.map(select, fresh.info[k], v) if k in fresh.info else v
                    for k, v in current.info.items()}
            if self.rollout_dir:
                for name in ROLLOUT_FIELDS:
                    key = "rollout_" + name
                    info[key] = select(jnp.zeros_like(info[key]), info[key])
            return current.replace(
                data=jax.tree.map(select, fresh.data, current.data),
                obs=select(fresh.obs, current.obs),
                info=info,
            )

        terminal = jax.lax.cond(jnp.any(terminal.done.astype(jnp.bool_)),
                                reset_finished, lambda current: current, terminal)
        info = dict(terminal.info)
        info["design_elapsed"] = elapsed
        info["design_rng"] = keys[:, 1]
        return terminal.replace(info=info)

    def init_recording(self, state):
        for name, value in [("qpos", state.data.qpos), ("qvel", state.data.qvel),
                            ("next_qpos", state.data.qpos), ("next_qvel", state.data.qvel),
                            ("action", jnp.zeros((len(state.done), self.action_size))),
                            ("ctrl", state.data.ctrl)]:
            state.info["rollout_" + name] = jnp.zeros((len(state.done), self.episode_length, value.shape[-1]), dtype=value.dtype)

    def save_finished(self, done, lengths, design_ids, clips, frames, failed, *arrays):
        """Host callback: write every finished episode, success or failure, to disk."""
        for i in np.flatnonzero(done):
            folder = Path(self.rollout_dir) / ("failed" if failed[i] else "completed")
            folder.mkdir(parents=True, exist_ok=True)
            n = int(lengths[i])
            np.savez_compressed(folder / f"{uuid4().hex}.npz",
                                E=self.host_uniforms[design_ids[i]],
                                clip=int(clips[i]), end_frame=int(frames[i]), failed=bool(failed[i]),
                                ctrl_dt=float(self.dt),
                                **{name: arr[i, :n] for name, arr in zip(ROLLOUT_FIELDS, arrays)})

    def record(self, terminal, qpos, qvel, action):
        values = [qpos, qvel, terminal.data.qpos, terminal.data.qvel, action, terminal.data.ctrl]
        index = jnp.minimum(terminal.info["steps"].astype(jnp.int32) - 1, self.episode_length - 1)
        for name, value in zip(ROLLOUT_FIELDS, values):
            key = "rollout_" + name
            terminal.info[key] = terminal.info[key].at[jnp.arange(len(index)), index].set(value)

        def emit(_):
            info = terminal.info
            failed = sum(info.get(k, jnp.zeros_like(terminal.done)) for k in FAILURE_KEYS) > 0
            jax.debug.callback(self.save_finished, terminal.done, info["steps"], info["design_id"],
                               info["clip"], info["frame"], failed,
                               *[info["rollout_" + n] for n in ROLLOUT_FIELDS], ordered=False)
            return jnp.int32(0)

        jax.lax.cond(jnp.any(terminal.done), emit, lambda _: jnp.int32(0), operand=None)


def wrap_for_multi_design(env, episode_length, action_repeat, randomization_fn, full_reset):
    if action_repeat != 1:
        raise ValueError("Multi-design tracking requires action_repeat=1 for aligned rollout recording")
    if not full_reset:
        raise ValueError("Episode morphology sampling requires full_reset=true")
    wrapped = MultiDesignVmapWrapper(env, env.design_bank)
    return EpisodeDesignReset(wrapped, episode_length, action_repeat)


def validate_split(train_designs, eval_designs, train_clips, eval_clips):
    for train, test, label in [(train_designs, eval_designs, "morphology"),
                               (train_clips, eval_clips, "motion")]:
        if not train or not test or set(train) & set(test):
            raise ValueError(f"Nonempty disjoint training/evaluation {label} sets required")
        if len(set(train)) != len(train) or len(set(test)) != len(test):
            raise ValueError(f"Duplicate {label} in split")
