"""Shared machinery for the L0~L3 design-search experiments (remind.md §4.3).

A design is the 13-dim vector of continuous `uniforms` of one discrete choice.
Every rung uses the same two primitives:

  uniforms_to_attrs : designs -> per-morphology MuJoCo model fields (CPU, ray)
  evaluate_designs  : those fields -> binary success rate under the frozen pi_theta

so that L0/L1/L2/L3 differ only in how they propose designs.
"""

import contextlib
import copy
import dataclasses
import os
import pickle
from typing import Callable, Optional

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import hydra
import jax
import jax.numpy as jnp
import ray
import zarr
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from dm_control import mjcf
from mujoco import mjx
from orbax import checkpoint as ocp

from t2.mjx_env.base_env import MjxEnv
from t2.mjx_env.stair_env import build_stair_scene

jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")


# -- design -> model fields --------------------------------------------------


def _attrs_for_uniforms(
    uniforms_list: list[list[float]],
    choices: list[int],
    robogen_from_params: Callable,
    data_attrs: list[str],
    rise: Optional[float] = None,
) -> list[dict[str, np.ndarray]]:
    out = []
    for uniforms in uniforms_list:
        mjcf_model = robogen_from_params(
            choices=copy.deepcopy(choices), uniforms=list(uniforms)
        )
        mj_model = mjcf.Physics.from_mjcf_model(
            build_stair_scene(mjcf_model, rise=rise)
        ).model.ptr
        out.append({attr: getattr(mj_model, attr).copy() for attr in data_attrs})
    return out


def uniforms_to_attrs(
    uniforms: np.ndarray,
    choices: list[int],
    robogen_from_params: Callable,
    data_attrs: list[str],
    num_processes: int = 48,
    rise: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """(B, num_uniforms) designs -> {attr: (B, ...)} model fields.

    `rise` fixes the staircase height so every design in a comparison faces the
    same terrain; None keeps `DEFAULT_STAIRS`.
    """
    if not ray.is_initialized():
        ray.init(num_cpus=num_processes, log_to_driver=False)
    remote = ray.remote(num_cpus=1)(_attrs_for_uniforms)
    groups = np.array_split(uniforms, min(num_processes, len(uniforms)))
    results = ray.get(
        [
            remote.remote(g.tolist(), choices, robogen_from_params, data_attrs, rise)
            for g in groups
        ]
    )
    flat = [d for group in results for d in group]
    return {a: np.stack([d[a] for d in flat], axis=0) for a in data_attrs}


# -- frozen policy -----------------------------------------------------------


def load_run(ckpt_path: str):
    """Returns (cfg, policy_fn) for a `train_rl_stairs.py` checkpoint directory."""
    run_dir = ckpt_path.split("/checkpoints/")[0]
    cfg = pickle.load(open(f"{run_dir}/config.pkl", "rb"))
    params = ocp.PyTreeCheckpointer().restore(os.path.abspath(ckpt_path))
    policy_params = params[1]["params"]
    action_size = len(policy_params["std_param"]["value"])
    obs_size = policy_params["MLP_0"]["hidden_0"]["kernel"].shape[0]

    normalize = (
        running_statistics.normalize
        if cfg.train.normalize_observations
        else (lambda x, y: x)
    )
    network_factory = hydra.utils.instantiate(cfg.train.network_factory)
    ppo_network = network_factory(
        obs_size, action_size, preprocess_observations_fn=normalize
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    # orbax restores the normalizer as a plain dict; the network expects the
    # RunningStatisticsState dataclass.
    normalizer = params[0]
    if isinstance(normalizer, dict):
        fields = {f.name for f in dataclasses.fields(
            running_statistics.RunningStatisticsState
        )}
        kwargs = {}
        for k, v in normalizer.items():
            if k not in fields:
                continue
            # `count` round-trips as a {hi, lo} pair under 32-bit jax.
            if isinstance(v, dict):
                v = sum(jnp.asarray(x) for x in v.values())
            kwargs[k] = jnp.asarray(v)
        normalizer = running_statistics.RunningStatisticsState(**kwargs)
    return cfg, make_policy((normalizer, params[1]), deterministic=True)


def latest_checkpoint(run_dir: str) -> str:
    ckpts = [
        d
        for d in os.listdir(f"{run_dir}/checkpoints")
        if d.isdigit() and os.path.isdir(f"{run_dir}/checkpoints/{d}")
    ]
    assert ckpts, f"no checkpoints under {run_dir}"
    return f"{run_dir}/checkpoints/{max(ckpts, key=int)}"


# -- evaluation --------------------------------------------------------------


@contextlib.contextmanager
def _swap_model(env: MjxEnv, model: mjx.Model):
    old = env.mjx_model
    try:
        env.mjx_model = model
        yield env
    finally:
        env.mjx_model = old


class DesignEvaluator:
    """Rolls designs out under the frozen policy and returns success rates.

    One env per (design, seed); the per-design model fields are vmapped over the
    same way `DomainRandomizationVmapWrapper` does during training, so training
    and evaluation see identical physics.
    """

    def __init__(
        self,
        env: MjxEnv,
        policy_fn: Callable,
        extra_obs_attrs: list[str],
        episode_length: int,
        num_seeds: int,
        max_batch: int = 4096,
    ):
        self.env = env
        self.policy_fn = policy_fn
        self.extra_obs_attrs = list(extra_obs_attrs)
        self.episode_length = episode_length
        self.num_seeds = num_seeds
        self.max_batch = max_batch
        self._attr_keys: Optional[tuple[str, ...]] = None
        self._rollout = None

    def _obs_fn(self, model: mjx.Model) -> jax.Array:
        return jnp.hstack(
            [getattr(model, attr).flatten() for attr in self.extra_obs_attrs]
        )

    def _build_rollout(self, attr_keys: tuple[str, ...]):
        base_model = self.env.mjx_model
        in_axes = jax.tree_util.tree_map(lambda x: None, base_model)
        in_axes = in_axes.tree_replace({k: 0 for k in attr_keys})

        def rollout(attr_values, rng):
            model_v = base_model.tree_replace(dict(zip(attr_keys, attr_values)))

            def single(model, key):
                with _swap_model(self.env, model) as env:
                    reset_key, scan_key = jax.random.split(key)
                    state = env.reset(reset_key)
                    state = state.replace(
                        obs=jnp.hstack([state.obs, self._obs_fn(model)])
                    )

                    def body(carry, _):
                        state, ever_success, ever_fell, key = carry
                        key, act_key = jax.random.split(key)
                        action, _ = self.policy_fn(state.obs, act_key)
                        state = env.step(state, action)
                        state = state.replace(
                            obs=jnp.hstack([state.obs, self._obs_fn(model)])
                        )
                        alive = 1.0 - jnp.maximum(ever_success, ever_fell)
                        ever_success = jnp.maximum(
                            ever_success, state.metrics["success"] * alive
                        )
                        ever_fell = jnp.maximum(
                            ever_fell, state.metrics["fell"] * alive
                        )
                        return (state, ever_success, ever_fell, key), None

                    carry = (state, jnp.zeros(()), jnp.zeros(()), scan_key)
                    (state, ever_success, ever_fell, _), _ = jax.lax.scan(
                        body, carry, (), self.episode_length
                    )
                    return ever_success, ever_fell, state.info["max_x"]

            return jax.vmap(single, in_axes=(in_axes, 0))(
                model_v, jax.random.split(rng, attr_values[0].shape[0])
            )

        return jax.jit(rollout)

    def __call__(self, attrs: dict[str, np.ndarray], seed: int = 0):
        attr_keys = tuple(sorted(attrs))
        if self._attr_keys != attr_keys:
            self._attr_keys = attr_keys
            self._rollout = self._build_rollout(attr_keys)

        n_designs = next(iter(attrs.values())).shape[0]
        per_call = max(1, self.max_batch // self.num_seeds)
        success, fell, max_x = [], [], []
        for start in range(0, n_designs, per_call):
            values = tuple(
                jnp.array(
                    np.repeat(attrs[k][start : start + per_call], self.num_seeds, 0)
                )
                for k in attr_keys
            )
            s, f, mx = self._rollout(values, jax.random.PRNGKey(seed + start))
            success.append(np.asarray(s))
            fell.append(np.asarray(f))
            max_x.append(np.asarray(mx))
        shape = (-1, self.num_seeds)
        return (
            np.concatenate(success).reshape(shape).mean(axis=1),
            np.concatenate(fell).reshape(shape).mean(axis=1),
            np.concatenate(max_x).reshape(shape).mean(axis=1),
        )


def build_evaluator(
    ckpt_path: str,
    num_seeds: int,
    max_batch: int = 4096,
    goal_x: Optional[float] = None,
) -> tuple[DesignEvaluator, dict]:
    cfg, policy_fn = load_run(ckpt_path)
    root = zarr.open(cfg.mj_model_data_path, mode="r")
    choices = list(root.attrs["choices"])
    num_uniforms = int(root.attrs["num_uniforms"])
    robogen_fn = hydra.utils.instantiate(root.attrs["robogen_fn"])
    data_attrs = list(root.keys())

    def mj_model_fn():
        mjcf_model = robogen_fn(
            choices=copy.deepcopy(choices), uniforms=[0.5] * num_uniforms
        )
        return mjcf.Physics.from_mjcf_model(build_stair_scene(mjcf_model)).model.ptr

    eval_cfg = cfg.eval_env
    if goal_x is not None:
        eval_cfg = eval_cfg.copy()
        eval_cfg.goal_x = goal_x
    env = hydra.utils.instantiate(eval_cfg, mj_model_fn=mj_model_fn)
    evaluator = DesignEvaluator(
        env=env,
        policy_fn=policy_fn,
        extra_obs_attrs=list(cfg.extra_obs_attrs),
        episode_length=int(cfg.env.max_episode_length),
        num_seeds=num_seeds,
        max_batch=max_batch,
    )
    space = dict(
        choices=choices,
        num_uniforms=num_uniforms,
        robogen_fn=robogen_fn,
        data_attrs=data_attrs,
    )
    return evaluator, space
