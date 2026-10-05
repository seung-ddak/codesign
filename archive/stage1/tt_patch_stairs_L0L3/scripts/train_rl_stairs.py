"""Train the morphology-conditioned stair-climbing policy (remind.md §4.2 pi_theta).

Same structure as `train_rl_procedural.py`, but the task is outcome-based stair
climbing instead of end-effector tracking, and the scene carries the staircase.
The morphology attributes are domain-randomised per env and appended to the
observation, so one policy covers the whole continuous design slice.
"""

import copy
import os
import pickle

import jax
import jax.numpy as jnp
from absl import logging

logging.set_verbosity(logging.WARNING)

import functools

from t2.mjx_env.base_env import MjxEnv, State
from t2.mjx_env.wrapper import wrap_for_brax_training

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import hydra
import zarr
from dm_control import mjcf
from flax.training import orbax_utils
from mujoco import mjx
from omegaconf import OmegaConf
from orbax import checkpoint as ocp

import wandb
from t2.mjx_env.stair_env import build_stair_scene


@hydra.main(config_path="../config", config_name="run_rl_stairs", version_base="1.2")
def main(cfg):
    OmegaConf.resolve(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="transformer-transformer",
        tags=cfg.wandb.tags,
        config=config_dict,
    )
    pickle.dump(cfg, open(f"{wandb.run.dir}/config.pkl", "wb"))

    root = zarr.open(cfg.mj_model_data_path, mode="r")
    hardware_dict = {k: jnp.array(root[k][:]) for k in root.keys()}
    num_hardware = next(iter(hardware_dict.values())).shape[0]
    choices = root.attrs["choices"]
    robogen_fn = hydra.utils.instantiate(root.attrs["robogen_fn"])
    num_uniforms = int(root.attrs["num_uniforms"])

    ckpt_path = wandb.run.dir + "/checkpoints"
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    def mj_model_fn(
        num_uniforms: int = num_uniforms,
        choices: list[int] = choices,
    ):
        mjcf_model = robogen_fn(
            choices=copy.deepcopy(choices), uniforms=[0.5] * num_uniforms
        )
        return mjcf.Physics.from_mjcf_model(build_stair_scene(mjcf_model)).model.ptr

    def randomization_fn(
        model: mjx.Model,
        rng: jax.Array,
        _hardware_dict: dict[str, jax.Array] = hardware_dict,
        _num_hardware: int = num_hardware,
    ):
        @jax.vmap
        def rand(rng):
            rng, key = jax.random.split(rng)
            hardware_idx = jax.random.randint(key, (), 0, _num_hardware)
            return {k: _hardware_dict[k][hardware_idx] for k in _hardware_dict.keys()}

        hardware_data = rand(rng)
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({k: 0 for k in _hardware_dict.keys()})
        model = model.tree_replace(hardware_data)  # type: ignore
        return model, in_axes

    env: MjxEnv = hydra.utils.instantiate(cfg.env, mj_model_fn=mj_model_fn)
    eval_env: MjxEnv = hydra.utils.instantiate(cfg.eval_env, mj_model_fn=mj_model_fn)

    def randomization_obs_fn(model: mjx.Model):
        return jnp.hstack(
            [getattr(model, attr).flatten() for attr in cfg.extra_obs_attrs]
        )

    jit_randomization_obs_fn = jax.jit(randomization_obs_fn)

    def policy_params_fn(current_step, make_policy, params):
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        orbax_checkpointer.save(
            ckpt_path + f"/{current_step}", params, force=True, save_args=save_args
        )

    def progress(num_steps, metrics):
        if any(jnp.isnan(v) for v in jax.tree_util.tree_leaves(metrics)):
            print(f"[{num_steps:06d}] NaN metrics detected")
            raise ValueError("NaN metrics detected")
        wandb.log(metrics, step=num_steps)
        print("-" * 40 + f"{num_steps}" + "-" * 40)
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("-" * 80, flush=True)

    hydra.utils.instantiate(
        cfg.train,
        policy_params_fn=policy_params_fn,
        wrap_env_fn=functools.partial(
            wrap_for_brax_training,
            full_reset=cfg.full_reset,
            randomization_obs_fn=(
                jit_randomization_obs_fn
                if cfg.observe_domain_randomized_attrs
                else None
            ),
        ),
        randomization_fn=randomization_fn,
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )


if __name__ == "__main__":
    main()
