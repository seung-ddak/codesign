"""Load a motion-tracking PPO checkpoint for evaluation."""

import dataclasses
import pickle
from pathlib import Path

import hydra
import jax.numpy as jnp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from orbax import checkpoint as ocp


def load_normalizer(saved):
    """orbax restores the dataclass as a dict; UInt64 count comes back as {hi, lo}."""
    if not isinstance(saved, dict):
        return saved
    fields = {f.name for f in dataclasses.fields(running_statistics.RunningStatisticsState)}
    return running_statistics.RunningStatisticsState(**{
        k: jnp.asarray(sum(jnp.asarray(x) for x in v.values()) if isinstance(v, dict) else v)
        for k, v in saved.items() if k in fields
    })


def load_run(ckpt_path):
    path = Path(ckpt_path)
    with (path.parent.parent / "config.pkl").open("rb") as file:
        cfg = pickle.load(file)
    params = ocp.PyTreeCheckpointer().restore(str(path.resolve()))
    policy_params = params[1]["params"]
    action_size = len(policy_params["std_param"]["value"])
    obs_size = policy_params["MLP_0"]["hidden_0"]["kernel"].shape[0]
    normalize = running_statistics.normalize if cfg.train.normalize_observations else lambda x, y: x
    network = hydra.utils.instantiate(cfg.train.network_factory)(
        obs_size, action_size, preprocess_observations_fn=normalize
    )
    policy = ppo_networks.make_inference_fn(network)((load_normalizer(params[0]), params[1]), deterministic=True)
    return cfg, policy
