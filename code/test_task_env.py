"""Minimal wiring check for the copied TT task environment."""

import os
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root.parent / "transformer-transformer")
    env = hydra.utils.instantiate(OmegaConf.load(root / "code/task_env.yaml"))
    state = env.step(env.reset(jax.random.PRNGKey(1)), jnp.zeros(env.action_size))
    assert state.obs.shape == (114,)
    assert env.reward_exp_pose_err == env.reward_pos_err == env.reward_orn_err == 0
    assert state.info["task_command"].shape == (3,)
    assert state.info["done/pos_err"] == state.info["done/orn_err"] == 0
    assert {"forward", "jump_air", "jump_height", "stairs_progress", "stairs_height"} <= state.metrics.keys()
    print("task environment wiring ok")


if __name__ == "__main__":
    main()
