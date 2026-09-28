"""Run directly: slot i of a structure group simulates design i % K with its own friction."""

import os
import sys
from pathlib import Path

TT = Path(os.environ.get("CODESIGN_TT", Path(__file__).resolve().parents[3] / "transformer-transformer"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(TT))
os.chdir(TT)

import jax
import jax.numpy as jp
import numpy as np

from multi_design_env import FRICTION_RANGE, MultiDesignEnv, structure_key


def main():
    g1 = [([0, 0, 0, 0, 0, 0, 0, b], s) for b in (0, 1) for s in (0, 5)]
    g2 = [([0, 1, 1, 1, 1, 1, 1, b], s) for b in (0, 1) for s in (0, 5)]
    assert len({structure_key(c) for c, _ in g1}) == 1 and structure_key(g1[0][0]) != structure_key(g2[0][0])
    env = MultiDesignEnv([g1, g2], 1000)
    n = 16
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), n))
    morph = np.asarray(state.obs[:, 48:])
    for gi, group in enumerate(env.groups):
        inner = group.env.env  # AutoReset -> Episode -> DesignGroupEnv
        for slot in range(n // 2):
            assert np.allclose(morph[gi * n // 2 + slot], inner.envs[slot % 4].morphology), (gi, slot)
    # Design-specific geometry reaches the physics: the two seeds of one choice differ.
    sub = state.info["groups"][0]
    assert not np.allclose(sub.data.xpos[0], sub.data.xpos[1])
    friction = np.random.RandomState(0).uniform(*FRICTION_RANGE, n // 2)
    assert len(np.unique(friction)) == n // 2 and FRICTION_RANGE[0] <= friction.min()
    state = jax.jit(env.step)(state, jp.zeros((n, env.action_size)))
    assert state.obs.shape == (n, 69) and np.isfinite(np.asarray(state.reward)).all()
    assert set(state.info) >= {"truncation", "episode_metrics", "episode_done", "groups"}
    # Auto-reset must not wipe the finished episode before the PPO metrics logger reads it.
    env = MultiDesignEnv([g1], 3)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    step = jax.jit(env.step)
    for _ in range(3):
        state = step(state, jp.zeros((4, env.action_size)))
    assert (np.asarray(state.info["episode_done"]) == 1).all()
    assert (np.asarray(state.info["episode_metrics"]["length"]) == 3).all()
    state = step(state, jp.zeros((4, env.action_size)))
    assert (np.asarray(state.info["episode_metrics"]["length"]) == 1).all()
    print("PASS")


if __name__ == "__main__":
    main()
