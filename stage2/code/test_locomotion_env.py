"""Run directly: T3 env interface/obs layout, T4 contact forces, termination and reward."""

import os
import sys
from pathlib import Path

TT = Path(os.environ.get("CODESIGN_TT", Path(__file__).resolve().parents[3] / "transformer-transformer"))
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(TT))
os.chdir(TT)  # TT component MJCF paths are repo-relative

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from t2.mjx_env.base_env import make_data
from locomotion_env import (EPISODE_STEPS, ONLY_POSITIVE_REWARDS, PUSH_STEPS, RESAMPLE_STEPS, LocomotionEnv,
                            quat_rotate)


def main():
    yaw90 = jp.array([np.cos(np.pi / 4), 0., 0., np.sin(np.pi / 4)])
    assert np.allclose(quat_rotate(yaw90, jp.array([1., 0., 0.])), [0., 1., 0.], atol=1e-6)
    assert np.allclose(quat_rotate(yaw90, jp.array([0., 1., 0.]), inverse=True), [1., 0., 0.], atol=1e-6)
    assert (EPISODE_STEPS, RESAMPLE_STEPS, PUSH_STEPS) == (1000, 500, 750)

    # 00000000: crouches onto a calf after a randomized reset (penalised, not terminated).
    # 01111001: stands with 10 cm thigh clearance.
    for choices, check in (([0, 0, 0, 0, 0, 0, 0, 0], "kneel"), ([0, 1, 1, 1, 1, 0, 0, 1], "stand")):
        env = LocomotionEnv(choices, seed=0, add_noise=False)
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        obs = np.asarray(state.obs)
        assert obs.shape == (48 + 21,), obs.shape
        assert np.allclose(obs[6:9], [0., 0., -1.], atol=1e-3), obs[6:9]  # upright at reset
        assert np.allclose(obs[48:], env.morphology)

        def advance(state, _):
            state = env.step(state, jp.zeros(env.action_size))
            feet = env.contact_forces(state.data, jp.array(env.feet))[:, 2].sum()
            return state, (state.info["command"], state.data.qpos[env.qadr + 2], state.reward,
                           state.done, feet, state.metrics["collision"])

        rollout = jax.jit(lambda s, n: jax.lax.scan(advance, s, None, n), static_argnums=1)
        if check == "kneel":
            state, (commands, height, reward, done, _, collision) = rollout(state, 760)
            commands, height, reward, done, collision = map(
                np.asarray, (commands, height, reward, done, collision))
            assert int(state.info["episode_step"]) == 760
            assert np.isfinite(np.asarray(state.obs)).all() and np.isfinite(height).all()
            assert (commands[RESAMPLE_STEPS - 2, [0, 1, 3]] != commands[RESAMPLE_STEPS - 1, [0, 1, 3]]).any()
            assert np.isfinite(reward).all() and (not ONLY_POSITIVE_REWARDS or (reward >= 0).all())
            # Calf contact is a collision penalty (-1 * dt per body), not a termination.
            assert (collision < 0).sum() > 100 and not done.any(), ((collision < 0).sum(), done.any())
            assert height.min() > 0.1, "trunk sank"
            # Trunk contact ends the episode: drop the robot upside down.
            flipped = state.replace(data=mjx.forward(env.mjx_model, make_data(
                env.mj_model, qpos=env.home.at[env.qadr + 3:env.qadr + 7].set(
                    jp.array([0., 1., 0., 0.])), ctrl=env.ctrl, impl="jax")))
            _, (_, flip_z, _, flip_done, _, _) = rollout(flipped, 100)
            flip_done, flip_z = np.asarray(flip_done), np.asarray(flip_z)
            assert flip_done.any(), "no termination on trunk contact"
            fell = int(np.argmax(flip_done))
            assert flip_z[fell:].min() > -0.05, "trunk sank through the floor"
            print(choices, "nu", env.mj_model.nu, "penalised steps", (collision < 0).sum(),
                  "flipped done at step", fell, "reward mean", reward.mean().round(4))
        else:
            # From the home pose, zero action holds the stance without termination and the
            # feet carry the robot's weight (net contact force sign and scale).
            home = state.replace(data=mjx.forward(env.mjx_model, make_data(
                env.mj_model, qpos=env.home, ctrl=env.ctrl, impl="jax")))
            _, (_, stand, reward, done, feet, _) = rollout(home, PUSH_STEPS - 1)
            stand, feet = np.asarray(stand), np.asarray(feet)
            weight = env.mj_model.body_subtreemass[1] * 9.81
            assert stand.min() > 0.15 and not np.asarray(done).any(), stand.min()
            assert abs(feet[-100:].mean() / weight - 1) < 0.05, (feet[-100:].mean(), weight)
            print(choices, "nu", env.mj_model.nu, "home stance z min", stand.min().round(3),
                  "feet Fz / mg", (feet[-100:].mean() / weight).round(3),
                  "reward mean", np.asarray(reward).mean().round(4))
    print("PASS")


if __name__ == "__main__":
    main()
