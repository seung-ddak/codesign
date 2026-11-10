"""Checks the physics/reward boundary used by the RL replacement pilot."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/tt_dynamics'))
from train_task_rl import TaskEnv


@pytest.fixture(params=['zigzag', 'circle'])
def env(request):
    task = request.param
    p = ROOT / f'runs/pipeline_v1/fast_mode24_v16_twin/evaluation/check_w1/joint_{task}_4s/scale_10.npz'
    return TaskEnv(dict(np.load(p)), 0, task)


def test_incremental_reward_equals_original_trajectory_reward(env):
    total = 0.
    for _ in range(10):
        obs, r, done, fell = env.step(np.zeros(12))
        total += r
        if done:
            break
    assert total == pytest.approx(env.score()[0] - (20. if fell else 0.), abs=1e-5)
    assert obs.shape == (68,) and np.isfinite(obs).all()
    np.testing.assert_array_equal(obs[42:55], env.E)


def test_reset_and_action_units(env):
    initial = env.reset().copy()
    env.step(np.full(12, 100.))
    np.testing.assert_allclose(env.data.ctrl, env.home + 2.5)
    np.testing.assert_allclose(env.reset(), initial)
    target = env.home + .1
    env.step(target, target=True)
    np.testing.assert_allclose(env.data.ctrl, target)
    with pytest.raises(ValueError):
        env.step(np.full(12, np.nan))


def test_fall_ends_episode_and_has_training_only_penalty(env):
    env.data.qpos[3:7] = [0., 1., 0., 0.]
    _, reward, done, fell = env.step(np.zeros(12))
    assert done and fell
    assert reward == pytest.approx(env.score()[0] - 20., abs=1e-5)


def test_reused_ppo_completes_finite_update():
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = ActorCritic(68, 68, 12, actor_hidden_dims=[16], critic_hidden_dims=[16])
    ppo = PPO(policy, num_learning_epochs=1, num_mini_batches=1)
    ppo.init_storage(2, 4, [68], [68], [12])
    with torch.no_grad():
        obs = torch.zeros(2, 68)
        for _ in range(4):
            ppo.act(obs, obs)
            ppo.process_env_step(torch.tensor([.1, -.1]), torch.zeros(2, dtype=torch.bool), {})
        ppo.compute_returns(obs)
    assert np.isfinite(ppo.update()).all()


@pytest.mark.parametrize('task', ['zigzag', 'circle'])
def test_dense_reward_prefers_progress_over_standing_reverse_and_off_path(task):
    from train_task_rl import dense_task_reward
    previous = np.zeros(42)
    previous[2], previous[3], previous[7] = .22, 1., -1.
    forward = previous.copy()
    if task == 'zigzag':
        tangent = np.array([1., .6]); tangent /= np.linalg.norm(tangent)
        forward[:2] = .01*tangent
    else:
        forward[:2] = [np.sin(.01), 1.-np.cos(.01)]
    reverse = previous.copy(); reverse[:2] = -forward[:2]
    off = forward.copy(); off[1] += .5
    def reward(s, t=30, delta=None, torque=None):
        return dense_task_reward(previous, s, np.zeros(12) if delta is None else delta,
                                 np.zeros(12) if torque is None else torque, task, t, .22)[0]
    assert reward(forward) > reward(previous) > reward(reverse)
    assert reward(forward) > reward(off)
    assert reward(previous, 30) < reward(previous, 1)
    assert reward(forward) > reward(forward, delta=np.ones(12), torque=np.full(12, 33.5))
    low = forward.copy(); low[2] = .05
    assert reward(forward) > reward(low)


def test_dense_env_rollout_uses_finite_reward(env):
    env.reward_version = 'dense'
    env.reset()
    rewards = [env.step(np.zeros(12))[1] for _ in range(8)]
    assert np.isfinite(rewards).all()
    assert env.training_return == pytest.approx(sum(rewards))
