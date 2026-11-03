"""Exercise RL inside reverse diffusion, without reward differentiation."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/tt_dynamics'))
from joint_diffusion_guidance import joint_diffusion_full_reward
from rl_diffusion_guidance import DenoisingPolicy


class Net(torch.nn.Module):
    def forward(self, x, e, *args):
        return torch.zeros_like(x), torch.zeros_like(e)


def forbidden_reward(*args):
    raise AssertionError('RL sampling must not call the differentiable reward')


def sample(policy=None):
    return joint_diffusion_full_reward(Net(), forbidden_reward, n=2, episode_states=7, horizon=4,
           state_dim=42, action_dim=12, e_dim=13, mu=torch.zeros(54), sd=torch.ones(54),
           beta=torch.tensor([.1,.3]), abar=torch.tensor([.9,.63]), guidance_scale=0.,
           clip=5., seed=42, denoising_policy=policy)


def test_zero_rl_matches_prior_and_nonzero_rl_changes_joint_tokens(monkeypatch):
    monkeypatch.setattr(torch.autograd, 'grad', lambda *a, **k: (_ for _ in ()).throw(AssertionError('reward gradient called')))
    base, be, _ = sample()
    controller = DenoisingPolicy(episode_states=7, horizon=4, diffusion_steps=2)
    zero, ze, audit = sample(controller)
    assert torch.equal(base, zero) and torch.equal(be, ze)
    with torch.no_grad():
        controller.policy.actor[-1].bias.fill_(.5)
    changed, ce, audit = sample(controller)
    assert not torch.equal(be[:,1:11], ce[:,1:11])
    assert not torch.equal(base[..., :42], changed[..., :42])
    assert not torch.equal(base[..., 42:], changed[..., 42:])
    assert torch.equal(ce[:,[0,11,12]], torch.full((2,3),.5))
    assert all(not audit[f'{k}_gradient_nonzero'] for k in ('e','s','a'))
    assert len(audit['rl_correction_norm_per_step']) == 2


def test_invalid_rl_correction_is_rejected():
    def bad(xs, e, k):
        return torch.full_like(e,float('nan')), [torch.zeros_like(x) for x in xs]
    with pytest.raises(ValueError, match='finite'):
        sample(bad)


def test_checkpoint_roundtrip_and_mismatch(tmp_path):
    controller = DenoisingPolicy(episode_states=7, horizon=4, diffusion_steps=2)
    path = tmp_path/'policy.pt'
    controller.save(path, {'task':'zigzag'})
    restored = DenoisingPolicy.load(path, episode_states=7, horizon=4, diffusion_steps=2)
    assert torch.equal(sample(controller)[0],sample(restored)[0])
    with pytest.raises(ValueError, match='episode_states'):
        DenoisingPolicy.load(path, episode_states=8, horizon=4, diffusion_steps=2)


def test_ppo_collects_denoising_steps_and_updates_policy(monkeypatch):
    monkeypatch.setitem(sys.modules,'triton',None)
    from rsl_rl.algorithms import PPO
    controller = DenoisingPolicy(episode_states=7, horizon=4, diffusion_steps=2)
    algo = PPO(controller.policy, num_learning_epochs=1, num_mini_batches=1)
    algo.init_storage(2,2,[controller.obs_dim],[controller.obs_dim],[controller.action_dim])
    controller.algorithm = algo
    old = controller.policy.actor[-1].weight.detach().clone()
    sample(controller)
    controller.finish(torch.tensor([1.,-1.]))
    assert algo.storage.step == 2
    metrics = algo.update()
    assert torch.isfinite(torch.tensor(metrics)).all()
    assert not controller.pending
    assert not torch.equal(old,controller.policy.actor[-1].weight)


def test_invalid_generated_start_is_penalized_before_execution(monkeypatch):
    import train_diffusion_rl as runner
    monkeypatch.setattr(runner, 'TaskEnv', lambda *a, **k: (_ for _ in ()).throw(AssertionError('invalid start executed')))
    x = torch.zeros(1,4,54)
    x[:,:,2],x[:,:,7] = .2,1.
    rewards, rows, saved = runner.replay_generated(x, torch.full((1,13),.5), 'zigzag')
    assert rewards.tolist() == [-20.]
    assert rows[0]['invalid_initial'] and not rows[0]['survived'] and rows[0]['steps'] == 0
