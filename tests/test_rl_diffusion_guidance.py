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


def test_fixed_prediction_reuses_tt_and_corrects_all_tokens():
    class CountedNet(Net):
        calls = 0
        def forward(self, *args):
            self.calls += 1
            return super().forward(*args)
    net = CountedNet()
    cache = {}
    controller = DenoisingPolicy(episode_states=7, horizon=4, diffusion_steps=1)
    def draw(seed=42):
        return joint_diffusion_full_reward(net, forbidden_reward, n=2, episode_states=7, horizon=4,
            state_dim=42, action_dim=12, e_dim=13, mu=torch.zeros(54), sd=torch.ones(54),
            beta=torch.tensor([.1,.3]), abar=torch.tensor([.9,.63]), guidance_scale=0.,
            clip=5., seed=seed, denoising_policy=controller, fixed_prediction_cache=cache)
    before, eb, audit = draw()
    calls = net.calls
    assert calls == 2 and audit['diffusion_steps'] == 1
    with torch.no_grad():
        controller.policy.actor[-1].bias.fill_(.5)
    after, ea, audit = draw()
    assert net.calls == calls and cache['tt_prediction_passes'] == 1
    assert not torch.equal(before[...,:42], after[...,:42])
    assert not torch.equal(before[...,42:], after[...,42:])
    assert not torch.equal(eb, ea)
    assert len(audit['rl_correction_norm_per_step']) == 1
    with pytest.raises(ValueError, match='identical initial noise'):
        draw(seed=43)


def test_fixed_prediction_ppo_uses_one_transition_per_candidate(monkeypatch):
    monkeypatch.setitem(sys.modules, 'triton', None)
    from rsl_rl.algorithms import PPO
    controller = DenoisingPolicy(episode_states=7, horizon=4, diffusion_steps=1)
    algo = PPO(controller.policy, num_learning_epochs=1, num_mini_batches=1)
    algo.init_storage(2,1,[controller.obs_dim],[controller.obs_dim],[controller.action_dim])
    controller.algorithm = algo
    cache = {}
    for _ in range(2):
        joint_diffusion_full_reward(Net(), forbidden_reward, n=2, episode_states=7, horizon=4,
            state_dim=42, action_dim=12, e_dim=13, mu=torch.zeros(54), sd=torch.ones(54),
            beta=torch.tensor([.1,.3]), abar=torch.tensor([.9,.63]), guidance_scale=0.,
            seed=42, denoising_policy=controller, fixed_prediction_cache=cache)
        controller.finish(torch.tensor([1.,-1.]))
        assert algo.storage.step == 1
        assert torch.isfinite(torch.tensor(algo.update())).all()


def test_shared_window_policy_reloads_at_different_episode_lengths(tmp_path):
    controller = DenoisingPolicy(episode_states=7,horizon=4,diffusion_steps=1,shared_window_actor=True)
    with torch.no_grad():
        controller.policy.actor.e_head[-1].bias.fill_(.2)
        controller.policy.actor.sa_head[-1].bias.fill_(.3)
    path = tmp_path/'shared.pt'
    controller.save(path, {'task':'zigzag'})
    for states in (4,7,13):
        loaded = DenoisingPolicy.load(path,episode_states=states,horizon=4,diffusion_steps=1)
        xs = [torch.zeros(2,4,54) for _ in range(loaded.windows)]
        e = torch.zeros(2,13)
        ce, cx = loaded(xs,e,0)
        assert ce.shape == (2,13) and len(cx) == loaded.windows
        assert all(torch.equal(c,cx[0]) for c in cx)
        obs = loaded.observation(xs,e,0)
        loaded.policy.update_distribution(obs)
        assert loaded.policy.action_std.shape == (2,loaded.action_dim)
        assert loaded.policy.std.numel() == 67
        assert loaded.policy.evaluate(obs).shape == (2,1)
        assert sum(p.numel() for p in loaded.policy.parameters()) == sum(p.numel() for p in controller.policy.parameters())


def test_shared_window_ppo_trains_global_e_and_shared_sa(monkeypatch):
    monkeypatch.setitem(sys.modules, 'triton', None)
    from rsl_rl.algorithms import PPO
    c = DenoisingPolicy(episode_states=7,horizon=4,diffusion_steps=1,shared_window_actor=True)
    algo = PPO(c.policy,num_learning_epochs=2,num_mini_batches=1)
    algo.init_storage(2,1,[c.obs_dim],[c.obs_dim],[c.action_dim])
    c.algorithm=algo
    ew=c.policy.actor.e_head[-1].weight.detach().clone()
    sw=c.policy.actor.sa_head[-1].weight.detach().clone()
    xs=[torch.randn(2,4,54) for _ in range(c.windows)]
    ce,cx=c(xs,torch.zeros(2,13),0)
    assert ce.shape==(2,13) and len(cx)==2
    c.finish(torch.tensor([1.,-1.]))
    assert torch.isfinite(torch.tensor(algo.update())).all()
    assert not torch.equal(ew,c.policy.actor.e_head[-1].weight)
    assert not torch.equal(sw,c.policy.actor.sa_head[-1].weight)
