"""Episodic PPO over TT reverse-diffusion noise residuals (NOT locomotion PPO).

The pretrained TT remains frozen. One Gaussian action per physical rollout
specifies compact E / predicted-state / joint-action epsilon corrections.
Those same low-rank corrections are broadcast across the DDIM denoising steps.
Policy updates use only realized MuJoCo task rewards, never TT-predicted rewards.

This is a one-step episodic PPO ablation, not a state-dependent per-frame
controller and not equivalent to training PPO motion-tracking experts.
"""
import math

import torch
from torch import nn


class EpisodicResidualPPO(nn.Module):
    """Diagonal Gaussian policy; one continuous action per 10-second episode."""
    def __init__(self, num_windows, e_dim=13, state_features=4, joints=12):
        super().__init__()
        self.num_windows = int(num_windows)
        self.e_dim = int(e_dim)
        self.state_features = int(state_features)
        self.joints = int(joints)
        self.dim = self.e_dim + self.num_windows * (self.state_features + self.joints)
        self.mean = nn.Parameter(torch.zeros(self.dim))
        self.log_std = nn.Parameter(torch.full((self.dim,), -0.7))
        self.value = nn.Parameter(torch.tensor(0.0))

    def distribution(self):
        return torch.distributions.Normal(self.mean, self.log_std.clamp(-3, 1).exp())

    def draw(self, count):
        dist = self.distribution()
        raw = dist.sample((count,)).detach()
        logp = dist.log_prob(raw).sum(-1).detach()
        return raw, logp

    def log_prob(self, raw):
        return self.distribution().log_prob(raw).sum(-1)

    def residual_fn(self, raw, *, e_scale=0.08, s_scale=0.06,
                    a_scale=0.08, inactive_e=(0, 11, 12)):
        """Return the exact denoising_policy callback expected by TT sampler."""
        if raw.ndim != 2 or raw.shape[1] != self.dim:
            raise ValueError("RL residual action has the wrong dimensions")
        n = len(raw)
        e = e_scale * raw[:, :self.e_dim].tanh()
        e = e.clone()
        if inactive_e:
            e[:, list(inactive_e)] = 0
        blocks = raw[:, self.e_dim:].reshape(
            n, self.num_windows, self.state_features + self.joints).tanh()

        def callback(xs, noisy_e, step):
            if len(xs) != self.num_windows or noisy_e.shape != (n, self.e_dim):
                raise ValueError("RL denoising window or morphology shape mismatch")
            corrections = []
            for i, xt in enumerate(xs):
                corr = torch.zeros_like(xt)
                # XY positions and heading-linear XY velocities are modified
                # only on non-anchor timesteps, respecting s0 conditioning.
                corr[:, 1:, 0:2] = s_scale * blocks[:, i, 0:2, None].transpose(1, 2)
                corr[:, 1:, 8:10] = s_scale * blocks[:, i, 2:4, None].transpose(1, 2)
                corr[:, :, 42:54] = a_scale * blocks[:, i, 4:, None].transpose(1, 2)
                corrections.append(corr)
            return e.to(noisy_e), corrections
        return callback


def ppo_update(policy, optimizer, raw_actions, old_log_probs, rewards,
               *, epochs=4, clip_ratio=0.2, value_weight=0.05):
    """On-policy clipped PPO update for one-step physical-rollout episodes."""
    raw = raw_actions.detach()
    old_lp = old_log_probs.detach()
    returns = rewards.detach().to(raw.device, dtype=raw.dtype)
    if not torch.isfinite(returns).all():
        raise ValueError("Non-finite physical rewards")
    baseline = policy.value.detach()
    advantage = returns - baseline
    # Batch normalized advantages: best of a batch must have better probability.
    if len(advantage) > 1:
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-5)
    losses = []
    for _ in range(epochs):
        new_lp = policy.log_prob(raw)
        ratio = (new_lp - old_lp).clamp(-20, 20).exp()
        clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
        actor_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
        critic_loss = (policy.value - returns).square().mean()
        loss = actor_loss + value_weight * critic_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append((float(actor_loss.detach()), float(critic_loss.detach())))
    return {
        "ppo_actor_loss": losses[-1][0],
        "ppo_value_loss": losses[-1][1],
        "reward_mean": float(returns.mean()),
        "reward_best": float(returns.max()),
        "policy_std_mean": float(policy.log_std.clamp(-3, 1).exp().mean().detach()),
    }
