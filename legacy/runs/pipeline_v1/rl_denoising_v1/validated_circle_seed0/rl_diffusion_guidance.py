"""PPO residuals inside joint E/S/A denoising, without reward differentiation."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / 'quad/rsl_rl'))
from rsl_rl.modules import ActorCritic
from joint_diffusion_guidance import _windows


class DenoisingPolicy:
    def __init__(self, *, episode_states, horizon, diffusion_steps, xdim=54, edim=13,
                 correction_scale=.25, device='cpu'):
        if correction_scale <= 0 or not torch.isfinite(torch.tensor(correction_scale)):
            raise ValueError('correction_scale must be finite and positive')
        self.config = dict(episode_states=episode_states, horizon=horizon, diffusion_steps=diffusion_steps,
                           xdim=xdim, edim=edim, correction_scale=correction_scale)
        self.windows = len(_windows(episode_states, horizon))
        self.obs_dim = 2*self.windows*xdim + edim + 1
        self.action_dim = self.windows*xdim + edim
        self.policy = ActorCritic(self.obs_dim, self.obs_dim, self.action_dim,
                                 actor_hidden_dims=[64,64], critic_hidden_dims=[64,64],
                                 init_noise_std=.1, max_std=.3).to(device)
        torch.nn.init.zeros_(self.policy.actor[-1].weight)
        torch.nn.init.zeros_(self.policy.actor[-1].bias)
        self.algorithm = None
        self.pending = False

    def observation(self, xs, e, k):
        if len(xs) != self.windows or e.shape[-1] != self.config['edim']:
            raise ValueError('policy dimensions differ from sampler')
        pooled = [v for x in xs for v in (x.mean(1), x.std(1, unbiased=False))]
        time = e.new_full((len(e),1), k/max(1,self.config['diffusion_steps']-1))
        obs = torch.cat(pooled + [e, time], dim=1).detach()
        if obs.shape[1] != self.obs_dim or not torch.isfinite(obs).all():
            raise ValueError('invalid denoising observation')
        return obs

    def __call__(self, xs, e, k):
        obs = self.observation(xs, e, k)
        with torch.no_grad():
            if self.algorithm is None:
                action = self.policy.act_inference(obs)
            else:
                if self.pending:
                    self.algorithm.process_env_step(obs.new_zeros(len(e)),
                                                    torch.zeros(len(e), dtype=torch.bool, device=e.device), {})
                action = self.algorithm.act(obs, obs)
                self.pending = True
            # shortcut: one epsilon offset per channel/window, use tokenwise control if this basis is insufficient.
            correction = self.config['correction_scale'] * torch.tanh(action)
            ce = correction[:, :self.config['edim']]
            offsets = correction[:, self.config['edim']:].reshape(len(e), self.windows, self.config['xdim'])
            return ce, [offsets[:,i,None,:].expand_as(x) for i,x in enumerate(xs)]

    def finish(self, physical_rewards):
        if self.algorithm is None or not self.pending:
            raise RuntimeError('no pending PPO denoising transition')
        if not torch.isfinite(physical_rewards).all():
            raise ValueError('physical rewards must be finite')
        self.algorithm.process_env_step(physical_rewards,
                                        torch.ones_like(physical_rewards, dtype=torch.bool), {})
        self.pending = False
        with torch.no_grad():
            self.algorithm.compute_returns(physical_rewards.new_zeros(len(physical_rewards), self.obs_dim))

    def save(self, path, metadata):
        torch.save(dict(config=self.config, state_dict=self.policy.state_dict(), metadata=metadata), path)

    @classmethod
    def load(cls, path, *, episode_states, horizon, diffusion_steps, device='cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = checkpoint['config']
        for key, expected in [('episode_states',episode_states),('horizon',horizon),('diffusion_steps',diffusion_steps)]:
            if config[key] != expected:
                raise ValueError(f'RL checkpoint {key}={config[key]} differs from sampler {expected}')
        obj = cls(**config, device=device)
        obj.policy.load_state_dict(checkpoint['state_dict'])
        obj.policy.eval()
        obj.metadata = checkpoint['metadata']
        return obj
