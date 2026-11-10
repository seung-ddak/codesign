"""Torch-only exponential moving average for diffusion network weights.

Maintains a distinct shadow copy without changing optimizer parameters.
The warmup-power schedule follows the common EMA power form:
decay(step) = min(max_decay, 1 - (1 + step / inv_gamma)**(-power)).
This is not a claim of byte-for-byte diffusers.EMAModel equivalence.
"""
import torch


class PowerEMA:
    def __init__(self, model, power=0.75, max_decay=0.9999, inv_gamma=1.0):
        if not 0 < power <= 1 or not 0 < max_decay < 1 or inv_gamma <= 0:
            raise ValueError("Invalid EMA schedule")
        self.power = float(power)
        self.max_decay = float(max_decay)
        self.inv_gamma = float(inv_gamma)
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        decay = min(self.max_decay, 1.0 - (1.0 + self.step / self.inv_gamma) ** (-self.power))
        for key, val in model.state_dict().items():
            if val.is_floating_point():
                self.shadow[key].mul_(decay).add_(val.detach(), alpha=1.0-decay)
            else:
                self.shadow[key].copy_(val.detach())
        return decay

    def state_dict(self):
        return {
            "step": self.step, "power": self.power,
            "max_decay": self.max_decay, "inv_gamma": self.inv_gamma,
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()}
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if abs(float(state["power"]) - self.power) > 1e-12:
            raise ValueError("EMA power differs from checkpoint")
        self.step = int(state["step"])
        if self.shadow.keys() != state["shadow"].keys():
            raise ValueError("EMA parameters differ from checkpoint")
        for key, val in state["shadow"].items():
            if self.shadow[key].shape != val.shape:
                raise ValueError("EMA shape differs from checkpoint: " + key)
            self.shadow[key].copy_(val.to(self.shadow[key].device))

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow)
