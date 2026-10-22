"""Full-horizon shared-morphology optimizer.

SPSA estimates a directional gradient of the *complete predicted task reward*
with respect to one morphology E shared by every TT window. Each evaluation
must use identical random noise; the caller owns that deterministic rollout.
This is gradient estimation, not backpropagation through MJX or through the
full sequence of TT denoising steps. MJX is reserved for final evaluation.
"""
import torch


@torch.no_grad()
def optimize_shared_e(e_init, score_fn, *, steps=4, perturb=0.04, step_size=0.08,
                      seed=0, low=0.05, high=0.95, inactive=(0, 11, 12)):
    """Projected, accepted-only SPSA ascent; score_fn(E[B,13]) -> [B].

    At every trial both perturbations and proposal are evaluated under common
    rollout noise (score_fn must enforce that). All active E dimensions can
    influence the full-horizon reward. Returns E, scores, and JSON-safe log.
    """
    if e_init.ndim != 2 or e_init.shape[-1] != 13:
        raise ValueError("expected E [batch,13]")
    if steps < 0 or not (perturb > 0 and step_size > 0 and 0 <= low < high <= 1):
        raise ValueError("invalid shared-E optimization settings")
    mask = torch.ones_like(e_init)
    mask[:, list(inactive)] = 0
    center = e_init.clone().clamp(low, high)
    center[:, list(inactive)] = 0.5
    generator = torch.Generator(device=e_init.device).manual_seed(seed)
    def evaluate(e):
        s = score_fn(e)
        if s.shape != (len(e),):
            raise ValueError("score_fn must return [batch] reward")
        return torch.where(torch.isfinite(s), s, torch.full_like(s, -1e12))
    current = evaluate(center)
    history = [{"iteration": 0, "predicted_reward": current.detach().cpu().tolist(),
                "accepted": [False] * len(center)}]
    for iteration in range(1, steps + 1):
        direction = (torch.randint(0, 2, center.shape, generator=generator,
                                   device=center.device) * 2 - 1).to(center.dtype) * mask
        plus = (center + perturb * direction).clamp(low, high)
        minus = (center - perturb * direction).clamp(low, high)
        # At clipped coordinates this is an approximate projected direction.
        plus[:, list(inactive)] = minus[:, list(inactive)] = 0.5
        rp, rm = evaluate(plus), evaluate(minus)
        g = ((rp - rm) / (2 * perturb))[:, None] * direction
        gnorm = torch.linalg.vector_norm(g, dim=1, keepdim=True)
        proposal = (center + step_size * g / gnorm.clamp(min=1e-12)).clamp(low, high)
        proposal[:, list(inactive)] = 0.5
        next_score = evaluate(proposal)
        accepted = (gnorm.squeeze(1) > 1e-12) & (next_score > current)
        center = torch.where(accepted[:, None], proposal, center)
        current = torch.where(accepted, next_score, current)
        history.append({"iteration": iteration, "predicted_reward": current.detach().cpu().tolist(),
                        "accepted": accepted.detach().cpu().tolist(),
                        "plus_reward": rp.detach().cpu().tolist(),
                        "minus_reward": rm.detach().cpu().tolist()})
    return center, current, history
