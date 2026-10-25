"""Shared-E reward-guided *joint diffusion* over overlapping TT windows.

Unlike outer-loop SPSA morphology search, this method guides the noisy E
token during each reverse-diffusion step, jointly with noisy S/A window tokens.

At diffusion step k:
  1. one noisy E latent is denoised by first (Mode 4) window;
  2. every later (Mode 2) window is conditionally denoised using that same
     provisional E and the prior window's predicted endpoint;
  3. the full 10-second differentiable reward on concatenated predicted
     states/actions is differentiated w.r.t. the noisy E AND X latents;
  4. the resulting reward gradient adjusts DDIM predicted noise at step k.

This is an approximate, full-horizon reward-guided joint diffusion sampler.
Its generated states are predictions of the learned TT, not MJX transitions.
It has no validity guarantee; physical MJX replay is required.
"""

import torch

from tt_model import condition_masks, rest_s0, unbase


def _windows(episode_states, horizon):
    if horizon < 2 or episode_states < 2:
        raise ValueError("episode_states and horizon must both be >= 2")
    starts = list(range(0, episode_states - 1, horizon - 1))
    return [(i, min(horizon, episode_states - i)) for i in starts]


def joint_diffusion_full_reward(
    net, reward, *,
    n, episode_states, horizon, state_dim, action_dim, e_dim,
    mu, sd, beta, abar, emu=0.5, esd=0.5,
    guidance_scale=1.0, clip=1.0, seed=0, no_guide_s=False,
    e_low=0.05, e_high=0.95, inactive_e=(0, 11, 12), s0_fn=None
):
    """Return (trajectory[B,episode_states,S+A], E[B,e_dim], audit).

    One E latent and one x latent per short window, all updated at every
    denoising step. Rewards depend on the *whole* predicted episode, and
    autograd sends the gradient into E's noisy token (not posthoc E search).
    DDIM noise is initialized once; gradient graphs are rebuilt per step.

    s0_fn (optional): physical E [B,e_dim] -> S0 [B,state_dim]. When given,
    S0 is NOT generated: the Mode-4 pass only denoises E, S0 = s0_fn(E) is
    recomputed at every step, and every window runs Mode 2 (E + S0 known).
    """
    if n < 1 or horizon < 2 or episode_states < 2 or guidance_scale < 0:
        raise ValueError("invalid sampler sizes or reward guidance")
    if not (0 <= e_low < e_high <= 1) or clip <= 0:
        raise ValueError("invalid output ranges")
    if len(beta) != len(abar) or not len(beta):
        raise ValueError("invalid diffusion schedule")

    dev, dtype = mu.device, mu.dtype
    xdim = state_dim + action_dim
    windows = _windows(episode_states, horizon)
    generator = torch.Generator(device=dev).manual_seed(int(seed))
    xs = [torch.randn((n, horizon, xdim), generator=generator, device=dev, dtype=dtype)
          for _ in windows]
    e = torch.randn((n, e_dim), generator=generator, device=dev, dtype=dtype)
    initial_x = rest_s0().to(device=dev, dtype=dtype)
    if len(initial_x) != state_dim:
        raise ValueError("rest_s0 dimension does not match model S_DIM")
    known_mode4 = torch.full((n,), 4, dtype=torch.long, device=dev)
    known_mode2 = torch.full((n,), 2, dtype=torch.long, device=dev)
    grad_e_norms = []
    grad_s_norms = []
    grad_a_norms = []
    guide_history = []
    inactive = list(inactive_e)

    def physical_e(en):
        physical = (en * esd + emu).clamp(e_low, e_high)
        # Keep inert morphology coordinates constant in the arm-free A1 model.
        if inactive:
            physical = physical.clone()
            physical[:, inactive] = 0.5
        return physical

    def normalized_e(en):
        return (physical_e(en) - emu) / esd

    def decode(xts, et, k):
        """One differentiable joint score pass for all coupled windows."""
        cur = torch.zeros((n, xdim), dtype=dtype, device=dev)
        cur[:, :state_dim] = initial_x
        predictions, epsilon_x, x0_parts, masks = [], [], [], []
        first_epsilon = None
        first_e0 = None
        sqrt_ab = abar[k].sqrt()
        sqrt_nm = (1 - abar[k]).sqrt()
        kk = torch.full((n,), float(k), dtype=dtype, device=dev)
        if s0_fn is not None:
            # Mode 4 pass on window 0 only supplies the E prior (epsilon).
            mask0 = (torch.arange(horizon, device=dev) < windows[0][1]).expand(n, horizon)
            kx4, ke4 = condition_masks(known_mode4, mask0)
            known4 = torch.zeros_like(xts[0])
            known4[:, 0] = (cur - mu) / sd
            _, first_epsilon = net(torch.where(kx4 > 0, known4, xts[0]), et, kx4, ke4, kk, ~mask0)
            first_e0 = normalized_e(((et - sqrt_nm * first_epsilon) / sqrt_ab).clamp(-clip, clip))
            cur[:, :state_dim] = s0_fn(physical_e(first_e0).detach())
        for i, ((t0, w), xt) in enumerate(zip(windows, xts)):
            mask = (torch.arange(horizon, device=dev) < w).expand(n, horizon)
            s = cur.clone()
            # Every window begins in its own heading-aligned frame.
            s[:, 0:2], s[:, 3], s[:, 4] = 0, 1, 0
            x_known = torch.zeros_like(xt)
            x_known[:, 0] = (s - mu) / sd
            gen_e = i == 0 and s0_fn is None
            mode = known_mode4 if gen_e else known_mode2
            kx, ke = condition_masks(mode, mask)
            input_e = et if gen_e else first_e0
            # The first E is genuinely noisy (mode 4). Later windows reuse
            # the *same denoised E* as a known condition (mode 2).
            x_in = torch.where(kx > 0, x_known, xt)
            ex, ee = net(x_in, input_e, kx, ke, kk, ~mask)
            if gen_e:
                first_epsilon = ee
                first_e0 = ((et - sqrt_nm * ee) / sqrt_ab).clamp(-clip, clip)
                first_e0 = normalized_e(first_e0)
            xhat = ((xt - sqrt_nm * ex) / sqrt_ab).clamp(-clip, clip)
            xhat = torch.where(kx > 0, x_known, xhat)
            ep = unbase(xhat * sd + mu, cur)
            # Windows overlap exactly at their boundary state. Only the
            # final window contributes its endpoint to the full trajectory.
            pred_slice = ep[:, :w] if i == len(windows) - 1 else ep[:, :w - 1]
            predictions.append(pred_slice)
            x0_parts.append((xhat, x_known, kx))
            epsilon_x.append(ex)
            masks.append(mask)
            cur = ep[:, w - 1].clone()
        trajectory = torch.cat(predictions, dim=1)
        if trajectory.shape != (n, episode_states, xdim):
            raise RuntimeError("joint diffusion window assembly inconsistent")
        return trajectory, epsilon_x, first_epsilon, x0_parts, masks

    for k in reversed(range(len(beta))):
        with torch.enable_grad() if guidance_scale > 0 else torch.no_grad():
            xts = [x.detach().requires_grad_(guidance_scale > 0) for x in xs]
            et = e.detach().requires_grad_(guidance_scale > 0)
            traj, epsx, epse, x_parts, masks = decode(xts, et, k)
            if guidance_scale > 0:
                full_mask = torch.ones(n, episode_states, dtype=torch.bool, device=dev)
                score, _ = reward(traj, full_mask)
                grads = torch.autograd.grad(score.sum(), [et] + xts, allow_unused=True)
                g_e = grads[0] if grads[0] is not None else torch.zeros_like(et)
                # The inactive morphology dimensions cannot physically matter.
                g_e = g_e.clone()
                if inactive:
                    g_e[:, inactive] = 0.0
                gx = [g if g is not None else torch.zeros_like(xt)
                      for g, xt in zip(grads[1:], xts)]
                if no_guide_s:
                    gx = [torch.cat((torch.zeros_like(g[..., :state_dim]),
                                     g[..., state_dim:]), dim=-1) for g in gx]
                # Audit each gradient component independently. E is one
                # shared morphology token; S/A span all generated windows.
                grad_e_norms.append(float(torch.linalg.vector_norm(g_e).detach().cpu()))
                grad_s_norms.append(float(torch.linalg.vector_norm(
                    torch.cat([g[..., :state_dim].reshape(n, -1) for g in gx], dim=1)
                ).detach().cpu()))
                grad_a_norms.append(float(torch.linalg.vector_norm(
                    torch.cat([g[..., state_dim:].reshape(n, -1) for g in gx], dim=1)
                ).detach().cpu()))
                guide_history.append(float(score.detach().mean().cpu()))
            else:
                g_e = torch.zeros_like(et)
                gx = [torch.zeros_like(xt) for xt in xts]

            ab = abar[k]
            prev = abar[k - 1] if k else torch.ones_like(ab)
            sqrt_nm = (1 - ab).sqrt()
            # Classifier/reward guidance directly changes E's reverse diffusion
            # epsilon, not an already-decoded E after generation.
            guided_e_eps = epse - guidance_scale * sqrt_nm * g_e
            e0 = ((et - sqrt_nm * guided_e_eps) / ab.sqrt()).clamp(-clip, clip)
            e0 = normalized_e(e0)
            e_est = (et - ab.sqrt() * e0) / sqrt_nm
            new_e = prev.sqrt() * e0 + (1 - prev).sqrt() * e_est
            new_x = []
            for xt, eps, g, (xhat, x_known, kx), mask in zip(xts, epsx, gx, x_parts, masks):
                guided_eps = eps - guidance_scale * sqrt_nm * g
                x0 = ((xt - sqrt_nm * guided_eps) / ab.sqrt()).clamp(-clip, clip)
                x0 = torch.where(kx > 0, x_known, x0)
                adjusted_eps = (xt - ab.sqrt() * x0) / sqrt_nm
                next_x = prev.sqrt() * x0 + (1 - prev).sqrt() * adjusted_eps
                new_x.append(next_x.detach())
        e = new_e.detach()
        xs = new_x

    # Final geometry-invariant boundary projection of all already-denoised
    # x tokens. This removes small stitching inconsistencies at window edges.
    with torch.no_grad():
        ef = physical_e(e)
        cur = torch.zeros((n, xdim), dtype=dtype, device=dev)
        cur[:, :state_dim] = initial_x if s0_fn is None else s0_fn(ef)
        frames = []
        for i, ((_, w), xt) in enumerate(zip(windows, xs)):
            mask = (torch.arange(horizon, device=dev) < w).expand(n, horizon)
            mode = known_mode4 if i == 0 and s0_fn is None else known_mode2
            kx, _ = condition_masks(mode, mask)
            s = cur.clone()
            s[:, 0:2], s[:, 3], s[:, 4] = 0, 1, 0
            known = torch.zeros_like(xt)
            known[:, 0] = (s - mu) / sd
            aligned = torch.where(kx > 0, known, xt)
            ep = unbase(aligned * sd + mu, cur)
            frames.append(ep[:, :w] if i == len(windows) - 1 else ep[:, :w - 1])
            cur = ep[:, w - 1].clone()
        final_traj = torch.cat(frames, dim=1)
    return final_traj, ef, {
        "algorithm": "per-step full-horizon reward-gradient joint DDIM on shared noisy E + X",
        "guidance_scale": float(guidance_scale),
        "s0_source": "generated (Mode 4)" if s0_fn is None else "s0_fn(E) rest stance (Mode 2 only)",
        "windows": len(windows),
        "diffusion_steps": len(beta),
        "e_gradient_norm_per_step": grad_e_norms,
        "s_gradient_norm_per_step": grad_s_norms,
        "a_gradient_norm_per_step": grad_a_norms,
        "predicted_reward_during_guidance": guide_history,
        "e_gradient_nonzero": bool(any(x > 1e-12 for x in grad_e_norms)),
        "s_gradient_nonzero": bool(any(x > 1e-12 for x in grad_s_norms)),
        "a_gradient_nonzero": bool(any(x > 1e-12 for x in grad_a_norms))
    }
