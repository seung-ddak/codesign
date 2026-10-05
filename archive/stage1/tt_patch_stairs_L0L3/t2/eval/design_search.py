"""L0~L3 설계 탐색 단 (remind.md §4.3).

평가기(objective)와 무관한 탐색 로직만 담는다. objective 는

    obj(designs: np.ndarray (B, D)) -> (success, fell, max_x)   각각 (B,)

를 만족하면 무엇이든 된다 — MJX 평가기든 TT ctrl 정책 배치 rollout 이든.
"""

import time

import numpy as np


def summarize(stage, success, fell, max_x, evals, elapsed, extra=None):
    out = {
        "stage": stage,
        "evaluations": int(evals),
        "elapsed_s": round(elapsed, 1),
        "success_mean": float(np.mean(success)),
        "success_max": float(np.max(success)),
        "success_gt0_frac": float(np.mean(np.asarray(success) > 0)),
        "fell_mean": float(np.mean(fell)),
        "max_x_mean": float(np.mean(max_x)),
        "max_x_p90": float(np.quantile(max_x, 0.9)),
    }
    if extra:
        out.update(extra)
    return out


# -- L0 : prior 무작위 샘플링 -------------------------------------------------


def run_l0(obj, args, rng):
    """prior 에서 그냥 뽑은 설계의 성공률 분포 = task 난이도 확인."""
    designs = rng.uniform(0.0, 1.0, size=(args.budget, args.dim))
    t0 = time.time()
    success, fell, max_x = obj(designs)
    best = designs[int(np.argmax(max_x))]
    row = summarize(
        "L0", success, fell, max_x, len(designs), time.time() - t0,
        extra={"best_design": best.tolist(),
               "distance_mean": float(np.mean(max_x)),
               "distance_std": float(np.std(max_x))},
    )
    # 목표 거리를 바꿔가며 변별력을 본다 (4p(1-p) 가 최대인 지점)
    row["goal_sweep"] = {
        f"{g:.1f}": round(float(np.mean(np.asarray(max_x) >= g)), 3)
        for g in (0.3, 0.5, 0.8, 1.0, 1.2)
    }
    print(f"[L0] 성공률 {row['success_mean']:.3f} | 이동 평균 "
          f"{row['distance_mean']:.3f} ± {row['distance_std']:.3f} | "
          f"목표별 성공률 {row['goal_sweep']}", flush=True)
    return row


# -- L1 : CMA-ES --------------------------------------------------------------


def run_l1(obj, args, rng):
    import cma

    t0 = time.time()
    es = cma.CMAEvolutionStrategy(
        [0.5] * args.dim, 0.25,
        {"bounds": [0.0, 1.0], "popsize": args.popsize,
         "seed": args.seed + 1, "verbose": -9},
    )
    best_success, best_design = -1.0, None
    all_s, all_f, all_x, history = [], [], [], []
    n_evals = 0
    while n_evals < args.budget:
        designs = np.array(es.ask())
        success, fell, max_x = obj(designs)
        es.tell(designs.tolist(), (-np.asarray(success)).tolist())
        n_evals += len(designs)
        all_s.append(success); all_f.append(fell); all_x.append(max_x)
        if success.max() > best_success:
            best_success = float(success.max())
            best_design = designs[int(np.argmax(success))]
        history.append(dict(n_evals=n_evals, gen_mean=float(success.mean()),
                            best=best_success))
        print(f"[L1] evals={n_evals} gen_mean={success.mean():.3f} "
              f"best={best_success:.3f}", flush=True)
    return summarize(
        "L1", np.concatenate(all_s), np.concatenate(all_f), np.concatenate(all_x),
        n_evals, time.time() - t0,
        extra={"best_success": best_success,
               "best_design": None if best_design is None else best_design.tolist(),
               "history": history},
    )


# -- diffusion prior (L2/L3 공용) ---------------------------------------------


def make_diffusion(dim, steps, device):
    import torch
    import torch.nn as nn

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 256), nn.SiLU(),
                nn.Linear(256, 256), nn.SiLU(),
                nn.Linear(256, dim),
            )

        def forward(self, x, t):
            return self.net(torch.cat([x, t[:, None].float() / steps], dim=-1))

    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    alphas = torch.cumprod(1.0 - betas, dim=0)
    return Denoiser().to(device), betas, alphas


def ddpm_fit(model, betas, alphas, data, steps, epochs, lr, device):
    import torch

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x0 = torch.as_tensor(np.asarray(data), dtype=torch.float32, device=device) * 2 - 1
    loss = torch.zeros(())
    for _ in range(epochs):
        t = torch.randint(0, steps, (len(x0),), device=device)
        noise = torch.randn_like(x0)
        a = alphas[t][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        loss = ((model(xt, t) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


def ddpm_sample(model, betas, alphas, n, dim, steps, device, guidance=None):
    import torch

    x = torch.randn(n, dim, device=device)
    with torch.no_grad():
        for t in reversed(range(steps)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps = model(x, tt)
            a, b = alphas[t], betas[t]
            mean = (x - b / (1 - a).sqrt() * eps) / (1 - b).sqrt()
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x_next = mean + b.sqrt() * noise
            if guidance is not None:
                x_next = x_next + guidance(x, tt)
            x = x_next
    return ((x + 1) / 2).clamp(0, 1).cpu().numpy()


# -- L2 : DiffuseBot 식 결과 기반 prior 재학습 --------------------------------


def run_l2(obj, args, rng):
    import torch

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, betas, alphas = make_diffusion(args.dim, args.diffusion_steps, device)
    pool = rng.uniform(0.0, 1.0, size=(args.l2_pool, args.dim))
    ddpm_fit(model, betas, alphas, pool, args.diffusion_steps,
             args.diffusion_epochs, args.diffusion_lr, device)

    per_round = max(1, args.budget // args.l2_rounds)
    best_success, best_design, n_evals = -1.0, None, 0
    all_s, all_f, all_x, rounds = [], [], [], []
    while n_evals < args.budget:
        n = min(per_round, args.budget - n_evals)
        designs = ddpm_sample(model, betas, alphas, n, args.dim,
                              args.diffusion_steps, device)
        success, fell, max_x = obj(designs)
        n_evals += n
        all_s.append(success); all_f.append(fell); all_x.append(max_x)
        if success.max() > best_success:
            best_success = float(success.max())
            best_design = designs[int(np.argmax(success))]
        k = max(4, int(n * args.l2_top_frac))
        elite = designs[np.argsort(-np.asarray(success))[:k]]
        loss = ddpm_fit(model, betas, alphas, elite, args.diffusion_steps,
                        args.diffusion_epochs, args.diffusion_lr, device)
        rounds.append(dict(n_evals=n_evals, mean=float(success.mean()),
                           best=best_success, refit_loss=loss))
        print(f"[L2] evals={n_evals} mean={success.mean():.3f} "
              f"best={best_success:.3f}", flush=True)
    return summarize(
        "L2", np.concatenate(all_s), np.concatenate(all_f), np.concatenate(all_x),
        n_evals, time.time() - t0,
        extra={"best_success": best_success,
               "best_design": None if best_design is None else best_design.tolist(),
               "rounds": rounds},
    )


# -- L3 : residual guidance (우리 방법, D11) ----------------------------------


def run_l3(obj, args, rng):
    """사전학습 denoiser 는 고정하고, 정책이 **마지막 몇 denoising step 에만**
    residual 을 더한다. 학습은 종단 보상 하나로 REINFORCE
    (Transform2Act 의 credit assignment: 설계 단계 보상 0, 끝에서 한 번 채점)."""
    import torch
    import torch.nn as nn

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, betas, alphas = make_diffusion(args.dim, args.diffusion_steps, device)
    pool = rng.uniform(0.0, 1.0, size=(args.l2_pool, args.dim))
    ddpm_fit(model, betas, alphas, pool, args.diffusion_steps,
             args.diffusion_epochs, args.diffusion_lr, device)
    for p in model.parameters():  # prior 동결 (D11)
        p.requires_grad_(False)

    class Guidance(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(args.dim + 1, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, args.dim),
            )
            self.log_std = nn.Parameter(
                torch.full((args.dim,), float(np.log(args.l3_std)))
            )

        def forward(self, x, t):
            return self.net(
                torch.cat([x, t[:, None].float() / args.diffusion_steps], dim=-1)
            )

    pi = Guidance().to(device)
    opt = torch.optim.Adam(pi.parameters(), lr=args.l3_lr)
    per_round = max(1, args.budget // args.l3_rounds)
    baseline, n_evals = None, 0
    best_success, best_design = -1.0, None
    all_s, all_f, all_x, rounds = [], [], [], []

    while n_evals < args.budget:
        n = min(per_round, args.budget - n_evals)
        saved = []

        def guidance(x, tt):
            # 샘플링은 t = T-1 -> 0 으로 진행하므로 작은 t 가 마지막 step 이다.
            if int(tt[0].item()) >= args.l3_guided_steps:
                return torch.zeros_like(x)
            with torch.no_grad():
                mean = pi(x, tt) * args.l3_scale
                action = mean + torch.randn_like(mean) * pi.log_std.exp()
            saved.append((x.detach(), tt.detach(), action.detach()))
            return action

        designs = ddpm_sample(model, betas, alphas, n, args.dim,
                              args.diffusion_steps, device, guidance=guidance)
        success, fell, max_x = obj(designs)
        n_evals += n
        all_s.append(success); all_f.append(fell); all_x.append(max_x)
        if success.max() > best_success:
            best_success = float(success.max())
            best_design = designs[int(np.argmax(success))]

        reward = torch.as_tensor(np.asarray(success), dtype=torch.float32,
                                 device=device)
        baseline = (reward.mean() if baseline is None
                    else 0.9 * baseline + 0.1 * reward.mean())
        adv = reward - baseline
        if float(adv.abs().max()) > 0:
            adv = adv / (adv.std() + 1e-6)
        loss = torch.zeros((), device=device)
        for x, tt, action in saved:
            mean = pi(x, tt) * args.l3_scale
            logp = (
                -0.5 * ((action - mean) / pi.log_std.exp()) ** 2 - pi.log_std
            ).sum(-1)
            loss = loss - (logp * adv).mean()
        if saved:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
            opt.step()
        rounds.append(dict(n_evals=n_evals, mean=float(success.mean()),
                           best=best_success, loss=float(loss)))
        print(f"[L3] evals={n_evals} mean={success.mean():.3f} "
              f"best={best_success:.3f}", flush=True)

    return summarize(
        "L3", np.concatenate(all_s), np.concatenate(all_f), np.concatenate(all_x),
        n_evals, time.time() - t0,
        extra={"best_success": best_success,
               "best_design": None if best_design is None else best_design.tolist(),
               "rounds": rounds},
    )


STAGES = {"l0": run_l0, "l1": run_l1, "l2": run_l2, "l3": run_l3}
