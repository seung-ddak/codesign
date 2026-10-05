"""L0~L3 design search on the stair task (remind.md §4.3).

  L0  prior 무작위 샘플링        -- task 난이도 확인
  L1  CMA-ES                     -- black-box 최적화 기준선
  L2  결과 기반 prior 재학습     -- DiffuseBot 식 상위 k% fine-tune
  L3  residual guidance          -- 우리 방법 (D11/D12)

All rungs spend the same number of design evaluations (`--budget`) and share the
frozen pi_theta and the binary success criterion, so only the proposal mechanism
differs. Results are appended to `--out` as json.
"""

import argparse
import json
import os
import time

import numpy as np

from t2.eval.stair_design import (
    build_evaluator,
    latest_checkpoint,
    uniforms_to_attrs,
)


def _summarize(stage, success, fell, max_x, evals, elapsed, extra=None):
    out = {
        "stage": stage,
        "evaluations": int(evals),
        "elapsed_s": round(elapsed, 1),
        "success_mean": float(np.mean(success)),
        "success_max": float(np.max(success)),
        "success_gt0_frac": float(np.mean(np.asarray(success) > 0)),
        "success_p90": float(np.quantile(success, 0.9)),
        "fell_mean": float(np.mean(fell)),
        "max_x_mean": float(np.mean(max_x)),
        "max_x_p90": float(np.quantile(max_x, 0.9)),
    }
    if extra:
        out.update(extra)
    return out


class Objective:
    """designs (B, D) in [0,1] -> per-design success rate, with a call counter."""

    def __init__(self, evaluator, space, num_processes, seed, rebuild_fn=None):
        self.evaluator = evaluator
        self.rebuild_fn = rebuild_fn
        self._goal = None
        self.space = space
        self.num_processes = num_processes
        self.seed = seed
        self.rise = None
        self.n_evals = 0
        self.history = []

    def set_goal(self, goal_x):
        """목표 지점을 바꾸려면 env 를 다시 만들어야 한다 (goal_x 는 env 상수)."""
        if goal_x == self._goal:
            return
        self._goal = goal_x
        if self.rebuild_fn is not None:
            self.evaluator, self.space = self.rebuild_fn(goal_x)

    def __call__(self, designs: np.ndarray, rise=None):
        designs = np.clip(np.atleast_2d(designs), 0.0, 1.0)
        attrs = uniforms_to_attrs(
            designs,
            self.space["choices"],
            self.space["robogen_fn"],
            self.space["data_attrs"],
            num_processes=self.num_processes,
            rise=rise if rise is not None else self.rise,
        )
        success, fell, max_x = self.evaluator(attrs, seed=self.seed + self.n_evals)
        self.n_evals += len(designs)
        self.history.append(
            dict(
                n_evals=self.n_evals,
                best=float(np.max(success)),
                mean=float(np.mean(success)),
            )
        )
        return success, fell, max_x


def run_l0(obj, args, rng):
    """같은 설계 집합을 여러 (단높이, 목표거리) 조합에서 평가한다.

    §4.3 L0 의 목적은 "설계가 변별되는 난이도 찾기"다. 목표 거리가 정책 능력보다
    훨씬 멀면 전부 0, 너무 가까우면 전부 1 이라 어느 쪽도 변별이 안 된다.
    """
    designs = rng.uniform(0.0, 1.0, size=(args.budget, obj.space["num_uniforms"]))
    rises = [float(r) for r in str(args.rises).split()]
    goals = [None] if args.goals is None else [
        float(g) for g in str(args.goals).split()
    ]
    rows = []
    for rise in rises:
        for goal in goals:
            obj.set_goal(goal)
            t0 = time.time()
            success, fell, max_x = obj(designs, rise=rise)
            best = designs[int(np.argmax(success))]
            row = _summarize(
                "L0", success, fell, max_x, len(designs), time.time() - t0,
                extra={"stair_rise": rise, "goal_x": goal,
                       "best_design": best.tolist()},
            )
            print(f"[L0] rise={rise:.2f} goal_x={goal} "
                  f"mean={row['success_mean']:.3f} "
                  f"gt0={row['success_gt0_frac']:.3f} fell={row['fell_mean']:.3f} "
                  f"max_x={row['max_x_mean']:.3f}", flush=True)
            rows.append(row)
    return rows


def run_l1(obj, args, rng):
    import cma

    t0 = time.time()
    dim = obj.space["num_uniforms"]
    es = cma.CMAEvolutionStrategy(
        [0.5] * dim,
        0.25,
        {
            "bounds": [0.0, 1.0],
            "popsize": args.popsize,
            "seed": args.seed + 1,
            "verbose": -9,
        },
    )
    best_success, best_design = -1.0, None
    all_success, all_fell, all_max_x = [], [], []
    while obj.n_evals < args.budget:
        designs = np.array(es.ask())
        success, fell, max_x = obj(designs)
        es.tell(designs.tolist(), (-success).tolist())
        all_success.append(success)
        all_fell.append(fell)
        all_max_x.append(max_x)
        if success.max() > best_success:
            best_success = float(success.max())
            best_design = designs[int(np.argmax(success))]
        print(
            f"[L1] evals={obj.n_evals} gen_best={success.max():.3f} "
            f"best={best_success:.3f}",
            flush=True,
        )
    return _summarize(
        "L1",
        np.concatenate(all_success),
        np.concatenate(all_fell),
        np.concatenate(all_max_x),
        obj.n_evals,
        time.time() - t0,
        extra={
            "best_success": best_success,
            "best_design": None if best_design is None else best_design.tolist(),
            "history": obj.history,
        },
    )


def _make_diffusion(dim, steps, device):
    """Tiny DDPM over the design vector -- the stand-in prior for L2/L3.

    TT's pretrained denoiser generates RoboTokens across all 138 discrete
    choices; the frozen pi_theta only controls one of them, so for this first
    experiment the prior lives in the same 13-dim continuous slice everything
    else uses (remind.md: deviation from D8, recorded in §4.4).
    """
    import torch
    import torch.nn as nn

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, dim),
            )

        def forward(self, x, t):
            return self.net(torch.cat([x, t[:, None].float() / steps], dim=-1))

    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    alphas = torch.cumprod(1.0 - betas, dim=0)
    return Denoiser().to(device), betas, alphas


def _ddpm_fit(model, betas, alphas, data, steps, epochs, lr, device):
    import torch

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x0 = torch.as_tensor(data, dtype=torch.float32, device=device) * 2 - 1
    for _ in range(epochs):
        t = torch.randint(0, steps, (len(x0),), device=device)
        noise = torch.randn_like(x0)
        a = alphas[t][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        loss = ((model(xt, t) - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item())


def _ddpm_sample(model, betas, alphas, n, dim, steps, device, guidance=None):
    """Ancestral sampling. `guidance(x, t)` adds a residual at each step (L3)."""
    import torch

    x = torch.randn(n, dim, device=device)
    traj = []
    with torch.no_grad():
        for t in reversed(range(steps)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps = model(x, tt)
            a, b = alphas[t], betas[t]
            mean = (x - b / (1 - a).sqrt() * eps) / (1 - b).sqrt()
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x_next = mean + b.sqrt() * noise
            if guidance is not None:
                residual = guidance(x, tt)
                traj.append((x.clone(), tt.clone(), residual.clone()))
                x_next = x_next + residual
            x = x_next
    designs = ((x + 1) / 2).clamp(0, 1).cpu().numpy()
    return designs, traj


def run_l2(obj, args, rng):
    """DiffuseBot-style: sample -> evaluate -> refit the prior on the top k%."""
    import torch

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = obj.space["num_uniforms"]
    model, betas, alphas = _make_diffusion(dim, args.diffusion_steps, device)

    pool = rng.uniform(0.0, 1.0, size=(args.l2_pool, dim))
    _ddpm_fit(model, betas, alphas, pool, args.diffusion_steps,
              args.diffusion_epochs, args.diffusion_lr, device)

    rounds, per_round = [], max(1, args.budget // args.l2_rounds)
    best_success, best_design = -1.0, None
    all_success, all_fell, all_max_x = [], [], []
    while obj.n_evals < args.budget:
        n = min(per_round, args.budget - obj.n_evals)
        designs, _ = _ddpm_sample(
            model, betas, alphas, n, dim, args.diffusion_steps, device
        )
        success, fell, max_x = obj(designs)
        all_success.append(success)
        all_fell.append(fell)
        all_max_x.append(max_x)
        if success.max() > best_success:
            best_success = float(success.max())
            best_design = designs[int(np.argmax(success))]
        k = max(4, int(len(designs) * args.l2_top_frac))
        elite = designs[np.argsort(-success)[:k]]
        loss = _ddpm_fit(model, betas, alphas, elite, args.diffusion_steps,
                         args.diffusion_epochs, args.diffusion_lr, device)
        rounds.append(
            dict(n_evals=obj.n_evals, mean=float(success.mean()),
                 best=float(success.max()), refit_loss=loss)
        )
        print(f"[L2] evals={obj.n_evals} mean={success.mean():.3f} "
              f"best={best_success:.3f}", flush=True)
    return _summarize(
        "L2", np.concatenate(all_success), np.concatenate(all_fell),
        np.concatenate(all_max_x), obj.n_evals, time.time() - t0,
        extra={"best_success": best_success,
               "best_design": None if best_design is None else best_design.tolist(),
               "rounds": rounds},
    )


def run_l3(obj, args, rng):
    """Residual guidance (D11): a policy adds a residual to the last denoising
    steps, trained by REINFORCE on the terminal success reward (D9/Transform2Act:
    zero reward during the design stage, task reward at the end)."""
    import torch
    import torch.nn as nn

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = obj.space["num_uniforms"]
    model, betas, alphas = _make_diffusion(dim, args.diffusion_steps, device)
    pool = rng.uniform(0.0, 1.0, size=(args.l2_pool, dim))
    _ddpm_fit(model, betas, alphas, pool, args.diffusion_steps,
              args.diffusion_epochs, args.diffusion_lr, device)
    for p in model.parameters():  # the prior stays frozen (D11)
        p.requires_grad_(False)

    class Guidance(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, dim),
            )
            self.log_std = nn.Parameter(torch.full((dim,), np.log(args.l3_std)))

        def forward(self, x, t):
            return self.net(torch.cat([x, t[:, None].float()
                                       / args.diffusion_steps], dim=-1))

    pi = Guidance().to(device)
    opt = torch.optim.Adam(pi.parameters(), lr=args.l3_lr)
    baseline, rounds = None, []
    best_success, best_design = -1.0, None
    all_success, all_fell, all_max_x = [], [], []
    per_round = max(1, args.budget // args.l3_rounds)

    while obj.n_evals < args.budget:
        n = min(per_round, args.budget - obj.n_evals)
        saved = []

        def guidance(x, tt):
            # Sampling runs t = T-1 -> 0, so the *last* denoising steps are the
            # small t (remind.md §3.2: short credit path, stays on the prior).
            if int(tt[0].item()) >= args.l3_guided_steps:
                return torch.zeros_like(x)
            with torch.no_grad():
                mean = pi(x, tt) * args.l3_scale
                std = pi.log_std.exp()
                action = mean + torch.randn_like(mean) * std
            saved.append((x.detach(), tt.detach(), action.detach()))
            return action

        designs, _ = _ddpm_sample(model, betas, alphas, n, dim,
                                  args.diffusion_steps, device, guidance=guidance)
        success, fell, max_x = obj(designs)
        all_success.append(success)
        all_fell.append(fell)
        all_max_x.append(max_x)
        if success.max() > best_success:
            best_success = float(success.max())
            best_design = designs[int(np.argmax(success))]

        reward = torch.as_tensor(success, dtype=torch.float32, device=device)
        baseline = (reward.mean() if baseline is None
                    else 0.9 * baseline + 0.1 * reward.mean())
        advantage = reward - baseline
        if advantage.abs().max() > 0:
            advantage = advantage / (advantage.std() + 1e-6)
        loss = 0.0
        for x, tt, action in saved:  # REINFORCE over the guided denoising steps
            mean = pi(x, tt) * args.l3_scale
            std = pi.log_std.exp()
            logp = (-0.5 * ((action - mean) / std) ** 2 - pi.log_std).sum(-1)
            loss = loss - (logp * advantage).mean()
        if saved:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
            opt.step()
        rounds.append(dict(n_evals=obj.n_evals, mean=float(success.mean()),
                           best=float(success.max()),
                           loss=float(loss) if saved else 0.0))
        print(f"[L3] evals={obj.n_evals} mean={success.mean():.3f} "
              f"best={best_success:.3f}", flush=True)

    return _summarize(
        "L3", np.concatenate(all_success), np.concatenate(all_fell),
        np.concatenate(all_max_x), obj.n_evals, time.time() - t0,
        extra={"best_success": best_success,
               "best_design": None if best_design is None else best_design.tolist(),
               "rounds": rounds},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["l0", "l1", "l2", "l3"])
    p.add_argument("--run-dir", required=True, help="wandb run dir of the pi_theta run")
    p.add_argument("--ckpt", default=None, help="checkpoint dir (default: latest)")
    p.add_argument("--budget", type=int, default=4096, help="design evaluations")
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--popsize", type=int, default=64, help="L1 CMA-ES population")
    p.add_argument("--num-processes", type=int, default=48)
    p.add_argument("--max-batch", type=int, default=3072)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="design_search_results.jsonl")
    p.add_argument("--rises", default="0.03 0.05 0.07 0.10",
                   help="L0: 평가할 단높이 목록")
    p.add_argument("--goals", default=None,
                   help="L0: 평가할 목표 x 좌표 목록 (출발 x = -0.5)")
    p.add_argument("--rise", type=float, default=None,
                   help="L1~L3: 고정 단높이 (L0 에서 고른 값)")
    p.add_argument("--goal-x", type=float, default=None,
                   help="L1~L3: 고정 목표 x (L0 에서 고른 값)")
    # L2/L3 diffusion prior
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--diffusion-epochs", type=int, default=2000)
    p.add_argument("--diffusion-lr", type=float, default=1e-3)
    p.add_argument("--l2-pool", type=int, default=20000)
    p.add_argument("--l2-rounds", type=int, default=8)
    p.add_argument("--l2-top-frac", type=float, default=0.1)
    # L3 residual guidance
    p.add_argument("--l3-rounds", type=int, default=16)
    p.add_argument("--l3-lr", type=float, default=3e-4)
    p.add_argument("--l3-std", type=float, default=0.05)
    p.add_argument("--l3-scale", type=float, default=0.1)
    p.add_argument("--l3-guided-steps", type=int, default=10)
    args = p.parse_args()

    ckpt = args.ckpt or latest_checkpoint(args.run_dir)
    print(f"checkpoint: {ckpt}", flush=True)
    def rebuild(goal_x):
        return build_evaluator(
            ckpt, num_seeds=args.num_seeds, max_batch=args.max_batch, goal_x=goal_x
        )

    evaluator, space = rebuild(args.goal_x)
    obj = Objective(evaluator, space, args.num_processes, args.seed, rebuild_fn=rebuild)
    rng = np.random.default_rng(args.seed)

    obj.rise = args.rise
    results = {"l0": run_l0, "l1": run_l1, "l2": run_l2, "l3": run_l3}[args.stage](
        obj, args, rng
    )
    if isinstance(results, dict):
        results = [results]
    with open(args.out, "a") as f:
        for result in results:
            result["checkpoint"] = ckpt
            result["num_seeds"] = args.num_seeds
            result["seed"] = args.seed
            f.write(json.dumps(result) + "\n")
    print(json.dumps(
        [{k: v for k, v in r.items() if k not in ("history", "rounds", "best_design")}
         for r in results], indent=2))


if __name__ == "__main__":
    main()
