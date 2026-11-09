"""Run independent circle/zigzag/fast-straight TT co-design ablations.

Frozen TT inference methods:
  gradient: reward gradients through denoising (existing joint sampler)
  rl: episodic PPO residual epsilon policy, scored from MuJoCo rollouts

All evaluations produce separate E, predicted S/A, MuJoCo S/A, masks and
candidate-level physical rewards. This script DOES NOT train the TT weights.
Default --device cpu intentionally avoids a GPU already running other jobs.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from generate_task_mujoco import STEPS, settled_start, replay
from joint_diffusion_guidance import joint_diffusion_full_reward, _windows
from task_reward import TrackReward, ZigzagReward, StraightReward
from tt_model import make_net, schedule, check_rest_checkpoint
from tt_rl_residual_ppo import EpisodicResidualPPO, ppo_update


TASKS = {"circle": TrackReward, "zigzag": ZigzagReward, "straight": StraightReward}
INVALID_REWARD = -100.0


def task_class(name):
    if name not in TASKS:
        raise ValueError("Unknown task: " + name)
    return TASKS[name]()


def physical_results(task, predicted, morphology):
    """Evaluate exactly each generated E/S0/A once in real CPU MuJoCo."""
    predicted = predicted.detach().cpu().numpy()
    morphology = morphology.detach().cpu().numpy()
    mask_list, actual_list, rows, policy_rewards = [], [], [], []
    for i, (e, path) in enumerate(zip(morphology, predicted)):
        actual, mask, meta = replay(e, path)
        actual_list.append(actual)
        mask_list.append(mask)
        invalid = not bool(mask.any())
        terms = {}
        score = None
        if not invalid:
            with torch.no_grad():
                value, components = task(
                    torch.as_tensor(actual[None]), torch.as_tensor(mask[None]))
            score = float(value[0])
            terms = {k: float(v[0]) for k, v in components.items()}
            if meta.get("fell", False):
                score -= 20.0
                terms["early_termination_penalty"] = 20.0
            if not math.isfinite(score):
                invalid, score = True, None
        rl_return = INVALID_REWARD if invalid else score
        progress = (terms.get("r_progress") * task.unit_scale
                    if score is not None else None)
        n_valid = int(mask.sum())
        pos_err_cm = (float(np.linalg.norm(path[:n_valid, :2] -
                        actual[:n_valid, :2], axis=1).mean()*100)
                      if n_valid else None)
        joint_rmse_deg = (float(np.degrees(np.sqrt(np.mean(
                          (path[:n_valid, 14:26] -
                           actual[:n_valid, 14:26])**2))))
                          if n_valid else None)
        seams = [t for t in range(63, len(path), 63)]
        seam_jump_rad = (max(float(np.abs(path[t, 42:54] -
                                           path[t-1, 42:54]).max()) for t in seams)
                         if seams else None)
        rows.append({
            "pred_vs_mujoco_xy_mean_cm": pos_err_cm,
            "pred_vs_mujoco_joint_rmse_deg": joint_rmse_deg,
            "max_seam_action_jump_rad": seam_jump_rad,
            "candidate": i,
            "morphology": e.tolist(),
            "reward": score,
            "rl_return": rl_return,
            "progress": progress,
            "progress_unit": task.unit,
            "reward_terms": terms,
            "valid_seconds": float(meta.get("valid_seconds", 0.0)),
            "complete_10s": bool(meta.get("valid_seconds", 0.0) >= 10),
            **meta,
        })
        policy_rewards.append(rl_return)
    return (np.stack(actual_list), np.stack(mask_list),
            rows, torch.tensor(policy_rewards, dtype=torch.float32))


def load_model(checkpoint, device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ck["args"]
    check_rest_checkpoint(config)
    if 2 not in config["modes"] or config.get("pose", "abs") != "abs":
        raise ValueError("Needs TT with mode 2, corrected mode 4, absolute pose")
    model = make_net(config).to(device)
    weights = ck["ema_net"] if ck.get("ema_net") is not None else ck["net"]
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False)
    mu = torch.as_tensor(ck["mu"], device=device)
    sd = torch.as_tensor(ck["sd"], device=device)
    beta, abar = schedule(config["diff_steps"], device)
    if config["norm"] == "minmax":
        emu, esd = 0.5, 0.5
    else:
        emu, esd = 0.5, 0.25
    return model, dict(config=config, mu=mu, sd=sd, beta=beta, abar=abar,
                       emu=emu, esd=esd, weights="ema" if ck.get("ema_net") is not None else "raw")


def generate(model, setup, task, *, count, seed, scale=0., rl_raw=None, rl_policy=None):
    config = setup["config"]
    if rl_raw is not None and scale != 0:
        raise ValueError("Do not combine gradient and PPO corrections")
    callback = rl_policy.residual_fn(rl_raw) if rl_raw is not None else None
    def s0_fn(e):
        x = np.stack([settled_start(ee) for ee in e.detach().cpu().numpy()])
        return torch.as_tensor(x, device=setup["mu"].device, dtype=setup["mu"].dtype)
    return joint_diffusion_full_reward(
        model, task, n=count, episode_states=STEPS,
        horizon=config["horizon"], state_dim=42, action_dim=12, e_dim=13,
        mu=setup["mu"], sd=setup["sd"], beta=setup["beta"],
        abar=setup["abar"], emu=setup["emu"], esd=setup["esd"],
        clip=config["clip"], guidance_scale=scale, seed=seed,
        s0_fn=s0_fn, denoising_policy=callback)


def run_and_store(folder, name, model, setup, task, *, count, seed,
                  scale=0., rl_raw=None, rl_policy=None):
    folder.mkdir(parents=True, exist_ok=True)
    result_file = folder / f"seed_{seed}.json"
    npz_file = folder / f"seed_{seed}.npz"
    if result_file.exists() or npz_file.exists():
        raise FileExistsError(f"Existing evaluation must not be overwritten: {result_file}")
    predicted, morphology, audit = generate(
        model, setup, task, count=count, seed=seed, scale=scale,
        rl_raw=rl_raw, rl_policy=rl_policy)
    actual, valid, rows, returns = physical_results(task, predicted, morphology)
    np.savez_compressed(npz_file, pred=predicted.detach().cpu().numpy(),
                        E=morphology.detach().cpu().numpy(), mujoco=actual, valid=valid)
    result = dict(method=name, task=task.describe(), seed=seed, samples=count,
                  gradient_scale=float(scale), candidates=rows, guidance_audit=audit,
                  npz=str(npz_file))
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(name, task.describe()["task"], seed,
          [(x["reward"], x["valid_seconds"]) for x in rows], flush=True)
    return returns


def train_rl(task_name, model, setup, task, folder, *, seed, updates,
             rollouts, lr, ppo_epochs, resume):
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = folder / "policy_state.pt"
    history_path = folder / "training.jsonl"
    num_windows = len(_windows(STEPS, setup["config"]["horizon"]))
    policy = EpisodicResidualPPO(num_windows).to(setup["mu"].device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    start = 0
    if resume and checkpoint_path.exists():
        snapshot = torch.load(checkpoint_path, map_location=setup["mu"].device, weights_only=False)
        if snapshot["num_windows"] != num_windows or snapshot["task"] != task_name:
            raise ValueError("RL checkpoint task/window count mismatch")
        policy.load_state_dict(snapshot["policy"])
        optimizer.load_state_dict(snapshot["optimizer"])
        start = int(snapshot["iteration"])
    elif checkpoint_path.exists() or history_path.exists():
        raise FileExistsError(f"Existing RL training in {folder}. Use --resume.")
    for iteration in range(start, updates):
        # Different random TT noise each iteration, but fixed reproducible seeds.
        torch.manual_seed(seed + 23 * iteration)
        raw, old_lp = policy.draw(rollouts)
        predicted, morph, audit = generate(
            model, setup, task, count=rollouts,
            seed=seed + 1000 + iteration, rl_raw=raw,
            rl_policy=policy)
        _, _, rows, rewards = physical_results(task, predicted, morph)
        metrics = ppo_update(policy, optimizer, raw, old_lp, rewards,
                             epochs=ppo_epochs)
        record = dict(task=task_name, iteration=iteration + 1, rollouts=rollouts,
                      initial_valid=sum(bool(x.get("initial_valid")) for x in rows),
                      completed=sum(x["complete_10s"] for x in rows),
                      **metrics)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp = checkpoint_path.with_suffix(".tmp")
        torch.save(dict(task=task_name, num_windows=num_windows,
                        iteration=iteration + 1, policy=policy.state_dict(),
                        optimizer=optimizer.state_dict()), tmp)
        tmp.replace(checkpoint_path)
        print("PPO", record, flush=True)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--methods", nargs="+", choices=("gradient", "rl"),
                        default=["gradient", "rl"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--rl-updates", type=int, default=20)
    parser.add_argument("--rl-rollouts", type=int, default=4)
    parser.add_argument("--rl-lr", type=float, default=0.001)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="resume interrupted RL policy and skip completed evaluations")
    args = parser.parse_args()
    if args.candidates < 1 or args.eval_seeds < 1 or args.rl_updates < 1 or args.rl_rollouts < 2:
        parser.error("Invalid episode count")
    if args.cpu_threads < 1 or any(not math.isfinite(s) or s < 0 for s in args.scales):
        parser.error("Invalid CPU threads or guidance scale")
    ckpt = args.ckpt / "checkpoint.pt" if args.ckpt.is_dir() else args.ckpt
    if not ckpt.is_file():
        parser.error(f"Missing checkpoint: {ckpt}")
    if args.out.exists() and not args.resume:
        parser.error("Output directory exists. Use a fresh --out or --resume")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    model, setup = load_model(ckpt, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config_file = args.out / "experiment_config.json"
    if config_file.exists():
        old = json.loads(config_file.read_text(encoding="utf-8"))
        # Only the resume flag is permitted to change.
        old["resume"], cfg["resume"] = False, False
        if old != cfg:
            raise ValueError("Existing experiment settings differ from current arguments")
    else:
        config_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Frozen TT:", ckpt, "weights:", setup["weights"],
          "device:", args.device, flush=True)
    for name in args.tasks:
        task = task_class(name)
        td = args.out / name
        td.mkdir(parents=True, exist_ok=True)
        policy = None
        if "rl" in args.methods:
            policy = train_rl(name, model, setup, task, td / "rl_training",
                              seed=args.seed, updates=args.rl_updates,
                              rollouts=args.rl_rollouts, lr=args.rl_lr,
                              ppo_epochs=args.ppo_epochs, resume=args.resume)
        for evaluation in range(args.eval_seeds):
            eval_seed = args.seed + 20000 + evaluation
            if "gradient" in args.methods:
                for scale in args.scales:
                    folder = td / ("gradient_scale_%g" % scale)
                    if args.resume and (folder / f"seed_{eval_seed}.json").exists():
                        continue
                    run_and_store(folder, "gradient", model, setup, task,
                                  count=args.candidates, seed=eval_seed, scale=scale)
            if "rl" in args.methods:
                folder = td / "rl_ppo"
                if args.resume and (folder / f"seed_{eval_seed}.json").exists():
                    continue
                # Deterministic policy mean for evaluation; stochastic TT
                # initial noise varies with seed and candidate index.
                raw = policy.mean.detach()[None, :].expand(args.candidates, -1)
                run_and_store(folder, "rl", model, setup, task,
                              count=args.candidates, seed=eval_seed,
                              rl_raw=raw, rl_policy=policy)
    from tt_task_report import build_report
    build_report(args.out)
    print("Saved result report:", args.out / "report.html", flush=True)


if __name__ == "__main__":
    main()
