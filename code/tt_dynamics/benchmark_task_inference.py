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
import subprocess
from pathlib import Path

import numpy as np
import torch

from generate_task_mujoco import EVALUATION_VERSION, STEPS, TASKS, replay, score_replay, settled_start
from joint_diffusion_guidance import joint_diffusion_full_reward, _windows
from tt_model import make_net, schedule, check_rest_checkpoint
from tt_rl_residual_ppo import EpisodicResidualPPO, ppo_update


INVALID_REWARD = -100.0


def task_class(name):
    if name not in TASKS:
        raise ValueError("Unknown task: " + name)
    return TASKS[name]()


def physical_results(task, predicted, morphology, *, horizon=64):
    """Evaluate exactly each generated E/S0/A once in real CPU MuJoCo."""
    predicted = np.asarray(predicted)
    morphology = np.asarray(morphology)
    if (predicted.ndim != 3 or predicted.shape[1:] != (STEPS, 54) or
            not len(predicted) or morphology.shape != (len(predicted), 13)):
        raise ValueError("Expected aligned nonempty candidate arrays pred[N,500,54] and E[N,13]")
    mask_list, actual_list, rows, policy_rewards = [], [], [], []
    for i, (e, path) in enumerate(zip(morphology, predicted)):
        actual, mask, meta = replay(e, path, task=task)
        actual_list.append(actual)
        mask_list.append(mask)
        result = score_replay(task, actual, mask, meta)
        n_compare = min(int(mask.sum()), len(path))
        pos_err_cm = (float(np.linalg.norm(path[:n_compare, :2] -
                        actual[:n_compare, :2], axis=1).mean()*100)
                      if meta["control_steps"] else None)
        joint_rmse_deg = (float(np.degrees(np.sqrt(np.mean(
                          (path[:n_compare, 14:26] -
                           actual[:n_compare, 14:26])**2))))
                          if meta["control_steps"] else None)
        seams = range(horizon - 1, meta["control_steps"], horizon - 1)
        seam_jump_rad = max((float(np.abs(actual[t, 42:54] -
                                         actual[t-1, 42:54]).max()) for t in seams), default=None)
        rl_return = INVALID_REWARD if result["reward"] is None else result["reward"]
        rows.append(dict(candidate=i, morphology=e.tolist(),
                         pred_vs_mujoco_xy_mean_cm=pos_err_cm,
                         pred_vs_mujoco_joint_rmse_deg=joint_rmse_deg,
                         max_seam_action_jump_rad=seam_jump_rad, rl_return=rl_return,
                         **result, **meta))
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
    if result_file.exists():
        raise FileExistsError(f"Existing evaluation must not be overwritten: {result_file}")
    predicted, morphology, audit = generate(
        model, setup, task, count=count, seed=seed, scale=scale,
        rl_raw=rl_raw, rl_policy=rl_policy)
    result = dict(method=name, seed=seed, samples=count, gradient_scale=float(scale),
                  guidance_audit=audit, horizon=setup["config"]["horizon"])
    return store_evaluation(result_file, task, predicted.detach().cpu().numpy(),
                            morphology.detach().cpu().numpy(), result)


def store_evaluation(result_file, task, predicted, morphology, provenance):
    actual, valid, rows, returns = physical_results(task, predicted, morphology,
                                                     horizon=provenance["horizon"])
    npz_file = result_file.with_suffix(".npz")
    temporary_npz = npz_file.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as f:
        np.savez_compressed(f, pred=predicted, E=morphology, mujoco=actual, valid=valid,
                            action_valid=valid[:, 1:])
    temporary_npz.replace(npz_file)
    result = dict(**provenance, evaluation_version=EVALUATION_VERSION,
                  task=task.describe(), candidates=rows, npz=str(npz_file))
    temporary_json = result_file.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_json.replace(result_file)
    print(result["method"], result["task"]["task"], result["seed"],
          [(x["reward"], x["valid_seconds"], x["termination_reason"]) for x in rows], flush=True)
    return returns


def replay_archive(source, out):
    """Re-score previously generated candidates, with no checkpoint or training."""
    from tt_task_report import collect

    groups = collect(source)
    jobs = [(name, entry) for name, methods in groups.items()
            for runs in methods.values() for entry in runs]
    if not jobs:
        raise ValueError(f"No archived candidate pairs under {source}")
    out.mkdir(parents=True, exist_ok=False)
    for name, (result_file, previous, archive) in jobs:
        target = out / result_file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        provenance = {key: previous[key] for key in
                      ("method", "seed", "samples", "gradient_scale", "guidance_audit")
                      if key in previous}
        provenance.update(horizon=previous.get("horizon", 64),
                          source_candidate_file=str(result_file),
                          source_evaluation_version=previous.get("evaluation_version", 1))
        with np.load(archive) as z:
            store_evaluation(target, task_class(name), z["pred"], z["E"], provenance)


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
        _, _, rows, rewards = physical_results(task, predicted.detach().cpu().numpy(),
                                              morph.detach().cpu().numpy(),
                                              horizon=setup["config"]["horizon"])
        metrics = ppo_update(policy, optimizer, raw, old_lp, rewards,
                             epochs=ppo_epochs)
        record = dict(task=task_name, iteration=iteration + 1, rollouts=rollouts,
                      initial_valid=sum(bool(x.get("initial_valid")) for x in rows),
                      completed=sum(x["complete_10s"] for x in rows),
                      task_successes=sum(x["task_success"] for x in rows),
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", type=Path)
    source.add_argument("--replay-from", type=Path,
                        help="re-evaluate archived E/S0/A into a new output directory")
    parser.add_argument("--expected-diffusion-steps", type=int,
                        help="fail if the checkpoint schedule has a different number of steps")
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
    parser.add_argument("--record-video", action="store_true",
                        help="record MuJoCo 3D MP4 after evaluations in a separate offscreen process")
    parser.add_argument("--video-selection", choices=("best", "all"), default="best",
                        help="record best physical reward candidate per method/seed or all candidates")
    parser.add_argument("--video-backend", choices=("auto", "osmesa", "egl", "wireframe"),
                        default="auto",
                        help="auto: CPU OSMesa mesh if installed, otherwise GL-free MuJoCo 3D FK wireframe")
    parser.add_argument("--video-fps", type=int, choices=(10, 25, 50), default=25)
    args = parser.parse_args()
    if args.candidates < 1 or args.eval_seeds < 1 or args.rl_updates < 1 or args.rl_rollouts < 2:
        parser.error("Invalid episode count")
    if args.cpu_threads < 1 or any(not math.isfinite(s) or s < 0 for s in args.scales):
        parser.error("Invalid CPU threads or guidance scale")
    torch.set_num_threads(args.cpu_threads)
    if args.replay_from:
        if args.resume or args.out.exists() or not args.replay_from.is_dir():
            parser.error("--replay-from needs an existing source and a fresh --out")
        if args.expected_diffusion_steps is not None:
            parser.error("--expected-diffusion-steps requires --ckpt")
        replay_archive(args.replay_from, args.out)
        report_and_video(args)
        return
    ckpt = args.ckpt / "checkpoint.pt" if args.ckpt.is_dir() else args.ckpt
    if not ckpt.is_file():
        parser.error(f"Missing checkpoint: {ckpt}")
    if args.out.exists() and not args.resume:
        parser.error("Output directory exists. Use a fresh --out or --resume")
    torch.manual_seed(args.seed)
    model, setup = load_model(ckpt, args.device)
    if (args.expected_diffusion_steps is not None and
            len(setup["beta"]) != args.expected_diffusion_steps):
        parser.error(f"Checkpoint has {len(setup['beta'])} diffusion steps")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg["evaluation_version"] = EVALUATION_VERSION
    config_file = args.out / "experiment_config.json"
    if config_file.exists():
        old = json.loads(config_file.read_text(encoding="utf-8"))
        if old.get("evaluation_version") != EVALUATION_VERSION:
            raise ValueError("Legacy evaluation: use --replay-from and a fresh --out")
        # Video flags are post-evaluation presentation options.
        for key in ("record_video", "video_selection", "video_backend", "video_fps"):
            old[key] = cfg[key]
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
    report_and_video(args)


def report_and_video(args):
    from tt_task_report import build_report
    build_report(args.out)
    print("Saved result report:", args.out / "report.html", flush=True)
    if args.record_video:
        video_script = Path(__file__).with_name("render_task_videos.py")
        command = [sys.executable, str(video_script),
                   "--out", str(args.out), "--selection", args.video_selection,
                   "--backend", args.video_backend, "--fps", str(args.video_fps),
                   "--resume"]
        env = os.environ.copy()
        if args.video_backend in ("auto", "osmesa", "wireframe"):
            env["CUDA_VISIBLE_DEVICES"] = ""
        # A separate process is mandatory because this process imported
        # MuJoCo before the CLI/video backend was selected.
        print("Rendering physical replay videos:", args.video_backend, flush=True)
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            # Numeric physical results are already saved and remain valid.
            print(f"Video rendering failed ({exc}). Task results remain saved. "
                  "Run render_task_videos.py manually after installing the backend.",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
