"""Alternate a shared MJX walking policy and TT guidance; run_round() closes the loop."""

import argparse
import functools
import html
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TT = Path(os.environ.get("CODESIGN_TT", ROOT.parent / "transformer-transformer")).resolve()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, str(TT))
sys.path.insert(0, str(ROOT / "stage2/code"))


def setup_imports():
    os.chdir(TT)
    import t2.eval
    t2.eval.__path__.append(str(ROOT / "tt_patch/t2/eval"))


def write_report(out, rows, status):
    document = {"status": status, "rounds": rows}
    (out / "progress.json").write_text(json.dumps(document, indent=2, allow_nan=False))
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>"
                                   for k in ("round", "unique_designs", "return_mean",
                                             "success_mean", "gradient_norm", "seconds"))
                   + "</tr>" for row in rows)
    (out / "overview.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>공진화 학습</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto}'
        'td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}</style>'
        f'<h1>형상·보행 정책 공진화</h1><p>{html.escape(status)}</p>'
        '<p>보상: 낙상 전 전진 변위 / 목표 거리. 같은 제어기로 각 후보를 평가합니다.</p>'
        '<table><tr><th>Round</th><th>형상 수</th><th>평균 return</th><th>성공률</th>'
        '<th>ψ gradient norm</th><th>소요 초</th></tr>' + body + '</table></html>')


def resume_state(out, args):
    config_file = out / "config.json"
    if not config_file.exists():
        raise ValueError(f"Cannot resume without config.json: {out}")
    saved = json.loads(config_file.read_text())
    current = json.loads(json.dumps(vars(args), default=str))
    # More rounds or fewer parallel environments can be requested after a restart.
    for key in ("rounds", "num_envs", "check", "resume"):
        saved.pop(key, None)
        current.pop(key, None)
    if saved != current:
        raise ValueError("Run settings differ from config.json; use a new --out")
    rows = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()] \
        if (out / "history.jsonl").exists() else []
    for index, row in enumerate(rows):
        if row["round"] != index:
            raise ValueError("Non-contiguous run history")
        directory = out / f"round_{index:04d}"
        policy = directory / "psi.pt"
        if not policy.is_file():
            policy = directory / "phi.pt"  # D58 이전 checkpoint 호환
        if not all((directory / name).is_file() for name in ("theta.pkl", "designs.json")) \
                or not policy.is_file():
            raise ValueError(f"Missing checkpoint for completed round {index}")
    # The checkpoint directory is renamed into place before history is appended.
    while (directory := out / f"round_{len(rows):04d}").exists():
        row_file = directory / "row.json"
        if not row_file.is_file():
            raise ValueError(f"Incomplete checkpoint: {directory}")
        row = json.loads(row_file.read_text())
        if row["round"] != len(rows) or not all((directory / name).is_file()
                                                for name in ("theta.pkl", "designs.json")) \
                or not any((directory / name).is_file() for name in ("psi.pt", "phi.pt")):
            raise ValueError(f"Incomplete checkpoint: {directory}")
        rows.append(row)
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return rows, out / f"round_{len(rows)-1:04d}" if rows else None


def preflight(checkpoint, bank, require_gpu=True):
    errors = []
    for path in (checkpoint, checkpoint.with_name("cfg.pkl")):
        if not path.is_file():
            errors.append(f"Missing prior: {path}")
    choices = [p for p in bank.glob("*.zarr")
               if len(p.stem) == 8 and set(p.stem) <= {"0", "1"}]
    if len(choices) != 128:
        errors.append(f"Incomplete quadruped RoboToken bank: {len(choices)}/128 in {bank}")
    if (bank / ".download_in_progress").exists() and not (bank / ".download_complete").exists():
        errors.append("RoboToken bank download has not completed")
    if require_gpu:
        import torch
        import jax
        if not torch.cuda.is_available():
            errors.append("PyTorch cannot access a CUDA GPU")
        if not any(d.platform == "gpu" for d in jax.devices()):
            errors.append("JAX cannot access a CUDA GPU")
    return errors


def run_round(learner, theta, args, round_index, out):
    import jax
    import numpy as np
    from brax.training.agents.ppo import networks
    from t2.mjx_env import train as ppo
    from coevolution_env import WalkEnv, evaluate
    from t2.mjx_env.wrapper import wrap_for_brax_training

    started = time.monotonic()
    designs, transitions = learner.generate(args.seed + round_index)
    unique = {(tuple(d["choices"]), d["seed"]): d for d in designs if "invalid" not in d}
    if len(unique) < 2:
        raise RuntimeError(f"Only {len(unique)} distinct valid designs; increase --designs. "
                           "Refusing to silently train one fixed morphology.")
    environments = {key: WalkEnv(**d, goal=args.goal, threshold=args.upright,
                                  action_scale=args.action_scale) for key, d in unique.items()}
    before = None if theta is None else np.concatenate(
        [np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(theta[1])])
    order = list(environments)
    np.random.default_rng(args.seed + round_index).shuffle(order)
    network_factory = functools.partial(networks.make_ppo_networks,
                                        policy_hidden_layer_sizes=(128, 128),
                                        value_hidden_layer_sizes=(128, 128))
    for index, key in enumerate(order):
        print(f"round {round_index}: controller {index + 1}/{len(order)} {key}", flush=True)
        # ponytail: Brax restore_params keeps weights/statistics, resets Adam per block.
        # Expose full training state if block-boundary optimizer resets limit learning.
        make_policy, theta, _ = ppo.train(
            environment=environments[key], num_timesteps=args.theta_steps,
            num_envs=args.num_envs, num_eval_envs=args.num_envs,
            episode_length=args.horizon, unroll_length=16,
            batch_size=args.num_envs, num_minibatches=1, num_updates_per_batch=4,
            learning_rate=args.theta_lr, entropy_cost=0.001, discounting=1.0,
            gae_lambda=0.95, normalize_observations=True, max_grad_norm=1.,
            network_factory=network_factory, restore_params=theta, run_evals=False,
            num_evals=1, seed=args.seed + round_index * args.designs + index,
            wrap_env_fn=functools.partial(wrap_for_brax_training, full_reset=True))
    policy = make_policy(theta, deterministic=True)
    seeds = np.arange(args.eval_seeds) + args.seed + 100000
    scores = {key: evaluate(env, policy, seeds, args.horizon)
              for key, env in environments.items()}
    returns = [-1. if "invalid" in d else float(scores[(tuple(d["choices"]), d["seed"])][:, 0].mean())
               for d in designs]
    update = learner.update(transitions, returns)
    after = np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(theta[1])])
    delta = None if before is None else float(np.linalg.norm(after - before))
    round_dir = out / f"round_{round_index:04d}"
    staging = out / f"round_{round_index:04d}.incomplete"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    with (staging / "theta.pkl").open("wb") as f:
        pickle.dump(jax.device_get(theta), f)
    learner.save(staging / "psi.pt")
    records = [{**design, "episodes": scores[key].tolist()} for key, design in unique.items()]
    (staging / "designs.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    best_key = max(scores, key=lambda k: scores[k][:, 0].mean())
    import mujoco
    mujoco.mj_saveLastXML(str(staging / "best.xml"), environments[best_key].mj_model)
    row = {"round": round_index, "unique_designs": len(unique),
           "invalid_designs": len(designs) - sum("invalid" not in d for d in designs),
           "return_mean": float(np.mean(returns)),
           "success_mean": float(np.mean([v[:, 2].mean() for v in scores.values()])),
           "theta_parameter_delta": delta, "theta_initialized": before is None,
           "best_design": unique[best_key], "checkpoint": str(round_dir),
           "seconds": round(time.monotonic() - started, 1), **update}
    (staging / "row.json").write_text(json.dumps(row, allow_nan=False))
    staging.rename(round_dir)
    return theta, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check resources without training")
    parser.add_argument("--resume", action="store_true", help="Resume completed rounds from --out")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--designs", type=int, default=8)
    parser.add_argument("--theta-steps", type=int, default=131072, help="PPO steps per candidate per round")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--eval-seeds", type=int, default=9)
    parser.add_argument("--theta-lr", type=float, default=1e-4)
    parser.add_argument("--goal", type=float, default=1.)
    parser.add_argument("--upright", type=float, default=0.1)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path,
                        default=TT / "checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt")
    parser.add_argument("--bank", type=Path, default=TT / "umi_on_legs_plus_plus_robotokens")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "stage3/runs" / time.strftime("coevolution_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if min(args.rounds, args.theta_steps, args.num_envs, args.horizon, args.eval_seeds) < 1 or args.designs < 2:
        parser.error("Positive budgets and at least two designs are required")
    if args.goal <= 0 or not 0 < args.upright < 1 or args.action_scale <= 0 or args.theta_lr <= 0:
        parser.error("Invalid reward/control parameters")
    args.checkpoint, args.bank, args.out = args.checkpoint.resolve(), args.bank.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "config.json").exists() and not args.resume:
        parser.error("Output directory already contains a run; pass --resume")
    try:
        rows, latest = resume_state(args.out, args) if args.resume else ([], None)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.resume:
        write_report(args.out, [], "Checking resources")
    errors = preflight(args.checkpoint, args.bank)
    if errors:
        write_report(args.out, [], "Blocked: " + "; ".join(errors))
        parser.exit(2, "\n".join(errors) + "\n")
    if args.check:
        write_report(args.out, [], "Resources available; no training started")
        print("Resources available")
        return
    if not os.environ.get("TMUX"):
        parser.error("Run training inside tmux")
    setup_imports()
    from ddpo_hardware import DesignLearner
    if not args.resume:
        (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    theta = None
    try:
        learner = DesignLearner(TT, args.checkpoint, args.bank, args.designs, "cuda", args.seed,
                                trajectory_dir=args.out)
        if latest is not None:
            with (latest / "theta.pkl").open("rb") as stream:
                theta = pickle.load(stream)
            policy = latest / "psi.pt"
            learner.load(policy if policy.is_file() else latest / "phi.pt")
            print(f"Resuming at round {len(rows)} from {latest}", flush=True)
        for round_index in range(len(rows), args.rounds):
            write_report(args.out, rows, f"Training round {round_index}")
            theta, row = run_round(learner, theta, args, round_index, args.out)
            rows.append(row)
            with (args.out / "history.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            write_report(args.out, rows, f"Finished round {round_index}")
            print(json.dumps(row), flush=True)
        write_report(args.out, rows, "Requested rounds completed; assess walking success in the metrics")
    except Exception as exc:
        write_report(args.out, rows, f"Failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
