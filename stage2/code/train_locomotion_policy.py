"""T5: pre-train the morphology-conditioned locomotion policy pi_theta(o, E, command).

PPO settings follow legged_gym LeggedRobotCfgPPO where brax PPO has an equivalent:
actor/critic [512, 256, 128] elu, Gaussian with init std 1.0, 24 steps/env per update,
4 minibatches x 5 epochs, lr 1e-3, entropy 0.01, gamma 0.99, lambda 0.95, clip 0.2,
grad norm 1.0, 4096 envs, 1500 iterations, adaptive-KL learning rate (desired_kl 0.01, added
to TT t2/mjx_env/train.py 2026-09-27). Not reproduced: clipped value loss.

Designs: every one of the 128 bank choices x --seeds-per-choice bank seeds (Q10).
"""

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TT = Path(os.environ.get("CODESIGN_TT", ROOT.parent / "transformer-transformer")).resolve()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # device 1 is a 2 GB display card
sys.path.insert(0, str(TT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
LAUNCH_DIR = Path.cwd()
os.chdir(TT)  # TT component MJCF paths are repo-relative

import jax
import numpy as np
from brax.training import logger as metric_logger
from brax.training.agents.ppo import networks

from locomotion_env import (ACTION_SCALE, COMMAND_RANGES, EPISODE_STEPS, ONLY_POSITIVE_REWARDS,
                            REWARD_SCALES, TERMINATE_AFTER_CONTACTS_ON, LocomotionEnv)
from multi_design_env import MultiDesignEnv, structure_key
from t2.mjx_env import train as ppo

# TT train.py calls update_episode_metrics without the train_metrics argument that the
# installed brax logger requires.
_update = metric_logger.EpisodeMetricsLogger.update_episode_metrics
metric_logger.EpisodeMetricsLogger.update_episode_metrics = (
    lambda self, episode_metrics, dones, train_metrics=None:
    _update(self, episode_metrics, dones, train_metrics or {}))

BANK_SEEDS = 10000  # hardware_meta/seed of every umi_on_legs_plus_plus_robotokens zarr is 0..9999
PPO = dict(num_envs=4096, unroll_length=24, batch_size=1024, num_minibatches=4,
           num_updates_per_batch=5, learning_rate=1e-3, entropy_cost=0.01, discounting=0.99,
           gae_lambda=0.95, clipping_epsilon=0.2, max_grad_norm=1.0, normalize_advantage=True,
           normalize_observations=True, reward_scaling=1.0, desired_kl=0.01)
NETWORK = dict(policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
               activation=jax.nn.elu, distribution_type="normal", noise_std_type="scalar",
               init_noise_std=1.0)


def sample_designs(seeds_per_choice, seed, max_groups=None):
    """All bank choices x sampled bank seeds, grouped by MJX structure."""
    choices = sorted(json.loads((TT / "choice_to_ckpt_path.json").read_text()))
    rng = np.random.RandomState(seed)
    groups = {}
    for key in choices:
        c = [int(x) for x in key]
        for s in rng.choice(BANK_SEEDS, seeds_per_choice, replace=False):
            groups.setdefault(structure_key(c), []).append((c, int(s)))
    return list(groups.values())[:max_groups]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds-per-choice", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=1500 * PPO["num_envs"] * PPO["unroll_length"])
    p.add_argument("--num-envs", type=int, default=PPO["num_envs"])
    p.add_argument("--checkpoints", type=int, default=30)
    p.add_argument("--max-groups", type=int, help="smoke tests only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", choices=("flat", "task"), default="flat",
                   help="flat = T5 LocomotionEnv, task = T8 TaskEnv (terrain + height scan + jump)")
    args = p.parse_args()
    out = (LAUNCH_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)

    groups = sample_designs(args.seeds_per_choice, args.seed, args.max_groups)
    ppo_kw = {**PPO, "num_envs": args.num_envs, "batch_size": args.num_envs // PPO["num_minibatches"]}
    config = {"args": {k: str(v) for k, v in vars(args).items()}, "ppo": ppo_kw,
              "network": {**NETWORK, "activation": "elu"}, "episode_steps": EPISODE_STEPS,
              "action_scale": ACTION_SCALE, "only_positive_rewards": ONLY_POSITIVE_REWARDS,
              "reward_scales_x_dt": REWARD_SCALES, "command_ranges": COMMAND_RANGES,
              "terminate_after_contacts_on": TERMINATE_AFTER_CONTACTS_ON,
              "groups": len(groups), "designs": sum(map(len, groups)),
              "jax_devices": [str(d) for d in jax.devices()]}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "designs.json").write_text(json.dumps(groups))

    t = time.time()
    env_cls = LocomotionEnv
    if args.env == "task":
        from task_env import TaskEnv as env_cls
    env = MultiDesignEnv(groups, EPISODE_STEPS, env_cls=env_cls)
    print(f"built {config['designs']} designs in {len(groups)} groups: {time.time() - t:.0f}s", flush=True)

    log = (out / "progress.jsonl").open("a")
    start = time.time()

    def progress(step, metrics):
        row = {"step": int(step), "wall_s": round(time.time() - start, 1),
               **{k: float(v) for k, v in metrics.items()}}
        log.write(json.dumps(row) + "\n")
        log.flush()
        keys = ("training/sps", "training/learning_rate", "training/kl_mean", "episode/sum_reward", "episode/length")
        print(" ".join(f"{k}={row[k]:.4g}" for k in ("step", *keys) if k in row), flush=True)

    _, params, _ = ppo.train(
        environment=env, wrap_env_fn=lambda e, **_: e, episode_length=EPISODE_STEPS,
        num_timesteps=args.timesteps, num_evals=args.checkpoints + 1, run_evals=False,
        log_training_metrics=True, progress_fn=progress, seed=args.seed,
        network_factory=functools.partial(networks.make_ppo_networks, **NETWORK),
        save_checkpoint_path=str(out / "checkpoints"), **ppo_kw)
    print(f"done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
