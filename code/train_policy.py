"""257 관측 locomotion expert 가중치에서 시작해 task expert 를 학습한다."""

import argparse
import functools
import itertools
import json
import os
import sys
import time
from pathlib import Path

# ── 경로·환경변수 (jax import 전에 설정) ─────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TT = Path(os.environ.get("CODESIGN_TT", ROOT / "transformer-transformer")).resolve()
LAUNCH_DIR = Path.cwd()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(TT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(TT)  # TT asset 경로가 상대경로

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from brax.training import logger as metric_logger  # noqa: E402
from brax.training.agents.ppo import checkpoint, networks  # noqa: E402

from task_env import (ACTION_SCALE, COMMAND_RANGES, EPISODE_STEPS, JUMP_FLAG_MEAN,  # noqa: E402
                      OBSERVATION_SIZE, ONLY_POSITIVE_REWARDS, T5_OBSERVATION_SIZE,
                      TASK_REWARD_SCALES, TERMINATE_AFTER_CONTACTS_ON, make_training_env,
                      structure_key)
from brax.training.agents.ppo import train as ppo  # noqa: E402


# brax 가 아직 제거된 jax.device_put_replicated 를 부른다. 이 머신은 CUDA 장치 1 개로 학습하므로
# pmap 의 replica 축을 앞에 붙이는 것으로 완전히 대체된다.
def _device_put_replicated(value, devices):
    if len(devices) != 1:
        raise ValueError("This compatibility path supports one training device")
    replicated = jax.tree.map(lambda x: jnp.expand_dims(x, 0), value)
    return jax.device_put(replicated, devices[0])


jax.device_put_replicated = _device_put_replicated

# ── 학습 설정 ────────────────────────────────────────────────────────────────
INITIAL_CHECKPOINT = ROOT / "codesign/runs/expert/locomotion/checkpoints/000000000000"
NETWORK = dict(policy_hidden_layer_sizes=(512, 256, 128),
               value_hidden_layer_sizes=(512, 256, 128), activation=jax.nn.elu,
               distribution_type="normal", noise_std_type="scalar", init_noise_std=1.0)
PPO = dict(num_envs=1024, unroll_length=24, batch_size=256, num_minibatches=4,
           num_updates_per_batch=5, learning_rate=5e-5, entropy_cost=0.01,
           discounting=0.99, gae_lambda=0.95, clipping_epsilon=0.2,
           max_grad_norm=1.0, normalize_advantage=True, normalize_observations=True,
           reward_scaling=1.0, desired_kl=0.01)


def main():
    # 1. 인자
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--phase", choices=("flat", "jump", "universal"), default="flat")
    parser.add_argument("--seeds-per-choice", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=24_576_000)
    parser.add_argument("--num-envs", type=int, default=PPO["num_envs"])
    parser.add_argument("--checkpoints", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=PPO["learning_rate"])
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.verify_only and args.out is None:
        parser.error("--out is required unless --verify-only is used")

    # 2. 시작 checkpoint (관측 크기 확인)
    checkpoint_config = json.loads((args.checkpoint / "ppo_network_config.json").read_text())
    if checkpoint_config["observation_size"] != OBSERVATION_SIZE:
        raise ValueError(f"Expected {OBSERVATION_SIZE}-observation checkpoint, got "
                         f"{checkpoint_config['observation_size']}")
    params = checkpoint.load(str(args.checkpoint))
    print(f"Loaded expert checkpoint: {args.checkpoint}")
    if args.phase == "jump":
        params = set_flag_statistics(params, JUMP_FLAG_MEAN)
    if args.verify_only:
        return

    # 3. run 폴더·설정 기록
    out = (LAUNCH_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    groups = sample_designs(args.seeds_per_choice, args.seed, args.max_groups)
    ppo_kw = {**PPO, "num_envs": args.num_envs, "learning_rate": args.learning_rate,
              "batch_size": args.num_envs // PPO["num_minibatches"]}
    config = {"source_checkpoint": str(args.checkpoint), "observation_size": OBSERVATION_SIZE,
              "args": {k: str(v) for k, v in vars(args).items()}, "ppo": ppo_kw,
              "network": {**NETWORK, "activation": "elu"}, "groups": len(groups),
              "designs": sum(map(len, groups)), "action_scale": ACTION_SCALE,
              "command_ranges": COMMAND_RANGES, "reward_scales": TASK_REWARD_SCALES,
              "only_positive_rewards": ONLY_POSITIVE_REWARDS,
              "terminate_after_contacts_on": TERMINATE_AFTER_CONTACTS_ON}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "designs.json").write_text(json.dumps(groups))

    # 4. 환경
    env = make_training_env(groups, args.phase)

    # 5. 진행 로그·checkpoint 저장 콜백
    log = (out / "progress.jsonl").open("a")
    started = time.time()

    def progress(step, metrics):
        row = {"step": int(step), "wall_s": round(time.time() - started, 1),
               **{k: float(v) for k, v in metrics.items()}}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(" ".join(f"{k}={row[k]:.4g}" for k in
                       ("step", "training/sps", "episode/sum_reward", "episode/length")
                       if k in row), flush=True)

    def save(step, _, saved_params):
        checkpoint.save(str(out / "checkpoints"), step, saved_params,
                        checkpoint.network_config(
                            observation_size=env.observation_size,
                            action_size=env.action_size,
                            normalize_observations=True,
                            network_factory=functools.partial(
                                networks.make_ppo_networks, **NETWORK)))

    # 6. PPO 학습
    ppo.train(environment=env, wrap_env_fn=lambda e, **_: e,
              episode_length=EPISODE_STEPS, num_timesteps=args.timesteps,
              num_evals=args.checkpoints + 1, run_evals=False, log_training_metrics=True,
              progress_fn=progress, policy_params_fn=save, seed=args.seed,
              network_factory=functools.partial(networks.make_ppo_networks, **NETWORK),
              restore_params=params, restore_value_fn=True, **ppo_kw)


def set_flag_statistics(params, p):
    """jump flag 관측(index 69)의 정규화 통계를 0/1 분포(평균 p)로 설정한다.
    locomotion 학습 내내 0 이라 그대로 두면 std 가 0 근처로 잡혀 flag=1 이 ~15 로 들어간다.
    누적 count 가 커서 이후 학습 중엔 거의 변하지 않는다."""
    norm, i = params[0], T5_OBSERVATION_SIZE
    count = norm.count
    count = float(count.hi) * 2. ** 32 + float(count.lo) if hasattr(count, "hi") else float(count)
    var = p * (1. - p)
    norm = norm.replace(mean=jnp.asarray(norm.mean).at[i].set(p),
                        summed_variance=jnp.asarray(norm.summed_variance).at[i].set(var * count),  # Welford: 분산 x count
                        std=jnp.asarray(norm.std).at[i].set(np.sqrt(var)))
    print(f"jump flag normalizer: mean {p:.3f}, std {np.sqrt(var):.3f}")
    return (norm, *params[1:])


def sample_designs(seeds_per_choice, seed, max_groups=None):
    """256 개 choice 마다 형상 seed 를 뽑아 구조 그룹별로 묶는다."""
    if not 1 <= seeds_per_choice <= 10000:
        raise ValueError("--seeds-per-choice must be in [1, 10000]")
    choices = sorted(json.loads((TT / "choice_to_ckpt_path.json").read_text()))
    rng, groups = np.random.RandomState(seed), {}
    for choice in ([int(x) for x in key] for key in choices):
        for morphology_seed in rng.choice(10000, seeds_per_choice, replace=False):
            groups.setdefault(structure_key(choice), []).append((choice, int(morphology_seed)))
        
    return list(groups.values())[:max_groups]


if __name__ == "__main__":
    # brax 학습 metric logger 가 train_metrics=None 을 받으면 실패해서 {} 로 바꿔 넘긴다
    update = metric_logger.EpisodeMetricsLogger.update_episode_metrics
    metric_logger.EpisodeMetricsLogger.update_episode_metrics = (
        lambda self, episode_metrics, dones, train_metrics=None:
        update(self, episode_metrics, dones, train_metrics or {}))
    main()
