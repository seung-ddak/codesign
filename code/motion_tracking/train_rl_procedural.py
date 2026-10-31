import functools
import json
import os
import pickle
import sys
from pathlib import Path

# Reuse the original TT runtime; only the tracking environment/config live here.
TT = Path(__file__).resolve().parents[3] / "transformer-transformer"
if str(TT) not in sys.path:
    sys.path.insert(0, str(TT))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from absl import logging
from flax.training import orbax_utils
from omegaconf import OmegaConf
from orbax import checkpoint as ocp

from t2.mjx_env.utils import render_policy
from t2.mjx_env.wrapper import wrap_for_brax_training

from multi_design import build_bank, load_catalog, validate_split, wrap_for_multi_design
from pd_replay import build_model
from policy_io import load_normalizer

logging.set_verbosity(logging.WARNING)  # brax info 로그 끔

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def resolve_cfg(cfg):
    """choice 와 형상별 reference 경로를 확정한다."""
    # hydra 가 00000010 을 정수 10 으로 읽는 경우를 막는다.
    # choice=A1 이면 A1 몸 + 리타겟 없는 원본 reference (artifacts/pd_replay) — 좌표 변환 대조용
    choice = "A1" if str(cfg.choice) == "A1" else str(cfg.choice).zfill(8)
    cfg.choice = choice
    OmegaConf.resolve(cfg)
    for e in (cfg.env, cfg.eval_env):
        e.reference_dir = (os.path.join(os.path.dirname(e.reference_dir), "pd_replay") if choice == "A1"
                           else os.path.join(e.reference_dir, choice))
    return choice


def build_envs(cfg, choice):
    """TT model and its retargeted reference must use the same morphology."""
    model_fn = functools.partial(build_model, kp=cfg.kp, tau_scale=cfg.get("tau_scale"),
                                 torque_limit=cfg.get("torque_limit"))
    if not cfg.get("multi_design"):
        single = functools.partial(model_fn, choice=None if choice == "A1" else choice)
        return (hydra.utils.instantiate(cfg.env, mj_model_fn=single),
                hydra.utils.instantiate(cfg.eval_env, mj_model_fn=single))

    root = Path(cfg.env.reference_dir).parents[1]
    designs = load_catalog(root, choice)
    validate_split(list(cfg.train_designs), list(cfg.eval_designs), list(cfg.env.clips), list(cfg.eval_env.clips))
    if choice != designs["topology"]:
        raise ValueError("Multi-design training requires matching topology")
    designs_model_fn = functools.partial(model_fn, choice=choice)
    env, bank = build_bank(list(cfg.train_designs), cfg.env, root, choice, designs, designs_model_fn,
                           list(cfg.design_unlock_steps))
    eval_env, eval_bank = build_bank(list(cfg.eval_designs), cfg.eval_env, root, choice, designs, designs_model_fn,
                                     [0] * len(cfg.eval_designs))
    bank["rollout_dir"] = cfg.get("rollout_dir")
    env.design_bank, eval_env.design_bank = bank, eval_bank
    return env, eval_env


def make_progress_fn(metric_path):
    """Save the raw metrics this PPO exposes; never fabricate missing KL or losses."""
    def progress(num_steps, metrics):
        if any(not bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(metrics)):
            raise ValueError(f"Nonfinite PPO metric at step {num_steps}")
        scalars = {str(k): float(np.asarray(v).reshape(-1)[0]) for k, v in metrics.items() if np.asarray(v).size == 1}
        with metric_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"step": int(num_steps), **scalars}, ensure_ascii=False) + "\n")
        wandb.log(metrics, step=num_steps)
        print("-" * 40 + f"{num_steps}" + "-" * 40)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("-" * 80)
    return progress


def load_restore_params(path):
    """이전 run checkpoint (normalizer·policy·value) 에서 이어 학습 (사용자 결정 2026-10-12)."""
    params = ocp.PyTreeCheckpointer().restore(os.path.abspath(path))
    normalizer = load_normalizer(params[0])
    # 학습 루프의 count dtype (int32) 와 맞춤. {hi,lo} 합은 uint32
    print(f"restored {path}")
    return normalizer.replace(count=normalizer.count.astype(jnp.int32)), params[1], params[2]


def render_final(cfg, eval_env, make_policy, params, video_path):
    def with_morphology(state):
        if not cfg.get("multi_design"):
            return state
        return state.replace(obs=jnp.concatenate((state.obs, eval_env.design_bank["uniforms"][0])))

    render_policy(
        reset_fn=jax.jit(lambda rng: with_morphology(eval_env.reset(rng))),
        step_fn=jax.jit(lambda state, action: with_morphology(eval_env.step(state, action))),
        policy=jax.jit(make_policy(params, deterministic=True)),
        env=eval_env,
        video_path=video_path,
        camera="body_cam",
        seed=0,
    )


@hydra.main(config_path="config", config_name="run_rl_procedural", version_base="1.2")
def main(cfg):
    choice = resolve_cfg(cfg)
    wandb.init(project="transformer-transformer", tags=cfg.wandb.tags,
               config=OmegaConf.to_container(cfg, resolve=True))
    run_dir = Path(wandb.run.dir)
    pickle.dump(cfg, (run_dir / "config.pkl").open("wb"))
    ckpt_path = run_dir / "checkpoints"
    ckpt_path.mkdir(exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    env, eval_env = build_envs(cfg, choice)

    def save_checkpoint(current_step, make_policy, params):
        ocp.PyTreeCheckpointer().save(str(ckpt_path / str(current_step)), params, force=True,
                                      save_args=orbax_utils.save_args_from_target(params))

    wrap_env_fn = (functools.partial(wrap_for_multi_design, full_reset=cfg.full_reset) if cfg.get("multi_design")
                   else functools.partial(wrap_for_brax_training, full_reset=cfg.full_reset))
    # restore_params 는 hydra merge 불가 (jax 배열) → partial 후 전달
    make_inference_fn, params, _ = hydra.utils.instantiate(
        cfg.train,
        _partial_=True,
        policy_params_fn=save_checkpoint,
        wrap_env_fn=wrap_env_fn,
        environment=env,
        progress_fn=make_progress_fn(run_dir / "training_metrics.jsonl"),
        eval_env=eval_env,
    )(restore_params=load_restore_params(cfg.restore_ckpt) if cfg.get("restore_ckpt") else None)

    if cfg.get("render_final", True):
        video = str(run_dir / "final-rollout.mp4")
        render_final(cfg, eval_env, make_inference_fn, params, video)
        wandb.log({"rollout_video": wandb.Video(video)})


if __name__ == "__main__":
    main()
