import functools
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Reuse the original TT runtime; only the tracking environment/config live here.
TT = Path(__file__).resolve().parents[3] / "transformer-transformer"
if str(TT) not in sys.path:
    sys.path.insert(0, str(TT))

import jax
import jax.numpy as jnp
from absl import logging
from mujoco import mjx

from t2.mjx_env.base_env import MjxEnv, State

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)
from t2.mjx_env.utils import render_policy
from t2.mjx_env.wrapper import wrap_for_brax_training

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import hydra
import jax
from flax.training import orbax_utils
from omegaconf import OmegaConf
from orbax import checkpoint as ocp

import wandb


@hydra.main(
    config_path="config", config_name="run_rl_procedural", version_base="1.2"
)
def main(cfg):
    # hydra 가 00000010 을 정수 10 으로 읽는 경우를 막는다. reference 는 형상별 폴더
    # choice=A1 이면 A1 몸 + 리타겟 없는 원본 reference (artifacts/pd_replay) — 좌표 변환 대조용
    choice = "A1" if str(cfg.choice) == "A1" else str(cfg.choice).zfill(8)
    cfg.choice = choice
    OmegaConf.resolve(cfg)
    for e in (cfg.env, cfg.eval_env):
        e.reference_dir = (os.path.join(os.path.dirname(e.reference_dir), "pd_replay") if choice == "A1"
                           else os.path.join(e.reference_dir, choice))
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="transformer-transformer",
        tags=cfg.wandb.tags,
        config=config_dict,
    )
    pickle.dump(cfg, open(f"{wandb.run.dir}/config.pkl", "wb"))

    # Set up checkpoint directory
    ckpt_path = wandb.run.dir + "/checkpoints"
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # TT model and its retargeted reference must use the same morphology.
    from pd_replay import build_model

    mj_model_fn = functools.partial(build_model, choice=None if choice == "A1" else choice, kp=cfg.kp,
                                    tau_scale=cfg.get("tau_scale"), torque_limit=cfg.get("torque_limit"))

    randomization_fn = None

    env: MjxEnv = hydra.utils.instantiate(cfg.env, mj_model_fn=mj_model_fn)
    # Load evaluation environment
    eval_env: MjxEnv = hydra.utils.instantiate(cfg.eval_env, mj_model_fn=mj_model_fn)
    if cfg.get("multi_design"):
        from multi_design import make_bank, randomize_bank, wrap_for_multi_design

        root = Path(cfg.env.reference_dir).parents[1]
        designs = json.loads((root / "designs.json").read_text())
        design_names = list(cfg.train_designs)
        if design_names[0] != "e000" or choice != designs["topology"]:
            raise ValueError("Multi-design training requires e000 as base and matching topology")
        design_envs = [env]
        for name in design_names[1:]:
            if name not in designs:
                raise ValueError(f"Unknown design: {name}")
            design_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env, resolve=True))
            design_cfg.reference_dir = str(root / name / choice)
            design_envs.append(hydra.utils.instantiate(
                design_cfg, mj_model_fn=functools.partial(build_model, choice=choice,
                    uniforms=designs[name], kp=cfg.kp, tau_scale=cfg.get("tau_scale"),
                    torque_limit=cfg.get("torque_limit"))))
        bank = make_bank(design_envs, [designs[name] for name in design_names])
        env.design_bank = bank
        eval_env.design_bank = bank
        randomization_fn = functools.partial(randomize_bank, bank)

    def randomization_obs_fn(model: mjx.Model):
        extra_obs = []
        for attr in cfg.extra_obs_attrs:
            extra_obs.append(getattr(model, attr).flatten())
        return jnp.hstack(extra_obs)

    jit_randomization_obs_fn = jax.jit(randomization_obs_fn)

    def reset(rng: jax.Array):
        state = eval_env.reset(rng)
        if cfg.get("multi_design"):
            state = state.replace(obs=jnp.concatenate((state.obs, eval_env.design_bank["uniforms"][0])))
        if cfg.observe_domain_randomized_attrs:
            extra_obs = jit_randomization_obs_fn(eval_env.mjx_model)
            state = state.replace(
                obs=jax.tree.map(
                    lambda x, y: jnp.hstack([x, y]),
                    state.obs,
                    extra_obs,
                )
            )
        return state

    jit_reset = jax.jit(reset)

    def step(state: State, action: jax.Array):
        state = eval_env.step(state, action)
        if cfg.get("multi_design"):
            state = state.replace(obs=jnp.concatenate((state.obs, eval_env.design_bank["uniforms"][0])))
        if cfg.observe_domain_randomized_attrs:
            extra_obs = jit_randomization_obs_fn(eval_env.mjx_model)
            state = state.replace(
                obs=jax.tree.map(
                    lambda x, y: jnp.hstack([x, y]),
                    state.obs,
                    extra_obs,
                )
            )
        return state

    jit_step = jax.jit(step)

    def render_rollout(
        make_policy,
        params,
        video_path: str,
        seed: int = 0,
    ):
        # Create inference function
        # params should be a tuple of (normalizer_params, policy_params, ....) (only first two are used)
        inference_fn = make_policy(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)

        return render_policy(
            reset_fn=jit_reset,
            step_fn=jit_step,
            policy=jit_inference_fn,
            env=eval_env,
            video_path=video_path,
            camera="body_cam",
            seed=seed,
        )

    # Define policy parameters function for saving checkpoints
    def policy_params_fn(current_step, make_policy, params):
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path + f"/{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        if cfg.render_on_checkpoint:
            render_path = path + f"/{current_step:06d}.mp4"
            start_time = time.time()
            render_rollout(
                make_policy,
                (params[0], params[1], params[2]),
                str(render_path),
            )
            end_time = time.time()
            print(f"Rendered rollout in {end_time - start_time:.2f} seconds")
            wandb.log(
                {"rollout_video": wandb.Video(str(render_path), format="mp4")},
                step=current_step,
            )

    # Progress function for logging
    def progress(num_steps, metrics):
        has_nan_metrics = any(jnp.isnan(v) for v in jax.tree_util.tree_leaves(metrics))
        wandb.log(metrics, step=num_steps)
        if has_nan_metrics:
            print(f"[{num_steps:06d}] NaN metrics detected")
            # terminate the training
            raise ValueError("NaN metrics detected")
        print("-" * 40 + f"{num_steps}" + "-" * 40)
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("-" * 80)

    # restore_ckpt: 이전 run checkpoint (normalizer·policy·value) 에서 이어 학습 (사용자 결정 2026-10-12)
    restore = None
    if cfg.get("restore_ckpt"):
        import dataclasses

        from brax.training.acme import running_statistics

        p = ocp.PyTreeCheckpointer().restore(os.path.abspath(cfg.restore_ckpt))
        fields = {f.name for f in dataclasses.fields(running_statistics.RunningStatisticsState)}
        norm = running_statistics.RunningStatisticsState(**{  # orbax 는 dataclass 를 dict 로 복원 (stair_design.load_run 과 같은 변환)
            k: jnp.asarray(sum(jnp.asarray(x) for x in v.values()) if isinstance(v, dict) else v)
            for k, v in p[0].items() if k in fields})
        norm = norm.replace(count=norm.count.astype(jnp.int32))  # 학습 루프의 count dtype (int32) 와 맞춤. {hi,lo} 합은 uint32
        restore = (norm, p[1], p[2])
        print(f"restored {cfg.restore_ckpt}")

    # Train or load the model
    make_inference_fn, params, _ = hydra.utils.instantiate(  # restore_params 는 hydra merge 불가 (jax 배열) → partial 후 전달
        cfg.train,
        _partial_=True,
        policy_params_fn=policy_params_fn,
        wrap_env_fn=(functools.partial(wrap_for_multi_design, full_reset=cfg.full_reset)
                     if cfg.get("multi_design") else functools.partial(
                         wrap_for_brax_training,
                         full_reset=cfg.full_reset,
                         randomization_obs_fn=(jit_randomization_obs_fn
                                               if cfg.observe_domain_randomized_attrs else None))),
        randomization_fn=randomization_fn,
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )(restore_params=restore)

    if cfg.get("render_final", True):
        render_path = wandb.run.dir + "/final-rollout.mp4"
        render_rollout(make_inference_fn, params, str(render_path))
        wandb.log({"rollout_video": wandb.Video(str(render_path))})


if __name__ == "__main__":
    main()
