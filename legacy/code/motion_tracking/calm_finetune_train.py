"""A1 CALM LLC 를 TT 형상들에서 PPO 로 미세조정한다 (calm_finetune.CalmTrackEnv + make_calm_ppo_networks).

계획 파일 (plan.json) 은 shared_calm 형식 (designs, assignments, config). references/<design>/<topology>/calm_XXXXX.npz 를 쓴다.
멈춘 뒤 같은 --out 으로 다시 실행하면 learner.msgpack 에서 이어 간다.
"""
import argparse, functools, json, math, pickle, sys, time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE.parents[2] / "transformer-transformer"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--visit", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seconds", type=float, required=True)
    a = p.parse_args(); out = a.out.resolve()
    import jax
    from omegaconf import OmegaConf
    from orbax import checkpoint as ocp
    from multi_design import build_bank, EpisodeDesignReset, MultiDesignVmapWrapper
    from pd_replay import build_model
    from t2.mjx_env.train import train
    import calm_finetune as cf

    plan = json.load(open(out / "plan.json")); c = plan["config"]
    names = [r["design"] for r in plan["designs"]]
    designs = {r["design"]: r["E"] for r in plan["designs"]}
    clips = {k: ["calm_%05d" % i for i in ids] for k, ids in plan["assignments"].items()}
    root = out / "references"; run = out / "stages" / f"{a.visit:03d}" / "files"; run.mkdir(parents=True, exist_ok=True)
    envcfg = OmegaConf.load(CODE / "config/env/a1_track.yaml")
    envcfg.update(dict(_target_="calm_finetune.CalmTrackEnv", reference_dir=str(root), clips=[], ctrl_dt=1 / 30, sim_dt=1 / c["physics_hz"],
                       max_episode_length=600, w_vel=1., termination_pose_consecutive_steps=15, reset_qpos=0., reset_qvel=0.,
                       velocity_kick=[0., 0.], penalty_foot_slip=float(c["penalty_foot_slip"])))
    model = functools.partial(build_model, choice=plan["topology"], torque_limit=33.5)
    env, bank = build_bank(names, envcfg, root, plan["topology"], designs, model, [0] * len(names), clips_by_design=clips)
    env.design_bank = bank

    def wrap(e, episode_length, action_repeat, randomization_fn, **kw):
        return EpisodeDesignReset(MultiDesignVmapWrapper(e, e.design_bank), episode_length, action_repeat)

    network = functools.partial(cf.make_calm_ppo_networks, init_log_std=float(np.log(c["init_std"])))
    (run / "config.json").write_text(json.dumps(dict(plan_config=c, env=OmegaConf.to_container(envcfg, resolve=True)), indent=1))
    initial = json.load(open(out / "learner.json"))["step"] if (out / "learner.json").exists() else 0

    def progress(step, metrics):
        step = initial + int(step) if all(str(k).startswith("episode/") for k in metrics) else int(step)
        scalars = {str(k): float(np.asarray(v)) for k, v in metrics.items() if np.asarray(v).size == 1}
        if not all(np.isfinite(v) for v in scalars.values()):
            raise FloatingPointError("Nonfinite PPO metrics")
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(time=time.time(), visit=a.visit, step=step, metrics=scalars)) + "\n")
        print(f"학습 전이 {step:,} | " + str({k: round(v, 4) for k, v in scalars.items() if k in ("episode/sum_reward", "episode/length", "training/sps")}), flush=True)

    def checkpoint(step, make_policy, params):
        path = run / "checkpoints" / str(step); path.parent.mkdir(exist_ok=True)
        ocp.PyTreeCheckpointer().save(str(path), params, force=True)
        (out / "checkpoint.json").write_text(json.dumps(dict(step=int(step), path=str(path), visit=a.visit, time=time.time())))

    train(environment=env, eval_env=env, num_timesteps=a.steps, num_envs=c["num_envs"], batch_size=c["batch_size"],
          unroll_length=c["unroll_length"], num_minibatches=c["num_minibatches"], num_updates_per_batch=c["num_updates_per_batch"],
          learning_rate=c["learning_rate"], entropy_cost=c["entropy_cost"], discounting=.99, gae_lambda=.95, clipping_epsilon=.2,
          max_grad_norm=1., normalize_observations=False, episode_length=600,
          num_evals=max(2, math.ceil(a.steps / c["checkpoint_steps"]) + 1), num_eval_envs=1, run_evals=False,
          network_factory=network, wrap_env_fn=wrap, progress_fn=progress, policy_params_fn=checkpoint,
          full_state_path=str(out / "learner.msgpack"), max_walltime_seconds=a.seconds,
          log_training_metrics=True, training_metrics_steps=c.get("metrics_steps", 2_000_000), seed=c["seed"] + a.visit)
    jax.effects_barrier()


if __name__ == "__main__":
    main()
