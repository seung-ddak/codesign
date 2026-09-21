"""설계 (choices, seed) 하나를 expert 로 굴려 **D22 reward** 를 낸다 — 2 단의 채점기.

π_φ 는 이 값 하나만 본다. gradient 는 어디에도 없다 (그게 2 단의 요점).

  .venv/bin/python tt_codesign/code/eval_design_d22.py --spec results/compare_designs.json
"""

import os as _os
_R = _os.environ.get("CODESIGN_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_TT = _os.environ.get("CODESIGN_TT", _os.path.join(_os.path.dirname(_R), "transformer-transformer"))

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # 평가는 CPU 로만

import hydra
import numpy as np
from hydra import compose, initialize_config_dir

TT = _TT
CODE = _os.path.join(_R, "code")
os.chdir(TT)
sys.path.insert(0, CODE)

from d22_reward import D22Reward


def make_runner(choices, expert_ckpt, pkl):
    with initialize_config_dir(config_dir=f"{TT}/config", version_base="1.3"):
        cfg = compose(
            config_name="evaluate_hardware_opt_rl",
            overrides=[
                "ckpt_path=checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt",
                f"pickle_path={pkl}",
            ],
        )
    rc = cfg.eval_fn.runner
    rc.env.robot_generator.choices = list(choices)
    rc.env.robot_generator.num_uniforms = 13
    rc.ckpt_path = expert_ckpt
    rc.env.pickle_path = pkl
    return hydra.utils.instantiate(rc, log_dir="/tmp/d22_eval")


def score(runner, hardware_seed, traj_idx, max_steps):
    env = runner.env
    env.reset_hardware(hardware_seed=hardware_seed)
    runner.post_hardware_reset(hardware_seed=hardware_seed)
    env.traj_idx = traj_idx
    env.reset_no_obs(seed=traj_idx)
    env.p.forward()
    obs = env.get_obs()

    rew = D22Reward(env, runner=runner)
    x0 = float(env.d.qpos[0])
    max_x = x0
    for _ in range(max_steps):
        action = runner.default_policy(obs)
        ctrl = np.asarray(action["ctrl/target_qpos"]).reshape(-1)[: env.d.ctrl.shape[0]]
        obs, _, done, _ = env.step(ctrl)
        rew.step(ctrl)
        max_x = max(max_x, float(env.d.qpos[0]))
        if done or rew.terminated:
            break
    out = rew.summary()
    out["distance"] = round(max_x - x0, 4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default=f"{_R}/results/compare_designs.json")
    p.add_argument("--traj", type=int, default=1)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--pkl", default="data/flat_walk_long.pkl")
    args = p.parse_args()

    spec = json.load(open(args.spec))
    for tag, s in spec.items():
        t0 = time.time()
        runner = make_runner(s["choices"], s["expert"], args.pkl)
        out = score(runner, s["seed"], args.traj, args.steps)
        print(f"\n[{tag}] choices {''.join(map(str, s['choices']))} seed {s['seed']} "
              f"({time.time() - t0:.0f}s)")
        print(f"  D22 reward {out['reward']:9.2f} | step {out['steps']:3d} "
              f"| 종료 {out['terminated']} | 전진 {out['distance']:+.3f} m")
        for k in ("exp_pose_err", "orientation", "bad_contact", "default_qpos",
                  "joint_pos_limit", "action_rate", "joint_vel", "joint_acc",
                  "torque", "termination"):
            print(f"    {k:18s} {out[k]:10.2f}")


if __name__ == "__main__":
    main()
