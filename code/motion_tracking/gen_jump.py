"""MetalHead (A1, AMP, Isaac Gym) 학습된 정책으로 제자리 점프를 녹화한다.

isaacgym 환경에서 MetalHead 코드를 PYTHONPATH 로만 연결해 실행 (설치 X):
  cd codesign/third_party/MetalHead && \
  PYTHONPATH=$PWD:$PWD/rsl_rl ~/.conda/envs/isaacgym/bin/python ../../code/motion_tracking/gen_jump.py \
      --task=a1_amp_jump_cmd --load_run=example --headless True
출력: codesign/artifacts/sim_motions/jumpInPlace_raw.npz (정책 주기 30 ms, 다리 순서 FR FL RR RL, quat wxyz)
pd_replay.py --import_sim 이 50 Hz reference(fix_*) 로 바꾼다.
"""
import os

import isaacgym  # noqa: F401  (torch 보다 먼저)
import numpy as np
import torch
from legged_gym.envs import *  # noqa: F401,F403  task 등록
from legged_gym.utils import get_args, task_registry

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "sim_motions"))
LEGS = ["FR", "FL", "RR", "RL"]
STAND_S, AFTER_S, N_JUMPS = 1.5, 2.0, 2  # 서 있기 → 점프 → 착지 후 안정, 2 회 반복
JUMP_HEIGHT = 0.5  # 목표 몸통 높이 (m). MetalHead 학습 범위 0.4~0.8, 기본 서기 높이 0.32


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.jump_up_command = False  # 점프 신호는 아래에서 직접 준다
    env_cfg.env.episode_length_s = 100.0
    env_cfg.env.reference_state_initialization = False
    env_cfg.env.recovery_init_prob = 0
    train_cfg.runner.amp_num_preload_transitions = 1
    train_cfg.runner.resume = True
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    names = env.dof_names
    order = [names.index(f"{l}_{j}_joint") for l in LEGS for j in ("hip", "thigh", "calf")]
    feet = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], f"{l}_foot") for l in LEGS]
    stand, after = int(STAND_S / env.dt), int(AFTER_S / env.dt)
    log = {k: [] for k in ("base_pos", "base_quat", "q", "contact", "jump_sig")}
    for _ in range(N_JUMPS):
        for k in range(stand + after):
            env.commands[:, :3] = 0.0  # 제자리
            if k == stand:
                env.commands[:, 3] = JUMP_HEIGHT
                env.jump_sig[:] = 1
            env.compute_observations()
            obs = env.get_observations()
            env.step(policy(obs.detach()).detach())
            rs = env.root_states[0].cpu().numpy()
            log["base_pos"].append(rs[:3])
            log["base_quat"].append(rs[[6, 3, 4, 5]])  # Isaac (x,y,z,w) → (w,x,y,z)
            log["q"].append(env.dof_pos[0].cpu().numpy()[order])
            log["contact"].append(env.contact_forces[0, feet, 2].cpu().numpy() > 1.0)
            log["jump_sig"].append(float(env.jump_sig[0, 0]))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "jumpInPlace_raw.npz")
    np.savez(out, dt=env.dt, **{k: np.array(v) for k, v in log.items()})
    z = np.array(log["base_pos"])[:, 2]
    print(f"-> {out}  frames={len(z)} dt={env.dt:.3f} base z min/max={z.min():.3f}/{z.max():.3f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
