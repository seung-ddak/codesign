"""legged_gym (ETH 원본) A1 rough 정책으로 계단 오르기·내리기를 녹화한다.

코스 (x 방향, 폭 전체): 평지 → 10 cm × 4 단 오르기 (단 깊이 0.3 m) → 1 m 평판 → 4 단 내리기 → 평지. 사용자 결정 2026-10-11.
isaacgym 환경에서 원본 legged_gym·rsl_rl v1.0.2 를 PYTHONPATH 로만 연결해 실행:
  cd codesign/third_party/legged_gym && \
  PYTHONPATH=$PWD:$PWD/../rsl_rl ~/.conda/envs/isaacgym/bin/python ../../code/motion_tracking/gen_stairs.py --task=a1 --headless
출력: codesign/artifacts/sim_motions/stairs_raw.npz (50 Hz, 다리 순서 FR FL RR RL, quat wxyz, 계단 상자 world 좌표)
"""
import os

import isaacgym  # noqa: F401  (torch 보다 먼저)
import numpy as np
import torch
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.terrain import Terrain

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "sim_motions"))
LEGS = ["FR", "FL", "RR", "RL"]
STEP_H, STEP_D, N_STEP, PLATFORM = 0.10, 0.3, 4, 1.0
TILE = 12.0  # m. 로봇은 타일 중앙 ±1 m 에 생성 → 계단은 중앙 +1.5 m 부터
STAIR_X0 = TILE / 2 + 1.5
VX, DURATION = 0.5, 12.0  # 전진 명령 m/s, 녹화 길이 s


def stairs(self):
    """Terrain.randomized_terrain 대체: 1 x 1 타일에 계단 코스를 만든다 (높이 단위 = vertical_scale)."""
    hs, vs = self.cfg.horizontal_scale, self.cfg.vertical_scale
    hf = np.zeros((self.length_per_env_pixels, self.width_per_env_pixels))
    self.boxes = []  # (x0, x1, top z) 타일 좌표
    x = STAIR_X0
    for k in range(1, N_STEP + 1):
        self.boxes.append((x, x + STEP_D, k * STEP_H))
        x += STEP_D
    self.boxes.append((x, x + PLATFORM, N_STEP * STEP_H))
    x += PLATFORM
    for k in range(N_STEP - 1, 0, -1):
        self.boxes.append((x, x + STEP_D, k * STEP_H))
        x += STEP_D
    for x0, x1, h in self.boxes:
        hf[int(round(x0 / hs)):int(round(x1 / hs)), :] = round(h / vs)
    self.height_field_raw[self.border:self.border + hf.shape[0], self.border:self.border + hf.shape[1]] = hf
    self.env_origins[0, 0] = [TILE / 2, TILE / 2, 0.0]


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = DURATION + 5
    t = env_cfg.terrain
    t.mesh_type, t.curriculum, t.selected = "trimesh", False, False
    t.num_rows = t.num_cols = 1
    t.terrain_length = t.terrain_width = TILE
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False
    Terrain.randomized_terrain = stairs
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    # 생성 위치를 타일 중앙(계단 1.5 m 앞), yaw 0 으로 고정
    env.root_states[0, :2] = torch.tensor([TILE / 2, TILE / 2], device=env.device)
    env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    env.gym.set_actor_root_state_tensor(env.sim, isaacgym.gymtorch.unwrap_tensor(env.root_states))

    names = env.dof_names
    order = [names.index(f"{l}_{j}_joint") for l in LEGS for j in ("hip", "thigh", "calf")]
    feet = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], f"{l}_foot") for l in LEGS]
    log = {k: [] for k in ("base_pos", "base_quat", "q", "contact")}
    obs = env.get_observations()
    for _ in range(int(DURATION / env.dt)):
        env.commands[:, :3] = torch.tensor([VX, 0.0, 0.0], device=env.device)
        env.compute_observations()
        obs = env.get_observations()
        obs, _, _, done, _ = env.step(policy(obs.detach()).detach())
        if bool(done[0]):
            print("episode 종료 (넘어짐 등) — 녹화 중단")
            break
        rs = env.root_states[0].cpu().numpy()
        log["base_pos"].append(rs[:3])
        log["base_quat"].append(rs[[6, 3, 4, 5]])
        log["q"].append(env.dof_pos[0].cpu().numpy()[order])
        log["contact"].append(env.contact_forces[0, feet, 2].cpu().numpy() > 1.0)
    # 타일 좌표 → world (trimesh 는 -border 만큼 옮겨져 있어 타일 원점 = world 원점)
    boxes = np.array(env.terrain.boxes)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "stairs_raw.npz")
    np.savez(out, dt=env.dt, boxes=boxes, **{k: np.array(v) for k, v in log.items()})
    p = np.array(log["base_pos"])
    print(f"-> {out}  frames={len(p)} dt={env.dt:.3f} x {p[0, 0]:.2f}->{p[-1, 0]:.2f} z {p[:, 2].min():.3f}~{p[:, 2].max():.3f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
