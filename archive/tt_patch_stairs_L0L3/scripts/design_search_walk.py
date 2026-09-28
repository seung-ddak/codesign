"""L0~L3 설계 탐색 — T1 보행 task, TT ctrl 정책 평가기 (remind.md §4.2/§4.3).

  L0  prior 무작위 샘플링        -- task 난이도 확인
  L1  CMA-ES                     -- black-box 최적화 기준선
  L2  결과 기반 prior 재학습     -- DiffuseBot 식 상위 k% fine-tune
  L3  residual guidance          -- 우리 방법 (D11)

네 단이 **같은 평가기·같은 예산·같은 성공 판정**을 공유하고 제안 방식만 다르다.
평가기는 TT 공개 ctrl 체크포인트를 그대로 쓰므로 **π_θ 학습이 없다**.

  python scripts/design_search_walk.py --stage l0 --budget 256
"""

import argparse
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np

from t2.eval.batched_ctrl import BatchedCtrlRollout
from t2.eval.design_search import STAGES


class CtrlObjective:
    """설계 (B, 13) -> (success, fell, distance).

    성공 = 제한 시간 내 베이스가 `--goal` 만큼 전진 (D10). 넘어지면 전진이
    멈추므로 낙상은 자연히 실패로 잡힌다.
    """

    def __init__(self, runner: BatchedCtrlRollout, goal: float, traj: int, seed: int):
        self.runner = runner
        self.goal = goal
        self.traj = traj
        self.seed = seed
        self.n_evals = 0
        self.rows = []

    def __call__(self, designs: np.ndarray):
        designs = np.clip(np.atleast_2d(designs), 0.0, 1.0)
        out = []
        for i in range(0, len(designs), self.runner.n_envs):
            chunk = designs[i : i + self.runner.n_envs]
            out.extend(self.runner.run(chunk, traj_idx=self.traj, seed=self.seed))
        self.n_evals += len(designs)
        self.rows.extend(out)
        dist = np.array([r["distance"] for r in out])
        success = (dist >= self.goal).astype(float)
        fell = np.array([0.0 if r["completed"] else 1.0 for r in out])
        return success, fell, dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=list(STAGES))
    p.add_argument("--ckpt", default="checkpoints/umi_on_legs_plus_plus/d7lra7j0/040.pt")
    p.add_argument("--env", default="umi_on_legs_plus_plus_from_choice")
    p.add_argument("--pkl", default="data/flat_walk_long.pkl")
    p.add_argument("--traj", type=int, default=1, help="명령 속도 (1 = 0.10 m/s)")
    p.add_argument("--goal", type=float, default=0.8, help="성공 판정 전진 거리 (m)")
    p.add_argument("--budget", type=int, default=256, help="설계 평가 수")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--episode-steps", type=int, default=600)
    p.add_argument("--inference-steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=10000000)
    p.add_argument("--dim", type=int, default=13)
    p.add_argument("--out", default="design_search_walk.jsonl")
    # L1
    p.add_argument("--popsize", type=int, default=32)
    # L2/L3 diffusion prior
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--diffusion-epochs", type=int, default=2000)
    p.add_argument("--diffusion-lr", type=float, default=1e-3)
    p.add_argument("--l2-pool", type=int, default=20000)
    p.add_argument("--l2-rounds", type=int, default=8)
    p.add_argument("--l2-top-frac", type=float, default=0.1)
    # L3 residual guidance
    p.add_argument("--l3-rounds", type=int, default=8)
    p.add_argument("--l3-lr", type=float, default=3e-4)
    p.add_argument("--l3-std", type=float, default=0.05)
    p.add_argument("--l3-scale", type=float, default=0.1)
    p.add_argument("--l3-guided-steps", type=int, default=10)
    args = p.parse_args()

    t0 = time.time()
    runner = BatchedCtrlRollout(
        ckpt=args.ckpt,
        env_name=args.env,
        pkl=args.pkl,
        n_envs=args.n_envs,
        inference_steps=args.inference_steps,
        episode_steps=args.episode_steps,
    )
    print(f"평가기 준비 {time.time() - t0:.1f}s", flush=True)

    obj = CtrlObjective(runner, goal=args.goal, traj=args.traj, seed=args.seed)
    result = STAGES[args.stage](obj, args, np.random.default_rng(args.seed))
    result.update(
        checkpoint=args.ckpt, goal=args.goal, traj=args.traj,
        episode_steps=args.episode_steps, inference_steps=args.inference_steps,
        seed=args.seed,
    )
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(
        {k: v for k, v in result.items()
         if k not in ("history", "rounds", "best_design")},
        indent=2, ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
