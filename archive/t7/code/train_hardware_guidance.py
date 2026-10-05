"""T7: frozen-pi_theta locomotion return으로 TT hardware denoiser를 DDPO 학습한다.

  python stage2/code/train_hardware_guidance.py --pi-theta stage2/runs/T5/try1 --out stage2/runs/T7/try1

각 update는 stochastic DDIM trajectory를 기록하고, frozen pi_theta의 7개 명령 평균 return을
terminal reward로 사용해 clipped DDPO-IS로 TT 자체를 갱신한다. Invalid design은 배치의
최저 valid return을 받는다. history.jsonl과 psi_XXXX.pt를 저장한다.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from eval_locomotion_policy import LAUNCH_DIR  # noqa: E402  (chdirs into TT)
from train_locomotion_policy import TT  # noqa: E402

import numpy as np  # noqa: E402

from ddpo_hardware import DesignLearner  # noqa: E402
from design_scorer import DesignScorer, log_memory, memory_gib  # noqa: E402

PRIOR = TT / "checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt"  # D28 fixed TT prior
BANK = TT / "umi_on_legs_plus_plus_robotokens"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pi-theta", type=Path, required=True, help="T5 run (frozen policy)")
    p.add_argument("--ckpt", help="pi_theta checkpoint step dir (default: latest)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--global-batch", "--batch", dest="global_batch", type=int, default=32)
    p.add_argument("--generation-microbatch", type=int, default=4)
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--ddpo-minibatch", type=int, default=32)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--ddpo-epochs", type=int, default=4)
    p.add_argument("--psi-lr", type=float, default=1e-6)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if min(args.updates, args.global_batch, args.generation_microbatch, args.slots,
           args.ddpo_minibatch, args.save_every, args.ddpo_epochs) < 1:
        p.error("batch sizes, updates, slots, save-every and ddpo-epochs must be positive")
    if args.psi_lr <= 0 or not 0 < args.clip_range < 1:
        p.error("psi-lr must be positive and clip-range must be between 0 and 1")
    run = (LAUNCH_DIR / args.pi_theta).resolve()
    out = (LAUNCH_DIR / args.out).resolve()
    # T7 시작 실패 진단:
    # - FileExistsError: --out은 기존 run과 겹치지 않는 새 폴더를 지정한다.
    # - checkpoints/designs.json 오류: --pi-theta가 완료된 T5 run을 가리키는지 확인한다.
    # - CUDA OOM: 생성은 --generation-microbatch, 평가는 --slots를 낮춘다.
    out.mkdir(parents=True, exist_ok=False)
    ckpt = run / "checkpoints" / (args.ckpt or sorted(os.listdir(run / "checkpoints"))[-1])
    (out / "config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "pi_theta_ckpt": str(ckpt),
         "prior": str(PRIOR), "bank": str(BANK)}, indent=1))

    t = time.time()
    scorer = DesignScorer(json.loads((run / "designs.json").read_text()), ckpt, args.slots)
    learner = DesignLearner(
        TT, PRIOR, BANK, args.global_batch, "cuda", args.seed,
        ddpo_epochs=args.ddpo_epochs, lr=args.psi_lr, clip_range=args.clip_range,
        generation_microbatch=args.generation_microbatch,
        ddpo_minibatch=args.ddpo_minibatch, trajectory_dir=out)
    print(f"setup {time.time() - t:.0f}s", flush=True)
    log_memory("after_setup")
    log = (out / "history.jsonl").open("a")
    for u in range(args.updates):
        t = time.time()
        designs, transitions = learner.generate(seed=args.seed * 100000 + u)
        log_memory("after_generation", len(scorer.group_cache))
        valid = [i for i, d in enumerate(designs) if "invalid" not in d]
        # valid=0이 반복되면 보행 평가기가 아니라 DesignLearner.generate의
        # RoboToken bank 매칭과 invalid 사유를 먼저 확인한다.
        returns = np.full(len(designs), np.nan)
        if valid:
            returns[valid] = scorer.score([(designs[i]["choices"], designs[i]["seed"]) for i in valid],
                                          seed=u)
        log_memory("after_scoring", len(scorer.group_cache))
        worst = np.nanmin(returns) if valid else 0.
        rewards = np.where(np.isnan(returns), worst, returns)
        stats = learner.update(transitions, rewards.tolist())
        log_memory("after_ddpo", len(scorer.group_cache))
        rss_gib, peak_rss_gib = memory_gib()
        # 학습 실패 진단:
        # - psi_updated=False: 형상별 return이 모두 같아 TT에 순위 신호가 없는 상태다.
        # - return이 NaN/비정상: DesignScorer의 rollout, 명령별 생존 시간과 체크포인트를 확인한다.
        # - gradient_norm이 계속 0에 가까움: DDPO loss·KL·clip fraction을 확인한다.
        row = {"update": u, "wall_s": round(time.time() - t, 1), "valid": len(valid),
               "rss_gib": round(rss_gib, 2), "peak_rss_gib": round(peak_rss_gib, 2),
               "compiled_groups": len(scorer.group_cache),
               "return_mean": float(np.nanmean(returns)) if valid else None,
               "return_max": float(np.nanmax(returns)) if valid else None,
               "designs": [{**d, "return": None if np.isnan(r) else float(r)}
                           for d, r in zip(designs, returns)], **stats}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"update {u} valid {len(valid)}/{len(designs)} return mean {row['return_mean']} "
              f"max {row['return_max']} {row['wall_s']}s", flush=True)
        if (u + 1) % args.save_every == 0 or u + 1 == args.updates:
            learner.save(out / f"psi_{u + 1:04d}.pt")


if __name__ == "__main__":
    main()
