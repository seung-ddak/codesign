"""T7/T8: learn pi_phi (TT residual guidance) from frozen-pi_theta locomotion returns (D38, D39, D52).

  python stage2/code/train_hardware_guidance.py --pi-theta stage2/runs/T5_try1 --out stage2/runs/T7_try1

Each update: fixed TT prior + pi_phi generate --batch designs (reference tokens masked, nearest
bank (choice, seed), D49 arm/rail removal at build) -> DesignScorer rolls each out with the frozen
pi_theta over the fixed EVAL_COMMANDS (no domain randomisation) -> mean episode return is the
terminal reward -> DesignLearner.update (critic + actor on the guidance transitions). Invalid
designs get the batch's worst valid return. Writes history.jsonl and phi_XXXX.pt.
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
sys.path.insert(0, str(ROOT / "stage3_coevolution"))
from eval_locomotion_policy import LAUNCH_DIR  # noqa: E402  (chdirs into TT)
from train_locomotion_policy import TT  # noqa: E402

import numpy as np  # noqa: E402

from coevolution_guidance import DesignLearner  # noqa: E402
from design_scorer import DesignScorer  # noqa: E402

PRIOR = TT / "checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt"  # D28 fixed TT prior
BANK = TT / "umi_on_legs_plus_plus_robotokens"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pi-theta", type=Path, required=True, help="T5 run (frozen policy)")
    p.add_argument("--ckpt", help="pi_theta checkpoint step dir (default: latest)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run = (LAUNCH_DIR / args.pi_theta).resolve()
    out = (LAUNCH_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ckpt = run / "checkpoints" / (args.ckpt or sorted(os.listdir(run / "checkpoints"))[-1])
    (out / "config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "pi_theta_ckpt": str(ckpt),
         "prior": str(PRIOR), "bank": str(BANK)}, indent=1))

    t = time.time()
    scorer = DesignScorer(json.loads((run / "designs.json").read_text()), ckpt, args.slots)
    learner = DesignLearner(TT, PRIOR, BANK, args.batch, "cuda", args.seed)
    print(f"setup {time.time() - t:.0f}s", flush=True)
    log = (out / "history.jsonl").open("a")
    for u in range(args.updates):
        t = time.time()
        designs, transitions = learner.generate(seed=args.seed * 100000 + u)
        valid = [i for i, d in enumerate(designs) if "invalid" not in d]
        returns = np.full(len(designs), np.nan)
        if valid:
            returns[valid] = scorer.score([(designs[i]["choices"], designs[i]["seed"]) for i in valid],
                                          seed=u)
        worst = np.nanmin(returns) if valid else 0.
        rewards = np.where(np.isnan(returns), worst, returns)
        stats = learner.update(transitions, rewards.tolist())
        row = {"update": u, "wall_s": round(time.time() - t, 1), "valid": len(valid),
               "return_mean": float(np.nanmean(returns)) if valid else None,
               "return_max": float(np.nanmax(returns)) if valid else None,
               "designs": [{**d, "return": None if np.isnan(r) else float(r)}
                           for d, r in zip(designs, returns)], **stats}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"update {u} valid {len(valid)}/{len(designs)} return mean {row['return_mean']} "
              f"max {row['return_max']} {row['wall_s']}s", flush=True)
        if (u + 1) % args.save_every == 0 or u + 1 == args.updates:
            learner.save(out / f"phi_{u + 1:04d}.pt")


if __name__ == "__main__":
    main()
