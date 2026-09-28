"""T5 gate and T6 evaluator check (D50, D52, D53).

  t5: python stage2/code/t6_evaluator_check.py t5 --run stage2/runs/T5_try1
      exit 0 if the run's latest eval meets the T5 criteria, else 3.
  t6: python stage2/code/t6_evaluator_check.py t6 --run stage2/runs/T5_try1 --out stage2/runs/T6_try1
      freezes the T5 checkpoint, samples one bank seed per choice that training never used, scores
      them twice (different reset seeds, no domain randomisation, the fixed EVAL_COMMANDS). Passes if
      (1) the two design rankings agree (Spearman >= 0.8), (2) the unseen designs keep >= 90 % of the
      T5 run's own eval mean on each T5 criterion (generalisation gap, D53) and (3) the choice ranking
      by unseen return agrees with the ranking by the T5 eval's per-choice mean return (Spearman >= 0.8,
      D53). The absolute T5 criteria are reported but belong to the T5 gate. Writes <out>/t6_result.json;
      exit 0 on pass, else 3.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_locomotion_policy import LAUNCH_DIR, evaluate  # noqa: E402  (chdirs into TT)

import numpy as np  # noqa: E402

from multi_design_env import structure_key  # noqa: E402
from train_locomotion_policy import BANK_SEEDS, TT  # noqa: E402

T5_CRITERIA = {"cmd_progress_ratio": .8, "cmd_turn_ratio": .5, "survived": .9}  # D50
T6_MIN_SPEARMAN = .8  # D52
T6_MIN_GAP_RATIO = .9  # D53: unseen mean >= 0.9 x the T5 eval (training seeds) mean


def t5_check(summary):
    return {k: {"value": summary[k], "min": v, "pass": summary[k] >= v} for k, v in T5_CRITERIA.items()}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def latest_eval(run):
    return run / sorted(p for p in os.listdir(run) if p.startswith("eval_"))[-1]


def unseen_designs(train_designs, seed):
    used = {(tuple(c), s) for g in train_designs for c, s in g}
    choices = sorted(json.loads((TT / "choice_to_ckpt_path.json").read_text()))
    rng = np.random.RandomState(seed)
    groups = {}
    for key in choices:
        c = [int(x) for x in key]
        s = int(rng.randint(BANK_SEEDS))
        while (tuple(c), s) in used:
            s = int(rng.randint(BANK_SEEDS))
        groups.setdefault(structure_key(c), []).append((c, s))
    return list(groups.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("t5", "t6"))
    p.add_argument("--run", type=Path, required=True, help="T5 run")
    p.add_argument("--out", type=Path, help="T6 output folder")
    p.add_argument("--ckpt", help="checkpoint step dir name (default: latest)")
    p.add_argument("--seed", type=int, default=0, help="unseen-seed sampling")
    args = p.parse_args()
    run = (LAUNCH_DIR / args.run).resolve()
    if args.stage == "t5":
        check = t5_check(json.loads(latest_eval(run).read_text())["summary"])
        print(json.dumps(check, indent=1))
        sys.exit(0 if all(v["pass"] for v in check.values()) else 3)

    out = (LAUNCH_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = run / "checkpoints" / (args.ckpt or sorted(os.listdir(run / "checkpoints"))[-1])
    groups = unseen_designs(json.loads((run / "designs.json").read_text()), args.seed)
    (out / "designs.json").write_text(json.dumps(groups))
    reps = []
    for rep, reset_seed in enumerate((1, 2), 1):
        evaluate(out, ckpt, reset_seed)
        path = out / f"eval_rep{rep}.json"
        (out / f"eval_{ckpt.name}.json").rename(path)
        reps.append(json.loads(path.read_text()))
    ret = [{(tuple(d["choices"]), d["seed"]): d["return"] for d in r["designs"]} for r in reps]
    keys = sorted(ret[0])
    rho = spearman([ret[0][k] for k in keys], [ret[1][k] for k in keys])
    seen = json.loads(latest_eval(run).read_text())
    gap = {k: {"value": reps[0]["summary"][k], "seen": seen["summary"][k],
               "min": T6_MIN_GAP_RATIO * seen["summary"][k],
               "pass": reps[0]["summary"][k] >= T6_MIN_GAP_RATIO * seen["summary"][k]} for k in T5_CRITERIA}
    seen_choice = {}
    for d in seen["designs"]:
        seen_choice.setdefault(tuple(d["choices"]), []).append(d["return"])
    choices = sorted({k[0] for k in keys})  # one unseen seed per choice
    unseen_choice = {k[0]: (ret[0][k] + ret[1][k]) / 2 for k in keys}
    rho_choice = spearman([np.mean(seen_choice[c]) for c in choices], [unseen_choice[c] for c in choices])
    result = {"checkpoint": str(ckpt), "designs": len(keys), "spearman": rho,
              "spearman_pass": rho >= T6_MIN_SPEARMAN,
              "generalisation_gap": gap,
              "choice_spearman": rho_choice, "choice_spearman_pass": rho_choice >= T6_MIN_SPEARMAN,
              "unseen_t5_criteria": t5_check(reps[0]["summary"]),
              "return_spread": float(np.std([ret[0][k] for k in keys])),
              "repeat_abs_diff": float(np.mean([abs(ret[0][k] - ret[1][k]) for k in keys]))}
    result["pass"] = (result["spearman_pass"] and result["choice_spearman_pass"]
                      and all(v["pass"] for v in gap.values()))
    (out / "t6_result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["pass"] else 3)


if __name__ == "__main__":
    main()
