"""T5 gate and T6 evaluator check (D50, D52, D56).

  t5: python stage2/code/t6_evaluator_check.py t5 --run stage2/runs/T5/try1
      exit 0 if the run's latest eval meets the T5 criteria, else 3.
  t6: python stage2/code/t6_evaluator_check.py t6 --run stage2/runs/T5/try1 --out stage2/runs/T6/try1
      freezes the T5 checkpoint, samples one bank seed per choice that training never used, scores
      them twice (different reset seeds, no domain randomisation, the fixed EVAL_COMMANDS). Passes if
      (1) the two design rankings agree (Spearman >= 0.8) and (2) the mean of both unseen-design
      evaluations keeps >= 90 % of the T5 run's own eval mean on each T5 criterion (D56).
      Per-choice rank correlation remains diagnostic because T7 scores concrete choice+seed designs,
      not choice averages. Writes <out>/t6_result.json; exit 0 on pass, else 3.
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
from train_locomotion_policy import BANK_SEEDS  # noqa: E402

T5_CRITERIA = {"cmd_progress_ratio": .8, "cmd_turn_ratio": .5, "survived": .9}  # D50
T6_MIN_SPEARMAN = .8  # D52
T6_MIN_GAP_RATIO = .9  # D56: two-repeat unseen mean >= 0.9 x T5 seen mean
# T6 필수 기준은 반복 순위 안정성과 unseen seed 평균 일반화다. choice_spearman은 같은
# discrete choice 내부의 continuous variation을 보는 진단값이며 통과 여부에 쓰지 않는다.


def t5_check(summary):
    return {k: {"value": summary[k], "min": v, "pass": summary[k] >= v} for k, v in T5_CRITERIA.items()}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def latest_eval(run):
    return run / sorted(p for p in os.listdir(run) if p.startswith("eval_"))[-1]


def unseen_designs(train_designs, seed):
    # T6 진단: choice 누락·중복 또는 T5 학습 seed 재사용 문제만 여기서 확인한다.
    used = {(tuple(c), s) for g in train_designs for c, s in g}
    choices = sorted({tuple(c) for g in train_designs for c, _ in g})
    rng = np.random.RandomState(seed)
    groups = {}
    for key in choices:
        c = list(key)
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
    # T6 진단: rho가 낮으면 반복 채점이 불안정한 것이다. 정책이나 기준을 바꾸기 전에
    # evaluate()의 reset·종료·명령 할당을 확인한다.
    seen = json.loads(latest_eval(run).read_text())
    unseen = {k: float(np.mean([r["summary"][k] for r in reps])) for k in T5_CRITERIA}
    gap = {k: {"value": unseen[k], "seen": seen["summary"][k],
               "min": T6_MIN_GAP_RATIO * seen["summary"][k],
               "pass": unseen[k] >= T6_MIN_GAP_RATIO * seen["summary"][k]} for k in T5_CRITERIA}
    # T6 진단: gap 실패는 T5 일반화 실패다. train_locomotion_policy.py의 학습 범위와
    # locomotion_env.py의 보상을 재검토한다. T6 자체에는 학습 과정이 없다.
    seen_choice = {}
    for d in seen["designs"]:
        seen_choice.setdefault(tuple(d["choices"]), []).append(d["return"])
    choices = sorted({k[0] for k in keys})  # one unseen seed per choice
    unseen_choice = {k[0]: (ret[0][k] + ret[1][k]) / 2 for k in keys}
    rho_choice = spearman([np.mean(seen_choice[c]) for c in choices], [unseen_choice[c] for c in choices])
    # T6 진단: choice당 unseen seed가 하나뿐이라 choice 평균의 재현성을 판정할 수 없다.
    # 값은 continuous morphology variation 분석용으로만 기록한다.
    result = {"checkpoint": str(ckpt), "designs": len(keys), "spearman": rho,
              "spearman_pass": rho >= T6_MIN_SPEARMAN,
              "generalisation_gap": gap,
              "choice_spearman": rho_choice, "choice_spearman_gate": False,
              "unseen_t5_criteria": t5_check(unseen),
              "return_spread": float(np.std([ret[0][k] for k in keys])),
              "repeat_abs_diff": float(np.mean([abs(ret[0][k] - ret[1][k]) for k in keys]))}
    result["pass"] = result["spearman_pass"] and all(v["pass"] for v in gap.values())
    (out / "t6_result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["pass"] else 3)


if __name__ == "__main__":
    main()
