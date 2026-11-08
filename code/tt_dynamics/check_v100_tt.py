"""Check whether new V100 TT training FINISHED; never equate that with convergence."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("runs/v100_mujoco_10k"))
    p.add_argument("--steps", type=int, default=25000)
    a = p.parse_args()
    run = a.run
    audit = json.loads((run/"audit.json").read_text()) if (run/"audit.json").exists() else {}
    tt = run/"tt/h32_fast"
    progress = tt/"progress.jsonl"
    config = json.loads((tt/"config.json").read_text()) if (tt/"config.json").exists() else {}
    split = json.loads((tt/"split_audit.json").read_text()) if (tt/"split_audit.json").exists() else {}
    last = {}
    if progress.is_file():
        with progress.open() as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
    checkpoint = tt/"checkpoint.pt"
    loaded = False
    ck_steps = ck_modes = ck_data = None
    if checkpoint.is_file():
        try:
            ck = torch.load(checkpoint,map_location="cpu",weights_only=False)
            ck_modes = ck["args"].get("modes")
            ck_data = ck["args"].get("data",[])
            ck_steps = ck["args"].get("steps")
            loaded = "net" in ck and "mu" in ck and "sd" in ck
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"Checkpoint is unreadable: {exc}",file=sys.stderr)
    data_match = bool(ck_data) and all(
        "v100_mujoco_10k/rollouts/" in str(path) for path in ck_data)
    val = last.get("val_loss")
    finite_val = isinstance(val, (int,float)) and math.isfinite(val)
    done = (audit.get("complete") is True
            and audit.get("source_coverage") == 10000
            and audit.get("training_designs") == 200
            and audit.get("episodes") == 40000
            and not audit.get("invalid_files")
            and loaded and data_match and ck_modes == [1,2,4]
            and ck_steps == a.steps
            and last.get("step") == a.steps
            and finite_val
            and split.get("independent_source_groups",0) >= 2
            and split.get("train_source_groups",0)>0
            and split.get("validation_source_groups",0)>0)
    summary = dict(training_finished=bool(done),
                   scientific_convergence_confirmed=False,
                   warning="Training completion is NOT tracking success, learned dynamics accuracy, or task success.",
                   source_audit_complete=audit.get("complete"),
                   original_sources=audit.get("source_coverage"),
                   training_designs=audit.get("training_designs"),
                   valid_physics_transitions=audit.get("valid_transitions"),
                   checkpoint_exists=checkpoint.exists(), checkpoint_loadable=loaded,
                   checkpoint_matches_new_rollouts=data_match,
                   checkpoint_modes=ck_modes, checkpoint_steps=ck_steps,
                   final_progress_step=last.get("step"),
                   final_val_loss=val, source_split_groups=split.get("independent_source_groups"),
                   expected_steps=a.steps)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not done:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
