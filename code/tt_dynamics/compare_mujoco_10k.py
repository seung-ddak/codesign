"""Compare old small-dataset TT vs new 10k MuJoCo TT under IDENTICAL CPU tasks.

Both checkpoints must use compatible mode 2/4, horizon and diffusion settings;
read the emitted protocol before interpreting any change as a data-only effect.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "code/tt_dynamics/generate_task_mujoco.py"


def checkpoint_config(path):
    import torch
    model = torch.load(path, map_location="cpu", weights_only=False)
    a = model["args"]
    return {k:a.get(k) for k in ("horizon","diff_steps","d_model","layers","film","norm",
                                  "clip","modes","pose","steps","data")}


def summarize(summary):
    rows = []
    for scale, run in summary["results"].items():
        candidates = run["rows"]
        valid = [r for r in candidates if r.get("initial_valid")]
        survived = [r for r in valid if not r.get("fell", True)]
        scored = [r for r in valid if r.get("reward") is not None]
        rows.append(dict(scale=scale, candidates=len(candidates),
                         valid_start=len(valid), survived_10s=len(survived),
                         reward_mean=(sum(r["reward"] for r in scored)/len(scored) if scored else None),
                         mean_progress=(sum(r["task_progress"] for r in scored)/len(scored) if scored else None),
                         progress_unit=scored[0]["progress_unit"] if scored else None))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-ckpt", type=Path, required=True)
    p.add_argument("--old-ckpt", type=Path, default=ROOT/"runs/pipeline_v1/fast_mode24_v16_twin/tt_joint/checkpoint.pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a=p.parse_args()
    if a.out.exists():
        p.error("Use a new output directory to preserve prior comparisons")
    new=a.new_ckpt/"checkpoint.pt" if a.new_ckpt.is_dir() else a.new_ckpt
    old=a.old_ckpt/"checkpoint.pt" if a.old_ckpt.is_dir() else a.old_ckpt
    for f in (new,old):
        if not f.is_file():raise FileNotFoundError(f)
    configs={"new_10k":checkpoint_config(new),"old_small":checkpoint_config(old)}
    # The comparison is controlled for the sampler, physics, task, seed and count.
    # A different number of optimizer updates/source-data-physics remains a confound.
    compatibility={k:configs["new_10k"][k]==configs["old_small"][k]
                   for k in ("horizon","diff_steps","d_model","layers","film","norm","clip","pose")}
    if not all(compatibility.values()):
        raise ValueError(f"Architecture / diffusion mismatch: {compatibility}")
    a.out.mkdir(parents=True)
    results=[]
    for task in ("zigzag","circle","straight"):
        for label,ck in (("old_small",old),("new_10k",new)):
            folder=a.out/f"{task}_{label}"
            subprocess.run([sys.executable,"-u",str(GEN),"--ckpt",str(ck),
                            "--out",str(folder),"--task",task,
                            "--candidates",str(a.candidates),
                            "--scales","0","1","--seed",str(a.seed)],
                           cwd=ROOT,check=True)
            summary=json.loads((folder/"summary.json").read_text())
            results.extend(dict(task=task,model=label,**row) for row in summarize(summary))
            print(task,label,results[-2:],flush=True)
    protocol=dict(task_set=["zigzag","circle","straight"],physics="MuJoCo CPU",
                  seconds=10,candidates=a.candidates,seed=a.seed,
                  old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
                  new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
                  model_settings=configs,compatibility=compatibility,
                  warning="PD vs PPO, CPU vs MJX dataset physics, and unequal optimizer budgets may confound attribution to dataset size",
                  results=results)
    (a.out/"comparison.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print("Saved:",a.out/"comparison.json")


if __name__=="__main__":
    main()
