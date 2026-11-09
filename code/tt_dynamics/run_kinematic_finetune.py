"""Run a matched 2,000-update kinematic TT fine-tune on a V100 machine.

Builds a new train_tt.py command directly from the source checkpoint args,
preserving the original data order, split seed, architecture and normalization.
The runner never edits the original training directory or checkpoint.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--kin-xy-weight", type=float, default=0.002)
    p.add_argument("--kin-joint-weight", type=float, default=0.01)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume the NEW fine-tuning run; do not load source weights again")
    args = p.parse_args()

    ckpt = args.ckpt / "checkpoint.pt" if args.ckpt.is_dir() else args.ckpt
    if not ckpt.is_file():
        p.error(f"Checkpoint does not exist: {ckpt}")
    if args.resume:
        if not (args.out / "training_state.pt").is_file():
            p.error(f"No resumable fine-tuning state in: {args.out}")
    elif args.out.exists():
        p.error(f"Output directory exists (do not overwrite): {args.out}")
    if args.steps < 2 or not 0 <= args.warmup_steps < args.steps:
        p.error("Require steps >= 2 and 0 <= warmup_steps < steps")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    original = ck["args"]
    if original.get("pose", "abs") != "abs" or 1 not in original["modes"]:
        p.error("Source must be absolute-pose TT trained with Mode 1 (E+s0+A)")
    data = [str(f) for f in original["data"]]
    missing = [f for f in data if not Path(f).is_file()]
    if missing:
        p.error(f"Missing original source data ({len(missing)}): {missing[:3]}")
    if not args.dry_run and not torch.cuda.is_available():
        p.error("Training requires a CUDA GPU; rerun on the V100 machine")

    train = Path(__file__).with_name("train_tt.py")
    command = [sys.executable, str(train), "--out", str(args.out)]
    if args.resume:
        command.append("--resume")
    else:
        command.extend(["--init-from", str(ckpt)])
    command.extend(["--data", *data])

    def scalar(name, value):
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])

    def sequence(name, value):
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), *map(str, value)])

    # Transfer exact conditioning and data-split contract from the source.
    for name in ("horizon", "batch", "d_model", "layers", "diff_steps",
                 "norm", "clip", "val_frac", "seed", "pose",
                 "loss_balance", "contact_range", "swap_fraction",
                 "split_key", "e_dropout", "delta_weight", "pair_batch",
                 "sampler", "stride", "eval_episodes"):
        scalar(name, original.get(name))
    for name in ("modes", "mode_weights", "loss_group_weights"):
        sequence(name, original.get(name))
    if original.get("film", False):
        command.append("--film")
    for name, value in (
        ("steps", args.steps), ("lr", args.lr),
        ("warmup_steps", args.warmup_steps),
        ("ema_power", original.get("ema_power", 0.75)),
        ("checkpoint_every", args.steps), ("val_every", min(200, args.steps)),
        ("kin_xy_weight", args.kin_xy_weight),
        ("kin_joint_weight", args.kin_joint_weight),
    ):
        scalar(name, value)

    print("Using source checkpoint:", ckpt, flush=True)
    print("Source dataset files:", len(data), flush=True)
    print("New run:", args.out, flush=True)
    print("Command:\n" + shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
