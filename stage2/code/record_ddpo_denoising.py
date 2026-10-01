"""Replay one DDPO noise trajectory and record prior-vs-checkpoint RoboToken heatmaps."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
TT = ROOT.parent / "transformer-transformer"
sys.path[:0] = [str(ROOT / "stage2/code"), str(TT)]

import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from ddpo_hardware import DesignLearner  # noqa: E402


PRIOR = TT / "checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt"
BANK = TT / "umi_on_legs_plus_plus_robotokens"


def _matrix(value):
    value = np.asarray(value)
    return value.reshape(1, -1) if value.ndim < 2 else value.reshape(value.shape[0], -1)


def replay(checkpoint, seed, work):
    learner = DesignLearner(
        TT, PRIOR, BANK, batch_size=1, device="cuda", seed=seed,
        generation_microbatch=1, trajectory_dir=work,
    )
    if checkpoint:
        learner.load(checkpoint)
    designs, trajectory = learner.generate(seed)
    root = zarr.open_group(trajectory, mode="r")
    names = list(root.attrs["names"])
    states = {name: np.asarray(root[f"states/{name}"][0]) for name in names}
    timesteps = np.asarray(root["timesteps"])
    del learner
    torch.cuda.empty_cache()
    return names, states, timesteps, designs[0], trajectory


def record(left, right, output, fps):
    names = left[0]
    if names != right[0]:
        raise ValueError("RoboToken modalities differ between runs")
    frames = min(min(len(x) for x in left[1].values()), min(len(x) for x in right[1].values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, macro_block_size=1) as writer:
        for step in range(frames):
            fig, axes = plt.subplots(len(names), 2, figsize=(12.8, max(4.8, 1.65 * len(names))),
                                     squeeze=False, layout="constrained")
            timestep = "noise" if step == 0 else str(left[2][min(step - 1, len(left[2]) - 1)])
            fig.suptitle(f"try2 DDPO denoising replay · state {step}/{frames - 1} · t={timestep}",
                         fontsize=14)
            for row, name in enumerate(names):
                for col, (label, run) in enumerate((("pretrained prior", left), ("try2 psi_0010", right))):
                    axes[row, col].imshow(_matrix(run[1][name][step]), aspect="auto",
                                          cmap="coolwarm", vmin=-3.0, vmax=3.0)
                    axes[row, col].set_title(f"{label} · {name}", fontsize=9)
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
            fig.text(.5, .005,
                     f"final decode: prior={left[3]}   |   try2={right[3]}\n"
                     "Raw RoboToken values; final deterministic scheduler step is summarized by decode status.",
                     ha="center", fontsize=9)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            height, width = frame.shape[:2]
            writer.append_data(frame[:height - height % 2, :width - width % 2])
            plt.close(fig)
            print(f"frame {step + 1}/{frames}", flush=True)


def self_check():
    assert _matrix(np.arange(6)).shape == (1, 6)
    assert _matrix(np.zeros((2, 3, 4))).shape == (2, 12)
    print("record_ddpo_denoising self-check: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "stage2/runs/T7/try2")
    parser.add_argument("--checkpoint", default="psi_0010.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    run = args.run.resolve()
    checkpoint = run / args.checkpoint
    output = (args.output or run / "denoising/prior_vs_psi0010.mp4").resolve()
    if not checkpoint.is_file():
        parser.error(f"missing checkpoint: {checkpoint}")

    work = Path(tempfile.mkdtemp(prefix="try2_denoising_", dir=run))
    try:
        prior = replay(None, args.seed, work)
        trained = replay(checkpoint, args.seed, work)
        record(prior, trained, output, args.fps)
        metadata = {
            "video": output.name,
            "seed": args.seed,
            "left": "pretrained prior",
            "right": str(checkpoint),
            "prior_final": prior[3],
            "trained_final": trained[3],
            "note": "Raw RoboToken trajectory; simulator is not run at intermediate steps.",
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        print(f"wrote {output}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
