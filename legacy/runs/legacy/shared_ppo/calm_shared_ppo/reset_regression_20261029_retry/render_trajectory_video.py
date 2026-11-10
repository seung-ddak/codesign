"""Render a small MP4 chart animation from one recorded shared-PPO rollout."""
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ROLLOUTS = ROOT / "rollouts" / "completed"
OUTPUT = ROOT / "trajectory_preview.mp4"
GIF_OUTPUT = ROOT / "trajectory_preview.gif"


def main() -> None:
    files = sorted(ROLLOUTS.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No completed rollout in {ROLLOUTS}")
    with np.load(files[0]) as data:
        qpos = np.asarray(data["qpos"], dtype=float)
        action = np.asarray(data["action"], dtype=float)
        ctrl = np.asarray(data["ctrl"], dtype=float)
        e = np.asarray(data["E"], dtype=float)
        dt = float(data["ctrl_dt"])
    t = np.arange(len(qpos)) * dt
    height = qpos[:, 2]
    joint_pos = qpos[:, 7:]
    action_norm = np.linalg.norm(action, axis=1)
    ctrl_norm = np.linalg.norm(ctrl, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    fig.patch.set_facecolor("#f4f8fa")
    fig.suptitle("Recorded PPO transition · trajectory data visualization",
                 fontsize=16, fontweight="bold", color="#16384a")
    ax = axes[0]
    ax.set_facecolor("white")
    ax.plot(t, height, "o-", color="#087f72", linewidth=2.2, label="base z (m)")
    ax.set_ylabel("Base height (m)")
    ax.set_xlabel("Recorded control step time (s)")
    ax.grid(alpha=.22)
    marker, = ax.plot([], [], "o", color="#e76f51", markersize=9, zorder=5)
    ax.legend(loc="best", frameon=False)

    ax2 = axes[1]
    ax2.set_facecolor("white")
    ax2.plot(t, action_norm, "o-", color="#277da1", linewidth=2,
             label="policy action L2")
    ax2.plot(t, ctrl_norm, "s--", color="#f28e2b", linewidth=1.8,
             label="applied ctrl L2")
    ax2.set_ylabel("Vector L2 norm")
    ax2.set_xlabel("Recorded control step time (s)")
    ax2.grid(alpha=.22)
    marker2, = ax2.plot([], [], "o", color="#e76f51", markersize=9, zorder=5)
    ax2.legend(loc="best", frameon=False)
    fig.text(.5, .005,
             f"Schematic chart animation, not a robot render  ·  30 Hz  ·  "
             f"{len(qpos)} transitions  ·  E[3:5]={e[3]:.1f},{e[4]:.1f}  ·  "
             f"joint q range=[{joint_pos.min():.2f}, {joint_pos.max():.2f}] rad",
             ha="center", fontsize=9, color="#526b79")

    # Interpolate marker motion for readability; the underlying measurements remain discrete.
    frames = []
    fps = 12
    frame_count = max(48, len(t) * 12)
    for frame in range(frame_count):
        idx = min(len(t) - 1, int(frame / max(frame_count - 1, 1) * len(t)))
        marker.set_data([t[idx]], [height[idx]])
        marker2.set_data([t[idx]], [action_norm[idx]])
        canvas = fig.canvas
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
    plt.close(fig)
    imageio.mimsave(OUTPUT, frames, fps=fps, codec="libx264", quality=8,
                    macro_block_size=16)
    # Animated GIF is a browser-friendly inline fallback for file:// report viewers.
    gif_frames = [
        Image.fromarray(frame)
        .resize((756, 528))
        .convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        for frame in frames[::2]
    ]
    gif_frames[0].save(
        GIF_OUTPUT, save_all=True, append_images=gif_frames[1:],
        duration=1000 / fps * 2, loop=0, optimize=True
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes) and "
          f"{GIF_OUTPUT} ({GIF_OUTPUT.stat().st_size} bytes), source={files[0]}")


if __name__ == "__main__":
    main()
