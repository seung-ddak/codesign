"""Render recorded CPU qpos: held-out candidate 12, guidance vs RL, no resimulation."""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CODESIGN = ROOT.parents[2]
sys.path.insert(0, str(CODESIGN / 'code/tt_dynamics'))
from train_task_rl import TaskEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, default=ROOT)
    root = parser.parse_args().experiment_root.resolve()
    videos = root / 'videos'
    videos.mkdir(exist_ok=True)
    for task, folder in [('zigzag', 'zigzag_seed0_retry'), ('circle', 'circle_seed0')]:
        run = root / folder
        if not run.exists() and task == 'zigzag':
            run = root / 'zigzag_seed0'
        summary = json.loads((run / 'summary.json').read_text())
        source = dict(np.load(summary['protocol']['source']))
        index = 12  # Fixed first held-out morphology, not selected by performance.
        env = TaskEnv(source, index, task)
        trajectories = [np.load(run / f'{method}_cpu.npz')[f'qpos_{index}']
                        for method in ('guidance', 'rl')]
        assert all(q.shape[1] == env.model.nq and np.isfinite(q).all() for q in trajectories)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, .25]
        camera.distance, camera.azimuth, camera.elevation = 2., 135., -25.
        option = mujoco.MjvOption()
        option.geomgroup[:] = 1  # TT robot geometry is in group 3, hidden by default.
        data = mujoco.MjData(env.model)
        path = videos / f'{task}_heldout12_comparison.mp4'
        with mujoco.Renderer(env.model, height=368, width=640) as renderer:
            with imageio.get_writer(path, fps=25, codec='libx264', quality=8) as writer:
                for t in range(0, 200, 2):
                    panels = []
                    for method, qs in zip(('guidance', 'rl'), trajectories):
                        frame_index = min(t, len(qs)-1)
                        data.qpos[:] = qs[frame_index]
                        mujoco.mj_forward(env.model, data)
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        panel = Image.fromarray(renderer.render().copy())
                        draw = ImageDraw.Draw(panel)
                        row = summary[method][index]
                        label = f'{task} | {method.upper()} | held-out E candidate 12'
                        draw.rectangle((0, 0, 640, 52), fill=(20, 25, 35))
                        draw.text((12, 8), label, fill='white')
                        draw.text((12, 28), f"reward {row['reward']:.3f} | progress {row['progress']:.4f} {env.reward.unit if task == 'zigzag' else 'laps'} | t={t*.02:.2f}s", fill='white')
                        if t >= len(qs):
                            draw.text((12, 65), 'FALL: last recorded frame held', fill=(210, 40, 40))
                        panels.append(np.asarray(panel))
                    writer.append_data(np.concatenate(panels, axis=1))
        print(path, flush=True)
        reader = imageio.get_reader(path)
        first = reader.get_data(0)
        assert first.shape[:2] == (368, 1280)
        reader.close()


if __name__ == '__main__':
    main()
