"""Render randomized A1 or retargeted TT reference poses as kinematic videos."""

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np

from pd_replay import ROOT, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(ROOT) / "artifacts/randomized_reference")
    parser.add_argument("--out", type=Path, default=Path(ROOT) / "runs/pipeline_v1/randomized_reference_videos")
    parser.add_argument("--only", nargs="*", help="Clip names without .npz; default: all clips")
    parser.add_argument("--choice", help="TT topology; infer morphology from the first retargeted NPZ")
    args = parser.parse_args()
    paths = [args.source / f"{name}.npz" for name in args.only] if args.only else sorted(args.source.glob("*.npz"))
    if not paths or any(not path.is_file() for path in paths):
        parser.error("No matching reference clips found")

    with np.load(paths[0]) as first:
        uniforms = first["uniforms"] if args.choice else None
    model = build_model(choice=args.choice, uniforms=uniforms)
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 1.55, 120, -18
    scene_option = mujoco.MjvOption()
    if args.choice:
        scene_option.geomgroup[3] = 1
    args.out.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, 480, 640) as renderer:
        for path in paths:
            with np.load(path) as clip:
                if args.choice and not np.array_equal(clip["uniforms"], uniforms):
                    raise ValueError(f"Morphology differs from first clip: {path}")
                poses = np.column_stack((clip["fix_base_pos"], clip["fix_base_quat"], clip["fix_q"]))
                dt = float(clip["dt"])
            if poses.shape[1] != model.nq or not np.isfinite(poses).all() or dt <= 0:
                raise ValueError(f"Invalid reference poses: {path}")
            stride = max(1, round(1 / (25 * dt)))
            fps = round(1 / (dt * stride))
            output = args.out / f"{path.stem}.mp4"
            with imageio.get_writer(output, fps=fps, codec="libx264", quality=7) as writer:
                for pose in poses[::stride]:
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = data.qpos[:3]
                    renderer.update_scene(data, camera=camera, scene_option=scene_option)
                    writer.append_data(renderer.render())
            print(f"{path.name} -> {output} ({len(poses[::stride])} frames, {fps} fps)", flush=True)


if __name__ == "__main__":
    main()
