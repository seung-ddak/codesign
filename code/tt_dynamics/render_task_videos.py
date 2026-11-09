"""Render actual 3-D MuJoCo task replays to MP4 without regenerating TT/RL.

This is a separate subprocess by design: MUJOCO_GL must be set BEFORE
importing mujoco. Default OSMesa performs CPU software offscreen rendering,
leaving a busy V100 alone. EGL is opt-in and may use a GPU.

Only archived E/S0/A are used. Each candidate is re-replayed with the same
physics/actions; its physical token/mask must match the saved experiment.
Reference track and predicted trajectories are explicitly marked as visual
overlays and never enter the physical simulation or reward.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def select_candidates(rows, selection):
    if selection == "all":
        return list(range(len(rows)))
    eligible = [i for i, row in enumerate(rows) if row.get("reward") is not None]
    if not eligible:
        return []
    return [max(eligible, key=lambda i: rows[i]["reward"])]


def encode_frames(path, frames, *, width, height, fps):
    """Stream uncompressed RGB frames through ffmpeg; write MP4 atomically."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found. Install ffmpeg for MP4 output.")
    if path.exists():
        raise FileExistsError("Refusing to overwrite existing video: " + str(path))
    tmp = path.with_name(path.stem + ".partial.mp4")
    if tmp.exists():
        raise FileExistsError("Stale partial video must be inspected: " + str(tmp))
    command = [
        exe, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-f", "mp4", str(tmp),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if frame.shape != (height,width,3):
                raise ValueError(f"Unexpected RGB frame shape {frame.shape}")
            process.stdin.write(np.ascontiguousarray(frame,dtype=np.uint8).tobytes())
        process.stdin.close()
        error = process.stderr.read()
        rc = process.wait()
        if rc:
            raise RuntimeError(f"ffmpeg failed ({rc}): {error.decode(errors='replace')[-2000:]}")
        if not tmp.is_file() or tmp.stat().st_size <= 0:
            raise RuntimeError("ffmpeg returned no video")
        tmp.replace(path)
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        process.stderr.close()


def render_candidate(video_path, *, task, descriptor, method, candidate,
                     morphology, pred, physical, mask, fps, width, height):
    """Replay and render a physical robot with trajectory inset, 10s runtime."""
    import mujoco
    from generate_task_mujoco import STEPS, make_physics, replay
    from tt_video_overlay import image_with_overlay

    trace = []
    rereplay, new_mask, metadata = replay(morphology, pred, state_trace=trace)
    count = int(mask.sum())
    if not np.array_equal(new_mask, mask) or len(trace) != count:
        raise RuntimeError("Saved and re-executed physical valid masks differ")
    if count and not np.allclose(rereplay[:count,:42],physical[:count,:42],atol=2e-5,rtol=2e-5):
        raise RuntimeError("Re-render replay diverges from saved physical state. No video written.")
    if count == 0:
        raise ValueError("Cannot render physically invalid initial state")
    if STEPS != 500:
        raise ValueError("Expect 500 simulation states (10s at 50Hz)")

    model = make_physics(morphology)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 145.0
    camera.elevation = -24.0
    camera.distance = 2.7
    stride = 50 // fps
    fall_frame = count-1 if count < STEPS else None
    actual_xy = physical[:count, :2].astype(float)
    predicted_xy = pred[:, :2].astype(float)

    def frames():
        for index in range(0, STEPS, stride):
            # After an early fall, hold the last valid physical frame so that
            # the video is EXACTLY 10 seconds and the fall time remains clear.
            physical_frame = min(index, count-1)
            data.qpos[:] = trace[physical_frame]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3] + np.array([0.,0.,0.08])
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            annotated = image_with_overlay(
                rgb, task=task, task_spec=descriptor,
                physical_xy=actual_xy, predicted_xy=predicted_xy,
                frame=physical_frame, fps=fps, candidate=candidate, method=method,
                fall_step=fall_frame, display_time=index*0.02)
            yield np.asarray(annotated, dtype=np.uint8)

    try:
        encode_frames(video_path, frames(), width=width, height=height, fps=fps)
    finally:
        renderer.close()
    return {"fps":fps, "width":width, "height":height, "backend":os.environ["MUJOCO_GL"],
            "seconds":10.0, "valid_seconds":float(metadata.get("valid_seconds",count*0.02)),
            "fell":bool(count<STEPS), "replay_consistency_checked":True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="Existing task benchmark root containing seed_*.json and .npz")
    p.add_argument("--selection", choices=("best", "all"), default="best")
    p.add_argument("--backend", choices=("osmesa", "egl"), default="osmesa")
    p.add_argument("--egl-device", type=int, default=1,
                   help="EGL device index (only with --backend egl); GPU0 may be busy")
    p.add_argument("--fps", choices=(10,25,50), type=int, default=25)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--resume", action="store_true",
                   help="Skip already encoded MP4 and keep existing candidate metadata")
    args = p.parse_args()
    if args.width < 240 or args.height < 180 or args.width%2 or args.height%2:
        p.error("Even width/height >= 240x180 required for yuv420p MP4")
    if not args.out.is_dir():
        p.error(f"Benchmark root missing: {args.out}")
    # Must be applied in a fresh process, before any MuJoCo import.
    if "mujoco" in sys.modules:
        p.error("Run video renderer in a fresh process so MUJOCO_GL is set before import")
    os.environ["MUJOCO_GL"] = args.backend
    if args.backend == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
    # Defer import until backend is configured.
    try:
        import mujoco  # noqa: F401
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "MuJoCo offscreen renderer unavailable with "
            +args.backend+". For CPU mode install OSMesa (e.g. libosmesa6), "
            "or explicitly choose EGL on a permitted non-V100 GPU."
        ) from exc

    job_files = sorted(args.out.glob("*/gradient_scale_*/seed_*.json"))
    job_files += sorted(args.out.glob("*/rl_ppo/seed_*.json"))
    completed, skipped = 0, 0
    for result_path in job_files:
        task = result_path.relative_to(args.out).parts[0]
        with result_path.open(encoding="utf-8") as f:
            result = json.load(f)
        archive = result_path.with_suffix(".npz")
        if not archive.is_file():
            raise FileNotFoundError(archive)
        from tt_task_report import _gradient_name
        label = _gradient_name(result_path)
        with np.load(archive) as z:
            e_all, predicted, actual, valid = (z[k] for k in ("E", "pred", "mujoco", "valid"))
        for index in select_candidates(result["candidates"], args.selection):
            row = result["candidates"][index]
            name = f"{result_path.stem}_candidate_{index}.mp4"
            video_path = result_path.parent / "videos" / name
            video_path.parent.mkdir(parents=True, exist_ok=True)
            relative_path = video_path.relative_to(args.out).as_posix()
            if video_path.is_file():
                if not args.resume:
                    raise FileExistsError("Already recorded video: "+str(video_path))
                row["video_path"] = relative_path
                skipped += 1
                continue
            if not bool(valid[index].any()):
                row["video_status"] = "skipped_invalid_initial_state"
                skipped += 1
                continue
            metadata = render_candidate(
                video_path, task=task, descriptor=result["task"], method=label,
                candidate=index, morphology=e_all[index],
                pred=predicted[index], physical=actual[index], mask=valid[index],
                fps=args.fps, width=args.width, height=args.height)
            row["video_path"] = relative_path
            row["video_metadata"] = metadata
            completed += 1
            print(f"MP4 saved: {video_path}", flush=True)
        temp = result_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(result_path)

    from tt_task_report import build_report
    build_report(args.out)
    print(f"Videos rendered={completed}, skipped={skipped}. "
          f"Report: {args.out/'report.html'}", flush=True)


if __name__ == "__main__":
    main()
