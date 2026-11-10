"""Render actual 3-D MuJoCo task replays to MP4 without regenerating TT/RL.

This is a separate subprocess by design: MUJOCO_GL must be set BEFORE
importing mujoco. Default OSMesa performs CPU software offscreen rendering,
leaving a busy V100 alone. EGL is opt-in and may use a GPU.

Only archived E/S0/A are used. Each candidate is re-replayed with the same
physics/actions; its physical token/mask must match the saved experiment.
The track and predictions are drawn as overlays. Slalom poles are compiled
into the MuJoCo scene and also marked on the mini-map.
"""
import argparse
import ctypes
import ctypes.util
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
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            exe = get_ffmpeg_exe()
        except (ImportError, RuntimeError, FileNotFoundError) as exc:
            raise RuntimeError("ffmpeg not found. Install ffmpeg or imageio-ffmpeg for MP4 output.") from exc
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
                     morphology, pred, physical, mask, row, fps, width, height):
    """Replay and render a physical robot with trajectory inset, 10s runtime."""
    import mujoco
    from generate_task_mujoco import DT, STEPS, TASKS, make_physics, replay
    from tt_video_overlay import image_with_overlay

    trace = []
    reward = TASKS[task]()
    if json.loads(json.dumps(reward.describe())) != descriptor:
        raise ValueError("Task geometry differs from the saved replay")
    rereplay, new_mask, metadata = replay(morphology, pred, task=reward, state_trace=trace)
    count = int(mask.sum())
    if not np.array_equal(new_mask, mask) or len(trace) != count:
        raise RuntimeError("Saved and re-executed physical valid masks differ")
    if count and not np.allclose(rereplay[:count], physical[:count], atol=2e-5, rtol=2e-5):
        raise RuntimeError("Re-render replay diverges from saved physical state. No video written.")
    for key in ("termination_reason", "control_steps", "pole_contacts"):
        if metadata[key] != row[key]:
            raise RuntimeError(f"Saved and re-executed replay differ in {key}")
    if count < 2:
        raise ValueError("Cannot render a candidate without physical transitions")

    model = make_physics(morphology, reward)
    backend = os.environ.get("MUJOCO_GL")
    data = mujoco.MjData(model)
    renderer = None
    camera = None
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 145.0
    camera.elevation = -24.0
    camera.distance = 2.7
    stride = max(1, round(STEPS / 10 / fps))
    fall_frame = count - 1 if not metadata["complete_10s"] else None
    actual_xy = physical[:count, :2].astype(float)
    predicted_xy = pred[:, :2].astype(float)

    def frames():
        for index in range(0, STEPS, stride):
            # After an early fall, hold the last valid physical frame so that
            # the video is EXACTLY 10 seconds and the fall time remains clear.
            # Include the endpoint even when the requested FPS skips it.
            sample = STEPS if index == STEPS - stride else index
            physical_frame = min(sample, count-1)
            data.qpos[:] = trace[physical_frame]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3] + np.array([0.,0.,0.08])
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            annotated = image_with_overlay(
                rgb, task=task, task_spec=descriptor,
                physical_xy=actual_xy, predicted_xy=predicted_xy,
                frame=physical_frame, fps=fps, candidate=candidate, method=method,
                fall_step=fall_frame, display_time=sample*DT,
                termination_reason=metadata["termination_reason"])
            yield np.asarray(annotated, dtype=np.uint8)

    try:
        encode_frames(video_path, frames(), width=width, height=height, fps=fps)
    finally:
        if renderer is not None:
            renderer.close()
    return {"fps":fps, "width":width, "height":height, "backend":backend,
            "seconds":10.0, "valid_seconds":float(metadata.get("valid_seconds",count*DT)),
            "fell": metadata["fell"], "termination_reason": metadata["termination_reason"],
            "replay_consistency_checked": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="Existing task benchmark root containing seed_*.json and .npz")
    p.add_argument("--selection", choices=("best", "all"), default="best")
    p.add_argument("--backend", choices=("auto", "osmesa", "egl"), default="auto",
                   help="auto selects OSMesa mesh if available, else GPU EGL")
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
    backend = args.backend
    if backend == "auto":
        lib = ctypes.util.find_library("OSMesa")
        try:
            if lib is None:
                raise OSError("libOSMesa is not registered with the system linker")
            ctypes.CDLL(lib)
            probe = subprocess.run(
                [sys.executable, "-c", "import mujoco"],
                env=dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                         CUDA_VISIBLE_DEVICES=""),
                capture_output=True, timeout=20)
            backend = "osmesa" if probe.returncode == 0 else "egl"
        except (OSError, subprocess.TimeoutExpired):
            backend = "egl"
        print(f"CPU video backend auto-selected: {backend}", flush=True)
    os.environ["MUJOCO_GL"] = backend
    if backend == "osmesa":
        # Explicitly avoid using the V100 (which runs an unrelated UNet).
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if backend == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    elif backend == "egl":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
    # Defer import until backend is selected.
    try:
        import mujoco  # noqa: F401
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError(
            f"MuJoCo failed to initialize backend={backend}: {exc}. "
            "Install libosmesa6 (--backend osmesa) or use a working GPU EGL (--backend egl)."
        ) from exc

    job_files = sorted(args.out.glob("*/gradient_scale_*/seed_*.json"))
    job_files += sorted(args.out.glob("*/rl_ppo/seed_*.json"))
    if not job_files:
        p.error(f"No saved physical evaluation files in {args.out}. Run the task benchmark first.")
    from generate_task_mujoco import EVALUATION_VERSION

    from tt_task_report import _gradient_name, build_report, is_mp4

    completed, skipped, failed = 0, 0, 0
    for result_path in job_files:
        task = result_path.relative_to(args.out).parts[0]
        with result_path.open(encoding="utf-8") as f:
            result = json.load(f)
        if result.get("evaluation_version") != EVALUATION_VERSION:
            raise ValueError("Legacy replay archive. Use benchmark_task_inference.py "
                             "--replay-from with a fresh --out before recording.")
        archive = result_path.with_suffix(".npz")
        if not archive.is_file():
            raise FileNotFoundError(archive)
        label = _gradient_name(result_path)
        with np.load(archive) as z:
            e_all, predicted, actual, valid = (z[k] for k in ("E", "pred", "mujoco", "valid"))
        selected = select_candidates(result["candidates"], args.selection)
        for index, row in enumerate(result["candidates"]):
            if index not in selected:
                row["video_status"] = "not_selected"
                continue
            suffix = f"_{backend}"
            name = f"{result_path.stem}_candidate_{index}{suffix}.mp4"
            video_path = result_path.parent / "videos" / name
            video_path.parent.mkdir(parents=True, exist_ok=True)
            relative_path = video_path.relative_to(args.out).as_posix()
            if video_path.is_file():
                if not args.resume:
                    raise FileExistsError("Already recorded video: "+str(video_path))
                if is_mp4(video_path):
                    if row.get("video_path") != relative_path:
                        row["video_metadata"] = {"backend": backend}
                    row["video_path"] = relative_path
                    row["video_status"] = "recorded"
                    row.pop("video_error", None)
                    skipped += 1
                    continue
                row["video_status"] = "render_failed"
                row["video_error"] = "Existing MP4 is invalid. Move it aside and retry."
                failed += 1
                continue
            if not row["control_steps"]:
                row["video_status"] = "no_physical_steps"
                skipped += 1
                continue
            try:
                metadata = render_candidate(
                    video_path, task=task, descriptor=result["task"], method=label,
                    candidate=index, morphology=e_all[index],
                    pred=predicted[index], physical=actual[index], mask=valid[index], row=row,
                    fps=args.fps, width=args.width, height=args.height)
            except (ImportError, RuntimeError, ValueError, OSError) as exc:
                row["video_status"] = "render_failed"
                row["video_error"] = str(exc)
                failed += 1
                print(f"Video failed: {video_path}: {exc}", file=sys.stderr, flush=True)
                continue
            row["video_path"] = relative_path
            row["video_status"] = "recorded"
            row.pop("video_error", None)
            row["video_metadata"] = metadata
            completed += 1
            print(f"MP4 saved: {video_path}", flush=True)
        temp = result_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(result_path)

    build_report(args.out)
    print(f"Videos rendered={completed}, skipped={skipped}, failed={failed}. "
          f"Report: {args.out/'report.html'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
