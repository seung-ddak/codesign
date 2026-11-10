"""v12 = v11 과 동일 (PD kp200 데이터, 실행 kp 200), 학습 Mode 만 2/4 → 1/2/4 (20/40/40 %). remind 8-38."""
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
V9 = ROOT / "runs/pipeline_v1/fast_mode24_v9_e_coverage"
STAGES = ["train", "held", "seen", "native", "smoke", "fixedE_zigzag", "joint_zigzag",
          "fixedE_circle", "joint_circle", "report"]


def available_ram():
    return int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines()
                    if x.startswith("MemAvailable:"))) * 1024


def worker(stage):
    os.chdir(ROOT)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    os.nice(10)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.60 if stage == "train" else .25, 0)
    sys.path[:0] = [str(ROOT / "code/tt_dynamics"), str(ROOT / "code/motion_tracking")]
    conditions = json.loads((RUN / "conditions.json").read_text())
    out = RUN / "evaluation"
    out.mkdir(exist_ok=True)
    if stage == "train":
        args = [str(ROOT / "code/tt_dynamics/train_tt.py"), *conditions["training_args"]]
    elif stage in ("held", "seen"):
        models = {**{k: [v, 1] for k, v in conditions["baseline_models"].items()}, "pd": [str(RUN / "tt_joint"), 1]}
        args = [str(V9 / "measure_given_a.py"), str(out / f"given_a_{stage}.json"), json.dumps(models)]
        if stage == "held":
            args.append(str(V9 / "data_test"))
        args += ["--only-requested", "--pair-checkpoint", conditions["pair_checkpoint"],
                 "--metric-reference", conditions["metric_reference"]]
    elif stage in ("native", "report"):
        args = [str(RUN / ("evaluate_native.py" if stage == "native" else "build_report.py"))]
    else:
        import jax
        jax.config.update("jax_compilation_cache_dir", str(ROOT / "jax_cache"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        args = [str(ROOT / "code/tt_dynamics/generate_task.py"), "--ckpt", str(RUN / "tt_joint"),
                "--expert-config", "runs/pipeline_v1/experts_wide16/walkForwardSlow__r00",
                "--anchor-s0", "--joint-full-guidance", "--e-rest-s0", "--no-s0-align", "--kp", "200"]
        if stage == "smoke":
            args += ["--no-render", "--out", str(out / "check_w1/smoke_zigzag_joint_c16"),
                     "--seconds", "2", "--candidates", "16", "--task", "zigzag", "--scales", "0"]
        else:
            kind, task = stage.split("_", 1)
            args += ["--out", str(out / f"check_w1/{kind}_{task}_4s"), "--seconds", "4",
                     "--scales", "10", "--task", task]
            if kind == "fixedE":
                designs = json.loads((ROOT / "runs/pipeline_v1/retargeting/designs.json").read_text())
                args += ["--no-render", "--candidates", "4", "--fixed-e",
                         *[k for k in designs if k.startswith("e") and k != "e003"]]
            else:
                args += ["--candidates", "16"]
    sys.argv = args
    print("STAGE", stage, flush=True)
    runpy.run_path(args[0], run_name="__main__")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
        return
    if available_ram() < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB available RAM before launch")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=".40", TMPDIR=str(ROOT / ".runtime/tmp"),
               MPLCONFIGDIR=str(ROOT / ".runtime/mpl"), XDG_CACHE_HOME=str(ROOT / ".runtime/cache"))
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    for name in ("tmp", "mpl", "cache"):
        (ROOT / ".runtime" / name).mkdir(parents=True, exist_ok=True)
    status_path = RUN / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for stage in STAGES:
        if status.get(stage, {}).get("status") == "completed":
            continue
        if stage == "train" and (RUN / "tt_joint/checkpoint.pt").exists():
            raise FileExistsError("Existing checkpoint without completed status; inspect before overwriting")
        with (RUN / f"{stage}.log").open("a") as log, (RUN / "resources.jsonl").open("a") as resources:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", stage],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status[stage] = dict(status="running", pid=child.pid, started=time.time())
            status_path.write_text(json.dumps(status, indent=2))
            low = 0
            while child.poll() is None:
                ram = available_ram()
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                vals = gpu.stdout.strip().split(",")
                free_gpu = float(vals[1]) if len(vals) >= 2 else 0
                disk = {p: shutil.disk_usage(p).free / 1024**3 for p in ("/", str(ROOT))}
                resources.write(json.dumps(dict(stage=stage, time=time.time(), ram_available_gib=ram / 1024**3,
                                                gpu=gpu.stdout.strip(), disk_free_gib=disk)) + "\n")
                resources.flush()
                low = low + 1 if ram < 4 * 1024**3 or free_gpu < 1024 or disk["/"] < 1 or disk[str(ROOT)] < 8 else 0
                if low >= 2:
                    status[stage]["stop_reason"] = "resource headroom threshold"
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(10)
            status[stage].update(status="completed" if child.returncode == 0 else "stopped",
                                 returncode=child.returncode, ended=time.time())
            status_path.write_text(json.dumps(status, indent=2))
            print(stage, status[stage]["status"], flush=True)
            if child.returncode:
                raise SystemExit(child.returncode)
    print("PD_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
