"""v12 (remind 8-38) 와 동일한 평가를 임의 run 폴더에 적용한다 — 학습 없음.

    python runs/pipeline_v1/eval_v12parity/run_eval.py runs/pipeline_v1/<run>

단계: held/seen (516쌍, Mode 2 A 고정) → held_mode1 (--pin-mode 1, 8-43 기준 1) → native (PD 검증)
→ smoke/fixedE/joint × zigzag/circle (MJX 176 후보) → report (build_eval_report.py, 8-43 사전 기준 판정).
상태는 <run>/eval_status.json (학습 status.json 과 분리). 완료 단계는 재실행 시 건너뛴다.
"""
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V9 = ROOT / "runs/pipeline_v1/fast_mode24_v9_e_coverage"
V12 = ROOT / "runs/pipeline_v1/fast_mode24_v12_pd_kp200_mode1"
STAGES = ["held", "seen", "held_mode1", "native", "delta_r2", "smoke", "fixedE_zigzag", "joint_zigzag",
          "fixedE_circle", "joint_circle", "report"]
SCRIPTS = dict(native="evaluate_native.py", report="build_eval_report.py", delta_r2="measure_delta_r2.py")


def run_dir():
    p = Path(sys.argv[1])
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def available_ram():
    return int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines()
                    if x.startswith("MemAvailable:"))) * 1024


def worker(run, stage):
    os.chdir(ROOT)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    os.nice(10)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    sys.path[:0] = [str(ROOT / "code/tt_dynamics"), str(ROOT / "code/motion_tracking")]
    conditions = json.loads((run / "conditions.json").read_text())
    out = run / "evaluation"
    out.mkdir(exist_ok=True)
    ref = ROOT / conditions.get("reference_run", V12.relative_to(ROOT))  # 비교 기준 run (기본 v12, Δ 실험은 twin)
    ref_models = {"ref": [str(ref / "tt_joint"), 1], "pd": [str(run / "tt_joint"), 1]}
    if stage in ("held", "seen", "held_mode1"):
        models = {**{k: [v, 1] for k, v in conditions["baseline_models"].items()}, **ref_models}
        if stage == "held_mode1":
            models = ref_models
        args = [str(V9 / "measure_given_a.py"), str(out / f"given_a_{stage}.json"), json.dumps(models)]
        if stage != "seen":
            args.append(str(V9 / "data_test"))
        args += ["--only-requested", "--pair-checkpoint", conditions["pair_checkpoint"],
                 "--metric-reference", conditions["metric_reference"]]
        if stage == "held_mode1":
            args += ["--pin-mode", "1"]
    elif stage in SCRIPTS:
        args = [str(HERE / SCRIPTS[stage]), str(run)]
    else:
        import jax
        jax.config.update("jax_compilation_cache_dir", str(ROOT / "jax_cache"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        args = [str(ROOT / "code/tt_dynamics/generate_task.py"), "--ckpt", str(run / "tt_joint"),
                "--expert-config", "runs/pipeline_v1/experts_wide16/walkForwardSlow__r00",
                "--anchor-s0", "--joint-full-guidance", "--e-rest-s0", "--no-s0-align", "--kp", str(conditions["execution_kp"])]
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
    run = run_dir()
    if len(sys.argv) == 4 and sys.argv[2] == "--worker":
        worker(run, sys.argv[3])
        return
    if not (run / "tt_joint/checkpoint.pt").exists():
        raise FileNotFoundError(run / "tt_joint/checkpoint.pt")
    if available_ram() < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB available RAM before launch")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=".40", TMPDIR=str(ROOT / ".runtime/tmp"),
               MPLCONFIGDIR=str(ROOT / ".runtime/mpl"), XDG_CACHE_HOME=str(ROOT / ".runtime/cache"))
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    env.pop("JAX_PLATFORMS", None)
    for name in ("tmp", "mpl", "cache"):
        (ROOT / ".runtime" / name).mkdir(parents=True, exist_ok=True)
    status_path = run / "eval_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for stage in STAGES:
        if status.get(stage, {}).get("status") == "completed":
            continue
        with (run / f"eval_{stage}.log").open("a") as log, (run / "eval_resources.jsonl").open("a") as resources:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(run), "--worker", stage],
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
    print("EVAL_COMPLETE", run.name, flush=True)


if __name__ == "__main__":
    main()
