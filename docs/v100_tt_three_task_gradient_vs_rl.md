# TT 10-second task benchmark: Reward Gradient vs episodic PPO

**Existing checkpoint only; TT weights and the six-step research pipeline stay unchanged.**
This is **task inference** using frozen TT parameters, not re-training TT or
PPO motion/transition experts. The physical evaluator is MuJoCo, not MJX.

## Tasks and metrics

| Task flag | Reward class | What "success" actually means |
|---|---|---|
| circle | TrackReward | counterclockwise progress along radius-1m circular track, lane/stability/effort penalties |
| zigzag | ZigzagReward | forward distance following triangular zigzag centerline, lane/stability/effort penalties |
| straight | StraightReward | maximize +X progress (fast forward) while staying in a straight corridor |

Each method generates a normalized 13-component morphology E, the 500-step
predicted 42D physical state and 12D target joint action. The **same**
generated E, S0, A run in physical MuJoCo (PD kp=30, torque=33.5,
control step 0.02s, integrator step 0.005s) without per-step corrections.
Predictions do not count as task success until physically replayed.

Methods:
- **Gradient scale=0**: frozen TT with no reward guidance (negative control).
- **Gradient scale=1**: existing differentiable reward guidance through 5 DDIM
  steps, modifying shared E and S/A latents.
- **RL (episodic PPO)**: frozen TT; a diagonal Gaussian low-dimensional policy
  chooses one E/S/A residual correction vector per episode, broadcast into
  reverse diffusion. Policy is updated with **actual MuJoCo task returns** via
  clipped PPO. Evaluation uses the policy mean and new held-out random seeds.
  It is *not* a reactive walking policy or full per-denoising-step PPO.

All three methods use the same evaluation seed(s), candidate counts,
reward functions and physics. PPO's *extra* training rollouts are logged
separately as its optimization budget; they are **not** included in evaluation
scores. A small PPO budget does not imply convergence.

## Run safely on the server while the UNet job uses the V100

**Never terminate or disturb the existing UNet process.** Explicitly disable
CUDA for this benchmark; TT inference and MuJoCo playback then use CPU.

```bash
cd ~/quad_ws/codesign
git switch experiment/v100-mujoco-10k-tt
git pull --ff-only origin experiment/v100-mujoco-10k-tt
export PYTHON="$PWD/../transformer-transformer/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

"$PYTHON" -m pytest -q tests/test_tt_task_benchmark.py
```

### Pilot: all 3 tasks × 2 guidance strengths × RL (small budget)

```bash
"$PYTHON" -u code/tt_dynamics/benchmark_task_inference.py \
  --ckpt runs/v100_mujoco_10k/tt/h64_warmup_ema_mode0124 \
  --out runs/v100_mujoco_10k/task_gradient_vs_ppo_pilot_v1 \
  --device cpu --cpu-threads 2 \
  --tasks circle zigzag straight \
  --methods gradient rl \
  --scales 0 1 --candidates 2 --eval-seeds 1 \
  --rl-updates 2 --rl-rollouts 4 --ppo-epochs 4 \
  2>&1 | tee runs/v100_mujoco_10k/task_gradient_vs_ppo_pilot_v1.log
```

### Larger comparison (distinct output directory)

```bash
"$PYTHON" -u code/tt_dynamics/benchmark_task_inference.py \
  --ckpt runs/v100_mujoco_10k/tt/h64_warmup_ema_mode0124 \
  --out runs/v100_mujoco_10k/task_gradient_vs_ppo_full_v1 \
  --device cpu --cpu-threads 2 \
  --tasks circle zigzag straight \
  --methods gradient rl \
  --scales 0 1 --candidates 4 --eval-seeds 3 \
  --rl-updates 40 --rl-rollouts 4 --ppo-epochs 4 \
  2>&1 | tee runs/v100_mujoco_10k/task_gradient_vs_ppo_full_v1.log
```

Important: TT forward+backward on CPU may consume nontrivial memory and compute
and slow any concurrent CPU-using job. Do the small pilot first and monitor
`top`, `free -h`, and `nvidia-smi`. GPU 0 should not be newly occupied.

If interrupted: repeat the **same** command, adding `--resume`. RL policy
checkpoint and candidate evaluation files are persisted per task; completed
evaluation JSON files are skipped. If interruption leaves only a partial NPZ
without JSON, inspect it rather than deleting prior data blindly.

## Result files

```text
runs/v100_mujoco_10k/task_gradient_vs_ppo_full_v1/
  experiment_config.json
  circle/
    gradient_scale_0/seed_*.json, seed_*.npz
    gradient_scale_1/seed_*.json, seed_*.npz
    rl_training/training.jsonl, policy_state.pt
    rl_ppo/seed_*.json, seed_*.npz
  zigzag/    [same]
  straight/  [same]
  report.json
  report.md
  report.html
```

Each NPZ stores `E`, `pred` (TT states/actions), `mujoco` (actual
states/actions), and `valid` (physical time masks). Candidate JSON also
contains physical reward terms, real 10s completion, duration, forward
distance, progress, morphology, and predicted-vs-replayed state errors.

To regenerate the reports without loading the TT checkpoint, GPU or MuJoCo:

```bash
"$PYTHON" code/tt_dynamics/tt_task_report.py \
  --out runs/v100_mujoco_10k/task_gradient_vs_ppo_full_v1
```

## Interpretation rules

A method succeeds **only if physical MuJoCo scores and completion improve**.
The first metric is 10s completion rate with all attempts in denominator;
then compare physical reward, valid episode duration, track progress, and
morphology. Invalid initial conditions appear separately and are not given
fake zero rewards. The PPO training uses -100 as a learning penalty for
invalid episodes; reporting omits these from the mean valid replay reward.
An early physical fall receives -20, matching the prior experimental task
script. The predicted-vs-physical XY and joint errors help diagnose the TT
model's previously measured dynamics inconsistency.

**No actual three-task results are contained in this commit.** They will only
appear after executing the run against the pre-trained checkpoint and
full MuJoCo environment on the research machine.

## 10-second MuJoCo MP4 video recording (new)

The video tool **never trains TT or RL**. It reads archived generated E/S0/A
from each method's NPZ, replays the same CPU MuJoCo model, and verifies the
rerun matches the saved physical state/mask before making the video.
The video displays the actual 3D robot and a top-down mini-map:
gray = task centerline/boundaries, orange = TT-predicted path,
green = actual physical MuJoCo path. The 2D guide is **not a physical obstacle**.

Each video is exactly **10 seconds**, even if the robot falls earlier.
After a fall, the last actual physical frame is held and the fall time is
shown explicitly. Default output: H.264 MP4, **960×540, 25 FPS**, no audio.

### Video added automatically during the 3-task experiment

Add these options to the command above:

```bash
  --record-video --video-selection best --video-backend osmesa --video-fps 25
```

Use `--video-selection all` to record every candidate, not only the best
physical reward candidate per method and evaluation seed.

### Add videos to an experiment already finished (recommended if running now)

This does not repeat the 3-task TT or RL inference, only physics replay
for validation and rendering:

```bash
cd ~/quad_ws/codesign
export PYTHON="$PWD/../transformer-transformer/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

"$PYTHON" -m pytest -q tests/test_tt_task_video.py

"$PYTHON" -u code/tt_dynamics/render_task_videos.py \
  --out runs/v100_mujoco_10k/task_gradient_vs_ppo_pilot_v1 \
  --backend osmesa --selection best --fps 25 --width 960 --height 540 \
  --resume
```

Change `--out` to your actual existing completed benchmark folder.
The tool emits paths such as:

```text
circle/gradient_scale_0/videos/seed_20000_candidate_0.mp4
circle/gradient_scale_1/videos/seed_20000_candidate_0.mp4
circle/rl_ppo/videos/seed_20000_candidate_0.mp4
zigzag/... /videos/...
straight/... /videos/...
report.html  (embedded MP4 players for best candidates)
report.md    (direct MP4 links)
```

You can inspect `report.html` in a browser; links are relative to the
benchmark folder. All videos are preserved during resume and never silently
overwritten.

### Renderer dependency and GPU safety

**Default = OSMesa CPU software rendering.** This is deliberate while the
V100 runs the unrelated UNet job. It needs the system OSMesa runtime and
Python Pillow plus ffmpeg (a bundled `imageio-ffmpeg` binary also works).
Before rendering, check:

```bash
ldconfig -p | grep -i osmesa
"$PYTHON" -c "import PIL; print('Pillow ready')"
ffmpeg -version | head -n 1
```

If OSMesa cannot import, install/activate the appropriate system OSMesa
package (`libosmesa6` on Debian/Ubuntu) using your administrator.
Do not terminate the UNet process to make a video.

EGL rendering is an explicit alternative only if a **separate permitted
GPU** is known to be available. For example, on a machine where EGL device
index 1 has been verified to be Quadro P620 and NOT the V100:

```bash
"$PYTHON" code/tt_dynamics/render_task_videos.py \
  --out runs/v100_mujoco_10k/task_gradient_vs_ppo_pilot_v1 \
  --backend egl --egl-device 1 --selection best --resume
```

Do not assume EGL device ordering matches `nvidia-smi` numbering without
checking. The safe default remains OSMesa; CPU video processing may still
compete with other CPU jobs.

### Recording integrity and scope

- Video recording only happens **after** numerical task evaluation.
- If recording fails, existing NPZ and JSON scores stay intact.
- No created MP4 is presented as evidence unless recorded real MuJoCo
  poses match the archived physical result within tolerance.
- The candidate's `video_path` and recording metadata are written back to
  its result JSON; the report is regenerated.
- A candidate with an invalid initial S0 has no physical video and is
  explicitly skipped. It is not animated as if successful.
- The 3D MuJoCo video is not a policy rollout; it is physical open-loop replay
  of TT-generated actions under the generated morphology.

## 2026-11-09: OSMesa error on the V100 server — CPU no-GL fallback

The first 3-task pilot **completed all 18 physical MuJoCo replays** but
`MUJOCO_GL=osmesa` failed during import with
`AttributeError: 'NoneType' object has no attribute 'glGetError'`.
This is a missing/unloadable OSMesa library error, **not** a failed TT,
PPO, or physical rollout. Do not rerun training just to get video.

Pull the current branch, then use the *existing* NPZ/JSON to produce MP4:

```bash
cd ~/quad_ws/codesign
git pull --ff-only origin experiment/v100-mujoco-10k-tt
export PYTHON="$PWD/../transformer-transformer/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# Check the MP4 encoder: either system ffmpeg or imageio-ffmpeg is needed.
command -v ffmpeg || "$PYTHON" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"

"$PYTHON" -m pytest -q tests/test_tt_task_video.py
"$PYTHON" -u code/tt_dynamics/render_task_videos.py \
  --out runs/v100_mujoco_10k/task_gradient_vs_ppo_video_v1 \
  --backend wireframe --selection best --fps 25 --resume
```

If both MP4 encoders are absent, install the **Python-only encoder wheel**
in the existing project venv, not the UNet conda env:

```bash
"$PYTHON" -m pip install imageio-ffmpeg
```

`--backend wireframe` renders the **actual MuJoCo forward-kinematics
joint anchors, foot geoms and body orientation** as an animated 3-D
isometric **line drawing**. It does not provide photorealistic 3-D meshes
or collision geometry visualization and it **does not use GPU/OSMesa/EGL**.
The right-side path overlay and fall marker are preserved. This is the
recommended zero-GL, zero-GPU option while UNet is using V100.

`--backend auto` uses OSMesa mesh rendering if its system library can
be loaded and otherwise falls back to wireframe. For fully rendered robot
meshes (distinct from kinematic line art), ask the system administrator to
install `libosmesa6` and validate `MUJOCO_GL=osmesa` in the research
Python environment. Do not kill or interfere with the ongoing UNet job.
Once OSMesa is available you can request `--backend osmesa`.
The recorder never silently overwrites existing MP4 files.

Pilot results: all three tasks had zero 10-second completions across
3 methods × 2 candidates = 18 physical rollouts. The successful PPO
iteration logs do **not** imply task success or policy convergence.

## Video does not appear in report.html (path / recording audit)

Starting with the report MP4 gallery update, `tt_task_report.py` now
**discovers existing video files directly** in each method's `videos/`
folder, even if the saved candidate JSON does not contain `video_path`.
It also shows an explicit available/missing count and a direct link for
each MP4 (in addition to HTML5 video players).

**Do not rerun TT inference or PPO training just to rebuild a report.**
Use the existing experiment directory:

```bash
cd ~/quad_ws/codesign
git pull --ff-only origin experiment/v100-mujoco-10k-tt
export PYTHON="$PWD/../transformer-transformer/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
ROOT=runs/v100_mujoco_10k/task_gradient_vs_ppo_video_v1

"$PYTHON" -m pytest -q tests/test_tt_task_benchmark.py
find "$ROOT" -type f -name '*.mp4' -printf '%p (%s bytes)\\n'
"$PYTHON" code/tt_dynamics/tt_task_report.py --out "$ROOT" >/dev/null

"$PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path("runs/v100_mujoco_10k/task_gradient_vs_ppo_video_v1")
s=json.loads((root/"report.json").read_text())
print("Videos available:",s["videos"]["available"],
      "missing:",s["videos"]["missing"])
for item in s["videos"]["candidates"]:
    print(item["task"],item["method"],item["candidate"],
          item["status"],item["video_path"])
print("HTML:",(root/"report.html").resolve())
PY
```

- If `available=0`, there are no completed MP4 files. Run
  `render_task_videos.py --out "$ROOT" --backend wireframe --selection best --resume`
  and inspect the traceback if it fails. This is a rendering issue, not a
  TT training issue.
- If `available > 0` but the browser cannot play the video,
  click `MP4 직접 열기` in the report and open the MP4 directly. Keep
  `report.html` and the sibling `circle/`, `zigzag/`,
  `straight/` folders together; copying just the HTML breaks relative
  video URLs. Open the actual report on the server via the browser/file
  manager, or download the **whole output folder**, not the HTML alone.
- Rebuilding the report does not install OSMesa or encode video files.
  It only validates their presence and links them correctly.
