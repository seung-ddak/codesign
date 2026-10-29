# Frozen CALM HLC+LLC -> TT e000 direct MJX pilot

This is a **one-morphology feasibility test**, not a replacement of the agreed PPO
motion-tracking/transition-expert stages unless its transfer quality is demonstrated.
It uses existing A1 CALM checkpoints without further training, with 30 Hz closed-loop
control and 600 Hz MJX physics. At each control tick the **current TT state** feeds
CALM HLC+LLC. No saved A1 motion is retargeted or replayed.

## Environment

Run from `~/quad_ws/codesign` with the previously used Python environment
containing `mujoco`, `mujoco.mjx`, JAX, PyTorch, SciPy, PyYAML, and the original
CALM dependencies. Prerequisites used by the existing
`check_menagerie_mjx_policy.py`:

- `../external/a1-calm-main/checkpoints/lowLevelController.pth`
- `../external/a1-calm-main/checkpoints/highLevelPolicy.pth`
- `../external/a1-calm-main/calm/` source and network configs
- `~/software/isaacgym/python/isaacgym/torch_utils.py` (pure torch helpers)
- `runs/menagerie_calm_check_final/runtime/rl_games-1.1.4-py3-none-any.whl`
- `../transformer-transformer/` for the TT `00000011` morphology generator

If the wheel is elsewhere, use `--wheel /absolute/path/to/rl_games.whl`.

## Step 1: one-episode smoke test

```bash
cd ~/quad_ws/codesign
git fetch origin
git switch experiment/calm-tt-e000-direct-mjx

../transformer-transformer/.venv/bin/python \
  code/motion_tracking/check_calm_tt_direct.py \
  --out runs/calm_tt_direct_e000_smoke --episodes 1
```

## Step 2: 10 episodes with varied commands

Do this **after** reviewing the one-episode logs and mappings.

```bash
../transformer-transformer/.venv/bin/python \
  code/motion_tracking/check_calm_tt_direct.py \
  --out runs/calm_tt_direct_e000_10 --episodes 10
```

The episode command schedule is the existing seeded `balanced_commands`
(5-second slots including command transitions). For a torque-limit sensitivity
check, rerun using a *different output directory* and `--torque-limit 33.5`.
The default 15 N*m matches the original CALM checkpoint drive, not the full TT
actuator rating. Output directories may not already exist.

## Results

- `summary.json`: episode length, termination, 10-second survival, command world
  XY velocity RMSE, path length, contact and slip diagnostics.
- `report.html`: compact episode table.
- `episode_000.npz` ...: actual MJX `qpos`, `qvel`, and `contact` at **T+1**
  states; applied CALM `target_q` / `action` and world-frame `command` at
  **T** control intervals, `E` (13 morphology coordinates), `time`, `valid`,
  foot positions, initial-substep PD torque. All episodes are saved, including
  early falls; no reset occurs within an episode.

`SCREENING_PASS` requires complete 10 s and speed RMSE <= 0.15 m/s for all
episodes. This is an **exploratory threshold**, not G2/G3 approval or unseen
composite-task success. Report failed episodes and inspect actual trajectories.
Data is provisional; validate the observation adapter and transition/s0
conventions before feeding it to the TT dataset loader.

## Known transfer caveats

1. TT and CALM A1 have matching 12-joint leg *topology* but different dimensions,
   inertias, joint axes/reference frames, and possibly learned normalization. The
   TT 23-slot observation adapter maps body names and uses TT hip as the CALM
   shoulder marker proxy. This can degrade policy even with stable physics.
2. The original TT position actuators are disabled; PD is applied explicitly as
   `clip(40*(target-q)-1.5*dq, -15, +15)` per 600 Hz physics step. Hence this
   is not a default TT actuator comparison.
3. Self-collision is disabled and floor contact enabled for compatibility with
   the prior A1 test. Foot slip is estimated from **foot-center** displacement
   during sustained contact, not the exact contact-point velocity.
4. Passing is not proof of cross-morphology generalization. Before scaling to
   other morphologies, compare this with the existing MJX A1 closed-loop
   command-response baseline and verify action-dependent transition data.
