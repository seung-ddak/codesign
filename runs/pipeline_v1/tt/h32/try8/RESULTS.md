# h32_freefix_cgr complete

Command options: `--horizon 32 --steps 100000 --batch 32 --lr 0.0002 --d-model 256 --layers 6 --diff-steps 5 --norm minmax --clip 1 --modes 0 1 2 3 --sampler ddim --seed 0`

Checkpoint and generated sample arrays remain on the CGR PC.

```json
{
  "last_progress": {
    "step": 100000,
    "loss": 0.02490704134106636,
    "val_loss": 0.007503607776015997,
    "lr": 0.0,
    "min": 10.42101670106252
  },
  "window": {
    "n": 1344,
    "yaw_err_deg": 6.172604560852051,
    "yaw_hold_deg": 8.365760803222656,
    "yaw_bias_deg": -0.3905884921550751,
    "yaw_slope": 0.5591085446539155,
    "yaw_corr": 0.6998068640243198,
    "vel_err": 0.18009643256664276,
    "vel_hold": 0.1803923100233078,
    "wz_err": 0.3157314956188202,
    "wz_hold": 0.3959124684333801,
    "pos_err_cm": 10.351564407348633,
    "pos_hold_cm": 26.509380340576172,
    "joint_err_deg": 7.737702369689941
  },
  "episode": {
    "joint_rmse_deg": 9.929598808288574,
    "pos_err_cm": 73.0331802368164
  }
}
```

## Free-s0 circle task check (8 candidates per scale)

Compared with `h32_try5` using the same `generate_task.py --free-s0 --task circle` settings. At scale 0, mean valid simulation time increased from 0.87 s to 6.04 s, but all 8 candidates still fell before the 10 s horizon. At scale 10, all 8 also fell; mean valid time changed from 6.53 s to 5.85 s. Mean path error was 0.294 m and 0.565 m at scales 0 and 10, respectively (previously 0.182 m and 0.348 m). The conditioning correction helped short-horizon survival at scale 0 but did not solve task generation.

Raw results: `runs/pipeline_v1/task/track/try6/summary.json` (local experiment artifact).
