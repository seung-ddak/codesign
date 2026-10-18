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
