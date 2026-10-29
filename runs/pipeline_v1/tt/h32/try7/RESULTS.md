# h32_try4_cgr complete

100,000 steps; horizon 32; modes 0/1/2; diffusion 5; minmax; DDIM.

Checkpoint and generated sample arrays remain on the CGR PC.

```json
{
  "last_progress": {
    "step": 100000,
    "loss": 0.010324859991669655,
    "val_loss": 0.005431043915450573,
    "lr": 0.0,
    "min": 10.953666889667511
  },
  "window": {
    "n": 1344,
    "yaw_err_deg": 5.945329189300537,
    "yaw_hold_deg": 8.365760803222656,
    "yaw_bias_deg": 0.03886009380221367,
    "yaw_slope": 0.6241187902877189,
    "yaw_corr": 0.7168372929560916,
    "vel_err": 0.15548007190227509,
    "vel_hold": 0.1803923100233078,
    "wz_err": 0.3019513189792633,
    "wz_hold": 0.3959124684333801,
    "pos_err_cm": 8.377057075500488,
    "pos_hold_cm": 26.509380340576172,
    "joint_err_deg": 7.124707221984863
  },
  "episode": {
    "joint_rmse_deg": 8.7682466506958,
    "pos_err_cm": 58.92914581298828
  }
}
```
