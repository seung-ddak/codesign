# Motion-tracking expert

원본 Transformer Transformer의 사족보행 motion-tracking 환경과 morphology-conditioned
PPO 학습 진입점을 현재 연구에서 수정할 수 있도록 가져온 최소 묶음이다.

## 구성

- `track_env.py`: reference pose tracking MJX 환경과 보상
- `train_rl_procedural.py`: morphology domain randomization PPO 학습
- `config/`: 학습·환경·평가 Hydra 설정

공통 MJX, PPO, RoboGen 코드는 중복하지 않고 형제 디렉터리
`../../../transformer-transformer/t2`를 사용한다. 따라서 원본 TT 저장소와 그 `.venv`,
motion pickle, morphology option Zarr가 필요하다.

```bash
cd /home/cgr-ugrad-2026/quad_ws
MUJOCO_GL=egl WANDB_MODE=offline transformer-transformer/.venv/bin/python \
  codesign/code/motion_tracking/train_rl_procedural.py \
  mj_model_data_path=transformer-transformer/umi_on_legs_plus_plus_options/00000000.zarr \
  env.trajectory_file=/path/to/train_motion.pkl \
  eval_env.trajectory_file=/path/to/test_motion.pkl
```

현재 환경은 UMI end-effector pose를 추적한다. 사족보행 full-body walk/jump tracking에는
`track_env.py`의 reference 표현과 reward를 base, joint, foot, contact 기준으로 확장해야 한다.
