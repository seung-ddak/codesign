# Motion-tracking expert

원본 Transformer Transformer의 사족보행 motion-tracking 환경과 morphology-conditioned
PPO 학습 진입점을 현재 연구에서 수정할 수 있도록 가져온 최소 묶음이다.

## 구성

- `pd_replay.py`: raw CSV → reference (`artifacts/pd_replay/<clip>.npz`, fix_*), A1 모델 `build_model`
- `track_env.py`: full-body reference tracking MJX 환경 (Peng 2020 보상)
- `train_rl_procedural.py`: PPO 학습 (현재 A1 1 개 형상)
- `check_track_env.py`: 환경 자가 점검 (RSI 오차 0, reference 관절각 재생 시 항별 보상)
- `config/`: 학습·환경·평가 Hydra 설정 (`env/a1_track*.yaml`)

공통 MJX, PPO 코드는 형제 디렉터리 `../../../transformer-transformer/t2`와 그 `.venv`를 쓴다.

```bash
cd codesign
MUJOCO_GL=egl WANDB_MODE=offline ../transformer-transformer/.venv/bin/python \
  code/motion_tracking/train_rl_procedural.py
```
