# CALM 로컬 실행 호환 수정

`standalone_ghost_drawing.patch`는 원본 ZIP에 포함되지 않은
`~/project/calm-root/scripts/csv_to_rollout_motion.py` 의존성을 제거한다.
Ghost 시각화에 필요한 X/Y 회전 행렬과 URDF 몸통 치수는 해당 파일 안에 두고,
Quaternion 회전은 설치된 Isaac Gym 함수를 재사용한다.
정책 가중치·학습 설정·시뮬레이션 제어 조건은 바꾸지 않는다.

현재 `external/a1-calm-main`에는 적용되어 있다. 원본 ZIP을 다시 풀었다면 다음과 같이 적용한다.

```bash
cd /home/cgr-ugrad-2026/quad_ws/external/a1-calm-main
patch --forward -p1 < /home/cgr-ugrad-2026/quad_ws/codesign-calm/code/motion_tracking/calm/patches/standalone_ghost_drawing.patch
```

`fixed_latent_per_env_test.py --help` import 검사와 회전 행렬 테스트를 통과했다.
이 검사는 정책의 추종 성능을 보증하지 않는다.
