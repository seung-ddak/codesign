# Pipeline v1: A1 무릎 방향 TT topology

- Topology: `00000011`; 연속 치수 13개를 사용하는 TT 형상 4개(중간값 `e000`, 무작위 표본 `e001`~`e003`). 표본값은 `retargeting/designs.json`에 기록했다. 관절 토크 한계는 A1과 같은 ±33.5 N·m.
- 입력: `artifacts/randomized_reference/manifest.csv`의 25개 모션. tiltYaw, backFlip, howAreYou, rollBack, 점프·계단 및 관련 전환 제외.
- 랜덤화된 A1 reference의 운동학적 미리보기 영상: `randomized_reference_videos/*.mp4` (25 fps). 저장된 자세를 프레임마다 표시한 것으로 PPO 정책의 물리 rollout 영상은 아니다.
- 리타겟 결과: `retargeting/e000`~`e003/00000011/*.npz` 각 25개와 형상별 `summary.csv`. 각 NPZ에 원본, 랜덤화 파라미터, 형상 치수를 저장했다. 총 100개 파일의 관절각 유한값을 확인했다.
- TT 리타겟 영상: `retargeting/videos/e000`~`e003/*.mp4`, 각 형상당 25개(총 100개, 25 fps). TT 모델에 저장된 리타겟 자세를 표시한 운동학적 미리보기이며, PPO 물리 rollout은 아니다.
- 기구학 검사: 25개 모두 foot IK 평균 오차 0.31 mm 이하, 접지 미끄러짐 0.002 m/s 이하, 관절 한계 프레임 0%.
- 역동역학 검사: 정적 자세 5개(`sitDown`, `squat`, `standUp`, `tiltPitch`, `tiltRoll`)는 불량 프레임 0%; 이동 모션은 14.2~65.2%로 PPO tracking 가능 여부를 별도 확인해야 함.
- `check_track_env.log`: `sitDown__r00` reference 관절각 제어 20 step, 생존율 100%.
- `smoke/sitDown__r00/`: CPU PPO 8,192 step 실행, checkpoint와 영상 저장. 평가 4 episode 모두 낙상(`eval/episode/done/fellover/mean=1.0`). **expert 아님.**
- PPO 진입점에 동일 topology의 여러 형상과 각각의 리타겟 reference를 함께 쓰는 학습 경로를 추가했다. `e000`~`e002` 공유 정책을 `sitDown__r00`으로 131,072 step 시험했고, `e003`은 미학습 형상으로 평가했다. 4개 형상 모두 추종 오차 기준을 통과하지 못해 추가 학습 중이다. 형상별 CSV·A1 주황색/TT 파란색 영상과 통합 `summary.csv`는 `training/sitDown_multidesign/`의 W&B offline run에 있다.
- GPU는 RTX 4070 SUPER로 다시 인식된다. `e000` 단독 `sitDown` 131,072-step 시험 결과는 낙상으로 평가 불합격이며, 해당 `motion_eval.csv`와 A1 주황색·TT 파란색 비교 영상은 `experts/sitDown__r00/`의 최신 W&B offline run에 있다.
