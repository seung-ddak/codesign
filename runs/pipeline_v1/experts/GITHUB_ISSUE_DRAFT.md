# Pipeline v1: TT 선정 모션 expert 순차 학습

`selection.csv`의 19개 모션을 순서대로 학습하고 평가합니다. 모션마다 e000~e002 TT 형상으로 공유 PPO 정책 하나를 학습하며, e003은 학습하지 않은 형상으로 평가합니다.

- 학습: 모션당 5,242,880 환경 스텝, 병렬 환경 32개, 평가 환경 4개, 배치 32개. GPU 메모리 선할당은 끕니다.
- 실행: `code/motion_tracking/train_selected_experts.sh`, tmux 세션 `pipeline_v1_selected_experts`.
- 로컬 상태: `runs/pipeline_v1/experts/queue.log`. 각 모션의 `train.log`, `eval.log` 및 W&B offline run은 `runs/pipeline_v1/experts/<motion>/` 아래에 저장됩니다.
- 각 모션 완료 후 e000~e003 추종 평가 CSV·A1 주황색/TT 파란색 비교 영상을 생성합니다. 오류가 발생하면 다음 모션을 시작하지 않습니다.

완료 알림 댓글에는 네 형상의 생존·관절/발/자세/위치 오차·접지 일치·통과 여부를 표로 게시합니다. 동일한 내용은 각 run의 `motion_eval/report.md`에도 저장합니다. GitHub 알림을 받으려면 이 Issue를 구독하고 알림 설정을 확인하세요.
