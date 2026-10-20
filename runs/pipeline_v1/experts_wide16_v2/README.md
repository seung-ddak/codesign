# Wide16 expert 재학습

학습 형상은 e000–e002, e004–e016의 16개이며 e003은 평가에만 사용한다. 이전 wide16은 `squat`과 `trotBackwardFast`가 중간 체크포인트에서 통과한 뒤 후반에 붕괴했다. 원본 `experts_wide16/`은 보존한다.

이번 실행은 동일한 19개 모션과 retarget reference를 사용하고, PPO 학습률을 `1e-3`에서 `2e-4`로 낮춘다. 모션당 14,000,000 환경 스텝을 학습한다. 모든 비영(非零) 체크포인트를 e000/e001/e002/e008/e012/e016에서 영상 없이 평가하고 통과 수, 생존 수, 정규화 오차 순서로 하나를 선택한다. 미학습 e003은 선택에 사용하지 않는다. 선택 체크포인트를 16개 학습 형상과 e003에서 영상과 함께 최종 평가한다. 체크포인트 선택은 후반 붕괴를 피하기 위한 것이며, 형상 분포의 유효 rank가 낮거나 e003이 분포 밖인 문제는 해결하지 않는다.

`queue.log`, 각 모션의 `train.log`, `checkpoint_selection.log`, `eval.log`, W&B 체크포인트·평가 결과를 여기에 저장한다. `github_issue_number.txt`에 새 이슈 번호를 저장하고 완료된 모션마다 평가표를 댓글로 남긴다. 댓글 성공 여부는 이슈의 고유 마커로 확인해 재시작 시 중복을 피한다.

실행 환경: `EXPERT_ROOT=runs/pipeline_v1/experts_wide16_v2`, `EXPERT_DESIGNS="e000 e001 e002 e004 e005 e006 e007 e008 e009 e010 e011 e012 e013 e014 e015 e016"`, `EXPERT_STEPS=14000000`, `EXPERT_LR=2e-4`. GPU 접근과 GitHub API 접근이 필요한 실행은 샌드박스 밖에서 수행한다.
