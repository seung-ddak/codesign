# Motion tracking pipeline

`randomize_reference.py`는 사용할 A1 reference를 `artifacts/randomized_reference/`에 만든다.
`pd_replay.py`는 이를 A1과 무릎 방향이 같은 TT topology `00000011`로 리타겟한다.
현재 결과는 `runs/pipeline_v1/retargeting/e000/00000011/` 및 `e000/summary.csv`에 있다.

```bash
cd codesign
../transformer-transformer/.venv/bin/python code/motion_tracking/pd_replay.py \
  --retarget --source-dir artifacts/randomized_reference \
  --out runs/pipeline_v1/retargeting/e000 --choices 00000011 --torque-limit 33.5
```

`train_rl_procedural.py`와 `track_env.py`는 PPO 학습용이다. 기본 설정은 같은 topology의
`e000`~`e002` 모델과 각각의 reference로 공유 정책을 학습하고 `e003`을 미학습 형상으로
남긴다. `check_multi_design.py`는 모델·reference 대응 및 독립 환경과의 1-step 일치를 검사한다.
모션별 학습은 `env.clips`와 `eval_env.clips`를 같은 모션으로 override한다. 학습한 run은
`eval_multi_designs.sh <run>`으로 평가하며 형상별 `motion_eval.csv`와 A1 주황색·TT 파란색
영상, 통합 `motion_eval/summary.csv`를 생성한다.
