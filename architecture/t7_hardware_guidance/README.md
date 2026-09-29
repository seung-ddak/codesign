# T7 DDPO Hardware Diffusion Fine-tuning

아키텍처는 두 패널로 나뉜다.

- [① Generation & Scoring](./t7_generation_scoring.html): stochastic DDIM 생성, Zarr 기록, morphology 그룹화, MJX return 계산
- [② Advantage & DDPO Update](./t7_ddpo_update.html): global advantage, transition shuffle, clipped DDPO-IS, checkpoint 저장

①의 `Terminal returns`와 `Zarr state chain`이 ②의 입력으로 이어진다.

## 구성

- `stage2/code/train_hardware_guidance.py`: T7 본체. 생성 → 채점 → DDPO 업데이트 → 저장
- `stage2/code/ddpo_hardware.py`: stochastic DDIM transition 기록과 clipped DDPO-IS
- `stage2/code/design_scorer.py`: 필요한 structure group만 lazy JIT하고 고정 명령 7개를 순차 채점
- `stage2/code/run_t7.sh`: 100 updates 실행 후 `report.html` 생성
- `stage2/code/make_stage_report.py`: `history.jsonl` 기반 T7 보고서

## 구현 흐름

한 update는 다음 순서로 동작한다.

1. TT가 `generation_microbatch` 단위로 morphology를 생성한다.
2. DDIM state는 RAM 리스트 대신 `<run>/buffer/update_*.zarr`에 FP32 state chain으로 즉시 기록한다. 인접 transition이 공유하는 state는 한 번만 저장한다.
3. 생성 결과를 `structure_key`로 묶고, 실제 등장한 group만 고정 크기 `slots`로 lazy JIT한다. 7개 command는 같은 compiled rollout에 동적 입력으로 순차 전달한다.
4. global batch의 return으로 advantage를 계산한 뒤, 모든 stochastic transition을 섞어 `ddpo_minibatch` 단위로 clipped DDPO-IS를 수행한다.
5. update가 끝나면 Zarr buffer를 삭제하고 `history.jsonl` 및 주기적인 `psi_*.pt` checkpoint를 남긴다.

`global_batch`는 한 update의 morphology 수, `generation_microbatch`는 TT 동시 생성 수, `slots`는 MJX 동시 평가 수, `ddpo_minibatch`는 TT 최적화 시 한 번에 읽는 transition 수다. 네 값은 서로 독립적이다.

## 기본 실행

```bash
MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python \
  stage2/code/train_hardware_guidance.py \
  --pi-theta stage2/runs/T5/try1 \
  --out stage2/runs/T7/try5 \
  --updates 100 --global-batch 32 --generation-microbatch 4 \
  --slots 1 --ddpo-minibatch 32 --ddpo-epochs 4 --psi-lr 1e-6
```

현재 PC에는 `../scripts/tmux_run.sh`가 없으므로 실행은 직접 `tmux` 세션에서 시작해야 한다. 기존 `T7_smoke`와 `T7_try1`은 residual 방식의 과거 결과이므로 DDPO 성능 근거로 사용하지 않는다.

메모리는 setup, generation, group compile 전후, scoring, DDPO 단계마다 `rss`, `peak`, `compiled_groups`로 출력된다. `--slots`는 안정성 확인 후에만 늘린다.
