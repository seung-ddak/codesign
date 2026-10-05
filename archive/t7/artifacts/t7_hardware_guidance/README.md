# T7 DDPO Hardware Diffusion Fine-tuning

[구현 해설](./t7_hardware_guidance.html) · [아티팩트](https://claude.ai/artifact/JZVKfQpj9f31HconRhmzV5) — 한 update(생성 → 채점 → 갱신)의 코드 흐름, 설정값, 산출물, 한계.

## 구성

- `stage2/code/train_hardware_guidance.py`: T7 본체. 생성 → 채점 → DDPO 업데이트 → 저장
- `stage2/code/ddpo_hardware.py`: stochastic DDIM transition 기록과 clipped DDPO-IS
- `stage2/code/design_scorer.py`: structure group lazy JIT, 고정 명령 7개 vmap 채점 (A안, 09-30)
- `stage2/code/run_t7.sh`: 학습 후 `report.html` 생성 + 텔레그램
- `stage2/code/make_stage_report.py`: `history.jsonl` 기반 T7 보고서

## 실행

```bash
TMUX_RUN_CWD=$PWD bash ../scripts/tmux_run.sh T7_tryN "bash stage2/code/run_t7.sh stage2/runs/T5/try1 stage2/runs/T7/tryN 100"
```
