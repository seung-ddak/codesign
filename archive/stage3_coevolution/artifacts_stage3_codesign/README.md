# Stage 3 Hardware–Policy Co-design

[다이어그램](./stage3_codesign.html)은 Stage 2 성공 후 TT `ψ`와 locomotion `πθ`를 번갈아 갱신하는 Stage 3 co-design 흐름을 보여준다.

- `stage3/code/`: 공진화 학습, 환경, guidance, viewer, 검사 및 실행 스크립트
- `stage3/runs/`: 공진화 실험별 checkpoint와 결과
- `stage3/README.md`: 실행법과 검증 상태

DDPO learner는 Stage 2 소유인 `stage2/code/ddpo_hardware.py`에 있다. Stage 3는 이를 가져와 `πθ`와 함께 갱신한다. 진입 조건은 T7이 동일 simulation budget에서 pretrained TT보다 나은 return을 보이고 invalid·mode collapse 기준을 통과하는 것이다.
