# factorial_init_action_kp30: 8-45 2×2 MJX 를 kp30 으로 재실행

- 목적: 8-45 는 kp30 에서 생성된 PPO 행동을 kp200 으로 재생해 공통 생존 0/76 (판정 불가). e000 사후 검증에서 kp30 은 76/76 생존. 이득을 일치시켜 "출발 상태 vs 행동 출처" 질문에 답한다.
- 바뀐 것 (단변수): `prepare.py` 의 `'kp':200` → `30` (수집·재생 모두). 사전 기준·분할·MLP·표본 조건은 8-45 그대로. 레퍼런스 행동도 kp30 으로 재생되므로 PD kp200 데이터 (v10~v15) 와 직접 비교하지 않는다.
- 실행: `python runs/pipeline_v1/factorial_init_action_kp30/run.py` (prepare → collect → evaluate, report.html). 사후 probe (`finalize.py`) 는 복사하지 않았다.
