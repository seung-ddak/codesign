# codesign 작업 규칙 (Claude 용)

## 폴더 구조 (2026-11-10 정리, 사용자 결정)
- `legacy/`: 2026-11-10 이전의 모든 내용 (코드·실험 결과·기록 remind.md 등). 옛 경로 `codesign/runs/...` 는 이제 `codesign/legacy/runs/...`.
- `code/`: 공통으로 쓰는 코드만 (motion_tracking, tt_dynamics, viz, env_osmesa.sh). 데이터 경로는 legacy/runs 를 가리킨다.
- `check/`: 새 점검 작업 (예: check/slide).
- **새 작업을 구현하기 전에 legacy/ 의 코드부터 찾아본다.** 있으면 고쳐 쓰고, 공통으로 쓰게 되면 code/ 로 옮긴다.

다른 PC 에서 이어받을 때 이 파일 → `legacy/remind.md` (§0 단계표 · §2 열린 안건 · TODO T8) → `stage2/README.md` 순서로 읽는다.

## 기록
- 정본은 `remind.md`. 결정 = §1 D 번호, 결과 = 해당 TODO 항목. **항목마다 결과를 먼저, 3 줄 이내**, 긴 산문 금지.
- 모든 실험은 고정 환경 + 조건 표를 명시한다 (§4). 비유 금지.
- cfg·기준·학습 시작/중단 결정은 사용자만 한다. 진단과 선택지 제시까지만 자동으로 한다.

## 학습 실행
- 모든 학습·평가는 tmux 에서 `bash ../scripts/tmux_run.sh <세션> "<명령>"` 로 돌린다 (nohup 금지). 시작·종료 텔레그램 알림은 자동이다.
- 알림은 한 줄만 (`<이름> 학습 완료`). 자격증명은 `~/.config/telegram_bot.env` (RL 봇). 비-RL 봇과 섞지 않는다.
- 여러 단계 명령은 `bash -c '...'` 한 덩어리로 넘긴다 (`;` 로 이으면 마지막 명령만 로그에 남는다).
- tmux 대기는 반드시 `tmux has-session -t "=이름"` (정확 일치). 접두 매칭 때문에 체인이 17 시간 멈춘 적이 있다.
- run 폴더 = `stage2/runs/T{TODO}/try{N}`, tmux 세션 = `T{TODO}_try{N}`.
- 학습 중에는 git 명령을 쓰지 않는다. 마지막 단계에서 기록한 뒤 한 번만 push 한다.

## 결과 보고
- 학습이 끝나면 run 폴더 하나에 `report.html` + `videos/` (task 수행 영상, `render_task_videos.py`) 를 한 세트로 저장하고, 영상까지 넣어 아티팩트로 올린다. 비교 대상 run 의 영상도 같은 `videos/` 에 복사한다. 실패한 run 은 원인 분석을 포함한다.
- 작업 폴더는 `codesign/` 하나만 쓴다 (`codesign_v100` 은 2026-11-10 통합, 사용 금지).
- 아티팩트 링크는 `stage2/README.md` 의 run 표에 추가한다.
- 평가는 DR 을 끈 고정 명령 7 개로 한다 (D51·D52). reward 수치보다 실제 이동·회전·생존을 먼저 본다.
