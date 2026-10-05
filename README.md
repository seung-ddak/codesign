# codesign — 로봇의 몸을 AI 가 설계하게 하기 (2026-09-19 ~)

로봇을 만들 때 "어떤 몸을 만들지" 와 "어떻게 움직일지" 를 **함께** 설계하는 연구다.
출발점은 [Transformer Transformer](https://arxiv.org/abs/2607.25798) 논문이고,
그 논문의 reward-gradient guidance 를 **로봇이 실제로 움직인 데이터(형상·명령·상태·행동)** 로 학습한 TT 위에서 다시 쓰는 것이 현재 방향이다 (D63).

[`remind.md`](remind.md) 가 작업 기록 정본이다 (§0 = 현재 방향과 단계표).

> 큰 자산(체크포인트 7.3 GB · RL expert 9.3 GB · 데이터 288 MB)은 **여기 없다.**
> 전부 공개 URL 에서 받을 수 있고, 절차는 [`SETUP.md`](SETUP.md) 에 있다.

## 한 줄 요약

```
TT 원본 : 형상 + UMI 추종 데이터로 학습, guidance = ∇ R̂(형상)
우리     : 형상 + command + π_θ rollout(state·action) 으로 학습, 새 task reward 의 ∇ 를 형상 토큰에 적용
```

동결 π_θ 로 데이터를 모으고(R1) TT 예측이 실제 return 을 맞히는지 확인(R2)한 뒤 guidance 를 건다. 모든 결과는 MJX 로 재채점한다.

## 폴더

| 경로 | 내용 |
|------|------|
| `remind.md` | **정본** — 현재 유효한 결정 · 열린 안건 · TODO (폐기 이력은 `archive/remind_history.md`) |
| `CLAUDE.md` | 작업 규칙 (tmux·알림·기록·보고서) |
| `stage2/` | 현재 코드·결과. 동결 π_θ (T5·T6) · T8 TT guidance. 코드 = `stage2/code/`, 결과 = `stage2/runs/`, run 표 = `stage2/README.md` |
| `artifacts/` | 작업별 Archify 다이어그램 + README (D55) |
| `tt_patch/` | TT 레포에 얹을 파일 (경로 구조 그대로) + `tt_compat.diff` (jax 0.10 호환) |
| `rl_scripts/` | tmux 런처·텔레그램 전송기. 다른 PC 에선 codesign 옆 `scripts/` 로 복사 (SETUP §6) |
| `archive/` | 종료된 경로: `stage1/` (Stage 1 코드·결과·자산) · `stage3_coevolution/` (구 공진화) · `t7/` (RL 형상 생성) · `remind_history.md` |

## 지금 상태 (2026-10-05)

**D63 방향 전환.** T7 의 RL 형상 생성(DDPO 등)을 종료하고, TT reward-gradient guidance 로 간다. 다음 작업 = T8 R1 데이터 수집.

- **Stage 1 (완료, 09-25)**: TT 행 재현 (Pos Err 3.0 cm / Survival 99.4 %), RL guidance 치환 동작 확인. `archive/stage1/`.
- **T5·T6 (완료, 09-29)**: 형상 조건부 보행 π_θ 동결, 구체 형상 채점기로 검증 (반복 순위상관 0.913).

다음 할 일은 `remind.md` §2 열린 안건과 TODO T8.

## 이어서 하기 (다른 컴퓨터)

```sh
git clone https://github.com/seung-ddak/codesign.git
cd codesign
source env.sh          # 경로 설정 — 스크립트를 고칠 필요 없다
less SETUP.md          # TT 레포 + 공개 자산 16 GB 받는 절차
```

이어받은 뒤 **가장 먼저 할 것**: `CLAUDE.md` → `remind.md` §0 · §2 · TODO T8 → `stage2/README.md`. 배치·점검은 SETUP §6.
