# codesign — 로봇의 몸을 AI 가 설계하게 하기 (2026-09-19 ~)

로봇을 만들 때 "어떤 몸을 만들지" 와 "어떻게 움직일지" 를 **함께** 설계하는 연구다.
출발점은 [Transformer Transformer](https://arxiv.org/abs/2607.25798) 논문이고,
그 논문이 **수식 미분으로** 하던 일을 **강화학습으로** 대체할 수 있는지를 확인한다.

**처음 읽는다면 [`OVERVIEW.md`](OVERVIEW.md) 부터.** 배경지식 없이 읽을 수 있게 썼다.
[`remind.md`](remind.md) 는 작업 기록 정본인데 내부용 속기라 용어가 압축돼 있다.

> 큰 자산(체크포인트 7.3 GB · RL expert 9.3 GB · 데이터 288 MB)은 **여기 없다.**
> 전부 공개 URL 에서 받을 수 있고, 절차는 [`SETUP.md`](SETUP.md) 에 있다.

## 한 줄 요약

```
TT  : guidance = scale · ∇_x R̂(x̂)      모델 예측 reward 를 미분   (학습 없음)
우리 : TT ψ ← DDPO(J(E, θ_fixed))       실제 rollout return으로 denoiser 학습
```

Stage 2에서는 `π_θ`를 동결해 TT `ψ`의 효과만 검증하고, 통과한 뒤 Stage 3에서 두 정책을 번갈아 갱신한다.

## 폴더

| 경로 | 내용 |
|------|------|
| `remind.md` | **정본** — 결정(D1~D53) · 열린 안건 · 실험 결과 · TODO |
| `CLAUDE.md` | 작업 규칙 (tmux·알림·기록·보고서) |
| `code/` | Stage 1 (TT vs RL tracking guidance) — 회귀 검사용 보존 |
| `stage2/` | Stage 2 (동결 π_θ + TT ψ DDPO). 코드 = `stage2/code/`, 결과 = `stage2/runs/`. run 별 보고서·아티팩트 링크는 `stage2/README.md` |
| `stage3/` | **Stage 2 성공 후의 hardware–policy co-design**. 코드 = `stage3/code/`, 결과 = `stage3/runs/`, 실행법 = `stage3/README.md` |
| `archive/` | 폐기된 D22 reward · L0~L3 · π_θ fine-tune 스크립트. `tt_patch_stairs_L0L3/` = 계단(D16)·L0~L3(D17) TT 패치 |
| `tt_patch/` | TT 레포에 얹을 파일 (경로 구조 그대로): `rl_guidance.py`·`rl_residual.yaml` (Stage 1 RL guidance) + `tt_compat.diff` (기존 파일 수정분) |
| `results/` | 결과 jsonl·json, 비교 영상, 설계 렌더, 복원된 MJCF |
| `assets/` | 단일 설계 store (`*.zarr`) — π_θ 학습 입력 |
| `rl_scripts/` | tmux 런처·텔레그램 전송기. 다른 PC 에선 codesign 옆 `scripts/` 로 복사 (SETUP §7) |

## 지금 상태 (2026-09-29)

**Stage 2 진행 중.** T5·T6 완료 후 T7 DDPO 전체 smoke를 준비 중이다. Stage 3 co-design은 아직 시작하지 않았다. 상세 `stage2/README.md`.

- **Stage 1 (완료, 09-25)**: TT 행 재현 (Pos Err 3.0 cm / Survival 99.4 %), RL guidance 치환 동작 확인. 코드 `code/`.
- **Stage 3 (보류)**: Stage 2에서 DDPO 효과가 확인된 뒤 `ψ`와 `π_θ`를 번갈아 갱신하는 co-design.

다음 할 일은 `remind.md`의 TODO와 "현재 우선순위".

## 이어서 하기 (다른 컴퓨터)

```sh
git clone https://github.com/seung-ddak/codesign.git
cd codesign
source env.sh          # 경로 설정 — 스크립트를 고칠 필요 없다
less SETUP.md          # TT 레포 + 공개 자산 16 GB 받는 절차
```

이어받은 뒤 **가장 먼저 할 것**: `CLAUDE.md` → `remind.md` §0 · §1 · TODO → `stage2/README.md`. Stage 2 배치·점검은 SETUP §7.
Stage 1 배선은 `archive/eval_design_d22.py` (40 초), Stage 2 는 SETUP §7 점검 명령으로 확인한다.
