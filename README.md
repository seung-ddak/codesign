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
우리 : guidance = π_φ(x_t, k)           결과로부터 학습된 residual  (critic 으로 학습)
```

나머지(prior·샘플러·평가기·지표)는 전부 동일하게 두고 **guidance 를 만드는 주체만** 바꾼다.

## 폴더

| 경로 | 내용 |
|------|------|
| `remind.md` | **정본** — 결정(D1~D25) · 열린 안건 · 실험 결과(§5.1~5.24) |
| `code/` | Stage 1 (TT vs RL tracking guidance) — 회귀 검사용 보존 |
| `stage2/` | Stage 2 (동결 π_θ + π_φ 형상 생성) — T3 부터 작성 |
| `stage3_coevolution/` | Stage 3 공진화 코드 (보류). 실행법 `COEVOLUTION.md` |
| `archive/` | 폐기된 D22 reward · L0~L3 · π_θ fine-tune 스크립트 |
| `tt_patch/` | TT 레포에 얹을 파일들. 경로 구조 그대로. `train.py.diff` 는 기존 파일 수정분 |
| `results/` | 결과 jsonl·json, 비교 영상, 설계 렌더, 복원된 MJCF |
| `assets/` | 단일 설계 store (`*.zarr`) — π_θ 학습 입력 |

## 지금 상태 (2026-09-21)

**2026-09-22 갱신:** reference 없는 전진 보행 공진화 코드를 연결했다. 전체 실행 검증은 미완료이며 사용자 요청으로 검증을 중지하고 형상 bank 다운로드만 진행한다. 현재 실행 경로는 [`COEVOLUTION.md`](stage3_coevolution/COEVOLUTION.md).

- **1 단 (동등성)**: TT 행 재현 ✅ (Pos Err 3.0 cm / Survival 99.4 %, 논문 2.6/99.4).
  RL 행 완주 ✅ (3.0 cm / 97.8 %). 단 π_φ 가 학습되지 않아 **치환의 기계적 성립까지만** 확인.
  guidance 효과가 유의한 지점을 아직 못 찾았다 (§5.15·§5.16).
- **2 단 (확장)**: reward 확정 (D22 — 접촉 종료 + 1.0 m 전진). π_θ fine-tune 진행 중.

다음 할 일은 `remind.md` §7.

## 이어서 하기 (다른 컴퓨터)

```sh
git clone https://github.com/seung-ddak/codesign.git
cd codesign
source env.sh          # 경로 설정 — 스크립트를 고칠 필요 없다
less SETUP.md          # TT 레포 + 공개 자산 16 GB 받는 절차
```

이어받은 뒤 **가장 먼저 할 것**: `remind.md` §0(연구 구조 2 단) → §1(결정 D1~D25) → §7(다음 할 일).
배선이 살아 있는지는 `archive/eval_design_d22.py` 로 40 초면 확인된다.
