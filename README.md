# codesign — TT 확장 연구 (2026-09-19 ~)

Transformer-Transformer(TT)의 **형상 생성 guidance 를 gradient 에서 RL 로 치환**하는 연구의
코드·기록 저장소. 연구 결정과 실험 결과의 **정본은 [`remind.md`](remind.md)** 이고,
이 저장소는 그것을 재현할 수 있는 최소 집합이다.

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
| `code/` | 우리가 쓴 스크립트 (학습·평가·진단·렌더) |
| `tt_patch/` | TT 레포에 얹을 파일들. 경로 구조 그대로. `train.py.diff` 는 기존 파일 수정분 |
| `results/` | 결과 jsonl·json, 비교 영상, 설계 렌더, 복원된 MJCF |
| `assets/` | 단일 설계 store (`*.zarr`) — π_θ 학습 입력 |

## 지금 상태 (2026-09-21)

- **1 단 (동등성)**: TT 행 재현 ✅ (Pos Err 3.0 cm / Survival 99.4 %, 논문 2.6/99.4).
  RL 행 완주 ✅ (3.0 cm / 97.8 %). 단 π_φ 가 학습되지 않아 **치환의 기계적 성립까지만** 확인.
  guidance 효과가 유의한 지점을 아직 못 찾았다 (§5.15·§5.16).
- **2 단 (확장)**: reward 확정 (D22 — 접촉 종료 + 1.0 m 전진). π_θ fine-tune 진행 중.

다음 할 일은 `remind.md` §7.

## 이어서 하기

```sh
git clone https://github.com/seung-ddak/codesign.git
cd codesign && less SETUP.md
```
