# π_θ fine-tune (RL 형상) — 2026-09-21 13:35~14:26

`design_RL_best.zarr` 에 출고 expert(`4006084608`) 를 warm start 시켜 40 M step fine-tune.
**40 M 완주 후 eval 영상 렌더 단계에서 X11 끊김으로 프로세스 사망** (학습 손실은 없음).

명령: `code/finetune_pi_theta.sh assets/design_RL_best.zarr <expert_ckpt> 40000000`
원본 run: `transformer-transformer/wandb/offline-run-20260921_133548-xvz1emr3`

## 결과 — baseline 보다 나아진 epoch 이 하나도 없다

| step | pos_err ↓ | orn_err ↓ | ep_len |
|---|---|---|---|
| 0 (= 출고 expert) | **0.323** | **0.563** | 772 |
| 4.5 M | 1.047 | 2.078 | 920 |
| 9.0 M | 0.697 | 1.339 | 365 |
| 13.6 M | 0.487 | 1.689 | 110 |
| 18.1 M | 0.885 | 2.866 | 1754 |
| 22.6 M | 0.551 | 1.331 | 28 |
| 27.1 M | 0.956 | 2.887 | 2000 |
| 31.7 M | 0.429 | 1.855 | 20 |
| 36.2 M | 1.263 | 2.792 | 1613 |
| 40.7 M | 0.594 | 2.529 | 41 |

## 읽는 법 — pos_err 만 보면 속는다

ep_len 이 **1754 ↔ 28 ↔ 2000 ↔ 20** 으로 진폭 100 배 진동한다. 그리고 **ep_len 이 짧은 epoch 일수록
pos_err 이 낮다** (20 step → 0.429, 2000 step → 0.956). 즉 낮은 pos_err 은 추종을 잘해서가 아니라
**드리프트가 쌓이기 전에 죽어서** 나온 값이다. 두 지표를 같이 봐야 한다.

orn_err 은 baseline 0.563 에서 전 구간 1.3~2.9 로 **한 번도 회복하지 않았다** — 이쪽이 진짜 신호다.

## 체크포인트

`checkpoint_40697856/` = 40.7 M step 시점 (2 MB, orbax). 이어하기는 `SETUP.md` §6.

## 남은 의심 (다음 머신에서 볼 것)

1. warm start 한 expert 는 **그 choice 의 연속 변형 2 만 개를 DR 로 학습한 범용 정책**인데,
   단일 형상에 40 M 를 더 태우자 오히려 무너졌다 → fine-tune LR 이 expert 학습 LR 그대로일 가능성.
2. `terminate_on_bad_contact=true` + `fellover 0.1` 종료가 warm start 직후부터 터져
   **조기종료 rollout 으로만 gradient 가 만들어지는 구간**이 있었을 수 있다 (ep_len 20~41).
3. reward −3699 (36.2 M) 같은 outlier epoch 의 decomp 는 원본 wandb run 에 남아 있다.
