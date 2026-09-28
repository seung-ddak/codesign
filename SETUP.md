# 다른 컴퓨터에서 이어하기

전제: NVIDIA GPU 1 장 + CUDA 12.x. V100 이면 `use_flash_attn=false use_mixed_precision=false
use_torch_compile=false` 가 **필수**다 (sm_80+ 전용 커널·bf16 없음). 그 이상 세대면 켜도 된다.

디스크는 약 **25 GB** 필요하다 (checkpoints 7.3 + rl_policies 10 + data 0.3 + 여유).

---

## 1. TT 레포

```sh
git clone https://github.com/seung-ddak/transformer-transformer.git
cd transformer-transformer
uv sync                     # .venv 생성 (python 3.12)
```

## 2. 공개 자산 3 종

```sh
wget -qO- https://real.stanford.edu/transformer-transformer/checkpoints.zip | bsdtar -xvf- -C ./
wget -qO- https://real.stanford.edu/transformer-transformer/data.zip        | bsdtar -xvf- -C ./
wget -qO- https://real.stanford.edu/transformer-transformer/rl_policies.zip | bsdtar -xvf- -C ./
```

| 아카이브 | 크기 | 왜 필요한가 |
|---|---|---|
| `checkpoints.zip` | 6.8 GB | 형상 생성 모델 `xrh4wk4l/040.pt`, ctrl `d7lra7j0/040.pt` |
| `data.zip` | 0.2 GB | UMI 20-skill 궤적 (학습/평가 분리) |
| `rl_policies.zip` | **9.3 GB** | **128 개 RL expert** + `choice_to_ckpt_path.json` + options/robotokens zarr. 이게 없으면 `evaluate_hardware_opt_rl.py` 가 `FileNotFoundError` 로 죽는다 |

`blender_templates.zip` 은 시각화 전용이라 불필요.

## 3. 보행 궤적 (우리가 합성한 것)

`data/flat_walk_long.pkl` — 팔 끝점 목표를 +x 로 등속 이동시키는 합성 궤적 3 개
(traj 0/1/2 = 1 m / 2 m / 3 m 이동). **이 저장소에 없으면** `code/` 의 생성 절차를 참고해
다시 만들거나, 기존 머신에서 복사한다 (384 KB).

## 4. 우리 코드 얹기

```sh
CODESIGN=/path/to/codesign
TT=/path/to/transformer-transformer

# 새 파일들 (경로 구조 그대로)
cp -r "$CODESIGN/tt_patch/t2"      "$TT/"
cp -r "$CODESIGN/tt_patch/config"  "$TT/"
cp -r "$CODESIGN/tt_patch/scripts" "$TT/"

# 기존 파일 수정분 (jax 0.10 호환 — train.py + track_env.py 2 곳)
cd "$TT" && git apply "$CODESIGN/tt_patch/tt_compat.diff"
```

경로는 **환경변수로 잡는다** — 파일을 고칠 필요 없다.

```sh
cd "$CODESIGN" && source env.sh          # CODESIGN_ROOT / CODESIGN_TT 설정
```

기본값은 "이 저장소가 `<어디든>/codesign`, TT 레포가 그 옆 `<어디든>/transformer-transformer`"
다. 다르게 두려면 `export CODESIGN_TT=/실제/경로` 를 먼저 하면 된다.

## 5. 동작 확인

```sh
cd "$CODESIGN" && source env.sh && cd "$CODESIGN_TT"
# ① TT 행 재현 — Pos Err 2.6~3.0 cm / Survival 99.4 % 근처면 배선 정상 (약 55 분)
bash "$CODESIGN/code/repro_table4_TT.sh" 16

# ② 2 단 reward 채점기 — 설계 2 개를 굴려 score 를 낸다 (약 40 초)
.venv/bin/python "$CODESIGN/archive/eval_design_d22.py"
```

## 6. 끊긴 π_θ fine-tune 이어하기 (2026-09-21)

`runs/pi_theta_RL4/checkpoint_40697856/` 가 **40 M step 까지 돈 결과물**이다 (2 MB, orbax).
`runs/pi_theta_RL4/README.md` 에 epoch 별 수치가 있다. 이어 돌리려면:

```sh
source env.sh
MUJOCO_GL=egl bash "$CODESIGN/archive/finetune_pi_theta.sh" \
  "$CODESIGN/assets/design_RL_best.zarr" \
  "$CODESIGN_ROOT/runs/pi_theta_RL4/checkpoint_40697856" 40000000
```

처음부터(출고 expert) 다시 할 거면 두 번째 인자를
`$CODESIGN_TT/umi_on_legs_plus_plus_rl_policies_11nov2025/run-20251110_145212-z08g0hc7/files/checkpoints/4006084608`
로 준다. **단 40 M 결과가 baseline 보다 나빴다** — remind.md §5.25 를 먼저 읽을 것.

## 7. Stage 2 이어하기 (2026-09-28)

폴더 배치 — 스크립트가 서로를 상대경로로 찾으므로 **세 개를 같은 부모 아래** 둔다.

```
<ROOT>/codesign/                 # 이 저장소
<ROOT>/transformer-transformer/  # §1 (+ §2 자산, §4 패치)
<ROOT>/scripts/                  # cp -r codesign/rl_scripts <ROOT>/scripts
```

- 텔레그램 알림을 쓰려면 `~/.config/telegram_bot.env` 에 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 를 넣는다 (git 에 없음). 없으면 `TMUX_RUN_NOTIFY=0`.
- 동결 π_θ 체크포인트(T5_try1, t5_norail_x3_1000it 마지막 step)는 git 에 들어 있다. 다른 step 은 없다.
- 동작 확인 (codesign/ 에서):

```sh
P=../transformer-transformer/.venv/bin/python
$P stage2/code/t6_evaluator_check.py t5 --run stage2/runs/T5_try1   # 수 초, exit 3 = 기준 미달 (정상 판정)
MUJOCO_GL=egl $P stage2/code/test_locomotion_env.py                  # env·reward·종료
MUJOCO_GL=egl $P stage2/code/test_multi_design_env.py                # 구조 그룹 vmap
```

이후 실행법과 run 목록은 `stage2/README.md`, 작업 규칙은 `CLAUDE.md`.

---

## 실행 규칙

모든 학습·평가는 **tmux** 에서 돌린다 (nohup 금지). 기존 머신에서는
`bash ../scripts/tmux_run.sh <세션> "<명령>"` (§7 배치).

## 주요 스크립트

`env.sh` 를 source 한 뒤에는 `code/` 의 스크립트를 어느 디렉터리에서 호출해도 된다.

| 파일 | 하는 일 |
|---|---|
| `repro_table4_TT.sh` / `repro_table4_RL.sh` | 논문 Table 4 의 TT 행 / RL 행 |
| `train_pi_phi_critic.py` | π_φ 학습 (critic, `∇_a Q`) |
| `guidance_effect.py` | guidance 효과 크기 쌍대 측정 |
| `walk_reward.py` | **2 단 reward** (접촉 종료 + 전진 진척도) |
| `eval_design_d22.py` | 설계 하나를 expert 로 굴려 채점 |
| `extract_best_designs.py` → `enumerate_design_store.py` → `finetune_pi_theta.sh` | 생성 설계 → 단일 설계 store → π_θ fine-tune |
| `compare_tt_rl_sim.py` | TT/RL 형상을 한 화면에서 비교 (색 구분·깊이 합성) |
| `contact_audit.py` · `posture_audit.py` | 접촉·자세 진단 |

## 함정 (겪은 것들)

- `evaluate_hardware_opt_rl.py` 는 **repo 루트**에서 `choice_to_ckpt_path.json` 과
  `umi_on_legs_plus_plus_{options,robotokens}/` 를 찾는다. 다른 데 두면 죽는다.
- hydra 에서 `train.restore_checkpoint_path` 는 **`+` 접두사**가 필요하다 (구조체에 없는 키).
- `RLEnvRunner` 리셋 순서: `env.reset_hardware()` → `runner.post_hardware_reset()`.
  뒤집으면 `env.m` 이 없어서 죽는다.
- MuJoCo 오프스크린 렌더는 모델 XML 의 `offwidth`(기본 640)보다 크면 죽는다 →
  컴파일 후 `m.vis.global_.offwidth` 를 직접 키운다.
- 로봇 링크 캡슐이 `group 3` 이라 `mjvOption.geomgroup` 기본값으로는 **안 보인다** → 전부 1 로.
- jax 0.10 에서 `jnp.clip(..., a_min=)` 와 `jax.device_put_replicated` 가 **없어졌다** →
  `tt_compat.diff` 가 두 곳을 고친다. 안 넣으면 학습이 첫 step 에서 `TypeError` 로 죽는다.
- 학습 중 eval 영상 렌더가 **X11 로 나간다**. SSH X11 forwarding 이 끊기면
  `XIO: fatal IO error on X server "localhost:10.0"` 로 **학습째로 죽는다** (실제로 9/21 에 당했다).
  → 학습 명령 앞에 **`MUJOCO_GL=egl`** 를 붙여 headless 로 고정한다.
