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

# 기존 파일 수정분 (jax 0.10 호환 등)
cd "$TT" && git apply "$CODESIGN/tt_patch/train.py.diff"
```

`code/` 의 스크립트들은 경로를 절대경로로 들고 있다. 머신이 다르면 각 파일 상단의
`TT = "..."`, `OUT = "..."`, `CODE = "..."` 를 새 경로로 바꾼다.

## 5. 동작 확인

```sh
cd "$TT"
# ① TT 행 재현 — Pos Err 2.6~3.0 cm / Survival 99.4 % 근처면 배선 정상 (약 55 분)
bash "$CODESIGN/code/repro_table4_TT.sh" 16

# ② 2 단 reward 채점기 — 설계 2 개를 굴려 score 를 낸다 (약 40 초)
.venv/bin/python "$CODESIGN/code/eval_design_d22.py"
```

---

## 실행 규칙

모든 학습·평가는 **tmux** 에서 돌린다 (nohup 금지). 기존 머신에서는
`bash scripts/tmux_run.sh <세션> "<명령>"` 을 썼다.

## 주요 스크립트

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
