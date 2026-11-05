# V100 32GB：48시간 공용 PPO 학습·rollout 수집

**코드를 읽지 않고 아래 명령으로 실행할 수 있습니다.** 원본 모션 10,000개 전체, 학습 형상 200개(실제 길이 ±20%), 별도 검증 형상 20개를 사용합니다. 공용 PPO 하나를 학습합니다.

목표는 **20,000개 원본·형상 쌍 × 서로 다른 시작 조건 2회 = rollout 40,000개**입니다. 각 원본을 서로 다른 학습 형상 2개에 배정합니다. 목표 수량과 실제 확보량을 대시보드에서 구분합니다.

## 1. 새 머신에 복원

Linux x86_64, **Python3.12 이상**, NVIDIA V100 32GB 한 장, 드라이버 525 이상, 권장 여유 디스크 100GB 이상이 필요합니다. 먼저 `nvidia-smi`로 확인하세요. 원격 머신 계정·주소는 본인의 값을 사용합니다.

```bash
mkdir -p ~/quad_ws
cd ~/quad_ws
git clone --depth 1 --branch cleanup/shared-ppo https://github.com/seung-ddak/codesign.git
cd codesign
python3.12 code/portable_workspace.py restore

cd ../transformer-transformer
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r ../codesign/portability/requirements-current.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install --no-deps -e .
cd ../codesign
```

복원 묶음에는 수정된 transformer-transformer 코드도 포함됩니다. **이 경로에서는 `tt_local.patch`를 다시 적용하지 않습니다.** 기존 머신 파일이 다르면 restore가 덮어쓰지 않고 중단하므로 새 `quad_ws`를 사용하세요.

**V100은 CUDA12용 JAX를 사용합니다.** `jax[cuda13]`으로 바꾸지 마세요. JAX 공식 문서상 CUDA 13은 SM 7.5 이상을 지원하므로 V100에 맞지 않습니다. 참고: https://docs.jax.dev/en/latest/installation.html

## 2. 사용할 Python 지정과 GPU 검사

아래 명령은 새 터미널에서도 실행하세요. GPU 번호 0은 원하는 V100 번호로 바꿀 수 있습니다. 한 장만 선택합니다.

```bash
cd ~/quad_ws/codesign
export PYTHON="$PWD/../transformer-transformer/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/calm-v100-mpl

"$PYTHON" -c 'import jax,mujoco; print("JAX 장치:",jax.devices()); print("MuJoCo:",mujoco.__version__)'
"$PYTHON" code/motion_tracking/shared_calm.py preflight
```

`CudaDevice`가 보여야합니다. `CpuDevice`만 보이거나 preflight가 실패하면 CUDA 12 환경과 드라이버를 확인하세요. 패키지 설치·데이터 복원은 48시간 예산에 포함하지 않습니다. V100에서의 실제 실행은 로컬에서 검증할 수 없으므로 첫 실행의 메모리와 속도를 확인해야 합니다.

## 3. 데이터 배정표 생성 — 처음 한 번

```bash
"$PYTHON" code/motion_tracking/shared_calm.py init \
  --config configs/shared_calm_v100_48h.yaml \
  --out runs/shared_calm_v100_48h
```

원본 경로가 다르면 `--source '/실제경로/calm_motion_10,000/dataset'`를 추가합니다. init은 모션 해시와 형상 배정표를 확정합니다. 이미 plan.json이 있으면 run으로 이어갑니다.

## 4. 수집 시작 — SSH를 닫아도 계속 실행

```bash
nohup "$PYTHON" -u code/motion_tracking/shared_calm.py run \
  --out runs/shared_calm_v100_48h \
  > runs/shared_calm_v100_48h/console.log 2>&1 < /dev/null &
echo $! > runs/shared_calm_v100_48h/runner.pid
```

순서는 **리타겟팅 → 공용 PPO 학습 → 실제 물리 rollout → 검증 형상 평가·파일 검사**입니다. 형상 8개씩 묶어 전체를 2회 순회합니다. 정책·가치망·관측 정규화·Adam 상태·누적 학습 전이를 이어갑니다. 환경 상태와 난수까지 같은 시점으로 복원하는 방식은 아닙니다.

설정은 병렬 환경 128개, batch 128, 제어 30Hz, 물리 240Hz, 관절 속도 추종 가중치 1.0, 학습 목표 8천만 전이, rollout 32개씩, JAX 선할당 해제·풀 60%입니다. 학습 형상 200개는 기준 1 + 단일 변화 40 + 두 토큰 극단 40 + 전체 극단 32 + 내부 복합 87입니다. 나머지 8개 토큰은 고정합니다.

## 5. 한국어 실시간 그래프 열기

V100 머신에서 별도 터미널로 실행합니다(`PYTHON`은2단계처럼 지정).

```bash
"$PYTHON" code/motion_tracking/shared_calm_dashboard.py serve \
  --out runs/shared_calm_v100_48h --port 8765
```

**본인 PC 터미널**에서 SSH 포트를 연결합니다. 아래 계정·머신 주소를 교체하세요.

```bash
ssh -N -L 8765:127.0.0.1:8765 사용자계정@V100머신주소
```

본인 PC 브라우저에서 **http://127.0.0.1:8765** 를 여세요. 10초마다 새로고침하며 인터넷 CDN이 필요 없습니다. 서버는 127.0.0.1에서만 열립니다.

학습 중에는 실제 에피소드 보상과 손실을 기록합니다. 별도 검증 형상 20개의 평가는 마지막 검증 단계에서 실행하므로, 그 전에는 검증 결과 표가 비어 있습니다.

### 그래프 읽는 법

| 화면 | 확인할 내용 |
|---|---|
| 누적학습진행 | 학습 전이가 늘면 실제 학습 중. 묶음 교체·컴파일 때 잠시 정지 가능 |
| 평균에피소드보상·관절자세/속도보상 | 같은 조건에서 증가하는지 확인. 모션과 에피소드 길이 변경 영향도 있음 |
| 정책손실·가치손실·탐색정도 | 실제 PPO 출력값. 낮아졌다는 이유만으로 과업 성공이라고 판정하지 않음 |
| 초당학습전이 | 실제 학습 구간 속도. 컴파일·리타겟팅 시간과 다름 |
| GPU메모리·활용률 | VRAM 여유와 실제 계산 여부. 메모리 예약량만으로 진전 판단 불가 |
| 형상별낙상률·관절/몸통오차 | 실패가 특정 형상에 집중되는지 |
| 수집모션커버리지·K분포 | 10,000개 모션 모두 유효 전이가 기록됐는지 |
| 실제속도·회전·높이분포 | 정지·유사 상태에 편중되지 않는지 |
| 검증형상결과 | 학습에 없는 20개 형상의 성능. 원본 모션은 학습했으므로 미학습 모션 평가는 아님 |

아직 측정되지 않은 그래프는 **아직 데이터 없음**으로 표시합니다. 손실이 출력되기 전에는 현재 단계·마지막 지표 시각·로그를 함께 확인하세요. 처음 컴파일은 수분 이상 걸릴 수 있습니다.

실시간 서버를 열 수 없으면 `runs/shared_calm_v100_48h/report.html`을 본인 PC로 복사해 브라우저로 여세요. 실행 중 약 1분 간격으로 갱신되는 스냅샷이며 복사한 뒤에는 자동 갱신되지 않습니다. 수동 갱신 명령:

```bash
"$PYTHON" code/motion_tracking/shared_calm_dashboard.py report --out runs/shared_calm_v100_48h
```

## 6. 로그·일시 중지·재개

```bash
# 현재단계와마지막상태
"$PYTHON" code/motion_tracking/shared_calm.py status --out runs/shared_calm_v100_48h

# 전체오류/실행로그
tail -f runs/shared_calm_v100_48h/console.log

# 현재학습방문번호에맞는로그선택 (예: 첫번째묶음)
tail -f runs/shared_calm_v100_48h/logs/train_000.log

# 일시중지 — 이실험의저장된PID만종료
kill -TERM "$(cat runs/shared_calm_v100_48h/runner.pid)"
```

저장된 체크포인트까지 복원하며 이후 전이는 다시 계산합니다. 재개는 4단계의 동일한 nohup 명령을 실행하세요. 동시 실행은 잠금 파일로 거부됩니다. **48시간은 최초 run부터의 실제 경과 시간이며 중지 시간도 포함**합니다. 만료되면 새 실행 폴더와 새 예산을 사용해야 합니다.

## 7. 완료·미완료 판정과 TT용 파일

48시간을 준비 8시간·학습 20시간·수집 18시간·검증 2시간으로 나눕니다. 앞 단계가 일찍 끝나면 뒤 단계가 남은 시간을 사용합니다. 저장·종료 정리 시간은 추가될 수 있습니다. 시간이 부족하면 **budget_finished_incomplete** 상태와 누락 목록을 남깁니다. 개별 작업 오류나 낙상률 50% 초과 형상도 기록하고 가능한 다음 작업으로 진행합니다. 실패의 유효 전이도 보존합니다.

| 파일·폴더 | 내용 |
|---|---|
| state.json | 현재단계·전체상태·최종요약 |
| metrics.jsonl | 누적step·실제PPO손실·보상·속도 |
| learner.msgpack / learner.json | 재개용전체학습기상태 / 누적step |
| checkpoint.json | 현재공유정책체크포인트위치 |
| training_coverage.json | 실제RSI시작모션·형상노출수 |
| hardware.jsonl | 5초간격VRAM·GPU활용률·디스크여유 |
| rollouts/형상/variant_0.npz | 모션첫상태에서실행한실제궤적 |
| rollouts/형상/variant_1.npz | 아직 기록되지 않은 동작 구간 우선 RSI + 작은 관절 상태 변화 궤적 |
| evaluation/ | 검증형상결과 |
| audit.json | 실제유효전이·원본커버리지·누락·형식오류 |
| report.html | 한국어오프라인진행·결과보고서 |

데이터 검사를 수동으로 다시 실행:

```bash
"$PYTHON" code/motion_tracking/shared_calm.py audit --out runs/shared_calm_v100_48h
```

NPZ의 `qpos/qvel/contact`는 T+1개 상태, `action/target_q/valid`는 T개 전이입니다. `valid=false` 패딩은 TT 학습에서 제외하세요. `group_id`는 원본 모션 해시입니다. 시작 조건 번호는 `start_variant`로 따로 저장해 행동 개입 쌍으로 오인하지 않도록 합니다. 형상과 시작 조건이 달라도 동일하므로 같은 원본이 TT train/val에 섞이지 않도록 분리할 수 있습니다. 목표 추종에 실패해도 물리적으로 유효한 전이일 수 있으나, 낙상 근처 상태에 편중되지 않았는지 확인해야 합니다.

**40,000개는 목표이며 48시간 내 달성이나 TT 품질을 보장하지 않습니다.** 원본 커버리지 100%, 원본당 서로 다른 2개 형상, 쌍마다 2개 시작 조건, 실제 상태 분포와 검증 형상 성능을 함께 확인하세요. 이전 모션을 잊을 가능성을 줄이기 위해 두 번 순회하지만 최종 검증이 필요합니다. 다른 아키텍처로 학습 상태를 복원할 때의 호환성도 확인해야 합니다.

## 전체 수집 전 짧은 실행 검사

환경 설치와 GPU 검사가 끝난 뒤 다음 파일럿을 먼저 실행할 수 있습니다. **원본 4개·학습 형상 2개만 쓰는 기능 검사이며 본 수집 데이터가 아닙니다.** 별도 폴더에 저장하며 정상 완료 후 48시간용 run을 실행하세요.

```bash
"$PYTHON" code/motion_tracking/shared_calm.py init \
  --config configs/shared_calm_v100_pilot.yaml --out runs/shared_calm_v100_pilot
"$PYTHON" -u code/motion_tracking/shared_calm.py run --out runs/shared_calm_v100_pilot
```

모든 모션의 정밀 추종 수렴을 평가하는 검사가 아닙니다. 모델 구성, 전체 학습 상태 저장·복원, 실제 물리 rollout 저장 경로가 정상인지 확인하는 용도입니다.

## 모션 이름만으로 다양성을 판단하지 마세요

원본 ID가 10,000개 모두 있어도 매번 처음에 넘어지면 실제 동작 데이터는 부족할 수 있습니다. 대시보드는 원본 ID 커버리지와 **목표 모션 시간 구간 커버리지**를 따로 보여 줍니다. 후자는 여러 실행에서 기록된 유효 구간을 합쳐 반복 부분을 한 번만 계산합니다. 동작 종류별 유효 전이 그래프도 확인하세요.

두 번째 시작 조건은 기존 수집 결과에서 아직 기록되지 않은 동작 구간을 우선 선택하고, 그 구간의 reference 상태에서 작은 관절 상태 변화를 주어 실행합니다. 목표 동작 라벨은 정확한 수행 성공을 뜻하지 않으므로 실제 이동·회전·높이·관절 속도·접지·행동 분포와 함께 확인해야 합니다.

`complete`는 배정된 실행 조건과 학습량·원본 노출·파일 검사를 충족했다는 뜻이며, 정밀 추종이나 충분한 TT 품질을 보장하지 않습니다. 형상별 `collection_pass`와 실제 목표 시간 커버리지도 확인하세요.

TT 로더는 유효 전이가 2개 미만인 episode를 사용하지 않습니다. 원본 커버리지도 이 기준으로 계산합니다. 1개 전이만 기록된 실패도 원본 파일에는 보존하지만 충분한 원본 확보로 세지 않습니다. 최종 audit에는 TT 사용 가능 episode 수와 1초 이상 전이가 기록된 episode 수도 남깁니다.
