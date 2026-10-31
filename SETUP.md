# 다른 PC에서 작업 이어받기

현재 브랜치의 코드·실험 결과 외에 `portability/`에 학습 체크포인트, 수정된 transformer-transformer 실행 코드, A1 CALM 코드·정책·모델 자산, 필요한 로컬 데이터를 함께 보관했다. 분할 파일은 최대 **90 MiB**이며 SHA-256으로 검사한다. GitHub의 파일당 100 MB 제한은 커밋을 나누는 것으로 해결되지 않으므로 파일 자체를 나눴다.

## 1. 복제·파일 복원

Linux PC에서 아래 순서로 실행한다. 기존 작업 폴더와 충돌하지 않도록 새 폴더를 사용한다.

```bash
mkdir -p ~/quad_ws
cd ~/quad_ws
git clone --branch experiment/calm-tt-e000-direct-mjx https://github.com/seung-ddak/codesign.git
cd codesign
python3 code/portable_workspace.py restore
```

`../transformer-transformer`, `../external/a1-calm-main`, Git에서 제외되었던 체크포인트가 원래 상대 위치로 복원된다. 기존 파일 내용이 다르면 덮어쓰지 않고 중단한다. `restore`는 Python 3.12 이상을 사용한다. 원본 PC에서 복원 명령을 다시 실행할 필요는 없다.

## 2. Python 환경과 경로

Python 환경은 PC별로 새로 만든다. NVIDIA GPU 드라이버는 새 PC에 별도로 설치되어 있어야 한다.

```bash
cd ~/quad_ws/transformer-transformer
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r ../codesign/portability/requirements-current.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install --no-deps -e .
cd ../codesign
../transformer-transformer/.venv/bin/python code/portable_workspace.py rebase
source env.sh
```

`rebase`는 원본 작업 경로를 새 PC 경로로 바꾼다. 보고서 영상의 절대 경로, 설정 파일, 신뢰하는 프로젝트의 `config.pkl`도 함께 갱신한다. 복원·환경 설치 뒤 실행한다. 이 과정으로 생긴 경로 변경은 새 PC의 로컬 변경사항이다. GPU 종류·드라이버에 따라 설치 호환성은 별도로 확인해야 한다.

```bash
MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python -c \
  'import jax, mujoco; print(jax.devices()); print(mujoco.__version__)'
```

공유 PPO 실행 방법은 [SHARED_PPO.md](docs/SHARED_PPO.md)를 따른다. 체크포인트 파일의 보존과 optimizer/환경까지 완전히 동일한 학습 재개는 별개다. 재개 방법은 해당 실험 설정을 따른다.

## 포함 범위와 제외 범위

- **Git 일반 파일:** 코드·설정·레퍼런스·실험 측정값·그래프·보고서·영상 및 아직 미추적이던 실험 파일.
- **분할 묶음:** Git ignore 때문에 빠져 있던 학습 파일, 외부 실행 코드의 로컬 수정 내용, 모델 자산과 의존 데이터. 정확한 파일 목록은 `portability/manifest.json`에 있다.
- **제외:** `.git`, 가상환경, 컴파일 캐시, 임시 파일, 재생성 가능한 `.mjb`, 중복 ZIP, W&B 서비스 로그·미디어 사본, 자격증명. 기존 원본·결과 파일은 삭제하지 않았다.
- **별도 다운로드 자료 제외:** transformer-transformer의 공개 사전학습 `checkpoints/` 및 `umi_on_legs_plus_plus_*` 대형 다운로드 데이터. 현재 CALM 공유 PPO와 codesign Dynamics 실행에서 사용하지 않는 자료다. 이 자료를 사용하는 별도 실험은 해당 다운로드가 추가로 필요하다.

새 PC의 실제 환경 설치·장시간 GPU 학습은 원본 PC에서 검증할 수 없다. 분할본 해시·전체 복원 및 로컬 자산으로 TT 모델 생성은 별도로 확인한다.

## 다음 커밋·푸시에서도 유지할 기준

```bash
cd ~/quad_ws/codesign
python3 code/build_portable_snapshot.py
python3 code/portable_workspace.py verify
git add -A
```

스냅샷 생성은 현재 작업 폴더의 외부 의존 코드와 Git에서 제외된 필수 파일을 갱신한다. 90 MiB 이상인 새 파일도 분할 묶음에 넣고 원본은 Git ignore에 추가한다. 캐시·중복본·비밀정보를 제외한 뒤 커밋·푸시한다. 스냅샷이 커지므로 충분한 디스크 공간이 필요하다. 원격에 올리는 모든 새 Git blob이 100 MB 미만인지 최종 확인한다.
