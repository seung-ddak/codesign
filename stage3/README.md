# 전진 보행 공진화

> **Stage 3 = Stage 2 성공 후의 hardware–policy co-design.** 현재는 코드만 보존된 보류 단계이며 T7 DDPO 대조 평가가 통과하기 전에는 실행하지 않는다. DDPO learner는 `stage2/code/ddpo_hardware.py`에서 가져오며 이 폴더에는 co-design orchestration만 둔다.

완료된 round의 두 정책과 결과를 저장하며, 중단 후에는 마지막 완료 round부터 재개한다.

## 실행 구조

1. pretrained TT hardware denoiser `ψ`로 stochastic DDIM trajectory와 후보 형상을 생성한다. 목표 궤적 토큰을 모두 attention에서 가려 reference motion을 사용하지 않는다.
2. TT RoboToken bank의 quadruped 128 choice 안에서 후보를 디코딩한다. 중복을 제거한 유효 형상이 2개 미만이면 고정 형상 학습으로 대체하지 않고 중단한다.
3. 형상마다 짧은 PPO 블록을 실행하며 **한 π_θ의 가중치와 정규화 통계를 계속 이어 간다**. 관측에는 choice·연속 설계값·상태·actuator mask를 넣고 팔 제어는 home 값으로 유지한다.
4. 갱신된 **동일 π_θ**로 모든 후보를 같은 seed에서 평가한다. 낙상 전 step reward의 합을 batch advantage로 정규화한다.
5. clipped DDPO-IS로 TT `ψ`를 갱신하고 다음 round에서 새 후보를 생성한다. 과거 제어기로 얻은 return은 다음 round에 재사용하지 않는다.

reward는 `Δ(world x) / goal`이다. 후진은 음수여서 왕복으로 보상을 늘릴 수 없다. 목표는 기본 1.0 m이고 낙상한 step의 추가 전진은 보상하지 않는다.
지정 calf_motor·gripper 접촉 또는 up_z < 0.1에서 종료한다. 성공은 무낙상 목표 도달이며, 별도 성공 보너스·alive 보상·추종 reward·관절 penalty는 없다.
제어 PPO discount는 1.0이며 형상 평가도 같은 누적 reward를 사용한다. 디코딩 불가 후보만 return -1로 처리한다.

## 실행

이전 `finetune_pi_theta.sh`와 `eval_design_d22.py`는 이 경로에 사용하지 않는다.
현재 머신의 Python은 `../transformer-transformer/.venv/bin/python`, `t2/mjx_env/train.py`에는 번들의 JAX 호환 수정이 반영돼 있다.
다른 머신에서는 `tt_patch/train.py.diff`를 TT에 적용해야 한다. `CODESIGN_TT`로 TT 경로를 지정할 수 있다.

```bash
cd /home/cgr-ugrad-2026/quad_ws/codesign
../transformer-transformer/.venv/bin/python stage3/code/train_coevolution.py --check
tmux new -s coevolution
../transformer-transformer/.venv/bin/python -u stage3/code/train_coevolution.py \
  --rounds 2 --designs 4 --theta-steps 4096 --num-envs 32 \
  --horizon 128 --eval-seeds 3 --out stage3/runs/coevolution_smoke
```

전원이 꺼지면 tmux도 종료된다. 다시 켠 뒤 아래 명령은 저장된 설정과 마지막 완료 round를 읽고 새 tmux에서 이어서 실행한다. 진행 중이던 round만 다시 계산한다.

```bash
bash stage3/code/start_coevolution_tmux.sh stage3/runs/coevolution_live_20260922
tmux attach -t coevolution_coevolution_live_20260922
```

부팅 뒤 자동 실행은 `stage3/code/coevolution-resume.service`를 사용자 systemd에 연결하고 활성화한다. 사용자 서비스가 로그인 전에도 실행되려면 해당 계정의 linger가 켜져 있어야 한다. 서비스의 실행 경로와 대상 run 폴더는 현재 머신 기준이다.

```bash
systemctl --user link "$PWD/stage3/code/coevolution-resume.service"
systemctl --user enable --now coevolution-resume.service
loginctl enable-linger "$USER"
```

### 학습 중 live MuJoCo viewer

학습은 headless로 계속 실행하고, 별도 터미널에서 아래 viewer를 실행한다. viewer는
`round_XXXX/theta.pkl`과 `designs.json`을 감시하다가 checkpoint가 생기면 해당 round의
최고 return 형상을 최신 `pi_theta`로 움직인다. 다음 round checkpoint가 저장되면 창을
교체해 새 형상·정책을 재생한다.

```bash
cd /home/cgr-ugrad-2026/quad_ws/codesign
../transformer-transformer/.venv/bin/python -u stage3/code/watch_coevolution_viewer.py \
  --run-dir stage3/runs/coevolution_full
```

GUI가 없는 SSH 세션에서는 MuJoCo viewer를 띄울 수 없다. 이 경우 viewer를 실행하지
말고 학습을 계속한다. 현재 checkpoint 복원과 MJX 한 step은 smoke 형상에서 검증됐다.

짧은 smoke는 연결 확인용이며 보행 학습 성공 판정용이 아니다. 본 학습의 기본 horizon은 1000이고 평가 seed는 9개다.
출력 디렉터리에는 실행 전에 `overview.html`이 생긴다. 새 round는 `theta.pkl`, `psi.pt`, `designs.json`, `best.xml`과 전체 `history.jsonl`을 보존한다. D58 이전 `phi.pt` checkpoint도 읽을 수 있다.
`designs.json`의 episode 값 순서는 `[return, fell, success, steps]`다. `best.xml`은 해당 round의 제어 정책으로 가장 높은 평균 return을 받은 형상이다.

## 검증 상태와 한계

- 통과: `stage3/code/test_coevolution.py`의 reward/종료·공유 관측/행동 규격·팔 제어 고정·DDPO loss 검사.
- 통과: 실제 TT GPU decoder 2-step 연결 검사, 첫 형상의 실제 GPU PPO 16 step 및 8-step rollout. 이 수치는 성능 결과가 아니다.
- 미완료: 두 번째 형상에서 정책 이어 학습, 100-step prior와 전체 bank 디코딩, 전체 2-round 실행. 사용자 요청으로 검증 세션 종료.
- 새 reference-free 관측은 TT expert와 호환되지 않아 π_θ를 fresh 초기화한다. 공개 prior만 재사용한다.
- 후보별 PPO 블록에서 가중치·관측 통계는 유지되지만 Adam 상태는 재초기화된다. 형상별 JIT 컴파일과 bank 최근접 검색 비용도 아직 측정하지 않았다.
- 목표 토큰을 가린 prior의 생성 품질은 아직 검증하지 않았다. 같은 return만 나오면 TT `ψ` 갱신을 건너뛰고 기록한다.
- 재개는 완료된 round 경계에서만 가능하다. 전원 차단 시 진행 중이던 후보·round는 다시 계산한다.
