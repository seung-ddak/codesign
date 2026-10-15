# Expert 학습 장비·설정 검증 (2026-10-15, KST)

- CPU: AMD Ryzen 7 7800X3D, 8코어/16스레드. RAM: 30 GiB, 학습 전 사용 가능 약 22 GiB.
- GPU: NVIDIA GeForce RTX 4070 SUPER, VRAM 12,282 MiB. 학습 전 사용 가능 약 10.8 GiB.
- MJX는 GPU에서 여러 시뮬레이션 환경을 한 정책의 학습 배치로 병렬 실행한다. 독립적인 정책 19개는 GPU 메모리 경합을 피하기 위해 한 프로세스씩 순차 실행한다.
- 이번 실행: `e000`~`e002` 세 TT 형상으로 모션당 정책 1개, e003은 미학습 형상 평가용. 모션당 5,242,880 환경 스텝, 환경 32개, 배치 32개, 평가 환경 4개, 평가 10회, W&B offline. `XLA_PYTHON_CLIENT_PREALLOCATE=false`로 VRAM 선할당을 끈다.
- 검증: 평지에서 비발 형상 접촉이 실제로 생성되고 자체 충돌이 없는 것을 CPU MuJoCo에서 확인했다. GPU MJX에서 세 형상 병렬 reset/step 통과. `squat__r00` 12,288-step/환경 32개 소규모 학습에서 최종 checkpoint가 생성됐고, 별도 e000 평가에서 CSV·MP4가 생성됐다. 짧은 학습의 평가 불합격은 expert 성능을 뜻하지 않는다.
- 위 값은 소규모 검증에 기반한 보수적 시작 설정이다. 19개 모두의 수렴·완료 시간은 보장하지 않는다. `train_selected_experts.sh`가 `selection.csv`를 읽어 순차 학습하고, 완료한 모션마다 e000~e003 평가 CSV·비교 영상을 저장한다. 오류가 나면 다음 모션으로 넘어가지 않는다.
- GitHub 진행 기록: [Issue #2](https://github.com/seung-ddak/codesign/issues/2). `notify_selected_experts.py`가 별도 tmux 세션 `pipeline_v1_github_notify`에서 완료 모션의 네 형상 오차 표를 `motion_eval/report.md`에 저장하고 같은 내용을 Issue 댓글로 게시한다. 로그는 `github_notify.log`, 댓글 완료 목록은 `github_notified.txt`. 토큰은 저장소 밖 `~/.config/codesign/github_token`에 권한 600으로 보관한다. 사용자 자신의 토큰으로 댓글을 올리므로 자신의 활동 알림 설정에 따라 실제 알림이 오지 않을 수 있다.
