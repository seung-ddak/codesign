# v16 twin / delta 로컬 A/B 실행

2026-11-03. 원격99555a830의8-49 계획을 가져와 실행한다. kp30 데이터만 사용하며 v13의 실제 손실비율 고정은 적용하지 않는다.

1. 두 모델 모두 wide16 PPO4,864 + e_swap8,192 + action_pairs9,216 =22,272episode(48파일)를 사용한다. 구조·정규화·기본batch32·조건1/2/4비율20/40/40·seed0·60,000update는 같다. delta만 E/A각16쌍의 몸통14채널 변화량 손실을 계수1로 추가한다.
2. 보조 쌍 샘플링 난수를 기본 학습 난수와 분리했다. 따라서 추가손실 계산 때문에 기본 표본·조건·잡음 추첨 순서가 달라지지 않는다. source/conditioning 누적 카운터도 같은step에서 대조한다. dropout은0이다.
3. 문서의 사전 기준은516개 동일 행동 물리쌍의 전진 변화 부호다. 원격 코드의 고정형상 task평균 비교(서로다른생성A)와 달라, 516쌍의 각 실제E+s0에서 예측한 변화로 연결했다. 기존task평균부호는 보조값으로 유지했다. 문턱70%는 그대로다. 실제변화가 절대1e-6m이하인 경우만 부호판정에서 제외하며 비정상예측은 맞은표본으로 세지 않는다.
4. 최종DDIM 변화량 평가에는 같은쌍의 공통초기잡음을 명시적으로 배정했다. 쌍수와128예제청크경계가 맞지 않아도 양쪽잡음이 같다. 학습중단일잡음단계R²은 실제미래에잡음을넣은값을 입력하므로 보조모니터일뿐, 최종 생성R²과 다르다.
5. 문턱은 그대로다: 잘못된A의속도추가오차가 twin2배이상이며0.01이상, 동일A516쌍전진변화부호70%이상, 검증E/A쌍생성몸통R²각각0초과, 움직임오차가twin1.1배이하, 실제공동생성생존이twin이상. 판단 대상은delta대twin이다. twin보고서의v12참조는데이터·kp세계가달라 원인비교로해석하지않는다.

쌍원본 검증은 pair_data_audit.json에 있다. A16파일·3072그룹의 초기qpos/qvel차이0, 변하지않은행동쌍0, E16파일의동일소스행동차이0을 확인했다. 입력·학습구현해시는각run/conditions.json, 평가수정사전기록은protocol.json에 있다.

실행 순서는 twin학습→twin평가→delta학습→delta평가다. 중간코드/조건을 바꾸지 않는다. 처음 실행할 때 이미진행중인 twin학습은status/PID로기다리고중복학습하지않았다. 학습은기존supervise_train 자원감시를재사용하고, 평가는eval_v12parity 자체감시를사용한다. /home임시파일·캐시,CPU2개,Torch학습60%/평가25%,JAX40%로순차실행한다.

GPU가 보이는일반터미널의저장소루트에서 실행한다. 체인이실행중이면다시실행하지않는다.

```bash
TMPDIR="$PWD/.runtime/tmp" ../transformer-transformer/.venv/bin/python -u runs/pipeline_v1/v16_local_execution/run_chain.py
```

현재상태는이폴더status.json과각run/status.json(학습),eval_status.json(평가)에있다. 코드오류/자원부족이발생하면후속단계는중단한다. 완료한학습checkpoint는덮어쓰지않고,평가완료단계는재사용한다.

한계: E쌍은형상과그형상의실제초기상태가함께달라져순수E토큰의독립인과효과가아니다. delta는추가Mode1쌍감독과계산량도늘어나므로 같은감독량의별도복원대조군과비교한실험은아니다. 하나의학습seed이며, 양수R²만으로추가손실의효과를단정하지않고twin수치를함께본다.


## 완료 결과

두 모델 학습 및 각11단계 평가 완료. 최종 판정은 5개 기준 중2개 통과, 추가손실 설정 채택 기준 미달이다. 결과는 [report.html](report.html), 상세 검증은 [summary.json](summary.json)에 있다. 실제352후보·영상64개를 확인했다. 학습/평가 동안 RAM·디스크 부족 중단은 없었다.

사전 퇴보 방지 지표는 conditions.json의 `legacy_moving_rmse_max_ratio_vs_reference`에 따라 기존 Mode2 A고정 평가를 사용한다. 이 조건은 통과했다. 별도 Mode1 A알려짐 평가의 오차는55~71% 악화했으며 post-hoc로 별도 기록했다. 평가 기준을 사후 변경하지 않았다.

분기B에 따라 [guidance_path_review.md](guidance_path_review.md)에 학습·생성 경로 점검을 기록했다. 이번 설정은 실패했으므로10초 확장·λ조정·MJX CEM 실험은 수행하지 않았다. 추론 알고리즘과 확정 공동 생성 방향은 유지했다.

완료 결과 재검증 및 비교 보고서 재생성(학습/물리실행 없음):

```bash
OMP_NUM_THREADS=2 ../transformer-transformer/.venv/bin/python runs/pipeline_v1/v16_local_execution/summarize.py
```
