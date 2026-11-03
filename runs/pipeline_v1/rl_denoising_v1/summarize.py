"""Small execution-check report; no performance or convergence claim."""
import html
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
run = ROOT/'validated_zigzag_seed0'
results = json.loads((run/'summary.json').read_text())
rows = []
for method, result in results.items():
    for r in result['rows']:
        rows.append(f'<tr><td>{method}</td><td>{r["candidate"]}</td><td>{r["progress"]:.4f}</td>'
                    f'<td>{r["terminal_reward"]:.4f}</td><td>{r["survived"]}</td>'
                    f'<td>{r["prediction_position_rmse"]:.4f}</td></tr>')
a=np.load(run/'rl_before.npz'); b=np.load(run/'rl_after.npz'); u=np.load(run/'unguided.npz')
assert np.array_equal(a['pred'],u['pred']) and np.array_equal(a['E'],u['E'])
changes = dict(E=float(np.abs(b['E']-a['E']).max()),
               S=float(np.abs(b['pred'][...,:42]-a['pred'][...,:42]).max()),
               A=float(np.abs(b['pred'][...,42:]-a['pred'][...,42:]).max()))
assert all(value > 0 for value in changes.values())
assert all(not results['rl_after']['audit'][f'{k}_gradient_nonzero'] for k in ('e','s','a'))
videos = []
for task, label in [('zigzag','지그재그'),('circle','원형')]:
    for candidate in range(2):
        path = html.escape(str(ROOT/'videos'/f'{task}_candidate{candidate}_rl_track.mp4'), quote=True)
        videos.append(f'<h3>{label} · RL 후보 {candidate}</h3><video controls preload="metadata" style="width:100%" src="{path}"></video>'
                      f'<p><a href="{path}">{path}</a></p>')
page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Diffusion guidance PPO 치환 실행 검증</title>
<style>body{font:16px/1.8 system-ui;max-width:1050px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd3dd;padding:8px}</style>
<h1>Diffusion 내부 reward-gradient guidance를 PPO로 치환</h1>
<p><strong>구현 및 실제 실행 검증 완료. task 성능 개선 검증은 아니다.</strong></p>
<p>동결된 TT의 각 denoising 단계에서 PPO가 E·S·A의 epsilon 보정을 선택한다. reward 미분은 계산하지 않는다. 최종 E·A를 물리 실행한 보상으로 PPO를 학습했다. S0는 E의 안착 자세, 미래 S와 A는 공동 생성이다. 이전 task_rl_v1/v2의 별도 관절 제어 정책과 다르다.</p>
<p>실제 v16 twin checkpoint, 200 상태 프레임, 후보 2개, PPO 2update(학습 물리 episode4개·denoising 전이20개). 학습과 분리된 동일 평가 잡음으로 비교했다. unguided와 rl_before는 정확히 동일하다. rl_after는 E·S·A 모두 변경됐으며 gradient audit은 모두 false다.</p>
<p>학습 전후 최대 변화량: '''+json.dumps(changes)+'''</p>
<table><tr><th>방법</th><th>후보</th><th>실제 전진(m)</th><th>물리 terminal reward</th><th>생존</th><th>예측/물리 위치 RMSE(m)</th></tr>'''+''.join(rows)+'''</table>
<p>terminal reward는 dense physical return에서 생성/실제 위치 MSE×5를 뺀 값이다. reward-gradient 대조군만 기존 미분 보상을 사용하며 RL 경로는 사용하지 않는다. 표의 각 후보는 4초 이내 단일 실행이므로 성능 우열을 주장하지 않는다.</p>
<h2>실제 물리 실행 영상</h2>
<p>학습 후 diffusion PPO 영상만 표시한다. 노란 선은 목표 중심선, 흰 선은 경계다. 지그재그는 진폭 0.3m·주기 2m·반폭 0.15m, 원형은 반지름 1m·반폭 0.3m다. 트랙 표시는 보상식의 guide_lines를 사용하며 충돌·물리에는 영향을 주지 않는다.</p>
<p>지그재그는 기존 저장 기록을 사용했다. 원형은 기존 기록이 없어 같은 조건(PPO 2update·학습 물리 episode4개·별도 평가 후보2개)으로 새로 학습·실행했다. 각 task의 PPO 정책은 별개다. 각 후보의 생성 E와 실제 물리 qpos로 재생하며, 4초 실제 속도다. 낙상 뒤에는 마지막 프레임을 유지한다. 위 비교 표는 지그재그 결과이며 원형 원자료는 아래 링크에 있다.</p>'''+''.join(videos)+'''
<p>window당 채널별 상수 보정으로 행동 공간을 제한했다. S0 자유 생성, 장기 학습, 다중 seed 성능 평가는 수행하지 않았다.</p>
<p><a href="README.md">설계·실행 명령</a> · <a href="validated_zigzag_seed0/summary.json">지그재그 원본 결과</a> · <a href="validated_circle_seed0/summary.json">원형 원본 결과</a></p></html>'''
(ROOT/'report.html').write_text(page)
(ROOT/'verification.json').write_text(json.dumps(dict(zero_matches_prior=True, joint_changes=changes,
                                                     reward_gradients_used=False),indent=2))
print(ROOT/'report.html')
