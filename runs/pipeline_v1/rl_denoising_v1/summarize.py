"""Report the latest zigzag denoising experiment, keeping the circle baseline."""
import html
import base64
import json
import runpy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
if (ROOT/'shared_window_zigzag_10s_100iter/training.jsonl').exists():
    runpy.run_path(str(ROOT/'compare_shared_window.py'), run_name='__main__')
    raise SystemExit(0)
run = ROOT/'zigzag_1000iter'
protocol = json.loads((run/'protocol.json').read_text())
results = json.loads((run/'summary.json').read_text())
training = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
assert len(training) == protocol['updates'] == training[-1]['update']
assert all(np.isfinite(r['terminal_reward_mean']) for r in training)
assert all(not results['rl_after']['audit'][f'{k}_gradient_nonzero'] for k in ('e','s','a'))

def table(data):
    labels = {'unguided':'보정 없음', 'reward_gradient':'보상 기울기', 'rl_before':'RL 학습 전', 'rl_after':'RL 학습 후'}
    rows = []
    for method, result in data.items():
        for r in result['rows']:
            rows.append(f'<tr><td>{labels[method]}</td><td>{r["candidate"]}</td><td>{r["terminal_reward"]:.3f}</td><td>{r["progress"]:.3f}</td><td>{"생존" if r["survived"] else "낙상/무효"}</td><td>{r["steps"]*.02:.2f}</td><td>{r["prediction_position_rmse"]:.3f}</td></tr>')
    return '<table><tr><th>방법</th><th>후보</th><th>실제 실행 보상 ↑</th><th>전진 거리(m)</th><th>종료 상태</th><th>실행 시간(초)</th><th>예측 위치 오차(m)</th></tr>'+''.join(rows)+'</table>'

def chart():
    values = [r['terminal_reward_mean'] for r in training]
    lo, hi = min(values)-1, max(values)+1
    def y(v): return 255-(v-lo)/(hi-lo)*215
    points = ' '.join(f'{60+i/(len(values)-1)*720:.1f},{y(v):.1f}' for i,v in enumerate(values))
    ticks = ''.join(f'<text x="50" y="{y(v):.1f}" text-anchor="end">{v:.1f}</text>' for v in np.linspace(lo,hi,5))
    return f'<svg viewBox="0 0 820 320" role="img" aria-label="PPO 업데이트별 평균 실제 실행 보상"><path d="M60 35V260H790" fill="none" stroke="#555"/><polyline points="{points}" fill="none" stroke="#2875ce" stroke-width="2"/>{ticks}<text x="60" y="280">1</text><text x="760" y="280">100</text><text x="350" y="310">PPO 업데이트 횟수</text><text x="60" y="20">후보 8개의 평균 실제 실행 보상 (높을수록 좋음)</text></svg>'

def videos(task, updates):
    blocks=[]
    for candidate in range(2):
        path=ROOT/'videos'/f'{task}_candidate{candidate}_rl_track.mp4'
        url=html.escape(str(path),quote=True)
        embedded = base64.b64encode(path.read_bytes()).decode("ascii")
        blocks.append(f'<h3>{task} · {updates}회 학습 · 후보 {candidate}</h3><video controls playsinline preload="metadata"><source src="data:video/mp4;base64,{embedded}" type="video/mp4">영상 재생을 지원하는 브라우저에서 열어주세요.</video><p><a href="{url}">{url}</a></p>')
    return ''.join(blocks)

before=np.mean([r['terminal_reward'] for r in results['rl_before']['rows']])
after=np.mean([r['terminal_reward'] for r in results['rl_after']['rows']])
circle=json.loads((ROOT/'validated_circle_seed0/summary.json').read_text())
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>RL 디노이징 · 지그재그 100 iter 결과</title>
<style>body{{font:16px/1.8 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#202b3b}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3dd;padding:8px}}video,svg{{width:100%}}.result{{background:#fff0eb;padding:20px}}a{{overflow-wrap:anywhere}}</style>
<h1>RL 디노이징 · 지그재그 100 iter 결과</h1>
<p class="result"><b>후보 8개·100회 학습을 완료했습니다. 평균 보상과 생존 여부를 아래에서 비교합니다.</b> 동일한 별도 평가 잡음의 후보 8개에서 평균 보상이 {before:.3f} → {after:.3f}로 변했습니다. 생존 후보는 {sum(r['survived'] for r in results['rl_before']['rows'])}/8 → {sum(r['survived'] for r in results['rl_after']['rows'])}/8개입니다. 생존만으로 지그재그 추종 성공을 의미하지 않습니다. 평가 8개·seed 1개이므로 일반적인 성능 결론으로 확대할 수 없습니다.</p>
<p>평균 전진 거리는 {np.mean([r['progress'] for r in results['rl_before']['rows']]):.3f} → {np.mean([r['progress'] for r in results['rl_after']['rows']]):.3f}m입니다. 평균 보상과 전진 거리는 개선됐지만 생존율은 감소해 안정적인 트랙 추종을 달성했다고 판단할 수 없습니다.</p>
<h2>실험 조건</h2><p>동결된 v16 TT, PPO 업데이트 100회(회당 PPO 최적화 4 epoch), 후보 8개, 디노이징 {protocol['diffusion_steps']}단계, 상태 200개(최대 제어 실행 199×0.02=3.98초). 학습 물리 에피소드 {training[-1]['physical_episodes']}개, 디노이징 전이 {training[-1]['denoising_transitions']}개. 학습 루프 {training[-1]['seconds']:.1f}초(초기/최종 평가 시간 제외). 이전 2회 실험과 같은 seed·궤적 길이에 후보 수를 8개로 늘려 처음부터 100회 학습했습니다.</p>
<p>PPO는 디노이징 중 E와 미래 S/A의 epsilon을 보정합니다. 생성된 E·A를 CPU MuJoCo에서 실행한 보상으로 정책을 업데이트하며 TT 가중치는 고정합니다. RL 경로에는 보상 기울기를 사용하지 않습니다. 초기 S0는 생성 E의 안착 자세입니다.</p>
<h2>학습 곡선</h2>{chart()}<p>각 점은 서로 다른 학습 잡음에서 생성한 후보 8개의 평균입니다. 고정 평가 곡선이 아니므로 점 사이 변화만으로 성능 향상을 판단할 수 없습니다.</p>
<h2>같은 평가 잡음에서의 비교</h2>{table(results)}
<p>실제 실행 보상은 dense task return − 5×생성/실제 위치 MSE이며 낙상 패널티를 포함합니다. 전진 거리만으로 지그재그 추종 성공을 의미하지 않습니다.</p>
<h2>최신 지그재그 RL 영상</h2><p>100회 학습 후 평가 후보 0·1의 실제 물리 궤적입니다. 전체 8개 평가 결과는 위 표에 표시합니다. 노란 선은 목표 중심선, 흰 선은 경계입니다. 진폭 0.3m·주기 2m·반폭 0.15m. 낙상 후에는 마지막 기록 프레임을 유지합니다.</p>{videos('zigzag',100)}
<h2>기존 원형 실험: 2회 학습</h2><p>원형 정책은 이번에 재학습하지 않았으며 아래 영상은 기존 2회 학습 결과입니다. 지그재그 100회 실험과 학습량이 다릅니다.</p>{videos('circle',2)}
<p><a href="zigzag_1000iter/summary.json">최신 평가 원자료</a> · <a href="zigzag_1000iter/training.jsonl">학습 로그</a> · <a href="zigzag_1000iter/protocol.json">실험 조건</a> · <a href="README.md">구현 설명</a></p>
<p>제한: 단일 seed·평가 후보 8개, 짧은 궤적, window별 채널 상수 보정. 장기 추종 성능과 다중 seed 통계는 검증하지 않았습니다.</p></html>'''
best_video = ROOT/'zigzag_1000iter/recovered_iter30/best_iter30_video.html'
if best_video.exists():
    page = page.replace('<h2>최신 지그재그 RL 영상</h2>', '<h2>최고 학습 보상 시점 영상</h2><p><a href="zigzag_1000iter/recovered_iter30/best_iter30_video.html">30 iter · 후보 3 영상 열기</a> — 동일 seed 재현 및 원본 로그 일치 검증. 학습 rollout 기준이며 고정 평가 최적 정책을 의미하지 않습니다.</p><h2>최신 지그재그 RL 영상</h2>')
(ROOT/'report.html').write_text(page)
(ROOT/'verification.json').write_text(json.dumps(dict(run=run.name, updates=len(training), reward_gradients_used=False, evaluation_reward_before=float(before), evaluation_reward_after=float(after)),indent=2))
print(ROOT/'report.html')
