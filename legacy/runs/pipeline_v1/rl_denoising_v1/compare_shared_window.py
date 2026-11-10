"""Write the fixed-TT comparison from recorded physics results, with inline videos."""
import base64
import html
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
RUNS = [('fixed_tt_zigzag_100iter','기존 전체 구간 보정 정책'),
        ('shared_window_zigzag_10s_100iter','구간별 공유 보정 정책')]


def video(path):
    if not path.exists():
        return '<p>평가 영상 준비 중입니다.</p>'
    source = base64.b64encode(path.read_bytes()).decode('ascii')
    url = html.escape(str(path.resolve()),quote=True)
    return f'<video controls playsinline preload="metadata"><source src="data:video/mp4;base64,{source}" type="video/mp4"></video><p><a href="{url}">영상 절대 경로</a></p>'


def chart(records):
    values = [r['terminal_reward_mean'] for rows in records for r in rows]
    low, high = min(values)-1, max(values)+1
    y = lambda v: 245-(v-low)/(high-low)*205
    lines = []
    for rows,color in zip(records,['#b04a27','#196cc2']):
        points = ' '.join(f'{65+(r["update"]-1)/99*700:.1f},{y(r["terminal_reward_mean"]):.1f}' for r in rows)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    ticks = ''.join(f'<text x="55" y="{y(v):.1f}" text-anchor="end">{v:.1f}</text>' for v in [low,(low+high)/2,high])
    return '<svg viewBox="0 0 810 310" role="img" aria-label="학습 횟수에 따른 평균 물리 실행 보상"><text x="65" y="20">평균 물리 실행 보상 · 높을수록 좋음</text><path d="M65 35V250H775" fill="none" stroke="#555"/>'+''.join(lines)+ticks+'<text x="65" y="275">1</text><text x="750" y="275">100</text><text x="340" y="302">정책 학습 횟수</text></svg>'


blocks, totals, training = [], [], []
for dirname,label in RUNS:
    run = ROOT/dirname
    rows = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
    training.append(rows)
    protocol = json.loads((run/'protocol.json').read_text())
    assert protocol['episode_states']==501 and protocol['batch']==8 and protocol['updates']==100
    if not (run/'summary.json').exists():
        blocks.append(f'<h2>{label}</h2><p>학습 중: {rows[-1]["update"]}/100회. 완료 평가 결과는 아직 없습니다.</p>')
        continue
    assert len(rows)==100 and rows[-1]['update']==100
    data = json.loads((run/'summary.json').read_text())
    assert all(len(v['rows'])==8 for v in data.values())
    for method in ['rl_before','rl_after']:
        a=data[method]['rows']
        totals.append(f'<tr><td>{label}</td><td>{"학습 전" if method=="rl_before" else "학습 후"}</td><td>{mean(x["terminal_reward"] for x in a):.3f}</td><td>{mean(x["progress"] for x in a):.3f}</td><td>{sum(x["survived"] for x in a)}/8</td><td>{sum(x["invalid_initial"] for x in a)}/8</td><td>{mean(x["steps"]*.02 for x in a):.3f}</td></tr>')
    details=[]
    for x in data['rl_after']['rows']:
        status='초기 상태 무효' if x['invalid_initial'] else ('최대 시간 도달' if x['survived'] else '낙상')
        details.append(f'<tr><td>{x["candidate"]}</td><td>{x["terminal_reward"]:.3f}</td><td>{x["progress"]:.3f}</td><td>{status}</td><td>{x["steps"]*.02:.2f}</td></tr>')
    valid=[x for x in data['rl_after']['rows'] if not x['invalid_initial']]
    candidate=max(valid or data['rl_after']['rows'],key=lambda x:x['terminal_reward'])['candidate']
    best=json.loads((run/'best_training.json').read_text())
    audit=json.loads((run/'tt_prediction_audit.json').read_text())
    assert all(x['joint_prediction_passes']==1 for x in audit.values())
    blocks.append(f'<h2>{label}</h2><p>100회 완료. 학습 물리 실행 800개, 실제 학습 시간 {rows[-1]["seconds"]:.1f}초. 가장 높은 평균 학습 보상은 {best["update"]}회째 {best["mean_reward"]:.3f}입니다. 학습 최고 보상은 별도 평가의 최적 정책을 의미하지 않습니다.</p><table><tr><th>평가 후보</th><th>물리 보상</th><th>전진 거리(미터)</th><th>종료 이유</th><th>실행 시간(초)</th></tr>'+''.join(details)+f'</table><h3>학습 후 대표 평가 영상 · 후보 {candidate}</h3><p>초기 상태가 유효한 후보 중 보상이 가장 높은 후보입니다. 최대 10초 표시하며, 조기 종료 후에는 마지막 장면을 유지합니다.</p>'+video(run/'videos'/f'zigzag_candidate{candidate}_rl_track.mp4')+f'<p><a href="{dirname}/summary.json">전체 평가 수치</a> · <a href="{dirname}/training.jsonl">학습 기록</a> · <a href="{dirname}/protocol.json">실험 조건</a> · <a href="{dirname}/best_training.json">최고 학습 보상 기록</a></p>')

finished=all((ROOT/name/'summary.json').exists() for name,_ in RUNS)
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>고정 TT 예측 · 공유 보정 정책 비교</title><style>body{font:16px/1.8 system-ui;max-width:1120px;margin:35px auto;padding:0 20px;color:#243044}table{border-collapse:collapse;width:100%;margin:15px 0}th,td{border:1px solid #ccd3dd;padding:8px}video,svg{width:100%}.note{padding:18px;background:#fff0e5}a{overflow-wrap:anywhere}</style><h1>고정 TT 예측 · 공유 보정 정책 비교</h1>'''
page+='<p class="note">'+('두 방식 모두 100회 학습과 평가를 완료했지만 10초 완주는 0/8입니다. 공유 정책의 평균 평가 보상은 기존 정책보다 낮았으며, 이번 조건에서 안정적인 지그재그 수행 개선은 확인되지 않았습니다.' if finished else '기존 10초 실험의 평가는 완료됐으며, 공유 보정 정책은 학습 중입니다.')+'</p>'
page+='''<h2>동일한 실험 조건</h2><p>지그재그, 후보 8개, 정책 학습 100회, 상태 501개, 제어 500회 × 0.02초 = 최대 10초. 같은 TT 체크포인트, 학습 잡음 번호 0, 별도 평가 잡음 번호 1000000, 학습률 0.0003, 최적화 반복 4회, 탐색 가중치 0.001, 기존 보상식을 유지합니다. 초기 상태가 무효이거나 넘어지면 조기 종료합니다.</p><p>최고 잡음 단계에서 TT 예측 묶음을 한 번 계산해 고정하고, 매번 새 정책 보정으로 최종 상태를 직접 복원합니다. 학습과 평가에 각각 예측 묶음 1회를 사용하며, 각 묶음에는 여러 구간과 몸체 예측용 TT 호출이 포함됩니다. 일반적인 5단계 또는 100단계 디노이징 실험이 아닙니다.</p><p>기존 정책은 모든 구간의 보정값을 하나의 큰 출력으로 계산합니다. 새 정책은 전체 구간의 정보를 모아 몸체 E 보정값 하나를 계산하고, 상태 S와 행동 A는 같은 작은 정책을 각 구간에 적용합니다. 구간별 입력에는 현재 구간의 잡음 통계, 전체 구간의 평균 통계, E, 잡음 단계와 구간의 상대 위치가 들어갑니다. 탐색 분산도 구간 간 공유합니다.</p><h2>별도 평가 잡음에서의 비교</h2><table><tr><th>정책</th><th>시점</th><th>평균 물리 보상</th><th>평균 전진 거리(미터)</th><th>10초 완주</th><th>초기 상태 무효</th><th>평균 실행 시간(초)</th></tr>'''+''.join(totals)+'</table><p>생존과 전진 거리만으로 트랙 추종 성공을 판단하지 않습니다. 초기 상태가 무효인 후보도 평균 보상과 후보 수에 포함합니다.</p><h2>학습 보상</h2><p>갈색: 기존 정책 · 파란색: 공유 정책. 같은 학습 잡음 후보를 반복 사용하지만 보정은 확률적으로 탐색합니다. 이 곡선은 고정 평가 성능 곡선이 아닙니다.</p>'+chart(training)+''.join(blocks)
page+='''<h2>해석의 한계</h2><p>단일 학습 잡음 묶음 8개와 별도 평가 묶음 8개만 사용합니다. 공유 정책은 다른 길이에서도 같은 가중치를 불러올 수 있지만, 긴 시간에 대한 성능 보장은 없습니다. 형태 E는 궤적 전체에서 하나로 공유합니다. 현재 버전은 실제 상태를 받아 다음 구간을 다시 만드는 실시간 제어기가 아니라, 전체 궤적을 미리 보정한 뒤 실행하는 방식입니다. 앞 구간의 실제 실패를 보고 뒤 구간을 수정하지 않습니다.</p><p><a href="report_4s_batch8.html">이전 4초 실험 보고서</a> · <a href="zigzag_1000iter/recovered_iter30/best_iter30_video.html">이전 4초 실험의 최고 학습 보상 영상</a> · <a href="README.md">실행 방법과 구현 설명</a></p></html>'''
(ROOT/'report.html').write_text(page)
print(ROOT/'report.html')
