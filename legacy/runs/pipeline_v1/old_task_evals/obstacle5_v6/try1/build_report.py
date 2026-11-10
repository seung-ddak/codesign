"""Build the obstacle smoke report from observed results only."""
import base64
import html
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / 'summary.json').read_text())

def uri(path, mime):
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()

sections = []
for scale, rows in summary['runs'].items():
    cards = []
    for i, row in enumerate(rows):
        result = row['initial_physics'].get('obstacle_task')
        outcome = ('실행 제외: ' + str(row['skip_reason'])) if result is None else (
            f"목표 도달 성공: {result['success']} · 막대기 접촉: {result['contact']} · 목표 거리: {result['goal_distance_m']:.3f} m")
        vid = root / 'videos' / f'{scale}_cand{i}.mp4'
        media = f'<video controls muted loop playsinline src="{uri(vid, "video/mp4")}"></video>' if vid.exists() else '<p>영상 없음</p>'
        cards.append(f'<article><h3>joint 후보 {i}</h3><p>{html.escape(outcome)}</p>'
                     f'<p>TT 예측 R: {row["pred_total"]:.3f} · MJX 실제 R: {row["mjx_total"]} · '
                     f'실행 시간: {row["mjx_valid_s"]:.2f}s · 낙상 step: {row["mjx_fell_step"]}</p>{media}</article>')
    sections.append(f'<section><h2>{html.escape(scale)}</h2><div class="grid">' + ''.join(cards) + '</div></section>')
counts = [(s, len(rows), sum(r['initial_physics'].get('obstacle_task', {}).get('success', False) for r in rows))
          for s, rows in summary['runs'].items()]
report = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TT v6 · 막대기 5개 연속 회피 실험</title><style>
body{font:16px/1.6 system-ui;background:#101820;color:#e5eef3;margin:0;padding:32px}main{max-width:1200px;margin:auto}
section,article{background:#192733;border:1px solid #354653;border-radius:12px;padding:20px;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}video,img{width:100%;height:auto}a{color:#8ed4ed}table{width:100%;border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #354653}
</style><main><h1>기존 TT v6 · 막대기 5개 연속 회피</h1>
<p>재학습 없이 60k 체크포인트로 E·S·A 공동 생성. 같은 초기 노이즈에서 guidance scale 0과 10을 비교한다.
{seconds}초 open-loop · 후보 각 2개 · CPU의 JAX/MJX 실행.</p>
<p>출발 (0,0). 막대기 5개를 (1,0), (2,0), (3,0), (4,0), (5,0)에 1m 간격으로 일렬 설치했다. 각 막대기는 반지름 0.06m, 높이 약 0.82m이며 겹친 구 10개로 구성한다.
목표 (6,0)에 0.2m 이내 도달하고, 낙상과 관측된 장애물 접촉이 없어야 성공이다. 단순 생존은 성공이 아니다.</p>
<p>Guidance는 전체 로봇 형상을 모든 관절 자세에서 감싸는 보수적인 반경과 5cm 여유를 사용한다.
반경은 E별 실제 모델에서 계산하지만 미분하지 않는다. E gradient는 TT dynamics를 통해 전달된다.
실제 충돌은 로봇 전체와 장애물의 MJX 접촉으로 따로 판정한다. 접촉은 50Hz 제어 시점에서 관측하며 제어 시점 사이의 짧은 접촉을 놓칠 수 있다.
작은 표본·짧은 실행의 경로 검증이며 일반적인 장애물 회피 능력의 증거는 아니다.</p>'''
report = report.replace('{seconds}', str(summary['seconds']))
report += f'<section><h2>실험 환경과 막대기 설치</h2><img alt="실제 MuJoCo 모델의 출발점, 막대기, 목표 배치" src="{uri(root / "environment.png", "image/png")}"><p>왼쪽 파란 원: 출발 (0,0). 빨간 막대기 5개: x=1·2·3·4·5m, y=0, 최대 지름 12cm, 높이 약 82cm. 오른쪽 초록 원: 목표 (6,0), 도달 반경 20cm. 출발에서 첫 막대기까지 1m, 막대기 사이도 1m, 마지막 막대기에서 목표까지 1m이다.</p><p>사진은 실제 평가에 저장된 후보 0의 물리 모델을 렌더했다. 막대기는 반지름 6cm인 구 10개씩 수직으로 겹쳐 설치한 고정 장애물이다. 파란·초록 바닥 표시는 설명용이며 물리적 장애물이 아니다. 로봇 형상은 후보마다 달라진다.</p></section>'
report += '<section><h2>실제 목표 도달 · 5개 막대기 뒤 6m 목표</h2>' + ''.join(f'<p>{s}: {ok}/{n}</p>' for s, n, ok in counts) + '</section>'
rows_html = []
for scale, rows in summary['runs'].items():
    for i, row in enumerate(rows):
        outcome = row['initial_physics'].get('obstacle_task', {})
        rows_html.append(f'<tr><td>{scale}</td><td>{i}</td><td>{row["pred_total"]:.3f}</td>'
                         f'<td>{row["mjx_total"]}</td><td>{row["mjx_valid_s"]:.2f}s</td>'
                         f'<td>{outcome.get("contact", "실행 제외")}</td><td>{outcome.get("goal_distance_m", "실행 제외")}</td>'
                         f'<td>{outcome.get("success", False)}</td></tr>')
report += '<section><h2>후보별 실제 평가</h2><table><tr><th>Guidance</th><th>후보</th><th>TT 예측 R</th><th>MJX R</th><th>유효 실행</th><th>접촉</th><th>목표 거리 m</th><th>성공</th></tr>' + ''.join(rows_html) + '</table></section>'
report += f'<section><h2>경로: 점선 TT 예측 / 실선 MJX</h2><img src="{uri(root / "paths.png", "image/png")}"></section>'
report += ''.join(sections) + '<p><a href="summary.json">원본 수치 JSON</a></p></main></html>'
assert len(counts) == 2 and all(n == 2 for _, n, _ in counts)
(root / 'report.html').write_text(report)
print(counts)
