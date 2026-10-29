"""Compare V7 observed physics results against V6; no training or simulation here."""
import base64
import html
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

root = Path(__file__).resolve().parent
base = root.parent / 'fast_mode24_v6_steps60k'
load = lambda p: json.loads(p.read_text())
uri = lambda p, mime: f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode()
cc = load(root / 'codesign_check/codesign_check.json')
old = load(base / 'codesign_check/codesign_check.json')
smoke = load(root / 'smoke_zigzag_joint_c16/summary.json')['runs']['scale_0']
errors = [r['path_err_m'] for r in smoke if r['path_err_m'] is not None and r['mjx_total'] is not None]
median = float(np.median(errors)) if errors else None
sections = []
gates = {}
for task in ('zigzag', 'circle'):
    folder = root / f'codesign_check/joint_{task}_4s'
    fixed = load(root / f'codesign_check/fixedE_{task}_4s/summary.json')['runs']['scale_10']
    joint = load(folder / 'summary.json')['runs']['scale_10']
    assert len(fixed) == 64 and len(joint) == 16
    predicted, actual = [], []
    for i in range(16):
        rows = [r for r in fixed[i*4:(i+1)*4] if r['mjx_total'] is not None]
        if rows:
            predicted.append(np.mean([r['pred_total'] for r in rows]))
            actual.append(np.mean([r['mjx_total'] for r in rows]))
    rho = float(spearmanr(predicted, actual).statistic) if len(actual) > 1 else float('nan')
    rho = rho if np.isfinite(rho) else None
    gates[task] = dict(predicted_E_p=cc[task]['pred_R']['p'], rank_correlation=rho,
                       smoke_median_error_m=median,
                       passed=bool(cc[task]['pred_R']['p'] < .05 and rho is not None and rho > .6 and median is not None and median < .15))
    result = cc[task]['joint']; previous = old[task]['joint']
    selected = [max(range(16), key=lambda i: joint[i]['pred_total'])]
    valid = [i for i in range(16) if joint[i]['mjx_total'] is not None]
    if valid: selected.append(max(valid, key=lambda i: joint[i]['mjx_total']))
    cards = []
    for label, i in zip(('TT 예측 보상 1위', 'MJX 실제 보상 1위'), selected):
        r = joint[i]; video = folder / f'videos/scale_10_cand{i}.mp4'
        media = f'<video controls muted loop playsinline src="{uri(video,"video/mp4")}"></video>' if video.exists() else '<p>영상 없음: 유효 시작 여부 확인</p>'
        cards.append(f'<article><h3>{label} · joint {i}</h3><p>예측 R {r["pred_total"]:.3f} / 실제 R {r["mjx_total"]} / 낙상 step {r["mjx_fell_step"]}</p>{media}</article>')
    details = []
    for i,r in enumerate(joint):
        if i in selected: continue
        v = folder / f'videos/scale_10_cand{i}.mp4'
        media = f'<video controls muted playsinline src="{uri(v,"video/mp4")}"></video>' if v.exists() else '<p>영상 없음</p>'
        details.append(f'<article><h3>joint {i}</h3><p>예측 R {r["pred_total"]:.3f} / 실제 R {r["mjx_total"]}</p>{media}</article>')
    sections.append(f'<section><h2>{task}</h2><p>V7 gate: {html.escape(str(gates[task]))}</p><table><tr><th>Joint</th><th>V6</th><th>V7</th></tr>'
                    + ''.join(f'<tr><td>{key}</td><td>{previous[key]}</td><td>{result[key]}</td></tr>' for key in ('median_R','median_progress','done','valid','percentile_vs_designs'))
                    + '</table><p>done은 제한 시간 생존이며 트랙 완주가 아니다.</p>'
                    + f'<img src="{uri(folder / "paths.png", "image/png")}" alt="TT 예측 및 MJX 실제 경로">'
                    + '<div class="grid">' + ''.join(cards) + '</div><details><summary>나머지 joint 영상</summary><div class="grid">' + ''.join(details) + '</div></details></section>')
assert set(gates) == {'zigzag','circle'}
(root / 'comparison.json').write_text(json.dumps(gates, indent=2, allow_nan=False))
report = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7 E-swap 결과</title><style>
body{font:16px/1.6 system-ui;background:#101820;color:#e5eef3;padding:24px}main{max-width:1200px;margin:auto}section,article{padding:20px;border:1px solid #354653;border-radius:10px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}video,img{width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #354653}a{color:#8ed4ed}
</style><main><h1>V7 · E 반사실 데이터 추가 결과</h1><p>V6와 같은 60k 학습 설정에 E 교체 데이터만 추가. 16 원본 형상 × 32 action sequence × 16 대상 형상 = 8,192 rollout. 각 대상 형상 고유 home 안착 상태에서 실행한다.</p>
<p>고정 E: 16형상 × 4 seed. Joint: 16후보. 두 task 모두 scale 10, 4초 open-loop. 2초 scale 0 smoke는 16후보. 재계획 없음.</p>'''
report += f'<p>2초 smoke 유효 후보 {len(errors)}/16 · 경로 차 중앙값 {median}m. 기준: 예측 보상의 형상 효과 p&lt;0.05, 예측/실제 형상 평균 보상 순위상관 &gt;0.6, smoke 경로 차 &lt;0.15m.</p>'
report += ''.join(sections) + '<p><a href="comparison.json">판정 JSON</a></p></main></html>'
(root / 'report.html').write_text(report)
print(json.dumps(gates, indent=2, allow_nan=False))
