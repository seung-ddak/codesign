"""video/slide.html: 형상 (E0~E4) 마다 한 페이지, 모션 5 개 영상과 수치를 한 화면에. 영상은 video/<E>/<motion>.mp4 (상대 경로)."""
import html, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from shapes import SHAPES, MOTIONS

rows = {(r['shape'], r['motion']): r for r in json.load(open(HERE / 'results/summary.json'))}
DIM = {'E0': '몸통 0.30 m · 허벅지 0.20 m · 종아리 0.175 m', 'E1': '허벅지 0.16 m · 종아리 0.14 m (몸통 0.30 m)',
       'E2': '허벅지 0.24 m · 종아리 0.21 m (몸통 0.30 m)', 'E3': '몸통 0.24 m (다리 기본)', 'E4': '몸통 0.36 m (다리 기본)'}


def fmt(v, f):
    return '—' if v is None else format(v, f)


pages = []
for K, s in SHAPES.items():
    trs, vids = [], []
    for M in MOTIONS:
        r = rows.get((K, M)); name = M.replace('__r00', '')
        if r is None:
            trs.append(f'<tr><td>{name}</td><td colspan="8" class="muted">결과 없음</td></tr>'); continue
        warn = ' class="warn"' if r['ref_infeasible_frac'] > .05 else ''
        trs.append(f"<tr{warn}><td>{name}</td><td>{'○' if r['survived'] else '×'} {r['seconds']:.1f}/{r['clip_seconds']:.1f} s</td>"
                   f"<td>{fmt(r['slip_m_s'], '.3f')}</td><td>{r['ref_slip_m_s']:.3f}</td><td>{r['joint_rmse_deg']:.1f}</td>"
                   f"<td>{r['root_rmse_cm']:.1f}</td><td>{r['contact_agree']:.2f}</td><td>{r['travel_ratio']:.2f} ({r['ref_travel_m']:.2f} m)</td><td>{r['ref_infeasible_frac'] * 100:.0f} %</td></tr>")
        if (HERE / 'video' / K / f'{M}.mp4').exists():
            vids.append(f'<figure><video src="{K}/{M}.mp4" controls muted loop playsinline preload="metadata"></video><figcaption>{name}</figcaption></figure>')
    pages.append(f'''<section class="page" id="{K}">
  <header><h2>{K} · {html.escape(s['name'])}</h2><p class="muted">{DIM[K]} · E = {np.round(s['E'], 3).tolist()}</p></header>
  <div class="wrap"><table>
    <thead><tr><th>모션</th><th>생존 (실행/모션 길이)</th><th>발 미끄러짐 (m/s)</th><th>레퍼런스 미끄러짐</th><th>관절 RMSE (°)</th><th>몸통 위치 RMSE (cm)</th><th>접지 일치</th><th>이동 비율 (실제/레퍼런스)</th><th>레퍼런스 무리 프레임</th></tr></thead>
    <tbody>{''.join(trs)}</tbody></table></div>
  <div class="grid">{''.join(vids)}</div>
</section>''')

nav = ''.join(f'<button data-k="{K}">{K}</button>' for K in SHAPES)
doc = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>형상별 모션 트래킹 점검</title>
<style>
:root{{--bg:#f7f7f5;--panel:#fff;--fg:#1c1c1a;--muted:#62615c;--line:#dedcd6;--accent:#2a78d6;--warn:#fff4e5;}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16171a;--panel:#1f2024;--fg:#ececea;--muted:#a3a29c;--line:#34353a;--accent:#5b9bea;--warn:#3a2f1e;}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,"Noto Sans KR",sans-serif}}
main{{max-width:1500px;margin:0 auto;padding:16px max(16px,env(safe-area-inset-left))}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:18px;margin:0}} .muted{{color:var(--muted);margin:2px 0}}
nav{{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}}
nav button{{border:1px solid var(--line);background:var(--panel);color:var(--fg);padding:6px 14px;border-radius:6px;cursor:pointer;font:inherit}}
nav button.on{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.page{{display:none;flex-direction:column;gap:12px}} .page.on{{display:flex}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;background:var(--panel)}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px}} tr.warn td{{background:var(--warn)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:12px}}
figure{{margin:0;min-width:0}} video{{width:100%;border-radius:6px;background:#000}} figcaption{{color:var(--muted);font-size:12px}}
.note{{color:var(--muted);font-size:12px}}
</style></head><body><main>
<h1>형상별 모션 트래킹 점검 · 발 미끄러짐</h1>
<p class="muted">형상 1 개 × 모션 1 개마다 PPO 정책 하나 (v7 전문가 설정: 동시 환경 32, 5.24M step, 미끄러짐 감점 없음. 넘어지면 감점 없이 조기 종료, step 보상 ≥ 0). 외란 없이 모션 처음부터 끝까지 실행. 영상: 왼쪽 A1 레퍼런스, 오른쪽 TT 트래킹.</p>
<nav>{nav}</nav>
{''.join(pages)}
<p class="note">발 미끄러짐 = 접지한 발 중심의 월드 수평 속도 평균. 레퍼런스 미끄러짐도 같은 방식. 이동 비율 = 실제 몸통 이동을 레퍼런스 이동 방향으로 투영한 값 / 레퍼런스 이동 거리 (1 = 같은 거리). 주황 배경 = 레퍼런스에서 발로 낼 수 없는 힘이 필요한 프레임이 5 % 넘는 모션 (물리적으로 무리). ←/→ 키로 페이지 이동.</p>
</main><script>
const keys={json.dumps(list(SHAPES))};
function show(k){{document.querySelectorAll('.page').forEach(p=>p.classList.toggle('on',p.id===k));document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.k===k));if(location.hash!=='#'+k)history.replaceState(null,'','#'+k);}}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>show(b.dataset.k));
addEventListener('keydown',e=>{{const i=keys.indexOf(location.hash.slice(1)||keys[0]);if(e.key==='ArrowRight')show(keys[Math.min(i+1,keys.length-1)]);if(e.key==='ArrowLeft')show(keys[Math.max(i-1,0)]);}});
addEventListener('hashchange',()=>show(location.hash.slice(1)||keys[0]));show(keys.includes(location.hash.slice(1))?location.hash.slice(1):keys[0]);
</script></body></html>'''
(HERE / 'video').mkdir(exist_ok=True)
(HERE / 'video' / 'slide.html').write_text(doc)
print(HERE / 'video' / 'slide.html')
