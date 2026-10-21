"""s0 보정 audit 결과 → report.html (그림은 inline SVG, 파일 하나로 완결).

python3 runs/pipeline_v1/diag/s0_align_audit/make_report.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SH = ("e000", "e001", "e002")
raw = json.loads((HERE / "raw/selection.json").read_text())
ali = json.loads((HERE / "aligned/selection.json").read_text())
POOL = 768


def valid_chart():
    """원래 형상별 '세 형상 모두 유효' 시작점 수, 보정 전/후 가로 막대."""
    W, row, x0, xw = 640, 58, 70, 500
    h = 40 + row * 3
    s = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="보정 전후 유효 시작점 수">']
    for v in (0, 192, 384, 576, 768):
        x = x0 + xw * v / POOL
        s.append(f'<line x1="{x}" y1="10" x2="{x}" y2="{h-24}" class="grid"/>'
                 f'<text x="{x}" y="{h-8}" class="tick" text-anchor="middle">{v}</text>')
    for i, src in enumerate(SH):
        y = 16 + row * i
        s.append(f'<text x="{x0-10}" y="{y+22}" class="lab" text-anchor="end">{src}</text>')
        for j, (data, cls) in enumerate(((raw, "raw"), (ali, "ali"))):
            n = data["selection"][src]["physically_valid"]
            w = max(1.5, xw * n / POOL)
            yy = y + j * 20
            s.append(f'<rect x="{x0}" y="{yy}" width="{w:.1f}" height="16" rx="2" class="{cls}"/>'
                     f'<text x="{x0+w+6:.1f}" y="{yy+12.5}" class="val">{n}</text>')
    s.append("</svg>")
    return "".join(s)


def zshift_chart():
    """원래→대상 형상별 몸통 z 이동 (선택된 192개, p5–p95 막대 + 중앙값 점)."""
    pairs = [(a, b) for a in SH for b in SH if a != b]
    stats = {p: np.percentile(np.array(ali["body_z_shift_m"][p[0]][p[1]]) * 100, [5, 50, 95]) for p in pairs}
    lo, hi = -8, 14
    W, row, x0, xw = 640, 34, 120, 480
    h = 30 + row * len(pairs) + 24
    X = lambda v: x0 + xw * (v - lo) / (hi - lo)
    s = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="형상별 몸통 z 이동량">']
    for v in range(lo, hi + 1, 2):
        cls = "zero" if v == 0 else "grid"
        s.append(f'<line x1="{X(v)}" y1="14" x2="{X(v)}" y2="{h-24}" class="{cls}"/>'
                 f'<text x="{X(v)}" y="{h-8}" class="tick" text-anchor="middle">{v}</text>')
    for i, (a, b) in enumerate(pairs):
        p5, p50, p95 = stats[(a, b)]
        y = 30 + row * i
        s.append(f'<text x="{x0-12}" y="{y+4}" class="lab" text-anchor="end">{a} → {b}</text>'
                 f'<line x1="{X(p5):.1f}" y1="{y}" x2="{X(p95):.1f}" y2="{y}" class="range"/>'
                 f'<circle cx="{X(p50):.1f}" cy="{y}" r="5" class="med"/>'
                 f'<text x="{X(p95)+8:.1f}" y="{y+4}" class="val">{p50:+.1f} cm</text>')
    s.append("</svg>")
    return "".join(s), stats


def pen_table():
    rows = []
    for src in SH:
        cells = []
        for tgt in SH:
            r = raw["selection"][src]["audit_counts"][f"{tgt}:severe_floor_penetration"]
            a = ali["selection"][src]["audit_counts"][f"{tgt}:severe_floor_penetration"]
            cls = ' class="self"' if src == tgt else (' class="bad"' if r > 100 else "")
            cells.append(f"<td{cls}>{r} → <b>{a}</b></td>")
        rows.append(f"<tr><th>{src}</th>{''.join(cells)}</tr>")
    return "".join(rows)


def sel_table():
    rows = []
    for src in SH:
        r, a = raw["selection"][src], ali["selection"][src]
        jl = a["audit_counts"][f"{src}:joint_limit_violation"]
        rows.append(f"<tr><th>{src}</th><td>{POOL}</td><td>{r['physically_valid']} → <b>{a['physically_valid']}</b></td>"
                    f"<td>{r['selected']} → <b>{a['selected']}</b></td>"
                    f"<td>{r['selected_contact_matched']} → <b>{a['selected_contact_matched']}</b></td><td>{jl}</td></tr>")
    return "".join(rows)


zsvg, zstats = zshift_chart()
tot_raw = sum(raw["selection"][s]["selected"] for s in SH)
tot_ali = sum(ali["selection"][s]["selected"] for s in SH)

html = f"""<title>s0 높이 보정 Audit</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: 단일 컬럼 실험 기록, 결과 먼저 → 그림 → 조건 → Q&A */
:root {{
  --bg:#f6f7f9; --surface:#ffffff; --fg:#1b1f27; --muted:#5d6675; --line:#d9dde4;
  --raw:#b7bdc8; --accent:#2f6fde; --warn:#c2410c;
  --sans:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#12151b; --surface:#1a1e26; --fg:#e6e9ef; --muted:#9aa3b2; --line:#2c323d;
  --raw:#4a5260; --accent:#6aa0ff; --warn:#fb923c; color-scheme:dark }} }}
:root[data-theme="dark"] {{
  --bg:#12151b; --surface:#1a1e26; --fg:#e6e9ef; --muted:#9aa3b2; --line:#2c323d;
  --raw:#4a5260; --accent:#6aa0ff; --warn:#fb923c; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.65 var(--sans);
  padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); padding-block:32px 64px }}
main {{ max-width:760px; margin:0 auto; display:grid; gap:28px }}
h1 {{ font-size:1.6rem; margin:0; text-wrap:balance }}
h2 {{ font-size:1.12rem; margin:0 0 10px; text-wrap:balance }}
.meta {{ color:var(--muted); font-family:var(--mono); font-size:.8rem }}
section {{ min-width:0 }}
.verdict {{ background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:6px; padding:14px 18px }}
.verdict p {{ margin:.2em 0 }}
.pass {{ display:inline-block; font-family:var(--mono); font-size:.75rem; letter-spacing:.06em; color:var(--surface); background:var(--accent); border-radius:3px; padding:1px 7px; margin-right:6px }}
.tw {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:.9rem }}
th,td {{ border-bottom:1px solid var(--line); padding:6px 10px; text-align:right; white-space:nowrap }}
th:first-child {{ text-align:left }}
thead th {{ color:var(--muted); font-weight:600; font-size:.8rem }}
td.bad {{ color:var(--warn) }}
td.self {{ color:var(--muted) }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:14px }}
figcaption {{ color:var(--muted); font-size:.85rem; margin-top:8px }}
svg {{ width:100%; height:auto; display:block; font-family:var(--mono) }}
svg .grid {{ stroke:var(--line); stroke-width:1 }}
svg .zero {{ stroke:var(--muted); stroke-width:1.2 }}
svg .tick {{ fill:var(--muted); font-size:11px }}
svg .lab {{ fill:var(--fg); font-size:12.5px }}
svg .val {{ fill:var(--fg); font-size:11.5px }}
svg .raw {{ fill:var(--raw) }}
svg .ali {{ fill:var(--accent) }}
svg .range {{ stroke:var(--accent); stroke-width:6; stroke-linecap:round; opacity:.45 }}
svg .med {{ fill:var(--accent) }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:.82rem; color:var(--muted); margin-bottom:6px }}
.sw {{ display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:5px }}
.qa {{ display:grid; gap:14px }}
.q {{ font-weight:600 }}
.q::before {{ content:"Q. "; color:var(--accent) }}
.a::before {{ content:"A. "; font-weight:600; color:var(--muted) }}
.a p {{ margin:.25em 0 }}
code {{ font-family:var(--mono); font-size:.86em }}
ul {{ margin:.3em 0; padding-left:1.2em }}
</style>
<main>
<header>
  <div class="meta">codesign · pipeline_v1 · diag/s0_align_audit · 2026-10-20</div>
  <h1>s0 높이 보정 Audit</h1>
</header>

<section class="verdict">
  <p><span class="pass">PASS</span>세 형상 모두에서 쓸 수 있는 시작점이 원래 형상별로 <b>733 / 745 / 759</b>개 (기준 각 ≥ 50).</p>
  <p>보정 전에는 e000 출신 6개, e002 출신 1개뿐이라 D4 평가 표본의 96 % ({192}/{tot_raw}) 가 e001 출신이었다. 보정 후 선택 표본은 {tot_ali}개, 형상당 192개로 균등하다.</p>
</section>

<section>
  <h2>1. 세 형상 모두 유효한 시작점 수</h2>
  <figure>
    <div class="legend"><span><span class="sw" style="background:var(--raw)"></span>보정 전 (원래 qpos 그대로)</span>
    <span><span class="sw" style="background:var(--accent)"></span>보정 후 (몸통 z 만 이동)</span></div>
    {valid_chart()}
    <figcaption>원래 형상별 후보 768개 중, e000·e001·e002 세 몸체 모두에서 관통 ≤ 5 mm · 관절 한계 · 유한성 · 공통 명령 범위를 통과한 수.</figcaption>
  </figure>
</section>

<section>
  <h2>2. 관통(> 5 mm) 탈락 수: 보정 전 → 후</h2>
  <div class="tw"><table>
    <thead><tr><th>원래 형상 ↓ / 실행 형상 →</th><th>e000</th><th>e001</th><th>e002</th></tr></thead>
    <tbody>{pen_table()}</tbody>
  </table></div>
  <p class="meta">대각선(회색) = 원래 형상 자신. 보정해도 값이 같고 (9~13), 원래 데이터에 있던 관통이다. 주황 = 보정 전 대부분이 탈락한 칸.</p>
</section>

<section>
  <h2>3. 형상별 몸통 z 이동량</h2>
  <figure>
    {zsvg}
    <figcaption>선택된 192개 시작점 기준. 막대 = 5~95 백분위, 점 = 중앙값 (cm). 양수 = 몸통을 올림 (대상 형상의 다리가 더 김). 원래 형상 자신은 0 (assert 로 확인).</figcaption>
  </figure>
</section>

<section>
  <h2>4. 선택 결과 요약</h2>
  <div class="tw"><table>
    <thead><tr><th>원래 형상</th><th>후보</th><th>세 형상 모두 유효</th><th>선택</th><th>접촉까지 일치</th><th>관절 한계 탈락</th></tr></thead>
    <tbody>{sel_table()}</tbody>
  </table></div>
  <p class="meta">관절 한계 탈락은 보정과 무관 (관절각을 바꾸지 않음). 접촉 일치가 적은 것은 형상마다 발 위치가 달라 실제 접촉이 달라지기 때문이다. 결정: D4 에서는 각 형상의 실제 MJX 접촉을 그대로 TT s0 에 넣는다.</p>
</section>

<section>
  <h2>5. 고정 조건</h2>
  <div class="tw"><table>
    <thead><tr><th>항목</th><th>값</th></tr></thead>
    <tbody>
      <tr><th>형상</th><td>e000, e001, e002</td></tr>
      <tr><th>시작점 출처</th><td>각 형상 rollout <code>seq_seed0</code> 의 test 에피소드, stride 8, 32-state window</td></tr>
      <tr><th>후보 / 선택</th><td>원래 형상별 768 / 최대 192 (접촉 일치 우선, 에피소드 round-robin)</td></tr>
      <tr><th>보정 방식</th><td>관절각·quat·x/y·qvel 유지, 몸통 z 만 이동 → 가장 낮은 발의 바닥 높이를 원래 형상과 같게</td></tr>
      <tr><th>유효 기준</th><td>발 외 관통 ≤ 5 mm, 관절 한계 ±1e-3, quat 정규, 유한, 명령이 세 형상 공통 범위 안</td></tr>
      <tr><th>seed</th><td>20261020</td></tr>
      <tr><th>코드</th><td><code>code/tt_dynamics/start_align.py</code>, <code>d4_morphology.py --audit-only [--align-height]</code></td></tr>
    </tbody>
  </table></div>
</section>

<section class="qa">
  <h2>6. Q&amp;A</h2>
  <div><div class="q">형상마다 s0가 달라서 제대로 평가가 안되고 있는 것이 문제로 기억하는데 맞아?</div>
  <div class="a"><p>거의 맞다. 정확히는 한 형상의 s0 (몸통 z 포함) 를 다른 형상에 그대로 넣어 발 아닌 부위가 바닥을 뚫는 것이 문제였다. e000 출신 시작점은 e001 에서 749/768 이 관통했다 (표 2).</p></div></div>
  <div><div class="q">그럼 s0 문제는 어떻게 해결하면 될까?</div>
  <div class="a"><p>몸통 z 만 형상별로 옮겨 가장 낮은 발 높이를 원래 형상과 맞춘다. 관절각·자세·속도는 그대로라 "같은 s0, 같은 명령" 조건은 유지된다. 발이 떠 있던 구간이면 그 높이도 유지된다.</p></div></div>
  <div><div class="q">이걸 해결하면 TT 가 형상마다 다른 움직임을 예측하는 병목도 풀려?</div>
  <div class="a"><p>이 보정만으로는 아니다. 보정 전에도 접촉까지 일치한 13개에서 TT 의 yaw 반응은 MJX 의 1/5~1/10 이었다. 보정은 평가 표본을 공정하게 만들고, 형상 교차 데이터 (같은 s0·명령을 다른 형상에서 실행) 를 만들 수 있게 하는 전제 조건이다. 병목 자체는 그 데이터로 다시 학습한 뒤 D4 로 확인한다.</p></div></div>
</section>
</main>
"""
(HERE / "report.html").write_text(html)
print(HERE / "report.html")
