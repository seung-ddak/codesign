"""tt_switchE report.html 생성: jumpBoost (전환 미학습) 대비 전환 평가 + 모션별 평가 + 학습 곡선.
usage (codesign/ 에서): python runs/TT_tracking/tt_switchE/make_report.py
"""
import csv
import re
import shutil
from html import escape
from pathlib import Path

R = Path(__file__).resolve().parent
BASE = R.parent / "tt_jumpBoost"
SHAPES = ["00000000", "00000001", "00000010", "00000011"]
SW = "switch_eval_start_end"


def rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def curve(s):
    t = (R / s / "train.log").read_text(errors="ignore")
    g = lambda k: [float(x) for x in re.findall(rf"^eval/{k}: ([-\d.e]+)", t, re.M)]
    return g("episode/reward/mean"), g("episode/done/fellover/mean")


def svg_lines(series, ymin, ymax, label):
    W, H, L, B = 360, 170, 40, 24
    n = max(len(v) for v in series.values())
    x = lambda i: L + i * (W - L - 10) / max(n - 1, 1)
    y = lambda v: 8 + (ymax - v) * (H - B - 8) / (ymax - ymin)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{label}">']
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        out.append(f'<line x1="{L}" x2="{W-10}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{L-4}" y="{y(v)+3:.1f}" class="ax" text-anchor="end">{v:.3g}</text>')
    for i in range(n):
        out.append(f'<text x="{x(i):.1f}" y="{H-8}" class="ax" text-anchor="middle">{(i+1)*5}</text>')
    for j, (s, v) in enumerate(series.items()):
        pts = " ".join(f"{x(i):.1f},{y(a):.1f}" for i, a in enumerate(v))
        out.append(f'<polyline points="{pts}" class="s{j}" fill="none"/>')
    out.append("</svg>")
    return "".join(out)


sw_b = {s: rows(BASE / s / SW / "switch_eval.csv") for s in SHAPES}
sw_e = {s: rows(R / s / SW / "switch_eval.csv") for s in SHAPES}
me_b = {s: rows(BASE / s / "motion_eval/motion_eval.csv") for s in SHAPES}
me_e = {s: rows(R / s / "motion_eval/motion_eval.csv") for s in SHAPES}
cur = {s: curve(s) for s in SHAPES}
P = lambda rs: sum(int(r["pass"]) for r in rs)
PA = lambda rs: sum(int(r["pass_all"]) for r in rs)
held = lambda rs: [r for r in rs if r["kind"] == "held-out"]
trn = lambda rs: [r for r in rs if r["kind"] == "trained"]

# 요약 표
summ = "".join(
    f"<tr><td>{s}</td><td>{P(trn(sw_b[s]))}/6 → <b>{P(trn(sw_e[s]))}/6</b></td>"
    f"<td>{P(held(sw_b[s]))}/3 → <b>{P(held(sw_e[s]))}/3</b></td>"
    f"<td>{PA(me_b[s])}/{len(me_b[s])} → <b>{PA(me_e[s])}/{len(me_e[s])}</b></td>"
    f"<td>{cur[s][0][0]:.1f} → {cur[s][0][-1]:.1f}</td><td>{cur[s][1][-1]:.2f}</td></tr>" for s in SHAPES)
tot = lambda d, f, k=None: sum(f(k(d[s]) if k else d[s]) for s in SHAPES)

# 전환 쌍 상세 (형상 × 쌍, recover 초)
def cell(rb, re_):
    cls = "ok" if int(re_["pass"]) else "bad"
    rec = lambda r: "—" if float(r["recover_s"]) < 0 else f'{float(r["recover_s"]):.2f}'
    return (f'<td class="{cls}">{rec(rb)} → <b>{rec(re_)}</b> s<br><small>관절 {float(re_["joint_deg"]):.1f}° · '
            f'자세 {float(re_["orn_deg"]):.1f}°</small></td>')

pair_rows = ""
for i, r0 in enumerate(sw_e[SHAPES[0]]):
    pair_rows += (f'<tr><th>{r0["A"]} → {r0["B"]}<br><small>{"학습 포함" if r0["kind"] == "trained" else "held-out"}</small></th>'
                  + "".join(cell(sw_b[s][i], sw_e[s][i]) for s in SHAPES) + "</tr>")

# 모션 평가: 변화가 있는 모션만
mot = ""
for i, r0 in enumerate(me_e[SHAPES[0]]):
    pb = [int(me_b[s][i]["pass_all"]) for s in SHAPES]
    pe = [int(me_e[s][i]["pass_all"]) for s in SHAPES]
    if pb == pe and all(pe):
        continue
    def mc(s):
        b, e = me_b[s][i], me_e[s][i]
        fails = [k[5:] for k in e if k.startswith("pass_") and k != "pass_all" and e[k] == "0"]
        return (f'<td class="{"ok" if int(e["pass_all"]) else "bad"}">{"통과" if int(b["pass_all"]) else "실패"} → '
                f'<b>{"통과" if int(e["pass_all"]) else "실패"}</b><br><small>관절 {float(e["joint_rms_deg"]):.1f}° · '
                f'끝 위치 {float(e["end_pos_err_cm"]):.1f} cm{" · " + ", ".join(fails) if fails else ""}</small></td>')
    mot += f'<tr><th>{r0["motion"]}</th>' + "".join(mc(s) for s in SHAPES) + "</tr>"

# 영상: switchE 전환 영상을 report 옆 videos/ 로 복사
vids = ""
for s in SHAPES:
    vd = R / "videos" / s
    vd.mkdir(parents=True, exist_ok=True)
    items = ""
    for r in sw_e[s]:
        fn = f'{r["A"]}_to_{r["B"]}.mp4'
        shutil.copy(R / s / SW / "videos" / fn, vd / fn)
        tag = "통과" if int(r["pass"]) else "실패"
        items += (f'<figure><video src="videos/{s}/{fn}" controls muted loop playsinline preload="metadata"></video>'
                  f'<figcaption><span class="pill {"ok" if int(r["pass"]) else "bad"}">{tag}</span> {r["A"]} → {r["B"]}</figcaption></figure>')
    vids += f'<details{" open" if s == SHAPES[0] else ""}><summary>형상 {s}</summary><div class="vgrid">{items}</div></details>'

legend = "".join(f'<span class="lg"><i class="s{j}"></i>{s}</span>' for j, s in enumerate(SHAPES))
rew = svg_lines({s: cur[s][0] for s in SHAPES}, 15, 20, "평가 보상")
fall = svg_lines({s: cur[s][1] for s in SHAPES}, 0, 0.06, "넘어짐 비율")

# 결과 요약 문장 (수치 기반 자동 생성)
fails = [f'{s} {r["A"]}→{r["B"]}' for s in SHAPES for r in sw_e[s] if not int(r["pass"])]
reg = [f'{s} {e["motion"]}' for s in SHAPES for b, e in zip(me_b[s], me_e[s]) if int(b["pass_all"]) and not int(e["pass_all"])]
fix = [f'{s} {e["motion"]}' for s in SHAPES for b, e in zip(me_b[s], me_e[s]) if not int(b["pass_all"]) and int(e["pass_all"])]
sb, se = tot(sw_b, P), tot(sw_e, P)
verdict = "개선" if se > sb else ("동일" if se == sb else "악화")
analysis = "".join(f"<li>{x}</li>" for x in [
    f"전환 통과 {sb}/36 → <b>{se}/36</b> ({verdict}). 학습 포함 쌍 {tot(sw_b, P, trn)}/24 → {tot(sw_e, P, trn)}/24, "
    f"held-out {tot(sw_b, P, held)}/12 → {tot(sw_e, P, held)}/12",
    "남은 실패: " + (", ".join(fails) if fails else "없음"),
    f"모션별 tracking {tot(me_b, PA)}/120 → {tot(me_e, PA)}/120. 새로 실패: " + (", ".join(reg) or "없음")
    + " · 새로 통과: " + (", ".join(fix) or "없음"),
    f"일반 tracking 평가 (전환 없음) 보상 {sum(cur[s][0][0] for s in SHAPES)/4:.1f} → {sum(cur[s][0][-1] for s in SHAPES)/4:.1f}, "
    f"넘어짐 0 → {sum(cur[s][1][-1] for s in SHAPES)/4:.2f} (형상 평균)",
])

html = (R / "template.html").read_text()
for k, v in dict(ANALYSIS=analysis,SUMM=summ, PAIRS=pair_rows, MOT=mot or '<tr><td colspan="5">변화 없음</td></tr>', VIDS=vids,
                 REW=rew, FALL=fall, LEGEND=legend,
                 TB=f"{tot(sw_b, P, trn)}/24", TE=f"{tot(sw_e, P, trn)}/24",
                 HB=f"{tot(sw_b, P, held)}/12", HE=f"{tot(sw_e, P, held)}/12",
                 MB=f"{tot(me_b, PA)}/120", ME=f"{tot(me_e, PA)}/120").items():
    html = html.replace("{{" + k + "}}", v)
(R / "report.html").write_text(html)
print(R / "report.html")
