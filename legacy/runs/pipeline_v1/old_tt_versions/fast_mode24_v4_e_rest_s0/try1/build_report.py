"""render_rerun/ 결과로 report.html 생성. 형상 사진 = MJX 영상 첫 프레임(S0 자세) 중앙 crop, 영상/그림은 base64 내장."""
import base64, io, json, sys
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image

run = Path(__file__).parent
out = run / "render_rerun"
orig = json.load(open(run / "smoke_zigzag_joint/summary.json"))
summ = json.load(open(out / "summary.json"))
chk = json.load(open(out / "smoke_check.json")) if (out / "smoke_check.json").exists() else None


def b64(p, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode()


def photo(mp4):
    f = imageio.get_reader(mp4).get_data(0)
    h, w = f.shape[:2]
    im = Image.fromarray(f[h // 2 - 120:h // 2 + 120, w // 2 - 180:w // 2 + 180])
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


cards, rows, same = [], [], True
for scale, label in (("scale_0", "guidance 없음 (baseline)"), ("scale_1", "guidance 적용")):
    for i, r in enumerate(summ["runs"][scale]):
        o = orig["runs"][scale][i]
        same &= max(abs(a - b) for a, b in zip(r["E"], o["E"])) < 1e-3
        name = f"{scale}_cand{i}"
        mass = json.load(open(out / "designs" / f"{name}.json"))["total_mass_kg"]
        ok = r["mjx_fell_step"] == -1
        mjx = r["mjx_total"]
        stat = "완주" if ok else f"낙상 {r['mjx_valid_s']:.2f}s"
        vid = out / "videos" / f"{name}.mp4"
        media = (f'<img src="{photo(vid)}" alt="{name} 형상"><video src="{b64(vid, "video/mp4")}" controls loop muted playsinline></video>'
                 if vid.exists() else '<p class="none">영상 없음 (실행 skip)</p>')
        cards.append(f"""<article class="card"><header><h3>{label} · 후보 {i}</h3>
<span class="pill {'ok' if ok else 'bad'}">{stat}</span></header>
<div class="media">{media}</div>
<dl><dt>TT 예측 R</dt><dd>{r['pred_total']:.2f}</dd><dt>MJX 실제 R</dt><dd>{mjx:.2f}</dd>
<dt>예측-실제 경로 차</dt><dd>{r['path_err_m']:.2f} m</dd><dt>총질량</dt><dd>{mass:.2f} kg</dd></dl>
<p class="vec">E = [{', '.join(f'{x:.2f}' for x in r['E'])}]</p></article>""")
        rows.append(f"<tr><td>{label}</td><td>{i}</td><td>{r['pred_total']:.2f}</td><td>{mjx:.2f}</td>"
                    f"<td>{r['pred_total'] - mjx:+.2f}</td><td>{stat}</td></tr>")

tpl = (run / "report_template.html").read_text()
html = (tpl.replace("{{ROWS}}", "\n".join(rows)).replace("{{CARDS}}", "\n".join(cards))
        .replace("{{PATHS}}", b64(out / "paths.png", "image/png"))
        .replace("{{SAME}}", "원래 run 과 E 일치 (seed 재현 확인)" if same else "주의: 원래 run 과 E 가 다름 (재현 실패, 이 보고서는 재실행 기준)"))
(run / "report.html").write_text(html)
print("same E:", same, "->", run / "report.html", f"{len(html) / 1e6:.1f} MB")
