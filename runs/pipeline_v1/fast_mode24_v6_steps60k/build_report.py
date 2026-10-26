"""v6 guidance scale sweep report.html. 통계 = 후보 16 run, 영상 = 후보 4 렌더 run. 그림·영상 base64 내장."""
import base64, io, json, statistics as st, sys
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

here = Path(__file__).parent
STATS = {"0": "smoke_zigzag_joint_c16", "1": "smoke_zigzag_joint_c16", "3": "smoke_zigzag_joint_c16_scale3_10", "10": "smoke_zigzag_joint_c16_scale3_10"}
RENDER = here / "render_c4_scales"


def uri(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def photo(mp4):
    f = imageio.get_reader(mp4).get_data(0)
    h, w = f.shape[:2]
    buf = io.BytesIO()
    Image.fromarray(f[h // 2 - 120:h // 2 + 120, w // 2 - 180:w // 2 + 180]).save(buf, "PNG")
    return uri(buf.getvalue(), "image/png")


def med(xs):
    return st.median(xs) if xs else float("nan")


rows, agg = [], {}
for sc, folder in STATS.items():
    rs = json.load(open(here / folder / "summary.json"))["runs"][f"scale_{sc}"]
    ex = [r for r in rs if r["mjx_total"] is not None]
    a = dict(valid=len(ex), n=len(rs), done=sum(r["mjx_fell_step"] == -1 for r in ex),
             pe=med([r["path_err_m"] for r in ex]), prog=med([r["mjx_progress"] for r in ex]),
             pprog=med([r["pred_progress"] for r in rs]), R=med([r["mjx_total"] for r in ex]),
             fall=med([20 * r["mjx_terms"]["p_fall"] for r in ex]), pfall=med([20 * r["pred_terms"]["p_fall"] for r in rs]))
    a["pass"] = a["prog"] > 0.15 and a["done"] >= 12 and a["pe"] < 0.15
    agg[sc] = a
    rows.append(f"<tr><td>{sc}</td><td>{a['valid']}/{a['n']}</td><td>{a['done']}/{a['valid']}</td><td>{a['pe']:.2f} m</td>"
                f"<td>{a['pprog']:+.2f} m</td><td>{a['prog']:+.2f} m</td><td>{a['pfall']:.3f} / {a['fall']:.3f}</td><td>{a['R']:+.2f}</td>"
                f"<td><span class='pill {'ok' if a['pass'] else 'bad'}'>{'통과' if a['pass'] else '미달'}</span></td></tr>")

xs = list(agg)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(xs, [agg[s]["pprog"] for s in xs], "o-", color="#8a959c", label="TT predicted")
ax[0].plot(xs, [agg[s]["prog"] for s in xs], "o-", color="#0f6f86", label="MJX actual")
ax[0].axhline(0.15, ls="--", lw=1, color="#1d7a46"); ax[0].set_title("progress median (m)", fontsize=10)
ax[1].plot(xs, [agg[s]["pe"] for s in xs], "o-", color="#c06a1b")
ax[1].axhline(0.15, ls="--", lw=1, color="#1d7a46"); ax[1].set_title("path error median (m)", fontsize=10)
for a_ in ax:
    a_.set_xlabel("guidance scale"); a_.grid(alpha=.25)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    for el in [a_.xaxis.label, a_.title, *a_.get_xticklabels(), *a_.get_yticklabels()]:
        el.set_color("#6b7a82")
ax[0].legend(frameon=False, fontsize=8)
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True); plt.close(fig)
chart = uri(buf.getvalue(), "image/png")

cards = []
for rdir, tag, scales in ((RENDER, "2 s", xs), (here / "render_c4_scale10_nogs", "2 s · no-guide-s", ["10"]),
                          (here / "render_10s_c2", "10 s", ["0", "10"])):
  rsum = json.load(open(rdir / "summary.json"))["runs"]
  for sc in scales:
    for i, r in enumerate(rsum[f"scale_{sc}"]):
        vid = rdir / "videos" / f"scale_{sc}_cand{i}.mp4"
        ok = r["mjx_fell_step"] == -1
        stat = ("완주" if ok else f"낙상 {r['mjx_valid_s']:.2f}s") if r["mjx_total"] is not None else "S0 무효"
        media = (f'<img src="{photo(vid)}" alt="scale {sc} 후보 {i} 형상"><video src="{uri(vid.read_bytes(), "video/mp4")}" controls loop muted playsinline></video>'
                 if vid.exists() else '<p class="dim">영상 없음 (MJX 실행 제외)</p>')
        f = lambda k: f"{r[k]:+.2f}" if r.get(k) is not None else "-"  # noqa: E731
        cards.append(f"""<article class="card"><header><h3>{tag} · scale {sc} · 후보 {i}</h3><span class="pill {'ok' if ok else 'bad'}">{stat}</span></header>
<div class="media">{media}</div><dl><dt>예측 progress</dt><dd>{f('pred_progress')}</dd><dt>MJX progress</dt><dd>{f('mjx_progress')}</dd>
<dt>경로 차</dt><dd>{(f"{r['path_err_m']:.2f} m" if r.get('path_err_m') is not None else '-')}</dd></dl></article>""")

html = (here / "report_template.html").read_text()
for k, v in dict(ROWS="\n".join(rows), CHART=chart, CARDS="\n".join(cards), VERDICT=(here / "verdict.html").read_text()).items():
    html = html.replace("{{" + k + "}}", v)
(here / "report.html").write_text(html)
print(here / "report.html", f"{len(html) / 1e6:.1f} MB")
print({s: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items()} for s, a in agg.items()}, file=sys.stderr)
