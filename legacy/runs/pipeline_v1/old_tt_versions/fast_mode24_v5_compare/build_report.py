"""v4 / v5_orig / v5_paired joint TT smoke 비교 report.html. 그림·영상 base64 내장 (폴더당 report 하나)."""
import base64, io, json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

here = Path(__file__).parent
pv = here.parent
RUNS = [  # (이름, 설명, smoke 폴더, checkpoint 폴더, 영상 카드 포함)
    ("v4", "원본 데이터 · 3000 step", pv / "fast_mode24_v4_e_rest_s0_try1/render_rerun", pv / "fast_mode24_v4/tt_joint", False),
    ("v5_orig", "원본 데이터 · 20000 step", pv / "fast_mode24_v5_orig/smoke_zigzag_joint", pv / "fast_mode24_v5_orig/tt_joint", True),
    ("v5_paired", "원본 + 교차 재실행 · 20000 step", pv / "fast_mode24_v5_paired/smoke_zigzag_joint", pv / "fast_mode24_v5_paired/tt_joint", True),
]


def uri(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return uri(buf.getvalue(), "image/png")


def photo(mp4):
    f = imageio.get_reader(mp4).get_data(0)
    h, w = f.shape[:2]
    buf = io.BytesIO()
    Image.fromarray(f[h // 2 - 120:h // 2 + 120, w // 2 - 180:w // 2 + 180]).save(buf, "PNG")
    return uri(buf.getvalue(), "image/png")


def agg(rs):
    ex = [r for r in rs if r["mjx_total"] is not None]
    pe = [r["path_err_m"] for r in ex if r["path_err_m"] is not None]
    return dict(valid=len(ex), n=len(rs), done=sum(r["mjx_fell_step"] == -1 for r in ex),
                pe=np.mean(pe) if pe else float("nan"), pe_rng=(min(pe), max(pe)) if pe else (np.nan, np.nan),
                mjx=np.mean([r["mjx_total"] for r in ex]) if ex else float("nan"),
                gap=np.mean([abs(r["pred_total"] - r["mjx_total"]) for r in ex]) if ex else float("nan"))


summary_rows, cards, curves = [], [], {}
for name, desc, smoke, ck, show in RUNS:
    s = json.load(open(smoke / "summary.json"))
    for scale, lab in (("scale_0", "guidance 없음"), ("scale_1", "guidance 적용")):
        a = agg(s["runs"][scale])
        summary_rows.append(f"<tr><td>{name}</td><td>{desc}</td><td>{lab}</td><td>{a['valid']}/{a['n']}</td><td>{a['done']}/{a['valid']}</td>"
                            f"<td>{a['pe']:.2f} m <span class=dim>({a['pe_rng'][0]:.2f}~{a['pe_rng'][1]:.2f})</span></td>"
                            f"<td>{a['gap']:.2f}</td><td>{a['mjx']:.2f}</td></tr>")
        if show:
            for i, r in enumerate(s["runs"][scale]):
                vid = smoke / "videos" / f"{scale}_cand{i}.mp4"
                ok = r["mjx_fell_step"] == -1
                stat = ("완주" if ok else f"낙상 {r['mjx_valid_s']:.2f}s") if r["mjx_total"] is not None else "S0 무효"
                media = (f'<img src="{photo(vid)}" alt="{name} {scale} 후보 {i} 형상"><video src="{uri(vid.read_bytes(), "video/mp4")}" controls loop muted playsinline></video>'
                         if vid.exists() else '<p class="dim">영상 없음 (MJX 실행 제외)</p>')
                mjx = f"{r['mjx_total']:.2f}" if r["mjx_total"] is not None else "-"
                pe = f"{r['path_err_m']:.2f} m" if r.get("path_err_m") is not None else "-"
                cards.append(f"""<article class="card"><header><h3>{name} · {lab} · 후보 {i}</h3>
<span class="pill {'ok' if ok else 'bad'}">{stat}</span></header><div class="media">{media}</div>
<dl><dt>TT 예측 R</dt><dd>{r['pred_total']:.2f}</dd><dt>MJX 실제 R</dt><dd>{mjx}</dd><dt>경로 차</dt><dd>{pe}</dd></dl></article>""")
    prog = [json.loads(l) for l in open(ck / "progress.jsonl") if '"val_loss"' in l]
    curves[name] = ([p["step"] for p in prog], [p["val_loss"] for p in prog])

fig, ax = plt.subplots(figsize=(7, 3.2))
for (name, (x, y)), c in zip(curves.items(), ("#8a959c", "#0f6f86", "#c06a1b")):
    ax.plot(x, y, label=name, color=c, lw=1.8)
ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.grid(alpha=.25); ax.legend(frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for el in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
    el.set_color("#6b7a82")
loss_png = png(fig)

paths = "".join(f'<figure class="fig"><img src="{uri((smoke / "paths.png").read_bytes(), "image/png")}" alt="{name} 경로"><figcaption>{name}</figcaption></figure>'
                for name, _, smoke, _, _ in RUNS[1:])
html = (here / "template.html").read_text()
for k, v in dict(SUMMARY="\n".join(summary_rows), CARDS="\n".join(cards), LOSS=loss_png, PATHS=paths,
                 VERDICT=(here / "verdict.html").read_text()).items():
    html = html.replace("{{" + k + "}}", v)
(here / "report.html").write_text(html)
print(here / "report.html", f"{len(html) / 1e6:.1f} MB")
