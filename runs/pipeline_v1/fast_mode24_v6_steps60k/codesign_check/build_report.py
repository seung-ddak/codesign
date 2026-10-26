"""co-design 검사 report.html. analyze.py 의 codesign_check.json + summary.json 4개. 그림·영상 base64 내장. 영상 = render/ (같은 명령, 렌더만 켠 재실행), 320 px · 25 fps 로 줄여 넣는다."""
import base64, io, json, tempfile
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

here = Path(__file__).parent
css = (here.parent / "report_template.html").read_text().split("<main>")[0].replace("Guidance Scale Sweep", "Co-design Check")
cc = json.load(open(here / "codesign_check.json"))
designs = [k for k in json.load(open(here.parents[1] / "retargeting/designs.json")) if k.startswith("e") and k != "e003"]


def uri(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def fig_uri(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True); plt.close(fig)
    return uri(buf.getvalue(), "image/png")


def style(a):
    a.grid(alpha=.25)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for el in [a.xaxis.label, a.yaxis.label, a.title, *a.get_xticklabels(), *a.get_yticklabels()]:
        el.set_color("#6b7a82")


per, rho = {}, {}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, task in zip(axs, ("zigzag", "circle")):
    rs = json.load(open(here / f"fixedE_{task}_4s/summary.json"))["runs"]["scale_10"]
    jt = json.load(open(here / f"joint_{task}_4s/summary.json"))["runs"]["scale_10"]
    g = np.repeat(np.arange(16), len(rs) // 16)
    m = np.array([r["mjx_total"] for r in rs]); p = np.array([r["pred_total"] for r in rs])
    prog = np.array([r["mjx_progress"] for r in rs]); pe = np.array([r["path_err_m"] for r in rs])
    done = np.array([r["mjx_fell_step"] == -1 for r in rs])
    per[task] = [dict(name=designs[k], mjx=m[g == k].mean(), pred=p[g == k].mean(), prog=prog[g == k].mean(),
                      pe=np.median(pe[g == k]), done=int(done[g == k].sum())) for k in range(16)]
    dm = [d["mjx"] for d in per[task]]; dp = [d["pred"] for d in per[task]]
    rho[task] = spearmanr(dm, dp)
    ax.scatter(dp, dm, color="#0f6f86", s=28, label="fixed E (design mean, 4 seeds)")
    jp = [r["pred_total"] for r in jt]; jm = [r["mjx_total"] for r in jt]
    ax.scatter(jp, jm, color="#c06a1b", marker="x", s=28, label="joint-generated E (16)")
    ax.set_xlabel("TT predicted R"); ax.set_ylabel("MJX actual R"); ax.set_title(f"{task} 4 s, scale 10", fontsize=10)
    style(ax)
axs[0].legend(frameon=False, fontsize=8)
scatter = fig_uri(fig)

paths = "".join(f'<figure class="fig"><img src="{uri((here / f"{k}_{t}_4s/paths.png").read_bytes(), "image/png")}" alt="{k} {t} 경로" style="max-width:100%">'
                f'<figcaption>{k} · {t} (점선 TT 예측, 실선 MJX, x 낙상)</figcaption></figure>'
                for t in ("zigzag", "circle") for k in ("fixedE", "joint"))


def row_sum(task, unit):
    c = cc[task]; j = c["joint"]
    return (f"<tr><td>{task}</td><td>{c['done']}/{c['n']}</td><td>{j['done']}/{j['valid']}</td>"
            f"<td>{c['fixed_median_R']:+.2f}</td><td>{j['median_R']:+.2f}</td>"
            f"<td>{c['fixed_median_progress']:+.2f} / {j['median_progress']:+.2f} {unit}</td>"
            f"<td>{c['design_mean_R']['min']:+.2f} / {c['design_mean_R']['median']:+.2f} / {c['design_mean_R']['max']:+.2f}</td>"
            f"<td>{j['percentile_vs_designs']:.0f} %</td></tr>")


def row_eta(task):
    c = cc[task]
    cell = lambda k: f"{c[k]['eta2']:.2f} (p {c[k]['p']:.3f})"  # noqa: E731
    return (f"<tr><td>{task}</td><td>{cell('mjx_R')}</td><td>{cell('mjx_progress')}</td><td>{cell('survive')}</td>"
            f"<td>{cell('pred_R')}</td><td>{rho[task].statistic:+.2f} (p {rho[task].pvalue:.2f})</td></tr>")


def design_rows(task):
    out = []
    for d in sorted(per[task], key=lambda d: -d["mjx"]):
        out.append(f"<tr><td>{d['name']}</td><td>{d['mjx']:+.2f}</td><td>{d['pred']:+.2f}</td><td>{d['prog']:+.2f}</td>"
                   f"<td>{d['pe']:.2f} m</td><td>{d['done']}/4</td></tr>")
    return "\n".join(out)


def small(mp4):
    """원본 640x368 50 fps → 320x184 25 fps (전체 160 개를 16 MB 안에)."""
    fr = [np.asarray(Image.fromarray(f).resize((320, 184))) for k, f in enumerate(imageio.get_reader(mp4)) if k % 2 == 0]
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        imageio.mimsave(tf.name, fr, fps=25, quality=5)
        return uri(Path(tf.name).read_bytes(), "video/mp4")


def card(rdir, i, title, r):
    vid = rdir / "videos" / f"scale_10_cand{i}.mp4"
    ok = r["mjx_fell_step"] == -1
    stat = ("완주" if ok else f"낙상 {r['mjx_valid_s']:.2f}s") if r["mjx_total"] is not None else "S0 무효"
    media = f'<video src="{small(vid)}" controls loop muted playsinline></video>' if vid.exists() else '<p class="dim">영상 없음</p>'
    unit = "바퀴" if "circle" in rdir.name else "m"
    return (f'<article class="card"><header><h3>{title}</h3><span class="pill {"ok" if ok else "bad"}">{stat}</span></header>{media}'
            f'<dl><dt>예측 R</dt><dd>{r["pred_total"]:+.2f}</dd><dt>MJX R</dt><dd>{r["mjx_total"]:+.2f}</dd>'
            f'<dt>MJX progress</dt><dd>{r["mjx_progress"]:+.2f} {unit}</dd></dl></article>')


videos = []
for task in ("zigzag", "circle"):
    jd, fd = here / f"render/joint_{task}_4s", here / f"render/fixedE_{task}_4s"
    jr = json.load(open(jd / "summary.json"))["runs"]["scale_10"]
    fr_ = json.load(open(fd / "summary.json"))["runs"]["scale_10"]
    videos.append(f'<h3>{task} · joint 생성 후보 16</h3><div class="grid">' + "".join(card(jd, i, f"joint 후보 {i}", r) for i, r in enumerate(jr)) + "</div>")
    per_d = len(fr_) // 16
    order = sorted(range(16), key=lambda k: -np.mean([fr_[k * per_d + s]["mjx_total"] for s in range(per_d)]))
    for k in order:
        m = np.mean([fr_[k * per_d + s]["mjx_total"] for s in range(per_d)])
        videos.append(f'<details><summary>{task} · 고정 E {designs[k]} (MJX R 평균 {m:+.2f})</summary><div class="grid">'
                      + "".join(card(fd, k * per_d + s, f"{designs[k]} · seed {s}", fr_[k * per_d + s]) for s in range(per_d)) + "</div></details>")
    same = all(abs(a["mjx_total"] - b["mjx_total"]) < 1e-3 for a, b in zip(fr_, json.load(open(here / f"fixedE_{task}_4s/summary.json"))["runs"]["scale_10"]))
    print(task, "render run 수치 = 원 run 수치:", same)


z, c_ = cc["zigzag"], cc["circle"]
html = css + f"""<main>
<section>
  <div class="eyebrow">codesign · fast_mode24_v6_steps60k · co-design check · 2026-10-25</div>
  <h1>형상 E 가 결과를 바꾸는가, joint 가 좋은 E 를 고르는가</h1>
  <div class="lead">
  <p><b>결론: 형상 E 는 MJX 실제 결과를 크게 바꾸지만 (R 분산의 η² {z['mjx_R']['eta2']:.2f} / {c_['mjx_R']['eta2']:.2f}, p &lt; 0.001), TT 예측은 E 차이를 거의 반영하지 못한다 (예측 R η² {z['pred_R']['eta2']:.2f} / {c_['pred_R']['eta2']:.2f}, p {z['pred_R']['p']:.2f} / {c_['pred_R']['p']:.2f}). 그래서 joint guidance 가 고른 E 는 좋은 형상이 아니다.</b></p>
  <p>joint 생성 E 의 MJX R 중앙값은 학습 형상 16 개의 평균 R 대비 zigzag 하위 {z['joint']['percentile_vs_designs']:.0f} 백분위, circle {c_['joint']['percentile_vs_designs']:.0f} 백분위다. circle 은 위쪽이지만 R 중앙값 {c_['joint']['median_R']:+.2f} 는 거의 제자리 (progress {c_['joint']['median_progress']:+.3f} 바퀴) 라서 "좋은 E 를 골랐다" 기보다 "나쁜 형상을 덜 골랐다" 에 가깝다.</p>
  <p>형상별 평균 기준으로 TT 예측 R 과 MJX R 의 순위 상관은 zigzag {rho['zigzag'].statistic:+.2f} (p {rho['zigzag'].pvalue:.2f}), circle {rho['circle'].statistic:+.2f} (p {rho['circle'].pvalue:.2f}) 로 유의하지 않다. 판정: 병목 = <b>TT 의 E 민감도 부족</b> (wide16 action 실험의 D4 Yaw slope ~0 과 같은 방향). guidance 를 어떻게 바꿔도 TT 가 E 차이를 모르면 E 선택은 개선되지 않는다.</p>
  </div>
</section>

<section>
  <h2>실험 조건</h2>
  <div class="scroll"><table>
    <tr><td class="k">checkpoint</td><td>runs/pipeline_v1/fast_mode24_v6_steps60k/tt_joint (Mode 2+4, 60000 step, 고정)</td></tr>
    <tr><td class="k">비교</td><td>(1) 고정 E: 학습 형상 16 개 (designs.json e000~e016 중 e003 제외) × seed 4 = 64 rollout, 형상마다 같은 seed 집합 (2) joint: E·S·A 동시 생성 후보 16</td></tr>
    <tr><td class="k">과제</td><td>zigzag (amp 0.3 m, period 2 m, lane 반폭 0.15 m) / circle (반지름 1 m, lane 반폭 0.3 m), 4 s, 재계획 없음 (open-loop)</td></tr>
    <tr><td class="k">reward</td><td>R = progress − 20·lane − 20·fall − 0.05·action_rate (circle progress 단위 = 바퀴)</td></tr>
    <tr><td class="k">guidance</td><td>joint full guidance, scale 10 (S 포함), 확산 5 step</td></tr>
    <tr><td class="k">S0</td><td>E-derived home stance (--e-rest-s0 --anchor-s0 --no-s0-align), expert walkForwardSlow__r00</td></tr>
    <tr><td class="k">통계</td><td>η² = 형상 그룹이 설명하는 분산 비율, p = permutation 10000 회. 백분위 = joint MJX R 중앙값보다 낮은 형상 평균의 비율</td></tr>
    <tr><td class="k">영상</td><td>render/ 재실행 (같은 명령 + 렌더). 흰 점 = lane 경계, 파란 점 = TT 예측 몸통 경로. 보고서용으로 320 px · 25 fps 축소</td></tr>
  </table></div>
</section>

<section class="qa">
  <h2>질문과 답</h2>
  <div><p class="q">co-design 검사부터 진행해줘 과제는 지금처럼 지그재그랑 원형 돌기만 해볼게 재계획은 취소하자</p>
    <p class="a">zigzag·circle 두 과제에서 고정 E 64 rollout 과 joint 후보 16 을 4 s open-loop 로 돌렸다 (재계획 없음). 결과는 위 결론과 아래 표.</p></div>
  <div><p class="q">야 최근 학습 종료된거 확인해보고 report 만들어봐</p>
    <p class="a">10-25 16:37~16:46 에 4 run 모두 에러 없이 종료 (MJX 유효 S0 64/64 · 16/16). 이 문서가 그 보고서다.</p></div>
  <div><p class="q">지그재그랑 원형 트랙 도는거 다 영상 찍어서 보고서에 넣어주고 1번으로 진행해보자</p>
    <p class="a">같은 명령에 렌더만 켜서 4 run 을 다시 돌려 160 개 영상을 아래 "MJX 영상" 에 넣었다. 재실행 대조: TT 생성은 원 run 과 비트 동일, MJX R 은 160 개 중 14 개에서 최대 0.07 차이 (낙상 시점 전부 동일) = GPU 실행 간 미세 차이 (8-12). 영상 카드의 수치는 재실행 값, 위 표·통계는 원 run 값. 1번 = E 반사실 데이터 → v7 학습 (remind 8-32).</p></div>
</section>

<section>
  <h2>요약 (MJX)</h2>
  <div class="scroll"><table>
    <tr><th>과제</th><th>고정 E 완주</th><th>joint 완주</th><th>고정 E R 중앙값</th><th>joint R 중앙값</th><th>progress 중앙값 고정 / joint</th><th>형상 평균 R 최소 / 중앙 / 최대</th><th>joint 백분위</th></tr>
    {row_sum('zigzag', 'm')}
    {row_sum('circle', '바퀴')}
  </table></div>
</section>

<section>
  <h2>E 의 효과: MJX 실제 vs TT 예측</h2>
  <div class="scroll"><table>
    <tr><th>과제</th><th>MJX R η²</th><th>MJX progress η²</th><th>MJX 완주 η²</th><th>TT 예측 R η²</th><th>형상 평균 예측-MJX 순위 상관</th></tr>
    {row_eta('zigzag')}
    {row_eta('circle')}
  </table></div>
  <p class="note">MJX 쪽 η² 는 소수의 나쁜 형상 (R −3 ~ −8, 넘어지지 않아도 lane·fall 벌점 큼) 이 크게 끌어올린다. TT 예측 R 은 형상 사이 차이가 seed 차이보다 작다.</p>
  <div class="fig"><img src="{scatter}" alt="TT 예측 R 대 MJX R 산점도" style="max-width:100%"></div>
  <p class="note">점이 대각선을 따르지 않는다: TT 는 거의 모든 형상에 R 0 ~ +1 을 예측하지만 MJX 는 −8 ~ +0.5 로 퍼진다. 4 s 경로 차 중앙값 고정 E 0.26 / 0.31 m, joint 0.30 / 0.21 m (2 s 0.05 m 보다 큼).</p>
</section>

<section>
  <h2>형상별 결과 (MJX R 내림차순)</h2>
  <div class="figs">
    <div class="scroll"><h3>zigzag</h3><table><tr><th>형상</th><th>MJX R</th><th>예측 R</th><th>progress</th><th>경로 차</th><th>완주</th></tr>{design_rows('zigzag')}</table></div>
    <div class="scroll"><h3>circle</h3><table><tr><th>형상</th><th>MJX R</th><th>예측 R</th><th>progress</th><th>경로 차</th><th>완주</th></tr>{design_rows('circle')}</table></div>
  </div>
</section>

<section>
  <h2>경로</h2>
  <div class="figs">{paths}</div>
</section>

<section>
  <h2>MJX 영상</h2>
  <p class="note">joint 후보는 전부 펼쳐 두고, 고정 E 는 형상별로 접어 두었다 (MJX R 평균 내림차순). </p>
  {"".join(videos)}
</section>

<section>
  <h2>다음 선택지 (결정 대기)</h2>
  <div class="lead">
  <p>(1) TT 의 E 민감도 보강: 같은 S0·A 에서 E 만 바꾼 MJX rollout 데이터 추가 (Foot-Anchored v2 의 목적과 같음, 이전에 prepare 단계에서 shared start 0 으로 실패) — 통과 기준 예: 형상 평균 예측-MJX 순위 상관 &gt; 0.6.</p>
  <p>(2) E 선택을 TT 대신 MJX 로: 학습 형상 16 개를 MJX 로 직접 평가해 고르고 TT 는 S·A 생성에만 사용 (연구 정의 변경).</p>
  <p>(3) 4 s 경로 차 0.2~0.3 m 는 open-loop 누적 — E 문제와 별개로 남는다.</p>
  </div>
</section>
</main>
"""
(here / "report.html").write_text(html)
print(here / "report.html", f"{len(html) / 1e6:.2f} MB")
