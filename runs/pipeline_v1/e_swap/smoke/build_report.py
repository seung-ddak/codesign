"""E 반사실 데이터 smoke (remind 8-32) 영상 + report.html. 저장된 qpos 를 그대로 재생 (재실행 없음).

행 = 같은 행동 A (원본 형상 · episode), 열 = 원본 episode (원래 형상, 원래 시작) / 대상 형상 e000 · e008 · e016 (home 안착 S0).
usage (codesign/): MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python runs/pipeline_v1/e_swap/smoke/build_report.py
"""
import base64, json, pickle, sys
from pathlib import Path

here = Path(__file__).resolve().parent
ROOT = here.parents[3]
sys.path[:0] = [str(ROOT.parent / "transformer-transformer"), str(ROOT / "code/motion_tracking")]

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from pd_replay import build_model  # noqa: E402

TARGETS = ["e000", "e008", "e016"]
DATA = ROOT / "runs/pipeline_v1/wide16_action/rollouts"
cfg = pickle.load(open(sorted((ROOT / "runs/pipeline_v1/experts_wide16/walkForwardSlow__r00").glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb"))
designs = json.loads((ROOT / "runs/pipeline_v1/retargeting/designs.json").read_text())
model_of = {}


def model(e):
    if e not in model_of:
        model_of[e] = build_model(cfg.get("choice"), kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                  torque_limit=cfg.get("torque_limit"), uniforms=np.asarray(designs[e], np.float64))
    return model_of[e]


def render(m, qs, path, size=(320, 240)):
    """qpos 재생. 카메라는 몸통을 따라간다 (generate_task.render 와 같은 각도)."""
    m.vis.global_.offwidth, m.vis.global_.offheight = size
    r, d = mujoco.Renderer(m, size[1], size[0]), mujoco.MjData(m)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 1.8, 135.0, -30.0
    frames = []
    for q in qs[::2]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        cam.lookat[:] = q[:3]
        r.update_scene(d, camera=cam, scene_option=opt)
        frames.append(r.render())
    imageio.mimsave(path, frames, fps=25, quality=5)
    r.close()


def yaw(q):
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def stats(qs):
    dx, dy = qs[-1, :2] - qs[0, :2]
    return dict(dx=float(dx), dy=float(dy), dyaw=float((yaw(qs[-1]) - yaw(qs[0]) + 180) % 360 - 180))


def uri(p):
    return "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()


(here / "videos").mkdir(exist_ok=True)
D = {j: np.load(here / j / "primitive_seed0.npz") for j in TARGETS}
src, ep, motions = D["e000"]["source_design"], D["e000"]["source_episode"], D["e000"]["motions"]
clip = D["e000"]["clip"][:, 0]
T = D["e000"]["qpos"].shape[1]
rows, table, spread, odx = [], [], [], []
for i in range(len(ep)):
    with np.load(DATA / src[i] / "primitive_seed0.npz") as z:
        oq = z["qpos"][ep[i], :T]
    cells, st = [], {}
    for col, (name, m, qs, fell) in enumerate([("원본 " + src[i], model(src[i]), oq, False)] +
                                              [(j, model(j), D[j]["qpos"][i, :D[j]["valid"][i].sum()], bool(D[j]["fell"][i])) for j in TARGETS]):
        p = here / "videos" / f"a{i:02d}_{'orig' if col == 0 else name}.mp4"
        if not p.exists():
            render(m, qs, p)
        s = stats(qs)
        st[name] = s
        tag = f'<span class="pill bad">낙상 {len(qs) * 0.02:.2f}s</span>' if fell else ""
        cells.append(f'<figure class="fig"><video src="{uri(p)}" controls loop muted playsinline preload="none"></video>'
                     f'<figcaption>{name} {tag}<br>전진 {s["dx"]:+.2f} m · yaw {s["dyaw"]:+.0f}°</figcaption></figure>')
    rows.append(f'<article class="card"><header><h3>A{i:02d} · {src[i]} episode {ep[i]} · {motions[clip[i]]}</h3></header>'
                f'<div class="vids">{"".join(cells)}</div></article>')
    odx.append(st["원본 " + src[i]]["dx"])
    dxs, yws = [st[j]["dx"] for j in TARGETS], [st[j]["dyaw"] for j in TARGETS]
    spread.append((max(dxs) - min(dxs), max(yws) - min(yws)))
    table.append(f"<tr><td>A{i:02d}</td><td>{src[i]}</td><td>{motions[clip[i]]}</td><td>{st['원본 ' + src[i]]['dx']:+.2f}</td>"
                 + "".join(f"<td>{st[j]['dx']:+.2f} / {st[j]['dyaw']:+.0f}°{' ✕' if D[j]['fell'][i] else ''}</td>" for j in TARGETS) + "</tr>")

orig_dx = np.array(odx)
corr = {j: float(np.corrcoef(orig_dx, D[j]["qpos"][:, -1, 0] - D[j]["qpos"][:, 0, 0])[0, 1]) for j in TARGETS}
sp = np.array(spread)
fell = {j: int(D[j]["fell"].sum()) for j in TARGETS}
css = (ROOT / "runs/pipeline_v1/fast_mode24_v6_steps60k/report_template.html").read_text().split("<main>")[0].replace("Guidance Scale Sweep", "E-swap Smoke")
css = css.replace("</style>", ".vids { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)); gap: 8px; }\n"
                  ".vids video { width: 100%; border-radius: 4px; display: block; background: var(--line); aspect-ratio: 4 / 3; }\n"
                  ".vids .fig { padding: 6px; }\ndetails summary { cursor: pointer; }\n</style>")
html = css + f"""<main>
<section>
  <div class="eyebrow">codesign · e_swap smoke · 2026-10-26</div>
  <h1>E 반사실 데이터 smoke: 같은 행동, 다른 형상</h1>
  <div class="lead">
  <p><b>결론: smoke 통과. 같은 형상에서 home 자세로 시작해 원본 행동을 재생하면 원본 episode 의 전진을 재현하고 (e000 상관 {corr['e000']:+.2f}), 형상을 바꾸면 같은 행동의 결과가 크게 달라진다 (형상 3 개 사이 전진 차 중앙값 {np.median(sp[:, 0]):.2f} m, 최대 {sp[:, 0].max():.2f} m · yaw 차 중앙값 {np.median(sp[:, 1]):.0f}°, 최대 {sp[:, 1].max():.0f}°).</b> TT 가 E 효과를 배울 신호가 데이터에 있다.</p>
  <p>낙상 e000 {fell['e000']}/16 · e008 {fell['e008']}/16 · e016 {fell['e016']}/16. 원본 전진과의 상관은 e000 {corr['e000']:+.2f} · e008 {corr['e008']:+.2f} · e016 {corr['e016']:+.2f} (원본 행동 절반이 e008 것이라 e000 상관도 형상 차를 포함한다).</p>
  </div>
</section>

<section>
  <h2>실험 조건</h2>
  <div class="scroll"><table>
    <tr><td class="k">목적</td><td>TT 가 형상 E 를 무시하는 문제 (remind 8-31) 해결용 데이터. 같은 A 를 여러 E 에서 실행해 E 효과만 분리</td></tr>
    <tr><td class="k">행동 A</td><td>wide16 expert rollout 의 target_q 앞 100 step (2 s). 원본 형상 e000 · e008 각 8 episode (seed 0 무작위, 100 step 이상 유효한 episode 중)</td></tr>
    <tr><td class="k">대상 형상</td><td>e000 · e008 · e016</td></tr>
    <tr><td class="k">시작 상태</td><td>각 대상 형상의 home 자세를 발 최저점에 맞춰 1 s 안착 (S0 = g(E), generate_task --e-rest-s0 와 같음). qpos 를 형상 사이에 공유하지 않는다</td></tr>
    <tr><td class="k">실행</td><td>MJX 50 Hz open-loop, target_q 를 대상 형상 관절 범위로 clip, 넘어지면 그 뒤 valid=False</td></tr>
    <tr><td class="k">원본 열</td><td>같은 A 의 원래 episode (원래 형상, expert 가 closed-loop 로 만든 시작 상태)</td></tr>
    <tr><td class="k">코드</td><td>code/tt_dynamics/collect_e_swap.py, 이 보고서 runs/pipeline_v1/e_swap/smoke/build_report.py</td></tr>
  </table></div>
</section>

<section class="qa">
  <h2>질문과 답</h2>
  <div><p class="q">그래 그렇게 진행해봐</p>
    <p class="a">수집 스크립트를 만들고 형상 3 개로 smoke 를 돌렸다 (이 문서). 통과해서 전체 수집 (원본 16 × 32 episode × 대상 16 = 8,192 rollout) → v7 학습 → co-design 검사 재실행 체인을 tmux v7_eswap 에서 시작했다.</p></div>
  <div><p class="q">smoke 성공한거 영상 녹화해서 report 만들고 깃허브에 커밋 및 푸시해둬</p>
    <p class="a">저장된 qpos 를 그대로 재생해 64 개 영상 (행동 16 × 원본 + 형상 3) 을 아래에 넣었다.</p></div>
</section>

<section>
  <h2>행동별 결과 (전진 m / yaw, ✕ = 낙상)</h2>
  <div class="scroll"><table>
    <tr><th>A</th><th>원본 형상</th><th>모션</th><th>원본 전진</th>{"".join(f"<th>{j}</th>" for j in TARGETS)}</tr>
    {"".join(table)}
  </table></div>
</section>

<section>
  <h2>영상</h2>
  <p class="note">한 행 = 같은 행동 A. 왼쪽부터 원본 episode, e000, e008, e016. 2 s, 25 fps.</p>
  <div class="grid" style="grid-template-columns: 1fr">{"".join(rows)}</div>
</section>
</main>
"""
(here / "report.html").write_text(html)
print(here / "report.html", f"{len(html) / 1e6:.1f} MB", "corr", corr, "fell", fell)
