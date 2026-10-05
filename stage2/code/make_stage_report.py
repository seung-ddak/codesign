"""report.html for T6 (evaluator check), T7 (TT DDPO training) and T8 (TaskEnv π_θ) runs, same styling as T5.

  python stage2/code/make_stage_report.py t6 --run stage2/runs/T6/try1 --title "T6 try1"
  python stage2/code/make_stage_report.py t7 --run stage2/runs/T7/try1 --title "T7 try1"

T6 reads t6_result.json, eval_rep1.json, eval_rep2.json. T7 reads config.json and history.jsonl.
Charts are static inline SVG (no script), so the file works as a Telegram attachment.
"""

import argparse
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAIN_LEGS = lambda c: not any(c[1:5])  # bits 1-4: every leg is the plain (non-linkage) leg


def pct(v):
    return "—" if v is None else f"{v * 100:.0f}%"


def tiles(rows):
    return '<div class="verdict">' + "".join(
        f'<div><span class="k">{k}</span><span class="v">{v}</span><span class="d">{d}</span></div>'
        for k, v, d in rows) + "</div>"


def table(head, rows):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f'<td class="mono">{x}</td>' for x in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def line_svg(series, ylabel, ref=None, w=460, h=220):
    """series: [(name, css_color_var, [(x, y), ...])]."""
    m = dict(l=48, r=14, t=10, b=26)
    xs = [x for _, _, p in series for x, _ in p] or [0, 1]
    ys = [y for _, _, p in series for _, y in p if y is not None] + ([ref] if ref is not None else [])
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    y0, y1 = min(ys + [0]), max(ys + [1e-6])
    pad = (y1 - y0) * .08 or 1
    y1 += pad
    X = lambda v: m["l"] + (v - x0) / (x1 - x0) * (w - m["l"] - m["r"])
    Y = lambda v: h - m["b"] - (v - y0) / (y1 - y0) * (h - m["t"] - m["b"])
    g = []
    for i in range(6):
        v = y0 + (y1 - y0) * i / 5
        g.append(f'<line x1="{m["l"]}" x2="{w - m["r"]}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/>'
                 f'<text x="{m["l"] - 6}" y="{Y(v) + 3.5:.1f}" text-anchor="end">{v:.3g}</text>')
    for i in range(6):
        v = x0 + (x1 - x0) * i / 5
        g.append(f'<text x="{X(v):.1f}" y="{h - 8}" text-anchor="middle">{v:.0f}</text>')
    if ref is not None:
        g.append(f'<line x1="{m["l"]}" x2="{w - m["r"]}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" '
                 f'stroke="var(--ink2)" stroke-dasharray="5 4"/>')
    for name, color, pts in series:
        pts = [(x, y) for x, y in pts if y is not None]
        if pts:
            d = "".join(("M" if i == 0 else "L") + f"{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
            g.append(f'<path d="{d}" fill="none" stroke="var({color})" stroke-width="2"><title>{name}</title></path>')
    legend = "".join(f'<span><i style="background:var({c})"></i>{n}</span>' for n, c, _ in series)
    return (f'<figure><figcaption>{ylabel}</figcaption><div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">{"".join(g)}</svg></figure>')


def scatter_svg(pts, xlabel, ylabel, w=460, h=300):
    m = dict(l=48, r=14, t=10, b=30)
    vs = [v for p in pts for v in p] or [0, 1]
    lo, hi = min(vs), max(vs)
    pad = (hi - lo) * .05 or 1
    lo, hi = lo - pad, hi + pad
    X = lambda v: m["l"] + (v - lo) / (hi - lo) * (w - m["l"] - m["r"])
    Y = lambda v: h - m["b"] - (v - lo) / (hi - lo) * (h - m["t"] - m["b"])
    g = [f'<line x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}" stroke="var(--muted)" stroke-dasharray="4 4"/>']
    for i in range(6):
        v = lo + (hi - lo) * i / 5
        g.append(f'<text x="{m["l"] - 6}" y="{Y(v) + 3.5:.1f}" text-anchor="end">{v:.3g}</text>'
                 f'<text x="{X(v):.1f}" y="{h - 12}" text-anchor="middle">{v:.3g}</text>')
    g += [f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="4" fill="var(--s1)" fill-opacity=".7" stroke="var(--surface)"/>'
          for a, b in pts]
    return (f'<figure><figcaption>{ylabel} <span>· 가로 {xlabel}, 점선 = 완전 일치</span></figcaption>'
            f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">{"".join(g)}</svg></figure>')


def page(title, eyebrow, lede, sections):
    css = (HERE / "report_style.css").read_text()
    body = "".join(f"<section><h2>{h}</h2>{c}</section>" for h, c in sections)
    return (f'<title>{html.escape(title)}</title>\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
            f'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">'
            f'<style>{css}</style><div class="wrap"><header><div class="eyebrow">{eyebrow}</div>'
            f'<h1>{html.escape(title)}</h1><p class="lede">{lede}</p></header>{body}</div>')


def t6(run, title):
    r = json.loads((run / "t6_result.json").read_text())
    reps = [json.loads((run / f"eval_rep{i}.json").read_text()) for i in (1, 2)]
    ret = [{(tuple(d["choices"]), d["seed"]): d["return"] for d in e["designs"]} for e in reps]
    c = r["unseen_t5_criteria"]
    lede = (f"T5 정책을 동결하고, 학습에 쓰지 않은 seed 로 만든 형상 {r['designs']} 개를 두 번 채점했습니다. "
            f"결과: <b>{'통과' if r['pass'] else '불합격'}</b>.")
    ok = lambda b: "통과" if b else "미달"
    sec = [("판정", tiles([
        ("① 재측정 순위 상관 (≥ 0.8)", f"{r['spearman']:.3f}", f"{ok(r['spearman_pass'])} · 2 회 채점의 형상 순위 일치도"),
        ("진단 · choice 순위 상관", f"{r['choice_spearman']:.3f}",
         "gate 제외 · 학습 seed 평균 vs unseen seed 1개")]
        + [(f"② {n} (학습 seed × 0.9)", f"{pct(g['value'])}", f"{ok(g['pass'])} · 학습 seed {pct(g['seen'])}, 기준 {pct(g['min'])}")
           for n, g in (("이동 달성", r["generalisation_gap"]["cmd_progress_ratio"]),
                        ("회전 달성", r["generalisation_gap"]["cmd_turn_ratio"]),
                        ("생존", r["generalisation_gap"]["survived"]))])
     + f'<p class="lede">참고 · 절대 T5 기준(이동 80 %, 회전 50 %, 생존 90 %)은 T5 게이트 몫: '
       f'{pct(c["cmd_progress_ratio"]["value"])} / {pct(c["cmd_turn_ratio"]["value"])} / {pct(c["survived"]["value"])}.</p>'),
        ("두 번 채점한 return", '<div class="charts">' + scatter_svg(
            [(ret[0][k], ret[1][k]) for k in ret[0]], "1 회차", "형상별 return, 세로 2 회차") + "</div>"
         + f'<p class="lede">형상 간 return 표준편차 {r["return_spread"]:.2f}, 같은 형상 2 회 차이 평균 {r["repeat_abs_diff"]:.2f}.</p>'),
        ("명령별 결과 (1 회차)", table(["명령", "생존", "이동 달성", "회전 달성", "collision"], [
            [k, pct(v["survived"]), pct(v["cmd_progress_ratio"]), pct(v.get("cmd_turn_ratio")), f'{v["collision"]:.2f}']
            for k, v in reps[0]["per_command"].items()]))]
    return page(title, "codesign · Stage 2 · T6 평가기 검증", lede, sec)


def t7(run, title):
    cfg = json.loads((run / "config.json").read_text())
    rows = [json.loads(x) for x in (run / "history.jsonl").open()]
    u = [r["update"] for r in rows]
    valid = [d for r in rows for d in r["designs"] if "invalid" not in d]
    share = [(r["update"], sum(PLAIN_LEGS(d["choices"]) for d in r["designs"] if "invalid" not in d)
              / max(1, r["valid"])) for r in rows]
    first, last = rows[:10], rows[-10:]
    # 유효 형상이 0 인 업데이트는 return 이 None 이다. 전부 None 이면 0.0 이 아니라 "—" 로 찍는다.
    avg = lambda rs, k: (lambda v: f"{sum(v) / len(v):.1f}" if v else "—")([r[k] for r in rs if r[k] is not None])
    bits = lambda rs, b: sum(d["choices"][b] for r in rs for d in r["designs"] if "invalid" not in d) / max(
        1, sum(1 for r in rs for d in r["designs"] if "invalid" not in d))
    top = sorted(valid, key=lambda d: -(d["return"] or -1e9))[:10]
    lede = (f"동결 π_θ (<code>{html.escape(Path(cfg['pi_theta_ckpt']).parent.parent.name)}</code>) 로 채점한 "
            f"return을 보상으로 TT ψ를 DDPO {len(rows)}회 갱신했습니다. "
            f"업데이트당 형상 {cfg.get('global_batch', cfg.get('batch'))}개.")
    sec = [("요약", tiles([
        ("업데이트", str(len(rows)), f"목표 {cfg['updates']}"),
        ("평균 return", f"{avg(first, 'return_mean')} → {avg(last, 'return_mean')}", "처음 10 → 마지막 10 업데이트"),
        ("기본 다리 형상 비율", f"{pct(sum(s for _, s in share[:10]) / max(1, len(share[:10])))} → "
                            f"{pct(sum(s for _, s in share[-10:]) / max(1, len(share[-10:])))}", "bit 1~4 모두 0"),
        ("유효 형상 비율", pct(sum(r["valid"] for r in rows) / max(1, sum(len(r["designs"]) for r in rows))),
         "bank 변환 성공")])),
        ("학습 곡선", '<div class="charts">'
         + line_svg([("평균", "--s2", [(r["update"], r["return_mean"]) for r in rows]),
                     ("최대", "--s1", [(r["update"], r["return_max"]) for r in rows])], "형상 return (업데이트별)")
         + line_svg([("기본 다리 비율", "--s3", share)], "기본 다리 형상 비율 (bit 1~4 = 0)")
         + line_svg([("gradient norm", "--s1", [(r["update"], r.get("gradient_norm")) for r in rows])],
                    "DDPO gradient norm") + "</div>"),
        ("비트별 선택 비율 변화", table(["비트", "의미", "처음 10", "마지막 10"], [
            [f"bit {b}", n, pct(bits(first, b)), pct(bits(last, b))] for b, n in
            [(1, "앞-왼 링크 다리"), (2, "앞-오 링크 다리"), (3, "뒤-왼 링크 다리"), (4, "뒤-오 링크 다리"),
             (5, "레일 (무효)"), (6, "앞무릎 반전"), (7, "뒷무릎 반전")]])),
        ("return 상위 형상 10", table(["choice", "seed", "return"], [
            ["".join(map(str, d["choices"])), d["seed"], f'{d["return"]:.2f}'] for d in top]))]
    return page(title, "codesign · Stage 2 · T7 RL guidance 형상 생성", lede, sec)


def t8(run, title):
    """T8 TaskEnv π_θ: conditions, training curves, scenario eval (eval_task_policy.py), videos.
    Optional <run>/notes.json = {"lede": str, "notes": [str]} (결과 먼저, 3 줄 이내)."""
    cfg = json.loads((run / "config.json").read_text())
    rows = [json.loads(x) for x in (run / "progress.jsonl").open()]
    ev = json.loads(sorted(run.glob("eval_task_*.json"))[-1].read_text())
    notes = json.loads((run / "notes.json").read_text()) if (run / "notes.json").exists() else {}
    vids = json.loads((run / "videos/task_videos.json").read_text()) if (run / "videos/task_videos.json").exists() else []
    a, ppo = cfg["args"], cfg["ppo"]
    ep = [r for r in rows if "episode/length" in r]
    curve = lambda k: [(r["step"], r.get(k)) for r in ep]
    sc = ev["scenarios"]
    lede = notes.get("lede", f"TaskEnv (지형·높이 측정·점프) 에서 π_θ 를 {rows[-1]['step']:,} step 학습했습니다.")
    sec = [("요약", tiles([
        ("학습 step", f"{rows[-1]['step']:,}", f"목표 {int(a['timesteps']):,} · env {ppo['num_envs']}"),
        ("소요", f"{rows[-1]['wall_s'] / 60:.0f} 분", f"첫 기록까지 {rows[0]['wall_s'] / 60:.0f} 분 (컴파일 포함)"),
        ("처리량", f"{sum(r.get('episode/sps', 0) for r in ep[-10:]) / max(1, len(ep[-10:])):,.0f} step/s", "마지막 10 기록 평균"),
        ("에피소드 길이", f"{ep[0]['episode/length']:.0f} → {ep[-1]['episode/length']:.0f}", f"최대 {1000} step (20 s)")]
        + [(f"생존 · {k}", pct(v["survived"]), f"평균 {v['steps']:.0f} step") for k, v in sc.items()])),
        ("조건", table(["항목", "값"], [
            ["env", "TaskEnv — legged_gym rough terrain (Q16 a) + 높이 측정 187 점 + 점프 (Q17 b), obs 257"],
            ["형상", f"{cfg['designs']} 개 ({cfg['groups']} 구조 그룹), choice 당 seed {a['seeds_per_choice']}"],
            ["PPO", f"env {ppo['num_envs']} · unroll {ppo['unroll_length']} · lr {ppo['learning_rate']} · desired_kl {ppo.get('desired_kl')}"],
            ["학습 DR", "관측 노이즈 · push · 마찰 (legged_gym 원본)"],
            ["평가", f"DR 끔 · 명령 고정 20 s · 계단 난이도 {ev['level']} (단 {0.05 + 0.018 * ev['level']:.3f} m) · seed {ev['seed']}"],
            ["checkpoint", html.escape(Path(ev["checkpoint"]).name)]])),
        ("학습 곡선", '<div class="charts">'
         + line_svg([("에피소드 길이", "--s1", curve("episode/length"))], "에피소드 길이 (step)")
         + line_svg([("합계 보상", "--s2", curve("episode/sum_reward"))], "에피소드 합계 보상")
         + line_svg([("jump_air", "--s1", curve("episode/jump_air")), ("jump_height", "--s3", curve("episode/jump_height"))],
                    "점프 보상 항 (에피소드 합)")
         + line_svg([("tracking_lin_vel", "--s2", curve("episode/tracking_lin_vel"))], "선속도 추종 보상 (에피소드 합)")
         + "</div>"),
        ("시나리오 평가 (형상 평균)", table(
            ["시나리오", "생존", "평균 step", "명령 방향 이동", "회전", "지면 높이 변화", "점프 최고 상승", "네 발 공중 비율"],
            [[k, pct(v["survived"]), f"{v['steps']:.0f}",
              f"{v['progress_m']:+.2f} / {v['asked_m']:.0f} m" if v["asked_m"] else "—",
              f"{v['turn_rad']:+.2f} / {v['asked_turn_rad']:.0f} rad" if v["asked_turn_rad"] else "—",
              f"{v['climb_m']:+.3f} m", f"{v['jump_rise_m']:.3f} m", pct(v["air_frac"])] for k, v in sc.items()]))]
    if vids:
        sec.append(("영상 (시나리오별 중앙값 형상)", '<div class="charts">' + "".join(
            f'<figure><figcaption>{html.escape(v["scenario"])} <span>· {v["choices"]} s{v["seed"]} · '
            f'{v["steps"]:.0f} step · 이동 {v["progress_m"]:+.2f} m</span></figcaption>'
            f'<video controls muted loop playsinline width="100%" src="videos/{html.escape(v["file"])}"></video></figure>'
            for v in vids) + "</div>"))
    if notes.get("notes"):
        sec.append(("해석", "<ul>" + "".join(f"<li>{n}</li>" for n in notes["notes"]) + "</ul>"))
    return page(title, "codesign · Stage 2 · T8 통합 π_θ", lede, sec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("t6", "t7", "t8"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--title", required=True)
    args = p.parse_args()
    run = args.run.resolve()
    (run / "report.html").write_text({"t6": t6, "t7": t7, "t8": t8}[args.stage](run, args.title))
    print(f"wrote {run / 'report.html'}")


if __name__ == "__main__":
    main()
