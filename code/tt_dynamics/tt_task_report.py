"""Build factual Markdown/HTML reports from completed TT task benchmark outputs.

No invented values: absent runs appear explicitly as not evaluated. Every
statistic is computed from saved candidate-level physical MuJoCo replay data.
"""
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


TASK_NAMES = {"circle": "원형 트랙", "zigzag": "지그재그 트랙",
              "straight": "최고속도 직진"}
TASK_ORDER = ("circle", "zigzag", "straight")


def _number(x, digits=3):
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{x:.{digits}f}"


def _gradient_name(path):
    if path.parent.name.startswith("gradient_scale_"):
        return f"Gradient (scale={path.parent.name.removeprefix('gradient_scale_')})"
    if path.parent.name == "rl_ppo":
        return "RL (episodic PPO)"
    return path.parent.name


def _trajectory_svg(task, npz_path, candidate, task_spec):
    with np.load(npz_path) as z:
        mask = z["valid"][candidate].astype(bool)
        actual = z["mujoco"][candidate, mask, :2]
        predicted = z["pred"][candidate, :, :2]
    if not len(actual):
        return '<p>초기 상태 무효: 표시할 실제 궤적이 없음</p>'
    if task == "circle":
        theta = np.linspace(0, 2 * math.pi, 180)
        radius = float(task_spec.get("radius", 1))
        center = (0, radius)
        lines = []
        for rad in (radius, radius - task_spec.get("half_width", 0.3),
                    radius + task_spec.get("half_width", 0.3)):
            lines.append(np.stack([rad * np.sin(theta),
                                   radius - rad * np.cos(theta)], -1))
    elif task == "zigzag":
        xp = np.linspace(0, max(6, float(actual[:, 0].max()) + 1), 200)
        amp = float(task_spec.get("amp", .3))
        period = float(task_spec.get("period", 2))
        y = amp * (2 / math.pi) * np.arcsin(
            np.clip(np.sin(2 * math.pi * xp / period), -1, 1))
        half = float(task_spec.get("half_width", .15))
        lines = [np.stack([xp, y + shift], -1) for shift in (0, half, -half)]
    else:
        xp = np.linspace(0, max(6, float(actual[:, 0].max()) + 1), 2)
        half = float(task_spec.get("half_width", .15))
        lines = [np.stack([xp, xp * 0 + shift], -1)
                 for shift in (0, half, -half)]
    series = [actual] + lines
    pts = np.concatenate(series)
    minimum = pts.min(0) - .35
    maximum = pts.max(0) + .35
    span = max(float(np.max(maximum - minimum)), 1)
    width, height, margin = 540, 360, 18
    def convert(p):
        x = margin + (p[:, 0] - minimum[0]) / span * (width - 2*margin)
        y = height - margin - (p[:, 1] - minimum[1]) / span * (width - 2*margin)
        return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    els = []
    for i, line in enumerate(lines):
        color = "#999" if i else "#64748b"
        els.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5"'
                   f' stroke-dasharray="{"" if i == 0 else "5 5"}" points="{convert(line)}"/>')
    els.append(f'<polyline fill="none" stroke="#167a62" stroke-width="2.5" '
               f'points="{convert(actual)}"/>')
    els.append('<text x="12" y="24" fill="#167a62" font-size="14">'
               '실제 MuJoCo 궤적 (초록색)</text>')
    return (f'<svg role="img" aria-label="Task 실제 궤적" '
            f'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'style="width:100%;max-width:640px;background:#f8fafc;border-radius:10px">'
            + "".join(els) + "</svg>")


def collect(root):
    data = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(root).glob("*/**/seed_*.json")):
        if path.parent.name not in ("rl_ppo",) and not path.parent.name.startswith("gradient_scale_"):
            continue
        task = path.relative_to(root).parts[0]
        if task not in TASK_ORDER:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        npz_path = path.with_suffix(".npz")
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing physical rollout archive for {path}: {npz_path}")
        data[task][_gradient_name(path)].append((path, result, npz_path))
    return data


def build_report(root):
    root = Path(root)
    groups = collect(root)
    markdown = ["# TT Reward Gradient vs PPO: 세 Task 비교 보고서", "",
                "## 실험 방법", "",
                "기존 TT의 파라미터를 동결한 상태에서 E·S·A를 생성하고, "
                "각 후보를 10초 동안 **CPU MuJoCo**에서 오픈루프로 실행하여 점수를 계산했다. "
                "Gradient는 추론 중 미분 가능한 예측 보상을 사용하며, "
                "RL은 에피소드 단위 Gaussian residual-noise 정책을 실제 MuJoCo 보상으로 PPO 학습한다. "
                "RL은 보행 제어 PPO나 매 프레임 폐루프 제어가 아니다.", "",
                "**중요:** 이 보고서의 숫자는 실제 replay 결과만 집계한다. "
                "낙상 시 기존 코드와 동일하게 보상에서 20을 감점하며 "
                "초기 상태 무효 후보는 평균 보상에서 제외한다. 완주율의 분모는 모든 후보이다.", ""]
    report = {"tasks": {}, "interpretation": "physical MuJoCo replay; frozen TT; no training-completion claims"}
    sections = []
    for task in TASK_ORDER:
        task_groups = groups.get(task, {})
        markdown.extend([f"## {TASK_NAMES[task]}", ""])
        sections.append(f"<h2>{html.escape(TASK_NAMES[task])}</h2>")
        if not task_groups:
            markdown.extend(["아직 실제 평가 결과 없음.", ""])
            sections.append("<p>아직 실제 MuJoCo 평가 결과 없음</p>")
            report["tasks"][task] = {"status": "not_evaluated"}
            continue
        rows_md = ["| 방식 | 후보 | 초기 유효 | 10초 완주 | 유효후보 평균 보상 | 최고 보상 | 최고 진행량 |",
                   "|---|---:|---:|---:|---:|---:|---:|"]
        rows_html = ["<table><thead><tr><th>방식</th><th>후보</th><th>초기 유효</th><th>10초 완주</th>"
                     "<th>유효후보 평균 보상</th><th>최고 보상</th><th>최고 진행량</th></tr></thead><tbody>"]
        stats = {}
        visuals = []
        for method, runs in sorted(task_groups.items()):
            candidates = [c for _, run, _ in runs for c in run["candidates"]]
            valid = [c for c in candidates if c.get("reward") is not None]
            completes = sum(bool(c.get("complete_10s")) for c in candidates)
            rewards = [float(c["reward"]) for c in valid]
            progress = [float(c["progress"]) for c in valid if c.get("progress") is not None]
            st = dict(candidates=len(candidates), initial_valid=len(valid),
                      completed_10s=completes, completion_rate=completes/max(1, len(candidates)),
                      mean_valid_reward=sum(rewards)/len(rewards) if rewards else None,
                      best_reward=max(rewards) if rewards else None,
                      best_progress=max(progress) if progress else None)
            stats[method] = st
            summary = (method, len(candidates), len(valid), completes,
                       _number(st["mean_valid_reward"]), _number(st["best_reward"]),
                       _number(st["best_progress"]))
            rows_md.append("| " + " | ".join(map(str, summary)) + " |")
            rows_html.append("<tr>" + "".join("<td>"+html.escape(str(x))+"</td>" for x in summary) + "</tr>")
            if valid:
                best = max(valid, key=lambda c: c["reward"])
                best_file = next((p for _, run, p in runs
                                  if any(c["candidate"] == best["candidate"] and
                                         c["morphology"] == best["morphology"]
                                         for c in run["candidates"])), None)
                descriptor = next(run["task"] for _, run, _ in runs)
                if best_file is not None:
                    svg = _trajectory_svg(task, best_file, best["candidate"], descriptor)
                    morph = ", ".join(_number(float(x), 3) for x in best["morphology"])
                    visuals.append("<h3>"+html.escape(method)+" — 최고 보상 후보</h3>"
                                   +svg+"<p>실제 보상 "+_number(best["reward"])+
                                   " · 유효 시간 "+_number(best.get("valid_seconds",0))+
                                   "초 · E=["+html.escape(morph)+"]</p>")
                    markdown.append(f"**{method} 최고 후보 E:** [{morph}]; "
                                    f"보상 {_number(best['reward'])}; "
                                    f"유효 시간 {_number(best.get('valid_seconds',0))}초.")
        markdown.extend(rows_md+[""])
        rows_html.append("</tbody></table>")
        sections.append("".join(rows_html)+"".join(visuals))
        report["tasks"][task] = {"status": "evaluated", "methods": stats}
    markdown.extend(["## 판정 기준", "",
                     "운동학적 일관성만으로 성공을 선언하지 않는다. "
                     "실제 MuJoCo 10초 완주율과 동일 조건에서의 실현 보상 및 진행량을 비교한다. "
                     "후보 수가 작거나 RL 평가 Seed가 적으면 차이를 통계적으로 확정하지 않는다.", "",
                     "## 실험 한계", "",
                     "RL은 에피소드 보상을 사용한 저차원 residual-noise PPO이며, "
                     "상태별 보행 PPO나 TT 재학습이 아니다. "
                     "TT 예측 상태가 물리적으로 일관되지 않을 수 있으므로 실현된 보상만 성능으로 인정한다.", ""])
    (root/"report.md").write_text("\n".join(markdown)+"\n", encoding="utf-8")
    (root/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    document = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                '<title>TT Task Benchmark</title><style>body{max-width:1050px;margin:30px auto;padding:0 20px;'
                'font:16px/1.6 system-ui,sans-serif;color:#1c2938}table{width:100%;border-collapse:collapse}'
                'td,th{border-bottom:1px solid #ddd;padding:9px;text-align:left;font-size:14px}'
                'th{background:#eef2f6}section{margin:26px 0}svg{display:block;margin:12px 0}</style>'
                '<h1>TT Reward Gradient vs PPO — 실제 MuJoCo 평가 보고서</h1>'
                '<p>TT 가중치 동결 · 10초 오픈루프 물리 검증 · Gradient vs episodic PPO</p>'
                '<p>초기 무효는 평균 보상에서 제외하고 완주율 분모에는 포함한다. '
                '실제 시뮬레이션 없이 숫자를 채우지 않는다.</p>'
                + "".join("<section>"+s+"</section>" for s in sections)
                +'<p>RL은 에피소드 단위 PPO이며 상태별 폐루프 PPO는 아님. '
                '성공 기준은 실제 완주율·실현 보상·진행량.</p></html>')
    (root/"report.html").write_text(document, encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    r = build_report(args.out)
    print(json.dumps(r, indent=2, ensure_ascii=False))
