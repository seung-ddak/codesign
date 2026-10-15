"""Create one GitHub Issue and comment after each selected expert finishes."""

import csv
import json
import math
import stat
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
EXPERTS = ROOT / "runs/pipeline_v1/experts"
TOKEN_FILE = Path.home() / ".config/codesign/github_token"
API = "https://api.github.com/repos/seung-ddak/codesign"


def make_report(motion, summary):
    csv_text = (ROOT / summary).read_text().strip()
    rows = list(csv.DictReader(csv_text.splitlines()))
    assert len(rows) == 4 and {row["design"] for row in rows} == {"e000", "e001", "e002", "e003"}
    lines = [
        f"### {motion} 학습·평가 완료",
        "e000~e002로 학습하고 e003은 미학습 형상으로 평가했습니다.",
        "",
        "| 형상 | 생존 | 관절 RMS (°) | 발 오차 (cm) | 자세 오차 (°) | 종료 위치 오차 (cm) | 접지 일치 | 종합 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['design']} | {'예' if row['survived'] == '1' else '아니요'} "
            f"| {float(row['joint_rms_deg']):.2f} | {float(row['foot_err_cm']):.2f} "
            f"| {float(row['orn_err_deg']):.2f} | {float(row['end_pos_err_cm']):.2f} "
            f"| {100 * float(row['contact_agree']):.1f}% | {'통과' if row['pass_all'] == '1' else '불합격'} |"
        )
    lines += ["", "오차는 낮을수록, 접지 일치는 높을수록 좋습니다. e003은 학습에 사용하지 않은 형상입니다.", "", "#### 형상별 평가 오차 (x축: 형상, y축: 오차)"]
    for key, title, unit in (
        ("joint_rms_deg", "Joint RMS", "deg"),
        ("foot_err_cm", "Foot error", "cm"),
        ("orn_err_deg", "Orientation error", "deg"),
        ("end_pos_err_cm", "End position error", "cm"),
    ):
        values = [float(row[key]) for row in rows]
        assert all(math.isfinite(value) and value >= 0 for value in values)
        upper = max(1, math.ceil(max(values) * 1.1))
        lines += [
            "", "```mermaid", "xychart-beta", f'    title "{title} ({unit})"',
            "    x-axis [e000, e001, e002, e003]",
            f'    y-axis "{unit}" 0 --> {upper}',
            f"    bar [{', '.join(f'{value:.3f}' for value in values)}]", "```",
        ]

    metrics = {}
    steps = {}
    step = 0
    for line in (EXPERTS / motion / "train.log").read_text().splitlines():
        if line.startswith("-" * 40) and line.endswith("-" * 40):
            middle = line[40:-40]
            if middle.isdigit():
                step = int(middle)
            continue
        key, separator, value = line.partition(": ")
        if separator and key in {
            "training/policy_loss", "training/v_loss", "training/kl_mean", "training/entropy_loss",
            "eval/avg_episode_length", "eval/episode/done/fellover/mean",
            "eval/episode/done/clip_end/mean",
        }:
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                metrics.setdefault(key, []).append(number)
                steps.setdefault(key, []).append(step)

    def trend(key, percent=False):
        values = metrics.get(key)
        if not values:
            return "미기록"
        unit = 100 if percent else 1
        suffix = "%" if percent else ""
        return f"{values[0] * unit:.3g}{suffix} → {values[-1] * unit:.3g}{suffix}"

    lines += [
        "", "#### PPO 학습 지표 (첫 기록 → 마지막 기록)",
        "| 지표 | 변화 | 간단한 해석 |", "| --- | ---: | --- |",
        f"| Policy Loss | {trend('training/policy_loss')} | 단독으로 좋고 나쁨을 판정하지 않습니다. NaN·급변 여부를 봅니다. |",
        f"| Value Loss | {trend('training/v_loss')} | 지속적으로 폭증하지 않고 안정적이면 좋습니다. |",
        f"| Approximate KL | {trend('training/kl_mean')} | 급증하면 정책 변화가 과격할 수 있습니다. 1 이상이면 점검이 필요합니다. |",
        "| Clip Fraction | 미기록 | 과도한 clipping 여부를 볼 지표지만 현재 코드가 기록하지 않습니다. |",
        f"| Entropy 관련 손실 | {trend('training/entropy_loss')} | 원래 Entropy는 미기록입니다. 손실값만으로 성능을 판정하지 않습니다. |",
        "| Gradient Norm | 미기록 | 실제 기울기 크기는 미기록입니다. |",
        "| Explained Variance | 미기록 | 1에 가까울수록 예측력이 좋지만 현재 미기록입니다. |",
        f"| Episode Length | {trend('eval/avg_episode_length')} | 길이만으로는 부족합니다. 클립 완료율과 함께 봅니다. |",
        f"| 조기 종료율 (낙상) | {trend('eval/episode/done/fellover/mean', True)} | 낮을수록 좋습니다. 전체 조기 종료율은 미기록입니다. |",
        f"| 클립 완료율 | {trend('eval/episode/done/clip_end/mean', True)} | 높을수록 좋습니다. |",
        "", "학습 손실보다 아래 평가 CSV의 생존·오차·`pass_all`과 영상을 우선 확인하세요.",
    ]
    lines += ["", "#### PPO 지표 추이 (x축: 학습 스텝, y축: 지표 값)"]
    for key, title, unit in (
        ("training/policy_loss", "Policy loss", "loss"),
        ("training/v_loss", "Value loss", "loss"),
        ("training/kl_mean", "Approximate KL", "KL"),
        ("training/entropy_loss", "Entropy loss", "loss"),
        ("eval/avg_episode_length", "Episode length", "steps"),
        ("eval/episode/done/fellover/mean", "Fall termination", "percent"),
        ("eval/episode/done/clip_end/mean", "Clip completion", "percent"),
    ):
        if len(metrics.get(key, [])) < 2:
            continue
        scale = 100 if unit == "percent" else 1
        values = [value * scale for value in metrics[key]]
        lower, upper = min(0, min(values) * 1.1), max(0, max(values) * 1.1)
        if lower == upper:
            upper = 1
        labels = ", ".join(f'"{value / 1_000_000:.2f}M"' for value in steps[key])
        lines += [
            "", "```mermaid", "---", "config:", "  theme: base",
            "  themeVariables:", "    xyChart:", "      plotColorPalette: '#155e75'",
            "  themeCSS: '.line-plot-0 path { stroke-width: 3.5px !important; }'",
            "---", "xychart-beta", f'    title "{title}"',
            f"    x-axis [{labels}]", f'    y-axis "{unit}" {lower:.3g} --> {upper:.3g}',
            f"    line [{', '.join(f'{value:.4g}' for value in values)}]", "```",
        ]
    lines += ["", f"#### motion_eval/summary.csv — `{summary}`", "```csv", csv_text, "```"]
    return "\n".join(lines) + "\n"


def post(path, data, token):
    request = Request(
        API + path,
        data=json.dumps(data).encode(),
        headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
                 "Content-Type": "application/json", "User-Agent": "codesign-expert-notifier"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def main():
    if not TOKEN_FILE.is_file() or TOKEN_FILE.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit(f"Create {TOKEN_FILE} with mode 600; do not put the token in the repo")
    token = TOKEN_FILE.read_text().strip()
    if not token:
        raise SystemExit("Empty GitHub token")

    selected = {row["motion"] for row in csv.DictReader((EXPERTS / "selection.csv").open())}
    assert len(selected) == 19
    issue_file = EXPERTS / "github_issue_number.txt"
    if issue_file.exists():
        issue = int(issue_file.read_text().strip())
    else:
        # shortcut: API succeeds but local save fails can create a duplicate Issue; reconcile manually if that happens.
        draft = (EXPERTS / "GITHUB_ISSUE_DRAFT.md").read_text()
        title, body = draft.split("\n", 1)
        created = post("/issues", {"title": title.removeprefix("# "), "body": body.strip()}, token)
        issue = created["number"]
        issue_file.write_text(f"{issue}\n")
        print(f"Created {created['html_url']}", flush=True)

    notified_file = EXPERTS / "github_notified.txt"
    while True:
        notified = set(notified_file.read_text().splitlines()) if notified_file.exists() else set()
        for line in (EXPERTS / "queue.log").read_text().splitlines():
            if not line.startswith("Completed "):
                continue
            motion, summary = line.removeprefix("Completed ").split(": ", 1)
            if motion not in selected or motion in notified:
                continue
            report = make_report(motion, summary)
            (ROOT / summary).with_name("report.md").write_text(report)
            post(f"/issues/{issue}/comments", {"body": report}, token)
            with notified_file.open("a") as output:
                output.write(motion + "\n")
            notified.add(motion)
            print(f"Notified {motion}", flush=True)
        if notified == selected:
            break
        if subprocess.run(["tmux", "has-session", "-t", "=pipeline_v1_selected_experts"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            raise SystemExit("Training queue stopped before all selected motions completed")
        time.sleep(30)


if __name__ == "__main__":
    main()
