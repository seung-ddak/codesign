"""Post one result comment per completed wide16_v2 expert to a new GitHub issue."""

import csv
import json
import stat
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


root = Path(__file__).resolve().parents[2]
run = root / "runs/pipeline_v1/experts_wide16_v2"
token_file = Path.home() / ".config/codesign/github_token"
api = "https://api.github.com/repos/seung-ddak/codesign"
if not token_file.is_file() or token_file.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
    raise SystemExit("GitHub token file is missing or is not mode 600")
token = token_file.read_text().strip()


def request(path, data=None):
    req = Request(api + path, data=None if data is None else json.dumps(data).encode(),
                  headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
                           "User-Agent": "codesign-wide-expert-notifier", "Content-Type": "application/json"},
                  method="GET" if data is None else "POST")
    with urlopen(req, timeout=30) as response:
        return json.load(response)


issue_file = run / "github_issue_number.txt"
if issue_file.exists():
    issue = int(issue_file.read_text())
else:
    created = request("/issues", {"title": "Wide16 expert 재학습 및 형상별 검증",
        "body": "e000–e002, e004–e016에서 19개 모션 expert를 재학습합니다. "
                "학습률을 낮추고, e003을 제외한 6개 학습 형상에서 중간 체크포인트를 비교해 선택합니다. "
                "각 expert의 17형상 최종 평가가 끝날 때마다 이 이슈에 결과를 남깁니다. "
                "e003은 미학습 형상 평가에만 사용합니다."})
    issue = int(created["number"])
    issue_file.write_text(f"{issue}\n")
    print(created["html_url"], flush=True)

if "--init-only" in sys.argv:
    raise SystemExit

while True:
    posted = {line.partition(" -->")[0].removeprefix("<!-- wide16_v2:")
              for comment in request(f"/issues/{issue}/comments?per_page=100")
              for line in comment["body"].splitlines() if line.startswith("<!-- wide16_v2:")}
    completed = []
    for line in (run / "queue.log").read_text().splitlines():
        if not line.startswith("Completed "):
            continue
        motion, path = line.removeprefix("Completed ").split(": ", 1)
        completed.append(motion)
        if motion in posted:
            continue
        summary = root / path
        with summary.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 17 or {r["design"] for r in rows} != {f"e{i:03d}" for i in range(17)}:
            raise ValueError(f"Unexpected evaluation rows: {summary}")
        selection = json.loads((summary.parents[1] / "checkpoint_selection.json").read_text())
        lines = [f"<!-- wide16_v2:{motion} -->", f"### {motion}",
                 f"선택 체크포인트: {selection['selected_step']:,} step. "
                 "6개 학습 형상으로 선택했고, e003은 선택에 사용하지 않았습니다.", "",
                 "| 형상 | 생존 | 관절 RMS (°) | 발 오차 (cm) | 자세 오차 (°) | 접지 일치 | 통과 |",
                 "|---|---|---:|---:|---:|---:|---|"]
        for row in rows:
            lines.append(f"| {row['design']} | {row['survived']} | {float(row['joint_rms_deg']):.2f} "
                         f"| {float(row['foot_err_cm']):.2f} | {float(row['orn_err_deg']):.2f} "
                         f"| {float(row['contact_agree']):.2f} | {row['pass_all']} |")
        lines += ["", f"통과: {sum(int(r['pass_all']) for r in rows)}/17; "
                  f"학습 형상 {sum(int(r['pass_all']) for r in rows if r['design'] != 'e003')}/16, "
                  f"미학습 e003 {'통과' if next(r for r in rows if r['design'] == 'e003')['pass_all'] == '1' else '실패'}."]
        report = "\n".join(lines) + "\n"
        summary.with_name("report.md").write_text(report)
        request(f"/issues/{issue}/comments", {"body": report})
        print(f"Commented {motion} on issue #{issue}", flush=True)
    if len(set(completed)) == 19:
        break
    time.sleep(30)
