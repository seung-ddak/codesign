"""Build <run>/report.html from a finished run (the result-report format, 2026-09-27).

  python stage2/code/make_report.py --run stage2/runs/T5_try1 --task "T5 π_θ 선학습" \
      --title "T5 try1" [--baseline stage2/runs/<run> --baseline-label "..."]

Reads config.json, progress.jsonl, the latest eval_*.json, videos/videos.json and
videos/fixed_videos.json. Narrative (lede, timeline, notes, flow link) comes from an optional
<run>/timeline.json: {"lede", "intro", "flow_url", "flow_text", "events": [{when, cls, title,
pill, pill_cls, what, res}], "notes": [...]}; sections without data are hidden. Publishing the
page as an artifact is a separate, manual step.
"""

import argparse
import base64
import datetime
import glob
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTRL_DT = 0.02


def curve(run, label, short):
    rows = [json.loads(line) for line in (run / "progress.jsonl").open()]
    scale = json.loads((run / "config.json").read_text())["reward_scales_x_dt"]["tracking_lin_vel"]
    ep = [r for r in rows if "episode/length" in r]
    k = max(1, len(ep) // 120)
    pts = []
    for i in range(0, len(ep), k):
        b = ep[i:i + k]
        mean = lambda q: sum(r[q] for r in b) / len(b)
        length = mean("episode/length")
        pts.append({"s": b[-1]["step"], "l": round(length, 1),
                    "t": round(mean("episode/tracking_lin_vel") / (scale * length) * 100, 2)})
    std = [[r["step"], round(r["training/policy_dist_mean_std"], 3)]
           for r in rows if "training/policy_dist_mean_std" in r]
    return {"label": label, "short": short, "curve": pts, "std": std}


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--baseline-label", default="이전 run")
    args = p.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / "config.json").read_text())
    ev = json.loads(Path(sorted(glob.glob(str(run / "eval_*.json")))[-1]).read_text())
    data = {
        "task": args.task, "title": args.title, "run": os.path.relpath(run, HERE.parents[1]),
        "ckpt": Path(ev["checkpoint"]).name, "date": datetime.date.today().isoformat() + " 기준",
        "cfg": {"designs": cfg["designs"], "groups": cfg["groups"], "envs": cfg["ppo"]["num_envs"],
                "steps": int(float(cfg["args"]["timesteps"])),
                "lin_scale": round(cfg["reward_scales_x_dt"]["tracking_lin_vel"] / CTRL_DT, 3)},
        "summary": ev["summary"], "perCommand": ev["per_command"],
        "designs": [{"c": "".join(map(str, r["choices"])), "s": r["seed"], "ret": round(r["return"], 2),
                     "sv": r["survived"], "pr": round(r["cmd_progress_ratio"], 3),
                     "tr": round(r.get("cmd_turn_ratio", 0.), 3)} for r in ev["designs"]],
        "videos": load_json(run / "videos/videos.json", []),
        "fixedVideos": load_json(run / "videos/fixed_videos.json", []),
        "timeline": load_json(run / "timeline.json", {}),
        "curves": {"this": curve(run, f"이번: {run.name}", "이번")},
    }
    if args.baseline:
        base = args.baseline.resolve()
        data["curves"]["base"] = curve(base, f"{args.baseline_label}: {base.name}", "이전")
    write_page(run / "report.html", args.title, data)
    # Telegram copy: one self-contained file with the videos inlined as data URIs.
    videos = {v["file"]: run / "videos" / v["file"] for v in data["videos"] + data["fixedVideos"]}
    data["videoData"] = {f: "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()
                         for f, p in videos.items() if p.exists()}
    write_page(run / "report_tg.html", args.title, data)


def write_page(path, title, data):
    page = (HERE / "report_template.html").read_text()
    page = (page.replace("__TITLE__", title)
            .replace("__CSS__", (HERE / "report_style.css").read_text())
            .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    path.write_text(page)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
