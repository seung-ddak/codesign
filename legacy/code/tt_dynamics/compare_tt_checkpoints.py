"""Compare saved TT checkpoints on CPU and build a single video report."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from benchmark_task_inference import check_task_checkpoint
from tt_task_report import TASK_NAMES, collect, video_for_candidate, video_player

ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = (ROOT / "runs/pipeline_v1", ROOT / "runs/v100_mujoco_10k")
EXPECTED = (
    "runs/pipeline_v1/tt/h8/try1",
    "runs/pipeline_v1/tt/h32/try6",
    "runs/pipeline_v1/tt/h500/try1",
    "runs/pipeline_v1/fast_mode24_v4/tt_joint",
    "runs/pipeline_v1/fast_mode24_v7_eswap/tt_joint",
    "runs/pipeline_v1/fast_mode24_v16_twin/tt_joint",
    "runs/v100_mujoco_10k/tt/h64_warmup_ema_mode0124",
    "runs/v100_mujoco_10k/tt/h64_kinematic_warmstart_v1",
)
METHODS = ("Gradient (scale=0)", "Gradient (scale=1)", "RL (episodic PPO)")
HEADERS = ("무유도 TT", "Reward gradient", "RL 유도 (residual PPO)")


def checkpoint_path(path):
    path = path.expanduser().resolve()
    return path if path.is_file() or path.suffix in (".pt", ".pth", ".ckpt") else path / "checkpoint.pt"


def checkpoint_inventory(paths, scan_roots):
    if paths:
        return list(dict.fromkeys(checkpoint_path(path) for path in paths))
    return sorted({ROOT / run / "checkpoint.pt" for run in EXPECTED} |
                  {path.resolve() for root in scan_roots for path in root.rglob("checkpoint.pt")})


def inspect_checkpoint(path):
    parent = path.parent
    label = parent.name
    if label == "tt_joint" or label.startswith("try"):
        label = f"{parent.parent.name}/{label}"
    identifier = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    entry = dict(id=identifier, label=label, path=str(path), status="ready", config={})
    if not path.is_file():
        entry.update(status="missing", detail="체크포인트 파일 없음")
        return entry
    try:
        with path.open("rb") as stream:
            entry["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            snapshot = torch.load(stream, map_location="cpu", weights_only=False)
        config = snapshot["args"]
        entry["config"] = {key: config.get(key) for key in
                           ("horizon", "diff_steps", "modes", "conditioning_version", "pose", "norm")}
        entry["config"]["training_steps"] = snapshot.get("step", config.get("steps"))
        entry["config"]["weights"] = "EMA" if snapshot.get("ema_net") is not None else "raw"
        check_task_checkpoint(config)
    except ValueError as exc:
        entry.update(status="incompatible", detail=str(exc))
    except Exception as exc:
        entry.update(status="unreadable", detail=f"{type(exc).__name__}: {exc}")
    return entry


def esc(value):
    return html.escape(str(value), quote=True)


def cell(root, version_root, runs, title, players):
    candidates = [(path, row) for path, run, _ in runs for row in run["candidates"]]
    if not candidates:
        return "<td>평가 결과 없음</td>"
    rows = [row for _, row in candidates]
    scored = [row for row in rows if row.get("reward") is not None]
    completed = sum(bool(row.get("complete_10s")) for row in rows)
    successful = sum(bool(row.get("task_success")) for row in rows)

    def mean(key):
        values = [row[key] for row in rows if row.get(key) is not None and math.isfinite(row[key])]
        return f"{sum(values) / len(values):.3f} ({len(values)}개)" if values else "—"

    metrics = (f"<p>10초 생존 {completed}/{len(rows)} · 과제 성공 {successful}/{len(rows)}"
               f"<br>평균 물리 실행 시간 {mean('valid_seconds')}초"
               f"<br>평균 TT 예측 목적함수 {mean('predicted_guidance_score')}"
               f"<br>평균 실제 MuJoCo 보상 {mean('reward')}"
               f"<br>평균 진행량 {mean('progress')}"
               f"<br>예측↔실제 XY 오차 {mean('pred_vs_mujoco_xy_mean_cm')} cm</p>")
    if not scored:
        reasons = sorted({video_for_candidate(version_root, path, row)[1]
                          for path, row in candidates})
        return f"<td>{metrics}<p>{esc(' · '.join(reasons))}</p></td>"
    ranked = sorted(((path, row) for path, row in candidates if row.get("reward") is not None),
                    key=lambda item: item[1]["reward"], reverse=True)
    reasons = []
    for path, row in ranked:
        video, status = video_for_candidate(version_root, path, row)
        if video:
            relative = (version_root / video).relative_to(root).as_posix()
            backend = (row.get("video_metadata", {}).get("backend")
                       if row.get("video_path") == video else None)
            display = {"osmesa": "MuJoCo Renderer · CPU OSMesa",
                       "egl": "MuJoCo Renderer · EGL"}.get(backend, "렌더러 정보 없음")
            player = video_player(root, relative,
                                  f"{title} · {path.stem} · 후보 {row['candidate']} · {display}")
            players.add(relative)
            clip = f"<p>영상 후보 실제 보상 {row['reward']:.3f}</p>{player}"
            return f"<td>{metrics}{clip}</td>"
        if status not in reasons:
            reasons.append(status)
    return f"<td>{metrics}<p>{esc(' · '.join(reasons))}</p></td>"


def build_comparison(root, index):
    root = Path(root)
    entries = index["versions"]
    groups = {entry["id"]: collect(root / "versions" / entry["id"]) for entry in entries}
    tables, players = [], set()
    for task, label in TASK_NAMES.items():
        rows = []
        for entry in entries:
            result = groups[entry["id"]].get(task)
            if not result:
                continue
            version_root = root / "versions" / entry["id"]
            cells = [cell(root, version_root, result.get(method, []),
                          f"{entry['label']} · {header}", players)
                     for method, header in zip(METHODS, HEADERS)]
            rows.append(f"<tr><th>{esc(entry['label'])}</th>{''.join(cells)}</tr>")
        if not rows:
            tables.append(f"<h2>{esc(label)}</h2><p>저장된 MuJoCo 평가 결과 없음</p>")
            continue
        headings = "".join(f"<th>{esc(header)}</th>" for header in HEADERS)
        tables.append(f"<h2>{esc(label)}</h2><div class='matrix'><table>"
                      f"<thead><tr><th>TT 버전</th>{headings}</tr></thead>"
                      f"<tbody>{''.join(rows)}</tbody></table></div>")

    inventory = []
    for entry in entries:
        config = entry["config"]
        details = (f"H={config.get('horizon')} · DDIM {config.get('diff_steps')} · "
                   f"{config.get('weights')} · 학습 {config.get('training_steps')} steps"
                   if config else "—")
        inventory.append(f"<tr><th>{esc(entry['label'])}<br><small>{esc(entry['path'])}</small></th>"
                         f"<td>{esc(details)}</td><td>{esc(entry['status'])}"
                         f"<br>{esc(entry.get('detail', ''))}"
                         f"<br><small>{esc(entry.get('sha256', '')[:12])}</small></td></tr>")
    options = index["options"]
    document = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>TT 버전별 MuJoCo 비교</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:2rem;color:#183043}}
table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;vertical-align:top;
padding:.65rem;border-bottom:1px solid #ccd7df}}th{{background:#eef3f6}}
.matrix{{overflow-x:auto}}.matrix td{{min-width:220px;width:29%}}small{{overflow-wrap:anywhere}}</style>
<h1>TT 버전별 MuJoCo 비교</h1>
<p>저장된 TT 가중치를 동결하고 CPU에서 10초 오픈루프 평가를 실행했습니다.
RL 열은 버전·과제마다 MuJoCo 보상으로 학습한 작은 denoising residual 정책입니다.</p>
<p>평가 seed {esc(options["seed"] + 20000)}부터 {esc(options["eval_seeds"])}개,
seed당 후보 {esc(options["candidates"])}개 · PPO {esc(options["rl_updates"])}회 ×
{esc(options["rl_rollouts"])} rollout · 원형 진행량 단위는 바퀴, 지그재그와 직선은 m.</p>
<p>TT 예측 목적함수는 생성 궤적 점수입니다. 실제 MuJoCo 보상에는 실행 종료와
과제별 물리 판정이 반영됩니다. 실제 보상 평균의 괄호에는 유효 후보 수를 표시합니다.
영상의 주황색은 TT 예상 경로, 초록색은 실제 MuJoCo 경로입니다.
체크포인트마다 원래 DDIM 스텝과 윈도 길이를 유지하며, 윈도 길이가 다르면 같은 seed에서도 노이즈 배열이 달라집니다.</p>
<p>재생 영상 {len(players)}개 · HTML 파일을 브라우저에서 직접 열면 영상이 재생됩니다.</p>
{''.join(tables)}
<h2>체크포인트와 실행 상태</h2><table><thead><tr><th>버전 및 파일</th>
<th>학습 설정</th><th>실행 상태</th></tr></thead><tbody>{''.join(inventory)}</tbody></table></html>"""
    target = root / "report.html"
    temporary = target.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(target)
    return len(players)


def save_index(out, index):
    path = out / "comparison.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_checkpoint(out, entry, options, resume):
    version_root = out / "versions" / entry["id"]
    version_root.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).with_name("benchmark_task_inference.py")),
               "--ckpt", entry["path"], "--out", str(version_root),
               "--device", "cpu", "--tasks", *TASK_NAMES, "--methods", "gradient", "rl",
               "--scales", "0", "1", "--record-video", "--video-backend", "auto",
               "--video-selection", "best", "--video-fps", "10"]
    for key, value in options.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if resume and version_root.exists():
        command.append("--resume")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS=str(options["cpu_threads"]),
               OPENBLAS_NUM_THREADS=str(options["cpu_threads"]))
    log_path = version_root.with_suffix(".log")
    print(f"{entry['label']}: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    entry["status"] = "complete" if completed.returncode == 0 else "failed"
    entry["detail"] = "" if completed.returncode == 0 else (
        f"종료 코드 {completed.returncode} · 로그 {log_path.relative_to(out)}")


def render_saved_versions(out, index):
    """Use only saved candidate archives to render mesh MP4s."""
    failed = []
    renderer = Path(__file__).with_name("render_task_videos.py")
    for entry in index["versions"]:
        root = out / "versions" / entry["id"]
        if not collect(root):
            continue
        command = [sys.executable, str(renderer), "--out", str(root),
                   "--backend", "osmesa", "--selection", "best", "--fps", "10", "--resume"]
        print(f"MuJoCo mesh: {entry['label']}", flush=True)
        if subprocess.run(command).returncode:
            failed.append(entry["label"])
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", help="Repeat to select exact checkpoints")
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    mode.add_argument("--render-only", action="store_true",
                      help="render saved physical replays with MuJoCo Renderer on CPU OSMesa")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--rl-updates", type=int, default=2)
    parser.add_argument("--rl-rollouts", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if not args.list_only and args.out is None:
        parser.error("--out is required unless using --list-only")
    if args.report_only or args.render_only:
        index = json.loads((args.out / "comparison.json").read_text(encoding="utf-8"))
        failed = render_saved_versions(args.out, index) if args.render_only else []
        build_comparison(args.out, index)
        if failed:
            raise SystemExit(f"MuJoCo mesh rendering failed: {', '.join(failed)}")
        print(f"Report: {args.out / 'report.html'}")
        return
    options = {key: getattr(args, key) for key in
               ("seed", "eval_seeds", "candidates", "rl_updates", "rl_rollouts", "cpu_threads")}
    if min(value for key, value in options.items() if key != "seed") < 1 or args.rl_rollouts < 2:
        parser.error("Use positive counts and at least 2 PPO rollouts per update")
    paths = checkpoint_inventory(args.checkpoint, [*SEARCH_ROOTS, *args.search_root])
    entries = [inspect_checkpoint(path) for path in paths]
    if args.list_only:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return
    out = args.out.resolve()
    index = dict(options=options, versions=entries)
    if args.resume:
        old = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
        if old["options"] != options or [v["path"] for v in old["versions"]] != [str(p) for p in paths]:
            parser.error("Resume requires the same checkpoint list and evaluation options")
        for entry, previous in zip(entries, old["versions"]):
            if ((out / "versions" / entry["id"]).exists() and
                    entry.get("sha256") != previous.get("sha256")):
                parser.error(f"Checkpoint changed since the first run: {entry['path']}. Use a fresh --out.")
            if entry["status"] == "ready" and previous["status"] == "complete":
                entry.update(status="complete", detail="")
    else:
        out.mkdir(parents=True, exist_ok=False)

    save_index(out, index)
    build_comparison(out, index)
    for entry in entries:
        if entry["status"] != "ready":
            continue
        run_checkpoint(out, entry, options, args.resume)
        save_index(out, index)
        build_comparison(out, index)
    print(f"Report: {out / 'report.html'}", flush=True)
    if not any(entry["status"] == "complete" for entry in entries):
        raise SystemExit("No checkpoint completed. See the report and version logs.")


if __name__ == "__main__":
    main()
