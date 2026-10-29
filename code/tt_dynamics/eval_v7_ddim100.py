"""Evaluate V7 100-step DDIM experiments using ACTUAL MJX rewards, not TT predictions.

Produces JSON; no success claim without executed, finite MJX trajectories.
Stage failure or missing artifacts is recorded as "not_run", never as success.
"""
import argparse
import base64
import hashlib
import html
import subprocess
import sys
import json
import math
from pathlib import Path
import numpy as np


def load(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def summarize(rows):
    valid = [r for r in rows if finite(r.get("mjx_total")) and r.get("execution_skipped") is not True
             and isinstance(r.get("initial_physics"), dict)
             and r["initial_physics"].get("physically_valid") is True]
    complete = [r for r in valid if r.get("mjx_fell_step") == -1]
    metrics = {"candidates": len(rows), "valid_mjx": len(valid),
               "completed": len(complete), "completion_rate": len(complete) / len(rows) if rows else 0}
    for key in ("mjx_total", "pred_total", "mjx_progress", "pred_progress", "path_err_m"):
        vals = np.array([r[key] for r in valid if finite(r.get(key))], dtype=float)
        metrics[key + "_median"] = float(np.median(vals)) if len(vals) else None
    gap = [abs(r["pred_total"] - r["mjx_total"]) for r in valid if finite(r.get("pred_total"))]
    metrics["reward_abs_gap_median"] = float(np.median(gap)) if gap else None
    return metrics


def rank_corr(a, b):
    """Spearman rho (average ties), no scipy dependency."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return None
    def ranks(v):
        idx = np.argsort(v, kind="stable")
        out = np.empty(len(v), float)
        s = 0
        while s < len(v):
            e = s + 1
            while e < len(v) and v[idx[e]] == v[idx[s]]:
                e += 1
            out[idx[s:e]] = (s + e - 1) / 2.0
            s = e
        return out
    x, y = ranks(a), ranks(b)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def eta2_p(values, groups, n=2000, seed=0):
    y, g = np.asarray(values, float), np.asarray(groups)
    good = np.isfinite(y)
    y, g = y[good], g[good]
    if len(y) < 4 or len(np.unique(g)) < 2:
        return {"eta2": None, "permutation_p": None}
    def score(labels):
        total = float(((y - y.mean()) ** 2).sum())
        if total <= 1e-12:
            return 0.0
        between = sum(int((labels == k).sum()) * (y[labels == k].mean() - y.mean()) ** 2
                      for k in np.unique(labels))
        return float(between / total)
    observed = score(g)
    rng = np.random.default_rng(seed)
    p = (1 + sum(score(rng.permutation(g)) >= observed for _ in range(n))) / (n + 1)
    return {"eta2": observed, "permutation_p": float(p)}


def analyze_fixed(fixed_summary, joint_summary, scale):
    tag = f"scale_{scale:g}"
    if fixed_summary is None or joint_summary is None:
        return {"status": "not_run"}
    rows = fixed_summary.get("runs", {}).get(tag, [])
    joint = joint_summary.get("runs", {}).get(tag, [])
    # generate_task.py currently does NOT persist a fixed_e list in summary.json.
    # Its contract is contiguous candidates per shape, in caller-specified order.
    # Reconstruct grouping from *observed E values*, fail if values are not
    # contiguous or do not match expected 16 morphology experiment.
    names = []
    groups = []
    previous = None
    for row in rows:
        e = row.get("E")
        if not isinstance(e, list) or len(e) != 13 or not np.isfinite(np.asarray(e, float)).all():
            return {"status": "invalid", "reason": "missing/nonfinite generated E"}
        current = tuple(round(float(x), 5) for x in e)
        if current != previous:
            if current in names:
                return {"status": "invalid", "reason": "fixed morphology appears in noncontiguous blocks"}
            names.append(current)
            previous = current
        groups.append(len(names) - 1)
    n_design = len(names)
    if n_design != 16 or len(rows) % n_design:
        return {"status": "invalid", "reason": f"expected 16 fixed shapes; got {n_design}"}
    per = len(rows) // n_design
    if any(groups.count(k) != per for k in range(n_design)):
        return {"status": "invalid", "reason": "unequal number of seeds per fixed shape"}
    grouped = np.asarray(groups, dtype=int)
    pred = np.array([r.get("pred_total", np.nan) for r in rows], dtype=float)
    mjx = np.array([r.get("mjx_total") if finite(r.get("mjx_total")) else np.nan for r in rows], float)
    pred_means, mjx_means = [], []
    for k in range(n_design):
        ps, ms = pred[grouped == k], mjx[grouped == k]
        pred_means.append(float(np.mean(ps)) if np.isfinite(ps).all() else None)
        mjx_means.append(float(np.mean(ms)) if np.isfinite(ms).all() else None)
    rho = rank_corr(pred_means, mjx_means)
    variation = eta2_p(pred, grouped)
    path = summarize(joint)
    ready = bool(rho is not None and rho > 0.6 and variation["permutation_p"] is not None
                 and variation["permutation_p"] < 0.05)
    return {"status": "evaluated", "fixed": summarize(rows), "joint": path,
            "n_shapes": n_design, "samples_per_shape": per,
            "predicted_reward_by_shape": pred_means,
            "actual_mjx_reward_by_shape": mjx_means,
            "valid_mjx_samples_per_shape": [int(np.isfinite(mjx[grouped == k]).sum()) for k in range(n_design)],
            "spearman_shape_reward_rho": rho,
            "predicted_reward_e_effect": variation,
            "morphology_gate_pass": ready,
            "morphology_gate_definition": "predicted E effect permutation p<0.05 AND shape-mean Spearman rho>0.6",
            "joint_mjx_reward_better_than_fixed_median": (
                path["mjx_total_median"] > summarize(rows)["mjx_total_median"]
                if finite(path["mjx_total_median"]) and finite(summarize(rows)["mjx_total_median"]) else None)}


def evaluate(root, stage, fixed_scale):
    result = {"stage": stage, "experiment": str(root), "diffusion_steps_requested": 100,
              "status": "audit_only_no_gpu_execution_by_this_script"}
    if stage in ("smoke", "all"):
        d = load(root / "smoke_zigzag_2s/summary.json")
        if d is None:
            result["smoke"] = {"status": "not_run"}
        else:
            result["smoke"] = {"status": "evaluated",
                               "scales": {scale: summarize(rows) for scale, rows in d["runs"].items()}}
            base = result["smoke"]["scales"].get("scale_0", {})
            # Gate is about trajectory agreement, not just total reward.
            err = base.get("path_err_m_median")
            result["smoke"]["dynamics_gate_pass"] = bool(finite(err) and err < 0.15
                                                   and base.get("valid_mjx", 0) > 0)
    if stage in ("codesign", "all"):
        result["codesign"] = {task: analyze_fixed(
            load(root / f"codesign/fixedE_{task}_4s/summary.json"),
            load(root / f"codesign/joint_{task}_4s/summary.json"),
            fixed_scale) for task in ("zigzag", "circle")}
    if stage in ("full", "all"):
        result["full"] = {}
        for task in ("zigzag", "circle"):
            d = load(root / f"full_{task}_10s/summary.json")
            result["full"][task] = {"status": "evaluated",
                                   "scales": {scale: summarize(rows) for scale, rows in d["runs"].items()}} if d else {"status": "not_run"}
    return result


def run_generation(command):
    """Reuse only completed results with the same request and checkpoint bytes."""
    out = Path(command[command.index("--out") + 1])
    checkpoint = Path(command[command.index("--ckpt") + 1]) / "checkpoint.pt"
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    request = {"command": command, "checkpoint_sha256": digest}
    if out.exists():
        if load(out / "request.json") == request and load(out / "summary.json") is not None:
            print("Reusing verified result:", out)
            return
        raise ValueError(f"Existing result has missing/different settings or is incomplete: {out}; choose a new V7_100_ROOT")
    subprocess.run(command, check=True)
    if load(out / "summary.json") is None:
        raise ValueError(f"Generation did not produce summary.json: {out}")
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")


def write_report(root, audit, scale):
    """Render observed statistics and existing original-resolution MJX videos."""
    sections = []
    for task in ("zigzag", "circle"):
        directory = root / f"codesign/joint_{task}_4s"
        summary = load(directory / "summary.json")
        rows = summary.get("runs", {}).get(f"scale_{scale:g}", []) if summary else []
        choices = []
        for label, key in (("TT 예측 보상 1위", "pred_total"), ("MJX 실제 보상 1위", "mjx_total")):
            valid = [i for i, row in enumerate(rows) if finite(row.get(key))
                     and (key != "mjx_total" or summarize([row])["valid_mjx"] == 1)]
            if valid:
                choices.append((label, max(valid, key=lambda i: rows[i][key])))
        cards = []
        for label, i in choices:
            row = rows[i]
            video = directory / f"videos/scale_{scale:g}_cand{i}.mp4"
            if video.is_file():
                encoded = base64.b64encode(video.read_bytes()).decode()
                media = f'<video controls muted loop playsinline src="data:video/mp4;base64,{encoded}"></video>'
            else:
                media = '<p>이 후보의 영상 없음: 실행 제외 여부 또는 생성 설정 확인</p>'
            cards.append(f'<article><h3>{label} · joint {i}</h3><p>예측 R {row.get("pred_total")} / MJX R {row.get("mjx_total")}</p>{media}</article>')
        selected = {i for _, i in choices}
        others = []
        for i, row in enumerate(rows):
            if i in selected:
                continue
            video = directory / f"videos/scale_{scale:g}_cand{i}.mp4"
            media = f'<video controls muted playsinline src="data:video/mp4;base64,{base64.b64encode(video.read_bytes()).decode()}"></video>' if video.is_file() else '<p>영상 없음</p>'
            others.append(f'<article><h3>joint {i}</h3><p>예측 R {row.get("pred_total")} / MJX R {row.get("mjx_total")}</p>{media}</article>')
        sections.append(f'<section><h2>{task} · joint 후보 비교</h2><div class="grid">' + ''.join(cards)
                        + '</div><details><summary>나머지 joint 후보</summary><div class="grid">' + ''.join(others) + '</div></details></section>')
    text = html.escape(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    report = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V7 DDIM100 평가</title><style>body{font:16px/1.6 system-ui;background:#101820;color:#e5eef3;padding:24px}main{max-width:1200px;margin:auto}section,article{padding:18px;border:1px solid #354653;border-radius:10px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}video{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<main><h1>V7 DDIM100 · 실제 MJX 평가</h1><p>왼쪽은 TT 예측 보상 1위, 오른쪽은 MJX 실제 보상 1위 후보의 실제 실행 영상이다. 같은 후보가 두 기준에서 1위이면 동일 영상이 표시된다.</p>
<p>completed는 제한 시간 생존이며 트랙 완주가 아니다. not_run은 미실행이고, gate false는 기준 미달이다. 100단계의 효과는 같은 E-swap 데이터를 사용한 V7-DDIM5와 비교해야 한다.</p>"""
    report += ''.join(sections) + '<section><h2>관측 수치 및 gate 판정</h2><pre>' + text + '</pre></section></main></html>'
    (root / "report.html").write_text(report)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        run_generation(sys.argv[2:])
        return
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("smoke", "codesign", "full", "all"), default="all")
    p.add_argument("--fixed-scale", type=float, default=10.)
    p.add_argument("--require-smoke-gate", action="store_true")
    args = p.parse_args()
    data = evaluate(args.root, args.stage, args.fixed_scale)
    out = args.root / f"audit_{args.stage}.json"
    args.root.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    write_report(args.root, data, args.fixed_scale)
    print("Saved:", out)
    if args.require_smoke_gate and data.get("smoke", {}).get("dynamics_gate_pass") is not True:
        raise SystemExit("Smoke gate failed or missing; stop before codesign/full. See report.html")


if __name__ == "__main__":
    main()
