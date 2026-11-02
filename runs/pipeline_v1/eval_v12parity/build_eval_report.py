"""run_eval.py 결과로 기준 run(ref) 대비 비교표 + 사전 기준 판정 (gate.json) + report.html 을 만든다. 물리 재실행 없음.

    python build_eval_report.py <run_dir>

ref = conditions["reference_run"] (기본 v12). 사전 기준은 conditions["evaluation_gate"] 의 키로 켜진다:
  wrong_action_velocity_extra_error_min, wrong_action_ratio_vs_reference_min, E_swap_forward_change_sign_agreement_min,
  joint_survival_at_least_v12_zigzag/circle 또는 joint_survival_at_least_reference, delta_r2_min, legacy_moving_rmse_max_ratio_vs_reference.
"""
import base64
import datetime
import html
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RUN = Path(sys.argv[1]).resolve()
ROOT = RUN.parents[2]
OUT = RUN / "evaluation"
V9 = RUN.parent / "fast_mode24_v9_e_coverage"
V12 = RUN.parent / "fast_mode24_v12_pd_kp200_mode1"
LABELS = dict(height="몸통 높이", tilt="몸통 기울기", linvel="이동 속도", yaw="방향", pos="위치", joint="관절각")
COLORS = dict(v8="#64748b", v9="#b7791f", ref="#c2410c", pd="#087f8c")
DEFAULT_GATE = dict(wrong_action_velocity_extra_error_min=0.01, E_swap_forward_change_sign_agreement_min=0.7,
                    joint_survival_at_least_v12_zigzag=7, joint_survival_at_least_v12_circle=10)
TASKS = ("zigzag", "circle")


def read(path):
    return json.loads(path.read_text())


def eta(values, groups):
    denominator = ((values - values.mean()) ** 2).sum()
    return float(sum((groups == k).sum() * (values[groups == k].mean() - values.mean()) ** 2
                     for k in np.unique(groups)) / denominator) if denominator else 0.


def sign_agreement(rows):
    """고정 E 64 후보: (a) 후보별 예측 dx 와 MJX dx 부호 일치, (b) 형상 평균 쌍 (E 교체) 변화 부호 일치 (8-43 기준 2)."""
    groups = np.repeat(np.arange(16), 4)
    pred = np.array([row["pred_progress"] for row in rows])
    actual = np.array([row["mjx_progress"] if row["mjx_progress"] is not None else np.nan for row in rows])
    ok = np.isfinite(actual)
    per_candidate = int((np.sign(pred[ok]) == np.sign(actual[ok])).sum())
    pm = np.array([pred[groups == i].mean() for i in range(16)])
    am = np.array([np.nanmean(actual[groups == i]) if ok[groups == i].any() else np.nan for i in range(16)])
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16) if np.isfinite(am[i]) and np.isfinite(am[j])]
    agree = sum(np.sign(pm[i] - pm[j]) == np.sign(am[i] - am[j]) for i, j in pairs)
    return dict(candidate_sign_agree=per_candidate, candidate_valid=int(ok.sum()),
                pair_sign_agree=int(agree), pairs=len(pairs),
                pair_sign_rate=float(agree / len(pairs)) if pairs else None,
                fixed_dx_median_m=float(np.nanmedian(actual)) if ok.any() else None)


def task_result(folder, task):
    rows = read(folder / f"fixedE_{task}_4s/summary.json")["runs"]["scale_10"]
    assert len(rows) == 64
    assert all(all(row["E"] == rows[i * 4]["E"] for row in rows[i * 4:(i + 1) * 4]) for i in range(16))
    groups = np.repeat(np.arange(16), 4)
    pred = np.array([row["pred_total"] for row in rows])
    actual = np.array([row["mjx_total"] if row["mjx_total"] is not None else np.nan for row in rows])
    pred_mean = np.array([pred[groups == i].mean() for i in range(16)])
    actual_mean = np.array([np.nanmean(actual[groups == i]) if np.isfinite(actual[groups == i]).any() else np.nan for i in range(16)])
    effect = eta(pred, groups)
    rng = np.random.default_rng(0)
    p = float(np.mean([eta(pred, rng.permutation(groups)) >= effect for _ in range(10000)]))
    correlation = float(spearmanr(pred_mean, actual_mean).statistic)
    rank = correlation if np.isfinite(correlation) else None
    joint = read(folder / f"joint_{task}_4s/summary.json")["runs"]["scale_10"]
    smoke = read(folder / "smoke_zigzag_joint_c16/summary.json")["runs"]["scale_0"]
    assert len(joint) == len(smoke) == 16
    paths = [row["path_err_m"] for row in smoke if row["path_err_m"] is not None]
    path_error = float(np.median(paths)) if paths else None
    rewards = [row["mjx_total"] for row in joint if row["mjx_total"] is not None]
    progress = [row["mjx_progress"] for row in joint if row["mjx_progress"] is not None]
    return dict(pred_effect_p=p, pred_effect_eta2=effect, rank_correlation=rank,
                smoke_path_error_m=path_error, smoke_survived=sum(row["mjx_fell_step"] == -1 for row in smoke),
                smoke_total=len(smoke), fixed_valid=int(np.isfinite(actual).sum()), fixed_total=len(rows),
                fixed_fell=sum(row["mjx_fell_step"] != -1 for row in rows),
                joint_survived=sum(row["mjx_fell_step"] == -1 for row in joint), joint_total=len(joint),
                joint_reward_median=float(np.median(rewards)) if rewards else None,
                joint_progress_median=float(np.median(progress)) if progress else None,
                joint_progress_unit="m" if task == "zigzag" else "laps",
                gate=p < .05 and rank is not None and rank > .6 and path_error is not None and path_error < .15,
                predicted_design_reward=pred_mean.tolist(),
                actual_design_reward=[float(x) if np.isfinite(x) else None for x in actual_mean],
                **sign_agreement(rows))


def fmt(value, digits=3):
    return "미정" if value is None else f"{value:.{digits}f}"


def extra(source, g, key):
    return source["start30"][g][key + "_moving"] - source["start30"][g]["pin_moving"]


def main():
    conditions = read(RUN / "conditions.json")
    ref = ROOT / conditions.get("reference_run", str(V12.relative_to(ROOT)))
    ref_name = ref.name
    held, seen, native, mode1 = [read(OUT / name) for name in ("given_a_held.json", "given_a_seen.json", "given_a_native.json", "given_a_held_mode1.json")]
    mode1_ref = mode1.get("ref", mode1.get("v12"))
    assert len({held[name]["start30"]["pairs"] for name in COLORS}) == 1
    assert held["pd"]["start30"]["pairs"] == 516
    audit = read(RUN / "data_audit.json") if (RUN / "data_audit.json").exists() else read(V12 / "data_audit.json")
    progress = [json.loads(line) for line in (RUN / "tt_joint/progress.jsonl").read_text().splitlines()]
    assert progress[-1]["step"] == 60000
    training = read(RUN / "tt_joint/config.json")
    baseline = read(ROOT / conditions["training_reference"])
    for key in ("horizon", "steps", "batch", "lr", "d_model", "layers", "diff_steps", "seed", "norm", "clip", "sampler", "film", "modes", "mode_weights"):
        assert training[key] == baseline[key], key
    previous = read(V9 / "evaluation_local/comparison.json")
    tasks = {name: previous["tasks"][name] for name in ("v8", "v9")}
    # ref 의 MJX 결과: ref 폴더의 comparison.json (v12 형식 "pd" 또는 eval_v12parity 형식 "pd") + fixedE summary 재계산
    ref_comparison = read(ref / "evaluation/comparison.json")
    ref_mode1_file = read(ref / "evaluation/given_a_held_mode1.json")
    ref_mode1 = ref_mode1_file["pd"]
    tasks["ref"] = ref_comparison["tasks"]["pd"]
    for task in TASKS:
        rows = read(ref / f"evaluation/check_w1/fixedE_{task}_4s/summary.json")["runs"]["scale_10"]
        tasks["ref"][task].update(sign_agreement(rows), fixed_fell=sum(row["mjx_fell_step"] != -1 for row in rows))
    tasks["pd"] = {task: task_result(OUT / "check_w1", task) for task in TASKS}
    metrics = {}
    for group in LABELS:
        metrics[group] = {}
        for name in COLORS:
            row = held[name]["start30"][group]
            n = native[name]["start30"][group]
            metrics[group][name] = dict(legacy_agreement=row["e_agree_moving"],
                legacy_moving_rmse=row["pin_moving"], legacy_wrongE_penalty=row["wrongE_moving"] - row["pin_moving"],
                legacy_wrongA_penalty=row["wrongA_moving"] - row["pin_moving"], native_track=n["pin_track"],
                native_moving_rmse=n["pin_moving"], native_wrongE_penalty=n["wrongE_moving"] - n["pin_moving"],
                native_wrongA_penalty=n["wrongA_moving"] - n["pin_moving"])
    main_pass = all(metrics[g]["pd"]["legacy_agreement"] - metrics[g]["v8"]["legacy_agreement"] >= .1
                    for g in ("height", "tilt", "linvel"))
    # Mode 1 조건 직접 측정: 틀린 A/E 추가 오차 = wrongX_moving - pin_moving (start30)
    m1 = {name: {g: dict(pin=src["start30"][g]["pin_moving"], wrongA_extra=extra(src, g, "wrongA"), wrongE_extra=extra(src, g, "wrongE"))
                 for g in LABELS} for name, src in (("ref", ref_mode1), ("pd", mode1["pd"]))}
    assert abs(mode1_ref["start30"]["linvel"]["pin_moving"] - ref_mode1["start30"]["linvel"]["pin_moving"]) < 1e-4, "ref Mode-1 재측정이 ref 폴더 기록과 다름"
    delta = read(OUT / "delta_r2.json") if (OUT / "delta_r2.json").exists() else None
    ref_delta = read(ref / "evaluation/delta_r2.json") if (ref / "evaluation/delta_r2.json").exists() else None

    gate_cfg = conditions.get("evaluation_gate", DEFAULT_GATE)
    gate = dict(criteria=gate_cfg, reference=ref_name)
    wa_ref, wa_run = m1["ref"]["linvel"]["wrongA_extra"], m1["pd"]["linvel"]["wrongA_extra"]
    ratio = gate_cfg.get("wrong_action_ratio_vs_reference_min")
    gate["wrong_action_velocity_extra_error"] = dict(ref=wa_ref, run=wa_run, pass_=bool(
        wa_run >= gate_cfg["wrong_action_velocity_extra_error_min"] and (ratio is None or wa_run >= ratio * wa_ref)))
    sign_run = {t: tasks["pd"][t]["pair_sign_rate"] for t in TASKS}
    gate["e_swap_sign"] = dict(ref={t: tasks["ref"][t]["pair_sign_rate"] for t in TASKS}, run=sign_run,
                               pass_=all(r is not None and r >= gate_cfg["E_swap_forward_change_sign_agreement_min"] for r in sign_run.values()))
    surv_ref = {t: tasks["ref"][t]["joint_survived"] for t in TASKS}
    surv_run = {t: tasks["pd"][t]["joint_survived"] for t in TASKS}
    thr = surv_ref if gate_cfg.get("joint_survival_at_least_reference") else {
        "zigzag": gate_cfg["joint_survival_at_least_v12_zigzag"], "circle": gate_cfg["joint_survival_at_least_v12_circle"]}
    gate["joint_survival"] = dict(ref=surv_ref, run=surv_run, threshold=thr, pass_=all(surv_run[t] >= thr[t] for t in TASKS))
    checks = ["wrong_action_velocity_extra_error", "e_swap_sign", "joint_survival"]
    if "delta_r2_min" in gate_cfg:
        assert delta is not None, "delta_r2.json 없음 (run_eval delta_r2 단계)"
        r2_run = {k: delta[k]["r2"]["body"] if delta[k].get("pairs") else None for k in ("E", "A")}
        r2_ref = {k: ref_delta[k]["r2"]["body"] if ref_delta and ref_delta[k].get("pairs") else None for k in ("E", "A")}
        gate["delta_r2"] = dict(ref=r2_ref, run=r2_run, pass_=all(v is not None and v > gate_cfg["delta_r2_min"] for v in r2_run.values()))
        checks.append("delta_r2")
    if "legacy_moving_rmse_max_ratio_vs_reference" in gate_cfg:
        rr = {g: metrics[g]["pd"]["legacy_moving_rmse"] / metrics[g]["ref"]["legacy_moving_rmse"] for g in ("height", "tilt", "linvel")}
        gate["no_regression_rmse"] = dict(ratio=rr, pass_=all(v <= gate_cfg["legacy_moving_rmse_max_ratio_vs_reference"] for v in rr.values()))
        checks.append("no_regression_rmse")
    gate["checks"] = checks
    gate["all_pass"] = all(gate[c]["pass_"] for c in checks)
    (RUN / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False))
    resources = [json.loads(line) for line in (RUN / "eval_resources.jsonl").read_text().splitlines()]
    limits = dict(min_ram_available_gib=min(r["ram_available_gib"] for r in resources),
                  max_gpu_used_mib=max(float(r["gpu"].split(",")[0]) for r in resources if r["gpu"]))
    result = dict(main_sensitivity_pass=main_pass, metrics=metrics, mode1=m1, gate=gate, tasks=tasks, delta_r2=delta, resources=limits,
                  paired_cases=516, native_conditions=native["_conditions"], dataset=audit,
                  native_cases_start30=native["pd"]["start30"]["episodes"],
                  final_validation_loss=progress[-1]["val_loss"], final_validation_raw_loss=progress[-1].get("val_raw_loss"),
                  training_minutes=progress[-1]["min"])
    (OUT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

    features = list(LABELS)
    x = np.arange(len(features))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for i, (name, color) in enumerate(COLORS.items()):
        label = ref_name if name == "ref" else (RUN.name if name == "pd" else name)
        axes[0].bar(x + (i - 1.5) * .2, [metrics[g][name]["legacy_agreement"] for g in features], .2, label=label, color=color)
        axes[1].bar(x + (i - 1.5) * .2, [metrics[g][name]["native_moving_rmse"] for g in features], .2, label=label, color=color)
    for ax in axes:
        ax.set_xticks(x, features, rotation=25)
        ax.legend(fontsize=8)
        ax.axhline(0, color="#999", linewidth=.7)
    axes[0].set_title("Frozen legacy cases: morphology-change agreement")
    axes[1].set_title("Native validation cases: moving RMSE (lower is better)")
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), layout="constrained")
    key = "val_raw_loss" if "val_raw_loss" in progress[-1] else "val_loss"
    axes[0].plot([r["step"] for r in progress], [r.get("raw_loss", r["loss"]) for r in progress], alpha=.45, label="train minibatch")
    axes[0].plot([r["step"] for r in progress], [r[key] for r in progress], label=f"validation ({key})")
    axes[0].set(xlabel="Training steps", ylabel="Loss", yscale="log")
    axes[0].legend()
    for kind in ("E", "A"):
        if f"val_delta_{kind}_r2" in progress[-1]:
            axes[1].plot([r["step"] for r in progress], [r[f"val_delta_{kind}_r2"] for r in progress], label=f"val R²_Δ {kind} (single-k monitor)")
    axes[1].axhline(0, color="#999", linewidth=.7)
    axes[1].set(xlabel="Training steps", ylabel="R²_Δ (monitor)", ylim=(-1, 1))
    axes[1].legend()
    fig.savefig(OUT / "training.png", dpi=160)
    plt.close(fig)

    run_label = RUN.name
    change = conditions.get("single_variable_vs_v12", conditions.get("loss_semantics", ""))
    if isinstance(change, dict):
        change = change.get("change", json.dumps(change, ensure_ascii=False))
    ok = lambda b: "통과" if b else "미달"
    g1 = gate["wrong_action_velocity_extra_error"]
    g1_thr = f"≥ {gate_cfg['wrong_action_velocity_extra_error_min']}" + (f" 이고 ≥ ref × {ratio}" if ratio else "")
    gate_rows = (
        f"<tr><td>1. 틀린 A 추가 오차 (속도, Mode 1 조건)</td><td>{g1_thr}</td><td>{fmt(g1['ref'], 4)}</td><td>{fmt(g1['run'], 4)}</td><td>{ok(g1['pass_'])}</td></tr>"
        f"<tr><td>2. E 교체 시 예측 전진 변화 부호 일치 (형상 평균 쌍, zigzag / circle)</td><td>≥ {gate_cfg['E_swap_forward_change_sign_agreement_min']}</td><td>{fmt(gate['e_swap_sign']['ref']['zigzag'], 2)} / {fmt(gate['e_swap_sign']['ref']['circle'], 2)}</td><td>{fmt(sign_run['zigzag'], 2)} / {fmt(sign_run['circle'], 2)}</td><td>{ok(gate['e_swap_sign']['pass_'])}</td></tr>"
        f"<tr><td>3. 공동 생성 생존 (zigzag / circle, /16)</td><td>≥ {thr['zigzag']} / {thr['circle']}</td><td>{surv_ref['zigzag']} / {surv_ref['circle']}</td><td>{surv_run['zigzag']} / {surv_run['circle']}</td><td>{ok(gate['joint_survival']['pass_'])}</td></tr>")
    if "delta_r2" in gate:
        d = gate["delta_r2"]
        gate_rows += f"<tr><td>4. held-out 개입 쌍 Δ 설명 비율 R²_Δ (몸통 14 채널, E 쌍 / A 쌍, DDIM Mode 1)</td><td>&gt; {gate_cfg['delta_r2_min']}</td><td>{fmt(d['ref']['E'], 3)} / {fmt(d['ref']['A'], 3)}</td><td>{fmt(d['run']['E'], 3)} / {fmt(d['run']['A'], 3)}</td><td>{ok(d['pass_'])}</td></tr>"
    if "no_regression_rmse" in gate:
        rr = gate["no_regression_rmse"]["ratio"]
        gate_rows += f"<tr><td>5. 퇴보 금지: 516쌍 움직임 RMSE 비 (높이 / 기울기 / 속도, run ÷ ref)</td><td>≤ {gate_cfg['legacy_moving_rmse_max_ratio_vs_reference']}</td><td>1.00</td><td>{rr['height']:.2f} / {rr['tilt']:.2f} / {rr['linvel']:.2f}</td><td>{ok(gate['no_regression_rmse']['pass_'])}</td></tr>"
    mode1_rows = "".join(
        f"<tr><td>{LABELS[g]}</td><td>{fmt(m1['ref'][g]['pin'])}</td><td>{fmt(m1['ref'][g]['wrongA_extra'], 4)}</td><td>{fmt(m1['ref'][g]['wrongE_extra'], 4)}</td>"
        f"<td>{fmt(m1['pd'][g]['pin'])}</td><td>{fmt(m1['pd'][g]['wrongA_extra'], 4)}</td><td>{fmt(m1['pd'][g]['wrongE_extra'], 4)}</td></tr>" for g in LABELS)
    delta_rows = ""
    if delta:
        for kind in ("E", "A"):
            if delta[kind].get("pairs"):
                r = delta[kind]
                rr_ = ref_delta[kind] if ref_delta and ref_delta[kind].get("pairs") else None
                for grp in ("body", "pos", "height", "yaw", "linvel"):
                    delta_rows += (f"<tr><td>{kind} 쌍 · {grp}</td><td>{r['pairs']}</td><td>{fmt(rr_['r2'][grp]) if rr_ else '없음'}</td><td>{fmt(r['r2'][grp])}</td>"
                                   f"<td>{fmt(r['delta_rms_true'][grp], 4)}</td><td>{fmt(r['delta_rms_pred'][grp], 4)}</td></tr>")
                delta_rows += f"<tr><td>{kind} 쌍 · 끝 step 전진 변화 부호 일치</td><td>{r['pairs']}</td><td>{fmt(rr_['end_forward_change_sign_agree'], 2) if rr_ else '없음'}</td><td>{fmt(r['end_forward_change_sign_agree'], 2)}</td><td></td><td></td></tr>"
    table_rows = native_rows = ""
    for g, label in LABELS.items():
        for name in COLORS:
            r = metrics[g][name]
            shown = ref_name if name == "ref" else name
            table_rows += f"<tr><td>{label}</td><td>{shown}</td><td>{fmt(r['legacy_agreement'])}</td><td>{fmt(r['legacy_moving_rmse'])}</td><td>{fmt(r['legacy_wrongE_penalty'])}</td><td>{fmt(r['legacy_wrongA_penalty'])}</td></tr>"
            native_rows += f"<tr><td>{label}</td><td>{shown}</td><td>{fmt(r['native_track'])}</td><td>{fmt(r['native_moving_rmse'])}</td><td>{fmt(r['native_wrongE_penalty'])}</td><td>{fmt(r['native_wrongA_penalty'])}</td></tr>"
    task_rows = ""
    for name in COLORS:
        for task, r in tasks[name].items():
            p_label = "&lt;0.0001" if r["pred_effect_p"] == 0 else fmt(r["pred_effect_p"], 4)
            unit = "m" if task == "zigzag" else "바퀴"
            sign = f"{r['candidate_sign_agree']}/{r['candidate_valid']} · 쌍 {fmt(r['pair_sign_rate'], 2)}" if "pair_sign_rate" in r else "미측정"
            shown = ref_name if name == "ref" else name
            task_rows += f"<tr><td>{shown} · {task}</td><td>{p_label}</td><td>{fmt(r['rank_correlation'])}</td><td>{sign}</td><td>{r.get('fixed_fell', '미측정')}</td><td>{fmt(r['smoke_path_error_m'] * 100 if r['smoke_path_error_m'] is not None else None,1)}</td><td>{r['smoke_survived']}/{r['smoke_total']}</td><td>{r['joint_survived']}/{r['joint_total']}</td><td>{fmt(r['joint_progress_median'])} {unit}</td><td>{'통과' if r['gate'] else '미달'}</td></tr>"
    n_pass = sum(gate[c]["pass_"] for c in checks)
    conclusion = f"{run_label}: 사전 기준 {len(checks)}개 중 {n_pass}개 통과 → " + ("통과" if gate["all_pass"] else "미달") + f" (기준 run = {ref_name}). "
    conclusion += "기존 v8 대비 E 효과 일치도 +0.1 기준은 " + ("통과." if main_pass else "미달.")
    videos = best_rows = best_videos = ""
    for task in TASKS:
        files = sorted((OUT / f"check_w1/joint_{task}_4s/videos").glob("*.mp4"))
        joint = read(OUT / f"check_w1/joint_{task}_4s/summary.json")["runs"]["scale_10"]
        survived = [i for i, row in enumerate(joint) if row["mjx_fell_step"] == -1 and row["mjx_total"] is not None]
        if not files or not survived:
            best_rows += f"<tr><td>{task}</td><td colspan=5>생존 후보 없음 (영상 없음)</td></tr>"
            continue
        best = max(survived, key=lambda i: joint[i]["mjx_total"])
        tt_pick = max(survived, key=lambda i: joint[i]["pred_total"])
        rank = 1 + sum(joint[i]["pred_total"] > joint[best]["pred_total"] for i in survived)
        unit = "m" if task == "zigzag" else "바퀴"
        best_path = OUT / f"check_w1/joint_{task}_4s/videos/scale_10_cand{best}.mp4"
        best_rows += (f"<tr><td>{task}</td><td>cand{best}</td><td>{joint[best]['mjx_total']:.3f}</td><td>{joint[best]['mjx_progress']:.3f} {unit}</td>"
                      f"<td>{joint[best]['pred_total']:.3f} ({rank}위/{len(survived)})</td><td>cand{tt_pick}: {joint[tt_pick]['mjx_total']:.3f}</td></tr>")
        if best_path.exists():
            best_videos += f'<h3>{task} · cand{best}</h3><video controls src="{best_path.relative_to(RUN)}"></video>'
        videos += f"<h3>{task}: 나머지 공동 생성 후보</h3>"
        for path in files:
            if path != best_path:
                videos += f'<details><summary>{html.escape(path.stem)}</summary><video controls src="{path.relative_to(RUN)}"></video></details>'
    heading = "<tr><th>항목</th><th>모델</th><th>일치도</th><th>움직임 RMSE</th><th>틀린 E 추가 오차</th><th>틀린 A 추가 오차</th></tr>"
    today = datetime.date.today().isoformat()
    delta_section = ""
    if delta:
        delta_section = f'''<h2 id="delta">held-out 개입 쌍 Δ 설명 비율 (delta_r2.json)</h2><p>checkpoint 의 검증 분할 안에서 둘 다 32 step 유효한 쌍 (종류별 최대 1024). Mode 1 (E·s0·A 알려짐) DDIM 결정적 예측 8 seed 평균, 창 32, 자기 시작 기준. R²_Δ = 1 − Σ(Δ_hat − Δ)² / Σ(Δ − mean)². 평균을 내놓는 모델은 ≈ 0, 차이를 맞히면 양수. 학습 중 로그의 R²_Δ 는 단일 noise 단계 x0_hat 기준이라 과대 (x_t 가 x0 를 담고 있음) → 여기 값이 판정용.</p>
<table><tr><th>쌍 · 채널</th><th>쌍 수</th><th>{html.escape(ref_name)} R²_Δ</th><th>{html.escape(run_label)} R²_Δ</th><th>실제 Δ RMS</th><th>예측 Δ RMS</th></tr>{delta_rows}</table>'''
    text = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(run_label)} · 평가</title><style>body{{font:16px/1.8 system-ui;max-width:1160px;margin:32px auto;padding:20px;color:#243746}}nav a{{margin-right:20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #d4dfe4}}th{{background:#eef4f6}}img,video{{width:100%}}h2{{margin-top:38px}}details{{margin:8px 0}}.lead{{background:#edf7f7;padding:20px;border-left:5px solid #087f8c}}code{{overflow-wrap:anywhere}}</style>
<h1>{html.escape(run_label)}: {html.escape(ref_name)} 와 같은 평가, 사전 기준 판정</h1><p>{today} · 단변수 변경: {html.escape(str(change))}</p>
<nav><a href="#result">판정</a><a href="#gate">사전 기준</a><a href="#mode1">Mode 1 측정</a><a href="#delta">Δ 설명 비율</a><a href="#best">최고 후보</a><a href="#legacy">기존 평가</a><a href="#native">검증 평가</a><a href="#mjx">MJX 실행</a><a href="#videos">영상</a></nav>
<h2 id="result">판정</h2><p class="lead"><strong>{conclusion}</strong></p>
<h2 id="gate">사전 기준 (기준 run = {html.escape(ref_name)}, 평가 조건 동일)</h2>
<table><tr><th>기준</th><th>통과선</th><th>{html.escape(ref_name)}</th><th>{html.escape(run_label)}</th><th>판정</th></tr>{gate_rows}</table>
<p>기준 1 은 기존 시험 형상 8개·516쌍·시작 프레임 30 에서 A 를 Mode 1 조건(알려진 입력)으로 넣고, 다른 모션의 A 로 바꿨을 때 늘어나는 움직임 RMSE (v8 오차 단위). 기준 2 는 고정 E 64 후보(16 형상 × 4)의 형상 평균 예측 전진과 실제 MJX 전진을 형상 쌍마다 비교해 변화 부호가 같은 비율 (낙상 형상은 쌍에서 제외). 기준 3 은 공동 생성 16 후보 중 4초 생존 수. 원자료 <a href="gate.json">gate.json</a>.</p>
<h2 id="mode1">Mode 1 조건 직접 측정 (given_a_held_mode1.json, start 30)</h2>
<table><tr><th>항목</th><th>{html.escape(ref_name)} 올바른 A 오차</th><th>틀린 A 추가</th><th>틀린 E 추가</th><th>{html.escape(run_label)} 올바른 A 오차</th><th>틀린 A 추가</th><th>틀린 E 추가</th></tr>{mode1_rows}</table>
{delta_section}
<h2 id="best">가장 잘 나온 후보</h2><p>Task별 4초 공동 생성 16개 중 낙상 없이 실제 MJX 보상이 가장 높은 후보. 사후 선택이므로 모델의 후보 선택 능력을 뜻하지 않는다.</p>
<table><tr><th>Task</th><th>후보</th><th>실제 보상</th><th>실제 진행</th><th>TT 예측 보상 (생존 후보 중 순위)</th><th>TT 예측 1위 후보의 실제 보상</th></tr>{best_rows}</table>{best_videos}
<h2 id="conditions">무엇을 비교했는가?</h2><p>데이터 {audit['files']}파일, {audit['episodes']:,}에피소드, 유효 제어 {audit['valid_controls']:,}스텝 ({audit['falls']}에피소드 낙상). 구조·길이 32·60,000스텝·배치 32·학습률 0.0002·Mode 1/2/4 (20/40/40)·diffusion 5단계·minmax·DDIM·seed 0 은 기준 run 과 동일. 바뀐 것은 위 단변수 하나. 모든 설정과 데이터 해시는 <a href="conditions.json">conditions.json</a>.</p>
<h2 id="legacy">기존과 동일한 형상·행동 관계 평가 (Mode 2 A 고정)</h2><p>네 모델 모두 동일한 기존 시험 형상 8개, 516쌍, 8개 seed, v8 오차 단위, 시작 프레임 30.</p>
<table>{heading}{table_rows}</table><img src="evaluation/comparison.png" alt="형상 변화 일치도 및 검증 움직임 오차">
<h2 id="native">이 run 의 검증 분할 시작점</h2><p>검증 에피소드 {native['pd']['start30']['episodes']}개 (시작 프레임 30). 일치도는 시간에 따른 움직임 상관, 추가 오차는 양수일수록 올바른 조건 예측이 더 정확했다는 뜻.</p><table>{heading}{native_rows}</table>
<h2 id="mjx">같은 Task 조건으로 실제 MJX 실행</h2><p>고정 16형상 × 후보 4개와 공동 생성 후보 16개, zigzag·circle 4초, scale 10, kp {conditions['execution_kp']}. 부호 일치 = 후보별 예측 dx·MJX dx 부호 일치 수 / 유효 후보, 쌍 = 형상 평균 쌍 변화 부호 일치율 (기준 2).</p>
<table><tr><th>모델·Task</th><th>형상 효과 p</th><th>보상 순위 상관</th><th>고정 E 부호 일치</th><th>고정 E 낙상/64</th><th>2초 경로 차 cm</th><th>2초 생존</th><th>4초 생존</th><th>4초 진행 중앙값</th><th>Gate</th></tr>{task_rows}</table>
<h2>학습과 자원</h2><p>60,000스텝, 최종 검증 loss {progress[-1]['val_loss']:.5f}, 학습 {progress[-1]['min']:.1f}분. 평가: RAM 여유 최소 {limits['min_ram_available_gib']:.1f}GiB, GPU 사용 최대 {limits['max_gpu_used_mib']/1024:.1f}GiB.</p><img src="evaluation/training.png" alt="학습 loss 와 R²_Δ 모니터">
<h2 id="videos">실제 MJX 영상</h2>{videos}
<h2>원자료</h2><p><a href="evaluation/comparison.json">종합 수치</a> · <a href="gate.json">사전 기준</a> · <a href="evaluation/delta_r2.json">Δ 설명 비율</a> · <a href="evaluation/given_a_held_mode1.json">Mode 1 측정</a> · <a href="evaluation/given_a_held.json">기존 시험 형상</a> · <a href="evaluation/given_a_seen.json">기존 학습 형상</a> · <a href="evaluation/given_a_native.json">검증 분할</a> · <a href="eval_status.json">단계별 실행 상태</a></p></html>'''
    embed = lambda m: 'src="data:{};base64,{}"'.format(
        "video/mp4" if m[1].endswith(".mp4") else "image/png", base64.b64encode((RUN / m[1]).read_bytes()).decode())
    (RUN / "report.html").write_text(re.sub(r'src="(evaluation/[^"]+)"', embed, text))
    print(conclusion, flush=True)
    print(json.dumps({k: gate[k] for k in checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
