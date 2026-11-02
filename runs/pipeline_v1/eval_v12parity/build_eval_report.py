"""run_eval.py 결과로 v12 동일 비교표 + remind 8-43 사전 기준 판정 (gate.json) + report.html 을 만든다. 물리 재실행 없음.

    python build_eval_report.py <run_dir>
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
COLORS = dict(v8="#64748b", v9="#b7791f", v12="#c2410c", pd="#087f8c")
DEFAULT_GATE = dict(wrong_action_velocity_extra_error_min=0.01, E_swap_forward_change_sign_agreement_min=0.7,
                    joint_survival_at_least_v12_zigzag=7, joint_survival_at_least_v12_circle=10)


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


def task_result(task):
    folder = OUT / "check_w1"
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


def main():
    held, seen, native, mode1 = [read(OUT / name) for name in ("given_a_held.json", "given_a_seen.json", "given_a_native.json", "given_a_held_mode1.json")]
    assert len({held[name]["start30"]["pairs"] for name in COLORS}) == 1
    assert held["pd"]["start30"]["pairs"] == 516
    conditions = read(RUN / "conditions.json")
    audit = read(RUN / "data_audit.json") if (RUN / "data_audit.json").exists() else read(V12 / "data_audit.json")
    progress = [json.loads(line) for line in (RUN / "tt_joint/progress.jsonl").read_text().splitlines()]
    assert progress[-1]["step"] == 60000
    training = read(RUN / "tt_joint/config.json")
    baseline = read(ROOT / conditions["training_reference"])
    for key in ("horizon", "steps", "batch", "lr", "d_model", "layers", "diff_steps", "seed", "norm", "clip", "sampler", "film", "modes", "mode_weights"):
        assert training[key] == baseline[key], key
    previous = read(V9 / "evaluation_local/comparison.json")
    tasks = {name: previous["tasks"][name] for name in ("v8", "v9")}
    v12_comparison = read(V12 / "evaluation/comparison.json")
    v12_mode1 = read(V12 / "evaluation/given_a_held_mode1.json")
    tasks["v12"] = v12_comparison["tasks"]["pd"]
    # v12 고정 E 부호 일치는 v12 폴더의 summary.json 에서 같은 함수로 다시 계산 (comparison.json 에 없음)
    for task in ("zigzag", "circle"):
        rows = read(V12 / f"evaluation/check_w1/fixedE_{task}_4s/summary.json")["runs"]["scale_10"]
        tasks["v12"][task].update(sign_agreement(rows), fixed_fell=sum(row["mjx_fell_step"] != -1 for row in rows))
    tasks["pd"] = {task: task_result(task) for task in ("zigzag", "circle")}
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
    # Mode 1 조건 직접 측정 (8-43 기준 1): 틀린 A 추가 오차 = wrongA_moving - pin_moving, start30
    m1 = {}
    for name, source in (("v12", v12_mode1["pd"]), ("pd", mode1["pd"])):
        m1[name] = {g: dict(pin=source["start30"][g]["pin_moving"],
                            wrongA_extra=source["start30"][g]["wrongA_moving"] - source["start30"][g]["pin_moving"],
                            wrongE_extra=source["start30"][g]["wrongE_moving"] - source["start30"][g]["pin_moving"])
                    for g in LABELS}
    assert abs(mode1["v12"]["start30"]["linvel"]["pin_moving"] - v12_mode1["pd"]["start30"]["linvel"]["pin_moving"]) < 1e-4, "v12 Mode-1 재측정이 8-38 기록과 다름"
    gate_cfg = conditions.get("evaluation_gate", DEFAULT_GATE)
    gate = dict(
        criteria=gate_cfg,
        wrong_action_velocity_extra_error=dict(v12=m1["v12"]["linvel"]["wrongA_extra"], run=m1["pd"]["linvel"]["wrongA_extra"],
                                               pass_=m1["pd"]["linvel"]["wrongA_extra"] >= gate_cfg["wrong_action_velocity_extra_error_min"]),
        e_swap_sign=dict(v12={t: tasks["v12"][t]["pair_sign_rate"] for t in ("zigzag", "circle")},
                         run={t: tasks["pd"][t]["pair_sign_rate"] for t in ("zigzag", "circle")}),
        joint_survival=dict(v12={t: tasks["v12"][t]["joint_survived"] for t in ("zigzag", "circle")},
                            run={t: tasks["pd"][t]["joint_survived"] for t in ("zigzag", "circle")}),
    )
    gate["e_swap_sign"]["pass_"] = all(r is not None and r >= gate_cfg["E_swap_forward_change_sign_agreement_min"] for r in gate["e_swap_sign"]["run"].values())
    gate["joint_survival"]["pass_"] = (gate["joint_survival"]["run"]["zigzag"] >= gate_cfg["joint_survival_at_least_v12_zigzag"]
                                       and gate["joint_survival"]["run"]["circle"] >= gate_cfg["joint_survival_at_least_v12_circle"])
    gate["all_pass"] = gate["wrong_action_velocity_extra_error"]["pass_"] and gate["e_swap_sign"]["pass_"] and gate["joint_survival"]["pass_"]
    (RUN / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False))
    resources = [json.loads(line) for line in (RUN / "eval_resources.jsonl").read_text().splitlines()]
    limits = dict(min_ram_available_gib=min(r["ram_available_gib"] for r in resources),
                  max_gpu_used_mib=max(float(r["gpu"].split(",")[0]) for r in resources if r["gpu"]))
    result = dict(main_sensitivity_pass=main_pass, metrics=metrics, mode1=m1, gate=gate, tasks=tasks, resources=limits,
                  paired_cases=516, native_conditions=native["_conditions"], dataset=audit,
                  native_cases_start30=native["pd"]["start30"]["episodes"],
                  final_validation_loss=progress[-1]["val_loss"], final_validation_raw_loss=progress[-1].get("val_raw_loss"),
                  training_minutes=progress[-1]["min"])
    (OUT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

    features = list(LABELS)
    x = np.arange(len(features))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for i, (name, color) in enumerate(COLORS.items()):
        axes[0].bar(x + (i - 1.5) * .2, [metrics[g][name]["legacy_agreement"] for g in features], .2, label=name, color=color)
        axes[1].bar(x + (i - 1.5) * .2, [metrics[g][name]["native_moving_rmse"] for g in features], .2, label=name, color=color)
    for ax in axes:
        ax.set_xticks(x, features, rotation=25)
        ax.legend()
        ax.axhline(0, color="#999", linewidth=.7)
    axes[0].set_title("Frozen legacy cases: morphology-change agreement")
    axes[1].set_title("Held-out PD cases: moving RMSE (lower is better)")
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    key = "val_raw_loss" if "val_raw_loss" in progress[-1] else "val_loss"
    ax.plot([r["step"] for r in progress], [r.get("raw_loss", r["loss"]) for r in progress], alpha=.45, label="train minibatch")
    ax.plot([r["step"] for r in progress], [r[key] for r in progress], label=f"PD validation ({key})")
    ax.set(xlabel="Training steps", ylabel="Loss", yscale="log")
    ax.legend()
    fig.savefig(OUT / "training.png", dpi=160)
    plt.close(fig)

    run_label = RUN.name
    change = conditions.get("single_variable_vs_v12", conditions.get("loss_semantics", ""))
    if isinstance(change, dict):
        change = change.get("change", json.dumps(change, ensure_ascii=False))
    ok = lambda b: "통과" if b else "미달"
    gate_rows = (
        f"<tr><td>1. 틀린 A 추가 오차 (속도, Mode 1 조건)</td><td>≥ {gate_cfg['wrong_action_velocity_extra_error_min']}</td><td>{fmt(gate['wrong_action_velocity_extra_error']['v12'], 4)}</td><td>{fmt(gate['wrong_action_velocity_extra_error']['run'], 4)}</td><td>{ok(gate['wrong_action_velocity_extra_error']['pass_'])}</td></tr>"
        f"<tr><td>2. E 교체 시 예측 전진 변화 부호 일치 (형상 평균 쌍, zigzag / circle)</td><td>≥ {gate_cfg['E_swap_forward_change_sign_agreement_min']}</td><td>{fmt(gate['e_swap_sign']['v12']['zigzag'], 2)} / {fmt(gate['e_swap_sign']['v12']['circle'], 2)}</td><td>{fmt(gate['e_swap_sign']['run']['zigzag'], 2)} / {fmt(gate['e_swap_sign']['run']['circle'], 2)}</td><td>{ok(gate['e_swap_sign']['pass_'])}</td></tr>"
        f"<tr><td>3. 공동 생성 생존 (zigzag / circle, /16)</td><td>≥ {gate_cfg['joint_survival_at_least_v12_zigzag']} / {gate_cfg['joint_survival_at_least_v12_circle']}</td><td>{gate['joint_survival']['v12']['zigzag']} / {gate['joint_survival']['v12']['circle']}</td><td>{gate['joint_survival']['run']['zigzag']} / {gate['joint_survival']['run']['circle']}</td><td>{ok(gate['joint_survival']['pass_'])}</td></tr>")
    mode1_rows = "".join(
        f"<tr><td>{LABELS[g]}</td><td>{fmt(m1['v12'][g]['pin'])}</td><td>{fmt(m1['v12'][g]['wrongA_extra'], 4)}</td><td>{fmt(m1['v12'][g]['wrongE_extra'], 4)}</td>"
        f"<td>{fmt(m1['pd'][g]['pin'])}</td><td>{fmt(m1['pd'][g]['wrongA_extra'], 4)}</td><td>{fmt(m1['pd'][g]['wrongE_extra'], 4)}</td></tr>" for g in LABELS)
    table_rows = ""
    native_rows = ""
    for g, label in LABELS.items():
        for name in COLORS:
            r = metrics[g][name]
            table_rows += f"<tr><td>{label}</td><td>{name}</td><td>{fmt(r['legacy_agreement'])}</td><td>{fmt(r['legacy_moving_rmse'])}</td><td>{fmt(r['legacy_wrongE_penalty'])}</td><td>{fmt(r['legacy_wrongA_penalty'])}</td></tr>"
            native_rows += f"<tr><td>{label}</td><td>{name}</td><td>{fmt(r['native_track'])}</td><td>{fmt(r['native_moving_rmse'])}</td><td>{fmt(r['native_wrongE_penalty'])}</td><td>{fmt(r['native_wrongA_penalty'])}</td></tr>"
    task_rows = ""
    for name in COLORS:
        for task, r in tasks[name].items():
            p_label = "&lt;0.0001" if r["pred_effect_p"] == 0 else fmt(r["pred_effect_p"], 4)
            unit = "m" if task == "zigzag" else "바퀴"
            sign = f"{r['candidate_sign_agree']}/{r['candidate_valid']} · 쌍 {fmt(r['pair_sign_rate'], 2)}" if "pair_sign_rate" in r else "미측정"
            fell = r.get("fixed_fell", "미측정")
            task_rows += f"<tr><td>{name} · {task}</td><td>{p_label}</td><td>{fmt(r['rank_correlation'])}</td><td>{sign}</td><td>{fell}</td><td>{fmt(r['smoke_path_error_m'] * 100 if r['smoke_path_error_m'] is not None else None,1)}</td><td>{r['smoke_survived']}/{r['smoke_total']}</td><td>{r['joint_survived']}/{r['joint_total']}</td><td>{fmt(r['joint_progress_median'])} {unit}</td><td>{'통과' if r['gate'] else '미달'}</td></tr>"
    conclusion = f"{run_label}: remind 8-43 사전 기준 3개 중 " + str(sum(gate[k]["pass_"] for k in ("wrong_action_velocity_extra_error", "e_swap_sign", "joint_survival"))) + "개 통과 → " + ("통과" if gate["all_pass"] else "미달") + ". "
    conclusion += "기존 v8 대비 E 효과 일치도 +0.1 기준은 " + ("통과." if main_pass else "미달.")
    videos = ""
    best_rows = best_videos = ""
    for task in ("zigzag", "circle"):
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
            if path == best_path:
                continue
            videos += f'<details><summary>{html.escape(path.stem)}</summary><video controls src="{path.relative_to(RUN)}"></video></details>'
    heading = "<tr><th>항목</th><th>모델</th><th>일치도</th><th>움직임 RMSE</th><th>틀린 E 추가 오차</th><th>틀린 A 추가 오차</th></tr>"
    today = datetime.date.today().isoformat()
    text = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(run_label)} · v12 동일 평가</title><style>body{{font:16px/1.8 system-ui;max-width:1160px;margin:32px auto;padding:20px;color:#243746}}nav a{{margin-right:20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #d4dfe4}}th{{background:#eef4f6}}img,video{{width:100%}}h2{{margin-top:38px}}details{{margin:8px 0}}.lead{{background:#edf7f7;padding:20px;border-left:5px solid #087f8c}}code{{overflow-wrap:anywhere}}</style>
<h1>{html.escape(run_label)}: v12 와 같은 평가, remind 8-43 사전 기준</h1><p>{today} · 단변수 변경: {html.escape(str(change))}</p>
<nav><a href="#result">판정</a><a href="#gate">사전 기준</a><a href="#mode1">Mode 1 측정</a><a href="#best">최고 후보</a><a href="#legacy">기존 평가</a><a href="#native">PD 평가</a><a href="#mjx">MJX 실행</a><a href="#videos">영상</a></nav>
<h2 id="result">판정</h2><p class="lead"><strong>{conclusion}</strong></p>
<h2 id="gate">remind 8-43 사전 기준 (v12 Mode 1 평가와 동일 조건)</h2>
<table><tr><th>기준</th><th>통과선</th><th>v12</th><th>{html.escape(run_label)}</th><th>판정</th></tr>{gate_rows}</table>
<p>기준 1 은 기존 시험 형상 8개·516쌍·시작 프레임 30 에서 A 를 Mode 1 조건(알려진 입력)으로 넣고, 다른 모션의 A 로 바꿨을 때 늘어나는 움직임 RMSE (v8 오차 단위) 다. 기준 2 는 고정 E 64 후보(16 형상 × 4)의 형상 평균 예측 전진과 실제 MJX 전진을 형상 쌍마다 비교해 변화 부호가 같은 비율이다 (낙상 형상은 쌍에서 제외). 기준 3 은 공동 생성 16 후보 중 4초 생존 수다. 원자료 <a href="gate.json">gate.json</a>.</p>
<h2 id="mode1">Mode 1 조건 직접 측정 (given_a_held_mode1.json, start 30)</h2>
<table><tr><th>항목</th><th>v12 올바른 A 오차</th><th>v12 틀린 A 추가</th><th>v12 틀린 E 추가</th><th>{html.escape(run_label)} 올바른 A 오차</th><th>틀린 A 추가</th><th>틀린 E 추가</th></tr>{mode1_rows}</table>
<h2 id="best">가장 잘 나온 후보</h2><p>Task별 4초 공동 생성 16개 중 낙상 없이 실제 MJX 보상이 가장 높은 후보다. 실행 결과를 보고 사후에 고른 것이므로 Gate 통과나 모델의 후보 선택 능력을 뜻하지 않는다.</p>
<table><tr><th>Task</th><th>후보</th><th>실제 보상</th><th>실제 진행</th><th>TT 예측 보상 (생존 후보 중 순위)</th><th>TT 예측 1위 후보의 실제 보상</th></tr>{best_rows}</table>{best_videos}
<h2 id="conditions">무엇을 비교했는가?</h2><p>데이터는 v12 와 같은 PD kp200 {audit['files']}형상, {audit['episodes']:,}에피소드, 유효 제어 {audit['valid_controls']:,}스텝 ({audit['falls']}에피소드 낙상, 해시 일치). 구조·길이 32·60,000스텝·배치 32·학습률 0.0002·Mode 1/2/4 (20/40/40)·diffusion 5단계·minmax·DDIM·seed 0 동일. 바뀐 것은 위 단변수 하나다. 모든 설정은 <a href="conditions.json">conditions.json</a>.</p>
<h2 id="legacy">기존과 동일한 형상·행동 관계 평가 (Mode 2 A 고정)</h2><p>네 모델 모두 동일한 기존 시험 형상 8개, 516쌍, 8개 seed, v8 오차 단위, 시작 프레임 30. v8 대비 높이·기울기·속도 일치도 +0.1 이 기존 기준이다.</p>
<table>{heading}{table_rows}</table><img src="evaluation/comparison.png" alt="기존 평가의 형상 변화 일치도 및 PD 검증 움직임 오차">
<h2 id="native">PD 데이터의 미학습 시작점</h2><p>동일한 PD 검증 에피소드 {native['pd']['start30']['episodes']}개 (시작 프레임 30). 일치도는 시간에 따른 움직임 상관, 추가 오차는 양수일수록 올바른 조건 예측이 더 정확했다는 뜻.</p><table>{heading}{native_rows}</table>
<h2 id="mjx">같은 Task 조건으로 실제 MJX 실행</h2><p>고정 16형상 × 후보 4개와 공동 생성 후보 16개, zigzag·circle 4초, scale 10, kp {conditions['execution_kp']}. 부호 일치 = 후보별 예측 dx·MJX dx 부호 일치 수 / 유효 후보, 쌍 = 형상 평균 쌍 변화 부호 일치율 (기준 2).</p>
<table><tr><th>모델·Task</th><th>형상 효과 p</th><th>보상 순위 상관</th><th>고정 E 부호 일치</th><th>고정 E 낙상/64</th><th>2초 경로 차 cm</th><th>2초 생존</th><th>4초 생존</th><th>4초 진행 중앙값</th><th>Gate</th></tr>{task_rows}</table>
<h2>학습과 자원</h2><p>60,000스텝, 최종 검증 loss {progress[-1]['val_loss']:.5f}{'' if result['final_validation_raw_loss'] is None else f" (raw {result['final_validation_raw_loss']:.5f})"}, 학습 {progress[-1]['min']:.1f}분. 평가: RAM 여유 최소 {limits['min_ram_available_gib']:.1f}GiB, GPU 사용 최대 {limits['max_gpu_used_mib']/1024:.1f}GiB.</p><img src="evaluation/training.png" alt="학습 및 검증 loss">
<h2 id="videos">실제 MJX 영상</h2>{videos}
<h2>원자료</h2><p><a href="evaluation/comparison.json">종합 수치</a> · <a href="gate.json">사전 기준</a> · <a href="evaluation/given_a_held_mode1.json">Mode 1 측정</a> · <a href="evaluation/given_a_held.json">기존 시험 형상</a> · <a href="evaluation/given_a_seen.json">기존 학습 형상</a> · <a href="evaluation/given_a_native.json">PD 검증</a> · <a href="eval_status.json">단계별 실행 상태</a></p></html>'''
    embed = lambda m: 'src="data:{};base64,{}"'.format(
        "video/mp4" if m[1].endswith(".mp4") else "image/png", base64.b64encode((RUN / m[1]).read_bytes()).decode())
    (RUN / "report.html").write_text(re.sub(r'src="(evaluation/[^"]+)"', embed, text))
    print(conclusion, flush=True)
    print(json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
