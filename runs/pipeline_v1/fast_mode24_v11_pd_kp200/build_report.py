"""Write the PD/PPO comparison from completed measurements, without rerunning physics."""
import base64
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
OUT = RUN / "evaluation"
V9 = RUN.parent / "fast_mode24_v9_e_coverage"
LABELS = dict(height="몸통 높이", tilt="몸통 기울기", linvel="이동 속도", yaw="방향", pos="위치", joint="관절각")
COLORS = dict(v8="#64748b", v9="#b7791f", pd="#087f8c")


def read(path):
    return json.loads(path.read_text())


def eta(values, groups):
    denominator = ((values - values.mean()) ** 2).sum()
    return float(sum((groups == k).sum() * (values[groups == k].mean() - values.mean()) ** 2
                     for k in np.unique(groups)) / denominator) if denominator else 0.


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
                joint_survived=sum(row["mjx_fell_step"] == -1 for row in joint), joint_total=len(joint),
                joint_reward_median=float(np.median(rewards)) if rewards else None,
                joint_progress_median=float(np.median(progress)) if progress else None,
                joint_progress_unit="m" if task == "zigzag" else "laps",
                gate=p < .05 and rank is not None and rank > .6 and path_error is not None and path_error < .15,
                predicted_design_reward=pred_mean.tolist(),
                actual_design_reward=[float(x) if np.isfinite(x) else None for x in actual_mean])


def fmt(value, digits=3):
    return "미정" if value is None else f"{value:.{digits}f}"


def main():
    held, seen, native = [read(OUT / name) for name in ("given_a_held.json", "given_a_seen.json", "given_a_native.json")]
    assert len({held[name]["start30"]["pairs"] for name in COLORS}) == 1
    assert held["pd"]["start30"]["pairs"] == 516
    conditions, audit = read(RUN / "conditions.json"), read(RUN / "data_audit.json")
    progress = [json.loads(line) for line in (RUN / "tt_joint/progress.jsonl").read_text().splitlines()]
    assert progress[-1]["step"] == 60000
    training = read(RUN / "tt_joint/config.json")
    baseline = read(Path(conditions["training_reference"]))
    for key in ("horizon", "steps", "batch", "lr", "d_model", "layers", "diff_steps", "seed", "norm", "clip", "sampler", "film", "modes"):
        assert training[key] == baseline[key], key
    previous = read(V9 / "evaluation_local/comparison.json")
    tasks = {name: previous["tasks"][name] for name in ("v8", "v9")}
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
    resources = [json.loads(line) for line in (RUN / "resources.jsonl").read_text().splitlines()]
    limits = dict(min_ram_available_gib=min(r["ram_available_gib"] for r in resources),
                  max_gpu_used_mib=max(float(r["gpu"].split(",")[0]) for r in resources if r["gpu"]))
    result = dict(main_sensitivity_pass=main_pass, metrics=metrics, tasks=tasks, resources=limits,
                  paired_cases=516, native_conditions=native["_conditions"], dataset=audit,
                  native_cases_start30=native["pd"]["start30"]["episodes"],
                  final_validation_loss=progress[-1]["val_loss"], training_minutes=progress[-1]["min"])
    (OUT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

    features = list(LABELS)
    x = np.arange(len(features))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for i, (name, color) in enumerate(COLORS.items()):
        axes[0].bar(x + (i - 1) * .25, [metrics[g][name]["legacy_agreement"] for g in features], .25, label=name, color=color)
        axes[1].bar(x + (i - 1) * .25, [metrics[g][name]["native_moving_rmse"] for g in features], .25, label=name, color=color)
    for ax in axes:
        ax.set_xticks(x, features, rotation=25)
        ax.legend()
        ax.axhline(0, color="#999", linewidth=.7)
    axes[0].set_title("Frozen legacy cases: morphology-change agreement")
    axes[1].set_title("Held-out PD cases: moving RMSE (lower is better)")
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    ax.plot([r["step"] for r in progress], [r["loss"] for r in progress], alpha=.45, label="train minibatch")
    ax.plot([r["step"] for r in progress], [r["val_loss"] for r in progress], label="PD validation")
    ax.set(xlabel="Training steps", ylabel="Loss", yscale="log")
    ax.legend()
    fig.savefig(OUT / "training.png", dpi=160)
    plt.close(fig)

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
            task_rows += f"<tr><td>{name} · {task}</td><td>{p_label}</td><td>{fmt(r['rank_correlation'])}</td><td>{fmt(r['smoke_path_error_m'] * 100 if r['smoke_path_error_m'] is not None else None,1)}</td><td>{r['smoke_survived']}/{r['smoke_total']}</td><td>{r['joint_survived']}/{r['joint_total']}</td><td>{fmt(r['joint_progress_median'])} {unit}</td><td>{'통과' if r['gate'] else '미달'}</td></tr>"
    native_better = [LABELS[g] for g in ("height", "tilt", "linvel") if metrics[g]["pd"]["native_moving_rmse"] < metrics[g]["v8"]["native_moving_rmse"]]
    conclusion = "기존 동일 평가에서 PD 데이터 모델은 형상 효과 개선 기준을 " + ("통과했다." if main_pass else "통과하지 못했다.")
    conclusion += " PD 미학습 시작점에서 v8보다 움직임 오차가 낮은 핵심 항목: " + (", ".join(native_better) if native_better else "없음") + "."
    videos = ""
    best_rows = best_videos = ""
    for task in ("zigzag", "circle"):
        files = sorted((OUT / f"check_w1/joint_{task}_4s/videos").glob("*.mp4"))
        assert files, task  # 낙상·무효 후보는 영상이 없을 수 있다 (v11: 15/16)
        joint = read(OUT / f"check_w1/joint_{task}_4s/summary.json")["runs"]["scale_10"]
        survived = [i for i, row in enumerate(joint) if row["mjx_fell_step"] == -1 and row["mjx_total"] is not None]
        best = max(survived, key=lambda i: joint[i]["mjx_total"])
        tt_pick = max(survived, key=lambda i: joint[i]["pred_total"])
        rank = 1 + sum(joint[i]["pred_total"] > joint[best]["pred_total"] for i in survived)
        unit = "m" if task == "zigzag" else "바퀴"
        best_path = OUT / f"check_w1/joint_{task}_4s/videos/scale_10_cand{best}.mp4"
        best_rows += (f"<tr><td>{task}</td><td>cand{best}</td><td>{joint[best]['mjx_total']:.3f}</td><td>{joint[best]['mjx_progress']:.3f} {unit}</td>"
                      f"<td>{joint[best]['pred_total']:.3f} ({rank}위/{len(survived)})</td><td>cand{tt_pick}: {joint[tt_pick]['mjx_total']:.3f}</td></tr>")
        best_videos += f'<h3>{task} · cand{best}</h3><video controls src="{best_path.relative_to(RUN)}"></video>'
        videos += f"<h3>{task}: 나머지 공동 생성 15개</h3>"
        for path in files:
            if path == best_path:
                continue
            videos += f'<details><summary>{html.escape(path.stem)}</summary><video controls src="{path.relative_to(RUN)}"></video></details>'
    heading = "<tr><th>항목</th><th>모델</th><th>일치도</th><th>움직임 RMSE</th><th>틀린 E 추가 오차</th><th>틀린 A 추가 오차</th></tr>"
    text = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PD rollout · PPO 데이터 비교</title><style>body{{font:16px/1.8 system-ui;max-width:1160px;margin:32px auto;padding:20px;color:#243746}}nav a{{margin-right:20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #d4dfe4}}th{{background:#eef4f6}}img,video{{width:100%}}h2{{margin-top:38px}}details{{margin:8px 0}}.lead{{background:#edf7f7;padding:20px;border-left:5px solid #087f8c}}code{{overflow-wrap:anywhere}}</style>
<h1>PD로 수집한 Dynamics를 TT가 더 잘 학습하는가?</h1><p>2026년 11월 1일 · 기존 v8/v9와 같은 학습·평가 조건</p>
<nav><a href="#result">판정</a><a href="#best">최고 후보</a><a href="#conditions">실험 조건</a><a href="#legacy">기존 평가</a><a href="#native">PD 평가</a><a href="#mjx">MJX 실행</a><a href="#videos">영상</a></nav>
<h2 id="result">판정</h2><p class="lead"><strong>{conclusion}</strong></p>
<h2 id="best">가장 잘 나온 후보</h2><p>Task별 4초 공동 생성 16개 중 낙상 없이 실제 MJX 보상이 가장 높은 후보다. 실행 결과를 보고 사후에 고른 것이므로 Gate 통과나 모델의 후보 선택 능력을 뜻하지 않는다.</p>
<table><tr><th>Task</th><th>후보</th><th>실제 보상</th><th>실제 진행</th><th>TT 예측 보상 (생존 후보 중 순위)</th><th>TT 예측 1위 후보의 실제 보상</th></tr>{best_rows}</table>{best_videos}
<h2 id="conditions">무엇을 비교했는가?</h2><p>PD 수집은 {audit['files']}형상, {audit['episodes']:,}에피소드, 유효 제어 {audit['valid_controls']:,}스텝이며 {audit['falls']}에피소드에서 낙상이 있었다. 실제 MJX의 관절·몸통 상태와 PD 목표를 저장했고, 낙상 뒤 구간은 학습에서 제외했다. PPO rollout도 실제 물리 데이터다. 이번에 바뀐 것은 목표 관절각을 만드는 방식과 데이터 분포다.</p>
<p>주 기준 v8은 같은 16형상이고, v9는 48학습 형상이므로 보조 기준이다. PD 모델도 FiLM, 길이 32, 60,000스텝, 배치 32, 학습률 0.0002, Mode 2·4, diffusion 5단계, minmax, DDIM, seed 0을 그대로 사용했다. PD 학습에는 PPO나 E-swap 데이터를 넣지 않았다. 데이터 양은 v8의 13,056에피소드와 달라 수집 제어기만의 인과 효과로 단정할 수 없다.</p>
<p>PD train/validation은 모션 이름·로컬 시작 프레임 그룹으로 형상 전체를 함께 분할했다. 검증 {native['_conditions']['validation_source_groups']}그룹·{native['_conditions']['validation_episodes']}에피소드이며, 같은 시작점의 형상 복제본이 학습·검증에 나뉘지 않았다. 같은 모션의 다른 시작점은 학습에 있으므로 미학습 모션 평가로 해석하지 않는다. 형상 파라미터의 독립 방향 수는 {audit['morphology_parameter_rank']}로 기존 16형상과 같다. 모든 설정과 데이터 해시는 <a href="conditions.json">conditions.json</a>, 수집 점검은 <a href="data_audit.json">data_audit.json</a>에 있다.</p>
<h2 id="legacy">기존과 동일한 형상·행동 관계 평가</h2><p>세 모델 모두 동일한 기존 시험 형상 8개, 516쌍, 8개 seed, v8 오차 단위, 시작 프레임 30을 사용했다. 형상쌍의 행동이 같다는 검사를 통과했다. 핵심 통과 기준은 높이·기울기·속도의 형상 변화 일치도가 각각 v8보다 0.1 이상 개선되는 것이다. 같은 원본 행동에서 반복 뽑은 형상쌍이므로 516개가 독립 표본은 아니다. 학습 형상 평가는 별도 원자료로 제공한다.</p>
<table>{heading}{table_rows}</table><img src="evaluation/comparison.png" alt="기존 평가의 형상 변화 일치도 및 PD 검증 움직임 오차">
<h2 id="native">PD 데이터의 미학습 시작점에서도 맞추는가?</h2><p>동일한 PD 검증 에피소드를 세 모델에 넣고, 올바른 E·A, 틀린 E, 다른 모션의 A를 비교했다. 주 평가인 시작 프레임 30에서 길이가 충분한 {native['pd']['start30']['episodes']}개 사례를 사용했다. 일치도는 시간에 따른 움직임 상관이다. 추가 오차는 양수일수록 올바른 조건을 쓴 예측이 더 정확했다는 뜻이다. 형상마다 리타겟팅 행동이 다르고 S0에도 형상 정보가 들어 있어 이 표는 순수한 동일-A 형상 인과 실험으로 해석하지 않는다.</p><table>{heading}{native_rows}</table>
<p>두 평가 모두 기존 Mode-2 샘플링 중 A를 고정하는 진단을 유지했다. Mode 1을 학습한 조건부 dynamics 모델의 평가로 바꿔 해석하지 않는다. PD 자체 검증이 개선돼도 기존 PPO 행동·미학습 형상 분포로의 일반화가 개선됐다는 뜻은 아니다.</p>
<h2 id="mjx">같은 Task 조건으로 실제 MJX 실행</h2><p>고정 16형상 × 후보 4개와 공동 생성 후보 16개, zigzag·circle 4초, scale 10을 유지했다. 추가로 scale 0·2초·16후보 경로 검사를 했다. Gate는 기존과 같이 예측 보상의 형상 효과 p&lt;0.05, 실제 형상 평균 보상과 순위 상관&gt;0.6, 2초 경로 차 중앙값&lt;15cm다. 생존 수를 함께 표시하며 최종 10초 성공 판정은 하지 않았다. v8/v9 물리 결과는 코드·체크포인트 해시가 같은 기존 결과를 보존해 비교했다.</p>
<table><tr><th>모델·Task</th><th>형상 효과 p</th><th>보상 순위 상관</th><th>2초 경로 차 cm</th><th>2초 생존</th><th>4초 생존</th><th>4초 진행 중앙값</th><th>Gate</th></tr>{task_rows}</table>
<h2>학습과 자원</h2><p>60,000스텝 완료, 최종 PD 검증 loss {progress[-1]['val_loss']:.5f}, 학습 {progress[-1]['min']:.1f}분. 정규화와 데이터가 달라 loss 숫자만으로 기존 모델과 성능 우열을 판정하지 않는다. 작업은 CPU 2코어, PyTorch GPU 할당 상한 60%(평가 25%), JAX 풀 40%로 순차 실행했다. RAM 여유 최소 {limits['min_ram_available_gib']:.1f}GiB, 전체 GPU 사용 최대 {limits['max_gpu_used_mib']/1024:.1f}GiB였다. 임시파일·캐시는 여유 공간이 큰 /home 아래에 둔다.</p><img src="evaluation/training.png" alt="PD 학습 및 검증 loss">
<h2 id="videos">실제 MJX 영상</h2>{videos}
<h2>원자료</h2><p><a href="evaluation/comparison.json">종합 수치</a> · <a href="evaluation/given_a_held.json">기존 시험 형상</a> · <a href="evaluation/given_a_seen.json">기존 학습 형상</a> · <a href="evaluation/given_a_native.json">PD 검증</a> · <a href="status.json">단계별 실행 상태</a></p></html>'''
    embed = lambda m: 'src="data:{};base64,{}"'.format(
        "video/mp4" if m[1].endswith(".mp4") else "image/png", base64.b64encode((RUN / m[1]).read_bytes()).decode())
    (RUN / "report.html").write_text(re.sub(r'src="(evaluation/[^"]+)"', embed, text))
    print(conclusion, flush=True)
    print(json.dumps(dict(main_sensitivity_pass=main_pass, tasks=tasks["pd"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
