"""Record the actual retarget_tt height path without running IK or PPO."""
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import pd_replay


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "runs/TT_retarget/jump_height_stages"
    # Fail loudly if diagnostic mode reaches the excluded solver or writes reference artifacts.
    with patch.object(pd_replay, "foot_lock", side_effect=AssertionError("IK excluded")), \
         patch.object(pd_replay, "inverse_check", side_effect=AssertionError("dynamics excluded")), \
         patch.object(pd_replay.np, "savez", side_effect=AssertionError("reference write excluded")):
        traces = pd_replay.retarget_tt(str(root / "artifacts/pd_replay"), None,
                                      only=["jumpInPlace"], height_trace_only=True)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    rows, metrics = [], {}
    for ((choice, name), v), ax in zip(traces.items(), axes.flat):
        a, c, i, f = (v[k] for k in ("a1_z", "contact_z", "interpolated_z", "smoothed_z"))
        dt, scale = v["dt"], v["scale"]
        t = np.arange(len(a))*dt
        valid = np.isfinite(c)
        event = int(np.flatnonzero(~v["contact"].any(axis=1))[0])
        x = t-t[event]
        # Independent equivalence checks for the unchanged interpolation/filter formulas.
        legacy = scale*a + np.interp(np.arange(len(a)), np.flatnonzero(valid), (c-scale*a)[valid])
        np.testing.assert_array_equal(i, legacy)
        np.testing.assert_array_equal(f, savgol_filter(legacy, pd_replay.BASE_SG_WIN, 2))
        np.testing.assert_allclose(i[valid], c[valid], atol=1e-14)
        series = {"A1": a, "A1_scaled": scale*a, "contact": c, "interpolated": i, "smoothed": f}
        colors = ["darkorange", "gray", "firebrick", "seagreen", "royalblue"]
        result = {}
        columns = [t, x, v["contact"].sum(axis=1)]
        for (stage, z), color in zip(series.items(), colors):
            # Contact-only height is undefined at takeoff; use the pre-filter interpolated event value.
            baseline = i[event] if stage == "contact" else z[event]
            h = z-baseline
            p = int(np.nanargmax(z))
            post = np.arange(len(z)) >= event
            result[stage] = dict(peak_time_s=float(t[p]), peak_from_takeoff_s=float(x[p]),
                max_relative_height_mm=float(np.nanmax(h)*1000),
                max_post_event_height_mm=float(np.nanmax(h[post])*1000),
                rise_from_start_mm=float((np.nanmax(z)-z[0])*1000),
                baseline_m=float(baseline))
            ax.plot(x, h*1000, color=color, label=stage,
                    ls="--" if stage == "A1_scaled" else "-",
                    marker="." if stage == "contact" else None, lw=1.8)
            ax.scatter(x[p], h[p]*1000, color=color, s=22)
            columns.extend([z, h*1000])
            rows.append(f"| {choice} | {stage} | {t[p]:.2f} | {np.nanmax(h)*1000:.2f} | {(np.nanmax(z)-z[0])*1000:.2f} |")
        result["contact_samples_unchanged_by_interpolation"] = True
        result["event_s"] = float(t[event])
        result["scale"] = float(scale)
        metrics[choice] = result
        np.savez(out / f"{choice}_stages.npz", **v)
        header = ["time_s", "time_from_A1_event_s", "contact_count"]
        for stage in series:
            header.extend([f"{stage}_absolute_m", f"{stage}_relative_mm"])
        np.savetxt(out / f"{choice}_stages.csv", np.column_stack(columns), delimiter=",", header=",".join(header), comments="")
        ax.axvline(0, color="gray", ls=":")
        ax.axhline(0, color="gray", lw=.5)
        ax.set(title=f"TT {choice}", xlabel="Time from A1 takeoff label [s]", ylabel="Height relative to event [mm]")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    assert len(traces) == 4
    fig.suptitle("Jump body-height calculation stages — IK / PPO excluded\nCommon time axis; contact gaps remain NaN; contact baseline uses interpolated event height")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out / "height_stages.png", dpi=160)
    fig.savefig(out / "height_stages.svg")
    plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    (out / "report.md").write_text("""# Jump 몸체 높이 단계별 검증 (IK·PPO 제외)

실제 `retarget_tt()` 계산 경로에서 중간값을 기록했다. 알고리즘 계산식과 기존 실행 기본값은 변경하지 않았다. 진단 모드만 IK 이전에 반환하며 원본/리타겟팅 파일을 쓰지 않는다.

## 기준

A1 입력은 `artifacts/pd_replay/jumpInPlace.npz`이다. 원시 Isaac Gym 파일을 새로 가공하지 않고 실제 리타겟팅 입력을 비교한다. 공통 이벤트는 A1 첫 무접촉 라벨(0.66 s)이며 시간축 이동은 모두 동일하다. A1, 보간 후, 평활화 후 곡선은 각각 이벤트 높이를 뺐다. 접촉 계산값은 공중에서 정의되지 않으므로 NaN을 유지하며, 정규화에만 보간 후 이벤트 높이를 사용했다. 접촉 곡선의 공중 구간을 별도로 채우지 않았다.

`A1_scaled`는 다리 길이 배율만 적용한 대조군이다. 진폭 감소와 시간 변형을 구분하기 위해 표시했다. 표의 최대 상대 높이는 이벤트 이전 정점도 포함하며, 이륙 이후 상승량을 뜻하지 않는다. 별도 `max_post_event_height_mm`도 JSON에 기록했다.

## 단계별 결과

| 형상 | 단계 | 정점 시각 (s, 클립 기준) | 최대 이벤트 상대 높이 (mm) | 시작 높이 대비 상승 (mm) |
|---|---|---:|---:|---:|
""" + "\n".join(rows) + """

## 해석 및 판정

네 형상에서 IK 이전 높이 계산 결과가 동일했다. 현재 입력에 대해 **궤적 변형**으로 판정한다.

1. 다리 길이 비율 0.9375만 적용하면 정점 0.76 s와 곡선 형태는 유지되고 진폭만 6.25% 줄어든다.
2. 접촉 표본 중 최대값은 0.64 s이며 시작 높이 대비 81.40 mm다. **접촉값은 공중에서 미정의이므로 이것을 점프 정점 이동으로 해석할 수 없다.** 접촉 프레임의 보정량 변화와 공중 보간의 추가 변형은 별도 검증 결과 `runs/TT_retarget/jump_interpolation/report.md`를 참고한다.
3. 보간 후에도 정점 0.64 s와 시작 높이 대비 상승 81.40 mm가 유지된다. 접촉 프레임은 원래 계산값을 유지하므로 보간으로 그 구간의 왜곡을 복원하지 않는다.
4. 평활화 후 정점은 여전히 0.64 s다. 시작 높이 대비 상승은 67.63 mm로 더 줄어든다. 필터가 조기 정점을 처음 만든 것은 아니다.

A1은 0.64→0.76 s 동안 428.34→497.49 mm로 상승하지만, TT 접촉 기반 목표는 390.17→287.67 mm로 하강한다. A1 정점 0.76 s에 발 하나가 접촉으로 표시되어 있어 공중 보간이 아닌 접촉 기반 계산을 적용받는다. 이것이 해당 구간의 경로 선택과 궤적 변형을 보여주지만, 라벨 자체가 틀렸다는 증명은 아니다.

최대 이벤트 상대 높이는 A1 47.37 mm, 단순 배율 적용 44.41 mm, 접촉/보간 15.46 mm, 평활화 5.24 mm다. TT의 이 최대값은 이벤트 이전에 발생한다. 보간/평활화 곡선은 이벤트 이후 기준 높이를 넘지 않는다.

## 검사와 한계

보간 결과는 접촉 프레임의 계산 높이를 그대로 유지하는지 검사했다. 기존 보간/필터 수식과 배열이 정확히 일치하는지도 검사했다. IK·역동역학·레퍼런스 쓰기에 도달하면 실패하도록 차단한 상태에서 실행했다.

연속 곡선의 형태와 접촉 라벨을 함께 해석해야 한다. 접촉 기반 정점은 관측 가능한 접촉 프레임 중 정점이며 공중 높이는 정의되지 않는다. 라벨 기반 이벤트가 실제 물리적 이륙인지, IK 제약 및 동역학 실행 가능성은 검증하지 않았다.

재현: codesign 루트에서 `../transformer-transformer/.venv/bin/python code/motion_tracking/validate_jump_height_stages.py`.
""")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
