"""Separate contact-offset variation from interpolation effects; no IK/PPO."""
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pd_replay


def gaps(ok):
    indices = np.flatnonzero(~ok)
    return [g for g in np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1) if len(g)]


def main():
    assert [g.tolist() for g in gaps(np.array([False, True, False, False, True, False]))] == [[0], [2, 3], [5]]
    assert gaps(np.ones(3, dtype=bool)) == []
    root = Path(__file__).resolve().parents[2]
    out = root / "runs/TT_retarget/jump_interpolation"
    with patch.object(pd_replay, "foot_lock", side_effect=AssertionError("IK excluded")), \
         patch.object(pd_replay, "inverse_check", side_effect=AssertionError("dynamics excluded")):
        traces = pd_replay.retarget_tt(str(root / "artifacts/pd_replay"), None, ["jumpInPlace"], height_trace_only=True)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    rows = []
    for (choice, _), v in traces.items():
        ok, off, dt = v["interp_ok"], v["off"], v["dt"]
        a = v["scale"]*v["a1_z"]
        t = np.arange(len(a))*dt
        event = int(np.flatnonzero(~ok)[0])
        x = t-t[event]
        np.testing.assert_allclose(np.diff(x), np.diff(t), atol=1e-14)
        assert x[event] == 0
        filled = np.interp(np.arange(len(a)), np.flatnonzero(ok), off[ok])
        np.testing.assert_allclose(a+filled, v["interpolated_z"], atol=1e-14)
        np.testing.assert_array_equal(ok, ~np.isnan(v["contact_z"]))
        np.testing.assert_allclose(v["interpolated_z"][ok], v["contact_z"][ok], atol=1e-14)
        contact_delta = off-off[np.flatnonzero(ok)[0]]
        groups = gaps(ok)
        fig, axes = plt.subplots(1+len(groups), 1, figsize=(11, 3.5*(1+len(groups))))
        axes = np.atleast_1d(axes)
        axes[0].plot(x, contact_delta*1000, "o-", markersize=3, label="Contact offset minus first contact offset")
        axes[0].axvline(0, color="gray", ls=":")
        axes[0].set(ylabel="Offset change [mm]", xlabel="Time from A1 first no-contact frame [s]", title=f"TT {choice}: actual np.interp valid mask (gaps remain NaN)")
        result = {"time_origin_clip_s": float(t[event]), "contact_offset_range_mm": float(np.ptp(off[ok])*1000),
                  "contact_delta_from_first_min_mm": float(np.nanmin(contact_delta)*1000),
                  "contact_delta_from_first_max_mm": float(np.nanmax(contact_delta)*1000), "gaps": []}
        for g, ax in zip(groups, axes[1:]):
            left, right = int(g[0]-1), int(g[-1]+1)
            bounded = left >= 0 and right < len(a)
            anchor = left if left >= 0 else right
            ix = np.arange(max(0, left), min(len(a)-1, right)+1)
            reference = a[ix]+off[anchor]
            actual = v["interpolated_z"][ix]
            baseline = reference[0]
            delta = actual-reference
            dv = float((off[right]-off[left])/((right-left)*dt)) if bounded else 0.0
            if bounded:
                np.testing.assert_allclose(delta, dv*(t[ix]-t[left]), atol=1e-14)
                np.testing.assert_allclose(np.diff(actual-reference)/dt, dv, atol=1e-12)
            else:
                np.testing.assert_allclose(delta, 0, atol=1e-14)
            peaks = {}
            for label, z in (("fixed_offset", reference), ("linear_offset", actual)):
                p = int(np.argmax(z))
                peaks[label] = {"window_max_time_s": float(t[ix[p]]),
                    "window_max_time_from_event_s": float(x[ix[p]]),
                    "window_max_relative_mm": float((z[p]-baseline)*1000),
                    "interior_peak": bool(0 < p < len(z)-1)}
            item = dict(first_air_frame=int(g[0]), last_air_frame=int(g[-1]),
                t_L_s=float(t[left]) if left >= 0 else None, t_R_s=float(t[right]) if right < len(a) else None,
                t_L_from_event_s=float(x[left]) if left >= 0 else None,
                t_R_from_event_s=float(x[right]) if right < len(a) else None,
                mode="linear" if bounded else "constant extrapolation",
                off_L_mm=float(off[left]*1000) if left >= 0 else None,
                off_R_mm=float(off[right]*1000) if right < len(a) else None,
                delta_v_mps=dv, max_air_abs_delta_mm=float(np.max(np.abs(filled[g]-off[anchor]))*1000),
                right_endpoint_delta_mm=float(delta[-1]*1000), **peaks)
            result["gaps"].append(item)
            ol = "—" if item['off_L_mm'] is None else f"{item['off_L_mm']:.2f}"
            ore = "—" if item['off_R_mm'] is None else f"{item['off_R_mm']:.2f}"
            tl = "—" if left < 0 else f"{x[left]:.2f}"
            tr = "—" if right >= len(a) else f"{x[right]:.2f}"
            rows.append(f"| {choice} | {tl}–{tr} | {ol} | {ore} | {dv:+.4f} | {item['max_air_abs_delta_mm']:.2f} |")
            ax.plot(x[ix], (reference-baseline)*1000, "o-", color="darkorange", label="Scaled A1 + fixed left offset")
            ax.plot(x[ix], (actual-baseline)*1000, "o-", color="royalblue", label="Actual linear offset")
            if x[ix[0]] <= 0 <= x[ix[-1]]:
                ax.axvline(0, color="gray", ls=":")
            ax.set(xlabel="Time from A1 first no-contact frame [s]", ylabel="Height from shared left baseline [mm]",
                   title=f"Gap {g[0]}–{g[-1]} | added vertical velocity {dv:+.4f} m/s")
            axes[0].axvspan(x[g[0]], x[g[-1]]+dt, alpha=.12, color="gray")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{choice}_interpolation.png", dpi=160)
        plt.close(fig)
        np.savez(out / f"{choice}_traces.npz", **v, time_clip_s=t, time_from_event_s=x,
                 interpolated_off=filled, contact_delta_from_first=contact_delta)
        report[choice] = result
    (out / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    (out / "report.md").write_text("""# 접촉 보정과 공중 보간 분리 검증

실제 `retarget_tt`의 `ok = ~np.isnan(zb)`와 `off`를 직접 기록했다. `foot_lock`의 정리된 라벨은 사용하지 않았다. IK/PPO/필터 효과는 이 비교에서 제외했다. 원본 레퍼런스를 변경하지 않았다.

접촉 프레임에서는 `off - off[first valid]`를 기록했다. 이는 일정한 형상 높이 차이를 제거한 **원본 스케일 궤적 대비 변화**이지, 지면 접촉 조건의 물리적 오류를 확정하는 값은 아니다.

각 연속 무효 구간을 별도로 구하고 인접 유효 프레임을 t_L, t_R로 정했다. 구간별 기준은 `scale*A1 + off_L`, 검증 대상은 실제 보간 결과이다. 두 곡선 모두 동일한 t_L 높이를 뺐다. 최대 공중 오차는 접촉 끝점을 제외한 실제 무효 프레임에서 계산했다. 양 끝점 차이는 JSON에 별도 기록했다. 경계 구간은 상수 외삽으로 처리한다.

그래프와 표는 A1 첫 무접촉 프레임(클립 0.66 s)을 공통 0초로 사용한다. 두 번째 구간도 별도로 원점을 재설정하지 않는다. 시간 표시만 이동했으며 높이 기준·속도·오차 계산은 그대로다. JSON의 기존 시간 필드는 클립 기준으로 보존하고 `*_from_event_s` 필드에 상대 시간을 추가했다.

| 형상 | t_L–t_R (s, 첫 무접촉 기준) | off_L (mm) | off_R (mm) | 추가 수직속도 (m/s) | 공중 표본 최대 높이 차이 (mm) |
|---|---|---:|---:|---:|---:|
"""+"\n".join(rows)+"""

`window_max_time_s`는 각 짧은 비교 구간의 최대값 시각이다. `interior_peak=false`이면 경계 최대값이므로 점프 정점으로 해석하지 않는다. 전체 점프 정점 이동을 공중 보간 하나의 영향으로 단정하지 않는다.

## 이번 데이터의 해석

네 형상 모두 동일하다. 접촉 보정량의 전체 클립 범위는 174.05 mm이며 첫 접촉 프레임 기준 변화는 -161.76~+12.30 mm이다. 이는 일정한 형상 오프셋으로는 원본 스케일 궤적과 접촉 기반 높이를 일치시킬 수 없음을 보여준다. 접촉 조건을 만족시키는 높이 자체가 잘못됐다는 뜻은 아니다.

- 첫 공중 표본은 0.00 s 한 프레임이다. 양쪽 접촉 프레임 -0.02/+0.02 s의 보정량 차이는 -71.74 mm, 추가 속도는 -1.7936 m/s이다. 실제 공중 표본은 고정 오프셋 기준보다 35.87 mm 낮다. 기준의 구간 최대 상승은 35.68 mm이지만 보간 결과는 왼쪽 경계가 최대(상대 높이 0)여서 상승이 억제된다.
- 두 번째 공중 표본은 +0.20/+0.22/+0.24 s이다. 양쪽 접촉 프레임 +0.18/+0.26 s의 보정량 차이는 +103.65 mm, 추가 속도는 +1.2956 m/s이다. 공중 표본은 기준보다 최대 77.73 mm 높다. 기준 곡선은 왼쪽 경계가 최대인 하강 구간인데, 보간 결과는 오른쪽 경계에서 +13.87 mm로 최대가 된다.
- 두 구간 모두 내부 최대점이 없으므로 점프 정점 이동량을 보고하지 않는다. 속도 변화는 20 ms 표본 사이 선형 오프셋의 기울기이며 실제 시뮬레이터 속도 측정이 아니다.

결론: **선형 보간은 각 공중 구간의 높이와 수직속도를 추가로 변형한다.** 동시에 접촉 프레임의 보정량도 일정하지 않다. 두 영향을 구분해야 하며 전체 변형을 보간 하나만의 문제로 단정할 수 없다. Savitzky–Golay 필터는 이번 정량 비교에 적용하지 않았다.

검사: 실제 보간 배열 재현, 접촉 표본 보존, 선형 보간 높이 차이와 분석적 속도 식 일치, 무효 구간 분리 self-check 통과. 실제 이륙·착지 여부와 접촉 라벨 정확성은 검증하지 않았다.

재현: `../transformer-transformer/.venv/bin/python code/motion_tracking/validate_jump_interpolation.py`
""")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
