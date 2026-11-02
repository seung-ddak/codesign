"""E→몸통 신호 측정 (학습 없음): PD kp200 rollout 에서 같은 (모션, 시작 프레임) group 을 16 형상에 걸쳐 비교.

MJX 는 결정적이라 같은 형상 반복 분산 = 0. 따라서 형상 간 분산은
  (1) 움직임 크기 (|group 평균|)  (2) TT 예측 오차 (v12 eval_dynamics pos_err)  (3) E 좌표 선형 설명력 (LODO CV R²)
와 비교한다. 결과 = <out>/e_signal.json + report.html.

usage (codesign_cleanup/ 에서):
  ../transformer-transformer/.venv/bin/python -I code/tt_dynamics/measure_e_signal.py --data runs/pipeline_v1/pd_rollouts_kp200 --out runs/pipeline_v1/e_signal
"""
import argparse, base64, io, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, default=Path("runs/pipeline_v1/pd_rollouts_kp200"))
p.add_argument("--out", type=Path, default=Path("runs/pipeline_v1/e_signal"))
p.add_argument("--horizons", type=int, nargs="+", default=[50, 100], help="step (0.02 s): 50 = 1 s, 100 = 2 s")
p.add_argument("--tt-ref", type=Path, default=Path("runs/pipeline_v1/fast_mode24_v12_pd_kp200_mode1/tt_joint/eval_dynamics_ddim.json"))
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

designs = sorted(d.name for d in args.data.iterdir() if (d / "primitive_seed0.npz").exists())
Z = {d: np.load(args.data / d / "primitive_seed0.npz", allow_pickle=True) for d in designs}
g0 = Z[designs[0]]["group_id"]; motions = [str(m) for m in Z[designs[0]]["motions"]]
clip0 = Z[designs[0]]["clip"][:, 0]
assert all(np.array_equal(Z[d]["group_id"], g0) for d in designs), "group_id 가 형상 간에 다름"
E = np.stack([Z[d]["E"] for d in designs])  # [D,13]
D, G = len(designs), len(g0)


def yaw_of(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def outcomes(z, Tg):
    """episode 별 [vx, vy, yaw_rate, z_mean, tilt_mean] — group 별 지평 Tg[B] step (0..Tg) 평균 속도, 생존 여부."""
    B = len(Tg); i = np.arange(B)
    q = z["qpos"]
    ok = np.array([z["valid"][b, :Tg[b]].all() for b in range(B)]) & ~z["fell"]
    q0, qT = q[i, 0], q[i, Tg]
    yaw0, yawT = yaw_of(q0[:, 3:7]), yaw_of(qT[:, 3:7])
    dxy = qT[:, :2] - q0[:, :2]
    c, s = np.cos(yaw0), np.sin(yaw0)
    dx, dy = c * dxy[:, 0] + s * dxy[:, 1], -s * dxy[:, 0] + c * dxy[:, 1]
    dyaw = np.arctan2(np.sin(yawT - yaw0), np.cos(yawT - yaw0))
    sec = Tg * 0.02
    zm, tl = np.zeros(B), np.zeros(B)
    for b in range(B):
        qq = q[b, :Tg[b] + 1]
        up = 1 - 2 * (qq[:, 4] ** 2 + qq[:, 5] ** 2)
        zm[b], tl[b] = qq[:, 2].mean(), np.arccos(np.clip(up, -1, 1)).mean()
    return np.stack([dx / sec, dy / sec, dyaw / sec, zm, tl], 1), ok


NAMES = ["vx [m/s]", "vy [m/s]", "yaw_rate [rad/s]", "z_mean [m]", "tilt_mean [rad]"]


def anova2(Y):
    """Y [G,D] 반복 없음 2원 분해. 반환 η² (group, E 주효과, E×group 교호작용)."""
    m = Y.mean(); rm, cm = Y.mean(1, keepdims=True), Y.mean(0, keepdims=True)
    tot = ((Y - m) ** 2).sum()
    sg, sd = ((rm - m) ** 2).sum() * Y.shape[1], ((cm - m) ** 2).sum() * Y.shape[0]
    return dict(group=float(sg / tot), E_main=float(sd / tot), E_x_group=float((tot - sg - sd) / tot))


def lodo_r2(Y, lams=(0.0, 0.1, 1.0, 10.0)):
    """group 평균 제거 후 잔차를 E 13 좌표 (표준화) ridge 선형으로 설명 — leave-one-design-out CV R². λ 별 반환 (14 파라미터 / 15 점이라 λ=0 은 거의 보간)."""
    R = Y - Y.mean(1, keepdims=True)  # [G,D]
    Es = (E - E.mean(0)) / (E.std(0) + 1e-9)
    out = {}
    for lam in lams:
        pred = np.zeros_like(R)
        for d in range(D):
            tr = np.arange(D) != d
            Xt, Rt = Es[tr], R[:, tr].T  # [15,13], [15,G]
            W = np.linalg.solve(Xt.T @ Xt + lam * np.eye(13), Xt.T @ (Rt - Rt.mean(0)))
            pred[:, d] = Es[d] @ W + Rt.mean(0)
        out[f"lam{lam:g}"] = float(1 - ((R - pred) ** 2).sum() / (R ** 2).sum())
    return out


def e_consistency(y, clips):
    """E×group 교호작용을 E×motion (시작 프레임 8 개에 걸쳐 일관된 형상 효과) 과 E×start 잔차로 분해.
    + 같은 모션 안 시작 프레임 쌍 사이 형상 효과 벡터 (D,) 의 Spearman 상관 평균 (1 = 형상 순위가 시작 프레임과 무관하게 같음)."""
    R = y - y.mean(1, keepdims=True)  # 형상 효과 (group 별)
    R = R - R.mean(0, keepdims=True)  # E 주효과 제거 → 순수 교호작용
    ss_int = (R ** 2).sum(); ss_motion = 0.0; rhos = []
    for c in np.unique(clips):
        Rc = R[clips == c]  # [n_start, D]
        ss_motion += (Rc.mean(0, keepdims=True) ** 2).sum() * len(Rc)
        rk = Rc.argsort(1).argsort(1).astype(float)
        if len(Rc) > 1:
            cc = np.corrcoef(rk); rhos.append(cc[np.triu_indices(len(Rc), 1)].mean())
    return dict(frac_E_x_motion=float(ss_motion / ss_int), frac_E_x_start=float(1 - ss_motion / ss_int), spearman_between_starts=float(np.mean(rhos)))


tt = json.load(open(args.tt_ref)) if args.tt_ref.exists() else {}
res = {"designs": designs, "motions": motions, "n_groups": int(G), "horizons": {}}
figs = {}
for T in args.horizons:
    # group 별 공통 지평 = 16 형상 모두 valid 인 최소 길이 (clip 끝 때문에 시작 프레임이 늦으면 짧음), 상한 T
    Tg = np.minimum(np.stack([Z[d]["valid"].sum(1) for d in designs], 1).min(1), T)
    O, OK = zip(*(outcomes(Z[d], Tg) for d in designs))
    O, OK = np.stack(O, 1), np.stack(OK, 1)  # [G,D,5], [G,D]
    keep = OK.all(1)
    Y = O[keep]  # [G',D,5]
    h = {"T_s": T * 0.02, "horizon_s_median": float(np.median(Tg[keep]) * 0.02), "horizon_s_min": float(Tg[keep].min() * 0.02),
         "groups_all_alive": int(keep.sum()), "fell_cells": int((~OK).sum()), "per_outcome": {}}
    for k, name in enumerate(NAMES):
        y = Y[:, :, k]
        e_std = y.std(1, ddof=1)  # group 별 형상 간 std
        mag = np.abs(y.mean(1))
        h["per_outcome"][name] = dict(
            e_std_mean=float(e_std.mean()), e_std_median=float(np.median(e_std)),
            movement_mag_mean=float(mag.mean()), ratio_e_std_over_mag=float(e_std.mean() / max(mag.mean(), 1e-9)),
            e_range_mean=float((y.max(1) - y.min(1)).mean()),
            eta2=anova2(y), lodo_cv_r2=lodo_r2(y), consistency=e_consistency(y, clip0[keep]),
            per_motion={motions[c]: dict(e_std=float(e_std[clip0[keep] == c].mean()), mag=float(mag[clip0[keep] == c].mean()))
                        for c in range(len(motions)) if (clip0[keep] == c).any()})
    h["design_mean_vx"] = {d: float(Y[:, i, 0].mean()) for i, d in enumerate(designs)}
    res["horizons"][f"T{T}"] = h

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    y = Y[:, :, 0]; ax[0].imshow(y - y.mean(1, keepdims=True), aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    ax[0].set(title=f"vx (horizon<={T*0.02:.0f}s), group-mean removed [m/s]", xlabel="design", ylabel="group (motion×start)")
    ax[0].set_xticks(range(D)); ax[0].set_xticklabels(designs, rotation=90, fontsize=6)
    mnames = [m.replace("__r00", "") for m in motions]
    for k, c in ((0, "C0"), (2, "C1")):
        pm = res["horizons"][f"T{T}"]["per_outcome"][NAMES[k]]["per_motion"]
        xs = np.arange(len(pm)); w = 0.4
        ax[1].bar(xs + (0 if k == 0 else w), [v["e_std"] for v in pm.values()], w, color=c, label=f"{NAMES[k]} E-std")
        ax[1].plot(xs + (0 if k == 0 else w), [v["mag"] for v in pm.values()], "k_", ms=10)
    ax[1].set_xticks(np.arange(len(pm)) + 0.2); ax[1].set_xticklabels([m.replace("__r00", "") for m in pm], rotation=90, fontsize=7)
    ax[1].set(title="per motion: bar = std across designs, _ = |mean| (movement size)"); ax[1].legend(fontsize=7)
    ax[2].bar(range(D), [res["horizons"][f"T{T}"]["design_mean_vx"][d] for d in designs])
    ax[2].set_xticks(range(D)); ax[2].set_xticklabels(designs, rotation=90, fontsize=7); ax[2].set(title=f"mean vx per design [m/s] (all motions)")
    fig.tight_layout(); b = io.BytesIO(); fig.savefig(b, format="png", dpi=110); plt.close(fig)
    figs[T] = base64.b64encode(b.getvalue()).decode()

json.dump(res, open(args.out / "e_signal.json", "w"), indent=1, ensure_ascii=False)

# ---------- report ----------
tt_pos = {k: v.get("pos_err_cm") for k, v in tt.items() if isinstance(v, dict)}
rows = []
for T in args.horizons:
    h = res["horizons"][f"T{T}"]
    for name, o in h["per_outcome"].items():
        rows.append(f"<tr><td>{h['T_s']:.0f} s</td><td>{name}</td><td>{o['movement_mag_mean']:.3f}</td><td>{o['e_std_mean']:.3f}</td>"
                    f"<td>{o['ratio_e_std_over_mag']:.2f}</td><td>{o['e_range_mean']:.3f}</td><td>{o['eta2']['E_main']:.2f}</td>"
                    f"<td>{o['eta2']['E_x_group']:.2f}</td><td>{o['eta2']['group']:.2f}</td><td>{o['consistency']['frac_E_x_motion']:.2f}</td><td>{o['consistency']['spearman_between_starts']:.2f}</td><td>{max(o['lodo_cv_r2'].values()):.2f}</td></tr>")
h2 = res["horizons"][f"T{args.horizons[-1]}"]
dx2, yaw2, z2 = (h2["per_outcome"][n] for n in (NAMES[0], NAMES[2], NAMES[3]))
loco_ok = max(dx2["ratio_e_std_over_mag"], yaw2["ratio_e_std_over_mag"]) >= 0.2
r2 = lambda o: max(o["lodo_cv_r2"].values())
learn_ok = max(r2(dx2), r2(yaw2)) >= 0.3
consist_ok = max(dx2["consistency"]["spearman_between_starts"], yaw2["consistency"]["spearman_between_starts"]) >= 0.3
verdict = ("A: 데이터에 E→이동 신호 있음 + E 좌표로 선형 설명 가능 → 2 번 (회귀 dynamics 통과 시험) 진행" if loco_ok and learn_ok else
           "B1: E 신호 크고 시작 프레임 간 일관 (구조적) 이나 E 13 좌표 선형으론 설명 안 됨 → 형상 16 개론 E→이동 함수 복원 불가 (형상 수 확대 또는 E 차원 축소) 후 2 번" if loco_ok and consist_ok else
           "B2: E 신호는 크지만 시작 프레임마다 형상 순위가 바뀜 (비구조적·과도 응답) → 학습 가능한 E 효과 아님. 모션·지평·초기화 재검토" if loco_ok else
           "C: E 가 이동 (dx·dyaw) 을 거의 안 바꿈 → 형상 범위 확대 또는 E 가 결과를 가르는 모션 추가 전엔 TT 가 배울 것이 없음")
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>E signal measurement</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:auto;padding:16px;line-height:1.45}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #999;padding:3px 7px;text-align:right}}th{{background:#eee}}td:nth-child(2),td:nth-child(1){{text-align:left}}.v{{background:#fff3cd;padding:8px;border:1px solid #e0c060}}img{{max-width:100%}}</style></head><body>
<h1>1 번 측정: PD kp200 데이터에 E→몸통 신호가 있는가</h1>
<p>2026-11-02. 학습 없음. 데이터 <code>{args.data}</code>, 형상 {D} 개, group (모션×시작프레임) {G} 개, 한 cell = 1 episode (MJX 결정적 → 같은 형상 반복 분산 0).</p>
<div class="v"><b>판정 (사전 기준):</b> {verdict}<br>
기준 = 상한 2 s 지평에서 vx 또는 yaw_rate 의 (형상 간 std / |움직임 크기|) ≥ 0.2 → 신호 있음; LODO-CV R² ≥ 0.3 → E 좌표로 학습 가능.
실측: vx ratio {dx2['ratio_e_std_over_mag']:.2f}, yaw_rate ratio {yaw2['ratio_e_std_over_mag']:.2f}, ridge LODO R² (λ 최적) dx {r2(dx2):.2f} / dyaw {r2(yaw2):.2f} / z {r2(z2):.2f}; 시작 프레임 간 형상 순위 Spearman dx {dx2['consistency']['spearman_between_starts']:.2f} / dyaw {yaw2['consistency']['spearman_between_starts']:.2f}.<br>
추가 기준 (B 분기 분리): Spearman ≥ 0.3 → 구조적 (B1), 미만 → 비구조적 (B2).</div>
<h2>조건</h2><table><tr><th>항목</th><th>값</th></tr>
<tr><td>데이터</td><td>{args.data} (v11/v12 학습 데이터와 동일, kp 200 kd 0.5)</td></tr>
<tr><td>형상</td><td>{', '.join(designs)} (e003 held-out 제외)</td></tr>
<tr><td>모션 / 시작</td><td>{len(motions)} 모션 × 8 시작 프레임 = {G} group</td></tr>
<tr><td>결과량</td><td>vx·vy (시작 heading 프레임 평균 속도), yaw_rate, z 평균, tilt 평균. 지평 = group 별 16 형상 공통 valid 길이 (clip 끝 때문에 시작 프레임이 늦을수록 짧음, 최소 0.64 s), 상한 {', '.join(f'{t*0.02:.0f} s' for t in args.horizons)} (상한 2 s 일 때 중앙값 {h2['horizon_s_median']:.2f} s)</td></tr>
<tr><td>제외</td><td>16 형상 중 하나라도 낙상인 group 은 제외 ({h2['groups_all_alive']}/{G} 유지, 낙상 cell {h2['fell_cells']}/{G*D})</td></tr>
<tr><td>TT 오차 참고</td><td>v12 eval_dynamics pos_err_cm: {json.dumps(tt_pos)} (t100 = 2 s, n 작음)</td></tr></table>
<h2>결과 표</h2>
<table><tr><th>지평</th><th>결과량</th><th>|움직임| 평균</th><th>형상 간 std</th><th>std/|움직임|</th><th>형상 간 범위</th><th>η² E 주효과</th><th>η² E×group</th><th>η² group</th><th>교호 중 E×motion 비율</th><th>Spearman(start 간)</th><th>ridge LODO R² (최적 λ)</th></tr>{''.join(rows)}</table>
<p>η² = 반복 없는 2원 분해 (group 주효과 / E 주효과 / E×group 교호작용, 합 1). E 주효과 = 모든 모션에서 같은 방향으로 미는 형상 효과, E×group = 모션에 따라 다른 형상 효과 (TT 가 배워야 할 것은 둘 다). 교호 중 E×motion 비율 = E×group 교호작용 중 같은 모션의 시작 프레임 8 개에 공통인 부분 (높을수록 형상 효과가 모션 단위로 일관). Spearman = 같은 모션 안 시작 프레임 쌍 사이 형상 효과 순위 상관 평균 (1 = 어느 시작에서나 같은 형상이 빠름/느림). ridge LODO R² = group 평균 제거 잔차를 표준화 E 13 좌표 ridge (λ 0/0.1/1/10 중 최적) 로, 형상 1 개씩 빼고 맞춘 CV (15 점으로 13 차원 → 낮게 나올 수밖에 없음, 참고치).</p>
{''.join(f'<h2>{T*0.02:.0f} s</h2><img src="data:image/png;base64,{figs[T]}">' for T in args.horizons)}
<h2>사실 / 추정</h2>
<ul><li>사실: 위 표·그림 전부 데이터 직접 계산. TT 미사용.</li>
<li>추정: 형상 간 std 가 v12 TT 2 s 위치 오차 ({tt_pos.get('t100', 'NA')} cm, n=2) 보다 작으면 TT 가 E 효과를 못 배우는 게 모델만의 문제가 아님.</li>
<li>한계: 반복 분산 0 이라 "노이즈 대비" 는 없음. 비교 기준은 움직임 크기·TT 오차·선형 설명력 세 가지.</li></ul>
<h2>재현</h2><pre>../transformer-transformer/.venv/bin/python -I code/tt_dynamics/measure_e_signal.py --data {args.data} --out {args.out}</pre>
</body></html>"""
(args.out / "report.html").write_text(html)
print(json.dumps({k: {n: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in o.items() if kk != "per_motion"}
                      for n, o in h["per_outcome"].items()} for k, h in res["horizons"].items()}, indent=1, ensure_ascii=False))
print("verdict:", verdict)
