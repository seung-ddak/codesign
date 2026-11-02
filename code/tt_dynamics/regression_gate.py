"""2 번 통과 시험: diffusion 없이 순수 회귀로 (E, s0, A 열) → 1 s 몸통 속도 (vx, vy, yaw_rate, z_mean, tilt_mean) 가 이 데이터에서 학습되는가.

remind 8-39 (데이터에 E 신호 있음) 후속. TT 가 E→S 를 못 배우는 게 데이터 문제인지 모델/학습 목표 문제인지 분리.
  fold "design": 16 형상을 4 개씩 빼고 학습 → 안 본 형상 예측 (E→S 일반화).
  fold "start" : 모션당 시작 프레임 8 개 중 2 개를 전 형상에서 빼고 학습 → 안 본 A 예측 (A→S, group 암기 배제). e_swap 형식은 group 무작위 4-fold.
  e_swap 형식 (v9 48 형상 등): --data 여러 dir, group = (source_design, source_episode), --min-alive 0.75.
  모델: M_E (E 입력), M_noE (E 제외, 같은 구조), B0 = 학습 형상 group 평균 (E 없는 상한선), ridge_E (선형).
통과 기준 (사전): design fold 에서 M_E vx RMSE < 0.5 × |vx| 평균 AND M_E 가 B0 대비 RMSE ≥ 20 % 감소 AND E 효과 부호 일치 ≥ 70 % (|true−B0| > 0.03 m/s cell).

usage (codesign_cleanup/ 에서):
  ../transformer-transformer/.venv/bin/python -I code/tt_dynamics/regression_gate.py --data runs/pipeline_v1/pd_rollouts_kp200 --out runs/pipeline_v1/regression_gate
"""
import argparse, base64, io, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_data import features  # noqa: E402
from measure_e_signal_lib import outcomes, NAMES  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, nargs="+", default=[Path("runs/pipeline_v1/pd_rollouts_kp200")], help="형상 폴더들의 상위 dir (여러 개 가능). npz 에 group_id 있으면 PD 형식, 없으면 e_swap 형식 (source_design, source_episode 로 group)")
p.add_argument("--min-alive", type=float, default=1.0, help="group 유지 조건: 생존 형상 비율 ≥ 이 값 (PD 1.0, e_swap 은 낙상 많아 0.75 권장)")
p.add_argument("--out", type=Path, default=Path("runs/pipeline_v1/regression_gate"))
p.add_argument("--horizon", type=int, default=50, help="step (0.02 s) 상한; group 별 공통 valid 길이와 min")
p.add_argument("--a-steps", type=int, default=16, help="A 열 다운샘플 수")
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--epochs", type=int, default=2000)
p.add_argument("--wd", type=float, default=1e-3)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--rest-features", type=Path, help="안착 자세 추가 입력 비교; 실제 s0 보존, 별도 사전 기준 사용")
args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

paths = sorted((d for root in args.data for d in root.iterdir() if (d / "primitive_seed0.npz").exists()), key=lambda d: d.name)
designs = [d.name for d in paths]; assert len(set(designs)) == len(designs), "형상 이름 중복"
Z = {d.name: dict(np.load(d / "primitive_seed0.npz", allow_pickle=True)) for d in paths}
D = len(designs); motions = [str(m) for m in Z[designs[0]]["motions"]]
fmt = "pd" if "group_id" in Z[designs[0]] else "eswap"
if fmt == "eswap":  # group = 같은 A 열 (source_design, source_episode); 형상마다 행 순서를 첫 형상 기준으로 정렬
    key = lambda z: list(zip(z["source_design"].tolist(), z["source_episode"].tolist()))
    ref = key(Z[designs[0]]); pos = {k: i for i, k in enumerate(ref)}
    for d in designs:
        k = key(Z[d]); assert set(k) == set(ref), f"{d}: A 열 집합이 다름"
        order = np.argsort([pos[x] for x in k])
        Z[d] = {n: (v[order] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(k) else v) for n, v in Z[d].items()}
        Z[d]["group_id"] = np.arange(len(ref))
G = len(Z[designs[0]]["group_id"]); assert all(np.array_equal(Z[d]["group_id"], Z[designs[0]]["group_id"]) for d in designs)
clip0 = Z[designs[0]]["clip"][:, 0]; sf0 = Z[designs[0]].get("start_frame")
# group 공통 지평 = 낙상 안 한 형상들의 valid 길이 최소 (PD 는 clip 끝, e_swap 은 100 step 고정), 상한 horizon
vl = np.stack([np.where(Z[d]["fell"], 10 ** 6, Z[d]["valid"].sum(1)) for d in designs], 1)
Tg = np.minimum(vl.min(1), args.horizon).astype(int)
E = np.stack([Z[d]["E"] for d in designs]).astype(np.float32)

# ---- 표본: cell (g, d) → x = [E 13 | s0 42 | A 다운샘플 a_steps×12], y = 5 outcome
X_E, X_S, X_A, Y, OK, GI, DI = [], [], [], [], [], [], []
for di, d in enumerate(designs):
    z = Z[d]; y, ok = outcomes(z, Tg)
    s0 = features(z["qpos"][:, :1].reshape(-1, 19), z["qvel"][:, :1].reshape(-1, 18), z["contact"][:, :1].reshape(-1, 4))  # [G,42]
    for g in range(G):
        idx = np.linspace(0, Tg[g] - 1, args.a_steps).astype(int)
        X_A.append(z["target_q"][g, idx].reshape(-1))
    X_E.append(np.repeat(E[di][None], G, 0)); X_S.append(s0); Y.append(y); OK.append(ok); GI.append(np.arange(G)); DI.append(np.full(G, di))
X_E, X_S, X_A = np.concatenate(X_E), np.concatenate(X_S), np.stack(X_A).astype(np.float32)
Y, OK, GI, DI = np.concatenate(Y).astype(np.float32), np.concatenate(OK), np.concatenate(GI), np.concatenate(DI)
# 낙상 cell 제외; 생존 형상 비율 < min_alive 인 group 은 통째로 제외 (B0 가 흔들림)
alive_g = np.array([OK[GI == g].mean() >= args.min_alive for g in range(G)]); use = alive_g[GI] & OK
X_E, X_S, X_A, Y, GI, DI = (a[use] for a in (X_E, X_S, X_A, Y, GI, DI))
N = len(Y); print(f"samples {N} = groups {alive_g.sum()}/{G} × designs {D} (format {fmt}, 낙상 cell 제외), dev {dev}", flush=True)


class MLP(nn.Module):
    def __init__(self, din, h, dout):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.f(x)


def fit_predict(Xtr, Ytr, Xte, seed):
    """표준화 + MLP, AdamW, 전 배치 학습. 반환 = Xte 예측 (원 단위)."""
    torch.manual_seed(seed)
    sx = Xtr.std(0); sx = np.maximum(sx, 1e-2 * max(float(sx.max()), 1e-6))  # 하한: fold 안 분산 ≈0 인 입력 (e_swap s0 등) 이 held-out 에서 폭주하는 것 방지
    mx, my, sy = Xtr.mean(0), Ytr.mean(0), Ytr.std(0) + 1e-6
    norm = lambda X: np.clip((X - mx) / sx, -10, 10).astype(np.float32)
    xt, yt = torch.tensor(norm(Xtr), device=dev), torch.tensor((Ytr - my) / sy, device=dev)
    net = MLP(xt.shape[1], args.hidden, yt.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for _ in range(args.epochs):  # 전 배치 (N ≈ 1.3k)
        loss = ((net(xt) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return net(torch.tensor(norm(Xte), device=dev)).cpu().numpy() * sy + my


def ridge_predict(Xtr, Ytr, Xte, lam=1.0, std=True):
    """std=False: 입력이 이미 표준화됨 (E 는 설계 입력이라 16 형상 전체로 표준화해도 누수 아님; fold 12 개로 표준화하면 분산 작은 좌표가 폭주)."""
    mx, sx = (Xtr.mean(0), Xtr.std(0) + 1e-6) if std else (0.0, 1.0); my = Ytr.mean(0)
    A = (Xtr - mx) / sx; W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - my))
    return ((Xte - mx) / sx) @ W + my


def group_mean_pred(tr, te):
    """B0: 같은 group 의 학습 형상 평균 (group 이 학습에 없으면 NaN)."""
    out = np.full((te.sum(), Y.shape[1]), np.nan, np.float32)
    for j, g in enumerate(GI[te]):
        m = tr & (GI == g)
        if m.any(): out[j] = Y[m].mean(0)
    return out


def metrics(pred, true, b0, mag):
    r = {}
    for k, name in enumerate(NAMES):
        err = pred[:, k] - true[:, k]; rmse = float(np.sqrt((err ** 2).mean()))
        d = {"rmse": rmse, "rmse_over_mag": float(rmse / mag[k])}
        if b0 is not None and np.isfinite(b0).all():
            rb = float(np.sqrt(((b0[:, k] - true[:, k]) ** 2).mean())); d["b0_rmse"] = rb; d["gain_vs_b0"] = float(1 - rmse / rb)
            te_eff, pr_eff = true[:, k] - b0[:, k], pred[:, k] - b0[:, k]
            thr = 0.03 if k < 3 else 0.01
            big = np.abs(te_eff) > thr
            d["sign_agree"] = float((np.sign(te_eff[big]) == np.sign(pr_eff[big])).mean()) if big.any() else float("nan")
            d["n_big"] = int(big.sum()); d["eff_corr"] = float(np.corrcoef(te_eff, pr_eff)[0, 1]) if pr_eff.std() > 0 else 0.0
        r[name] = d
    return r


mag = np.abs(Y).mean(0)  # 움직임 크기 (outcome 별 |y| 평균)
X_Es = ((X_E - E.mean(0)) / (E.std(0) + 1e-9)).astype(np.float32)  # E 전역 표준화 (ridge 잔차 모델용)
if sf0 is not None:
    start_rank = np.zeros(G, int)
    for c in np.unique(clip0):
        m = clip0 == c; start_rank[m] = sf0[m].argsort().argsort()
    second = ("start", [np.isin(start_rank[GI], [i, i + 4]) for i in range(4)])  # 모션 안 시작 프레임 순위 0..7
else:
    gperm = np.random.default_rng(0).permutation(G)
    second = ("group", [np.isin(GI, gperm[i::4]) for i in range(4)])  # e_swap: group (= A 열) 무작위 4-fold
folds = {"design": [np.isin(DI, np.arange(D)[i::4]) for i in range(4)], second[0]: second[1]}
feats = {"M_E": np.concatenate([X_E, X_S, X_A], 1), "M_noE": np.concatenate([X_S, X_A], 1), "ridge_E": np.concatenate([X_E, X_S, X_A], 1)}
if args.rest_features is not None:
    from rest_ablation import run
    run(args, designs, X_E, X_S, X_A, Y, GI, DI, folds, fit_predict, metrics, group_mean_pred)
    raise SystemExit(0)
res = {"samples": int(N), "designs": designs, "format": fmt, "groups_used": int(alive_g.sum()), "groups_total": int(G), "horizon_s_median": float(np.median(Tg[alive_g]) * 0.02), "movement_mag": dict(zip(NAMES, mag.tolist())), "folds": {}}
preds_all = {}
for fname, fl in folds.items():
    P = {m: np.zeros_like(Y) for m in list(feats) + (["M_E_resid", "ridge_Eonly_resid", "ridge_pergroup_resid"] if fname == "design" else [])}; B0 = np.zeros_like(Y)
    for te in fl:
        tr = ~te
        B0[te] = group_mean_pred(tr, te)
        for m, Xf in feats.items():
            if m == "ridge_E": P[m][te] = ridge_predict(Xf[tr], Y[tr], Xf[te])
            else: P[m][te] = np.mean([fit_predict(Xf[tr], Y[tr], Xf[te], s) for s in range(args.seeds)], 0)
        if fname == "design":  # 사후 추가 (post-hoc): group 평균 B0 를 알려주고 잔차 y − B0 만 E 로 맞춤 → group 식별 부담 제거, 순수 E→S
            B0tr = group_mean_pred(tr, tr)  # 학습 cell 의 B0 = 자기 포함 group 평균 (학습 형상 12 개)
            Rtr = Y[tr] - B0tr
            P["M_E_resid"][te] = B0[te] + np.mean([fit_predict(feats["M_E"][tr], Rtr, feats["M_E"][te], s) for s in range(args.seeds)], 0)
            P["ridge_Eonly_resid"][te] = B0[te] + ridge_predict(X_Es[tr], Rtr, X_Es[te], std=False)
            # group 별 별도 선형 (e_signal LODO 와 같은 구조, E×group 교호작용 허용; group 당 학습 형상 12 개 / 13 차원 → ridge λ 1)
            for g in np.unique(GI[te]):
                mtr, mte = tr & (GI == g), te & (GI == g)
                P["ridge_pergroup_resid"][mte] = B0[mte] + ridge_predict(X_Es[mtr], Y[mtr] - B0tr[GI[tr] == g], X_Es[mte], std=False)
        print(f"  fold {fname} te={te.sum()} done {time.time()-t0:.0f}s", flush=True)
    b0 = B0 if fname == "design" else None  # start/group fold 는 group 이 학습에 없어 B0 정의 안 됨
    res["folds"][fname] = {m: metrics(P[m], Y, b0, mag) for m in P}
    if b0 is not None: res["folds"][fname]["B0"] = metrics(B0, Y, None, mag)
    preds_all[fname] = (P, B0)
    print(fname, json.dumps({m: {n: round(v["rmse_over_mag"], 3) for n, v in r.items()} for m, r in res["folds"][fname].items()}, ensure_ascii=False), flush=True)

# ---- 판정
dm = res["folds"]["design"]["M_E"]["vx [m/s]"]
c1, c2, c3 = dm["rmse_over_mag"] < 0.5, dm["gain_vs_b0"] >= 0.2, dm["sign_agree"] >= 0.7
verdict = ("A: 통과 — 순수 회귀는 안 본 형상의 E→몸통 속도를 배움. TT 의 E 무시는 데이터가 아니라 TT 구조/학습 목표 문제 → 3 번 (TT 를 delta·Mode 1 중심으로)" if c1 and c2 and c3 else
           "B: 부분 — E 가 오차를 줄이지만 기준 미달 (비선형·형상 16 개 한계 추정). 3 번 전에 형상 수 또는 E 표현 재검토" if c1 and (c2 or c3) else
           "C: 실패 — E 를 줘도 group 평균 이상 못 맞힘. 이 데이터·형상 범위로는 E→S 학습 불가 → 형상 범위·모션 재설계")
pr = res["folds"]["design"]["M_E_resid"]["vx [m/s]"]; pq = res["folds"]["design"]["ridge_Eonly_resid"]["vx [m/s]"]; pg = res["folds"]["design"]["ridge_pergroup_resid"]["vx [m/s]"]
posthoc = (f"사후 (post-hoc, 사전 기준 아님): B0 를 알려준 잔차 모델 — M_E_resid vx gain {pr['gain_vs_b0']*100:+.0f} %, sign {pr['sign_agree']*100:.0f} %, r {pr['eff_corr']:.2f}; "
           f"ridge_Eonly_resid (E 효과 모션 공통 가정) gain {pq['gain_vs_b0']*100:+.0f} %, sign {pq['sign_agree']*100:.0f} %, r {pq['eff_corr']:.2f}; "
           f"ridge_pergroup_resid (group 별 선형, e_signal 구조) gain {pg['gain_vs_b0']*100:+.0f} %, sign {pg['sign_agree']*100:.0f} %, r {pg['eff_corr']:.2f}. "
           + ("→ E→S 신호는 group 별로는 학습되나 하나의 모델로 교호작용까지 배우진 못함" if pg["gain_vs_b0"] >= 0.2 > max(pr["gain_vs_b0"], pq["gain_vs_b0"]) else
              "→ E→S 신호는 학습됨" if max(pr["gain_vs_b0"], pq["gain_vs_b0"], pg["gain_vs_b0"]) >= 0.2 else "→ 잔차로 떼어내도 20 % 미달"))
res["verdict"] = {"text": verdict, "posthoc": posthoc, "c1_rmse<0.5mag": bool(c1), "c2_gain>=20%": bool(c2), "c3_sign>=70%": bool(c3)}
res["elapsed_s"] = time.time() - t0
json.dump(res, open(args.out / "regression_gate.json", "w"), indent=1, ensure_ascii=False)

# ---- 그림
P, B0 = preds_all["design"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
k = 0
ax[0].scatter(Y[:, k] - B0[:, k], P["M_E"][:, k] - B0[:, k], s=6, alpha=.5, label="M_E"); ax[0].scatter(Y[:, k] - B0[:, k], P["ridge_E"][:, k] - B0[:, k], s=6, alpha=.4, label="ridge_E")
lim = np.abs(Y[:, k] - B0[:, k]).max(); ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=.8); ax[0].set(xlabel="true E effect (vx − group mean) [m/s]", ylabel="predicted E effect", title="held-out designs: E effect on vx"); ax[0].legend()
models = ["B0", "M_noE", "ridge_E", "M_E", "ridge_Eonly_resid", "ridge_pergroup_resid", "M_E_resid"]
for j, n in enumerate(NAMES):
    ax[1].bar(np.arange(len(models)) + j * 0.15, [res["folds"]["design"][m][n]["rmse_over_mag"] for m in models], 0.15, label=n)
ax[1].set_xticks(np.arange(len(models)) + 0.3); ax[1].set_xticklabels(models, rotation=30, ha="right", fontsize=8); ax[1].set(title="design fold: RMSE / |movement|", ylim=(0, 1.2)); ax[1].axhline(0.5, color="k", ls=":", lw=.8); ax[1].legend(fontsize=7)
per_d = [float(np.sqrt(((P["M_E"][DI == i, 0] - Y[DI == i, 0]) ** 2).mean())) for i in range(D)]
per_d0 = [float(np.sqrt(((B0[DI == i, 0] - Y[DI == i, 0]) ** 2).mean())) for i in range(D)]
ax[2].bar(np.arange(D) - 0.2, per_d0, 0.4, label="B0 (group mean)"); ax[2].bar(np.arange(D) + 0.2, per_d, 0.4, label="M_E")
ax[2].set_xticks(range(D)); ax[2].set_xticklabels(designs, rotation=90, fontsize=7); ax[2].set(title="vx RMSE per held-out design [m/s]"); ax[2].legend()
fig.tight_layout(); b = io.BytesIO(); fig.savefig(b, format="png", dpi=110); plt.close(fig); img = base64.b64encode(b.getvalue()).decode()

# ---- report
SECOND = second[0]
def tbl(fname):
    ms = ["B0", "M_noE", "ridge_E", "M_E", "ridge_Eonly_resid", "ridge_pergroup_resid", "M_E_resid"] if fname == "design" else ["M_noE", "ridge_E", "M_E"]
    rows = "".join(f"<tr><td>{m}</td>" + "".join(
        f"<td>{res['folds'][fname][m][n]['rmse']:.3f} ({res['folds'][fname][m][n]['rmse_over_mag']:.2f})" +
        (f"<br>gain {res['folds'][fname][m][n]['gain_vs_b0']*100:+.0f}%, sign {res['folds'][fname][m][n]['sign_agree']*100:.0f}% (n={res['folds'][fname][m][n]['n_big']}), r {res['folds'][fname][m][n]['eff_corr']:.2f}" if "gain_vs_b0" in res['folds'][fname][m][n] else "") + "</td>"
        for n in NAMES) + "</tr>" for m in ms)
    return f"<table><tr><th>모델</th>{''.join(f'<th>{n}</th>' for n in NAMES)}</tr>{rows}</table>"

html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Regression dynamics gate</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:auto;padding:16px;line-height:1.45}}table{{border-collapse:collapse;font-size:12px;margin:6px 0}}td,th{{border:1px solid #999;padding:3px 7px;text-align:right;vertical-align:top}}th{{background:#eee}}td:first-child{{text-align:left}}.v{{background:#fff3cd;padding:8px;border:1px solid #e0c060}}img{{max-width:100%}}</style></head><body>
<h1>2 번 통과 시험: 순수 회귀 dynamics 가 E→몸통 속도를 배우는가</h1>
<p>2026-11-02. remind 8-39 후속. diffusion·TT 없이 작은 MLP 로 (E, s0, A 열) → 1 s 평균 몸통 속도 예측. 표본 {N} (group {int(alive_g.sum())} × 형상 {D}, 낙상 cell 제외). 소요 {res['elapsed_s']/60:.1f} 분.</p>
<div class="v"><b>판정 (사전 기준):</b> {verdict}<br>{posthoc}<br>기준 (design fold, vx): (1) RMSE &lt; 0.5×|vx| → {dm['rmse_over_mag']:.2f} {'통과' if c1 else '미달'}; (2) B0 대비 RMSE 감소 ≥ 20 % → {dm['gain_vs_b0']*100:+.0f} % {'통과' if c2 else '미달'}; (3) E 효과 부호 일치 ≥ 70 % → {dm['sign_agree']*100:.0f} % {'통과' if c3 else '미달'}.</div>
<h2>조건</h2><table><tr><th>항목</th><th>값</th></tr>
<tr><td>데이터</td><td>{' + '.join(str(d) for d in args.data)} (형식 {fmt}, 형상 {D}: {', '.join(designs)}). 낙상 cell 제외, 생존 비율 ≥ {args.min_alive} 인 group {int(alive_g.sum())}/{G} 사용</td></tr>
<tr><td>입력</td><td>E 13 + s0 42 (rollout_data.features 첫 step) + A = target_q 를 {args.a_steps} step 균등 다운샘플 (×12). M_noE 는 E 제외</td></tr>
<tr><td>출력</td><td>vx, vy (시작 heading 프레임), yaw_rate, z 평균, tilt 평균. 지평 = group 공통 valid 길이 ∧ {args.horizon*0.02:.0f} s (중앙값 {res['horizon_s_median']:.2f} s)</td></tr>
<tr><td>모델</td><td>MLP {args.hidden}×2 GELU, AdamW lr 1e-3 wd {args.wd}, 전 배치 {args.epochs} step cosine, seed {args.seeds} 평균. 입력 표준화 (std 하한 = 최대 std 의 1 %, ±10 clip)·출력 표준화. ridge_E = 같은 입력 선형 (λ 1)</td></tr>
<tr><td>fold design</td><td>형상 {D//4} 개씩 held-out (4 fold, 전 형상 1 회씩, 학습 형상 {D - D//4}). B0 = 같은 group 의 학습 형상 평균 (E 를 모르는 최선)</td></tr>
<tr><td>fold {SECOND}</td><td>{'모션당 시작 프레임 8 개 중 2 개를 전 형상에서 held-out (4 fold)' if SECOND == 'start' else 'group (= A 열) 을 무작위 4-fold 로 held-out'}. 안 본 A 열 → A→S 를 group 암기 없이 측정. B0 정의 불가</td></tr>
<tr><td>사후 추가 모델</td><td>M_E_resid = B0 + MLP(E, s0, A → y − B0); ridge_Eonly_resid = B0 + ridge(E 13 → y − B0, λ 1, 전 group 공유); ridge_pergroup_resid = group 마다 별도 ridge(E 13 → y − B0) (e_signal LODO 와 같은 구조, 학습 형상 12 개). 사전 기준 모델이 B0 (group 평균) 조차 재현 못 해 E 효과와 group 식별 실패가 섞여서 추가. 사전 판정에는 미반영</td></tr>
<tr><td>E 효과 부호 일치</td><td>|true − B0| &gt; 0.03 m/s (yaw 0.03 rad/s, z·tilt 0.01) 인 cell 에서 sign(pred − B0) = sign(true − B0) 비율</td></tr></table>
<h2>결과 — design fold (안 본 형상)</h2><p>칸 = RMSE (RMSE/|움직임|). gain = B0 대비 RMSE 감소율, sign = E 효과 부호 일치, r = E 효과 상관.</p>{tbl('design')}
<h2>결과 — {SECOND} fold (안 본 {'시작 프레임' if SECOND == 'start' else 'group'} = 안 본 A 열)</h2>{tbl(SECOND)}
<img src="data:image/png;base64,{img}">
<h2>사실 / 추정</h2><ul>
<li>사실: 표·그림은 held-out 예측 직접 계산. B0 가 "E 를 몰라도 이 group 은 이만큼 움직인다" 의 상한이므로 M_E 의 gain·sign 이 E→S 학습 여부의 직접 증거.</li>
<li>사실: start fold 는 같은 모션의 다른 위상이라 A 가 완전히 새롭지는 않음 (같은 모션 안 보간). 다른 모션으로의 일반화는 측정하지 않음.</li>
<li>사실: group 별 ridge 의 B0 대비 vx gain 은 학습 형상 수에 민감 (별도 계산, λ 1: 15 형상 LODO +31 %, 12 형상 4-fold +11 %, 8 형상 2-fold +13 %). 13 차원 E 를 12~15 점으로 맞추는 과소결정 상태.</li>
<li>추정: 통과 시 TT 의 E 무시 (D4·v8·v12) 는 학습 목표 (절대 S 복원이 직전 S 복사로 풀림) 쪽. 미통과 시 13 차원 E 를 16 점으로 보간하는 한계.</li></ul>
<h2>재현</h2><pre>../transformer-transformer/.venv/bin/python -I code/tt_dynamics/regression_gate.py --data {' '.join(str(d) for d in args.data)} --min-alive {args.min_alive} --out {args.out}</pre>
</body></html>"""
(args.out / "report.html").write_text(html)
print("verdict:", verdict, f"({res['elapsed_s']/60:.1f} min)")
