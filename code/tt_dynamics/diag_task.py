"""task 결과 진단 (선택지 A): TT 예측 S vs MJX 실제 S 를 32 step window 증분으로 비교, 회전/직진 구간별 오차.

절대 오차는 open-loop 누적이 섞이므로 window 안의 증분 (Δyaw, heading frame Δxy) 만 비교한다.
구간 분류 = TT 예측 window 의 |Δyaw| > TURN_DEG (회전) / 그 외 (직진). MJX 낙상 이후 step 은 제외.

usage (codesign/ 에서): python3 code/tt_dynamics/diag_task.py runs/pipeline_v1/task/track_try1
"""

import json
import sys
from pathlib import Path

import numpy as np

W, TURN_DEG = 32, 10.0
run = Path(sys.argv[1])


def yaw(x):
    return np.unwrap(np.arctan2(x[..., 4], x[..., 3]), axis=-1)


def window_stats(pred, mjx, mask):
    rows = []
    for b in range(len(pred)):
        yp, ym = yaw(pred[b]), yaw(mjx[b])
        for t0 in range(0, pred.shape[1] - W + 1, W - 1):
            t1 = t0 + W - 1
            if not mask[b, t0:t1 + 1].all():
                break
            dyp, dym = np.degrees(yp[t1] - yp[t0]), np.degrees(ym[t1] - ym[t0])
            out = []
            for x, y in ((pred[b], yp), (mjx[b], ym)):  # window 시작 heading frame 의 Δxy
                c, s = np.cos(y[t0]), np.sin(y[t0])
                d = x[t1, :2] - x[t0, :2]
                out.append(np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]))
            rows.append(dict(cand=b, t0=t0, turn=abs(dyp) > TURN_DEG, dyaw_pred=dyp, dyaw_mjx=dym,
                             yaw_err=abs(dyp - dym), pos_err_cm=100 * np.linalg.norm(out[0] - out[1]),
                             joint_err_deg=float(np.degrees(np.sqrt(((pred[b, t0:t1 + 1, 14:26] - mjx[b, t0:t1 + 1, 14:26]) ** 2).mean())))))
    return rows


def summarize(rows):
    res = {}
    for name, sel in (("회전", lambda r: r["turn"]), ("직진", lambda r: not r["turn"]), ("전체", lambda r: True)):
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        g = lambda k: float(np.mean([r[k] for r in rs]))
        res[name] = dict(n=len(rs), yaw_err_deg=g("yaw_err"), pos_err_cm=g("pos_err_cm"), joint_err_deg=g("joint_err_deg"),
                         dyaw_pred_deg=g("dyaw_pred"), dyaw_mjx_deg=g("dyaw_mjx"),
                         sign_agree=float(np.mean([np.sign(r["dyaw_pred"]) == np.sign(r["dyaw_mjx"]) for r in rs])))
    if "회전" in res and "직진" in res:
        res["ratio_yaw_turn_over_straight"] = res["회전"]["yaw_err_deg"] / res["직진"]["yaw_err_deg"]
    return res


out = {}
for f in sorted(run.glob("scale_*.npz"), key=lambda p: float(p.stem.split("_")[1])):
    z = np.load(f)
    out[f.stem] = summarize(window_stats(z["pred"], z["mjx"], z["mjx_mask"]))
(run / "diag_task.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
for k, v in out.items():
    print(k, {c: {m: round(x, 2) if isinstance(x, float) else x for m, x in d.items()} if isinstance(d, dict) else round(d, 2) for c, d in v.items()})
