"""Mode 1 예측의 몸통 속도·이동 방향 지표 (넘어지지 않은 에피소드)."""
import json, sys, numpy as np
D = sys.argv[1]
z = np.load(D + "/samples.npz"); g = ~z["fell"]
t, p, m = z["true"][g], z["mode1"][g], z["mask"][g]
ok = m[:, 10]
r = dict(vel_corr_x=float(np.corrcoef(p[ok, 1:11, 8].ravel(), t[ok, 1:11, 8].ravel())[0, 1]),
         vel_corr_y=float(np.corrcoef(p[ok, 1:11, 9].ravel(), t[ok, 1:11, 9].ravel())[0, 1]))
k = 10
ps = np.stack([p[:, i:i + k, 8:10].mean(1) for i in range(1, 490, k)], 1)
ts = np.stack([t[:, i:i + k, 8:10].mean(1) for i in range(1, 490, k)], 1)
ms = np.stack([m[:, i:i + k].all(1) for i in range(1, 490, k)], 1)
r["vel_corr_all_x"] = float(np.corrcoef(p[:, 1:, 8][m[:, 1:]], t[:, 1:, 8][m[:, 1:]])[0, 1])
r["vel_corr_all_y"] = float(np.corrcoef(p[:, 1:, 9][m[:, 1:]], t[:, 1:, 9][m[:, 1:]])[0, 1])
r["smoothed10_corr_x"] = float(np.corrcoef(ps[ms][:, 0], ts[ms][:, 0])[0, 1])
r["smoothed10_corr_y"] = float(np.corrcoef(ps[ms][:, 1], ts[ms][:, 1])[0, 1])
r["vel_err"] = float(np.linalg.norm(p[:, 1:, 8:10] - t[:, 1:, 8:10], axis=-1)[m[:, 1:]].mean())
r["vel_err_zero"] = float(np.linalg.norm(t[:, 1:, 8:10], axis=-1)[m[:, 1:]].mean())
mm = m[:, 1:-1]
r["speed_from_pred_pos"] = float((np.linalg.norm(p[:, 2:, 0:2] - p[:, 1:-1, 0:2], axis=-1)[mm] / .02).mean())
r["speed_from_true_pos"] = float((np.linalg.norm(t[:, 2:, 0:2] - t[:, 1:-1, 0:2], axis=-1)[mm] / .02).mean())
for k in (10, 25, 63):
    o = m[:, k]; dt = t[o, k, :2] - t[o, 0, :2]; dp = p[o, k, :2] - p[o, 0, :2]
    cos = (dt * dp).sum(1) / (np.linalg.norm(dt, axis=1) * np.linalg.norm(dp, axis=1) + 1e-9)
    r[f"dir_cos_step{k}"] = float(cos.mean())
    r[f"move_true_cm_step{k}"] = float(np.linalg.norm(dt, axis=1).mean() * 100)
    r[f"move_pred_cm_step{k}"] = float(np.linalg.norm(dp, axis=1).mean() * 100)
json.dump(r, open(D + "/velocity_result.json", "w"), indent=1)
print(json.dumps(r, indent=1))
