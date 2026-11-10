"""1 s body-motion agreement of TT predictions with real data (from repro_check samples.npz).

For each episode and each full 1 s window inside the valid mask: yaw change and
displacement (forward/lateral in the heading frame at the window start), for the
real data (true) and the TT prediction given the real action (mode1).
"""
import json, sys
import numpy as np


def window_stats(x, L, per=50):
    yaw = np.unwrap(np.arctan2(x[:L, 4], x[:L, 3]))
    out = []
    for k in range((L - 1) // per):
        a, b = k * per, (k + 1) * per
        d = x[b, :2] - x[a, :2]
        c, s = np.cos(yaw[a]), np.sin(yaw[a])
        out.append((yaw[b] - yaw[a], c * d[0] + s * d[1], -s * d[0] + c * d[1]))
    return out


def metrics(path, group="nonfall"):
    z = np.load(path)
    sel = ~z["fell"] if group == "nonfall" else z["fell"]
    T, P = [], []
    for i in np.flatnonzero(sel):
        L = int(z["mask"][i].sum())
        T += window_stats(z["true"][i], L)
        P += window_stats(z["mode1"][i], L)
    T, P = np.asarray(T), np.asarray(P)
    res = dict(windows=len(T))
    for j, name in enumerate(("yaw_change", "forward", "lateral")):
        t, p = T[:, j], P[:, j]
        res[name] = dict(corr=float(np.corrcoef(t, p)[0, 1]), slope=float(np.polyfit(t, p, 1)[0]),
                         mae=float(np.abs(t - p).mean()), mean_abs_true=float(np.abs(t).mean()))
    return res


if __name__ == "__main__":
    print(json.dumps({p: metrics(p) for p in sys.argv[1:]}, indent=1))
