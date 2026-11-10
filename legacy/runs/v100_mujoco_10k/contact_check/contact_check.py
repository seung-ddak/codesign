"""H64 Mode 1 (실제 E·s0·A 제공) 예측에서 발 접촉 정확도와 몸통 속도 오차의 관계."""
import json, sys, numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "runs/v100_mujoco_10k/repro_nonfall"
z = np.load(D + "/samples.npz")
T, P, M, F = z["true"], z["mode1"], z["mask"], z["fell"]
C, V = slice(38, 42), slice(8, 10)
out = {}
for name, g in (("nonfall", ~F), ("fall", F)):
    t, p, m = T[g][:, 1:], P[g][:, 1:], M[g][:, 1:]
    tc, pc = t[..., C] > .5, p[..., C] > .5
    acc = (tc == pc)[m].mean()
    # 기준: 매 step 직전 실제 접촉을 그대로 쓴다 / 항상 4발 접지
    prev = T[g][:, :-1, C] > .5
    acc_prev = (tc == prev)[m].mean()
    acc_all4 = tc[m].mean()
    # 발별 정확도, 실제 접지 / 공중 각각
    on, off = tc[m], pc[m]
    recall_on = (off & on).sum() / on.sum()
    recall_air = (~off & ~on).sum() / (~on).sum()
    # step 마다 틀린 발 수 vs 몸통 수평 속도 오차
    wrong = (tc != pc).sum(-1)[m]
    verr = np.linalg.norm(p[..., V] - t[..., V], axis=-1)[m]
    vtrue = np.linalg.norm(t[..., V], axis=-1)[m]
    by = {int(k): dict(share=float((wrong == k).mean()), vel_err_m_s=float(verr[wrong == k].mean()))
          for k in range(5) if (wrong == k).sum() > 50}
    # 10 step 구간: 구간 접촉 정확도 상/하위 25% 에서 이동 방향 일치
    seg = []
    for i in range(len(t)):
        for s in range(0, 490, 10):
            if not m[i, s:s + 10].all():
                break
            a = (tc[i, s:s + 10] == pc[i, s:s + 10]).mean()
            dt = t[i, s + 9, :2] - t[i, s, :2]
            dp = p[i, s + 9, :2] - p[i, s, :2]
            if np.linalg.norm(dt) < .01:
                continue
            cos = float(dt @ dp / (np.linalg.norm(dt) * np.linalg.norm(dp) + 1e-9))
            seg.append((a, cos))
    seg = np.array(seg)
    lo, hi = np.quantile(seg[:, 0], [.25, .75])
    out[name] = dict(contact_acc=float(acc), baseline_prev_step=float(acc_prev), baseline_all_feet_down=float(acc_all4),
                     stance_recall=float(recall_on), swing_recall=float(recall_air), true_stance_share=float(on.mean()),
                     mean_true_feet_down=float(on.reshape(-1, 4).sum(1).mean() if False else tc[m].sum(-1).mean()),
                     mean_pred_feet_down=float(pc[m].sum(-1).mean()),
                     vel_err_by_wrong_feet=by, true_speed_m_s=float(vtrue.mean()),
                     segments=int(len(seg)),
                     dir_cos_low_contact_acc=float(seg[seg[:, 0] <= lo, 1].mean()),
                     dir_cos_high_contact_acc=float(seg[seg[:, 0] >= hi, 1].mean()),
                     contact_acc_quartiles=[float(lo), float(hi)],
                     corr_seg_contact_acc_vs_dir=float(np.corrcoef(seg[:, 0], seg[:, 1])[0, 1]))
json.dump(out, open(D + "/contact_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
