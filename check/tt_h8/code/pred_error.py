"""open-loop · MPC 의 TT 예측 (pred) 대 MuJoCo 실제 (actual) 오차 (2026-11-17 사용자 요청). summary.json 있는 과제만.
비교: pred[t] 대 actual[t] (actual 행 t = 제어 t 직전 상태), t < 300, 실제 valid 구간만. 후보 8 개 평균.
관절 = 특징 14:26 (rad) RMSE (°), 위치 = 몸통 xy 거리 (cm, 평균 · 마지막 valid 시점). 출력: pred_error.json"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; rows = []
for model in ['h8_e7', 'h64_e7']:
    for task in ['straight', 'circle', 'zigzag']:
        d = HERE / 'gen' / model / task
        if not (d / 'summary.json').exists(): continue
        for f in sorted(d.glob('scale_*.npz')):
            z = np.load(f)
            for mode in ['open', 'mpc']:
                p, a, v = z[f'{mode}_pred'], z[f'{mode}_actual'][:, :300], z[f'{mode}_valid'][:, :300]
                j, pm, pf, alive = [], [], [], []
                for i in range(len(p)):
                    m = v[i]
                    if m.sum() < 2: continue
                    j.append(np.degrees(np.sqrt(np.mean((p[i, m, 14:26] - a[i, m, 14:26]) ** 2))))
                    e = np.linalg.norm(p[i, m, :2] - a[i, m, :2], axis=1) * 100; pm.append(e.mean()); pf.append(e[-1]); alive.append(m.sum() / 30)
                rows.append(dict(model=model, task=task, scale=f.stem, mode=mode, n=len(j), joint_deg=float(np.mean(j)), pos_mean_cm=float(np.mean(pm)), pos_final_cm=float(np.mean(pf)), valid_s=float(np.mean(alive))))
(HERE / 'pred_error.json').write_text(json.dumps(rows, indent=1))
for r in rows: print(r['model'], r['task'], r['scale'], r['mode'], r['n'], f"{r['joint_deg']:.1f} {r['pos_mean_cm']:.1f} {r['pos_final_cm']:.1f} {r['valid_s']:.1f}")
