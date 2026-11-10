"""PPO 수렴 판정 (학습 중 기록 metrics.jsonl 사용). 수렴이면 exit 0, 아니면 1.

기준 (2026-11-10 사전 고정, 사용자 요청 "수렴하면 다음 단계"):
- 최소 학습량 200M step 이후에만 판정한다.
- 20M step 구간별 평균: 에피소드 길이, step 당 몸통 위치 오차, step 당 발 미끄러짐.
- 최근 3 구간 (60M step) 동안 에피소드 길이 증가 < 2 % 이고, 위치 오차·미끄러짐 감소 < 2 % 이면 수렴.
"""
import json, sys
from pathlib import Path
import numpy as np
O = Path(__file__).resolve().parent
MIN_STEP, BIN, WIN, TOL = 200_000_000, 20_000_000, 3, 0.02
rows = [json.loads(l) for l in open(O / 'metrics.jsonl')] if (O / 'metrics.jsonl').exists() else []
ep = [r for r in rows if 'episode/length' in r['metrics']]
if not ep or ep[-1]['step'] < MIN_STEP:
    print(json.dumps(dict(converged=False, reason='min_steps', step=ep[-1]['step'] if ep else 0))); sys.exit(1)
bins = {}
for r in ep:
    m = r['metrics']; L = max(m['episode/length'], 1)
    bins.setdefault(r['step'] // BIN, []).append((L, m.get('episode/err/root_pos', np.nan) / L, abs(m.get('episode/foot_slip', np.nan)) / L))  # 감점이라 음수로 쌓인다
keys = sorted(bins)[:-1]  # 마지막 구간은 아직 차는 중
if len(keys) < WIN + 1:
    print(json.dumps(dict(converged=False, reason='few_bins'))); sys.exit(1)
avg = {k: np.nanmean(np.array(bins[k]), 0) for k in keys}
a, b = avg[keys[-WIN - 1]], avg[keys[-1]]
gain_len = (b[0] - a[0]) / a[0]
gain_pos = (a[1] - b[1]) / a[1] if a[1] > 0 else 0.0
gain_slip = (a[2] - b[2]) / a[2] if np.isfinite(a[2]) and a[2] > 0 else 0.0
conv = gain_len < TOL and gain_pos < TOL and gain_slip < TOL
res = dict(converged=bool(conv), step=ep[-1]['step'], length=float(b[0]), pos_per_step=float(b[1]), slip_per_step=float(b[2]),
           gain_len=float(gain_len), gain_pos=float(gain_pos), gain_slip=float(gain_slip))
with open(O / 'converge_log.jsonl', 'a') as f: f.write(json.dumps(res) + '\n')
print(json.dumps(res)); sys.exit(0 if conv else 1)
