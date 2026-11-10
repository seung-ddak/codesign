"""최대 속도 직진, 같은 후보 (형상 같음): 왼쪽 open-loop | 오른쪽 MPC (1 초마다 다시 계획). render_task_pair.Run 사용."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import render_task_pair as rp
gtm, rm = rp.gtm, rp.rm
z = np.load(HERE / 'tokens_straight_every50.npz'); res = json.load(open(HERE / 'result_straight_every50.json'))
i = int(np.argmax(res['mpc']['forward_m']))
task = gtm.TASKS['straight']()
runs = []
for key, label in (('open_loop', 'open-loop (one 10 s plan)'), ('mpc', 'MPC (replan every 1 s)')):
    tok = z[key][i].astype(np.float32); e = z['E'][i]
    actual, valid, meta = gtm.replay(e, tok, task=task)
    r = gtm.score_replay(task, actual, valid, meta)
    b = dict(e=e.astype(np.float64), token=tok, saved=(actual, valid), reward=r['reward'], source=key)
    run = rp.Run('straight', b, label); run.sub = f"{label}   forward {meta['forward_m']:.2f} m in 10 s"
    runs.append(run)
out = HERE / 'video' / f'straight_cand{i}_open_vs_mpc.mp4'
stride = max(1, round(1 / gtm.DT / rp.FPS))
def frames():
    for t in range(0, gtm.STEPS + 1, stride):
        yield np.concatenate([r.frame(t) for r in runs], 1)
try:
    rm.encode_frames(out, frames(), width=2 * rp.W, height=rp.H, fps=rp.FPS)
finally:
    for r in runs: r.r.close()
print(out)
