"""same_scene.json 의 모션을 A1 원본 (레퍼런스 a1_qpos 재생) 만 따로 녹화. 출력: runs/s7_eval/videos/same_scene/a1_only/<묶음>_<모션>.mp4"""
import json, sys
import numpy as np
from common import ROOT, RUNS, log
from same_scene import OUT, refdir
sys.path.insert(0, str(ROOT / 'legacy/runs/v100_mujoco_10k/mesh_render'))
import render_three as r3
rm = r3.rm; D = OUT / 'a1_only'; D.mkdir(exist_ok=True)
for x in json.load(open(OUT / 'same_scene.json')):
    z = np.load(refdir(x['shape'], x['clip']) / f"{x['clip']}.npz"); dt = float(z['dt']); a = z['a1_qpos'].astype(float).copy(); a[:, :2] -= a[0, :2]
    gn = x['group'].replace('trot', '').replace('Slow__r00', ''); p = r3.Panel(r3.a1_model()); out = D / f"{gn}_{x['clip']}.mp4"

    def frames():
        for t in np.arange(0, len(a) * dt, 1 / r3.FPS):
            k = min(int(round(t / dt)), len(a) - 1)
            yield rm.label(p.frame(a[k], a[:k + 1, :2], a[k, :2]), f"{x['clip']} · {gn}", 'A1: original CALM motion', f't = {t:4.1f} s   moved {np.linalg.norm(a[k, :2]):.2f} m', False)
    try: rm.encode_frames(out, frames(), width=r3.W, height=r3.H, fps=r3.FPS)
    finally: p.r.close()
    log(f'A1 영상 {out.name}')
