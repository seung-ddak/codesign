"""중간 체크포인트 평가: runs/<L>/<motion>/ 의 모든 체크포인트를 eval_slide.evaluate 와 같은 조건 (외란 없음, 모션 처음부터) 으로 평가.
출력: results/ckpts.json, video/ckpt_curves.png (모션별: 생존 시간 · 관절 RMSE · 학습 중 eval 보상, x = 학습 step)."""
import glob, json, os, re, sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; sys.path.insert(0, str(CODE_DIR))
import numpy as np
import eval_slide as ev
from shapes import SHAPES, MOTIONS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus'] = False

rows = []
for K in SHAPES:
    for M in MOTIONS:
        run = HERE / 'runs' / K / M
        cks = sorted(glob.glob(str(run / 'wandb/offline-run-*/files/checkpoints/*')), key=lambda p: int(os.path.basename(p)))
        rew = [float(x) for x in re.findall(r'^eval/episode/reward/mean: (\S+)', (run / 'train.log').read_text(), re.M)] if (run / 'train.log').exists() else []
        for j, ck in enumerate(cks):
            r = ev.evaluate(K, M, ck=ck, save=False)
            r['step'] = int(os.path.basename(ck)); r['train_eval_reward'] = rew[j + 1] if j + 1 < len(rew) else None
            rows.append(r)
            print(K, M, r['step'], 'surv', r['survived'], r['seconds'], f"joint {r['joint_rmse_deg']:.1f}", flush=True)
(HERE / 'results').mkdir(exist_ok=True)
json.dump(rows, open(HERE / 'results/ckpts.json', 'w'), indent=1, ensure_ascii=False)

cols = plt.cm.viridis(np.linspace(0, .9, len(SHAPES)))
fig, ax = plt.subplots(len(MOTIONS), 3, figsize=(15, 4 * len(MOTIONS)), squeeze=False)
for i, M in enumerate(MOTIONS):
    for c, K in zip(cols, SHAPES):
        rr = sorted([r for r in rows if r['shape'] == K and r['motion'] == M], key=lambda r: r['step'])
        if not rr: continue
        x = np.array([r['step'] for r in rr]) / 1e6
        ax[i, 0].plot(x, [r['seconds'] for r in rr], 'o-', color=c, label=K)
        ax[i, 1].plot(x, [r['joint_rmse_deg'] for r in rr], 'o-', color=c, label=K)
        ax[i, 2].plot(x, [r['train_eval_reward'] if r['train_eval_reward'] is not None else np.nan for r in rr], 'o-', color=c, label=K)
    name = M.replace('__r00', '')
    ax[i, 0].set_title(f'{name}: 생존 시간 (s)'); ax[i, 1].set_title(f'{name}: 관절 RMSE (°)'); ax[i, 2].set_title(f'{name}: 학습 중 eval 보상')
    ax[i, 1].set_yscale('log')
    for a in ax[i]: a.set_xlabel('학습 step (M)'); a.grid(alpha=.3)
    ax[i, 0].legend()
fig.tight_layout(); (HERE / 'video').mkdir(exist_ok=True); fig.savefig(HERE / 'video/ckpt_curves.png', dpi=110)
print(HERE / 'video/ckpt_curves.png')
