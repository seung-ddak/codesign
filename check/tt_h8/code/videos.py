"""tt_h8 생성 결과 과제 영상 (2026-11-17 사용자 요청: 리포트 없이 영상만). 모델 (h8_e7, h64_e7) × 과제 × scale × {open, mpc} 마다 보상 최고 후보 1 개.
과제 환경 한 장면에 MuJoCo 실제 (초록) · TT 예측 (주황 반투명), chain/code/s10_generate.videos 와 같은 방식. summary.json 이 있는 과제만, 이미 있는 영상은 건너뜀.
출력: videos/<모델>_<과제>_<scale>_<open|mpc>_c<후보>.mp4"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'chain/code')); sys.path.insert(0, str(ROOT / 'code/tt_dynamics'))
import s9_report
from s10_generate import to50
from render_tt import render_task_overlay
from generate_task_mujoco import TASKS as TK


def main():
    out = HERE / 'videos'; out.mkdir(exist_ok=True)
    for model in ['h8_e7', 'h64_e7']:
        for task in ['straight', 'circle', 'zigzag']:
            d = HERE / 'gen' / model / task
            if not (d / 'summary.json').exists(): print('skip', model, task, flush=True); continue
            S = json.load(open(d / 'summary.json'))['results']
            for sc, r in S.items():
                z = np.load(d / f'{sc}.npz')
                for mode, key in [('open', 'open_loop'), ('mpc', 'mpc')]:
                    ok = [x for x in r[key] if x['initial_valid'] and x['reward'] is not None]
                    if not ok: continue
                    b = max(ok, key=lambda x: x['reward']); i = b['candidate']
                    dst = out / f'{model}_{task}_{sc}_{mode}_c{i}.mp4'
                    if dst.exists(): continue
                    act, val, pred = z[f'{mode}_actual'][i], z[f'{mode}_valid'][i], z[f'{mode}_pred'][i]
                    render_task_overlay(s9_report.to_qpos(to50(act, val)), s9_report.to_qpos(to50(pred, np.ones(len(pred), bool))), z['E'][i], TK[task](), dst,
                                        title=f'{model} · {task} · {sc} · {mode} · candidate {i} · reward {b["reward"]:.2f}', fell_t=b['valid_seconds'] if b['fell'] else None)
                    print('video', dst.name, flush=True)


if __name__ == '__main__':
    main()
