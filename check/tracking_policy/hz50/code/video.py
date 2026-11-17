"""30 Hz 대 50 Hz 정책 겹친 영상 (2026-11-17 사용자 요청). 같은 형상·모션·평가 환경 (evalpol, 외란·잡음 없음, 처음 10 초).
30 Hz = s6c50 20M 체크포인트 (원래 색, 불투명), 50 Hz = hz50 20M (반투명 파랑). 30 Hz qpos 는 50 Hz 로 시간 보간해 맞춘다.
출력: video/<형상>_<모션>_30vs50.mp4"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())
sys.path.insert(0, str(ROOT / 'chain/code')); sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(ROOT / 'code/viz'))
import evalpol, s6c_calm as s6c
from s7_eval import ckpts
from render_tt import render_overlay
from run import TARGETS, STEPS
SHAPES = json.load(open(ROOT / 'chain/runs/s5_retarget/assignment.json'))['shapes']; PICK = ['G_l+00_b+00', 'G_l-20_b+20']


def to50(q, hz):
    t = np.arange(len(q)) / hz; t2 = np.arange(0, t[-1] + 1e-9, 1 / 50)
    out = np.stack([np.interp(t2, t, q[:, j]) for j in range(q.shape[1])], 1); out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True); return out


def main():
    out = HERE / 'video'; out.mkdir(exist_ok=True)
    for g, c, why in TARGETS:
        ck30 = min(ckpts(ROOT / 'chain/runs/s6c50_calm' / g / c), key=lambda k: abs(int(Path(k).name) - STEPS)); ck50 = ckpts(HERE / 'runs' / g / c)[-1]
        for s in PICK:
            E = SHAPES[s]['E']
            r30 = evalpol.evaluate(ck30, E, s6c.REFS / s / '00000011', c, multi=True, keep_qpos=True, max_seconds=10)
            r50 = evalpol.evaluate(ck50, E, HERE / 'refs' / s / '00000011', c, multi=True, keep_qpos=True, max_seconds=10)
            q30, q50 = to50(np.asarray(r30['qpos']), 30), np.asarray(r50['qpos'])
            tag = lambda r: f"joint {r['joint_rmse_deg']:.1f} deg, root {r['root_rmse_cm']:.1f} cm{'' if r['survived'] else ', FELL'}"
            render_overlay(q30, q50, E, out / f'{s}_{c}_30vs50.mp4', title=f'{s} · {c} ({why.replace("30Hz ", "30 Hz ").replace("실패", "fail").replace("통과", "pass")})',
                           sub=f'solid = 30 Hz ({tag(r30)})  /  blue = 50 Hz ({tag(r50)})', status_fn=lambda t: f't = {t * .02:4.1f} s', color=(0.2, 0.5, 1.0, 0.45))
            print(s, c, '30Hz', tag(r30), '| 50Hz', tag(r50), flush=True)


if __name__ == '__main__':
    main()
