"""30 Hz 리타겟 (s5, 초록) 대 100 Hz 리타겟 (이번, 흰색) 레퍼런스 나란히 (2026-11-17). 30 Hz 리타겟이 있는 형상만, 처음 10 초. 출력: video/<형상>_<모션>_ref30_vs_ref100.mp4"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; H100 = HERE.parent / 'hz100'
sys.path.insert(0, str(H100 / 'code'))
import video_side as vs
import importlib.util as _u
_sp = _u.spec_from_file_location('rt100_run', HERE / 'code/run.py'); _m = _u.module_from_spec(_sp); _sp.loader.exec_module(_m)
CLIPS, S5 = _m.CLIPS, _m.S5


def q(f, sec=10):
    z = np.load(f); hz = round(1 / float(z['dt'])); x = np.c_[z['fix_base_pos'], z['fix_base_quat'], z['fix_q']][:int(sec * hz) + 1]; return vs.to50(x, hz)


def main():
    out = HERE / 'video'; out.mkdir(exist_ok=True); M = {r['clip'] + r['shape']: r for r in json.load(open(HERE / 'metrics.json'))}
    shapes = json.load(open(S5 / 'assignment.json'))['shapes']
    for c, g in CLIPS.items():
        for s in shapes:
            f30 = S5 / 'references' / s / '00000011' / f'{c}.npz'
            if not f30.exists(): continue
            r = M[c + s]; t = lambda m: f"slide {m['contact_foot_speed_cm_s']:.2f}cm/s tau {m['tau_max']:.0f}"
            vs.render_side([q(f30), q(HERE / 'refs' / s / '00000011' / f'{c}.npz')], shapes[s]['E'], out / f'{s}_{c}_ref30_vs_ref100.mp4',
                           title=f'{s} · {c} ({g.split("__")[0]})', sub='green = 30 Hz retarget (s5)   white = 100 Hz retarget (new)',
                           status=f"30Hz {t(r['hz30'])} | 100Hz {t(r['hz100'])}")
            print('video', s, c, flush=True)


if __name__ == '__main__':
    main()
