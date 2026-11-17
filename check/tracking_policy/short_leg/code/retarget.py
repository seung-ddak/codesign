"""A1 레퍼런스 (legacy/artifacts/randomized_reference) 를 짧은 다리 형상 L05~L25 로 리타겟."""
import shutil, sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관
sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(CODE_DIR))
import pd_replay as pd
from shapes import SHAPES, MOTIONS
SRC = ROOT / 'legacy/artifacts/randomized_reference'
for k, s in SHAPES.items():
    dst = HERE / 'references' / k
    if False:  # 모든 형상을 새로 리타겟
        (dst / '00000011').mkdir(parents=True, exist_ok=True)
        for m in MOTIONS: shutil.copy(ROOT / f'legacy/runs/pipeline_v1/retargeting/e000/00000011/{m}.npz', dst / '00000011' / f'{m}.npz')
        continue
    pd.retarget_tt(str(SRC), str(dst), only=MOTIONS, choices=['00000011'], uniforms=s['E'], torque_limit=33.5)
    print('retargeted', k, flush=True)
