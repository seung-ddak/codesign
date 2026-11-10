"""A1 레퍼런스 (legacy/artifacts/randomized_reference) 를 형상 E0~E4 로 리타겟. E0 는 기존 e000 리타겟을 복사."""
import shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(HERE))
import pd_replay as pd
from shapes import SHAPES, MOTIONS
SRC = ROOT / 'legacy/artifacts/randomized_reference'
for k, s in SHAPES.items():
    dst = HERE / 'references' / k
    if k == 'E0':
        (dst / '00000011').mkdir(parents=True, exist_ok=True)
        for m in MOTIONS: shutil.copy(ROOT / f'legacy/runs/pipeline_v1/retargeting/e000/00000011/{m}.npz', dst / '00000011' / f'{m}.npz')
        continue
    pd.retarget_tt(str(SRC), str(dst), only=MOTIONS, choices=['00000011'], uniforms=s['E'], torque_limit=33.5)
    print('retargeted', k, flush=True)
