# 2026-11-12 좌·우 이동 (walkLeftSlow, walkRightSlow) 추가: 리타겟 → 학습 (끝난 정책은 건너뜀) → 평가 → 새 모션만 렌더 → 슬라이드
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
S=check/slide; NEW="walkLeftSlow__r00 walkRightSlow__r00"
echo "START lr retarget $(date '+%F %T')" >> $S/logs/after.log
CUDA_VISIBLE_DEVICES="" $PY - <<PYEOF || { echo "FAIL lr retarget" >> $S/logs/after.log; exit 1; }
import shutil, sys
from pathlib import Path
sys.path.insert(0, 'code/motion_tracking'); sys.path.insert(0, '$S/code')
import pd_replay as pd
from shapes import SHAPES
NEW = '$NEW'.split(); ROOT = Path('.').resolve(); SRC = ROOT / 'legacy/artifacts/randomized_reference'
for k, s in SHAPES.items():
    dst = ROOT / '$S/references' / k
    if k == 'E0':
        for m in NEW: shutil.copy(ROOT / f'legacy/runs/pipeline_v1/retargeting/e000/00000011/{m}.npz', dst / '00000011' / f'{m}.npz')
        continue
    pd.retarget_tt(str(SRC), str(dst), only=NEW, choices=['00000011'], uniforms=s['E'], torque_limit=33.5)
    print('retargeted', k, flush=True)
PYEOF
echo "START lr train $(date '+%F %T')" >> $S/logs/after.log
bash $S/code/train_all.sh
grep -q FAIL <(grep -E "walk(Left|Right)" $S/logs/train.log) && echo "WARN lr train FAIL 있음" >> $S/logs/after.log
echo "START lr eval $(date '+%F %T')" >> $S/logs/after.log
CUDA_VISIBLE_DEVICES="" $PY $S/code/eval_slide.py > $S/logs/eval_lr.log 2>&1 || { echo "FAIL lr eval" >> $S/logs/after.log; exit 1; }
source code/env_osmesa.sh
ONLY=$(for K in E0 E1 E2 E3 E4; do for M in $NEW; do printf "%s/%s " $K $M; done; done)
CUDA_VISIBLE_DEVICES="" $PY $S/code/render_slide.py $ONLY > $S/logs/render_lr.log 2>&1 || { echo "FAIL lr render" >> $S/logs/after.log; exit 1; }
$PY $S/code/build_slide.py >> $S/logs/after.log 2>&1
echo "END lr $(date '+%F %T')" >> $S/logs/after.log
