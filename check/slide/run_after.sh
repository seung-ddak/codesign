cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
S=check/slide
until grep -q "ALL DONE" $S/train.log 2>/dev/null; do sleep 60; done
echo "START eval $(date '+%F %T')" >> $S/after.log
CUDA_VISIBLE_DEVICES="" $PY $S/eval_slide.py > $S/eval.log 2>&1 || { echo "FAIL eval" >> $S/after.log; exit 1; }
echo "START render $(date '+%F %T')" >> $S/after.log
source code/env_osmesa.sh
CUDA_VISIBLE_DEVICES="" $PY $S/render_slide.py > $S/render.log 2>&1 || { echo "FAIL render" >> $S/after.log; exit 1; }
$PY $S/build_slide.py >> $S/after.log 2>&1
echo "END $(date '+%F %T')" >> $S/after.log
