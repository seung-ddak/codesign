cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
S=check/slide
until grep -q "ALL DONE" $S/logs/train.log 2>/dev/null; do sleep 60; done
echo "START eval $(date '+%F %T')" >> $S/logs/after.log
CUDA_VISIBLE_DEVICES="" $PY $S/code/eval_slide.py > $S/logs/eval.log 2>&1 || { echo "FAIL eval" >> $S/logs/after.log; exit 1; }
echo "START render $(date '+%F %T')" >> $S/logs/after.log
source code/env_osmesa.sh
CUDA_VISIBLE_DEVICES="" $PY $S/code/render_slide.py > $S/logs/render.log 2>&1 || { echo "FAIL render" >> $S/logs/after.log; exit 1; }
$PY $S/code/build_slide.py >> $S/logs/after.log 2>&1
echo "END $(date '+%F %T')" >> $S/logs/after.log
