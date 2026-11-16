# 짧은 다리 단계 점검: 리타겟 → 학습 (slide 와 같은 설정) → 평가 → 영상 → 슬라이드
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
S=check/short_leg; L=$S/logs/after.log
echo "START retarget $(date '+%F %T')" >> $L
[ -f $S/references/L25/00000011/walkBackwardSlow__r00.npz ] || CUDA_VISIBLE_DEVICES="" $PY $S/code/retarget.py > $S/logs/retarget.log 2>&1 || { echo "FAIL retarget" >> $L; exit 1; }
echo "START train $(date '+%F %T')" >> $L
bash $S/code/train_all.sh
echo "START eval $(date '+%F %T')" >> $L
CUDA_VISIBLE_DEVICES="" $PY $S/code/eval_slide.py > $S/logs/eval.log 2>&1 || { echo "FAIL eval" >> $L; exit 1; }
echo "START ckpt eval $(date '+%F %T')" >> $L
CUDA_VISIBLE_DEVICES="" $PY $S/code/eval_ckpts.py > $S/logs/eval_ckpts.log 2>&1 || echo "FAIL ckpt eval" >> $L
echo "START render $(date '+%F %T')" >> $L
source code/env_osmesa.sh
CUDA_VISIBLE_DEVICES="" $PY $S/code/render_slide.py > $S/logs/render.log 2>&1 || { echo "FAIL render" >> $L; exit 1; }
$PY $S/code/build_slide.py >> $L 2>&1
echo "END $(date '+%F %T')" >> $L
