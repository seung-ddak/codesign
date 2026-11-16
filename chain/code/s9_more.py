"""s9 TT 재학습 (2026-11-16 사용자 결정: step 늘리기). 출력 runs/s9_tt_200k/, 기존 s9_tt 는 그대로 둠."""
import s9_tt, s9_report
from common import RUNS
OUT = RUNS / 's9_tt_200k'
s9_tt.OUT = s9_report.OUT = OUT; s9_report.TT = OUT / 'tt'; s9_report.V = OUT / 'videos'
s9_tt.main('--horizon 64 --steps 200000 --batch 32 --norm minmax --clip 1 --sampler ddim')
