"""Controlled 100k-step retraining: try5 settings, corrected free-mode mask only."""
from pathlib import Path
import run_cgr

if __name__ == '__main__':
    run_cgr.OUT = Path('runs/pipeline_v1/tt/h32/try8')
    run_cgr.BRANCH = 'fix/free-conditioning'
    index = run_cgr.FLAGS.index('--modes')
    run_cgr.FLAGS[index + 1:index + 4] = ['0', '1', '2', '3']
    run_cgr.main()
