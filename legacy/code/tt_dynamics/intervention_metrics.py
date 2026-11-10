"""Metrics for physically paired changes, distinct from task-shape ranking proxies."""
import numpy as np


def forward_change_sign(pred_delta,true_delta,tolerance=1e-6):
    p=np.asarray(pred_delta);t=np.asarray(true_delta)
    assert p.shape==t.shape
    eligible=np.isfinite(t)&(np.abs(t)>tolerance)
    correct=eligible&np.isfinite(p)&(np.sign(p)==np.sign(t))
    count=int(eligible.sum())
    return {'rate':float(correct.sum()/count) if count else None,'correct':int(correct.sum()),
            'eligible':count,'pairs':int(t.size),'true_zero_tolerance_m':tolerance,
            'definition':'sign of paired final-window forward displacement change, with each member own E+s0 and identical physical A'}
