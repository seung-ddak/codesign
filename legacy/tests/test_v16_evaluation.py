import ast
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/tt_dynamics'))
from intervention_metrics import forward_change_sign
from tt_model import condition_masks,schedule


def test_forward_sign_excludes_true_zero_not_failed_predictions():
    r=forward_change_sign([.2,-.1,np.nan,0.,99.],[.1,-.2,.3,-.1,0.])
    assert r['pairs']==5 and r['eligible']==4 and r['correct']==2 and r['rate']==.5


def test_delta_eval_noise_is_paired_even_across_non_aligned_chunks():
    p=ROOT/'runs/pipeline_v1/eval_v12parity/measure_delta_r2.py'
    f=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='predict_mode1')
    ns=dict(torch=torch,condition_masks=condition_masks,S_DIM=42,SEEDS=8)
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),ns)
    x=torch.zeros(6,4,54);mask=torch.ones(6,4,dtype=torch.bool);e=torch.ones(6,13)
    beta,abar=schedule(5,'cpu')
    g=torch.Generator().manual_seed(5)
    noise=torch.randn(8,3,4,54,generator=g).repeat(1,2,1,1)
    def net(x,e,*args):return torch.zeros_like(x),torch.zeros_like(e)
    pred=torch.cat([ns['predict_mode1'](net,x[i:i+2],mask[i:i+2],e[i:i+2],torch.zeros(54),beta,abar,10,noise[:,i:i+2]) for i in (0,2,4)])
    assert torch.equal(pred[:3],pred[3:])
