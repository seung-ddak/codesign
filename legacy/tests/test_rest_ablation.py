import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/tt_dynamics'))
from rest_ablation import feature_sets, gate


def test_rest_is_additional_and_measured_state_is_preserved():
    e=np.ones((3,13)); s=np.arange(126).reshape(3,42); a=np.zeros((3,192)); g=np.full((3,42),-7.)
    before=s.copy()
    x=feature_sets(e,s,a,g)
    assert np.array_equal(s,before)
    for name in ('s0_A','E_s0_A','g_s0_A','E_g_s0_A'):
        assert np.array_equal(x[name][:,-234:-192],s)
        assert np.array_equal(x[name][:,-192:],a)
    assert np.array_equal(x['E_g_s0_A'][:,13:55],g)


def test_gate_requires_all_thresholds_and_improvement():
    baseline={'rmse':.1}
    good={'rmse':.08,'gain_vs_b0':.2,'sign_agree':.8}
    assert all(gate(good,baseline).values())
    for key,value in [('rmse',.1),('gain_vs_b0',.19),('sign_agree',.79)]:
        assert not all(gate({**good,key:value},baseline).values())



def test_complete_run_writes_results_and_reuses_predictions(tmp_path):
    import json
    from types import SimpleNamespace
    from rest_ablation import run
    e=np.arange(52,dtype=np.float32).reshape(4,13)
    di=np.repeat(np.arange(4),2); gi=np.tile([0,1],4)
    state=np.zeros((8,42),np.float32); action=np.ones((8,192),np.float32)
    y=np.arange(40,dtype=np.float32).reshape(8,5)
    rest=tmp_path/'rest.npz'
    np.savez(rest,designs=np.array(['a','b','c','d']),E=e,rest_s0=np.zeros((4,42)))
    data=tmp_path/'data';data.mkdir()
    args=SimpleNamespace(rest_features=rest,out=tmp_path,data=[data],seeds=1)
    calls=[]
    def fit(xtr,ytr,xte,seed):
        calls.append(1)
        return np.tile(ytr.mean(0),(len(xte),1))
    def metrics(pred,true,b0,mag):
        return {'vx [m/s]':{'rmse':float(np.sqrt(((pred[:,0]-true[:,0])**2).mean())),
                           'gain_vs_b0':0.,'sign_agree':.5}}
    def mean(tr,te):
        return np.stack([y[tr & (gi==g)].mean(0) for g in gi[te]])
    folds={'design':[di==i for i in range(4)]}
    run(args,['a','b','c','d'],e[di],state,action,y,gi,di,folds,fit,metrics,mean)
    assert len(calls)==16
    result=json.loads((tmp_path/'results.json').read_text())
    assert not result['pass'] and result['args']['data']==[str(data)]
    assert (tmp_path/'report.html').exists()
    with np.load(tmp_path/'predictions.npz') as pred:
        assert np.array_equal(pred['s0'],state)
        assert pred['design_E_g_s0_A'].shape==(1,8,5)
    run(args,['a','b','c','d'],e[di],state,action,y,gi,di,folds,fit,metrics,mean)
    assert len(calls)==16
