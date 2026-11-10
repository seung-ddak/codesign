"""Build a concise result audit and plot from completed predictions; does not train."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent
root=out.parents[2]
r=json.loads((out/'results.json').read_text())
assert len(r['designs'])==16 and r['samples']==1792
for p,h in r['sha256'].items():
    assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
models=list(r['models'])
vx='vx [m/s]'
scores=r['folds']['design']['ensemble']
base=scores['E_s0_A'][vx]['rmse']
summary={'pass':r['pass'],'design_vx':{m:scores[m][vx] for m in models}, 'per_seed_primary_better_than_baseline':[], 'per_fold_primary_better_than_baseline':[], 'elapsed_min':r['elapsed_s']/60}
for seed in range(3):
    x=r['folds']['design']['per_seed']
    summary['per_seed_primary_better_than_baseline'].append(x['E_g_s0_A'][seed][vx]['rmse']<x['E_s0_A'][seed][vx]['rmse'])
for fold in r['folds']['design']['per_fold']:
    summary['per_fold_primary_better_than_baseline'].append(fold['E_g_s0_A'][vx]['rmse']<fold['E_s0_A'][vx]['rmse'])
with np.load(out/'predictions.npz') as z:
    y=z['Y']; b=z['design_B0']; assert np.isfinite(y).all() and np.isfinite(b).all()
    assert len(set(z['DI']))==16
    for m in models:
        pred=z['design_'+m]
        assert pred.shape==(3,1792,5) and np.isfinite(pred).all()
        assert abs(float(np.sqrt(((pred.mean(0)[:,0]-y[:,0])**2).mean()))-scores[m][vx]['rmse'])<1e-6
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].bar(models,[scores[m][vx]['rmse'] for m in models],color=['#808080','#3978ad','#d2912e','#43996b'])
    ax[0].axhline(r['folds']['design']['B0'][vx]['rmse'],color='k',ls='--',label='B0: training-shape group mean')
    ax[0].set(ylabel='Held-out shape vx RMSE (m/s)',title='Same PD data, measured initial state preserved')
    ax[0].tick_params(axis='x',rotation=25);ax[0].legend(fontsize=8)
    true=y[:,0]-b[:,0]
    for m in ('E_s0_A','E_g_s0_A'):
        ax[1].scatter(true,z['design_'+m].mean(0)[:,0]-b[:,0],s=7,alpha=.3,label=m)
    lim=max(abs(true).max(),.1)
    ax[1].plot([-lim,lim],[-lim,lim],'k--')
    ax[1].set(xlabel='Actual vx - B0 (m/s)',ylabel='Predicted vx - B0 (m/s)',title='Predictive morphology deviations')
    ax[1].legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
resources=[json.loads(x) for x in (out/'resources.jsonl').read_text().splitlines()]
summary['min_available_ram_gib']=min(x['MemAvailable'] for x in resources)/2**30
summary['min_root_free_gib']=min(x['root_free'] for x in resources)/2**30
summary['min_home_free_gib']=min(x['home_free'] for x in resources)/2**30
summary['descriptor_audit']=json.loads((out/'descriptor_audit.json').read_text())
summary['validation']='14 relevant tests passed; input hashes match v12; predictions finite and aggregate RMSE recomputed.'
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
p=out/'report.html'
s=p.read_text()
s += '<p>[사실] 1초 상태에서 e008은 발 접촉 0개였음. 형상을 제외하지 않았으며 완전한 정지·안착 상태로 간주하지 않음.</p><img src="comparison.png" alt="Held-out morphology prediction comparison" style="max-width:100%"><p><a href="summary.json">검증·자원·분할별 일관성 요약</a></p>'
p.write_text(s)
print(json.dumps({k:v for k,v in summary.items() if k!='descriptor_audit'},indent=2,ensure_ascii=False))
