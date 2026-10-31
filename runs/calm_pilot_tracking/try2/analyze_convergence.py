"""Plot only recorded PPO epoch means and evaluation points; no interpolation."""
import os,json,re
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-calm')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(__file__).resolve().parent
blocks=re.split(r'-{5,}(\d+)-{5,}',(p/'train.log').read_text());rows=[]
for i in range(1,len(blocks),2):
 values={k:float(v) for k,v in re.findall(r'^([A-Za-z0-9_/]+): ([0-9.eE+\-]+)$',blocks[i+1],re.M)}
 rows.append(dict(step=int(blocks[i]),**values))
assert [r['step'] for r in rows]==[0,2623488,5246976]
loss=[r for r in rows if 'training/total_loss' in r];assert len(loss)==2
for r in loss:assert np.isclose(r['training/total_loss'],r['training/policy_loss']+r['training/v_loss']+r['training/entropy_loss'],atol=1e-6)
(p/'convergence_data.json').write_text(json.dumps(dict(points=rows,loss_points=2,loss_aggregation='mean over each 2,623,488-step training epoch',convergence='undetermined: only two loss averages and three internal evaluations'),indent=2))
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
items=[('training/total_loss','PPO total loss','loss',loss),('training/v_loss','Value loss','loss',loss),('training/policy_loss','Policy loss','loss',loss),('eval/episode/reward/mean','Internal evaluation reward','reward',rows),('eval/summary/pos_err/mean','Internal body position error','cm',rows),('training/kl_mean','Policy update KL','KL',loss)]
for ax,(key,title,unit,points) in zip(axes.flat,items):
 x=[r['step']/1e6 for r in points];y=[r[key]*(100 if unit=='cm' else 1) for r in points]
 ax.plot(x,y,'o--',color='#247ab5');ax.set(title=title,xlabel='Training environment steps (millions)',ylabel=unit);ax.grid(alpha=.2)
 for a,b in zip(x,y):ax.annotate(f'{b:.4g}',(a,b),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
 ax.margins(x=.15,y=.25)
 if points is loss:ax.text(.03,.04,'2 epoch averages only',transform=ax.transAxes,fontsize=9,color='#9b3d2b')
fig.suptitle('Recorded PPO history: sparse points do not establish convergence',fontsize=15)
fig.savefig(p/'learning_curves.png',dpi=180)
print('Recorded losses:',[(r['step'],r['training/total_loss'],r['training/v_loss']) for r in loss])
