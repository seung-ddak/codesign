"""Score every prefix of saved TT trajectories using the original task reward."""
import sys,json,csv,base64
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;repo=root.parents[2];out=root/'smoke_zigzag_joint'
sys.path.insert(0,str(repo/'code/tt_dynamics'))
from task_reward import ZigzagReward
s=json.loads((out/'summary.json').read_text());cfg=s['reward']
assert s['task']=='zigzag'
reward=ZigzagReward(**{k:cfg[k] for k in ('amp','period','half_width','min_height_frac','rewards','penalties')})
series={};table=[]
for tag in ('scale_0','scale_1'):
 with np.load(out/f'{tag}.npz') as z:x=torch.as_tensor(z['pred'])
 assert len(x)==1
 points=[]
 for length in range(1,x.shape[1]+1):
  total,terms=reward(x[:,:length],torch.ones(1,length,dtype=torch.bool))
  weighted={**{f'r_{k}':float(terms[f'r_{k}'][0])*w for k,w in reward.rewards.items()},**{f'p_{k}':-float(terms[f'p_{k}'][0])*w for k,w in reward.penalties.items()}}
  row={'time_s':(length-1)*.02,'reward':float(total[0]),**weighted}
  assert np.isclose(row['reward'],sum(weighted.values()),atol=1e-5)
  points.append(row);table.append({'scale':tag,**row})
 assert np.isclose(points[-1]['reward'],s['runs'][tag][0]['pred_total'],atol=1e-5)
 for k,v in s['runs'][tag][0]['pred_terms'].items():assert np.isclose(float(terms[k][0]),v,atol=1e-5)
 series[tag]=points
(root/'predicted_reward_time.json').write_text(json.dumps({'definition':'Reward evaluated on trajectory prefix [0,t]; penalties are prefix means, not summed per-step returns','dt_s':.02,'series':series},indent=2))
with (root/'predicted_reward_time.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=table[0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(table)
fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
for tag,points in series.items():
 t=[p['time_s'] for p in points];axes[0].plot(t,[p['reward'] for p in points],label=f"{tag}: final {points[-1]['reward']:.4f}")
axes[0].set(title='TT-predicted task reward of trajectory prefix [0,t]',ylabel='Prefix task score');axes[0].legend()
for ax,(tag,points) in zip(axes[1:],series.items()):
 for k,label in [('r_progress','Forward progress'),('p_lane','Lane penalty'),('p_fall','Tilt / low-height penalty'),('p_action_rate','Action-rate penalty')]:ax.plot([p['time_s'] for p in points],[p[k] for p in points],label=label)
 ax.set(title=f'{tag}: weighted contributions (penalties shown negative)',ylabel='Weighted contribution');ax.legend(loc='best',fontsize=9)
for ax in axes:ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.2)
axes[-1].set_xlabel('Trajectory time (s), last state at 1.98 s');fig.tight_layout();fig.savefig(root/'predicted_reward_time.png',dpi=160);plt.close(fig)
print('Saved time plots, CSV and JSON; final rewards match summary.json')
