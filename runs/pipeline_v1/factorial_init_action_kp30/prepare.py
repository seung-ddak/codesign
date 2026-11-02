"""Freeze paired e000 action windows before looking at target-shape physics outcomes."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
STEPS=50
WINDOWS=4


def choose_windows(z, steps=50, count=4):
    selected=[]
    for c,motion in enumerate(z['motions']):
        options=[]
        for row in np.flatnonzero(z['clip'][:,0]==c):
            candidates=[]
            for t in range(z['valid'].shape[1]-steps+1):
                sl=slice(t,t+steps)
                frames=z['ref_frame'][row,sl]
                if (z['valid'][row,sl].all() and (z['clip'][row,sl]==c).all()
                    and not z['switched'][row,sl].any() and np.all(np.diff(frames)==1)):
                    candidates.append(t)
            if len(candidates)>=count:options.append((len(candidates),int(row),candidates))
        assert options,f'No paired windows for {motion}'
        _,row,candidates=max(options,key=lambda x:(x[0],-x[1]))
        for k in np.linspace(0,len(candidates)-1,count).astype(int):
            t=candidates[k]
            frame=int(z['ref_frame'][row,t]-z['clip_start'][c])
            selected.append(dict(motion=str(motion),clip=int(c),source_episode=row,source_t=t,start_frame=frame))
    return selected


def main():
    source=ROOT/'runs/pipeline_v1/wide16_action/rollouts/e000/primitive_seed0.npz'
    config=sorted((ROOT/'runs/pipeline_v1/experts_wide16/walkForwardSlow__r00').glob('wandb/offline-run-*/files/config.pkl'))[-1]
    with config.open('rb') as f:cfg=pickle.load(f)
    choice=cfg.get('choice')
    with np.load(source,allow_pickle=False) as z:
        rows=choose_windows(z,STEPS,WINDOWS)
        ppo=np.stack([z['target_q'][r['source_episode'],r['source_t']:r['source_t']+STEPS] for r in rows])
        off=z['action_offset'];scale=float(z['action_scale']);lo,hi=z['clip_actions']
        motions=z['motions'].copy()
    reference=[];ref_files=[]
    for r in rows:
        path=ROOT/'runs/pipeline_v1/retargeting/e000'/choice/(r['motion']+'.npz')
        with np.load(path) as z:
            # Existing PD collector uses the next reference frame as the command.
            reference.append(z['fix_q'][r['start_frame']+1:r['start_frame']+STEPS+1])
        ref_files.append(path)
    raw=np.stack([np.stack(reference),ppo]).astype(np.float32)
    target=np.clip(raw,off+lo*scale,off+hi*scale)
    assert target.shape==(2,len(rows),STEPS,12) and np.isfinite(target).all()
    designs=sorted(p.parent.name for p in (ROOT/'runs/pipeline_v1/pd_rollouts_kp200').glob('*/primitive_seed0.npz'))
    e=json.loads((ROOT/'runs/pipeline_v1/retargeting/designs.json').read_text())
    np.savez_compressed(OUT/'action_bank.npz',target_q=target,raw_target_q=raw,motions=motions,
                        clip=np.array([r['clip'] for r in rows]),start_frame=np.array([r['start_frame'] for r in rows]),
                        source_episode=np.array([r['source_episode'] for r in rows]),source_t=np.array([r['source_t'] for r in rows]),
                        E=np.stack([e[d] for d in designs]).astype(np.float32),designs=np.array(designs))
    files=set(ref_files)|{source,config,ROOT/'runs/pipeline_v1/retargeting/designs.json',Path(__file__),ROOT/'code/tt_dynamics/regression_gate.py',ROOT/'code/motion_tracking/track_env.py',ROOT/'code/motion_tracking/pd_replay.py',ROOT/'code/tt_dynamics/collect_e_swap.py'}
    for d in designs:
        for m in motions:files.add(ROOT/'runs/pipeline_v1/retargeting'/d/choice/(str(m)+'.npz'))
    conditions={'source_design':'e000','designs':designs,'groups':len(rows),'motions':len(motions),'steps':STEPS,'dt':.02,
                'choice':choice,'kp':30,'home_settle_s':1.,'reference_initialization':'shape-specific retargeted qpos/qvel at same source clip/frame; no reset noise',
                'actions':'same absolute target_q bank across all shapes and initializations; PPO actions from existing kp30 source, replayed at matching kp30 (parity rerun of factorial_init_action, remind 8-45 follow-up)',
                'arms':['reference_reference','reference_ppo','rest_reference','rest_ppo'],
                'primary_population':'same group intersection surviving all 4 arms and 16 designs through full 1 second; conditional-on-survival, not fall prediction',
                'minimum_common_groups':24,'minimum_common_motions':8,'minimum_effect_cells':32,
                'models':['E_s0_A','s0_A'],'fold':'4-fold held-out design, sorted design index mod4; seed 0/1/2 predictions averaged',
                'hidden':256,'epochs':2000,'wd':.001,'seeds':3,'a_steps':16,
                'gate':{'vx_gain_vs_B0_min':.2,'vx_effect_sign_min':.8,'effect_threshold_m_s':.03},
                'factor_contrasts':'difference in normalized morphology gain, rest minus reference averaged across action origins; PPO minus reference averaged across initializations; interaction also reported. Descriptive 16-shape comparison, no population causal generalization claim.',
                'secondary_population':'arm-specific all-16-shape surviving groups; explicitly secondary, never replaces common-population gate',
                'resources':'serial stages; GPU JAX40% / Torch30%, CPU2 threads, temp/cache on /home; watchdog stops own worker at RAM<4GiB, GPUfree<1GiB, rootfree<1GiB, homefree<8GiB twice',
                'clipped_action_values':int((raw!=target).sum()),'windows':rows,
                'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
                'config':str(config.relative_to(ROOT))}
    for p in (OUT/'collect.py',OUT/'evaluate.py',OUT/'run.py'):
        conditions['sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    conditions['sha256'][str((OUT/'action_bank.npz').relative_to(ROOT))]=hashlib.sha256((OUT/'action_bank.npz').read_bytes()).hexdigest()
    (OUT/'conditions.json').write_text(json.dumps(conditions,indent=2,ensure_ascii=False))
    print('Frozen',len(rows),'paired windows across',len(motions),'motions and',len(designs),'designs',flush=True)


if __name__=='__main__':main()
