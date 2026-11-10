"""Validated original-TT -> guidance -> PPO -> evaluation, one stage at a time."""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from native_tt_data import ROOT,TT,prepare,save_json,atomic_torch,digest,reference_candidate
from native_tt_train import configure,model_and_augmentation,normalization,attach_normalization,train,load_model

TASKS=('straight','circle','zigzag')
STAGES=('preflight','prepare','train','gradient','residual','evaluate')


def preflight(source,out,device):
    from native_tt_data import NativeDataset
    from native_tt_eval import sample,physical_replay
    from native_tt_residual import Controller,train_policy
    save_json(out/'preflight.json',dict(passed=False,status='running',updated=time.time()))
    torch.manual_seed(0);np.random.seed(0)
    prepare(source,out/'data',limit=1)
    pilot=out/'pilot_data';pilot.mkdir(exist_ok=True)
    name='e0000_variant_0'
    if not (pilot/name).exists():(pilot/name).symlink_to((out/'data'/name).resolve(),target_is_directory=True)
    save_json(pilot/'manifest.json',dict(shards=[name]))
    cfg=configure(out/'tt_base');model,aug=model_and_augmentation(cfg)
    norms=normalization(pilot,cfg,model,pilot/'normalization.pt')
    attach_normalization(model,norms)
    dataset=NativeDataset(pilot,cfg,'train')
    raw=torch.utils.data.default_collate([dataset[0]])
    adapter=model.tasks.hardware_gen.adapters['ctrl']
    encoded=adapter.modality.to_tensor(raw,encoders=adapter.attr_encoders)
    decoded=adapter.modality.from_tensor(encoded,decoders=adapter.attr_decoders)
    mask=~raw['ctrl/mask'].expand_as(raw['ctrl/target_qpos'])
    torch.testing.assert_close(decoded['ctrl/target_qpos'][mask],raw['ctrl/target_qpos'][mask],atol=1e-5,rtol=1e-5)
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA not available; do not start long CPU training')
        torch.cuda.set_per_process_memory_fraction(.75)
    model=model.to(device)
    bf16=device=='cuda' and torch.cuda.is_bf16_supported()
    if not bf16:model.tasks.hardware_gen.use_flash_attn=False
    microbatch=1
    timings=[]
    # Calibrate real native forward/backward passes before
    # creating an optimizer or a long-running training process.
    for count in [1,2,4,8,16,32]:
        batch=torch.utils.data.default_collate([dataset[i%len(dataset)] for i in range(count)])
        batch=aug({k:v.to(device) for k,v in batch.items()})
        model.zero_grad(set_to_none=True);started=time.time()
        try:
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=bf16):
                loss=model([(['hardware_gen'],batch)],is_eval=False)['loss']
            if not torch.isfinite(loss):raise FloatingPointError('Native loss')
            loss.backward()
            if device=='cuda':torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache();break
        peak=torch.cuda.max_memory_allocated() if device=='cuda' else 0
        if peak>9*1024**3:break
        microbatch=count;timings.append(dict(batch=count,seconds=time.time()-started,peak_bytes=peak))
    model.zero_grad(set_to_none=True);model.eval().requires_grad_(False)
    if device=='cuda':torch.cuda.empty_cache()
    baseline=sample(model,cfg,'straight',1,123)
    controller=Controller(model.tasks.hardware_gen,device)
    zero=sample(model,cfg,'straight',1,123,controller=controller)
    if controller.steps!=100:raise AssertionError('Sampler was not called 100 times')
    for key in baseline:np.testing.assert_allclose(baseline[key],zero[key],atol=1e-5,rtol=1e-5)
    guided=sample(model,cfg,'straight',1,123,scale=.1)
    if not all(np.isfinite(v).all() for v in guided.values()):raise FloatingPointError('Gradient guidance')
    with torch.no_grad():controller.policy.actor.heads['ctrl'][-1].bias.fill_(.5)
    changed=sample(model,cfg,'straight',1,123,controller=controller)
    if np.allclose(changed['ctrl/target_qpos'],zero['ctrl/target_qpos']):
        raise AssertionError('Residual did not change physical motor targets')
    with np.load(out/'data'/name/'hardware.npz') as z:hardware={k:z[k] for k in z.files}
    fixed=sample(model,cfg,'straight',1,123,hardware=hardware)
    for key,value in hardware.items():
        if key in fixed:
            np.testing.assert_allclose(fixed[key][0,:len(value)].reshape(value.shape),value,atol=1e-4,rtol=1e-4)
    candidate=reference_candidate(out/'data'/name)
    row,arrays=physical_replay(candidate,'straight')
    repeated,_=physical_replay(candidate,'straight')
    if row['invalid'] or row['steps']<2:raise AssertionError(row)
    if row['return_value']!=repeated['return_value']:raise AssertionError('Physics replay mismatch')
    np.testing.assert_allclose(arrays['rewards'].sum(),row['return_value'],atol=1e-9)
    pilot_base=out/'pilot_base.pt'
    if pilot_base.exists():
        model.load_state_dict(torch.load(pilot_base,map_location=device,weights_only=False)['model'])
    else:atomic_torch(pilot_base,dict(model=model.state_dict()))
    # Exercise the real 16-candidate × 100-step PPO storage/update as well as
    # the small known-answer test. This policy is never reused for actual RL.
    train_policy(model,cfg,'straight',0,pilot_base,out/'pilot_residual',stop_after=1)
    subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_native_tt_chain.py')],check=True)
    report=dict(passed=True,microbatch=microbatch,calibration=timings,
        zero_residual_matches=True,denoising_steps=100,motor_targets_changed=True,
        full_policy_update_passed=True,
        fixed_hardware_unchanged=True,
        physical_replay=row,source_plan_sha256=digest(source/'plan.json'),
        approximate_seconds_per_optimizer_update=timings[-1]['seconds']*1024/timings[-1]['batch'],
        warning='Calibration excludes full-data I/O, validation and checkpoint costs; not a completion promise.')
    resume=out/'resume_check.json'
    if not resume.exists() or not json.loads(resume.read_text())['passed']:
        raise RuntimeError('Actual effective-batch interruption/resume test must pass first')
    report['resume_check']=json.loads(resume.read_text())
    save_json(out/'preflight.json',report)
    print(json.dumps(report),flush=True)


def gradient(out,device):
    from native_tt_eval import evaluate
    model,cfg,_=load_model(out/'tt_base',device)
    selected={}
    for task in TASKS:
        scores={}
        for scale in [0.,.1,1.,10.]:
            rows=[]
            for seed in range(3):
                noises=range(1000000+seed*100,1000016+seed*100)
                rows+=evaluate(model,cfg,task,noises,out/'gradient'/task/f'scale_{scale}'/f'seed_{seed}',scale=scale)
            scores[str(scale)]=float(np.mean([r['return_value'] for r in rows]))
        selected[task]=dict(scale=float(max(scores,key=scores.get)),development_returns=scores)
        save_json(out/'gradient/selected.json',selected)


def residual(out,device):
    from native_tt_residual import train_policy
    model,cfg,_=load_model(out/'tt_base',device)
    for task in TASKS:
        for seed in range(3):
            folder=out/'residual'/task/f'seed_{seed}'
            started=time.time()
            train_policy(model,cfg,task,seed,out/'tt_base/checkpoint.pt',folder,stop_after=5)
            save_json(folder/'connection.json',dict(passed=True,first_five_update_seconds=time.time()-started))
            train_policy(model,cfg,task,seed,out/'tt_base/checkpoint.pt',folder)


def final_evaluation(source,out,device):
    from native_tt_eval import evaluate
    from native_tt_residual import Controller
    from native_tt_data import native_env
    model,cfg,_=load_model(out/'tt_base',device)
    selected=json.loads((out/'gradient/selected.json').read_text());summaries=[]
    # Held-out hardware is first loaded here, after all model/scale selection.
    plan=json.loads((source/'plan.json').read_text())
    for task in TASKS:
        for seed in range(3):
            controller=Controller(model.tasks.hardware_gen,device)
            checkpoint=torch.load(out/'residual'/task/f'seed_{seed}'/'policy.pt',map_location=device,weights_only=False)
            controller.policy.load_state_dict(checkpoint['policy'])
            noises=range(2000000+seed*100,2000064+seed*100)
            for method,scale,control in [('unguided',0.,None),('gradient',selected[task]['scale'],None),('rl',0.,controller)]:
                folder=out/'evaluation'/task/f'seed_{seed}'/method
                rows=evaluate(model,cfg,task,noises,folder,scale=scale,controller=control)
                summaries.append(dict(task=task,seed=seed,method=method,mean_return=float(np.mean([r['return_value'] for r in rows])),
                    survival=float(np.mean([r['survived'] for r in rows])),invalid=float(np.mean([r['invalid'] for r in rows]))))
            for shape in plan['eval_designs']:
                _,_,hardware=native_env(shape['E'])
                for method,scale,control in [('unguided',0.,None),('gradient',selected[task]['scale'],None),('rl',0.,controller)]:
                    folder=out/'evaluation/heldout'/shape['design']/task/f'seed_{seed}'/method
                    evaluate(model,cfg,task,[2000000+seed*100],folder,scale=scale,controller=control,hardware=hardware)
    save_json(out/'evaluation/comparison.json',summaries)
    body='<meta charset="utf-8"><title>Native TT guidance comparison</title><h1>Actual MuJoCo results</h1>'
    body+='<p>Same frozen EMA TT; all final candidates included. Invalid starts are failures. Held-out hardware results are separate.</p>'
    body+='<table><tr><th>Task</th><th>Seed</th><th>Method</th><th>Return</th><th>Survival</th><th>Invalid</th></tr>'
    for row in summaries:
        body+='<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in ['task','seed','method','mean_return','survival','invalid'])+'</tr>'
    (out/'evaluation/report.html').write_text(body+'</table>')


def completed(stage,out):
    if stage=='preflight':
        p=out/'preflight.json'
        return (p.exists() and json.loads(p.read_text()).get('passed')
                and bool(json.loads(p.read_text()).get('resume_check',{}).get('passed'))
                and json.loads(p.read_text()).get('full_policy_update_passed',False))
    if stage=='prepare':
        p=out/'data/manifest.json'
        if not p.exists():return False
        data=json.loads(p.read_text())
        if len(data['shards'])!=400 or data['original_sources']!=10000 or data['episodes']!=40000:return False
        return all((out/'data'/name/'complete.json').is_file() for name in data['shards'])
    if stage=='train':
        p=out/'tt_base/checkpoint.pt'
        if not p.exists():return False
        ck=torch.load(p,map_location='cpu',weights_only=False)
        return ck['step']==20480 and np.isfinite(ck['ema_val_loss'])
    if stage=='gradient':
        p=out/'gradient/selected.json'
        return p.exists() and set(json.loads(p.read_text()))==set(TASKS)
    if stage=='residual':
        for task in TASKS:
            for seed in range(3):
                p=out/'residual'/task/f'seed_{seed}'/'training_state.pt'
                if not p.exists():return False
                if torch.load(p,map_location='cpu',weights_only=False)['update']!=1000:return False
        return True
    return (out/'evaluation/report.html').is_file()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('run',)+STAGES)
    p.add_argument('--source',type=Path,default=ROOT/'runs/local_mujoco_10k_20261109')
    p.add_argument('--out',type=Path,default=ROOT/'runs/tt_gradient_rl_chain_v1')
    p.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    args=p.parse_args();source=args.source.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT);torch.set_num_threads(2)
    if args.stage in ['preflight','train','gradient','residual','evaluate'] and args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_per_process_memory_fraction(.75)
    if args.stage!='run':
        import fcntl
        worker_lock=(out/'worker.lock').open('a');fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=='run':
        import fcntl
        lock=(out/'chain.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract=dict(source_plan_sha256=digest(source/'plan.json'),
            code={f.name:digest(f) for f in Path(__file__).parent.glob('native_tt_*.py')},
            runner_sha256=digest(__file__),device=args.device)
        runtime_files=[TT/'t2'/name for name in ['model/core.py','model/modality.py','model/task/task.py',
            'robotok/tokenizer.py','robotok/io.py','data/normalization.py','train/augment.py']]
        runtime_files += [ROOT.parent/'quad/rsl_rl/rsl_rl'/name for name in
                          ['algorithms/ppo.py','modules/actor_critic.py','storage/rollout_storage.py']]
        contract['runtime']={str(f):digest(f) for f in runtime_files}
        protocol=out/'protocol.json'
        if protocol.exists() and json.loads(protocol.read_text())!=contract:raise ValueError('Chain code or inputs changed; preserve this run')
        save_json(protocol,contract)
        stages={}
        for stage in STAGES:
            if any(digest(path)!=value for path,value in contract['runtime'].items()):
                raise RuntimeError('Original TT/PPO runtime changed during the chain')
            if completed(stage,out):
                stages[stage]='complete';continue
            save_json(out/'chain.json',dict(status='running',stage=stage,stages=stages,updated=time.time()))
            with (out/f'{stage}.log').open('a') as log:
                result=subprocess.run([sys.executable,'-u',__file__,stage,'--source',str(source),
                    '--out',str(out),'--device',args.device],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode or not completed(stage,out):
                save_json(out/'chain.json',dict(status='failed',stage=stage,returncode=result.returncode,stages=stages,updated=time.time()))
                raise RuntimeError(f'{stage} failed; inspect {out/stage}.log; later stages were not started')
            stages[stage]='complete'
        save_json(out/'chain.json',dict(status='complete',stages=stages,updated=time.time()))
    elif args.stage=='preflight':preflight(source,out,args.device)
    elif args.stage=='prepare':prepare(source,out/'data')
    elif args.stage=='train':
        gate=json.loads((out/'preflight.json').read_text())
        if not gate['passed']:raise RuntimeError('Preflight has not passed')
        train(out/'data',out/'tt_base',microbatch=gate['microbatch'],device=args.device)
    elif args.stage=='gradient':gradient(out,args.device)
    elif args.stage=='residual':residual(out,args.device)
    else:final_evaluation(source,out,args.device)


if __name__=='__main__':main()
