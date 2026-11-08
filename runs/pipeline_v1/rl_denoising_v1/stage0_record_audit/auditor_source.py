"""Stage 0: verify saved rewards, termination and deterministic physics replay."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch
from train_task_rl import DENSE_REWARD_KEYS, dense_task_reward, features, build_task_model
from train_diffusion_rl import replay_generated, mujoco


def trace_errors(data, i, row, task):
    """Recompute per-step reward from saved states, controls and motor forces."""
    required=[f'{name}_{i}' for name in ['tokens','qpos','qvel','ctrl','torque','step_reward','reward_components']]
    if any(name not in data for name in required):
        return None
    errors=[]
    def check(ok, message):
        if not ok: errors.append(message)
    t=int(row['steps']);x=np.asarray(data[f'tokens_{i}']);q=np.asarray(data[f'qpos_{i}']);v=np.asarray(data[f'qvel_{i}'])
    ctrl=np.asarray(data[f'ctrl_{i}']);torque=np.asarray(data[f'torque_{i}'])
    components=np.asarray(data[f'reward_components_{i}']);rewards=np.asarray(data[f'step_reward_{i}'])
    check(x.shape==(t+1,54) and q.shape==(t+1,19) and v.shape==(t+1,18),'state_length')
    check(ctrl.shape==(t,12) and torque.shape==(t,12) and components.shape==(t,len(DENSE_REWARD_KEYS)) and rewards.shape==(t,),'transition_length')
    if errors:return errors
    check(all(np.isfinite(a).all() for a in [x,q,v,ctrl,torque,components,rewards]),'nonfinite_trace')
    check(list(np.asarray(data['reward_component_names']))==list(DENSE_REWARD_KEYS),'reward_component_names')
    check(np.isclose(row['execution_seconds'],t*.02,atol=1e-10),'execution_seconds')
    max_steps=len(data['pred'][i])-1
    check(np.isclose(row['max_seconds'],max_steps*.02,atol=1e-10),'max_seconds')
    if row['invalid_initial']:
        check(t==0 and row['termination_reason']=='invalid_initial','invalid_termination')
        check(row['terminal_reward']==-20. and row['invalid_penalty']==-20.,'invalid_reward')
        check(row['physical_return']==0. and row['consistency_penalty']==0.,'invalid_components')
        check(bool(row['initial_invalid_reasons']),'missing_invalid_reason')
        return errors
    check(np.allclose(features(q,v,x[:,38:42]),x[:,:42],rtol=1e-6,atol=1e-6),'state_features')
    home=np.asarray(build_task_model(data['E'][i]).keyframe('home').ctrl)
    expected_ctrl=home+np.clip(data['pred'][i,:t,42:]-home,-2.5,2.5)
    check(np.allclose(expected_ctrl,ctrl,rtol=0,atol=1e-7),'applied_controls')
    check(np.allclose(x[:-1,42:],ctrl,rtol=0,atol=1e-7),'token_actions')
    expected=[];previous=home
    for k in range(t):
        _,rates=dense_task_reward(x[k,:42],x[k+1,:42],ctrl[k]-previous,torque[k],task,k+1,data['pred'][i,0,2])
        parts={name:value*.02 for name,value in rates.items()};parts['fall']=-20. if x[k+1,7]>-.5 else 0.
        expected.append([parts[name] for name in DENSE_REWARD_KEYS]);previous=ctrl[k]
    expected=np.asarray(expected).reshape(t,len(DENSE_REWARD_KEYS))
    check(np.allclose(expected,components,rtol=1e-6,atol=1e-7),'reward_components')
    check(np.allclose(expected.sum(1),rewards,rtol=1e-6,atol=1e-7),'step_rewards')
    physical=float(expected.sum())
    penalty=5.*float(np.mean((data['pred'][i,:t+1,:3]-x[:,:3])**2))
    check(np.isclose(physical,row['physical_return'],rtol=1e-6,atol=1e-6),'physical_return')
    check(np.isclose(penalty,row['consistency_penalty'],rtol=1e-6,atol=1e-6),'consistency_penalty')
    check(np.isclose(physical-penalty,row['terminal_reward'],rtol=1e-6,atol=1e-6),'terminal_reward')
    check(np.allclose(expected.sum(0),[row['reward_components'][k] for k in DENSE_REWARD_KEYS],rtol=1e-6,atol=1e-6),'component_totals')
    fell=bool(x[-1,7]>-.5)
    check(not np.any(x[1:-1,7]>-.5),'continued_after_fall')
    check(t>0 and (fell or t==max_steps),'early_termination')
    check(row['termination_reason']==('fall' if fell else 'time_limit'),'termination_reason')
    check(row['survived']==(not fell),'survived')
    check(row['invalid_penalty']==0.,'invalid_penalty')
    return errors


def audit_case(path, rows, task):
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    if len(rows)!=len(data['E']):raise ValueError('Candidate count differs from archive')
    _,fresh,saved=replay_generated(torch.as_tensor(data['pred']),torch.as_tensor(data['E']),task)
    checked=[]
    for i,(old,new) in enumerate(zip(rows,fresh)):
        errors=[]
        for key in ['candidate','steps','survived','invalid_initial']:
            if old[key]!=new[key]:errors.append('replay_'+key)
        for key in ['terminal_reward','original_reward','progress','prediction_position_rmse']:
            if old[key] is None or new[key] is None:
                if old[key]!=new[key]:errors.append('replay_'+key)
            elif not np.isclose(old[key],new[key],rtol=1e-6,atol=1e-5):errors.append('replay_'+key)
        for name in ['tokens','qpos','qvel','ctrl','torque','step_reward','reward_components']:
            key=f'{name}_{i}'
            if key in data:
                if data[key].shape!=saved[key].shape or not np.allclose(data[key],saved[key],rtol=1e-6,atol=1e-6):errors.append('replay_'+name)
        verified=trace_errors(data,i,old,task)
        if verified is not None:errors.extend(verified)
        delta=abs(float(old['terminal_reward'])-float(new['terminal_reward']))
        checked.append(dict(candidate=i,errors=errors,reward_abs_error=delta,stored_trace_complete=verified is not None,
                            replayed=new))
    return dict(file=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidates=checked)


def audit_run(run):
    protocol=json.loads((run/'protocol.json').read_text());summary=json.loads((run/'summary.json').read_text())
    records=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
    errors=[];cases=[]
    if protocol.get('mujoco_version') is not None and protocol['mujoco_version']!=mujoco.__version__:
        errors.append('physics_version_mismatch')
    for name,digest in protocol.get('source_hashes',{}).items():
        path=run/name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            errors.append('source_snapshot_'+name)
    if len(records)!=protocol['updates']:errors.append('training_update_count')
    for u,record in enumerate(records,1):
        if record['update']!=u or len(record['rows'])!=protocol['batch']:errors.append(f'update_{u}_counter')
        expected=np.asarray([r['terminal_reward'] for r in record['rows']],dtype=np.float32)
        if not np.isclose(expected.mean(),record['terminal_reward_mean'],rtol=1e-6,atol=1e-5):errors.append(f'update_{u}_mean')
        if record['physical_episodes']!=u*protocol['batch'] or record['denoising_transitions']!=u*protocol['batch']*protocol.get('diffusion_steps',5):errors.append(f'update_{u}_transitions')
        if u>1 and record['seconds']<records[u-2]['seconds']:errors.append(f'update_{u}_time')
        if 'ppo_terminal_rewards' in record and not np.allclose(record['ppo_terminal_rewards'],expected,rtol=0,atol=1e-6):errors.append(f'update_{u}_ppo_buffer')
        if 'policy_update_before' in record and (record['policy_update_before']!=u-1 or record['policy_update_after']!=u):errors.append(f'update_{u}_policy_step')
        path=run/'training_rollouts'/f'update_{u:05d}.npz'
        if path.exists():cases.append(audit_case(path,record['rows'],protocol['task']))
    for name,result in summary.items():cases.append(audit_case(run/f'{name}.npz',result['rows'],protocol['task']))
    if (run/'best_training.npz').exists():
        best=json.loads((run/'best_training.json').read_text());cases.append(audit_case(run/'best_training.npz',best['rows'],protocol['task']))
        if not np.isclose(best['mean_reward'],max(r['terminal_reward_mean'] for r in records),rtol=1e-6,atol=1e-5):errors.append('best_training_selection')
    checked=[candidate for case in cases for candidate in case['candidates']]
    missing=[r['update'] for r in records if not (run/'training_rollouts'/f'update_{r["update"]:05d}.npz').exists()]
    complete=not missing and bool(protocol.get('source_hashes')) and protocol.get('mujoco_version')==mujoco.__version__ and all(c['stored_trace_complete'] for c in checked) and all('ppo_terminal_rewards' in r for r in records)
    errors.extend(f'{Path(case["file"]).name}:candidate_{r["candidate"]}:{e}' for case in cases for r in case['candidates'] for e in r['errors'])
    return dict(run=str(run.resolve()),status='fail' if errors else ('pass' if complete else 'partial'),errors=errors,
                training_updates_checked=len(records),physics_candidates_replayed=len(checked),missing_training_rollout_updates=missing,
                complete_saved_traces=sum(c['stored_trace_complete'] for c in checked),
                maximum_reward_abs_error=max((c['reward_abs_error'] for c in checked),default=0.),cases=cases,
                provenance=dict(recorded_mujoco=protocol.get('mujoco_version'),current_mujoco=mujoco.__version__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
    results=[]
    for run in a.runs:
        print('기록 검사:',run,flush=True)
        try:results.append(audit_run(run))
        except Exception as error:results.append(dict(run=str(run.resolve()),status='fail',errors=[str(error)],cases=[]))
    result=dict(stage=0,scope='Record consistency, not task success',mujoco_version=mujoco.__version__,results=results,
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    blocks=[]
    for r in results:
        label={'pass':'통과','partial':'부분 확인','fail':'불일치 발견'}[r['status']]
        blocks.append(f'<h2>{html.escape(Path(r["run"]).name)}: {label}</h2><p>학습 기록 {r.get("training_updates_checked",0)}회, 물리 재실행 {r.get("physics_candidates_replayed",0)}개, 저장된 상세 궤적 {r.get("complete_saved_traces",0)}개.</p><p>최대 보상 차이: {r.get("maximum_reward_abs_error",0):.9g}</p><p>학습 궤적 누락: {len(r.get("missing_training_rollout_updates",[]))}회. 이전 파일에 없는 관절 속도·힘·보상 항목은 재실행으로 복원했으며 당시 파일에 저장돼 있던 기록과 구분합니다.</p><pre>{html.escape(json.dumps(r["errors"],ensure_ascii=False,indent=2))}</pre>')
    (a.out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>0단계 기록 확인</title><style>body{font:17px/1.8 system-ui;max-width:1050px;margin:35px auto;padding:0 20px}pre{white-space:pre-wrap}</style><h1>0단계 기록 확인 결과</h1><p>같은 E와 생성 행동을 물리 환경에 다시 적용해 저장 결과와 비교했습니다. 새 기록은 저장된 상태·모터 목표·힘에서 보상 항목도 다시 계산합니다. 부분 확인은 확인 가능한 자료에서 불일치가 없지만 전체 학습 궤적이나 상세 기록이 없다는 뜻입니다. 통과는 과제 수행 성공을 의미하지 않습니다.</p>'+''.join(blocks)+'<p><a href="audit.json">후보별 상세 결과</a></p></html>')
    print(a.out/'report.html',flush=True)
    if any(r['status']=='fail' for r in results):raise SystemExit(1)

if __name__=='__main__':main()
