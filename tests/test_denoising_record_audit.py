"""Stage-0 auditing must detect corrupted rewards, controls and termination."""
import sys
from pathlib import Path
import copy
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/tt_dynamics'))
from train_diffusion_rl import physical_rest_s0,replay_generated
from train_task_rl import build_task_model
from audit_denoising_records import trace_errors,audit_case


@pytest.fixture(scope='module')
def recording():
    e=torch.full((1,13),.5)
    state=physical_rest_s0(e)
    pred=torch.zeros(1,6,54);pred[:,:,:42]=state[:,None,:]
    pred[:,:,42:]=torch.as_tensor(build_task_model(e[0].numpy()).keyframe('home').ctrl.copy())
    rewards,rows,saved=replay_generated(pred,e,'zigzag')
    assert rows[0]['steps']==5 and rows[0]['termination_reason']=='time_limit'
    data=dict(pred=pred.numpy(),E=e.numpy(),**saved)
    return data,rows


def test_physics_recording_and_reward_decomposition_are_reproducible(recording,tmp_path):
    data,rows=recording
    assert trace_errors(data,0,rows[0],'zigzag')==[]
    path=tmp_path/'rollout.npz';np.savez_compressed(path,**data)
    result=audit_case(path,rows,'zigzag')
    assert result['candidates'][0]['errors']==[]
    assert result['candidates'][0]['stored_trace_complete']


@pytest.mark.parametrize('field,error',[('ctrl_0','applied_controls'),('reward_components_0','reward_components'),('step_reward_0','step_rewards')])
def test_corrupted_trace_is_detected(recording,field,error):
    data,rows=recording;broken={k:v.copy() for k,v in data.items()}
    broken[field].flat[0]+=.1
    assert error in trace_errors(broken,0,rows[0],'zigzag')


def test_wrong_termination_and_final_reward_are_detected(recording):
    data,rows=recording;row=copy.deepcopy(rows[0])
    row['termination_reason']='fall';row['terminal_reward']+=1
    errors=trace_errors(data,0,row,'zigzag')
    assert 'termination_reason' in errors and 'terminal_reward' in errors


def test_old_records_are_not_reported_as_fully_auditable(recording):
    data,rows=recording
    old={k:v for k,v in data.items() if k in ['pred','E','tokens_0','qpos_0']}
    assert trace_errors(old,0,rows[0],'zigzag') is None


def test_invalid_initial_record_has_no_physics_steps(recording):
    data,_=recording;pred=torch.as_tensor(data['pred'].copy());pred[:,:,2]=-.1
    _,rows,saved=replay_generated(pred,torch.as_tensor(data['E']),'zigzag')
    archive=dict(pred=pred.numpy(),E=data['E'],**saved)
    assert rows[0]['termination_reason']=='invalid_initial' and rows[0]['steps']==0
    assert rows[0]['physical_return']==0 and rows[0]['invalid_penalty']==-20
    assert trace_errors(archive,0,rows[0],'zigzag')==[]


def test_audit_rejects_wrong_training_mean(recording,tmp_path):
    import json
    from audit_denoising_records import audit_run
    data,rows=recording
    (tmp_path/'protocol.json').write_text(json.dumps(dict(updates=1,batch=1,diffusion_steps=1,task='zigzag')))
    (tmp_path/'summary.json').write_text(json.dumps({'rl_after':{'rows':rows}}))
    np.savez_compressed(tmp_path/'rl_after.npz',**data)
    record=dict(update=1,rows=rows,terminal_reward_mean=rows[0]['terminal_reward']+1,
                physical_episodes=1,denoising_transitions=1,seconds=.1)
    (tmp_path/'training.jsonl').write_text(json.dumps(record)+'\n')
    result=audit_run(tmp_path)
    assert result['status']=='fail' and 'update_1_mean' in result['errors']
