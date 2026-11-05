import sys,json
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/motion_tracking'))
from plan_calm_dynamics import allocate,morphologies,build,prepare_reference


def test_full_allocation_balances_reuse_and_keeps_common_and_extra_disjoint():
    motions=[dict(id=i,k=i%12+1) for i in range(10000)]
    shared,rows,stats=allocate(motions)
    assert len(shared)==30 and len(rows)==1000
    assert len({tuple(sorted(r['extra'])) for r in rows})==1000
    assert stats['extra_reuse_min']==7 and stats['extra_reuse_max']==8
    for r in rows:
        assert r['common']==shared and len(r['extra'])==70
        assert len(set(r['common']+r['extra']))==100
    assert {m['k'] for m in motions if m['id'] in shared}==set(range(1,13))
    assert allocate(motions)==(shared,rows,stats)


def test_catalog_changes_real_lengths_by_twenty_percent_with_isolated_and_combined_tokens():
    from plan_calm_dynamics import LENGTH_TOKENS
    catalog=morphologies();e=np.array([r['E'] for r in catalog]);base=catalog[0]['physical_dimensions']
    assert len({tuple(row) for row in e})==1000
    np.testing.assert_equal(e[0],.5)
    assert sum(r['variation']=='single_token' for r in catalog)==200
    assert sum(r['variation']=='multiple_tokens' for r in catalog)==799
    active=set(LENGTH_TOKENS)
    for r in catalog:
        changed=set(np.flatnonzero(np.abs(np.array(r['E'])-.5)>1e-10))
        assert changed==set(r['changed_tokens']) and changed<=active
        assert len(changed)==1 if r['variation']=='single_token' else len(changed)>=2 if r['variation']=='multiple_tokens' else not changed
        for name,v in r['physical_dimensions'].items():assert .8-1e-10<=v/base[name]<=1.2+1e-10
    for i in active:
        deltas=[r['relative_dimension_changes'][str(i)] for r in catalog if r['variation']=='single_token' and i in r['changed_tokens']]
        assert len(deltas)==40 and min(deltas)==pytest.approx(-.2) and max(deltas)==pytest.approx(.2)
    assert np.all(e[:,list(set(range(13))-active)]==.5)


def test_long_retarget_summary_names_fit_the_filesystem():
    from pd_replay import retarget_summary_name
    names=['calm_%05d'%i for i in range(100)]
    short=retarget_summary_name(names)
    assert len(short.encode())<240 and short==retarget_summary_name(names)
    assert short!=retarget_summary_name(names[:-1])
    assert retarget_summary_name(['walk'])=='summary_walk.csv'
    assert retarget_summary_name(None)=='summary.csv'


def test_partial_source_cannot_become_final_manifest(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'status.json').write_text(json.dumps(dict(status='running',accepted=9999)))
    with pytest.raises(ValueError,match='10,000'):build(source,tmp_path/'jobs')


def test_source_conversion_preserves_duration_quaternion_and_joint_order(tmp_path):
    n=601;root=np.zeros((n,13));root[:,6]=1;root[:,2]=.3;root[:,0]=np.arange(n)/30
    names=[leg+'_'+j+'_joint' for leg in ['FL','FR','RL','RR'] for j in ['hip','thigh','calf']]
    q=np.tile(np.arange(12),(n,1));src=tmp_path/'src.npz';dst=tmp_path/'dst.npz'
    np.savez(src,root=root,q=q,time=np.arange(n)/30,joint_names=names,foot_pos=np.zeros((n,4,3)),foot_floor_contact=np.zeros((n,4),bool))
    prepare_reference(src,dst)
    with np.load(dst) as z:
        assert z['fix_q'].shape==(601,12) and float(z['dt'])==pytest.approx(1/30)
        np.testing.assert_equal(z['fix_q'][0],[3,4,5,0,1,2,9,10,11,6,7,8])
        np.testing.assert_allclose(z['fix_base_quat'],np.tile([1,0,0,0],(601,1)))
        assert z['fix_base_pos'][-1,0]==pytest.approx(20)


def test_training_command_uses_the_assigned_shape_and_all_clips(tmp_path):
    from run_calm_dynamics import train_arguments
    e=dict(E=[.1]*13,common=list(range(30)),extra=list(range(30,100)))
    args=[str(x) for x in train_arguments(e,tmp_path/'refs/00000011',tmp_path/'policy',100)]
    assert 'multi_design=false' in args and 'uniforms='+json.dumps(e['E'],separators=(',',':')) in args
    assert 'env.ctrl_dt='+str(1/30) in args and 'eval_env.ctrl_dt='+str(1/30) in args
    assert 'env.sim_dt='+str(1/240) in args and 'env.max_episode_length=600' in args
    assert round((1/30)/(1/240))==8
    clips=next(x for x in args if x.startswith('env.clips='))
    assert clips.count('calm_')==100
    assert 'train.num_envs=32' in args and 'train.batch_size=32' in args and 'train.num_eval_envs=1' in args


def test_single_design_environment_really_builds_its_assigned_morphology(monkeypatch):
    import train_rl_procedural as trainer
    from omegaconf import OmegaConf
    expected=[.15]*13;seen=[]
    def model(**kwargs):seen.append(kwargs);return kwargs['uniforms']
    monkeypatch.setattr(trainer,'build_model',model)
    monkeypatch.setattr(trainer.hydra.utils,'instantiate',lambda cfg,mj_model_fn:mj_model_fn())
    cfg=OmegaConf.create(dict(multi_design=False,kp=30,tau_scale=None,torque_limit=33.5,uniforms=expected,env={},eval_env={}))
    trainer.build_envs(cfg,'00000011')
    assert len(seen)==2 and all(list(e['uniforms'])==expected for e in seen)


def test_failed_baseline_tracking_stops_before_other_morphologies(tmp_path,monkeypatch):
    import run_calm_dynamics as runner
    out=tmp_path/'queue';out.mkdir();designs=[dict(design='e0000'),dict(design='e0001')]
    (out/'manifest.json').write_text(json.dumps(dict(designs=designs)))
    called=[]
    def process(manifest,design,root,stage,steps):
        called.append(design['design']);p=root/'rollouts'/design['design'];p.mkdir(parents=True)
        (p/'primitive_seed0.json').write_text(json.dumps(dict(tracking_pass_fraction=.5)))
    monkeypatch.setattr(runner,'process',process)
    monkeypatch.setattr(sys,'argv',['run','--source',str(tmp_path/'source'),'--out',str(out)])
    runner.main()
    assert called==['e0000']
    status=json.loads((out/'status.json').read_text())
    assert status['status']=='tracking_quality_gate_not_met' and status['pending_designs']==1


def test_rollout_batches_preserve_episode_order_and_shared_morphology():
    from collect_tracking_dynamics import merge_batches
    shared=np.arange(13)
    batches=[dict(E=shared,dt=.02,source_motion_id=np.arange(i,i+n),valid=np.ones((n,1000),bool),qpos=np.full((n,1001,19),i)) for i,n in [(0,16),(16,4)]]
    result=merge_batches(batches,{'source_motion_id','valid','qpos'})
    np.testing.assert_array_equal(result['source_motion_id'],np.arange(20))
    np.testing.assert_array_equal(result['E'],shared)
    assert result['qpos'].shape==(20,1001,19) and result['valid'].shape==(20,1000)
    assert result['dt']==.02


def test_worker_disables_gpu_preallocation_and_limits_pool(tmp_path,monkeypatch):
    import run_calm_dynamics as runner
    calls=[]
    monkeypatch.setattr(runner.subprocess,'run',lambda *args,**kw:calls.append(kw))
    runner.command(['python','worker.py'],tmp_path/'worker.log')
    assert calls[0]['env']['XLA_PYTHON_CLIENT_PREALLOCATE']=='false'
    assert calls[0]['env']['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.35'
