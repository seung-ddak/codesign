import json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/motion_tracking'))
from shared_calm_common import create_plan,load_config,shapes,stage_groups,save
from shared_calm import deadline,audit
from shared_calm_dashboard import snapshot,report
ROOT=Path(__file__).resolve().parents[1]

def test_v100_config_covers_all_shapes_with_balanced_sparse_pairs(tmp_path):
    c=load_config(ROOT/'configs/shared_calm_v100_48h.yaml');c.update(source_motions=20,train_designs=4,eval_designs=1,resident_designs=2)
    source=tmp_path/'source';source.mkdir();save(source/'status.json',{'accepted':10000});records=[]
    for i in range(20):
        (source/f'{i}.npz').write_bytes(bytes([i]));records.append(dict(id=i,file=f'{i}.npz',accepted=True,plan=dict(k=i%12+1,segments=[{'end':300}])))
    (source/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));p=create_plan(source,c)
    assert p['target_pairs']==40 and p['target_episodes']==80
    seen={i:set() for i in range(20)}
    for e in p['designs']:
        ids=p['assignments'][e['design']];assert len(ids)==len(set(ids))==10
        for i in ids:seen[i].add(e['design'])
    assert all(len(v)==2 for v in seen.values())
    assert stage_groups(p)==[['e0000','e0001'],['e0002','e0003']]
    assert create_plan(source,c)==p

def test_catalog_has_axes_pairs_corners_and_interior():
    rows=shapes(1);assert len(rows)==220
    assert len({tuple(r['E']) for r in rows})==220
    base=rows[0]['physical_dimensions']
    for r in rows:
        for k,v in r['physical_dimensions'].items():assert .8-1e-9<=v/base[k]<=1.2+1e-9
    assert all(len(r['changed_tokens'])==1 for r in rows[1:41])
    assert all(len(r['changed_tokens'])==2 for r in rows[41:81])
    assert all(len(r['changed_tokens'])==5 for r in rows[81:113])

def test_deadlines_do_not_restart_on_resume():
    c=load_config(ROOT/'configs/shared_calm_v100_48h.yaml');clock={'started':100,'deadline':100+48*3600}
    assert deadline(clock,c,'prepare')==100+8*3600
    assert deadline(clock,c,'collect')==100+46*3600
    assert deadline(clock,c,'verify')==clock['deadline']

def test_audit_counts_unique_sources_and_both_initial_conditions(tmp_path):
    plan={'motions':[{'id':1}],'target_pairs':1,'target_episodes':2}
    folder=tmp_path/'rollouts/e0000';folder.mkdir(parents=True)
    for v in [0,1]:
        np.savez(folder/f'variant_{v}.npz',valid=np.array([[1,1,0]],bool),qpos=np.full((1,4,19),v*.01),qvel=np.zeros((1,4,18)),action=np.zeros((1,3,12)),target_q=np.zeros((1,3,12)),dt=1/30,design='e0000',source_motion_id=[1],variant=v,fell=[False])
    result=audit(tmp_path,plan);assert result['coverage_complete'] and result['episodes']==2 and result['source_coverage']==1 and result['pairs_with_both_starts']==1
    (folder/'variant_1.npz').unlink();assert not audit(tmp_path,plan)['coverage_complete']

def test_dashboard_handles_partial_json_and_exports_offline_korean_page(tmp_path):
    save(tmp_path/'plan.json',dict(config={},motions=[],designs=[],target_pairs=0,target_episodes=0))
    (tmp_path/'metrics.jsonl').write_text('{"step":1,"time":0,"metrics":{}}\n{"partial":')
    d=snapshot(tmp_path);assert len(d['metrics'])==1
    report(tmp_path);page=(tmp_path/'report.html').read_text()
    for label in ['누적 학습 진행','학습 전이 (백만 회)','관절 속도 추종 보상','아직 데이터 없음']:assert label in page
    assert 'window.STATIC_SNAPSHOT=' in page and 'https://' not in page

def test_distinct_reference_lengths_stay_with_their_physics_models(tmp_path):
    import sys,functools
    sys.path.insert(0,str(ROOT.parent/'transformer-transformer'))
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from pd_replay import build_model
    from multi_design import build_bank,MultiDesignVmapWrapper
    cfg=OmegaConf.load(ROOT/'code/motion_tracking/config/env/a1_track.yaml')
    cfg.update(dict(ctrl_dt=1/30,sim_dt=1/240,clips=[],reference_dir=str(tmp_path)))
    catalog={'one':[.5]*13,'two':[.5]*13};catalog['two'][2]=.51
    clips={'one':['short'],'two':['long']}
    for name,n in [('one',301),('two',601)]:
        model=build_model('00000011',uniforms=catalog[name]);home=model.key('home').qpos
        folder=tmp_path/name/'00000011';folder.mkdir(parents=True)
        np.savez(folder/(clips[name][0]+'.npz'),fix_q=np.tile(home[7:],(n,1)),fix_base_pos=np.tile(home[:3],(n,1)),fix_base_quat=np.tile(home[3:7],(n,1)),fix_contact=np.zeros((n,4)),dt=1/30,uniforms=catalog[name])
    env,bank=build_bank(['one','two'],cfg,tmp_path,'00000011',catalog,functools.partial(build_model,choice='00000011'),[0,0],clips_by_design=clips)
    assert bank['refs']['ref_qpos'].shape==(2,601,19)
    np.testing.assert_array_equal(bank['refs']['clip_end'],[[301],[601]])
    wrapped=MultiDesignVmapWrapper(env,bank)
    with wrapped.design_env(wrapped.model_for(jnp.int32(1)),jnp.int32(1)) as selected:
        assert int(selected.clip_end[0])==601
    assert int(env.clip_end[0])==301


def test_expired_child_budget_terminates_worker_without_success(tmp_path):
    import time
    from shared_calm import run_child
    okay=run_child(['-c','import time;time.sleep(60)'],tmp_path,tmp_path/'child.log',time.time()-1,{'pool_fraction':.1,'reserve_disk_gb':0})
    assert okay is False


def test_event_can_accept_rollout_summary_timestamp(tmp_path):
    from shared_calm_common import event
    event(tmp_path,'rollout_complete',time=1,episodes=4)
    row=json.loads((tmp_path/'events.jsonl').read_text())
    assert row['kind']=='rollout_complete' and row['episodes']==4 and row['time']>1


def test_unprepared_shape_does_not_discard_other_shapes_in_bank(tmp_path):
    from shared_calm_common import ready_batch
    plan={'config':{'resident_designs':2},'designs':[{'design':'a'},{'design':'b'}],'eval_designs':[{'design':'v'}]}
    save(tmp_path/'ready/b.json',{})
    assert ready_batch(tmp_path,plan,0)==(['b'],[])
    save(tmp_path/'ready/v.json',{})
    assert ready_batch(tmp_path,plan,0)==(['b'],['v'])


@pytest.mark.parametrize('key,value',[('num_envs',0),('pool_fraction',1.1),('train_sweeps',0)])
def test_invalid_resource_settings_are_rejected(tmp_path,key,value):
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(ROOT/'configs/shared_calm_v100_48h.yaml');cfg[key]=value;p=tmp_path/'bad.yaml';OmegaConf.save(cfg,p)
    with pytest.raises(ValueError):load_config(p)


def test_rsi_prioritizes_reference_segments_not_observed_before():
    from shared_calm_common import covered_length,choose_undercovered_start
    assert covered_length([(0,30),(20,50),(80,100)])==70
    motion={'seconds':10,'segments':[{'index':0,'start':0,'end':150},{'index':1,'start':150,'end':300}]}
    rng=np.random.default_rng(1)
    starts=[choose_undercovered_start(motion,[(0,150)],rng) for _ in range(10)]
    assert all(150<=s<225 for s in starts)


def test_shared_rollout_loads_into_tt_without_becoming_action_intervention(tmp_path):
    sys.path.insert(0,str(ROOT/'code/tt_dynamics'))
    from rollout_data import load,source_group_keys,intervention_pairs
    q=np.zeros((2,5,19));q[:,:,3]=1;p=tmp_path/'variant_1.npz'
    np.savez(p,qpos=q,qvel=np.zeros((2,5,18)),contact=np.zeros((2,5,4)),target_q=np.zeros((2,4,12)),valid=np.ones((2,4),bool),E=np.full(13,.5),motions=['walk','turn'],clip=np.array([[0]*4,[1]*4]),group_id=np.array(['a'*64,'b'*64]),start_variant=1)
    x,e,m,src,labels=load([p],4)
    assert x.shape==(2,4,54) and e.shape==(2,13) and m.all()
    with np.load(p) as z:assert source_group_keys(p,z,length=2)==['group:'+'a'*64,'group:'+'b'*64]
    pair_e,pair_a=intervention_pairs([p],4)
    assert pair_e.shape==pair_a.shape==(0,2)


def test_single_transition_is_preserved_but_not_counted_as_tt_coverage(tmp_path):
    folder=tmp_path/'rollouts/e0000';folder.mkdir(parents=True)
    np.savez(folder/'variant_0.npz',valid=np.array([[1,0]],bool),qpos=np.zeros((1,3,19)),qvel=np.zeros((1,3,18)),action=np.zeros((1,2,12)),target_q=np.zeros((1,2,12)),dt=1/30,design='e0000',source_motion_id=[1],start_variant=0,fell=[True])
    result=audit(tmp_path,{'motions':[{'id':1}],'target_pairs':1,'target_episodes':2})
    assert result['valid_transitions']==1 and result['tt_usable_episodes']==0 and result['source_coverage']==0 and not result['coverage_complete']


def test_v100_preset_rejects_multiple_visible_gpus(monkeypatch):
    import shared_calm
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    monkeypatch.setattr(shared_calm.subprocess,'check_output',lambda *a,**kw:'Tesla V100, 32768, 550.0\nTesla V100, 32768, 550.0\n')
    with pytest.raises(ValueError,match='exactly one GPU'):shared_calm.preflight({})
