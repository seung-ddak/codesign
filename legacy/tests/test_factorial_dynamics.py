import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/pipeline_v1/factorial_init_action'


def module(name):
    spec=importlib.util.spec_from_file_location('factorial_'+name,RUN/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_initial_state_is_identical_across_action_origins():
    m=module('collect')
    q=np.arange(57).reshape(3,19);v=np.arange(54).reshape(3,18)
    qr=np.full(19,-1);vr=np.full(18,-2)
    qb,vb=m.initial_batches(q,v,qr,vr)
    assert qb.shape==(2,2,3,19)
    assert np.array_equal(qb[:,0],qb[:,1]) and np.array_equal(vb[:,0],vb[:,1])
    assert np.array_equal(qb[0,0],q)
    assert np.array_equal(qb[1,1],np.broadcast_to(qr,q.shape))


def test_windows_exclude_switched_invalid_or_nonconsecutive_frames():
    m=module('prepare')
    z={'motions':np.array(['walk']), 'clip':np.zeros((3,8),int),'valid':np.ones((3,8),bool),
       'switched':np.zeros((3,8),bool),'ref_frame':np.tile(np.arange(8),(3,1)), 'clip_start':np.array([0])}
    z['valid'][0]=False;z['switched'][1]=True
    rows=m.choose_windows(z,steps=3,count=4)
    assert len(rows)==4 and {r['source_episode'] for r in rows}=={2}
    assert len({r['source_t'] for r in rows})==4
    assert rows[0]['start_frame']==0 and rows[-1]['start_frame']==5


def test_factor_contrasts_preserve_interaction():
    m=module('evaluate')
    x=m.crossed_contrasts([0,1,2,5])
    assert x=={'rest_minus_reference':3.,'ppo_minus_reference_action':2.,'interaction':2.}
