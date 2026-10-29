import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/tt_dynamics'))
from motion_screen import trajectory_metrics, zigzag_y


def evaluate(points, task='zigzag', survived=True):
    x=np.zeros((len(points),54));x[:,:2]=points
    row={'initial_physics':{'physically_valid':True}, 'mjx_fell_step':-1 if survived else 2,
         'mjx_total':1.,'path_err_m':0.}
    settings={'amp':.3,'period':2.,'half_width':.15,'radius':1.}
    return trajectory_metrics(x,np.ones(len(x),bool),row,task,settings,len(x)*.02)


def test_stationary_and_short_straight_are_not_zigzag_success():
    assert not evaluate(np.zeros((100,2)))['moving']
    result=evaluate(np.stack([np.linspace(0,.15,100), np.zeros(100)],axis=1))
    assert result['survived'] and result['moving']
    assert result['inside_lane_fraction']==1 and not result['task_success']
    assert not evaluate(np.array([[0.,0.],[2.,0.]]))['task_success']


def test_zigzag_requires_ordered_opposite_turns_and_survival():
    x=np.linspace(0,2,400);points=np.stack([x,zigzag_y(x,.3,2)],axis=1)
    assert evaluate(points)['task_success']
    assert not evaluate(points,survived=False)['task_success']


def test_circle_uses_arc_progress_and_requires_full_lap():
    angle=np.linspace(0,2*np.pi,400)
    points=np.stack([np.sin(angle),1-np.cos(angle)],axis=1)
    assert evaluate(points,'circle')['task_success']
    half=evaluate(points[:200],'circle')
    assert half['moving'] and not half['task_success']
