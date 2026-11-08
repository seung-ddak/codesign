"""Stage-1 comparison detects motor/model differences without hiding early termination."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/tt_dynamics'))
from audit_denoising_physics import compare_traces,model_differences
from train_task_rl import build_task_model


def states(n=4):
    q=np.zeros((n,19));q[:,2]=.3;q[:,3]=1
    return q


def test_identical_physics_inputs_and_traces_match():
    q=states();ctrl=np.ones((3,12))
    result=compare_traces(q,q.copy(),ctrl,ctrl.copy())
    assert result['common_states']==4 and result['control_max_error']==0
    assert result['first_step_body_error_m']==0 and result['prefix_2s_joint_max_rmse_deg']==0


def test_control_sign_and_motion_difference_are_detected():
    q=states();other=q.copy();other[1:,0]=.03;other[1:,7:]=np.deg2rad(3)
    result=compare_traces(q,other,np.ones((3,12)),-np.ones((3,12)))
    assert result['control_max_error']==2
    assert result['first_step_body_error_m']==.03
    np.testing.assert_allclose(result['first_step_joint_rmse_deg'],3)


def test_rejected_or_truncated_execution_is_not_a_full_comparison():
    q=states()
    rejected=compare_traces(q,q[:1],np.ones((3,12)),np.empty((0,12)))
    assert rejected['common_states']==1 and rejected['first_step_body_error_m'] is None
    assert rejected['control_max_error'] is None
    empty=compare_traces(q,np.empty((0,19)),np.ones((3,12)),np.empty((0,12)))
    assert empty['common_states']==0


def test_model_parameters_and_control_period_difference_are_detected():
    one=build_task_model(np.full(13,.5));two=build_task_model(np.full(13,.5))
    assert model_differences(one,two)==[]
    two.actuator_gainprm[0,0]*=2;two.opt.timestep*=2
    assert 'actuator_gainprm' in model_differences(one,two)
    assert 'opt.timestep' in model_differences(one,two)


def test_report_leads_with_conclusion_and_hides_raw_metrics(tmp_path):
    import json
    from audit_denoising_physics import render_report
    home=dict(case='home_generated_E_10s',cpu_training_report_identical=True,
              cpu=dict(invalid_initial=False,termination_reason='time_limit',execution_seconds=10.,max_seconds=10.,survived=True),
              mjx=dict(execution_valid=True,steps=500,fell_step=-1),status='pass',metrics={})
    fall=dict(case='shared_window_zigzag_10s_100iter_candidate2',cpu_training_report_identical=True,
              cpu=dict(invalid_initial=False,termination_reason='fall',execution_seconds=.04,max_seconds=10.,survived=False),
              mjx=dict(execution_valid=True,steps=2,fell_step=2),status='pass',metrics={'control_max_error':0.})
    (tmp_path/'results.json').write_text(json.dumps({'results':[home,fall]}))
    render_report(tmp_path);page=(tmp_path/'report.html').read_text()
    assert '이번 즉시 낙상은 학습·평가 환경 차이로 설명되지 않습니다.' in page
    assert '0.04초' in page and '10초' in page
    assert 'control_max_error' not in page and '<pre>' not in page
    assert page.index('이번 즉시 낙상')<page.index('각 비교 사례')


def test_report_keeps_mjx_difference_visible(tmp_path):
    import json
    from audit_denoising_physics import render_report
    row=dict(case='zigzag_1000iter_candidate0',cpu_training_report_identical=True,
             cpu=dict(invalid_initial=False,termination_reason='time_limit',execution_seconds=3.98),
             mjx=dict(execution_valid=True,steps=199,fell_step=-1),status='different',
             metrics={'prefix_2s_orientation_max_error_deg':3.24})
    (tmp_path/'results.json').write_text(json.dumps({'results':[row]}))
    render_report(tmp_path);page=(tmp_path/'report.html').read_text()
    assert '완전한 일치는 확인되지 않았습니다.' in page and '3.24도' in page
    assert 'prefix_2s_orientation_max_error_deg' not in page and '<pre>' not in page
