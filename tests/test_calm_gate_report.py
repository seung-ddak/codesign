import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('report_gates',Path(__file__).parents[1]/'code/motion_tracking/calm/report_gates.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_survival_alone_never_passes_g0():
    assert m.gate_status(dict(continuous_10s=False))=='FAIL'
    assert m.gate_status(dict(continuous_10s=True,finite_states=True,finite_actions=True))=='BLOCKED'
    assert m.gate_status(dict(continuous_10s=True,finite_states=True,finite_actions=False))=='FAIL'
    assert m.gate_status(dict(continuous_10s=True,resets_after_initial=1))=='FAIL'


def test_state_action_time_contract():
    z=dict(state=np.zeros((10,37)),llc_action=np.zeros((9,12)),target_q=np.zeros((9,12)),
           latent=np.zeros((9,64)),command=np.zeros((9,3)),time=np.arange(10)/30,dt=1/30)
    assert m.validate_arrays(z)
    z['llc_action']=np.zeros((10,12))
    with pytest.raises(ValueError,match='alignment'):m.validate_arrays(z)
    z['llc_action']=np.zeros((9,12));z['time'][4]+=.01
    with pytest.raises(ValueError,match='Timestamp'):m.validate_arrays(z)
    z['time']=np.arange(10)/30;z['target_q'][1,2]=np.nan
    assert not m.validate_arrays(z)


def test_execution_error_is_blocked_and_existing_report_is_preserved(tmp_path):
    p=tmp_path/'failed';p.mkdir();(tmp_path/'failed.log').write_text('CUDA execution error')
    result=m.diagnostic_reports(tmp_path)
    assert result[0]['status']=='BLOCKED'
    original=(p/'report.html').read_bytes()
    assert m.diagnostic_reports(tmp_path)==[]
    assert (p/'report.html').read_bytes()==original
