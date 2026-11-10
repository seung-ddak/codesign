import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/pipeline_v1/fast_mode24_v13_group_loss/evaluation_tasks'
spec=importlib.util.spec_from_file_location('v13_tasks',RUN/'run_tasks.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_task_grid_has_matched_4s_10s_runs_and_no_training():
    assert len(m.STAGES)==11 and len(set(m.STAGES))==11
    total=sum(64 if m.spec(s)['kind']=='fixedE' else 16 for s in m.STAGES)
    assert total==368
    assert {m.spec(s)['task'] for s in m.STAGES}=={'circle','zigzag'}
    assert all(m.spec(s)['seconds']==10 for s in m.STAGES if s.startswith('v12'))


def test_command_keeps_seed_kp_guidance_and_separates_fixed_from_joint():
    designs=['e000','e001']
    a=m.command('v13_fixedE_zigzag_10s',designs)
    assert a[a.index('--kp')+1]=='200' and a[a.index('--seconds')+1]=='10'
    assert a[a.index('--seed')+1]=='0' and a[a.index('--scales')+1]=='10'
    assert a[a.index('--fixed-e')+1:]==designs
    b=m.command('v13_joint_circle_4s',designs)
    assert '--fixed-e' not in b and '--no-render' not in b
    assert all(s in b for s in ('--joint-full-guidance','--e-rest-s0','--no-s0-align'))
