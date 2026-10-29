import sys
from pathlib import Path
import json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/tt_dynamics'))
import run_v7_ablations as runner


def test_issue_notifications_are_persisted_and_not_resent(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    token=tmp_path/'token';token.write_text('test-token');token.chmod(0o600)
    monkeypatch.setattr(runner,'TOKEN_FILE',token)
    called=[]
    def post(path,data,secret):
        assert path=='/issues/8/comments' and secret=='test-token'
        called.append(data);return {'id':42}
    monkeypatch.setattr(runner,'post',post)
    out=tmp_path/'run';out.mkdir()
    runner.record(out,8,'测试完成',{'status':'completed_with_screen_failures'})
    runner.flush_notifications(out,8)
    assert len(called)==1
    event=json.loads(next((out/'events').glob('*.json')).read_text())
    assert event['comment_id']==42 and 'test-token' not in json.dumps(event)


def test_failed_notification_keeps_event_for_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    token=tmp_path/'token';token.write_text('test-token');token.chmod(0o600)
    monkeypatch.setattr(runner,'TOKEN_FILE',token)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    def fail(*args):raise OSError('offline')
    monkeypatch.setattr(runner,'post',fail)
    out=tmp_path/'run';out.mkdir()
    with pytest.raises(RuntimeError,match='retained'):
        runner.record(out,8,'Failure',{'status':'execution_error'})
    event=json.loads(next((out/'events').glob('*.json')).read_text())
    assert event['comment_id'] is None and event['notification_error']=='offline'


def test_report_handles_a_stage_error_without_claiming_success(tmp_path):
    (tmp_path/'events').mkdir()
    (tmp_path/'events/001.json').write_text(json.dumps({'title':'Training failed','data':{'status':'execution_error'}}))
    runner.report(tmp_path)
    text=(tmp_path/'report.html').read_text()
    assert 'execution_error' in text and '75%' in text
