"""Offline check: only notify success after verified results have been pushed."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import run_cgr as runner

for fail_push in (False, True):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / 'run'
        out.mkdir()
        (out / 'issue.json').write_text('{"number": 3}')
        token = root / 'token'
        token.write_text('fake')
        messages, calls = [], []
        def run(cmd, **kwargs):
            calls.append(cmd)
            if 'code/tt_dynamics/train_tt.py' in cmd:
                (out / 'progress.jsonl').write_text('{"step":100000,"loss":0.1}\n')
                (out / 'checkpoint.pt').write_bytes(b'fake')
                (out / 'config.json').write_text('{}')
                (out / 'eval_dynamics_ddim.json').write_text('{"all":{"error":1}}')
                (out / 'diag_window.json').write_text(json.dumps({'전체': {'error': 1}}))
            if cmd[:2] == ['git', 'push'] and fail_push:
                raise RuntimeError('simulated push failure')
        def output(cmd, **kwargs):
            return runner.BRANCH if cmd[1] == 'branch' else 'commit123'
        with patch.object(runner, 'ROOT', root), patch.object(runner, 'OUT', out), patch.object(runner, 'TOKEN_FILE', token), patch.object(runner.os, 'chdir'), patch.object(runner.subprocess, 'run', run), patch.object(runner.subprocess, 'check_output', output), patch.object(runner, 'post', lambda path, data, token: messages.append(data['body'])):
            try:
                runner.main()
            except RuntimeError:
                assert fail_push
        assert len(messages) == 1
        assert ('업로드 완료' in messages[0]) == (not fail_push)
        assert (out / 'completed.txt').exists() == (not fail_push)
        assert (out / 'failed.txt').exists() == fail_push
        assert any(c[:2] == ['git', 'push'] for c in calls)
print('PASS: successful publication and failed push notify correctly')
