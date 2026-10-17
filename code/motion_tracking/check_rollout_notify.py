"""Offline check of the actual workflow: data commits only, safe reruns, API failure."""
import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

root = Path(__file__).resolve().parents[2]
workflow = yaml.safe_load((root / '.github/workflows/rollout-notify.yml').read_text())
run = workflow['jobs']['notify']['steps'][0]['run']
source = run.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
commits = [dict(sha=s, html_url='https://example.com/' + s, commit={'message': s})
           for s in ('data', 'code', 'deleted')]
posted = []
comments = []

def respond(req, timeout):
    url = req.full_url
    if req.data:
        posted.append(json.loads(req.data)['body'])
        return io.BytesIO(b'{}')
    if '/comments?' in url:
        result = [{'body': body} for body in comments]
    elif '/compare/' in url:
        result = {'commits': commits}
    else:
        sha = url.rsplit('/', 1)[-1]
        result = {'files': [dict(filename='code/a.py' if sha == 'code' else
                                'runs/pipeline_v1/rollouts/e000/seq_seed0.npz',
                                status='removed' if sha == 'deleted' else 'added')]}
    return io.BytesIO(json.dumps(result).encode())

with tempfile.TemporaryDirectory() as tmp:
    event = Path(tmp) / 'event.json'
    event.write_text(json.dumps({'before': 'old', 'after': 'new'}))
    with patch.dict(os.environ, GITHUB_REPOSITORY='test/repo', GITHUB_EVENT_PATH=str(event), GH_TOKEN='fake'), patch('urllib.request.urlopen', respond):
        exec(source, {})
        assert len(posted) == 1 and 'rollout-commit:data' in posted[0]
        comments.extend(posted)
        exec(source, {})
        assert len(posted) == 1
    with patch.dict(os.environ, GITHUB_REPOSITORY='test/repo', GITHUB_EVENT_PATH=str(event), GH_TOKEN='fake'), patch('urllib.request.urlopen', side_effect=OSError('offline')):
        try:
            exec(source, {})
        except OSError:
            pass
        else:
            raise AssertionError('API errors must fail the workflow')
print('PASS: data-only notification, rerun deduplication, API failure propagation')
