"""GitHub 이슈·댓글·보고서 게시·승인 대기 (토큰: ~/.config/codesign/github_token, 저장소 seung-ddak/codesign).
보고서·그림·영상은 별도 작업 사본 (~/.cache/codesign_reports) 에서 `reports` 브랜치로 SSH push (토큰에 contents 권한 없음). codesign 작업 폴더 git 은 건드리지 않는다.
승인: 체인이 결정 댓글을 단 뒤, 저장소 소유자·협업자가 '승인' (기본값) / '승인 key=value ...' / '중단' 으로 시작하는 댓글을 달면 읽는다."""
import json, time, urllib.error, urllib.request
from pathlib import Path
OWNER, REPO, BRANCH = 'seung-ddak', 'codesign', 'reports'
API = f'https://api.github.com/repos/{OWNER}/{REPO}'
TOKEN = (Path.home() / '.config/codesign/github_token').read_text().strip()


def call(method, url, body=None):
    url = url if url.startswith('http') else API + url
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={'Authorization': f'Bearer {TOKEN}', 'Accept': 'application/vnd.github+json'})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read() or b'null')
        except urllib.error.HTTPError as e:
            if e.code in (404, 409, 422) or k == 4: raise
        except urllib.error.URLError:
            if k == 4: raise
        time.sleep(10 * (k + 1))


REPO_DIR = Path.home() / '.cache/codesign_reports'  # 보고서 전용 작업 사본 (codesign 작업 폴더의 git 은 건드리지 않음)
SSH = f'git@github.com:{OWNER}/{REPO}.git'


def _git(*a):
    import subprocess
    subprocess.run(['git', '-C', str(REPO_DIR), *a], check=True, capture_output=True, timeout=600)


def _ensure_repo():
    import subprocess
    if (REPO_DIR / '.git').exists():
        return
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    _git('init', '-q'); _git('remote', 'add', 'origin', SSH)
    r = subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', '-q', '--depth', '1', 'origin', BRANCH], capture_output=True)
    if r.returncode == 0:
        _git('checkout', '-q', '-b', BRANCH, 'FETCH_HEAD')
    else:
        _git('checkout', '-q', '--orphan', BRANCH)
        (REPO_DIR / 'README.md').write_text('연속 실험 체인 보고서 (codesign/chain 이 자동으로 올림).\n'); _git('add', 'README.md')
        _git('-c', 'user.name=codesign-chain', '-c', 'user.email=chain@localhost', 'commit', '-q', '-m', 'reports branch')


def upload_dir(local_dir, remote_dir, exts=('.html', '.png', '.mp4', '.json', '.csv')):
    """폴더 안 파일들을 reports 브랜치 remote_dir 에 복사·commit·push (SSH). 50 MB 넘는 파일 제외. 반환: {상대경로: (blob URL, raw URL)}."""
    import shutil
    _ensure_repo(); out = {}
    for f in sorted(Path(local_dir).rglob('*')):
        if f.is_file() and f.suffix in exts and f.stat().st_size < 50e6 and 'references' not in f.parts:
            rel = f.relative_to(local_dir).as_posix(); dst = REPO_DIR / remote_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, dst)
            out[rel] = (f'https://github.com/{OWNER}/{REPO}/blob/{BRANCH}/{remote_dir}/{rel}',
                        f'https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{remote_dir}/{rel}')
    _git('add', '-A', remote_dir)
    import subprocess
    if subprocess.run(['git', '-C', str(REPO_DIR), 'diff', '--cached', '--quiet']).returncode != 0:
        _git('-c', 'user.name=codesign-chain', '-c', 'user.email=chain@localhost', 'commit', '-q', '-m', f'report: {remote_dir}')
    for k in range(3):
        try:
            _git('push', '-q', 'origin', BRANCH); break
        except Exception:
            if k == 2: raise
            time.sleep(30)
    return out


def create_issue(title, body):
    return call('POST', '/issues', dict(title=title, body=body))


def comment(issue_number, body):
    return call('POST', f'/issues/{issue_number}/comments', dict(body=body))


def wait_approval(issue_number, since_comment_id, poll=600, log=print):
    """since_comment_id 이후 소유자 댓글 중 '승인' 또는 '중단' 으로 시작하는 첫 댓글을 기다린다.
    반환: ('approve', {key: value}) 또는 ('stop', {})."""
    while True:
        page = 1
        while True:
            cs = call('GET', f'/issues/{issue_number}/comments?per_page=100&page={page}')
            for c in cs:
                if c['id'] <= since_comment_id or c.get('author_association') not in ('OWNER', 'MEMBER', 'COLLABORATOR'): continue
                text = c['body'].strip()
                if text.startswith('중단'):
                    return 'stop', {}
                if text.startswith('승인'):
                    kv = dict(t.split('=', 1) for t in text[2:].split() if '=' in t)
                    return 'approve', kv
            if len(cs) < 100: break
            page += 1
        log(f'승인 대기 중 (issue #{issue_number})')
        time.sleep(poll)


def edit_comment(comment_id, body):
    """기존 댓글 내용을 바꾼다 (후속 결과는 새 댓글 대신 기존 댓글 수정, 2026-11-12 사용자 요청)."""
    return call('PATCH', f'/issues/comments/{comment_id}', dict(body=body))


def get_comment(comment_id):
    return call('GET', f'/issues/comments/{comment_id}')
