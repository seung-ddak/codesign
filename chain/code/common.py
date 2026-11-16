"""체인 공용: 경로, 로그, 상태 파일, HTML 보고서, 단계 결과 게시."""
import base64, html, json, os, subprocess, sys, time
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent; CHAIN = CODE_DIR.parent; ROOT = CHAIN.parent  # 코드 = chain/code, 결과 = chain/runs, 로그 = chain/logs
PY = str(ROOT.parent / 'transformer-transformer/.venv/bin/python')
CODE = ROOT / 'code/motion_tracking'
RUNS = CHAIN / 'runs'; RUNS.mkdir(exist_ok=True)
STATE = CHAIN / 'state.json'
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE_DIR))


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    (CHAIN / 'logs').mkdir(exist_ok=True)
    with open(CHAIN / 'logs/chain.log', 'a') as f: f.write(line + '\n')


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(s, indent=1, ensure_ascii=False)); tmp.replace(STATE)


def run(cmd, logfile, cwd=ROOT, env=None):
    """명령 실행, 출력은 logfile 로. 실패하면 RuntimeError."""
    with open(logfile, 'a') as fh:
        r = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})))
    if r.returncode != 0:
        raise RuntimeError(f'{cmd[:3]} exit {r.returncode} (log {logfile})')


CSS = """:root{--bg:#f7f7f5;--panel:#fff;--fg:#1c1c1a;--muted:#62615c;--line:#dedcd6;--bad:#fde8e8;--good:#e8f5e9}
@media (prefers-color-scheme: dark){:root{--bg:#16171a;--panel:#1f2024;--fg:#ececea;--muted:#a3a29c;--line:#34353a;--bad:#3a1f1f;--good:#1f3324}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,"Noto Sans KR",sans-serif}
main{max-width:1400px;margin:0 auto;padding:16px}
h1{font-size:21px;margin:0 0 6px} h2{font-size:17px;margin:22px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.muted{color:var(--muted)} .wrap{overflow-x:auto}
table{border-collapse:collapse;background:var(--panel);font-variant-numeric:tabular-nums}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap} th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:12px} tr.bad td{background:var(--bad)} tr.good td{background:var(--good)}
img{max-width:100%;background:#fff;border-radius:6px} video{width:100%;max-width:640px;border-radius:6px;background:#000}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px} figure{margin:0} figcaption{color:var(--muted);font-size:12px}
ul{margin:4px 0}
.flt{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}
.flt select{font:inherit;padding:4px 8px;margin-left:6px;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px}"""


def table(headers, rows, row_class=None, filter_col=None):
    """filter_col: 이 열 값으로 보고서 상단 드롭다운 필터 (모션 등)."""
    head = ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)
    body = ''
    for i, r in enumerate(rows):
        cls = f' class="{row_class(i)}"' if row_class and row_class(i) else ''
        f = f' data-f="{html.escape(str(r[filter_col]))}"' if filter_col is not None else ''
        body += f'<tr{cls}{f}>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in r) + '</tr>'
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def img(path, caption=''):
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def video(rel, caption='', tag=None):
    f = f' data-f="{html.escape(tag)}"' if tag else ''
    return f'<figure{f}><video src="{rel}" controls muted loop playsinline preload="metadata"></video><figcaption>{html.escape(caption)}</figcaption></figure>'


def report(path, title, sections, conditions):
    """sections: [(제목, html)], conditions: [(항목, 값)] 고정 환경·조건 표."""
    cond = table(['항목', '값'], conditions)
    body = ''.join(f'<h2>{html.escape(t)}</h2>{h}' for t, h in sections)
    import re
    tags = sorted(set(re.findall(r'data-f="([^"]+)"', body)))
    flt = ''
    if tags:  # 모션별 보기: 표 행·영상 중 data-f 가 있는 것만 거른다
        opts = '<option value="">전체</option>' + ''.join(f'<option>{t}</option>' for t in tags)
        flt = (f'<div class="flt"><label>모션 <select id="flt">{opts}</select></label></div>'
               '<script>document.getElementById("flt").onchange=e=>{const v=e.target.value;'
               'document.querySelectorAll("[data-f]").forEach(x=>x.style.display=(!v||x.dataset.f===v)?"":"none")};</script>')
    Path(path).write_text(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                          f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main><h1>{html.escape(title)}</h1>'
                          f'<p class="muted">생성 {time.strftime("%F %T")}</p>{flt}<h2>고정 환경·조건</h2>{cond}{body}</main></body></html>')


def publish(stage, out_dir, summary_md, images=(), decision=None):
    """단계 결과를 이슈에 게시. out_dir 를 reports/<stage>/ 에 올리고, 댓글 = 요약 + 그림 + 보고서 링크 (+ 결정 요청).
    반환: 댓글 id."""
    import gh
    s = state()
    if 'issue' not in s:
        body = ('연속 실험 체인 (codesign/chain). 단계마다 결과 보고서 (HTML) 를 댓글로 올린다.\n\n'
                '결정이 필요한 단계에서는 체인이 멈추고 기다린다. 이어가려면 `승인` (기본값) 또는 `승인 key=value ...` , 끝내려면 `중단` 으로 시작하는 댓글을 단다 (10 분마다 확인).')
        s['issue'] = gh.create_issue(f"연속 실험 체인 {time.strftime('%Y-%m-%d')}", body)['number']; save_state(s)
    files = gh.upload_dir(out_dir, f"{s.get('run_tag', time.strftime('%Y%m%d'))}/{stage}")
    rep = files.get('report.html')
    md = f'## {stage}\n\n{summary_md}\n\n'
    for p in images:
        if Path(p).name in files: md += f'![{Path(p).stem}]({files[Path(p).name][1]})\n\n'
    if rep: md += f'**보고서**: [{stage}/report.html]({rep[0]}) (Download raw file → 브라우저로 열기. 영상은 같은 폴더의 videos/)\n\n'
    if decision:
        md += '### 결정 요청 (체인 대기 중)\n\n' + decision + '\n\n`승인` = 위 기본값으로 진행 · `승인 key=value ...` = 값 바꿔 진행 · `중단` = 체인 종료\n'
    c = gh.comment(s['issue'], md)
    log(f'게시 {stage} → issue #{s["issue"]} comment {c["id"]}')
    return c['id']
