"""check/slide 결과로 GitHub 이슈를 만들고 결과 표를 댓글로 남긴다 (토큰: ~/.config/codesign/github_token)."""
import json, urllib.request
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent
TOKEN = (Path.home() / '.config/codesign/github_token').read_text().strip()
API = 'https://api.github.com/repos/seung-ddak/codesign'


def call(method, url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method=method,
                                 headers={'Authorization': f'Bearer {TOKEN}', 'Accept': 'application/vnd.github+json'})
    return json.load(urllib.request.urlopen(req))


rows = json.load(open(HERE / 'results/summary.json'))
body = ('형상 5 개 (E0 일반, E1 다리 짧음, E2 다리 긺, E3 몸통 짧음, E4 몸통 긺; ±20 %) × A1 대표 모션 5 개, '
        '형상·모션마다 PPO 정책 하나 (v7 전문가 설정: 동시 환경 32, 5.24M step, lr 1e-3, 미끄러짐 감점 없음, '
        '넘어지면 감점 없이 조기 종료, step 보상 ≥ 0).\n\n목적: 형상 1 × 모션 1 정책에서도 발이 미끄러지는지 확인.\n\n'
        '파일: `check/slide/` (영상 `video/E0~E4/`, 슬라이드 `video/slide.html`, 수치 `results/summary.json`). 브랜치 `experiment/h64-delta-trim`.')
issue = call('POST', f'{API}/issues', dict(title='형상별 모션 트래킹 점검: 발 미끄러짐 (check/slide)', body=body))
lines = ['| 형상 | 모션 | 생존 | 발 미끄러짐 (m/s) | 레퍼런스 미끄러짐 | 관절 RMSE (°) | 몸통 위치 RMSE (cm) | 이동 비율 | 레퍼런스 무리 프레임 |',
         '|---|---|---|---|---|---|---|---|---|']
for r in rows:
    lines.append(f"| {r['shape']} | {r['motion'].replace('__r00', '')} | {'○' if r['survived'] else '×'} {r['seconds']:.1f}/{r['clip_seconds']:.1f} s | "
                 f"{r['slip_m_s']:.3f} | {r['ref_slip_m_s']:.3f} | {r['joint_rmse_deg']:.1f} | {r['root_rmse_cm']:.1f} | {r['travel_ratio']:.2f} | {r['ref_infeasible_frac'] * 100:.0f} % |")
call('POST', issue['comments_url'], dict(body='학습·평가 완료. 결과 (외란 없이 모션 처음부터 끝까지):\n\n' + '\n'.join(lines)))
print(issue['html_url'])
