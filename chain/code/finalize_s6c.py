"""2026-11-14 사용자 결정 (선택 1): s6c 를 199/267 에서 중단. 지금까지 결과로 영상·보고서를 만들어 게시하고,
결정 ① (after_s6c) 을 기본값 (s6 100M, s6a 80M) 으로 승인된 것으로 기록 → 체인이 retrain 부터 이어간다."""
import json
from common import state, save_state, publish, log
import s6c_calm as s6c
from chain import S6C_NOTE
res = json.loads((s6c.OUT / 'results.json').read_text())
shapes = json.load(open(s6c.s6.S5 / 'assignment.json'))['shapes']
s6c.videos(res, shapes); g = s6c.make_report(res, note=S6C_NOTE)
n = sum(d['n'] for d in g.values())
publish('s6c_calm', s6c.OUT, f'CALM 단독 정책 {n}/267 개에서 중단 (사용자 결정). 단독 5M 은 CALM 에 부족: 공유 정책 (50M) 보다 모든 묶음에서 나쁨 → 모션 품질 판단 불가.\n\n'
        '사용자 결정: s6 100M · s6a 80M 재학습으로 진행 (s6 → s7 → s8 → s6c 비교 갱신 → s6a → s9 전 결정).', [s6c.OUT / 'scatter.png'])
s = state(); s.setdefault('values', {})['after_s6c'] = dict(s6_steps=100_000_000, s6a_steps=80_000_000)
if 's6c' not in s['done']: s['done'].append('s6c')
save_state(s); log('s6c 중단·게시, after_s6c = s6 100M · s6a 80M (사용자 결정 11-14)')
