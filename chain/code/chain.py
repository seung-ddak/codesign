"""연속 실험 체인 (2026-11-12 사용자 결정). tmux 에서 실행: tmux new-session -d -s chain "bash -c '... python chain/code/chain.py'"
순서: 짧은 다리 실험 종료 대기 → 1 그래프·하한 [결정: 형상 격자] → 4 CALM 재구성 → 5 리타겟 → 6 공유 정책 학습 → 7 평가 [결정: 수집 형상 수]
→ 8 rollout 수집 [결정: TT 인자] → 9 TT 학습. 단계마다 GitHub 이슈에 보고서 댓글. 결정 지점에서는 '승인' 댓글을 기다린다.
끝난 단계는 state.json 에 기록되어, 다시 실행하면 이어서 한다. 단계가 실패하면 실패 댓글을 달고 멈춘다."""
import json, os, subprocess, sys, time, traceback
# 2026-11-13: s7·s8 평가가 체인 프로세스 안에서 JAX 를 써서 GPU 75 % (8.9 GiB) 를 선점·유지 → s9 TT 학습 OOM. 선점 끔.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from pathlib import Path
from common import ROOT, RUNS, CHAIN, log, state, save_state, publish
import gh

STAGES = ['wait_short_leg', 's1', 's4', 's5', 's6a', 's6', 's7', 's8', 's6c', 's9']


def done(s, k): return k in s.get('done', [])


def mark(s, k, **kv):
    s.setdefault('done', []).append(k); s.setdefault('values', {}).update(kv); save_state(s)


def gate(s, key, stage, out_dir, summary, images, defaults):
    """결정 댓글을 달고 승인을 기다린다. 반환: 적용 값 dict."""
    if key in s.get('values', {}): return s['values'][key]
    if f'{key}_comment' not in s:
        txt = '\n'.join(f'- `{k}={v}`' for k, v in defaults.items())
        s[f'{key}_comment'] = publish(stage, out_dir, summary, images, decision=f'기본값:\n{txt}'); save_state(s)
    kind, kv = gh.wait_approval(s['issue'], s[f'{key}_comment'], log=log)
    if kind == 'stop':
        gh.comment(s['issue'], f'`중단` 확인. 체인을 {stage} 에서 끝낸다.'); log('사용자 중단'); sys.exit(0)
    val = dict(defaults, **kv)
    gh.comment(s['issue'], f'승인 확인 ({stage}). 적용 값: ' + ', '.join(f'`{k}={v}`' for k, v in val.items()))
    s.setdefault('values', {})[key] = val; save_state(s)
    return val


S6C_NOTE = ('2026-11-14 사용자 결정으로 267 개 중 199 개에서 중단. 단독 정책 5M step 은 CALM 모션에 부족 (보상이 오르는 중에 종료, '
            '공유 정책이 통과하는 CW·CCW 모션도 대부분 실패) → "모션 문제" 판정은 학습량 부족 때문이며 모션 품질 판단에 쓰지 않는다.')


def extend(out, steps):
    """2026-11-13: 정책 폴더들의 학습량을 steps 로 늘리고 done 을 지워 이어서 학습 (learner.msgpack 에서 재개)."""
    for f in sorted(Path(out).glob('trot*')):
        p = json.load(open(f / 'plan.json')); old = p['config']['train_transitions']
        if old >= steps: continue
        p['config']['train_transitions'] = steps; p.setdefault('notes', []).append(f"{time.strftime('%Y-%m-%d')} train_transitions {old} -> {steps} (사용자 결정, 체인 retrain)")
        json.dump(p, open(f / 'plan.json', 'w'), indent=1)
        if (f / 'done').exists(): (f / 'done').unlink()
    log(f'연장 {Path(out).name} → {steps / 1e6:.0f}M')


def archive(dirs):
    dst = RUNS / f"_old_{time.strftime('%Y%m%d_%H%M')}"; dst.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        if Path(d).exists(): Path(d).rename(dst / Path(d).name)
    log(f'보관 → {dst}')


def nums(x): return [float(v) for v in str(x).split(',')]


def main():
    s = state(); s.setdefault('run_tag', time.strftime('%Y%m%d')); save_state(s)
    try:
        if not done(s, 'wait_short_leg'):
            Ls = [ROOT / 'check/short_leg/after.log', ROOT / 'check/short_leg/logs/after.log']  # 정리 전·후 위치
            while not any(L.exists() and any(x.startswith(('END', 'FAIL')) for x in L.read_text().splitlines()) for L in Ls):
                time.sleep(300)
            mark(s, 'wait_short_leg')
        if not done(s, 's1'):
            import s1_shape_graph as s1
            r = s1.main(); lo = r['lower_leg']; pct = {'E0': 0}.get(lo, -int(lo[1:]) if lo else -10)
            summ = (f"짧은 다리 실험 결과로 다리 하한을 정했다. 규칙: squat·후진 중 하나라도 실패한 가장 작은 축소 비율의 바로 앞 단계 → **{lo}**"
                    + (' (실패가 들쭉날쭉: 자동 판정 보류, 값을 확인해 줘)' if r['irregular'] else ''))
            v = gate(s, 'grid', 's1_shape_graph', s1.OUT, summ, r['images'], dict(legs='-20,0,20', bodies='-20,0,20'))  # 2026-11-12 사용자 결정: 다리 하한 L20 (seed 차이로 L10 실패), 몸통 폭은 바꾸지 않음
            mark(s, 's1')
        legs, bodies = nums(s['values']['grid']['legs']), nums(s['values']['grid']['bodies'])
        if not done(s, 's4'):
            import s4_calm as s4
            s4.main(); publish('s4_calm', s4.OUT / 'report', '조이스틱 방식 CALM 모션 재구성 완료 (trot 레퍼런스 6 개, 실제 속도 분류, 할당량 1,000).', [s4.OUT / 'report/counts.png'])
            mark(s, 's4')
        if not done(s, 's5'):
            import s5_retarget as s5
            r = s5.main(legs, bodies); publish('s5_retarget', s5.OUT, f"학습 형상 {len(r['shapes'])} 개 리타겟 완료 (누락 {r['missing']}).", [])
            mark(s, 's5')
        if not done(s, 's6a'):  # 2026-11-12 추가: 레퍼런스 1 개 × 형상 전부
            import s6a_ref as s6a
            st = s6a.main(); publish('s6a_ref', s6a.OUT, f"레퍼런스 단독 정책 학습: 완료 {sum(st.values())}/{len(st)}.", [s6a.OUT / 'curves.png'])
            mark(s, 's6a')
        if not done(s, 's6'):
            import s6_train as s6
            st = s6.main(); publish('s6_train', s6.OUT, f"공유 정책 학습: 완료 {sum(st.values())}/{len(st)}.", [s6.OUT / 'curves.png'])
            mark(s, 's6')
        if not done(s, 's7'):
            import s7_eval as s7
            sel = s7.main()
            v = gate(s, 'collect', 's7_eval', s7.OUT, f'체크포인트 선택·평가 완료 (정책 {len(sel)} 개). 다음 = 학습하지 않은 연속 형상에서 rollout 수집.',
                     [s7.OUT / 'pass_map.png'], dict(n_shapes=20, per_group=100))
            mark(s, 's7')
        if not done(s, 's8'):
            import s8_collect as s8
            c = s['values']['collect']; files = s8.main(legs, bodies, int(c['n_shapes']), int(c['per_group']))
            import s9_tt
            v = gate(s, 'tt', 's8_collect', s8.OUT, f'rollout 수집 완료 ({len(files)} 파일). 다음 = TT 학습.', [s8.OUT / 'shapes.png'], dict(tt_args=s9_tt.DEFAULT_ARGS.replace(' ', '_')))
            mark(s, 's8')
        if not done(s, 's6c'):  # 2026-11-13 사용자 결정: CALM 모션 1 개 × 학습 형상 9 개 정책 (s7 CALM 전부)
            import s6c_calm as s6c
            g = s6c.main(); n = sum(d['n'] for d in g.values()); m = sum(d['motion'] for d in g.values()); pol = sum(d['policy'] for d in g.values())
            gate(s, 'after_s6c', 's6c_calm', s6c.OUT,
                 f'CALM 단독 정책 {n} 개 학습·평가 완료. 모션 문제 (단독도 통과 < 50 %) {m} 개, 공유 정책 문제 (단독 통과, s7 < 50 %) {pol} 개.\n\n'
                 '승인하면: s6 공유 정책 연장 → s7 → s8 → s6c 비교 갱신 → s6a 연장 → s9 전 결정. 값 0 = 그 학습 건너뜀.',
                 [s6c.OUT / 'scatter.png'], dict(s6_steps=100_000_000, s6a_steps=80_000_000))
            mark(s, 's6c')
        if not done(s, 'retrain'):  # 2026-11-13 사용자 결정: s6c 결과 보고 승인 → 재학습 연속 진행
            import s6_train as s6, s6a_ref as s6a, s6c_calm as s6c, s7_eval as s7, s8_collect as s8
            v = s['values']['after_s6c']; n6, n6a = int(float(v['s6_steps'])), int(float(v['s6a_steps']))
            if n6 > 0:
                if not done(s, 'retrain_s6'):
                    extend(s6.OUT, n6); st = s6.main()
                    publish('s6_train', s6.OUT, f'공유 정책 연장 학습 ({n6 / 1e6:.0f}M): 완료 {sum(st.values())}/{len(st)}.', [s6.OUT / 'curves.png'])
                    archive([s7.OUT, s8.OUT, RUNS / 's9_tt']); mark(s, 'retrain_s6')
                if not done(s, 'retrain_s7'):
                    sel = s7.main(); publish('s7_eval', s7.OUT, f'연장 학습 정책 평가 완료 (정책 {len(sel)} 개).', [s7.OUT / 'pass_map.png']); mark(s, 'retrain_s7')
                if not done(s, 'retrain_s8'):
                    c = s['values']['collect']; files = s8.main(legs, bodies, int(c['n_shapes']), int(c['per_group']))
                    publish('s8_collect', s8.OUT, f'연장 학습 정책으로 rollout 재수집 ({len(files)} 파일).', [s8.OUT / 'shapes.png']); mark(s, 'retrain_s8')
                if not done(s, 'retrain_s6c'):
                    s6c.make_report(json.loads((s6c.OUT / 'results.json').read_text()), note=S6C_NOTE)
                    publish('s6c_calm', s6c.OUT, 's6c 비교 갱신 (새 s7 공유 정책 기준).', [s6c.OUT / 'scatter.png']); mark(s, 'retrain_s6c')
            if n6a > 0 and not done(s, 'retrain_s6a'):
                extend(s6a.OUT, n6a); st = s6a.main()
                publish('s6a_ref', s6a.OUT, f'레퍼런스 단독 정책 연장 학습 ({n6a / 1e6:.0f}M): 완료 {sum(st.values())}/{len(st)}.', [s6a.OUT / 'curves.png'])
                mark(s, 'retrain_s6a')
            mark(s, 'retrain')  # 2026-11-14 사용자 결정: s9 전 결정 없이 이전 TT 인자로 바로 진행
        if not done(s, 's9'):
            import s9_tt
            s9_tt.main(s['values']['tt']['tt_args'].replace('_', ' ') if s['values']['tt']['tt_args'] else s9_tt.DEFAULT_ARGS)
            publish('s9_tt', s9_tt.OUT, 'TT 학습 완료.', []); mark(s, 's9')
        if not done(s, 's6c50'):  # 2026-11-14 사용자 결정: s9 뒤 CALM 단독 정책 50M, 묶음당 4 개 (s7 실패 2 + 통과 2)
            import s6c50
            g = s6c50.main(); publish('s6c50_calm', s6c50.OUT, g, [s6c50.OUT / 'scatter.png']); mark(s, 's6c50')
        gh.comment(s['issue'], '체인 전체 완료.'); log('체인 완료')
    except Exception as e:
        tb = traceback.format_exc(); log(tb)
        try:
            if 'issue' in s: gh.comment(s['issue'], f'**체인 실패** — 멈춤.\n\n```\n{tb[-3000:]}\n```')
        except Exception: pass
        raise


if __name__ == '__main__':
    main()
