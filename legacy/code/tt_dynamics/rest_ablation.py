"""Controlled regression comparison: preserve measured s0 and append settled morphology features."""
import hashlib
import html
import json
from pathlib import Path
import time
import numpy as np
from measure_e_signal_lib import NAMES


def feature_sets(e, s, a, g):
    assert len(e) == len(s) == len(a) == len(g)
    assert g.shape[1] == s.shape[1] == 42
    assert all(np.isfinite(v).all() for v in (e, s, a, g))
    return {
        's0_A': np.concatenate([s, a], 1),
        'E_s0_A': np.concatenate([e, s, a], 1),
        'g_s0_A': np.concatenate([g, s, a], 1),
        'E_g_s0_A': np.concatenate([e, g, s, a], 1),
    }


def gate(candidate, baseline):
    return {
        'gain_vs_group_mean_ge_20pct': candidate['gain_vs_b0'] >= .20,
        'effect_sign_ge_80pct': candidate['sign_agree'] >= .80,
        'rmse_better_than_E_s0_A': candidate['rmse'] < baseline['rmse'],
    }


def run(args, designs, X_E, X_S, X_A, Y, GI, DI, folds, fit, metrics, group_mean):
    start = time.monotonic()
    with np.load(args.rest_features, allow_pickle=False) as z:
        assert z['designs'].tolist() == designs, 'rest feature design order mismatch'
        rest = z['rest_s0'].copy()
        assert rest.shape == (len(designs), 42)
        assert np.allclose(z['E'], np.stack([X_E[DI == d][0] for d in range(len(designs))]), atol=1e-7)
    feats = feature_sets(X_E, X_S, X_A, rest[DI])
    mag = np.abs(Y).mean(0)
    result = {'models': {m: int(x.shape[1]) for m, x in feats.items()}, 'samples': len(Y),
              'designs': designs, 'args': json.loads(json.dumps(vars(args), default=str)),
              'folds': {}, 'criteria': 'Primary design-fold vx: B0 gain >=20%, effect sign >=80%, RMSE lower than E_s0_A. No s0 replacement. g_s0_A is a representation control.',
              'note': 'Baseline B0 uses training-design means per group. PD actions are retargeted by shape; this is predictive morphology generalization, not a fixed-action causal intervention.'}
    saved = {'Y': Y, 'GI': GI, 'DI': DI, 's0': X_S, 'rest_s0': rest, 'designs': np.array(designs)}
    for fname, split in folds.items():
        pred = {m: np.zeros((args.seeds, *Y.shape), np.float32) for m in feats}
        b0 = np.full_like(Y, np.nan)
        fold_scores = []
        for fi, te in enumerate(split):
            tr = ~te
            if fname == 'design':
                assert not set(DI[tr]) & set(DI[te])
                b0[te] = group_mean(tr, te)
            for m, x in feats.items():
                for seed in range(args.seeds):
                    cache = args.out/'checkpoints'/f'{fname}_{fi}_{m}_{seed}.npy'
                    cache.parent.mkdir(exist_ok=True)
                    if cache.exists():
                        value = np.load(cache, allow_pickle=False)
                    else:
                        value = fit(x[tr], Y[tr], x[te], seed)
                        np.save(cache, value)
                    assert value.shape == Y[te].shape and np.isfinite(value).all()
                    pred[m][seed, te] = value
                print(f'{fname} fold {fi+1}/4 {m} done ({time.monotonic()-start:.0f}s)', flush=True)
            fold_scores.append({m: metrics(v[:,te].mean(0),Y[te],b0[te] if fname=='design' else None,mag) for m,v in pred.items()})
        base = b0 if fname == 'design' else None
        result['folds'][fname] = {'ensemble': {m: metrics(v.mean(0),Y,base,mag) for m,v in pred.items()},
                                  'per_seed': {m: [metrics(p,Y,base,mag) for p in v] for m,v in pred.items()}, 'per_fold': fold_scores}
        if base is not None:
            result['folds'][fname]['B0'] = metrics(b0,Y,None,mag)
        for m,v in pred.items(): saved[f'{fname}_{m}'] = v
        saved[f'{fname}_test_fold'] = sum((te.astype(int)*(i+1) for i,te in enumerate(split)))
        saved[f'{fname}_B0'] = b0
        # Preserve completed stages if a later stage is interrupted.
        (args.out/'results.partial.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    scores = result['folds']['design']['ensemble']
    vx = NAMES[0]
    result['gates'] = {m: gate(scores[m][vx], scores['E_s0_A'][vx]) for m in ('g_s0_A','E_g_s0_A')}
    result['pass'] = all(result['gates']['E_g_s0_A'].values())
    result['elapsed_s'] = time.monotonic()-start
    files = [p/'primitive_seed0.npz' for root in args.data for p in sorted(root.iterdir()) if (p/'primitive_seed0.npz').exists()]
    files += [args.rest_features, Path(__file__), Path(__file__).with_name('regression_gate.py')]
    result['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    np.savez_compressed(args.out/'predictions.npz', **saved)
    (args.out/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    rows = ''
    for fname, data in result['folds'].items():
        for m, values in data['ensemble'].items():
            v=values[vx]
            rows += '<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (fname,m,f"{v['rmse']:.4f}", f"{100*v['gain_vs_b0']:+.1f}%" if 'gain_vs_b0' in v else '—', f"{100*v['sign_agree']:.1f}%" if 'sign_agree' in v else '—'))+'</tr>'
    report = f'''<!doctype html><meta charset="utf-8"><title>PD 안착 자세 추가 입력 실험</title><style>body{{font:16px sans-serif;max-width:1000px;margin:32px auto;line-height:1.6}}td,th{{border:1px solid #aaa;padding:6px}}table{{border-collapse:collapse}}</style>
<h1>PD 안착 자세 추가 입력 실험</h1><p>[사실] 사전 기준 {'통과' if result['pass'] else '미달'}. 실제 초기 상태와 정답 궤적을 보존하고 입력 표현만 변경했음.</p>
<p>안 본 형상 4-fold (= 16형상을 4개씩 빼고 학습), seed 3개 평균. s0=실제 초기 상태, A=관절 목표각 행동 열, E=형상 좌표, g=home 자세에서 1초 안착한 상태. B0=같은 모션·시작 그룹의 학습 형상 평균.</p>
<p>사전 기준: E_g_s0_A 속도 오차가 B0보다 20% 이상 감소, 형상 효과 부호 80% 이상 일치, E_s0_A보다 오차 감소. 기존 실험의 다른 판정 기준은 이 실험에 소급 적용하지 않음.</p>
<table><tr><th>평가</th><th>입력</th><th>전진 속도 RMSE (m/s)</th><th>B0 대비 감소</th><th>형상 효과 부호</th></tr>{rows}</table>
<p>[한계] start 평가는 같은 모션의 안 본 시작 프레임임. 새로운 모션 일반화, TT 생성, 10초 MJX 성공은 검증하지 않음. 행동은 형상별 리타겟되므로 같은 행동에서 형상만 바꾼 인과 실험이 아님. 낙상 그룹을 제외한 예측 평가임.</p><p><a href="results.json">전체 지표·seed·fold별 결과</a> / <a href="predictions.npz">원단위 예측</a> / <a href="conditions.json">사전 조건</a></p>'''
    (args.out/'report.html').write_text(report)
    print(json.dumps({'pass':result['pass'],'gates':result['gates'],'design_vx':{m:v[vx] for m,v in scores.items()}},ensure_ascii=False),flush=True)
