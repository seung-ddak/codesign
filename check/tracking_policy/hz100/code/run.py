"""제어 100 Hz 모션트랙킹 확인 (2026-11-17 사용자 요청: hz50 과 같은 구성). s6c50 (30 Hz / 240 Hz, 50M) 과 같은 모션·형상·학습량으로 100 Hz / 200 Hz 학습 → 같은 기준 평가.
모션 = s6c50 실패 2 (통과 0/9) + 통과 2 (9/9). 레퍼런스 = s5 A1 30 Hz 를 100 Hz 로 보간 (resample_reference) → 형상 9 개 리타겟.
학습량 20M transition (hz50 와 같음. 시뮬레이션 시간은 30 Hz 의 0.3 배). max_episode_length 2000 (20 초).
출력: a1/, refs/<형상>/00000011/, runs/<모션>/, results.json"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())
sys.path.insert(0, str(ROOT / 'chain/code')); sys.path.insert(0, str(ROOT / 'code/motion_tracking'))
import s6_train as s6, s6c_calm as s6c
from common import log
# 2026-11-17 사용자 결정: 빠른 확인으로 축소 (모션 2 개, 20M). 30 Hz 비교 = s6c50 의 20M 체크포인트 (같은 학습량) 를 같은 기준으로 평가.
TARGETS = [('trotCCWSlow__r00', 'calm_round_00_00939', '30Hz 실패 0/9'), ('trotCWSlow__r00', 'calm_round_02_00546', '30Hz 통과 9/9')]
HZ, PHYS, STEPS = 100, 200, 20_000_000


def main():
    from plan_calm_dynamics import resample_reference
    import pd_replay as pd
    shapes = json.load(open(s6.S5 / 'assignment.json'))['shapes']; a1 = HERE / 'a1'; a1.mkdir(exist_ok=True)
    for _, c, _ in TARGETS:
        if not (a1 / f'{c}.npz').exists(): resample_reference(s6.S5 / 'a1_reference' / f'{c}.npz', a1 / f'{c}.npz', hz=HZ)
    for s, v in shapes.items():
        d = HERE / 'refs' / s; todo = [c for _, c, _ in TARGETS if not (d / '00000011' / f'{c}.npz').exists()]
        if todo: pd.retarget_tt(str(a1), str(d), only=todo, choices=['00000011'], uniforms=v['E'], torque_limit=33.5)
    REF30 = s6c.REFS; s6c.REFS = HERE / 'refs'; rf = HERE / 'results.json'; res = json.loads(rf.read_text()) if rf.exists() else {}
    for g, c, why in TARGETS:
        if c in res: continue
        f = s6c.build(g, c, shapes, out=HERE / 'runs', steps=STEPS); p = json.load(open(f / 'plan.json'))
        p['config'].update(train_transitions=STEPS, checkpoint_steps=5_000_000, control_hz=HZ, physics_hz=PHYS, max_episode_length=2000); json.dump(p, open(f / 'plan.json', 'w'), indent=1)
        ok = (f / 'done').exists() or s6.train(f); rr = s6c.evaluate(f, c, shapes) if ok else []
        res[c] = dict(group=g, clip=c, why=why, failed=not ok, results=rr)
        log(f"hz100 {c} ({why}) 100Hz 통과 {sum(r['passed'] for r in rr)}/{len(rr)}"); rf.write_text(json.dumps(res, indent=1, default=float))
    # 30 Hz 비교: s6c50 의 20M 근처 체크포인트를 같은 기준 (evalpol, 최대 s6c.MAXS 초) 으로 평가
    import evalpol
    from s7_eval import ckpts
    bf = HERE / 'baseline30.json'; base = json.loads(bf.read_text()) if bf.exists() else {}
    for g, c, why in TARGETS:
        if c in base: continue
        ck = min(ckpts(ROOT / 'chain/runs/s6c50_calm' / g / c), key=lambda k: abs(int(Path(k).name) - STEPS))
        rr = [dict(evalpol.evaluate(ck, v['E'], REF30 / s / '00000011', c, multi=True, max_seconds=s6c.MAXS), shape=s, checkpoint=ck) for s, v in shapes.items()]
        for r in rr: r['passed'] = evalpol.passes(r)
        base[c] = rr; bf.write_text(json.dumps(base, indent=1, default=float))
        print(f"30Hz {c} {Path(ck).name} 통과 {sum(r['passed'] for r in rr)}/9", flush=True)
    for g, c, why in TARGETS:
        a, b = base[c], res[c]['results']; import numpy as np
        print(c, why, '| 30Hz', sum(r['passed'] for r in a), f"관절 {np.median([r['joint_rmse_deg'] or 99 for r in a]):.1f}", '| 100Hz', sum(r['passed'] for r in b), f"관절 {np.median([r['joint_rmse_deg'] or 99 for r in b]):.1f}", flush=True)
    print('END')


if __name__ == '__main__':
    main()
