"""8 단계: 학습하지 않은 연속 형상에서 rollout 수집 (생성 모델 학습 데이터).
형상: 학습 격자 범위 (다리 legs 최소~최대, 몸통 bodies 최소~최대) 안에서 학습 형상과 겹치지 않게 n_shapes 개 (seed 고정 farthest-point).
형상마다: 7 단계에서 고른 묶음 정책 각각으로 [레퍼런스 + 그 묶음 CALM 모션 per_group 개] 를 이 형상으로 리타겟해 실행 (MJX, vmap 256, 결정적 정책, 외란 없음).
저장 형식 = code/tt_dynamics/rollout_data.load 가 읽는 형식: valid, qpos, qvel, contact, target_q, E, motions, clip (+ fell, group, policy).
출력: runs/s8_collect/<형상>/<묶음>.npz, shapes.json, report.html"""
import functools, json, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from common import ROOT, RUNS, log, report, table, img, video
from shapes_grid import token, name, dims
OUT = RUNS / 's8_collect'; S5 = RUNS / 's5_retarget'; S6 = RUNS / 's6_train'; S7 = RUNS / 's7_eval'


def pick_shapes(legs, bodies, n, seed=0):
    rng = np.random.default_rng(seed); cand = np.c_[rng.uniform(min(legs), max(legs), 4000), rng.uniform(min(bodies), max(bodies), 4000)]
    chosen = [np.array([l, b], float) for l in legs for b in bodies]; k0 = len(chosen); scale = np.array([max(legs) - min(legs), max(bodies) - min(bodies)]) + 1e-9
    while len(chosen) - k0 < n:
        d = np.min(np.linalg.norm((cand[:, None] - np.array(chosen)[None]) / scale, axis=-1), axis=1); c = np.round(cand[d.argmax()])
        if not any((c == x).all() for x in chosen): chosen.append(c)
        cand = np.delete(cand, d.argmax(), 0)
    return [(int(c[0]), int(c[1])) for c in chosen[k0:]]  # 정수 % (형상 이름 중복 방지)


def retarget(job):
    os.environ['OMP_NUM_THREADS'] = '1'
    import pd_replay as pd
    shape, E, clips = job; dst = OUT / 'references' / shape
    todo = [c for c in clips if not (dst / '00000011' / f'{c}.npz').exists()]
    if todo: pd.retarget_tt(str(S5 / 'a1_reference'), str(dst), only=todo, choices=['00000011'], uniforms=E, torque_limit=33.5)
    return shape, len(todo)


def rollout(ck, E, ref_dir, clips, path, batch=256):
    import jax, jax.numpy as jnp
    from omegaconf import OmegaConf
    from policy_io import load_run
    from pd_replay import build_model
    from track_env import TrackMjxEnv
    cfg, policy = load_run(ck); ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True); ecfg.pop('_target_')
    ecfg.update(reference_dir=str(ref_dir), clips=list(clips), reset_qpos=0., reset_qvel=0., velocity_kick=[0., 0.]); ecfg['obs_noise'] = {k: 0. for k in ecfg['obs_noise']}
    env = TrackMjxEnv(mj_model_fn=functools.partial(build_model, '00000011', uniforms=list(E), torque_limit=33.5), **ecfg)
    e = jnp.asarray(E, jnp.float32); starts = np.asarray(env.clip_start); ends = np.asarray(env.clip_end)
    reset = jax.jit(jax.vmap(env.reset_with_traj))
    floor = int(env.mj_model.geom('floor').id)

    @jax.jit
    def step(s, keys):
        obs = jnp.concatenate((s.obs, jnp.broadcast_to(e, (s.obs.shape[0], 13))), axis=1)
        a = jax.vmap(lambda o, k: policy(o, k)[0])(obs, keys); s = jax.vmap(env.step)(s, a)
        def one(d):
            g, dist = d._impl.contact.geom, d._impl.contact.dist
            return jax.vmap(lambda f: ((((g[:, 0] == f) & (g[:, 1] == floor)) | ((g[:, 1] == f) & (g[:, 0] == floor))) & (dist <= 1e-3)).any())(env._foot_ids)
        return s, jax.vmap(one)(s.data)
    out = {k: [] for k in ['qpos', 'qvel', 'contact', 'target_q', 'valid', 'clip', 'fell']}
    for o in range(0, len(clips), batch):
        idx = np.arange(o, min(o + batch, len(clips))); n = len(idx); idx = np.pad(idx, (0, batch - n), mode='edge')
        T = int((ends[idx] - starts[idx] - 1).max())
        s = reset(jax.random.split(jax.random.PRNGKey(o), batch), jnp.asarray(idx), jnp.asarray(starts[idx]))
        alive = np.ones(batch, bool); fell = np.zeros(batch, bool); length = ends[idx] - starts[idx] - 1
        rec = {k: [] for k in ['qpos', 'qvel', 'contact', 'target_q', 'valid']}; key = jax.random.PRNGKey(1000 + o)
        for t in range(T):
            key, sub = jax.random.split(key); s, touch = step(s, jax.random.split(sub, batch))
            valid = alive & (t < length); f = np.asarray(s.info['done/fellover']) != 0
            q, v = np.asarray(s.data.qpos), np.asarray(s.data.qvel); ok = np.isfinite(q).all(1) & np.isfinite(v).all(1); valid &= ok
            for k, val in [('qpos', q), ('qvel', v), ('contact', np.asarray(touch)), ('target_q', np.asarray(s.data.ctrl)[:, :12]), ('valid', valid)]:
                rec[k].append(np.nan_to_num(val))
            fell |= valid & f; alive = valid & ~f
            if not alive.any(): break
        for k in rec: out[k].append(np.stack(rec[k], 1)[:n])
        out['clip'].append(np.broadcast_to(idx[:n, None], (n, len(rec['valid']))).copy()); out['fell'].append(fell[:n])
    Tm = max(x.shape[1] for x in out['valid'])
    pad = lambda x: np.pad(x, [(0, 0), (0, Tm - x.shape[1])] + [(0, 0)] * (x.ndim - 2))
    data = {k: np.concatenate([pad(x) for x in v]) if k != 'fell' else np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(path, **data, E=np.asarray(E, np.float32), motions=np.array(list(clips)), checkpoint=str(ck))
    return dict(episodes=len(clips), steps=int(data['valid'].sum()), fell=int(data['fell'].sum()))


def main(legs, bodies, n_shapes=20, per_group=100, workers=14):
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.load(open(S7 / 'selected.json')); a = json.load(open(S5 / 'assignment.json'))
    shapes = {name(l, b): (l, b) for l, b in pick_shapes(legs, bodies, n_shapes)}
    json.dump({k: dict(leg=v[0], body=v[1], E=token(*v)) for k, v in shapes.items()}, open(OUT / 'shapes.json', 'w'), indent=1)
    rng = np.random.default_rng(0); plan = {}
    for g in sel:
        allc = sorted({c for v in a['by_group'][g].values() for c in v})
        for s in shapes: plan[(s, g)] = [g] + list(rng.choice(allc, min(per_group, len(allc)), replace=False))
    jobs = [(s, token(*shapes[s]), sorted({c for (ss, g), cl in plan.items() if ss == s for c in cl})[i::3]) for s in shapes for i in range(3)]
    with Pool(workers) as p:
        for s, n in p.imap_unordered(retarget, jobs): log(f's8 retarget {s} +{n}')
    stats = {}
    for (s, g), clips in plan.items():
        f = OUT / s / f'{g}.npz'; f.parent.mkdir(exist_ok=True)
        have = {p.stem for p in (OUT / 'references' / s / '00000011').glob('*.npz')}; clips = [c for c in clips if c in have]
        if not f.exists():
            stats[(s, g)] = rollout(sel[g], token(*shapes[s]), OUT / 'references' / s / '00000011', clips, f); log(f"s8 {s} {g} {stats[(s, g)]}")
        else:
            with np.load(f) as d: stats[(s, g)] = dict(episodes=len(d['valid']), steps=int(d['valid'].sum()), fell=int(d['fell'].sum()))
    make_report(shapes, stats, legs, bodies)
    return sorted(str(p) for p in OUT.glob('G_*/*.npz'))


def make_report(shapes, stats, legs, bodies):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter([l for l in legs for b in bodies], [b for l in legs for b in bodies], s=180, marker='s', c='k', label='학습 형상')
    sv = {s: sum(v['steps'] for (ss, _), v in stats.items() if ss == s) for s in shapes}
    sc = ax.scatter([v[0] for v in shapes.values()], [v[1] for v in shapes.values()], c=[sv[s] for s in shapes], s=160, cmap='viridis', label='수집 형상 (색 = 유효 step)')
    plt.colorbar(sc); ax.set_xlabel('다리 길이 변화 (%)'); ax.set_ylabel('몸통 길이 변화 (%)'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'shapes.png', dpi=105); plt.close(fig)
    groups = sorted({g for _, g in stats})
    rows = [(g.replace('__r00', ''), sum(v['episodes'] for (s, gg), v in stats.items() if gg == g), sum(v['steps'] for (s, gg), v in stats.items() if gg == g),
             f"{sum(v['fell'] for (s, gg), v in stats.items() if gg == g) / max(1, sum(v['episodes'] for (s, gg), v in stats.items() if gg == g)) * 100:.0f} %") for g in groups]
    report(OUT / 'report.html', '8 단계: 학습하지 않은 연속 형상 rollout 수집', [
        ('수집 형상', img(OUT / 'shapes.png')), ('묶음별 수집량', table(['묶음', '에피소드', '유효 step (30 Hz)', '넘어짐 비율'], rows)),
        ('형상별', table(['형상', '치수', '유효 step'], [(s, dims(*shapes[s]), sv[s]) for s in shapes]))],
        [('형상', f'학습 격자 범위 안 farthest-point {len(shapes)} 개 (학습 형상 제외)'), ('정책', '7 단계 선택 체크포인트 (묶음별 공유 정책, 형상 토큰 입력)'),
         ('모션', '묶음 레퍼런스 + 묶음 CALM 모션 표본, 이 형상으로 새로 리타겟'), ('실행', 'MJX, 결정적 정책, 외란·잡음 없음, 넘어지면 그 에피소드 끝'),
         ('저장', 'rollout_data.load 형식 (valid, qpos, qvel, contact, target_q, E, motions, clip)')])
