"""8z 단계 (2026-11-17 사용자 결정): TT 학습용 rollout. 학습하지 않은 형상 500 개 + 학습 형상 9 개, 형상당 모션 60 개.
형상당 모션 = 모션 세트 6 개 × (레퍼런스 1 + 그 세트 CALM 9). CALM 은 세트마다 섞은 뒤 형상들에 순환 배정 (세트당 1,000 개가 약 4.6 형상씩 고르게)
→ 같은 모션 × 여러 형상, 같은 형상 × 여러 모션, 둘 다 다른 조합이 모두 들어간다.
형상 = 학습 격자 범위 (다리·몸통 -20~+20 %) 안 정수 % farthest-point 500 개 (학습 형상 제외, s8_collect.pick_shapes, seed 0).
정책 = s7h 선택 100 Hz 체크포인트 (세트별). 여러 형상을 한 GPU 묶음에 섞는다 (multi_design.make_bank, 25 형상 × 10 모션 = 250 env).
결정적 정책, 외란·잡음 없음, 넘어지면 그 에피소드 끝 (fell 표시).
단계: plan → retarget (s5h 종료 뒤, CPU) → rollout (s7h 종료 뒤, GPU). 이어하기 = 있는 파일은 건너뜀.
출력: runs/s8z_collect/ (plan.json, references/<형상>/00000011/, rollouts/<세트>/chunk_XXX.npz, report.html)
chunk npz: qpos, qvel, contact, target_q, valid [N,T…], fell [N], E [N,13], shape [N], motion [N], checkpoint."""
import argparse, functools, json, os, time
from multiprocessing import Pool
import numpy as np
from common import RUNS, CHAIN, log, report, table
from shapes_grid import token, name
from s8_collect import pick_shapes
OUT = RUNS / 's8z_collect'; S5 = RUNS / 's5_retarget'; S5H = RUNS / 's5h_retarget100'; S7H = RUNS / 's7h_eval100'
LEGS = BODIES = [-20, 0, 20]; N_SHAPES = 500; PER_SET = 9; CHUNK = 25
os.environ['TRACK_SG_SEC'] = '0.07'  # 100 Hz 정책 학습 값


def make_plan():
    if (OUT / 'plan.json').exists(): return json.load(open(OUT / 'plan.json'))
    a = json.load(open(S5 / 'assignment.json'))
    sets = {g: sorted({c for v in per.values() for c in v}) for g, per in a['by_group'].items()}
    seen = {name(l, b): dict(E=token(l, b), seen=True, ref_dir=str(S5H / 'references' / name(l, b) / '00000011')) for l in LEGS for b in BODIES}
    unseen = {name(l, b): dict(E=token(l, b), seen=False, ref_dir=str(OUT / 'references' / name(l, b) / '00000011')) for l, b in pick_shapes(LEGS, BODIES, N_SHAPES)}
    shapes = {**seen, **unseen}; order = list(shapes); rng = np.random.default_rng(0); motions = {s: {} for s in order}
    for g, clips in sets.items():
        pool = list(rng.permutation(clips)); k = 0
        for s in order:
            motions[s][g] = [g] + [str(pool[(k + i) % len(pool)]) for i in range(PER_SET)]; k += PER_SET
    p = dict(shapes=shapes, motions=motions, per_set=PER_SET)
    OUT.mkdir(parents=True, exist_ok=True); json.dump(p, open(OUT / 'plan.json', 'w'), indent=1); return p


def retarget(job):
    os.environ['OMP_NUM_THREADS'] = '1'
    import pd_replay as pd
    shape, E, clips = job; dst = OUT / 'references' / shape
    todo = [c for c in clips if not (dst / '00000011' / f'{c}.npz').exists()]
    if todo: pd.retarget_tt(str(S5H / 'a1_100'), str(dst), only=todo, choices=['00000011'], uniforms=E, torque_limit=33.5)
    return shape, len(todo)


def retarget_all(p, workers=16):
    jobs = [(s, v['E'], sorted({c for cl in p['motions'][s].values() for c in cl})[i::2]) for s, v in p['shapes'].items() if not v['seen'] for i in range(2)]
    done = 0
    with Pool(workers) as pool:
        for s, n in pool.imap_unordered(retarget, jobs):
            done += 1
            if done % 50 == 0: log(f's8z retarget {done}/{len(jobs)}')
    log('s8z retarget 완료')


def rollout_set(g, ck, p):
    import jax, jax.numpy as jnp
    from omegaconf import OmegaConf
    from policy_io import load_run
    from pd_replay import build_model
    from track_env import TrackMjxEnv
    from multi_design import make_bank, MultiDesignVmapWrapper
    cfg, policy = load_run(ck); ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True); ecfg.pop('_target_')
    ecfg.update(reset_qpos=0., reset_qvel=0., velocity_kick=[0., 0.]); ecfg['obs_noise'] = {k: 0. for k in ecfg['obs_noise']}
    mk = lambda s: TrackMjxEnv(mj_model_fn=functools.partial(build_model, '00000011', kp=cfg.get('kp'), tau_scale=cfg.get('tau_scale'),
                                                           torque_limit=cfg.get('torque_limit') or 33.5, uniforms=list(p['shapes'][s]['E'])),
                               **dict(ecfg, reference_dir=p['shapes'][s]['ref_dir'], clips=p['motions'][s][g]))
    order = list(p['shapes']); d = OUT / 'rollouts' / g; d.mkdir(parents=True, exist_ok=True); fns = {}
    for ci, o in enumerate(range(0, len(order), CHUNK)):
        f = d / f'chunk_{ci:03d}.npz'
        if f.exists(): continue
        names = order[o:o + CHUNK]; envs = [mk(s) for s in names]
        bank = make_bank(envs, [p['shapes'][s]['E'] for s in names], distinct_clips=True)
        if 'base' not in fns: fns['base'], fns['fields'] = bank['base'], sorted(bank['model_fields'])  # 묶음마다 같은 필드 목록 → jit 재사용
        assert set(bank['model_fields']) <= set(fns['fields']), set(bank['model_fields']) - set(fns['fields'])
        mf = {k: bank['model_fields'][k] if k in bank['model_fields'] else jnp.stack([getattr(bank['base'], k)] * len(envs)) for k in fns['fields']}
        dyn = dict(model_fields=mf, refs=bank['refs'], uniforms=bank['uniforms']); W = MultiDesignVmapWrapper(envs[0], bank)
        D, M = len(names), len(p['motions'][names[0]][g]); B = D * M
        did = np.repeat(np.arange(D), M); cid = np.tile(np.arange(M), D)
        starts = np.asarray(bank['refs']['clip_start'])[did, cid]; ends = np.asarray(bank['refs']['clip_end'])[did, cid]
        floor = int(envs[0].mj_model.geom('floor').id); feet = envs[0]._foot_ids
        if D not in fns:  # 형상 수가 같은 묶음은 같은 jit 재사용 (bank 는 인자로)
            base = fns['base']
            def r1(bk, key, dd, cc, st):
                W.bank = dict(bk, base=base)
                with W.design_env(W.model_for(dd), dd) as env: return env.reset_with_traj(key, cc, st)
            def s1(bk, s, a, dd):
                W.bank = dict(bk, base=base)
                with W.design_env(W.model_for(dd), dd) as env:
                    s = env.step(s, a); g_, dist = s.data._impl.contact.geom, s.data._impl.contact.dist
                    touch = jax.vmap(lambda ft: ((((g_[:, 0] == ft) & (g_[:, 1] == floor)) | ((g_[:, 1] == ft) & (g_[:, 0] == floor))) & (dist <= 1e-3)).any())(feet)
                    return s, touch
            def step(bk, s, keys, dd):
                obs = jnp.concatenate((s.obs, bk['uniforms'][dd]), axis=1)
                a = jax.vmap(lambda o, k: policy(o, k)[0])(obs, keys)
                return jax.vmap(lambda ss, aa, d_: s1(bk, ss, aa, d_))(s, a, dd)
            fns[D] = (jax.jit(jax.vmap(r1, in_axes=(None, 0, 0, 0, 0))), jax.jit(step))
        reset, step = fns[D]; dd = jnp.asarray(did)
        s = reset(dyn, jax.random.split(jax.random.PRNGKey(ci), B), dd, jnp.asarray(cid), jnp.asarray(starts))
        alive = np.ones(B, bool); fell = np.zeros(B, bool); length = ends - starts - 1; T = int(length.max())
        rec = {k: [] for k in ['qpos', 'qvel', 'contact', 'target_q', 'valid']}; key = jax.random.PRNGKey(1000 + ci)
        for t in range(T):
            key, sub = jax.random.split(key); s, touch = step(dyn, s, jax.random.split(sub, B), dd)
            valid = alive & (t < length); fo = np.asarray(s.info['done/fellover']) != 0
            q, v = np.asarray(s.data.qpos), np.asarray(s.data.qvel); valid &= np.isfinite(q).all(1) & np.isfinite(v).all(1)
            for k, val in [('qpos', q), ('qvel', v), ('contact', np.asarray(touch)), ('target_q', np.asarray(s.data.ctrl)[:, :12]), ('valid', valid)]:
                rec[k].append(np.nan_to_num(val).astype(np.float32) if k != 'valid' and k != 'contact' else val)
            fell |= valid & fo; alive = valid & ~fo
            if not alive.any(): break
        data = {k: np.stack(v, 1) for k, v in rec.items()}
        np.savez_compressed(f, **data, fell=fell, E=np.asarray([p['shapes'][names[i]]['E'] for i in did], np.float32),
                            shape=np.array([names[i] for i in did]), motion=np.array([p['motions'][names[i]][g][j] for i, j in zip(did, cid)]), checkpoint=str(ck))
        log(f"s8z {g} chunk {ci} episodes {B} steps {int(data['valid'].sum())} fell {int(fell.sum())}")
        del envs, bank, W


def wait(pattern, logfile):
    while not (logfile.exists() and pattern in logfile.read_text()): time.sleep(300)


def make_rep(p):
    rows = []
    for g in sorted({g for m in p['motions'].values() for g in m}):
        ep = st = fe = 0
        for f in sorted((OUT / 'rollouts' / g).glob('chunk_*.npz')):
            with np.load(f) as z: ep += len(z['fell']); st += int(z['valid'].sum()); fe += int(z['fell'].sum())
        rows.append((g.replace('__r00', ''), ep, st, f'{fe / max(ep, 1) * 100:.1f} %'))
    report(OUT / 'report.html', '8z 단계: 형상 509 개 rollout 수집 (100 Hz)', [('세트별 수집량', table(['세트', '에피소드', '유효 step (100 Hz)', '넘어짐'], rows))],
           [('형상', f'학습 9 + 학습 범위 안 새 형상 {N_SHAPES} (farthest-point, 정수 %)'), ('모션', f'형상당 60 = 세트 6 × (레퍼런스 + CALM {PER_SET}), CALM 순환 배정'),
            ('정책', 's7h 선택 100 Hz 체크포인트'), ('실행', 'MJX, 결정적, 외란·잡음 없음, 25 형상 × 10 모션 묶음')])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--wait', action='store_true'); a = ap.parse_args()
    p = make_plan(); log(f"s8z plan 형상 {len(p['shapes'])}, 형상당 모션 {sum(len(v) for v in next(iter(p['motions'].values())).values())}")
    if a.wait: wait('EXIT', CHAIN / 'logs/s5h.log')
    retarget_all(p)
    if a.wait: wait('s7h END', CHAIN / 'logs/chain.log')
    sel = json.load(open(S7H / 'selected.json'))
    for g in sel: rollout_set(g, sel[g], p)
    make_rep(p); log('s8z END')
