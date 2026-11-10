"""4 번: 오전 CPU 파일럿 (형상 16 × 모션 32 = 512) 을 MJX 로 다시 실행하고 같은 분석으로 비교."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(ROOT / 'runs/calm_tt_policy_pilot/try1/code'))
import jax, jax.numpy as jnp
import calm_jax, calm_tt_mjx as ctm
import calm_tt_pilot
calm_tt_pilot.MOTIONS = ROOT / 'runs/calm_motion_10,000/dataset/accepted'
calm_tt_pilot.PD_RUN = ROOT / 'runs/local_mujoco_10k_20261109'
import calm_tt_pilot_analyze as an
PILOT = ROOT / 'runs/calm_tt_policy_pilot/try1'
cfg = json.load(open(PILOT / 'config.json'))
(HERE / 'episodes').mkdir(exist_ok=True)
json.dump(cfg, open(HERE / 'config.json', 'w'), indent=1)
params = calm_jax.load_params(); T = 300; SIT = 10
t0 = time.time()
for shape, E in cfg['shapes'].items():
    motions = sorted({int(f.stem.split('_')[1]) for f in (PILOT / 'episodes').glob(f'{shape}_*.npz')})
    m, idx = ctm.make_model(E); roll = ctm.make_rollout(m, idx, params, T)
    q0, cmd, lat, use = [], [], [], []
    for mo in motions:
        ref = dict(np.load(calm_tt_pilot.MOTIONS / f'{mo:05d}.npz'))
        sit = np.zeros(T, bool)
        for k, a, b in zip(ref['motion_index'], ref['segment_start'], ref['segment_end']):
            if int(k) == SIT: sit[a:b] = True
        q0.append(ctm.initial_qpos(m, idx, ref['root'][0], ref['q'][0])); cmd.append(ref['command'][:T]); lat.append(ref['latent'][:T]); use.append(sit)
    out = jax.device_get(roll(jnp.asarray(np.array(q0)), jnp.asarray(np.array(cmd), jnp.float32), jnp.asarray(np.array(lat), jnp.float32), jnp.asarray(np.array(use))))
    for i, mo in enumerate(motions):
        f = np.asarray(out['fell_step'][i]); steps = int(np.argmax(f)) if f.any() else T
        np.savez_compressed(HERE / 'episodes' / f'{shape}_{mo:05d}.npz', qpos=np.asarray(out['qpos'][i, :steps + 1], np.float32),
                            target_q=np.asarray(out['target'][i, :steps], np.float32), E=np.asarray(E, np.float32), motion=mo, shape=shape,
                            fell=bool(f.any()), dt=ctm.CONTROL_DT, steps=steps)
    print(shape, len(motions), 'eps', f'{time.time() - t0:.0f}s', flush=True)
rows = an.calm_rows(HERE)
mjx_sum = an.summarize(rows)
pilot_sum = json.load(open(PILOT / 'analysis.json'))['calm_all']
by_shape = {s: (an.summarize([r for r in rows if r['shape'] == s]), json.load(open(PILOT / 'analysis.json'))['calm_by_shape'][s]) for s in cfg['shapes']}
res = dict(mjx=mjx_sum, cpu_pilot=pilot_sum, by_shape={s: dict(mjx_cos=a['direction_cos_mean'], cpu_cos=b['direction_cos_mean'], mjx_surv=a['survived_10s'], cpu_surv=b['survived_10s']) for s, (a, b) in by_shape.items()},
           seconds=time.time() - t0)
json.dump(res, open(HERE / 'replicate.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('mjx', 'cpu_pilot')}, indent=1))
