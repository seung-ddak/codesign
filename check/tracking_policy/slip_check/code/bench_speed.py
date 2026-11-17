"""접촉 설정별 MJX 환경 step 속도 (GPU, vmap). E0 slide 정책 (walkForwardSlow) 으로 실제 접촉이 생기게 한다.
설정마다 jit 후 워밍업, 그다음 설정들을 번갈아 R 회 측정 (다른 GPU 작업과 겹친 영향을 고르게). 출력: results/bench_speed.json"""
import functools, json, sys, time
from pathlib import Path
import numpy as np
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관
sys.path.insert(0, str(CODE_DIR)); sys.path.insert(0, str(ROOT / 'chain/code'))
import slip_eval as se
import jax, jax.numpy as jnp
from omegaconf import OmegaConf
from pd_replay import build_model
from policy_io import load_run
from track_env import TrackMjxEnv
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024; STEPS = 100; R = 3
M = 'walkForwardSlow__r00'; ck = se.ckpt('E0', M); cfg, policy = load_run(ck)
fns = {}
for name, c in se.CONFIGS.items():
    if name[0] not in 'ABD': continue
    ecfg = OmegaConf.to_container(cfg.env, resolve=True); ecfg.pop('_target_')
    ecfg.update(reference_dir=str(ROOT / 'check/slide/references/E0/00000011'), clips=[M], impratio=c['impratio'], solver_iterations=c['it'], solver_ls_iterations=c['ls'])
    def model(c=c):
        m = build_model('00000011', torque_limit=33.5, uniforms=se.SHAPES['E0']); m.opt.cone = c['cone']; return m
    env = TrackMjxEnv(mj_model_fn=model, **ecfg)
    reset = jax.jit(jax.vmap(env.reset))

    @jax.jit
    def roll(s, key, env=env):
        def body(carry, k):
            s = carry; a = jax.vmap(lambda o, kk: policy(o, kk)[0])(s.obs, jax.random.split(k, N))
            return jax.vmap(env.step)(s, a), None
        s, _ = jax.lax.scan(body, s, jax.random.split(key, STEPS)); return s
    s = reset(jax.random.split(jax.random.PRNGKey(0), N)); t0 = time.time(); s = roll(s, jax.random.PRNGKey(1)); jax.block_until_ready(s.data.qpos)
    print(name, 'compile+first', round(time.time() - t0, 1), 's', flush=True); fns[name] = (roll, s)
res = {n: [] for n in fns}
for r in range(R):
    for n, (roll, s) in fns.items():
        t0 = time.time(); s2 = roll(s, jax.random.PRNGKey(10 + r)); jax.block_until_ready(s2.data.qpos); dt = time.time() - t0
        res[n].append(N * STEPS / dt); print(n, r, f'{N * STEPS / dt:,.0f} env step/s', flush=True)
base = np.median(res[next(iter(res))])
out = {n: dict(env_steps_per_s=float(np.median(v)), relative=float(np.median(v) / base), runs=v) for n, v in res.items()}
out['_조건'] = dict(num_envs=N, steps=STEPS, repeats=R, gpu_shared_with='short_leg 학습 5 개 (동시 실행)', policy=str(ck))
json.dump(out, open(HERE / 'results/bench_speed.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps({n: (round(v['env_steps_per_s']), round(v['relative'], 2)) for n, v in out.items() if n[0] != '_'}, ensure_ascii=False))
