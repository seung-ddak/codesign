"""5 번 준비 검증: 학습 전 CalmTrackEnv + CALM 신경망이 원본 CALM (저장된 z 경로) 과 같은 관절 목표를 내는지, 그리고 10 초 생존."""
import json, sys, functools
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(ROOT.parent / 'transformer-transformer'))
import jax, jax.numpy as jnp
from omegaconf import OmegaConf
import calm_jax, calm_finetune as cf
from pd_replay import build_model
REF = ROOT / 'runs/ppo_tracking/try2/references/e0000/00000011'
plan = json.load(open(ROOT / 'runs/ppo_tracking/try2/plan.json'))
clips = ['calm_%05d' % i for i in plan['assignments']['e0000'][:32]]
cfg = OmegaConf.load(ROOT / 'code/motion_tracking/config/env/a1_track.yaml')
cfg.update(dict(reference_dir=str(REF), clips=clips, ctrl_dt=1 / 30, sim_dt=1 / 600, max_episode_length=300, w_vel=1., termination_pose_consecutive_steps=600,
                termination_pos_err_threshold=100., termination_orn_err_threshold=100., reset_qpos=0., reset_qvel=0., velocity_kick=[0., 0.],
                obs_noise={k: 0. for k in cfg.obs_noise}, penalty_foot_slip=-2.5))
c = {k: v for k, v in OmegaConf.to_container(cfg).items() if not k.startswith('_')}
env = cf.CalmTrackEnv(mj_model_fn=functools.partial(build_model, choice='00000011', uniforms=[.5] * 13, torque_limit=33.5), **c)
nets = cf.make_calm_ppo_networks(env.observation_size, env.action_size)
pp = nets.policy_network.init(jax.random.PRNGKey(0)); params = calm_jax.load_params()
reset = jax.jit(jax.vmap(env.reset_with_traj)); step = jax.jit(jax.vmap(env.step))
mean_fn = jax.jit(lambda o: nets.policy_network.apply(None, pp, o)[0])
n = len(clips)
st = reset(jax.random.split(jax.random.PRNGKey(0), n), jnp.arange(n), env.clip_start[:n])
# (1) 같은 상태에서 원본 calm_jax 와 비교
mu = mean_fn(st.obs)
tgt_env = jax.vmap(lambda a: env.action_target(a, None))(mu)
states = jax.vmap(env.body_states)(st.data)
tgt_ref = jax.vmap(lambda s, z: calm_jax.llc_target(params, calm_jax.calm_obs(s), z))(states, env.ref_latent[env.clip_start[:n]])
diff = float(jnp.abs(tgt_env - tgt_ref).max()); print('target max diff vs calm_jax:', diff)
# (2) 10 초 결정적 실행
fell = np.zeros(n, bool); steps = np.full(n, 300)
for t in range(300):
    st = step(st, mean_fn(st.obs))
    q = np.asarray(st.data.qpos); up = 1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)
    new = (~fell) & ((up < .5) | (q[:, 2] < .07)); steps[new] = t + 1; fell |= new
res = dict(target_max_diff=diff, episodes=n, survived_10s=int((~fell).sum()), mean_seconds=float(steps.mean() / 30))
print(res); (Path(__file__).with_name('check_finetune_env.json')).write_text(json.dumps(res, indent=1))
