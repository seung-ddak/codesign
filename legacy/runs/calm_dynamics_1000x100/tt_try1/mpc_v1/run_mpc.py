"""MPC 시험: e0000 TT, 원형 트랙, reward gradient (강도 3). open-loop (10 초 한 번 생성) vs 1 초마다 실제 상태에서 다시 생성 (형상 고정).
실제로 실행한 행동 전체를 기존 채점 (generate_task_mujoco.replay + score_replay) 으로 평가한다."""
import json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'code/tt_dynamics'))
os.environ.setdefault('TT_REWARD_PROFILE', 'v2')
import torch
torch.set_num_threads(12)
import generate_task_mujoco as gtm
from benchmark_task_inference import load_model, generate
from joint_diffusion_guidance import joint_diffusion_full_reward
TASK, SCALE, N, SEED, EVERY = os.environ.get('MPC_TASK', 'circle'), 3.0, 4, 0, int(os.environ.get('MPC_EVERY', 50))
model, setup = load_model(HERE.parent / 'tt/checkpoint.pt', 'cpu'); cfg = setup['config']
task = gtm.TASKS[TASK]()

def delta_of(prev, cur):
    yaw = np.arctan2(prev[4], prev[3]); c, s = np.cos(yaw), np.sin(yaw); d = cur[:2] - prev[:2]
    dy = np.arctan2(cur[4], cur[3]) - yaw; dy = (dy + np.pi) % (2 * np.pi) - np.pi
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], dy], np.float32)

def score(E, tok):
    out = []
    for e, t in zip(E, tok):
        actual, valid, meta = gtm.replay(e, t, task=task)
        r = gtm.score_replay(task, actual, valid, meta)
        out.append(dict(reward=r['reward'], progress_laps=r['progress'], forward_m=meta['forward_m'], speed=meta['mean_forward_speed_m_s'], complete_10s=meta['complete_10s'], seconds=meta['valid_seconds'],
                        lane=r['lane_compliance'], success=r['task_success'], actual=actual, valid=valid))
    return out

t0 = time.time()
traj, E, _ = generate(model, setup, task, count=N, seed=SEED, scale=SCALE)
plan0 = traj.detach().cpu().numpy(); E = E.detach().cpu().numpy()
open_loop = score(E, plan0[:, :gtm.STEPS])
# MPC
exec_tok = plan0[:, :gtm.STEPS].copy(); pred_err = []; replans = 0
for t in range(EVERY, gtm.STEPS, EVERY):
    rows = [gtm.replay(e, tk, task=task) for e, tk in zip(E, exec_tok)]
    alive = np.array([v[t] for _, v, _ in rows])
    if not alive.any(): break
    cur = np.stack([a[t, :42] for a, _, _ in rows]); prev = np.stack([a[t - 1, :42] for a, _, _ in rows])
    d0 = np.stack([delta_of(p, c) for p, c in zip(prev, cur)])
    L = gtm.STEPS - t + 1
    new, _, _ = joint_diffusion_full_reward(model, task, n=N, episode_states=L, horizon=cfg['horizon'], state_dim=42, action_dim=12, e_dim=13,
        mu=setup['mu'], sd=setup['sd'], beta=setup['beta'], abar=setup['abar'], emu=setup['emu'], esd=setup['esd'], clip=cfg['clip'],
        guidance_scale=SCALE, seed=SEED + t, pose=cfg.get('pose', 'abs'), fixed_e=E, init_state=torch.as_tensor(cur), init_delta=torch.as_tensor(d0))
    new = new.detach().cpu().numpy()
    for i in range(N):
        if alive[i]:
            exec_tok[i, t:, 42:] = new[i, :gtm.STEPS - t, 42:]
            exec_tok[i, t:, :42] = new[i, :gtm.STEPS - t, :42]   # 기록용 (재생에는 행동만 쓰인다)
    replans += 1
    print(f't={t * gtm.DT:.1f}s replanned, alive {int(alive.sum())}/{N}, {time.time() - t0:.0f}s', flush=True)
mpc = score(E, exec_tok)
def summ(rs): return dict(survived_10s=sum(r['complete_10s'] for r in rs), success=sum(r['success'] for r in rs),
                          progress_laps=[round(float(r['progress_laps']), 3) for r in rs], reward=[None if r['reward'] is None else round(r['reward'], 2) for r in rs],
                          lane=[round(float(r['lane']), 2) for r in rs], forward_m=[round(float(r['forward_m']), 2) for r in rs])
res = dict(task=TASK, scale=SCALE, candidates=N, replan_every_s=EVERY * gtm.DT, replans=replans, E_fixed=E.round(3).tolist(),
           open_loop=summ(open_loop), mpc=summ(mpc), seconds=time.time() - t0)
print(json.dumps({k: res[k] for k in ('open_loop', 'mpc', 'seconds')}, indent=1))
json.dump(res, open(HERE / f'result_{TASK}_every{EVERY}.json', 'w'), indent=1)
np.savez(HERE / f'tokens_{TASK}_every{EVERY}.npz', E=E, open_loop=plan0[:, :gtm.STEPS], mpc=exec_tok,
         open_actual=np.stack([r['actual'] for r in open_loop]), mpc_actual=np.stack([r['actual'] for r in mpc]))
