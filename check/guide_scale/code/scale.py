"""s9z TT (H 32, 100 Hz) 로 straight 과제 open-loop 계획만 생성: 보상 유도 세기 0·1·10·100 에서 TT 계획의 전진 거리·경로 길이·기울기 크기 (2026-11-18, '왜 과제 유도가 안 되나' 진단).
MuJoCo 실행 없음 (s0 정착만). 후보 8, seed 0. 출력: ../result.json"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault('TT_CONTROL_HZ', '100'); os.environ.setdefault('TT_PHYSICS_HZ', '200')
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / 'code/tt_dynamics')); sys.path.insert(0, str(ROOT / 'chain/runs/s10z_mpc_h32/code'))
import numpy as np, torch
import generate_task_mujoco as gtm
from joint_diffusion_guidance import joint_diffusion_full_reward
from mpc_generate import load
dev = 'cuda'; net, ta, kw = load(ROOT / 'chain/runs/s9z_tt_h32/tt', dev); task = gtm.TASKS['straight']()
s0_fn = lambda eb: torch.as_tensor(np.stack([gtm.settled_start(e) for e in eb.detach().cpu().numpy()]), device=dev, dtype=kw['mu'].dtype)
out = {}
for sc in [0, 1, 10, 100]:
    t0 = time.time(); plan, E, audit = joint_diffusion_full_reward(net, task, n=8, episode_states=gtm.STEPS, guidance_scale=sc, seed=0, s0_fn=s0_fn, **kw)
    p = plan.detach().cpu().numpy(); xy = p[:, :, :2]
    r, _ = task(plan.detach(), torch.ones(p.shape[:2], dtype=torch.bool, device=dev))
    g = {k: v for k, v in audit.items() if isinstance(v, (int, float, str))} if isinstance(audit, dict) else {}
    gl = {k: (float(np.mean(v)) if len(v) else None) for k, v in audit.items() if isinstance(v, list) and v and isinstance(v[0], float)} if isinstance(audit, dict) else {}
    out[sc] = dict(forward_x_m=np.round(xy[:, -1, 0] - xy[:, 0, 0], 2).tolist(), path_m=np.round(np.linalg.norm(np.diff(xy, axis=1), axis=2).sum(1), 2).tolist(),
                   mean_speed=float(np.linalg.norm(p[:, :, 8:10], axis=2).mean()), tt_reward=np.round(r.cpu().numpy(), 2).tolist(), audit_means=gl, sec=round(time.time() - t0))
    out[sc]['pred_reward_first_last'] = [audit['predicted_reward_during_guidance'][0], audit['predicted_reward_during_guidance'][-1]] if audit['predicted_reward_during_guidance'] else None
    print(sc, {k: v for k, v in out[sc].items() if k != 'audit_means'}, flush=True); print('  audit', gl, flush=True)
json.dump(out, open(Path(__file__).resolve().parents[1] / 'result.json', 'w'), indent=1)
