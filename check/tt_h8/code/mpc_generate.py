"""TT 생성: open-loop (10 초 한 번 생성) 대 MPC (EVERY step 마다 MuJoCo 실제 상태 S 를 S0 로 넣고 남은 구간 다시 생성, 형상 E 고정).
2026-11-17 사용자 결정: 학습 창 8 step + 예측 시점을 더 자주. legacy/runs/calm_dynamics_1000x100/tt_try1/mpc_v1/run_mpc.py 를 지금 생성기 (30 Hz, s0_fn) 에 맞춤.
실행: python mpc_generate.py --ckpt <tt 폴더> --task circle --every 7 --out <새 폴더>
출력: <out>/scale_<s>.npz (E, open_pred, open_actual, mpc_pred, mpc_actual, valid), summary.json"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault('TT_CONTROL_HZ', '30'); os.environ.setdefault('TT_PHYSICS_HZ', '240')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/tt_dynamics'))
import numpy as np, torch
import generate_task_mujoco as gtm
from joint_diffusion_guidance import joint_diffusion_full_reward
from tt_model import make_net, schedule, check_rest_checkpoint


def load(ckdir, dev):
    ck = torch.load(Path(ckdir) / 'checkpoint.pt', map_location=dev, weights_only=False); ta = ck['args']; check_rest_checkpoint(ta)
    net = make_net(ta).to(dev); net.load_state_dict(ck['ema_net'] if ck.get('ema_net') is not None else ck['net']); net.eval().requires_grad_(False)
    mu = torch.as_tensor(ck['mu'], device=dev); sd = torch.as_tensor(ck['sd'], device=dev)
    beta, abar = schedule(ta['diff_steps'], dev); emu, esd = ((.5, .5) if ta['norm'] == 'minmax' else (.5, .25))
    return net, ta, dict(mu=mu, sd=sd, beta=beta, abar=abar, emu=emu, esd=esd, clip=ta['clip'], horizon=ta['horizon'], state_dim=42, action_dim=12, e_dim=13)


def evaluate(task, E, tok):
    rows, act, val = [], [], []
    for i, (e, t) in enumerate(zip(E, tok)):
        a, v, meta = gtm.replay(e, t, task=task); r = gtm.score_replay(task, a, v, meta)
        rows.append(dict(candidate=i, morphology=e.tolist(), **r, **meta)); act.append(a); val.append(v)
    return rows, np.stack(act), np.stack(val)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--task', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--every', type=int, default=7); p.add_argument('--candidates', type=int, default=8); p.add_argument('--scales', nargs='+', type=float, default=[0, 1])
    p.add_argument('--seed', type=int, default=0); a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; net, ta, kw = load(a.ckpt, dev); task = gtm.TASKS[a.task](); a.out.mkdir(parents=True, exist_ok=True)
    S, N = gtm.STEPS, a.candidates

    def s0_fn(eb):
        return torch.as_tensor(np.stack([gtm.settled_start(e) for e in eb.detach().cpu().numpy()]), device=dev, dtype=kw['mu'].dtype)
    summary = {}
    for scale in a.scales:
        t0 = time.time()
        plan, E, _ = joint_diffusion_full_reward(net, task, n=N, episode_states=S, guidance_scale=scale, seed=a.seed, s0_fn=s0_fn, **kw)
        plan = plan.detach().cpu().numpy()[:, :S]; E = E.detach().cpu().numpy()
        open_rows, open_act, open_val = evaluate(task, E, plan)
        tok = plan.copy(); replans = 0
        for t in range(a.every, S, a.every):
            rr = [gtm.replay(e, k, task=task) for e, k in zip(E, tok)]; alive = np.array([v[t] for _, v, _ in rr])
            if not alive.any(): break
            cur = np.stack([x[t, :42] for x, _, _ in rr])  # MuJoCo 실제 상태 (특징)
            new, _, _ = joint_diffusion_full_reward(net, task, n=N, episode_states=S - t + 1, guidance_scale=scale, seed=a.seed + t,
                                                    fixed_e=E, init_state=torch.as_tensor(cur, device=dev), **kw)
            new = new.detach().cpu().numpy()
            for i in np.flatnonzero(alive): tok[i, t:] = new[i, :S - t]
            replans += 1
            if replans % 10 == 0: print(f'{a.task} scale {scale:g} t={t * gtm.DT:.1f}s replans {replans} alive {int(alive.sum())}/{N} {time.time() - t0:.0f}s', flush=True)
        mpc_rows, mpc_act, mpc_val = evaluate(task, E, tok)
        np.savez_compressed(a.out / f'scale_{scale:g}.npz', E=E, open_pred=plan, open_actual=open_act, open_valid=open_val, mpc_pred=tok, mpc_actual=mpc_act, mpc_valid=mpc_val)
        summary[f'scale_{scale:g}'] = dict(open_loop=open_rows, mpc=mpc_rows, replans=replans, seconds=time.time() - t0)
        f = lambda rs: [(r['candidate'], None if r['reward'] is None else round(r['reward'], 2), r['valid_seconds']) for r in rs]
        print(f'{a.task} scale {scale:g} open {f(open_rows)}\n  mpc {f(mpc_rows)}  ({time.time() - t0:.0f}s)', flush=True)
    (a.out / 'summary.json').write_text(json.dumps(dict(task=a.task, ckpt=a.ckpt, horizon=ta['horizon'], every=a.every, hz=gtm.CONTROL_HZ, results=summary), indent=1, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
