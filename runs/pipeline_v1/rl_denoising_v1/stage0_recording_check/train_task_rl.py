"""CPU PPO replacement for task actions on frozen, previously generated E.

Uses the repository's rsl_rl PPO and the original task reward for evaluation.
Run from codesign: ../transformer-transformer/.venv/bin/python code/tt_dynamics/train_task_rl.py --out runs/pipeline_v1/task_rl_v1
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# CPU-only run: the installed Triton native extension crashes during Adam import.
sys.modules['triton'] = None
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT.parent / 'quad/rsl_rl'), str(ROOT / 'code/motion_tracking')]
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from pd_replay import build_model
from rollout_data import features
from s0_geometry import s0_to_q
from task_reward import TrackReward, ZigzagReward


DENSE_REWARD_KEYS = ("speed", "progress", "lane", "tilt", "heading", "low",
                     "stall", "action_rate", "torque", "fall")

def dense_task_reward(previous, current, action_delta, torque, task, step, initial_height):
    """Physical per-step task reward; rates are integrated at the 50 Hz control dt."""
    dt, speed = .02, .5
    x, y = current[:2]
    if task == 'zigzag':
        phase = 2 * np.pi * x / 2.
        center_y = .3 * 2 / np.pi * np.arcsin(np.sin(phase))
        tangent = np.array([1., .6 * np.sign(np.cos(phase))])
        lane_error = abs(y - center_y)
        progress = current[0] - previous[0]
    else:
        radial = np.array([x, y - 1.])
        tangent = np.array([-radial[1], radial[0]])
        lane_error = abs(np.linalg.norm(radial) - 1.)
        old = previous[:2] - np.array([0., 1.])
        progress = np.arctan2(old[0]*radial[1]-old[1]*radial[0], old @ radial)
    tangent /= max(np.linalg.norm(tangent), 1e-8)
    velocity = (current[:2] - previous[:2]) / dt
    along = float(velocity @ tangent)
    terms = dict(speed=2.*(np.exp(-((along-speed)/.25)**2)-np.exp(-(speed/.25)**2)),
                 progress=2.*np.clip(progress/dt, -1., 1.),
                 lane=-2.*min((lane_error/.15)**2, 4.),
                 tilt=-.5*max(0., 1.+current[7]),
                 heading=-.2*(1.-np.clip(current[3:5] @ tangent, -1., 1.)),
                 low=-2.*(max(0., .7*initial_height-current[2])/initial_height)**2,
                 stall=-.5*float(step > 25 and along < .05),
                 action_rate=-.01*float(np.mean((action_delta/.25)**2)),
                 torque=-.005*float(np.mean((torque/33.5)**2)))
    if not np.isfinite(list(terms.values())).all():
        raise FloatingPointError('Nonfinite dense reward')
    return dt*sum(terms.values()), terms


def build_task_model(E):
    m = build_model('00000011', kp=30., torque_limit=33.5, uniforms=E)
    # Match TrackMjxEnv.preprocess_mj_model without loading tracking references/MJX.
    m.opt.timestep, m.opt.impratio = .005, 1
    m.opt.iterations, m.opt.ls_iterations = 5, 7
    floor = m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    stairs = np.array([m.geom(i).name.startswith('stair_') for i in range(m.ngeom)])
    feet = np.array([m.geom(i).name.endswith('_foot') for i in range(m.ngeom)])
    m.geom_contype[:] = np.select((floor, stairs, feet), (1, 8, 4), default=2)
    m.geom_conaffinity[:] = np.select((floor, stairs, feet), (6, 4, 9), default=1)
    return m


class TaskEnv:
    def __init__(self, source, index, task, reward_version='prefix'):
        self.task, self.reward_version = task, reward_version
        self.E = source['E'][index].copy()
        self.actions = source['pred'][index, :, 42:].copy()
        self.start = source['pred'][index, 0, :42].copy()
        self.horizon = len(self.actions)
        self.reward = ZigzagReward() if task == 'zigzag' else TrackReward()
        self.model = m = build_task_model(self.E)
        self.feet = [m.geom(f'{leg}_foot').id for leg in ('FR', 'FL', 'RR', 'RL')]
        self.floor = m.geom('floor').id
        self.home = m.keyframe('home').ctrl.copy()
        self.data = mujoco.MjData(m)
        self.reset()

    def state(self):
        d = self.data
        contacts = [any(self.floor in (c.geom1, c.geom2) and f in (c.geom1, c.geom2) and c.dist <= .001
                        for c in d.contact[:d.ncon]) for f in self.feet]
        return features(np.stack([self.q0, d.qpos.copy()]),
                        np.stack([self.v0, d.qvel.copy()]), np.array([contacts, contacts]))[-1]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.q0, self.v0 = s0_to_q(self.start)
        self.data.qpos[:], self.data.qvel[:] = self.q0, self.v0
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or self.start[7] > -.5:
            raise ValueError('Invalid initial state')
        self.t, self.prev_score = 0, 0.
        self.last = self.home.copy()
        self.tokens = [np.r_[self.state(), self.last]]
        self.qraw = [self.data.qpos.copy()]
        self.training_return = 0.
        self.vraw = [self.data.qvel.copy()]
        self.applied_controls, self.torque_history = [], []
        self.reward_history, self.reward_component_history = [], []
        return self.obs()

    def obs(self):
        s = self.tokens[-1][:42].copy()
        s[8:14] *= .1
        s[26:38] *= .05
        return np.r_[s, self.E, self.last - self.home, self.t / self.horizon].astype(np.float32)

    def score(self):
        x = torch.tensor(np.asarray(self.tokens), dtype=torch.float32)[None]
        m = torch.ones(x.shape[:2], dtype=torch.bool)
        total, terms = self.reward(x, m)
        return float(total[0]), {k: float(v[0]) for k, v in terms.items()}

    def step(self, action, target=False):
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (12,) or not np.isfinite(a).all():
            raise ValueError('Action must be 12 finite values')
        # PPO outputs home-relative targets in radians; guidance uses absolute targets.
        ctrl = a if target else self.home + a
        ctrl = self.home + np.clip(ctrl - self.home, -2.5, 2.5)
        previous, action_delta = self.tokens[-1][:42].copy(), ctrl - self.last
        self.tokens[-1][42:] = ctrl
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=4)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Nonfinite physics state')
        self.t += 1
        self.last = ctrl.copy()
        self.tokens.append(np.r_[self.state(), ctrl])
        self.qraw.append(self.data.qpos.copy())
        fell = self.tokens[-1][7] > -.5
        done = fell or self.t >= self.horizon - 1
        if self.reward_version == 'dense':
            reward, rates = dense_task_reward(previous, self.tokens[-1][:42], action_delta,
                                          self.data.actuator_force, self.task, self.t, self.start[2])
        else:
            score, _ = self.score()
            reward = score - self.prev_score
            self.prev_score = score
        # Prevent a policy from avoiding future penalties by deliberately ending early.
        if fell:
            reward -= 20.
        self.training_return += reward
        self.vraw.append(self.data.qvel.copy())
        self.applied_controls.append(ctrl.copy())
        self.torque_history.append(self.data.actuator_force.copy())
        self.reward_history.append(float(reward))
        if self.reward_version == 'dense':
            parts = {name: value*.02 for name,value in rates.items()}
            parts['fall'] = -20. if fell else 0.
            self.reward_component_history.append([parts[name] for name in DENSE_REWARD_KEYS])
        return self.obs(), reward, done, bool(fell)


def evaluate(envs, policy=None):
    rows, trajectories = [], {}
    for i, env in enumerate(envs):
        obs = env.reset()
        for t in range(env.horizon - 1):
            if policy is None:
                action = env.actions[t]
            else:
                with torch.no_grad():
                    action = policy.act_inference(torch.tensor(obs)[None])[0].numpy()
            obs, _, done, fell = env.step(action, target=policy is None)
            if done:
                break
        score, terms = env.score()
        rows.append(dict(candidate=i, reward=score, survived=not fell, steps=env.t,
                         training_return=env.training_return,
                         progress=terms['r_progress'] * env.reward.unit_scale, **terms))
        trajectories[f'tokens_{i}'] = np.asarray(env.tokens)
        trajectories[f'qpos_{i}'] = np.asarray(env.qraw)
    return rows, trajectories


def aggregate(rows):
    return dict(reward_median=float(np.median([r['reward'] for r in rows])),
                progress_median=float(np.median([r['progress'] for r in rows])),
                survived=sum(r['survived'] for r in rows), candidates=len(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--task', choices=['zigzag', 'circle'], default='zigzag')
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--self-check', action='store_true')
    p.add_argument('--reward-version', choices=['prefix', 'dense'], default='prefix')
    args = p.parse_args()
    if args.updates <= 0:
        p.error('--updates must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source_path = ROOT / f'runs/pipeline_v1/fast_mode24_v16_twin/evaluation/check_w1/joint_{args.task}_4s/scale_10.npz'
    source = dict(np.load(source_path))
    envs = [TaskEnv(source, i, args.task, args.reward_version) for i in range(len(source['E']))]
    if args.self_check:
        env = envs[0]
        env.reset()
        total = 0.
        for _ in range(8):
            _, r, done, fell = env.step(np.zeros(12))
            total += r
            if done:
                break
        assert np.isfinite(total)
        if args.reward_version == 'prefix':
            assert abs(total - (env.score()[0] - (20. if fell else 0.))) < 1e-5
        assert env.obs().shape == (68,) and np.isfinite(env.obs()).all()
        try:
            env.step(np.full(12, np.nan))
            raise AssertionError('NaN accepted')
        except ValueError:
            pass
        print('self-check passed: reward telescoping, observation, action validation')
        return
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(task=args.task, updates=args.updates, seed=args.seed, source=str(source_path),
                    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    train_candidates=list(range(12)), heldout_candidates=list(range(12, 16)),
                    backend='MuJoCo CPU; both guidance and PPO replayed here',
                    horizon_frames=200, ctrl_dt=.02, sim_dt=.005,
                    ppo=dict(rollout_steps=64, epochs=4, minibatches=4, gamma=.99,
                             learning_rate=3e-4, entropy_coef=.001, hidden=[64, 64]),
                    reward_version=args.reward_version,
                    train_reward=('dense speed/progress/lane/stall/stability/action/torque; -20 fall'
                                  if args.reward_version == 'dense' else 'original task prefix-score increments; -20 terminal fall penalty'),
                    eval_reward='unmodified task_reward; no extra terminal penalty',
                    observation='S42 (scaled velocities), E13, previous home-relative target12, time1',
                    limitation='single seed pilot; E frozen; S produced by physics; no joint diffusion integration')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (args.out / 'train_task_rl_snapshot.py').write_bytes(Path(__file__).read_bytes())
    baseline, traj = evaluate(envs)
    np.savez_compressed(args.out / 'guidance_cpu.npz', **traj)
    (args.out / 'baseline.json').write_text(json.dumps(baseline, indent=2))
    print('baseline', aggregate(baseline), flush=True)
    policy = ActorCritic(68, 68, 12, actor_hidden_dims=[64, 64], critic_hidden_dims=[64, 64],
                         init_noise_std=.3, max_std=.6)
    # Start at home stance, without cloning the failing guided action sequence.
    torch.nn.init.zeros_(policy.actor[-1].weight)
    torch.nn.init.zeros_(policy.actor[-1].bias)
    untrained, _ = evaluate(envs, policy)
    (args.out / 'untrained.json').write_text(json.dumps(untrained, indent=2))
    algo = PPO(policy, num_learning_epochs=4, num_mini_batches=4, gamma=.99,
               learning_rate=3e-4, entropy_coef=.001, device='cpu')
    train = envs[:12]
    algo.init_storage(len(train), 64, [68], [68], [12])
    obs = torch.tensor(np.stack([e.reset() for e in train]))
    start = time.monotonic()
    for update in range(args.updates):
        with torch.no_grad():
            for _ in range(64):
                actions = algo.act(obs, obs).numpy()
                nxt, rew, done = [], [], []
                for e, a in zip(train, actions):
                    o, r, d, _ = e.step(a)
                    nxt.append(e.reset() if d else o)
                    rew.append(r)
                    done.append(d)
                obs = torch.tensor(np.stack(nxt))
                algo.process_env_step(torch.tensor(rew), torch.tensor(done), {})
            algo.compute_returns(obs)
        metrics = algo.update()
        if not np.isfinite(metrics).all() or not all(torch.isfinite(x).all() for x in policy.parameters()):
            raise FloatingPointError('Nonfinite PPO update')
        record = dict(update=update+1, transitions=(update+1)*64*len(train),
                      seconds=time.monotonic()-start, metrics=list(map(float, metrics)))
        with (args.out / 'training.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if (update+1) % 10 == 0:
            torch.save(policy.state_dict(), args.out / 'policy.pt')
            print(record, flush=True)
        if update+1 == 100:
            torch.save(policy.state_dict(), args.out / 'policy_update100.pt')
    torch.save(policy.state_dict(), args.out / 'policy.pt')
    final, traj = evaluate(envs, policy)
    np.savez_compressed(args.out / 'rl_cpu.npz', **traj)
    result = dict(protocol=protocol, guidance=baseline, untrained=untrained, rl=final, elapsed_seconds=time.monotonic()-start)
    for name, sl in [('train', slice(0, 12)), ('heldout', slice(12, 16)), ('all', slice(None))]:
        result[name] = dict(guidance=aggregate(baseline[sl]), untrained=aggregate(untrained[sl]), rl=aggregate(final[sl]))
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    lines = [f'# {args.task}: RL action replacement pilot', '',
             'E frozen. Guidance and RL use identical CPU physics, S0, and evaluation reward.', '',
             '| Split | Method | Median reward | Median progress | Survived |',
             '|---|---|---:|---:|---:|']
    for split in ('train', 'heldout', 'all'):
        for method in ('guidance', 'untrained', 'rl'):
            a = result[split][method]
            lines.append(f"| {split} | {method} | {a['reward_median']:.4f} | {a['progress_median']:.4f} | {a['survived']}/{a['candidates']} |")
    lines += ['', f'Training transitions: {args.updates*64*len(train):,}. Seed: {args.seed}.',
              f'Training reward: {protocol["train_reward"]}. Evaluation uses the original task score.',
              'Held-out means four generated morphologies excluded from policy training, not a new task or seed.',
              'CPU results must not be interpreted as identical to previous MJX results.',
              'This pilot does not train E or generate S with diffusion; no success conclusion from survival alone.']
    (args.out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('train','heldout','all')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
