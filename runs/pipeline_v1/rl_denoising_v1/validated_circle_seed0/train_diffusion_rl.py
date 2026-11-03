"""Train PPO to steer joint diffusion epsilon using terminal CPU physics reward.

The TT is frozen. E and future S/A are generated jointly; S0 = rest(E).
No reward gradients and no separate closed-loop action policy are used.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_task_rl import TaskEnv, build_task_model, features, mujoco
from rl_diffusion_guidance import DenoisingPolicy
from rsl_rl.algorithms import PPO
from joint_diffusion_guidance import joint_diffusion_full_reward
from tt_model import make_net, schedule, check_rest_checkpoint
from task_reward import TrackReward, ZigzagReward
from s0_geometry import s0_to_q

ROOT = Path(__file__).resolve().parents[2]


def physical_rest_s0(E):
    states = []
    for e in E.detach().cpu().numpy():
        m = build_task_model(e)
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe('home').qpos
        d.qpos[2] = 1.
        d.ctrl[:] = m.keyframe('home').ctrl
        mujoco.mj_forward(m, d)
        feet = np.array([m.geom(f'{leg}_foot').id for leg in ('FR','FL','RR','RL')])
        d.qpos[2] -= (d.geom_xpos[feet,2]-m.geom_size[feet,0]).min()
        mujoco.mj_step(m, d, nstep=round(1./m.opt.timestep))
        floor = m.geom('floor').id
        contact = [any(floor in (c.geom1,c.geom2) and f in (c.geom1,c.geom2) and c.dist <= .001
                       for c in d.contact[:d.ncon]) for f in feet]
        state = features(d.qpos[None].copy(), d.qvel[None].copy(), np.array([contact]))[0]
        if not np.isfinite(state).all():
            raise FloatingPointError('nonfinite rest state')
        states.append(state)
    return torch.tensor(np.stack(states), dtype=E.dtype, device=E.device)


def replay_generated(trajectory, E, task):
    """Actual physical reward; position consistency penalizes fictitious generated S."""
    source = dict(pred=trajectory.detach().cpu().numpy(), E=E.detach().cpu().numpy())
    rows, saved, rewards = [], {}, []
    for i in range(len(E)):
        start = source['pred'][i,0,:42]
        if not np.isfinite(source['pred'][i]).all():
            raise FloatingPointError('nonfinite generated trajectory')
        if start[2] <= 0 or start[7] > -.5:
            rewards.append(-20.)
            rows.append(dict(candidate=i, terminal_reward=-20., original_reward=None,
                             progress=0., survived=False, invalid_initial=True, steps=0,
                             prediction_position_rmse=0.))
            saved[f'tokens_{i}'] = source['pred'][i,:1].copy()
            saved[f'qpos_{i}'] = s0_to_q(start)[0][None]
            continue
        env = TaskEnv(source, i, task, 'dense')
        feet = set(env.feet)
        invalid = env.start[2] <= 0 or env.start[7] > -.5
        invalid |= sum(env.tokens[0][38:42]) < 2
        invalid |= any(c.dist < -.002 and env.floor in (c.geom1,c.geom2)
                       and (c.geom2 if c.geom1 == env.floor else c.geom1) not in feet
                       for c in env.data.contact[:env.data.ncon])
        fell = False
        if not invalid:
            for action in env.actions[:-1]:
                _, _, done, fell = env.step(action, target=True)
                if done:
                    break
        score, terms = env.score()
        actual = np.asarray(env.tokens)
        position_mse = float(np.mean((source['pred'][i,:len(actual),:3]-actual[:,:3])**2))
        reward = -20. if invalid else env.training_return - 5.*position_mse
        if not np.isfinite(reward):
            raise FloatingPointError('nonfinite physical terminal reward')
        rewards.append(reward)
        rows.append(dict(candidate=i, terminal_reward=reward, original_reward=score,
                         progress=terms['r_progress']*env.reward.unit_scale,
                         survived=bool(not invalid and not fell), invalid_initial=bool(invalid),
                         steps=env.t, prediction_position_rmse=position_mse**.5))
        saved[f'tokens_{i}'], saved[f'qpos_{i}'] = actual, np.asarray(env.qraw)
    return trajectory.new_tensor(rewards), rows, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, default=ROOT/'runs/pipeline_v1/fast_mode24_v16_twin/tt_joint/checkpoint.pt')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task', choices=['zigzag','circle'], default='zigzag')
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--episode-states', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.updates < 1 or args.batch < 2 or args.episode_states < 2:
        parser.error('updates >=1, batch >=2, episode-states >=2 required')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, weights_only=False, map_location='cpu')
    ta = ck['args']
    check_rest_checkpoint(ta)
    if 2 not in ta['modes']:
        parser.error('TT must support Mode 2')
    net = make_net(ta).eval().requires_grad_(False)
    net.load_state_dict(ck['net'])
    beta, abar = schedule(ta['diff_steps'], 'cpu')
    reward = ZigzagReward() if args.task == 'zigzag' else TrackReward()
    controller = DenoisingPolicy(episode_states=args.episode_states, horizon=ta['horizon'], diffusion_steps=len(beta))
    metadata = dict(task=args.task, tt_sha256=hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
                    seed=args.seed, updates=args.updates, batch=args.batch,
                    terminal_reward='dense physical return - 5 * predicted/physical position MSE; invalid start=-20',
                    s0='physical rest(E)', backend='MuJoCo CPU',
                    action='bounded epsilon residual per E coordinate and per S/A channel/window',
                    train_noise_seeds=[args.seed+i for i in range(args.updates)],
                    eval_noise_seed=args.seed+1000000,
                    limitation='PPO denoising implementation check; short training does not establish task success')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'protocol.json').write_text(json.dumps(metadata, indent=2))
    for filename in ['train_diffusion_rl.py','rl_diffusion_guidance.py','joint_diffusion_guidance.py','train_task_rl.py']:
        source = Path(__file__).parent/filename
        (args.out/filename).write_bytes(source.read_bytes())
    def sample(seed, policy=None, scale=0.):
        return joint_diffusion_full_reward(net, reward, n=args.batch, episode_states=args.episode_states,
                   horizon=ta['horizon'], state_dim=42, action_dim=12, e_dim=13,
                   mu=torch.as_tensor(ck['mu']), sd=torch.as_tensor(ck['sd']), beta=beta, abar=abar,
                   emu=.5, esd=.5 if ta['norm']=='minmax' else .25,
                   clip=ta['clip'], seed=seed, guidance_scale=scale,
                   s0_fn=physical_rest_s0, denoising_policy=policy)
    comparisons = {}
    def evaluate(name, policy=None, scale=0.):
        trajectory, E, audit = sample(metadata['eval_noise_seed'], policy, scale)
        scores, rows, saved = replay_generated(trajectory, E, args.task)
        comparisons[name] = dict(rows=rows, audit=audit)
        np.savez_compressed(args.out/f'{name}.npz', pred=trajectory.numpy(), E=E.numpy(), **saved)
        print(name, rows, flush=True)
    evaluate('unguided')
    evaluate('reward_gradient', scale=10.)
    evaluate('rl_before', controller)
    algo = PPO(controller.policy, num_learning_epochs=4, num_mini_batches=1, learning_rate=3e-4,
               gamma=1., lam=.95, entropy_coef=.001)
    algo.init_storage(args.batch, len(beta), [controller.obs_dim], [controller.obs_dim], [controller.action_dim])
    controller.algorithm = algo
    start = time.monotonic()
    for update in range(args.updates):
        trajectory, E, audit = sample(args.seed+update, controller)
        scores, rows, _ = replay_generated(trajectory, E, args.task)
        controller.finish(scores)
        metrics = algo.update()
        if not np.isfinite(metrics).all() or not all(torch.isfinite(p).all() for p in controller.policy.parameters()):
            raise FloatingPointError('nonfinite PPO update')
        record = dict(update=update+1, terminal_reward_mean=float(scores.mean()),
                      physical_episodes=(update+1)*args.batch, denoising_transitions=(update+1)*args.batch*len(beta),
                      seconds=time.monotonic()-start, metrics=list(map(float,metrics)), rows=rows, audit=audit)
        with (args.out/'training.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        controller.save(args.out/'policy.pt', metadata)
        print('update',update+1,'physical reward',float(scores.mean()),flush=True)
    controller.algorithm = None
    controller.policy.eval()
    evaluate('rl_after', controller)
    (args.out/'summary.json').write_text(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
