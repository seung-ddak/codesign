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

from train_task_rl import TaskEnv, build_task_model, features, mujoco, DENSE_REWARD_KEYS
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
    rows, saved, rewards = [], {'reward_component_names': np.array(DENSE_REWARD_KEYS), 'record_version': np.array(1)}, []
    for i in range(len(E)):
        start = source['pred'][i,0,:42]
        if not np.isfinite(source['pred'][i]).all():
            raise FloatingPointError('nonfinite generated trajectory')
        initial_reasons = []
        if start[2] <= 0: initial_reasons.append('nonpositive_height')
        if start[7] > -.5: initial_reasons.append('initial_tilt')
        if initial_reasons:
            rewards.append(-20.)
            rows.append(dict(candidate=i, terminal_reward=-20., original_reward=None,
                             progress=0., survived=False, invalid_initial=True, steps=0,
                             prediction_position_rmse=0., execution_seconds=0., max_seconds=(len(trajectory[i])-1)*.02,
                             termination_reason='invalid_initial', initial_invalid_reasons=initial_reasons,
                             physical_return=0., consistency_penalty=0., invalid_penalty=-20.,
                             reward_components={k:0. for k in DENSE_REWARD_KEYS}))
            saved[f'tokens_{i}'] = source['pred'][i,:1].copy()
            q0,v0 = s0_to_q(start)
            saved[f'qpos_{i}'],saved[f'qvel_{i}'] = q0[None],v0[None]
            saved[f'ctrl_{i}']=np.empty((0,12))
            saved[f'torque_{i}']=np.empty((0,12))
            saved[f'step_reward_{i}']=np.empty(0)
            saved[f'reward_components_{i}']=np.empty((0,len(DENSE_REWARD_KEYS)))
            continue
        env = TaskEnv(source, i, task, 'dense')
        feet = set(env.feet)
        if sum(env.tokens[0][38:42]) < 2: initial_reasons.append('insufficient_foot_contacts')
        if any(c.dist < -.002 and env.floor in (c.geom1,c.geom2)
               and (c.geom2 if c.geom1 == env.floor else c.geom1) not in feet
               for c in env.data.contact[:env.data.ncon]): initial_reasons.append('nonfoot_floor_penetration')
        invalid = bool(initial_reasons)
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
                         steps=env.t, prediction_position_rmse=position_mse**.5,
                         execution_seconds=env.t*.02, max_seconds=(env.horizon-1)*.02,
                         termination_reason='invalid_initial' if invalid else ('fall' if fell else 'time_limit'),
                         initial_invalid_reasons=initial_reasons, physical_return=float(env.training_return),
                         consistency_penalty=0. if invalid else 5.*position_mse, invalid_penalty=-20. if invalid else 0.,
                         reward_components=dict(zip(DENSE_REWARD_KEYS,np.asarray(env.reward_component_history).reshape(-1,len(DENSE_REWARD_KEYS)).sum(0).tolist()))))
        saved[f'tokens_{i}'], saved[f'qpos_{i}'] = actual, np.asarray(env.qraw)
        saved[f'qvel_{i}'] = np.asarray(env.vraw)
        saved[f'ctrl_{i}'] = np.asarray(env.applied_controls).reshape(-1,12)
        saved[f'torque_{i}'] = np.asarray(env.torque_history).reshape(-1,12)
        saved[f'step_reward_{i}'] = np.asarray(env.reward_history)
        saved[f'reward_components_{i}'] = np.asarray(env.reward_component_history).reshape(-1,len(DENSE_REWARD_KEYS))
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
    parser.add_argument('--fixed-tt-prediction', action='store_true', help='Cache one joint TT prediction; train PPO on the same candidate noise without further TT calls')
    parser.add_argument('--record-rollouts', action='store_true', help='Save every training rollout for stage-0 verification')
    parser.add_argument('--shared-window-actor', action='store_true', help='Share S/A actor weights and exploration noise across windows; retain one global E head')
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
    controller = DenoisingPolicy(episode_states=args.episode_states, horizon=ta['horizon'], diffusion_steps=1 if args.fixed_tt_prediction else len(beta),
                                 shared_window_actor=args.shared_window_actor)
    metadata = dict(task=args.task, tt_sha256=hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
                    seed=args.seed, updates=args.updates, batch=args.batch,
                    episode_states=args.episode_states, ctrl_dt=.02, horizon=ta['horizon'],
                    diffusion_steps=controller.config['diffusion_steps'], tt_schedule_steps=len(beta),
                    fixed_tt_prediction=args.fixed_tt_prediction, shared_window_actor=args.shared_window_actor, ppo_epochs_per_update=4,
                    record_rollouts=args.record_rollouts, record_version=1, mujoco_version=mujoco.__version__,
                    numpy_version=np.__version__, torch_version=str(torch.__version__),
                    source_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['train_diffusion_rl.py','train_task_rl.py','rl_diffusion_guidance.py','joint_diffusion_guidance.py']},
                    terminal_reward='dense physical return - 5 * predicted/physical position MSE; invalid start=-20',
                    s0='physical rest(E)', backend='MuJoCo CPU',
                    action='bounded epsilon residual per E coordinate and per S/A channel/window',
                    train_noise_seeds=[args.seed if args.fixed_tt_prediction else args.seed+i for i in range(args.updates)],
                    eval_noise_seed=args.seed+1000000,
                    limitation='PPO denoising implementation check; short training does not establish task success')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'protocol.json').write_text(json.dumps(metadata, indent=2))
    for filename in ['train_diffusion_rl.py','rl_diffusion_guidance.py','joint_diffusion_guidance.py','train_task_rl.py']:
        source = Path(__file__).parent/filename
        (args.out/filename).write_bytes(source.read_bytes())
    prediction_caches = {}
    def sample(seed, policy=None, scale=0.):
        return joint_diffusion_full_reward(net, reward, n=args.batch, episode_states=args.episode_states,
                   horizon=ta['horizon'], state_dim=42, action_dim=12, e_dim=13,
                   mu=torch.as_tensor(ck['mu']), sd=torch.as_tensor(ck['sd']), beta=beta, abar=abar,
                   emu=.5, esd=.5 if ta['norm']=='minmax' else .25,
                   clip=ta['clip'], seed=seed, guidance_scale=scale,
                   s0_fn=physical_rest_s0, denoising_policy=policy,
                   fixed_prediction_cache=prediction_caches.setdefault(seed, {}) if args.fixed_tt_prediction else None)
    comparisons = {}
    def evaluate(name, policy=None, scale=0.):
        trajectory, E, audit = sample(metadata['eval_noise_seed'], policy, scale)
        scores, rows, saved = replay_generated(trajectory, E, args.task)
        comparisons[name] = dict(rows=rows, audit=audit)
        np.savez_compressed(args.out/f'{name}.npz', pred=trajectory.numpy(), E=E.numpy(), **saved)
        print(name, rows, flush=True)
    evaluate('unguided')
    if not args.fixed_tt_prediction:
        evaluate('reward_gradient', scale=10.)
    evaluate('rl_before', controller)
    algo = PPO(controller.policy, num_learning_epochs=4, num_mini_batches=1, learning_rate=3e-4,
               gamma=1., lam=.95, entropy_coef=.001)
    algo.init_storage(args.batch, controller.config['diffusion_steps'], [controller.obs_dim], [controller.obs_dim], [controller.action_dim])
    controller.algorithm = algo
    best_reward = -float('inf')
    start = time.monotonic()
    for update in range(args.updates):
        trajectory, E, audit = sample(args.seed if args.fixed_tt_prediction else args.seed+update, controller)
        scores, rows, rollout_saved = replay_generated(trajectory, E, args.task)
        if update == 0 or float(scores.mean()) > best_reward:
            best_reward = float(scores.mean())
            controller.save(args.out/'policy_best_training.pt', dict(metadata, rollout_update=update+1, checkpoint_timing='before PPO update', training_reward=best_reward))
            np.savez_compressed(args.out/'best_training.npz', pred=trajectory.numpy(), E=E.numpy(), **rollout_saved)
            (args.out/'best_training.json').write_text(json.dumps(dict(update=update+1, mean_reward=best_reward, rows=rows), indent=2))
        if args.record_rollouts:
            folder = args.out/'training_rollouts'
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'update_{update+1:05d}.npz', pred=trajectory.numpy(), E=E.numpy(), **rollout_saved)
        controller.finish(scores)
        buffer_rewards = algo.storage.rewards.detach().cpu().numpy()[...,0]
        if not np.allclose(buffer_rewards[-1],scores.numpy(),rtol=0,atol=1e-6) or np.any(buffer_rewards[:-1] != 0):
            raise AssertionError('PPO reward buffer differs from physical terminal rewards')
        ppo_terminal_rewards = buffer_rewards[-1].tolist()
        metrics = algo.update()
        if not np.isfinite(metrics).all() or not all(torch.isfinite(p).all() for p in controller.policy.parameters()):
            raise FloatingPointError('nonfinite PPO update')
        record = dict(update=update+1, terminal_reward_mean=float(scores.mean()),
                      physical_episodes=(update+1)*args.batch, denoising_transitions=(update+1)*args.batch*controller.config['diffusion_steps'],
                      seconds=time.monotonic()-start, metrics=list(map(float,metrics)), rows=rows, audit=audit,
                      policy_update_before=update, policy_update_after=update+1, ppo_terminal_rewards=ppo_terminal_rewards)
        with (args.out/'training.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        controller.save(args.out/'policy.pt', metadata)
        print('update',update+1,'physical reward',float(scores.mean()),flush=True)
    controller.algorithm = None
    controller.policy.eval()
    evaluate('rl_after', controller)
    if args.fixed_tt_prediction:
        (args.out/'tt_prediction_audit.json').write_text(json.dumps({str(seed): {'joint_prediction_passes': cache['tt_prediction_passes']} for seed, cache in prediction_caches.items()}, indent=2))
    (args.out/'summary.json').write_text(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
