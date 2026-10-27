"""Bounded CALM HLC/LLC verification; never join states across resets.

Run with CALM's Python 3.8 venv, not the MJX/TT interpreter.
"""
import argparse
import datetime
import subprocess
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def command(t):
    """World-frame velocity and facing yaw: forward, turn, lateral, stop."""
    knots = np.array([0, 2, 4, 6, 8, 10.0])
    values = np.array([[.35, 0, 0], [.35, 0, 0], [0, .35, math.pi/2],
                       [-.30, 0, math.pi/2], [0, 0, math.pi/2], [0, 0, math.pi/2]])
    i = min(np.searchsorted(knots, t, side='right') - 1, len(knots)-2)
    i = max(i, 0)
    u = np.clip((t-knots[i])/(knots[i+1]-knots[i]), 0, 1)
    u = u*u*(3-2*u)
    return values[i]*(1-u) + values[i+1]*u


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calm-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sync-reset', action='store_true', help='Separate reset-observation diagnostic, off for the original baseline')
    args = p.parse_args()
    root, out = args.calm_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    git_root = Path(__file__).resolve().parents[3]
    git_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=git_root,text=True).strip()
    source_paths = [Path(__file__).resolve(), root/'calm/env/tasks/humanoid.py',
                    root/'calm/learning/hrl_players.py', root/'calm/learning/calm_players.py']
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    os.environ['CALM_ROOT'] = str(root)
    os.chdir(root)
    sys.path.insert(0, str(root/'calm'))
    sys.argv = [sys.argv[0], '--task', 'QuadrupedHeading', '--test', '--headless',
                '--num_envs', '1', '--seed', str(args.seed),
                '--cfg_env', 'calm/data/cfg/a1_heading_lat_up_c8_hit2.yaml',
                '--cfg_train', 'calm/data/cfg/train/rlg/hrl_a1_heading_lat31zs.yaml',
                '--motion_file', 'calm/data/motions_a1/v2v3_31motions_squat2s_bfcut.pt',
                '--llc_checkpoint', 'checkpoints/lowLevelController.pth',
                '--checkpoint', 'checkpoints/highLevelPolicy.pth']
    import run as calm_run  # imports Isaac Gym before torch
    from utils.config import get_args, load_cfg, set_seed
    from isaacgym import gymapi
    from reset_observations import install
    if args.sync_reset:
        install(root/'assets/urdf/a1_description/urdf/a1.urdf')
    import torch
    import imageio.v2 as imageio

    calm_run.args = ca = get_args()
    cfg, train, _ = load_cfg(ca)
    calm_run.cfg, calm_run.cfg_train = cfg, train
    train['params']['seed'] = set_seed(args.seed, False)
    cfg['env']['motion_file'] = ca.motion_file
    # Allow a real 10-second rollout; do not change physical/controller parameters.
    original_episode_length = cfg['env']['episodeLength']
    cfg['env']['episodeLength'] = 302
    assert cfg['env']['stateInit'] == 'Default'
    train['params']['config']['train_dir'] = str(out/'unused_train_dir')
    manifest = dict(seed=args.seed, argv=sys.argv, cfg_env=cfg, cfg_train=train,
                    git_commit=git_commit, source_sha256=sources, sync_reset=args.sync_reset,
                    timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    original_episode_length=original_episode_length,
                    python=sys.version, torch_version=torch.__version__, numpy_version=np.__version__,
                    simulator='Isaac Gym Preview 4 GPU PhysX', robot='calm_a1',
                    asset_sha256=hashlib.sha256((root/'assets/urdf/a1_description/urdf/a1.urdf').read_bytes()).hexdigest(),
                    checkpoints={n: hashlib.sha256((root/'checkpoints'/n).read_bytes()).hexdigest()
                                 for n in ['lowLevelController.pth', 'highLevelPolicy.pth']})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=lambda x: vars(x) if isinstance(x, argparse.Namespace) else str(x)))
    # The upstream task disables graphics in headless mode unless sensors are enabled.
    from env.tasks.base_task import BaseTask
    original_init = BaseTask.__init__

    def camera_init(self, cfg, enable_camera_sensors=False):
        original_init(self, cfg, enable_camera_sensors=True)

    BaseTask.__init__ = camera_init
    runner = calm_run.build_alg_runner(calm_run.RLGPUAlgoObserver())
    runner.load(train); runner.reset()
    player = runner.create_player(); player.restore(ca.checkpoint)
    set_seed(args.seed, False)  # checkpoint restoration may replace random generator state
    task = player.env.task
    assert task.num_envs == 1 and player._llc_steps == 1
    dt = float(task.dt)
    steps = round(10/dt)
    assert abs(steps*dt-10) < 1e-6
    manifest['runtime'] = dict(hlc_observation_size=player.obs_shape[0],
                               llc_observation_size=player._llc_agent.obs_shape[0],
                               latent_dim=player._latent_dim, joint_actions=task.num_actions,
                               dt=dt)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    task._reset_task = lambda ids: task._heading_change_steps.__setitem__(ids, 1000000)
    obs = player.env_reset()
    initial_knees = task._knee_world_pos(task._dof_pos,task._humanoid_root_states[:,:3],task._humanoid_root_states[:,3:7])
    reset_fk_error = float((initial_knees-task._rigid_body_pos[:,task._knee_body_ids]).norm(dim=-1).max())
    if player.is_rnn:
        player.init_rnn()
    names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
    joints = task.gym.get_actor_dof_names(task.envs[0], task.humanoid_handles[0])
    feet = [names.index(n) for n in ['FR_foot','FL_foot','RR_foot','RL_foot']]
    camera_props = gymapi.CameraProperties(); camera_props.width = 960; camera_props.height = 544
    camera = task.gym.create_camera_sensor(task.envs[0], camera_props)
    if camera < 0:
        raise RuntimeError('Camera creation failed')
    states, actions, latents, targets, commands, contacts, foot_pos, torques = [], [], [], [], [], [], [], []

    def record():
        states.append(torch.cat([task._humanoid_root_states[0], task._dof_pos[0], task._dof_vel[0]]).cpu().numpy().copy())
        foot_pos.append(task._rigid_body_pos[0, feet].cpu().numpy().copy())
        contacts.append((task._contact_forces[0, feet].norm(dim=-1)>1).cpu().numpy())

    reason = 'duration'
    diagnostics = []
    try:
        with torch.no_grad(), imageio.get_writer(str(out/'rollout.mp4'), fps=round(1/dt)) as video:
            record()
            for k in range(steps):
                c = command(k*dt)
                speed = np.linalg.norm(c[:2])
                task._nocmd[:] = False
                task._tar_speed[:] = float(speed)
                task._tar_dir[:] = torch.tensor(c[:2]/speed if speed>1e-8 else [1.,0.], device=task.device)
                task._tar_facing_dir[:] = torch.tensor([math.cos(c[2]),math.sin(c[2])], device=task.device)
                task._tar_face_lat[:] = 0; task._tar_up_roll[:] = 0; task._tar_up_pitch[:] = 0
                task._compute_observations()
                # Refresh task observations after setting commands, before policy inference.
                obs = {'obs': torch.clamp(task.obs_buf, -player.env.clip_obs, player.env.clip_obs).to(player.device)}
                latent = player.get_action(obs, True).reshape(1,-1)
                llc = player._compute_llc_action(obs['obs'], latent)
                applied = llc.clamp(-player.env.clip_actions, player.env.clip_actions)
                actions.append(applied[0].cpu().numpy().copy())
                latents.append(torch.nn.functional.normalize(latent,dim=-1)[0].cpu().numpy().copy())
                targets.append(task._action_to_pd_targets(applied)[0].cpu().numpy().copy())
                commands.append(c)
                _, _, done, info = player.env.step(applied)
                record()
                diagnostics.append({name: getattr(task, name)[0].detach().cpu().tolist()
                                    for name in ['_terminate_buf', '_pos_protection_buf', '_knee_low_count', '_knee_term_buf']
                                    if hasattr(task, name)})
                task.gym.refresh_dof_force_tensor(task.sim)
                torques.append(task.dof_force_tensor[0].cpu().numpy().copy())
                if not np.isfinite(states[-1]).all():
                    reason = 'nonfinite'; break
                pos = states[-1][:3]
                task.gym.set_camera_location(camera, task.envs[0], gymapi.Vec3(*(pos+[2.,2.,1.])), gymapi.Vec3(*pos))
                task.gym.step_graphics(task.sim); task.gym.render_all_camera_sensors(task.sim)
                rgba = task.gym.get_camera_image(task.sim, task.envs[0], camera, gymapi.IMAGE_COLOR).reshape(544,960,4)
                video.append_data(rgba[:,:,:3])
                if bool(done[0]):
                    reason = 'terminated'; break
    finally:
        np.savez_compressed(out/'raw_partial.npz',state=np.asarray(states),llc_action=np.asarray(actions),
                            target_q=np.asarray(targets),command=np.asarray(commands),dt=dt)
    state = np.asarray(states)
    data = dict(state=state, foot_pos=foot_pos, contact=contacts, llc_action=actions,
                latent=latents, target_q=targets, torque=torques, command=commands,
                dt=dt, time=np.arange(len(state))*dt, joint_names=joints,
                body_names=names, quaternion_order='xyzw', frame='world_z_up')
    np.savez_compressed(out/'episode.npz', **data)
    velocity = state[1:,7:9]
    error = np.linalg.norm(velocity-np.asarray(commands)[:,:2],axis=1)
    finite_actions = all(np.isfinite(x).all() for x in [actions,targets,latents,commands])
    continuous = reason=='duration' and len(actions)==steps and finite_actions
    result = dict(status='FAIL' if not continuous else 'BLOCKED',
                  status_note='Full G0 approval also requires measured command fidelity; continuity alone never passes G0',
                  reset_fk_error_m=reset_fk_error, finite_states=bool(np.isfinite(state).all()),
                  finite_actions=finite_actions,continuous_10s=continuous, termination_reason=reason, resets_after_initial=0,
                  steps=len(actions), seconds=len(actions)*dt, dt=dt,
                  command_velocity_rmse_mps=float(np.sqrt(np.mean(error**2))),
                  path_length_m=float(np.linalg.norm(np.diff(state[:,:2],axis=0),axis=1).sum()),
                  final_displacement_m=float(np.linalg.norm(state[-1,:2]-state[0,:2])),
                  note='Continuity is separate from command tracking accuracy; not a reference approval.')
    result['terminal_diagnostics'] = diagnostics[-1] if diagnostics else {}
    (out/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    task.gym.destroy_sim(task.sim)
    return 0 if continuous else 2


if __name__ == '__main__':
    sys.exit(main())
