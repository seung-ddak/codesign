"""Actual MuJoCo rollouts -> original t2 modality data, with bounded memory."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TT = ROOT.parent / 'transformer-transformer'
sys.path[:0] = [str(TT), str(ROOT / 'code/motion_tracking')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def native_env(E):
    """Tokenization copy only: physical replay always uses the original model."""
    from dm_control import mjcf
    from pd_replay import build_model
    from t2.env.base_env import BaseEnv
    seen = []
    model = build_model('00000011', uniforms=E, kp=30, torque_limit=33.5,
                        flat_ground=True, scene_fn=seen.append)
    model.opt.timestep = .005
    spec = seen[0]
    for i, actuator in enumerate(spec.actuators):
        actuator.gainprm = model.actuator_gainprm[i]
        actuator.biasprm = model.actuator_biasprm[i]
        actuator.forcerange = model.actuator_forcerange[i]
    root = mjcf.from_xml_string(spec.to_xml(), assets=spec.assets, escape_separators=True)
    for site in list(root.find_all('site')):
        site.remove()
    # A tracking site on the collision-free wrapper creates a second free-link
    # token. Attach it to the rigid trunk that already owns collision geometry.
    trunk = next(b for b in root.find_all('body') if b.name.endswith('quadruped_body'))
    trunk.add('site', name='chain_track', group=1, pos=[0, 0, 0], size=[.001])
    env = BaseEnv(500, .005, .02, lambda _: root)
    env.reset_hardware(0)
    if (model.nq, model.nv, model.nu) != (env.m.nq, env.m.nv, env.m.nu):
        raise ValueError('Tokenizer changed joint/actuator counts')
    for kind, count in [('joint', model.njnt), ('actuator', model.nu)]:
        for i in range(count):
            if getattr(model, kind)(i).name.replace('/', '\\') != getattr(env.m, kind)(i).name:
                raise ValueError(f'Tokenizer reordered {kind} {i}')
    np.testing.assert_allclose(model.actuator_gainprm, env.m.actuator_gainprm)
    np.testing.assert_allclose(model.actuator_biasprm, env.m.actuator_biasprm)
    import mujoco
    original_data=mujoco.MjData(model)
    original_data.qpos[:]=model.keyframe('home').qpos
    env.d.qpos[:]=original_data.qpos
    mujoco.mj_forward(model,original_data);mujoco.mj_forward(env.m,env.d)
    mass=np.zeros((model.nv,model.nv));token_mass=np.zeros_like(mass)
    mujoco.mj_fullM(model,original_data,mass);mujoco.mj_fullM(env.m,env.d,token_mass)
    env.tokenization_mass_relative_error=float(np.max(np.abs(mass-token_mass))/np.max(np.abs(mass)))
    if env.tokenization_mass_relative_error>.01:
        raise ValueError('Original tokenizer changed the mass matrix by more than 1%')
    hardware = {k: v for k, v in env.hardware_dict.items() if not k.startswith('metadata/')}
    return model, env, hardware


def observations(model, env, qpos, qvel, previous_controls):
    import mujoco
    data = mujoco.MjData(model)
    rows = {}
    for q, v, control in zip(qpos, qvel, previous_controls):
        data.qpos[:], data.qvel[:], data.ctrl[:] = q, v, control
        mujoco.mj_forward(model, data)
        env.d.qpos[:], env.d.qvel[:], env.d.ctrl[:] = q, v, control
        mujoco.mj_forward(env.m, env.d)
        obs = env.get_obs()
        # No dynamics from the tokenization copy enter the training observations.
        obs['actuator_obs/force'] = data.actuator_force[:, None].astype(np.float32)
        obs['actuator_obs/velocity'] = data.actuator_velocity[:, None].astype(np.float32)
        site = env.track_mj_site_ids[0]
        obs['track_link_obs/pos'] = env.d.site_xpos[[site]].astype(np.float32)
        obs['track_link_obs/rotmat'] = env.d.site_xmat[[site]].astype(np.float32)
        for k, value in obs.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(k)
            rows.setdefault(k, []).append(value)
    return {k: np.asarray(v) for k, v in rows.items()}


def reference_candidate(folder, episode_index=0):
    folder=Path(folder);rec=json.loads((folder/'complete.json').read_text())
    ep=rec['episodes'][episode_index]
    times=np.linspace(0,ep['length']-1,8,dtype=int)
    with np.load(folder/'hardware.npz') as z:
        candidate={k:z[k] for k in z.files}
    for group in ['link','dyna_joint','fixed_joint','actuator']:
        count=next(len(v) for k,v in candidate.items() if k.startswith(group+'/'))
        candidate[group+'/mask']=np.zeros((count,1),bool)
    for key,filename in rec['fields'].items():
        array=np.load(folder/filename,mmap_mode='r')
        candidate[key]=np.array(array[ep['start']+times]).reshape(-1,array.shape[-1])
        group=key.split('/')[0]
        candidate[group+'/time/id']=np.broadcast_to(times[:,None,None],(8,array.shape[1],1)).reshape(-1,1)
    return candidate


def prepare(source, dest, limit=0):
    """Commit one shard at a time; an interrupted temporary shard is never reused."""
    source, dest = Path(source).resolve(), Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    audit = json.loads((source / 'audit.json').read_text())
    if not audit['complete'] or audit['invalid_files'] or audit.get('failures'):
        raise ValueError('Full source audit has not passed')
    plan = json.loads((source / 'plan.json').read_text())
    hashes = sorted({m['sha256'] for m in plan['motions']})
    if len(hashes) != 10000:
        raise ValueError('Expected 10,000 unique original source hashes')
    order = np.random.default_rng(0).permutation(hashes)
    validation = set(order[:2000].tolist())
    split = dict(train=sorted(set(hashes)-validation), validation=sorted(validation))
    save_json(dest / 'source_split.json', split)
    manifest_hash = digest(source / 'plan.json')
    files = sorted((source / 'rollouts').glob('*/variant_*.npz'))
    if len(files) != 400:
        raise ValueError('Expected 200 training morphologies × two variants')
    if limit:
        files = files[:limit]
    records = []
    current_design = None
    started=time.time();processed=0
    for index, path in enumerate(files):
        name = path.parent.name + '_' + path.stem
        folder = dest / name
        marker = folder / 'complete.json'
        checksum = digest(path)
        if marker.exists():
            rec = json.loads(marker.read_text())
            if rec['source_sha256'] != checksum or rec['plan_sha256'] != manifest_hash:
                raise ValueError('Prepared data source changed')
            records.append(rec)
            continue
        with np.load(path, allow_pickle=False) as z:
            values={key:z[key] for key in ['E','dt','qpos','qvel','target_q','valid','group_id']}
            E = values['E'].copy()
            if current_design != path.parent.name:
                model, env, hardware = native_env(E)
                current_design = path.parent.name
            if not np.isclose(values['dt'], .02) or not np.isfinite(values['qvel']).all():
                raise ValueError('Invalid physical timestep/state')
            episodes = []
            chunks = {}
            offset = 0
            for i, mask in enumerate(values['valid']):
                length = int(mask.sum())
                if np.any(mask[1:] & ~mask[:-1]):
                    raise ValueError('Non-prefix valid mask')
                target = values['target_q'][i, :length].copy()
                if length < 8:
                    # Original RandomTimestepSampler samples eight distinct times.
                    # These short episodes cannot satisfy that native contract.
                    continue
                previous = np.concatenate([target[:1], target[:-1]])
                obs = observations(model, env, values['qpos'][i, :length],
                                   values['qvel'][i, :length], previous)
                obs['ctrl/target_qpos'] = target[..., None]
                obs['target_pose/pos'] = obs['track_link_obs/pos'].copy()
                obs['target_pose/rotmat'] = obs['track_link_obs/rotmat'].copy()
                for k, v in obs.items():
                    chunks.setdefault(k, []).append(v)
                source_hash = str(values['group_id'][i])
                if source_hash not in hashes:
                    raise ValueError('Unknown source hash')
                episodes.append(dict(start=offset, length=length, source_hash=source_hash,
                                     split='validation' if source_hash in validation else 'train'))
                offset += length
            with tempfile.TemporaryDirectory(dir=dest, prefix=name + '.tmp-') as tmp:
                tmp = Path(tmp)
                mapping = {}
                for j, (key, values) in enumerate(chunks.items()):
                    filename = f'field_{j:02d}.npy'
                    np.save(tmp / filename, np.concatenate(values))
                    mapping[key] = filename
                np.savez(tmp / 'hardware.npz', **hardware)
                rec = dict(name=name, source_sha256=checksum, plan_sha256=manifest_hash,
                           fields=mapping, episodes=episodes, E=E.tolist(), frames=offset,
                           tokenization_mass_relative_error=env.tokenization_mass_relative_error)
                save_json(tmp / 'complete.json', rec)
                if folder.exists():
                    raise FileExistsError(folder)
                tmp.rename(folder)
                # TemporaryDirectory cleanup is safe after the atomic rename.
            records.append(rec)
            processed+=1
        elapsed=time.time()-started
        save_json(dest / 'progress.json', dict(shards=index+1,total=len(files),seconds=elapsed,
                  estimated_remaining_seconds=elapsed/max(processed,1)*(len(files)-index-1)))
        print(f'prepared {index+1}/{len(files)} {name} frames={offset}', flush=True)
    if not limit:
        covered={ep['source_hash'] for rec in records for ep in rec['episodes']}
        if covered!=set(hashes):
            raise ValueError('Native temporal sampling lost original source coverage')
        save_json(dest / 'manifest.json', dict(source=str(source), plan_sha256=manifest_hash,
                                              shards=[r['name'] for r in records], split=split,
                                              frames=sum(r['frames'] for r in records),
                                              episodes=sum(len(r['episodes']) for r in records),
                                              original_sources=len(covered)))
    return records


class NativeDataset:
    def __init__(self, data, cfg, split):
        from collections import OrderedDict
        self.data, self.cfg = Path(data), cfg
        self.entries = []
        self.cache = OrderedDict()
        manifest = json.loads((self.data / 'manifest.json').read_text())
        for name in manifest['shards']:
            rec = json.loads((self.data / name / 'complete.json').read_text())
            self.entries.extend((name, ep) for ep in rec['episodes'] if ep['split'] == split)
        if not self.entries:
            raise ValueError('Empty source split')

    def __len__(self):
        return len(self.entries)

    def shard(self, name):
        if name not in self.cache:
            folder = self.data / name
            rec = json.loads((folder / 'complete.json').read_text())
            with np.load(folder / 'hardware.npz') as z:
                hardware = {k: z[k] for k in z.files}
            arrays = {k: np.load(folder / filename, mmap_mode='r') for k, filename in rec['fields'].items()}
            self.cache[name] = (hardware, arrays)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name]

    def __getitem__(self, index):
        from t2.data.pad import pad
        name, ep = self.entries[index]
        hardware, arrays = self.shard(name)
        times = np.random.choice(ep['length'], 8, replace=False)
        anchor = times.min()
        raw = {**hardware, **{k: np.array(v[ep['start'] + times]) for k, v in arrays.items()}}
        for group in {k.split('/')[0] for k in arrays}:
            n = next(v.shape[1] for k, v in arrays.items() if k.startswith(group + '/'))
            raw[group + '/time/id'] = np.broadcast_to((times-anchor)[:, None, None], (8, n, 1))
        return pad(raw, dict(self.cfg.seq_len))
