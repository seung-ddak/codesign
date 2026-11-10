"""Original t2 model/loss/EMA with streaming native data and resumable updates."""
import copy
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from native_tt_data import TT, NativeDataset, atomic_torch, save_json, digest


def configure(out):
    import hydra
    import t2.utils.misc  # registers the original eval resolver
    from omegaconf import OmegaConf
    config_dir = Path(out) / 'original_config'
    if not config_dir.exists():
        shutil.copytree(TT / 'config', config_dir)
        path = config_dir / 'train.yaml'
        path.write_text(path.read_text().replace('  - addon_hardware_bimanual_diffusion\n',
                                                 '  - addon_hardware_quadruped_diffusion\n'))
    with hydra.initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        cfg = hydra.compose(config_name='train', overrides=['hardware_dataset_path=unused_native_shards'])
    assert cfg.model.backbone.depth == 8 and cfg.model.backbone.num_heads == 4
    assert cfg.model.tasks.hardware_gen.noise_scheduler.num_train_timesteps == 100
    assert cfg.seq_len.rollout_steps == 8
    OmegaConf.save(cfg, Path(out) / 'resolved_config.yaml', resolve=True)
    return cfg


def model_and_augmentation(cfg):
    import hydra
    model = hydra.utils.instantiate(cfg.model)
    augmentation = hydra.utils.instantiate(cfg.datasets.clean.batch_process_fn)
    return model, augmentation


def attach_normalization(model, state):
    from t2.data.normalization_utils import normalization_dict_from_state_dict
    norms = normalization_dict_from_state_dict(state)
    for task in model.tasks.values():
        for adapter in task.adapters.values():
            for key, norm in norms.items():
                if key.split('/')[0] == adapter.modality.name:
                    adapter.modality.attr_norms['/' + key.split('/', 1)[1]] = norm
    return norms


def normalization(data, cfg, model, path):
    """Exact streaming minima/maxima, after original encoders, training sources only."""
    if Path(path).exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    from t2.data.normalization import Normalization
    dataset = NativeDataset(data, cfg, 'train')
    adapters = {a.modality.name: a for a in model.tasks.hardware_gen.adapters.values()}
    bounds = {}
    manifest = json.loads((Path(data) / 'manifest.json').read_text())
    for index, name in enumerate(manifest['shards']):
        rec = json.loads((Path(data) / name / 'complete.json').read_text())
        hardware, arrays = dataset.shard(name)
        spans = [(ep['start'], ep['start']+ep['length']) for ep in rec['episodes'] if ep['split']=='train']
        if not spans:
            continue
        for key, value in {**hardware, **arrays}.items():
            group, attr = key.split('/', 1)
            adapter = adapters[group]
            if key.endswith('/id') or attr not in adapter.modality.attrs:
                continue
            pieces = [(0, len(value))] if key in hardware else spans
            for lo, hi in pieces:
                for start in range(lo, hi, 4096):
                    x = torch.tensor(np.array(value[start:min(hi,start+4096)]), dtype=torch.float32)
                    if attr in adapter.attr_encoders:
                        x = adapter.attr_encoders[attr](x)
                    x = x.reshape(-1, adapter.modality.attrs[attr])
                    low, high = x.amin(0), x.amax(0)
                    if key.endswith('/rotmat') or key.endswith('/imat'):
                        low, high = torch.full_like(low, -1), torch.ones_like(high)
                    if not torch.isfinite(x).all():
                        raise FloatingPointError(key)
                    if key in bounds:
                        low, high = torch.minimum(bounds[key][0],low), torch.maximum(bounds[key][1],high)
                    bounds[key] = (low, high)
        print(f'normalization {index+1}/{len(manifest["shards"])}', flush=True)
    norms = torch.nn.ModuleDict()
    for key, (low, high) in bounds.items():
        # Same positional augmentation envelope used by the original train.py.
        if key.endswith('/pos') and key.split('/')[0] in ['target_pose','free_link_obs','track_link_obs']:
            low = low - torch.tensor([.3,.3,0.]); high = high + torch.tensor([.3,.3,0.])
        norms[key] = Normalization({'q0/0':low, 'q100/0':high})
    atomic_torch(path, norms.state_dict())
    return norms.state_dict()


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    np.random.set_state(state['numpy']); random.setstate(state['python'])


def load_model(out, device):
    cfg = configure(out)
    model, aug = model_and_augmentation(cfg)
    checkpoint = torch.load(Path(out)/'checkpoint.pt',map_location='cpu',weights_only=False)
    attach_normalization(model, checkpoint['normalization'])
    model.load_state_dict(checkpoint['model'])
    from diffusers.training_utils import EMAModel
    ema = EMAModel(model.parameters(),power=.75,update_after_step=0)
    ema.load_state_dict(checkpoint['ema']); ema.copy_to(model.parameters())
    return model.to(device).eval().requires_grad_(False), cfg, aug


def train(data, out, steps=20480, microbatch=1, device='cuda', stop_after=None):
    from diffusers.training_utils import EMAModel
    from transformers import get_scheduler
    from torch.utils.data import default_collate
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    if 1024 % microbatch:
        raise ValueError('Microbatch must divide original effective batch 1024')
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    cfg = configure(out)
    model, augmentation = model_and_augmentation(cfg)
    norms = normalization(data,cfg,model,out/'normalization.pt')
    attach_normalization(model,norms)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
    scheduler = get_scheduler('cosine',optimizer,num_warmup_steps=500,num_training_steps=steps)
    ema = EMAModel(model.parameters(),power=.75,update_after_step=0); ema.to(device)
    dataset = NativeDataset(data,cfg,'train')
    validation = NativeDataset(data,cfg,'validation')
    # Draw with replacement from the saved RNG: no prefetched batches or hidden
    # iterator position can be lost on restart.
    checkpoint_path = out/'training_state.pt'
    first = 0
    contract=dict(steps=steps,effective_batch=1024,microbatch=microbatch,
                  data_sha256=digest(Path(data)/'manifest.json'),config_sha256=digest(out/'resolved_config.yaml'))
    if checkpoint_path.exists():
        ck = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if ck['contract'] != contract:
            raise ValueError('Resume configuration differs')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); ema.load_state_dict(ck['ema']);ema.to(device)
        first=ck['step']; restore_rng(ck['rng'])
    bf16 = device=='cuda' and torch.cuda.is_bf16_supported()
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(.75)
    if not bf16:
        model.tasks.hardware_gen.use_flash_attn=False
    def save(step):
        atomic_torch(checkpoint_path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),ema=ema.state_dict(),step=step,rng=rng_state(),contract=contract,
            normalization=norms))
    if first==0:save(0)
    started=time.time()
    model.train()
    def validate():
        state=rng_state();torch.manual_seed(1);np.random.seed(1)
        model.eval();losses=[]
        with torch.no_grad():
            for _ in range(32):
                batch=default_collate([validation[np.random.randint(len(validation))]])
                batch=augmentation({k:v.to(device) for k,v in batch.items()})
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=bf16):
                    losses.append(float(model([(['hardware_gen'],batch)],is_eval=False)['loss']))
        restore_rng(state);model.train()
        value=float(np.mean(losses))
        if not np.isfinite(value):raise FloatingPointError('Validation loss')
        return value
    for step in range(first+1,min(steps,stop_after)+1 if stop_after else steps+1):
        optimizer.zero_grad(set_to_none=True)
        total=0.
        for _ in range(1024//microbatch):
            batch=default_collate([dataset[np.random.randint(len(dataset))] for _ in range(microbatch)])
            batch={k:v.to(device) for k,v in batch.items()}
            batch=augmentation(batch)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=bf16):
                loss=model([(['hardware_gen'],batch)],is_eval=False)['loss']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native TT loss')
            (loss/(1024//microbatch)).backward();total+=float(loss.detach())/(1024//microbatch)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();scheduler.step();ema.step(model.parameters())
        rec=dict(step=step,loss=total,seconds=time.time()-started,
                 estimated_remaining_seconds=(time.time()-started)/(step-first)*(steps-step),
                 gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
        if step==1 or step%100==0 or step==steps:
            rec['val_loss']=validate()
        with (out/'progress.jsonl').open('a') as log:log.write(json.dumps(rec)+'\n')
        print(rec,flush=True)
        if step%10==0 or step==steps or step==stop_after:save(step)
    if stop_after is not None and stop_after<steps:return checkpoint_path
    ema.store(model.parameters());ema.copy_to(model.parameters())
    ema_val=validate();ema.restore(model.parameters())
    atomic_torch(out/'checkpoint.pt',dict(model=model.state_dict(),ema=ema.state_dict(),
        normalization=norms,step=steps,contract=contract,ema_val_loss=ema_val))
    return out/'checkpoint.pt'
