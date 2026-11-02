"""Evaluate only the trusted rest_s0_of function, without generate_task's training/rendering side effects."""
import ast
import hashlib
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'code/tt_dynamics'), str(ROOT/'code/motion_tracking'), str(ROOT.parent/'transformer-transformer')]
from pd_replay import build_model
from track_env import TrackMjxEnv
from rollout_data import features

out = Path(__file__).resolve().parent
config = sorted((ROOT/'runs/pipeline_v1/experts_wide16/walkForwardSlow__r00').glob('wandb/offline-run-*/files/config.pkl'))[-1]
with config.open('rb') as f:
    cfg = pickle.load(f)
choice = cfg.get('choice')
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop('_target_')
ecfg.update(impl='jax', reference_dir=str(ROOT/'runs/pipeline_v1/retargeting/e000'/choice), clips=['walkForwardSlow__r00'])
source = ROOT/'code/tt_dynamics/generate_task.py'
tree = ast.parse(source.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='rest_s0_of')
ns = dict(np=np, torch=torch, mujoco=mujoco, TrackMjxEnv=TrackMjxEnv, build_model=build_model,
          features=features, choice=choice, KP=200, cfg=cfg, ecfg=ecfg,
          args=SimpleNamespace(task='straight'), dev='cpu')
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
paths = sorted((ROOT/'runs/pipeline_v1/pd_rollouts_kp200').glob('*/primitive_seed0.npz'))
es=[]; gs=[]
for p in paths:
    with np.load(p) as z: e=z['E'].astype(np.float32)
    assert e.shape==(13,)
    g=ns['rest_s0_of'](torch.tensor(e[None])).numpy()[0]
    assert g.shape==(42,) and np.isfinite(g).all()
    es.append(e); gs.append(g)
    print(p.parent.name, 'height',round(float(g[2]),4),'contact',g[-4:].tolist(),flush=True)
np.savez_compressed(out/'rest_features.npz',designs=np.array([p.parent.name for p in paths]),E=np.stack(es),rest_s0=np.stack(gs))
(out/'rest_provenance.json').write_text(json.dumps({'backend':'MuJoCo CPU, existing generate_task.rest_s0_of exact AST function', 'kp':200,'settle_s':1,'config':str(config.relative_to(ROOT)), 'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,config,Path(__file__))}},indent=2))
