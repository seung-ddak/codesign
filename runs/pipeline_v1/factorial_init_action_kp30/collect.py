"""Actual MJX GPU full-factorial replay: initialization x action origin, common controls across E."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def initial_batches(qref,vref,qrest,vrest):
    """Axes: initialization(reference/rest), action(reference/ppo), group, coordinate."""
    q=np.stack([qref,np.broadcast_to(qrest,qref.shape)])
    v=np.stack([vref,np.broadcast_to(vrest,vref.shape)])
    return np.repeat(q[:,None],2,axis=1),np.repeat(v[:,None],2,axis=1)


def main():
    sys.path[:0]=[str(ROOT/'code/motion_tracking'),str(ROOT.parent/'transformer-transformer')]
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx
    from omegaconf import OmegaConf
    from pd_replay import build_model
    from collect_pd_rollouts import save_rollout
    from track_env import TrackMjxEnv
    from t2.mjx_env.base_env import make_data,step
    assert jax.default_backend()=='gpu',jax.devices()
    c=json.loads((OUT/'conditions.json').read_text())
    fingerprint=hashlib.sha256((OUT/'conditions.json').read_bytes()).hexdigest()
    with (ROOT/c['config']).open('rb') as f:cfg=pickle.load(f)
    ecfg=OmegaConf.to_container(cfg.eval_env,resolve=True);ecfg.pop('_target_')
    with np.load(OUT/'action_bank.npz') as z:bank={k:z[k].copy() for k in z.files}
    tq=np.broadcast_to(bank['target_q'][None],(2,*bank['target_q'].shape)).reshape(-1,c['steps'],12).copy()
    G=c['groups']; B=4*G; T=c['steps']
    ecfg.update(impl='jax',clips=bank['motions'].tolist(),clip_boost=None,switch_prob=0.,reset_qpos=0.,reset_qvel=0.)
    rest_source=ROOT/'code/tt_dynamics/collect_e_swap.py'
    node=next(n for n in ast.parse(rest_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='rest_state')
    ns=dict(np=np,mujoco=mujoco,args=SimpleNamespace(settle=c['home_settle_s']))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(rest_source),'exec'),ns)
    forward=jax.jit(jax.vmap(mjx.forward,in_axes=(None,0)))
    substeps=round(ecfg['ctrl_dt']/ecfg['sim_dt'])
    vstep=jax.jit(lambda model,data,ctrl:jax.vmap(lambda d,u:step(model,d,u,substeps))(data,ctrl))
    audit={}
    for name,e in zip(bank['designs'],bank['E']):
        name=str(name);path=OUT/'rollouts'/name/'factorial.npz'
        if path.exists():
            with np.load(path) as z:
                assert str(z['fingerprint'])==fingerprint,'Existing rollout conditions mismatch'
                assert z['qpos'].shape==(2,2,G,T+1,19) and np.isfinite(z['qpos']).all()
            print(name,'verified existing rollout',flush=True);continue
        ecfg['reference_dir']=str(ROOT/'runs/pipeline_v1/retargeting'/name/c['choice'])
        env=TrackMjxEnv(mj_model_fn=lambda:build_model(c['choice'],kp=c['kp'],tau_scale=cfg.get('tau_scale'),torque_limit=cfg.get('torque_limit'),uniforms=e),**ecfg)
        assert env.n_substeps==substeps
        frame=np.asarray(env.clip_start)[bank['clip']]+bank['start_frame']
        assert (frame+T<np.asarray(env.clip_end)[bank['clip']]).all()
        qref=np.asarray(env.ref_qpos)[frame];vref=np.asarray(env.ref_qvel)[frame]
        qr,vr=ns['rest_state'](env)
        q0,v0=initial_batches(qref,vref,qr,vr)
        assert np.array_equal(q0[:,0],q0[:,1]) and np.array_equal(v0[:,0],v0[:,1])
        off=np.asarray(env.action_offset)
        lo,hi=env.clip_actions
        assert (tq>=off[:12]+lo*env.action_scale-1e-6).all() and (tq<=off[:12]+hi*env.action_scale+1e-6).all(),'Controls would be shape-dependently clipped'
        d0=make_data(env.mj_model,impl='jax',nconmax=env._nconmax,njmax=env._njmax)
        data=jax.tree.map(lambda x:jnp.broadcast_to(x,(B,*x.shape)),d0)
        data=data.replace(qpos=jnp.asarray(q0.reshape(B,19)),qvel=jnp.asarray(v0.reshape(B,18)))
        data=forward(env.mjx_model,data)
        feet=np.asarray(env._foot_ids);floor=env.mj_model.geom('floor').id
        qseq=[];vseq=[];contacts=[];valid=[]
        alive=np.ones(B,bool)
        for t in range(T+1):
            q=np.asarray(data.qpos);v=np.asarray(data.qvel)
            assert np.isfinite(q).all() and np.isfinite(v).all(),f'{name} nonfinite at {t}'
            geom=np.asarray(data._impl.contact.geom);dist=np.asarray(data._impl.contact.dist)
            contact=np.stack([(((((geom[...,0]==f)&(geom[...,1]==floor))|((geom[...,1]==f)&(geom[...,0]==floor)))&(dist<=1e-3)).any(-1)) for f in feet],-1)
            alive &= 1-2*(q[:,4]**2+q[:,5]**2)>=env.termination_fellover_threshold
            qseq.append(q);vseq.append(v);contacts.append(contact)
            if t:valid.append(alive.copy())
            if t<T:
                ctrl=np.broadcast_to(off,(B,len(off))).copy();ctrl[:,:12]=tq[:,t]
                data=vstep(env.mjx_model,data,jnp.asarray(ctrl))
        out={'qpos':np.stack(qseq,1).reshape(2,2,G,T+1,19),'qvel':np.stack(vseq,1).reshape(2,2,G,T+1,18),
             'contact':np.stack(contacts,1).reshape(2,2,G,T+1,4),'valid':np.stack(valid,1).reshape(2,2,G,T),'fell':(~alive).reshape(2,2,G),
             'target_q':tq.reshape(2,2,G,T,12),'E':e,'design':name,'fingerprint':fingerprint,'dt':np.float64(env.dt),
             'group_id':np.arange(G),'clip':bank['clip'],'start_frame':bank['start_frame'],'motions':bank['motions'],
             'initial_qpos':q0,'initial_qvel':v0,'rest_contact_initial':np.stack(contacts,1).reshape(2,2,G,T+1,4)[1,0,0,0],
             'kp':np.float32(c['kp']),'backend':'mjx_gpu'}
        assert np.array_equal(out['qpos'][:,0,:,0],out['qpos'][:,1,:,0])
        save_rollout(path,out)
        audit[name]={'fell':out['fell'].sum(-1).tolist(),'rest_contacts':out['rest_contact_initial'].tolist()}
        print(name,'fell per arm',out['fell'].sum(-1).tolist(),'saved',flush=True)
        del env,data,d0,out
        gc.collect()
    (OUT/'collection_audit.json').write_text(json.dumps({'backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()],'new_rollouts':audit},indent=2))


if __name__=='__main__':main()
