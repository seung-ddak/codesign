"""Evaluate a CALM tracking checkpoint with the saved native MJX environment."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'transformer-transformer'))
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from policy_io import load_run
from pd_replay import build_model,LEGS
from track_env import TrackMjxEnv
from eval_calm_pd_tracking import foot_contacts,tracking_errors


def set_sliding_friction(model,value):
    if not np.isfinite(value) or value<=0:raise ValueError('Friction must be finite and positive')
    for i in range(model.ngeom):
        if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE or model.geom(i).name in [x+'_foot' for x in LEGS]:
            model.geom_friction[i,0]=value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--sliding-friction',type=float,help='Override only foot and plane sliding friction')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.sliding_friction is not None and (not np.isfinite(args.sliding_friction) or args.sliding_friction<=0):parser.error('Use finite positive friction')
    if (out/'metrics.json').exists() or list(out.glob('*.npz')):raise ValueError('Existing evaluation results')
    cfg,policy=load_run(args.checkpoint);ecfg=OmegaConf.to_container(cfg.eval_env,resolve=True);ecfg.pop('_target_')
    def model_fn():
        model=build_model('00000011',uniforms=[.5]*13,kp=cfg.kp,torque_limit=cfg.torque_limit)
        if args.sliding_friction is not None:set_sliding_friction(model,args.sliding_friction)
        return model
    env=TrackMjxEnv(mj_model_fn=model_fn,**ecfg)
    assert np.isclose(env.dt,1/30) and env.n_substeps==10 and not cfg.multi_design
    model=env.mj_model;planes=set(np.flatnonzero(model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE).tolist());feet=np.array([model.geom(x+'_foot').id for x in LEGS])
    names=list(ecfg['clips']);assert len(names)==5
    refs=[];vel=[];hashes=[]
    for name in names:
        path=Path(ecfg['reference_dir'])/(name+'.npz')
        with np.load(path) as d:ref=np.c_[d['fix_base_pos'],d['fix_base_quat'],d['fix_q']]
        assert ref.shape==(301,19)
        qv=np.zeros(model.nv);mujoco.mj_differentiatePos(model,qv,1/30,ref[0],ref[1]);vel.append(qv);refs.append(ref)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    keys=jnp.repeat(jax.random.PRNGKey(0)[None],5,axis=0)
    states=jax.jit(jax.vmap(env.reset_with_traj))(keys,jnp.arange(5,dtype=jnp.int32),env.clip_start)
    # Match the previous CPU evaluation's initial finite-difference velocity and control.
    def initialize(state,qvel,ctrl):
        data=mjx.forward(env.mjx_model,state.data.replace(qvel=qvel,ctrl=ctrl))
        return state.replace(data=data)
    states=jax.jit(jax.vmap(initialize))(states,jnp.asarray(np.array(vel)),jnp.asarray(np.array(refs)[:,0,7:]))
    def advance(state,key):
        info=dict(state.info);info['rng']=key
        action=jnp.clip(policy(env._get_obs(state.data,info),key)[0],*env.clip_actions)
        state=env.step(state,action)
        # Refresh measurement fields without feeding the refresh into the integration state.
        return state,mjx.forward(env.mjx_model,state.data)
    advance=jax.jit(jax.vmap(advance,in_axes=(0,None)))
    logs=[{k:[] for k in ['time','qpos','qvel','foot_pos','reference_foot_pos','contact','slip_mps','up_z']} for _ in names]
    max_clock_error=0.
    ghost=mujoco.MjData(model);active=np.ones(5,dtype=bool);fell=np.zeros(5,dtype=bool);rng=jax.random.PRNGKey(0)
    conditions=dict(training_backend='MJX GPU',evaluation_backend='MJX GPU',checkpoint=str(args.checkpoint.resolve()),reference_hz=30,physics_hz=300,
                    solver_iterations=ecfg['solver_iterations'],solver_ls_iterations=ecfg['solver_ls_iterations'],
                    seed=0,initialization='first reference pose; same raw finite-difference qvel and initial ctrl as prior CPU comparison',
                    observation='native environment observation; CPU comparison key sequence and previous action; saved observation noise retained',
                    measurement='native MJX forward contact forces; refreshed measurement state not fed back into integration',
                    termination='body up-axis Z < 0.5; no autoreset; metrics truncated at first fall',
                    observation_noise=ecfg['obs_noise'],jax_devices=[str(x) for x in jax.devices()],
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    conditions['sliding_friction_override']=args.sliding_friction
    conditions['foot_and_floor_friction']={model.geom(i).name:model.geom_friction[i].tolist() for i in range(model.ngeom) if i in planes or i in feet}
    (out/'conditions.json').write_text(json.dumps(conditions,ensure_ascii=False,indent=2))
    for k in range(300):
        rng,key=jax.random.split(rng);states,snapshots=advance(states,key)
        if np.asarray(states.info['state_is_nan']).any():raise ValueError('Nonfinite simulation state')
        snapshots=mjx.get_data(model,snapshots);assert len(snapshots)==5
        for ci,d in enumerate(snapshots):
            if not active[ci]:continue
            assert np.isfinite(d.qpos).all()
            max_clock_error=max(max_clock_error,abs(d.time-(k+1)/30))
            assert np.isclose(d.time,(k+1)/30,atol=1e-3)
            ghost.qpos[:]=refs[ci][k+1];mujoco.mj_forward(model,ghost)
            contact,slip=foot_contacts(model,d,feet,planes);up=1-2*(d.qpos[4]**2+d.qpos[5]**2)
            values=[(k+1)/30,d.qpos.copy(),d.qvel.copy(),d.geom_xpos[feet].copy(),ghost.geom_xpos[feet].copy(),contact,slip,up]
            for label,value in zip(logs[ci],values):logs[ci][label].append(value)
            if up<.5:fell[ci]=True;active[ci]=False
        if (k+1)%60==0:print('Evaluated control frames:',k+1,flush=True)
        if not active.any():break
    conditions['max_float32_clock_error_s']=max_clock_error
    (out/'conditions.json').write_text(json.dumps(conditions,ensure_ascii=False,indent=2))
    rows=[]
    for ci,name in enumerate(names):
        d={k:np.asarray(v) for k,v in logs[ci].items()};n=len(d['time']);ref=refs[ci][1:n+1];q=d['qpos']
        np.savez_compressed(out/(name+'.npz'),**d,reference_qpos=ref)
        rot=Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix();rr=Rotation.from_quat(ref[:,[4,5,6,3]]).as_matrix()
        f=np.einsum('tfi,tij->tfj',d['foot_pos']-q[:,None,:3],rot);rf=np.einsum('tfi,tij->tfj',d['reference_foot_pos']-ref[:,None,:3],rr)
        slip=d['slip_mps'][np.isfinite(d['slip_mps'])]
        row=dict(motion=name,survived_10s=bool(not fell[ci] and n==300),fell=bool(fell[ci]),evaluated_seconds=n/30,
                 reference_sha256=hashes[ci],foot_body_mean_error_m=float(np.linalg.norm(f-rf,axis=-1).mean()),
                 slip_mean_mps=float(slip.mean()) if len(slip) else None,slip_p95_mps=float(np.percentile(slip,95)) if len(slip) else None,
                 supported_foot_samples=len(slip),**tracking_errors(q,ref,d['foot_pos'],d['reference_foot_pos']))
        row['tracking_error_screen_pass']=bool(row['survived_10s'] and row['joint_rmse_deg']<5 and row['foot_body_mean_error_m']<.03 and row['orientation_mean_deg']<5)
        rows.append(row)
    (out/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
