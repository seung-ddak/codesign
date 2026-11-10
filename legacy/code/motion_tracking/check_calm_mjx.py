"""Replay recorded CALM PD targets in MJX without resetting the trajectory."""
import argparse
import base64
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def build_model(source, output):
    import mujoco
    root=ET.parse(source).getroot()
    for link in root.findall('link'):
        for visual in link.findall('visual'):link.remove(visual)
    for joint in root.findall('joint'):
        for dynamics in joint.findall('dynamics'):joint.remove(dynamics)
    ET.SubElement(ET.SubElement(root,'mujoco'),'compiler',fusestatic='false',discardvisual='true',balanceinertia='true')
    urdf=output/'collision_only.urdf';ET.ElementTree(root).write(urdf)
    spec=mujoco.MjSpec.from_file(str(urdf));spec.body('base').add_freejoint(name='root')
    spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[20,20,.1],contype=2,conaffinity=1)
    for geom in spec.geoms:
        if geom.name!='floor':geom.contype=1;geom.conaffinity=2
        geom.friction=[1,.005,.0001]
    spec.option.timestep=1/120;spec.option.gravity=[0,0,-9.81]
    model=spec.compile()
    # Match the mass inferred by Isaac Gym for collision-only shoulder links.
    for leg in ['FL','FR','RL','RR']:
        b=model.body(leg+'_thigh_shoulder').id
        model.body_inertia[b]*=.166226/model.body_mass[b];model.body_mass[b]=.166226
    spec.compiler.inertiafromgeom=mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO
    spec.visual.global_.offwidth=960;spec.visual.global_.offheight=544
    for body in spec.bodies:
        if body.name.endswith('_thigh_shoulder'):
            b=model.body(body.name).id;body.mass=model.body_mass[b];body.inertia=model.body_inertia[b];body.iquat=model.body_iquat[b];body.ipos=model.body_ipos[b];body.explicitinertial=True
    model=spec.compile()
    (output/'model.xml').write_text(spec.to_xml())
    return model


def initial_state(trace, model):
    import mujoco
    root=trace['root'][0];qpos=np.zeros(model.nq);qvel=np.zeros(model.nv)
    qpos[:3]=root[:3];qpos[3:7]=root[[6,3,4,5]];qvel[:3]=root[7:10]
    from scipy.spatial.transform import Rotation
    qvel[3:6]=Rotation.from_quat(root[3:7]).inv().apply(root[10:13])
    addresses=[];velocities=[]
    for i,name in enumerate(trace['joint_names']):
        joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,str(name));assert joint>=0
        a=model.jnt_qposadr[joint];b=model.jnt_dofadr[joint];addresses.append(a);velocities.append(b)
        qpos[a]=trace['q'][0,i];qvel[b]=trace['qvel'][0,i]
    assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
    return qpos,qvel,np.array(addresses),np.array(velocities)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    import mujoco,jax,jax.numpy as jnp
    from mujoco import mjx
    model=build_model(Path(__file__).resolve().parents[3]/'external/a1-calm-main/assets/urdf/a1_description/urdf/a1.urdf',args.out)
    mm=mjx.put_model(model);cases=sorted(args.source.glob('hlc_random_*/trace.npz'));assert cases
    traces=[np.load(p) for p in cases];init=[initial_state(t,model) for t in traces];qa,va=init[0][2:]
    def replay(qpos,qvel,targets):
        data=mjx.make_data(model).replace(qpos=qpos,qvel=qvel)
        def frame(data,target):
            def substep(_,d):
                force=jnp.zeros(model.nv).at[va].set(jnp.clip(40*(target-d.qpos[qa])-1.5*d.qvel[va],-15,15))
                return mjx.step(mm,d.replace(qfrc_applied=force))
            data=jax.lax.fori_loop(0,4,substep,data)
            return data,(data.qpos,data.qvel)
        return jax.lax.scan(frame,data,targets)[1]
    print('Compiling MJX',jax.devices(),flush=True)
    qp,qv=jax.jit(jax.vmap(replay))(jnp.array([i[0] for i in init]),jnp.array([i[1] for i in init]),jnp.array([t['target_q'] for t in traces]));qp=np.asarray(qp);qv=np.asarray(qv)
    results=[]
    import imageio.v2 as imageio
    renderer=mujoco.Renderer(model,height=544,width=960);data=mujoco.MjData(model)
    cards=[]
    for i,(path,t) in enumerate(zip(cases,traces)):
        q=np.vstack([init[i][0],qp[i]]);ref=t['root'];finite=np.isfinite(q).all(axis=1);up=1-2*(q[:,4]**2+q[:,5]**2);fall=(up<0)|(q[:,2]<.06)|~finite;ids=np.flatnonzero(fall)
        err=np.linalg.norm(q[:,:2]-ref[:,:2],axis=1)
        r=dict(case=path.parent.name,first_fall_seconds=float(ids[0]/30) if len(ids) else None,xy_rmse_m=float(np.sqrt(np.mean(err**2))),final_xy_error_m=float(err[-1]),final_height_m=float(q[-1,2]),finite=bool(finite.all()))
        results.append(r);np.savez_compressed(args.out/(path.parent.name+'.npz'),qpos=q,qvel=qv[i],source_root=ref)
        video=args.out/(path.parent.name+'.mp4')
        with imageio.get_writer(video,fps=30) as writer:
            for row in q[1:]:
                if not np.isfinite(row).all():break
                data.qpos[:]=row;mujoco.mj_forward(model,data);camera=mujoco.MjvCamera();camera.lookat[:]=row[:3];camera.distance=1.6;camera.azimuth=45;camera.elevation=-25
                renderer.update_scene(data,camera);writer.append_data(renderer.render())
        cards.append('<h2>'+path.parent.name+'</h2><video controls muted playsinline style="width:100%" src="data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()+'"></video><pre>'+json.dumps(r,indent=2)+'</pre>')
    renderer.close()
    summary=dict(fall_definition='body up dot world up < 0, root height < 0.06 m, or nonfinite',replay='open loop saved PD targets; policy not reevaluated',backend=str(jax.devices()),model_mass_kg=float(model.body_mass.sum()),dt=1/120,control_dt=1/30,kp=40,kd=1.5,torque_limit=15,self_collision=False,results=results)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    falls=[r['first_fall_seconds'] for r in results if r['first_fall_seconds'] is not None]
    conclusion=f'<p>{len(results)}개 중 {len(falls)}개에서 전도 조건이 발생했습니다. 기준은 몸통 위쪽 축의 수직 성분이 음수이거나 루트 높이가 0.06 m 미만인 경우입니다. 이 결과는 저장된 행동의 직접 재현 실패이며, 정책을 MJX에서 다시 실행했을 때의 성능은 미평가입니다.</p>'
    (args.out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><body style="max-width:1000px;margin:auto"><h1>CALM 연속 이동 MJX 재생</h1><p>같은 초기 상태와 저장된 관절 목표각을 사용했습니다. 로봇 자기 충돌은 끄고 바닥 충돌은 유지했습니다. HLC·LLC를 다시 실행하지 않은 개방 루프 재생입니다. 충돌 형상만 표시합니다. 접촉 해법과 마찰·관절 구동 구현은 PhysX와 완전히 같지 않습니다.</p>'+conclusion+''.join(cards)+'</body></html>')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
