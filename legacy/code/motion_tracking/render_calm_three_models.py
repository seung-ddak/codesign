"""Replay source A1 visual meshes and two TT records in one shared scene."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from pd_replay import build_model,LEGS


def dae_mesh(path):
    # shortcut: these CALM assets contain identity-transformed triangle meshes; use a full Collada loader for other assets.
    root=ET.parse(path).getroot();ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
    assert root.find('c:asset/c:up_axis',ns).text=='Z_UP'
    assert float(root.find('c:asset/c:unit',ns).get('meter'))==1
    for node in root.findall('.//c:node',ns):
        if node.find('c:instance_geometry',ns) is not None:
            matrix=node.find('c:matrix',ns)
            assert matrix is not None and np.allclose(np.fromstring(matrix.text,sep=' ').reshape(4,4),np.eye(4))
    meshes=root.findall('c:library_geometries/c:geometry/c:mesh',ns);assert len(meshes)==1
    mesh=meshes[0];vertices=mesh.find('c:vertices',ns)
    source=vertices.find("c:input[@semantic='POSITION']",ns).get('source')[1:]
    src=mesh.find(f"c:source[@id='{source}']",ns)
    assert int(src.find('c:technique_common/c:accessor',ns).get('stride'))==3
    points=np.fromstring(src.find('c:float_array',ns).text,sep=' ').reshape(-1,3)
    faces=[]
    for tris in mesh.findall('c:triangles',ns):
        inputs=tris.findall('c:input',ns);stride=max(int(i.get('offset')) for i in inputs)+1
        v=next(i for i in inputs if i.get('semantic')=='VERTEX');assert v.get('source')=='#'+vertices.get('id')
        indices=np.fromstring(tris.find('c:p',ns).text,sep=' ',dtype=int).reshape(-1,stride)
        triangle=indices[:,int(v.get('offset'))].reshape(-1,3)
        assert len(triangle)==int(tris.get('count'));faces.append(triangle)
    assert faces and mesh.find('c:polylist',ns) is None
    faces=np.concatenate(faces);assert faces.min()>=0 and faces.max()<len(points) and np.isfinite(points).all()
    return points,faces


def source_spec(root):
    spec=mujoco.MjSpec.from_file(str(root/'runs/calm_pilot_retarget/try3/source_model/model.xml'))
    for g in list(spec.geoms):
        if g.type==mujoco.mjtGeom.mjGEOM_PLANE:spec.delete(g)
        else:g.group=5
    urdf=root.parent/'external/a1-calm-main/assets/urdf/a1_description/urdf/a1.urdf'
    robot=ET.parse(urdf).getroot();loaded=set()
    for link in robot.findall('link'):
        for visual in link.findall('visual'):
            origin=visual.find('origin');pos=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            quat=Rotation.from_euler('xyz',rpy).as_quat()[[3,0,1,2]]
            geom=visual.find('geometry');mesh=geom.find('mesh')
            kw=dict(name="visual_"+link.get("name"),pos=pos,quat=quat,contype=0,conaffinity=0,group=2,rgba=[.95,.48,.12,1],mass=0)
            if mesh is not None:
                path=(urdf.parent/mesh.get('filename')).resolve();name='visual_'+path.stem
                if name not in loaded:
                    vertices,faces=dae_mesh(path)
                    spec.add_mesh(name=name,uservert=vertices.ravel(),userface=faces.ravel(),scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' '));loaded.add(name)
                spec.body(link.get('name')).add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,**kw)
            elif geom.find('sphere') is not None:
                spec.body(link.get('name')).add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[float(geom.find('sphere').get('radius')),0,0],**kw)
            elif geom.find('box') is not None:
                spec.body(link.get('name')).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=np.fromstring(geom.find('box').get('size'),sep=' ')/2,**kw)
            else:raise ValueError('Unrecognized original visual geometry')
    assert len(loaded)==5
    return spec


def shared_scene(root,uniforms=None):
    captured=[]
    original=build_model('00000011',uniforms=[.5]*13 if uniforms is None else uniforms,torque_limit=33.5,scene_fn=lambda spec:captured.append(spec.copy()))
    scene=captured[0];child=scene.copy()
    for key in list(child.keys):child.delete(key)
    for geom in list(child.geoms):
        if geom.type==mujoco.mjtGeom.mjGEOM_PLANE:child.delete(geom)
    a1=source_spec(root);a1.option.timestep=scene.option.timestep
    scene.attach(a1,prefix='a1/',frame=scene.worldbody.add_frame())
    scene.attach(child,prefix='target/',frame=scene.worldbody.add_frame())
    for geom in scene.geoms:
        geom.contype=0;geom.conaffinity=0
        if geom.type!=mujoco.mjtGeom.mjGEOM_PLANE:
            geom.rgba=[.95,.48,.12,1] if geom.name.startswith('a1/') else ([.65,.35,.87,1] if geom.name.startswith('target/') else [.2,.5,.95,1])
            geom.material=''
    model=scene.compile();assert model.nq==57 and model.nmesh>=5
    starts=[int(model.joint('a1/root').qposadr[0]),int(model.joint('target/'+original.joint(0).name).qposadr[0]),0]
    assert len(set(starts))==3
    assert sum(model.geom(i).name.startswith('a1/visual_') and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH for i in range(model.ngeom))==13
    np.testing.assert_allclose(model.geom_rgba[model.geom('a1/visual_trunk').id],[.95,.48,.12,1],atol=1e-6)
    source_order=[int(model.joint(f'a1/{leg}_{j}_joint').qposadr[0]) for leg in LEGS for j in ['hip','thigh','calf']]
    assert source_order==list(range(starts[0]+7,starts[0]+19))
    return model,starts


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();root=Path(__file__).resolve().parents[2]
    conditions=json.loads((run/'conditions.json').read_text())
    refs=Path(conditions.get('reference_directory',root/'runs/calm_pilot_retarget/try3/references/e000/00000011'))
    model,starts=shared_scene(root,conditions.get('E'));data=mujoco.MjData(model)
    model.vis.global_.offwidth,model.vis.global_.offheight=1280,720
    model.vis.headlight.ambient[:]=[.5,.5,.5];model.vis.headlight.diffuse[:]=[.7,.7,.7]
    renderer=mujoco.Renderer(model,height=720,width=1280)
    options=mujoco.MjvOption();options.geomgroup[3]=1;options.geomgroup[5]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.azimuth=125;camera.elevation=-30
    offsets=np.array([[-.95,-.65,0],[0,0,0],[.95,.65,0]])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    rows=json.loads((run/'metrics.json').read_text());dest=run/'shared_scene_videos';dest.mkdir(exist_ok=True)
    for row in rows:
        name=row['motion']
        with np.load(run/(name+'.npz')) as log,np.load(refs/(name+'.npz')) as ref:
            times=log['time'];indices=np.rint(times*30).astype(int);assert np.allclose(indices/30,times,atol=1e-8)
            states=[ref['a1_qpos'][indices],log['reference_qpos'],log['qpos']]
            assert all(q.shape==(len(times),19) and np.isfinite(q).all() for q in states)
            np.testing.assert_allclose(states[1],np.c_[ref['fix_base_pos'],ref['fix_base_quat'],ref['fix_q']][indices],atol=1e-8)
            path=dest/(name+'.mp4')
            with imageio.get_writer(path,fps=30,codec='libx264',quality=7,macro_block_size=1) as writer:
                for i,time in enumerate(times):
                    centers=[]
                    for start,states_i,offset in zip(starts,states,offsets):
                        q=states_i[i].copy();q[:3]+=offset;data.qpos[start:start+19]=q;centers.append(q[:3])
                    mujoco.mj_forward(model,data)
                    centers=np.asarray(centers);camera.lookat[:]=centers.mean(0);camera.distance=max(3.2,1.2+np.linalg.norm(np.ptp(centers,axis=0))*.95)
                    renderer.update_scene(data,camera=camera,scene_option=options)
                    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                    draw.rectangle((0,0,1280,43),fill=(22,30,40))
                    for x,label,color in [(18,'A1 SOURCE · full visual',(.95,.48,.12)),(445,'TT TARGET',(.65,.35,.87)),(865,'TT ACTUAL',(.2,.5,.95))]:draw.text((x,8),label,font=font,fill=tuple(int(v*255) for v in color))
                    draw.rectangle((0,679,1280,720),fill=(22,30,40));draw.text((18,687),f't={time:.2f}s  |  shared floor / shared camera  |  constant display spacing only',font=font,fill='white')
                    writer.append_data(np.asarray(frame))
                    if i==min(150,len(times)-1):imageio.imwrite(dest/(name+'_preview.png'),np.asarray(frame))
            row['shared_scene_video_absolute_path']=str(path);print(name,len(times),'frames',flush=True)
    renderer.close()
    (run/'shared_scene_videos.json').write_text(json.dumps(dict(display_offsets_m=offsets.tolist(),source_visual='original CALM A1 DAE meshes and URDF visual origins',replay_only=True,rows=rows),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
