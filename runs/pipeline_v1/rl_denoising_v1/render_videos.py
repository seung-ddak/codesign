"""Render trained diffusion-PPO physical rollouts with task track overlays."""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]/'code/tt_dynamics'))
from train_task_rl import build_task_model
from rollout_data import yaw_of
from task_reward import TrackReward, ZigzagReward


def track_segments(reward, q0, length=2.):
    yaw = yaw_of(q0[3:7])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c,-s],[s,c]])
    lines = reward.guide_lines(length=length) if isinstance(reward,ZigzagReward) else reward.guide_lines()
    segments=[]
    for points,center in lines:
        xy = points @ rotation.T + q0[:2]
        xyz = np.c_[xy,np.full(len(xy),.012)]
        color = np.array([1.,.82,.05,1.] if center else [.95,.95,.95,1.],dtype=np.float32)
        segments.extend((a,b,color) for a,b in zip(xyz[:-1],xyz[1:]))
    return segments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',type=Path,default=ROOT)
    args=parser.parse_args()
    root=args.experiment_root.resolve()
    videos=root/'videos'
    videos.mkdir(exist_ok=True)
    option=mujoco.MjvOption()
    option.geomgroup[:]=1
    for task in ('zigzag','circle'):
        run=root/f'validated_{task}_seed0'
        if not run.exists():
            run=root/f'{task}_seed0'
        protocol=json.loads((run/'protocol.json').read_text())
        summary=json.loads((run/'summary.json').read_text())['rl_after']['rows']
        reward=ZigzagReward() if task=='zigzag' else TrackReward()
        with np.load(run/'rl_after.npz') as archive:
            states=archive['pred'].shape[1]
            for candidate,row in enumerate(summary[:2]):
                model=build_task_model(archive['E'][candidate])
                model.vis.global_.offwidth,model.vis.global_.offheight=960,544
                qs=archive[f'qpos_{candidate}']
                assert qs.shape[1]==model.nq and np.isfinite(qs).all()
                length=max(2.,float(qs[:,0].max()-qs[0,0])+.3)
                segments=track_segments(reward,qs[0],length)
                camera=mujoco.MjvCamera()
                camera.lookat[:]=[length/2,0.,.1] if task=='zigzag' else [0.,1.,.1]
                camera.distance,camera.azimuth,camera.elevation=(max(3.2,length*1.6),90.,-55.) if task=='zigzag' else (4.2,90.,-60.)
                data=mujoco.MjData(model)
                path=videos/f'{task}_candidate{candidate}_rl_track.mp4'
                with mujoco.Renderer(model,height=544,width=960) as renderer:
                    with imageio.get_writer(path,fps=25,codec='libx264',quality=8) as writer:
                        for t in range(0,states,2):
                            data.qpos[:]=qs[min(t,len(qs)-1)]
                            mujoco.mj_forward(model,data)
                            renderer.update_scene(data,camera=camera,scene_option=option)
                            assert renderer.scene.ngeom+len(segments)<=renderer.scene.maxgeom
                            for a,b,color in segments:
                                geom=renderer.scene.geoms[renderer.scene.ngeom]
                                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),color)
                                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.006,a,b)
                                renderer.scene.ngeom+=1
                            frame=Image.fromarray(renderer.render().copy())
                            draw=ImageDraw.Draw(frame)
                            draw.rectangle((0,0,960,56),fill=(20,25,35))
                            draw.text((12,8),f'{task.upper()} | diffusion PPO after {protocol["updates"]} updates | candidate {candidate} | t={t*.02:.2f}s',fill='white')
                            unit='m' if task=='zigzag' else 'laps'
                            draw.text((12,29),f"Final progress {row['progress']:.3f} {unit} | yellow: track center | white: boundaries",fill='white')
                            if t>=len(qs):
                                draw.text((12,68),'FALL: last recorded frame held',fill=(220,35,35))
                            writer.append_data(np.asarray(frame))
                reader=imageio.get_reader(path)
                assert reader.count_frames()==len(range(0,states,2)) and reader.get_data(0).shape[:2]==(544,960)
                if candidate==0:
                    imageio.imwrite(videos/f'{task}_rl_track_preview.png',reader.get_data(0))
                reader.close()
                print(path,flush=True)


if __name__=='__main__':
    main()
