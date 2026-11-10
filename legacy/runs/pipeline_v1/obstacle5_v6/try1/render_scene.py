"""Photograph the actual compiled obstacle model; colored floor marks are visual only."""
from pathlib import Path
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[2] / 'code/tt_dynamics'))
from s0_geometry import s0_to_q
model = mujoco.MjModel.from_binary_path(str(root / 'designs/scale_0_cand0.mjb'))
with np.load(root / 'scale_0.npz') as z:
    q, v = s0_to_q(z['pred'][0, 0, :42])
data = mujoco.MjData(model)
data.qpos[:], data.qvel[:] = q, v
mujoco.mj_forward(model, data)
poles = [i for i in range(model.ngeom) if model.geom(i).name.startswith('obstacle_pole_')]
assert len(poles) == 50 and np.allclose(model.geom_pos[poles, 0], np.repeat(np.arange(1., 6.), 10))
model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
camera = mujoco.MjvCamera()
camera.lookat[:] = [3., 0., .18]
camera.distance, camera.azimuth, camera.elevation = 6.8, 90., -30.
option = mujoco.MjvOption(); option.geomgroup[3] = 1
with mujoco.Renderer(model, 720, 1280) as renderer:
    renderer.update_scene(data, camera=camera, scene_option=option)
    for center, radius, color in [((0., 0.), .15, [0.15, .5, 1., 1.]), ((6., 0.), .2, [.1, .9, .35, 1.])]:
        for angle in np.linspace(0, 2*np.pi, 60, endpoint=False):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.012, 0., 0.]),
                               np.array([center[0] + radius*np.cos(angle), center[1] + radius*np.sin(angle), .015]),
                               np.eye(3).ravel(), np.array(color))
            renderer.scene.ngeom += 1
    frame = renderer.render().copy()
assert frame.shape == (720, 1280, 3)
image = Image.fromarray(frame)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
draw.rectangle((12, 12, 1268, 62), fill=(15, 24, 32))
draw.text((30, 23), 'BLUE: start (0,0)    RED: 5 poles, x=1..5m, spacing 1m    GREEN: goal (6,0), radius 0.2m', font=font, fill='white')
image.save(root / 'environment.png')
print(root / 'environment.png')
