"""L0~L3 best design 비교 figure (측면 / 정면 / 사시 3 뷰 + 라벨)."""
import os, json
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont

SC = "/mnt/gstore/tmp/claude-1000/-mnt-gstore-home-xiangyue-RL/a2ea98c9-3261-4f2d-b674-52ae9f8190cc/scratchpad/designs"
OUT = "/mnt/gstore/home/xiangyue/RL/tt_codesign/results"
os.makedirs(OUT, exist_ok=True)
W = H = 560
VIEWS = [("측면", 90, -8), ("정면", 180, -8), ("사시", 140, -22)]
EXTRA = """
  <visual><headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65" specular="0.15 0.15 0.15"/>
   <global offwidth="1200" offheight="1200"/></visual>
  <asset><texture name="_g" type="2d" builtin="checker" rgb1="0.78 0.81 0.85" rgb2="0.66 0.70 0.75"
    width="300" height="300"/><material name="_g" texture="_g" texrepeat="8 8" reflectance="0.03"/></asset>
  <worldbody>
    <light name="_l1" pos="1.5 -1.5 2.5" dir="-0.5 0.5 -1" directional="true"/>
    <light name="_l2" pos="-1.5 1.5 2.0" dir="0.5 -0.5 -1" directional="true" diffuse="0.35 0.35 0.35"/>
    <geom name="_floor" type="plane" size="4 4 0.05" material="_g"/>
  </worldbody>
"""
rows = {json.loads(l)["stage"]: json.loads(l)
        for l in open("/mnt/gstore/home/xiangyue/RL/transformer-transformer/design_search_walk.jsonl")}
LBL = {"L0": "L0 prior 무작위", "L1": "L1 CMA-ES", "L2": "L2 prior 재학습", "L3": "L3 residual guidance (우리)"}

try:
    F = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 26)
    Fs = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)
except OSError:
    F = Fs = ImageFont.load_default()

stats, panels = {}, []
for st in ["L0", "L1", "L2", "L3"]:
    xml = open(f"{SC}/{st}/umi_on_legs_plus_plus.xml").read().replace("</mujoco>", EXTRA + "</mujoco>")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    d.qpos[2] += 0.02 - d.geom_xpos[1:, 2].min()
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(opt); opt.geomgroup[:] = 1
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    gp = d.geom_xpos[1:]
    center = (gp.max(0) + gp.min(0)) / 2
    span = float((gp.max(0) - gp.min(0)).max())
    imgs = []
    for _, az, el in VIEWS:
        cam.lookat[:] = center
        cam.distance, cam.azimuth, cam.elevation = span * 2.0, az, el
        r.update_scene(d, cam, scene_option=opt)
        imgs.append(r.render())
    r.close()
    feet = [i for i in range(m.nbody)
            if "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")]
    fp = d.xpos[feet]
    com = (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum()
    stats[st] = dict(succ=rows[st]["success_mean"], mass=float(m.body_mass.sum()),
                     base_z=float(d.qpos[2]),
                     wheelbase=float(fp[:, 0].max() - fp[:, 0].min()),
                     track=float(fp[:, 1].max() - fp[:, 1].min()),
                     com_x=float(com[0] - d.qpos[0]), com_z=float(com[2]))

    strip = np.concatenate(imgs, axis=1)
    im = Image.fromarray(strip)
    pad = Image.new("RGB", (im.width, im.height + 52), "white")
    pad.paste(im, (0, 52))
    dr = ImageDraw.Draw(pad)
    s = stats[st]
    dr.text((14, 8), f"{LBL[st]}   성공률 {s['succ']:.3f}", font=F, fill=(15, 15, 15))
    dr.text((im.width - 690, 14),
            f"질량 {s['mass']:.2f} kg | 앞뒤 발간격 {s['wheelbase']:.3f} m | "
            f"좌우 {s['track']:.3f} m | 기립높이 {s['base_z']:.3f} m",
            font=Fs, fill=(70, 70, 70))
    panels.append(np.asarray(pad))

fig = np.concatenate(panels, axis=0)
p = f"{OUT}/L0_L3_best_designs.png"
Image.fromarray(fig).save(p)
print(json.dumps(stats, indent=2, ensure_ascii=False))
print("saved", p, fig.shape)
