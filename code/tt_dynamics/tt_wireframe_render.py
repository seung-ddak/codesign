"""GL-free animated 3D kinematic stick-figure for a physically replayed MuJoCo state.

This does not call MuJoCo Renderer, EGL, OSMesa, X11, OpenGL, or CUDA.
It uses forward-kinematics world-space joint anchors and foot geom positions
from the original MuJoCo model, and an orthographic isometric projection.
It is explicitly labeled a line-art/kinematics visualization, not mesh video.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LEG_COLORS = {
    "FR": (213, 113, 34),
    "FL": (44, 115, 202),
    "RR": (149, 91, 185),
    "RL": (40, 144, 116),
}
LEGS = ("FR", "FL", "RR", "RL")


def _body_and_leg_paths(model, data):
    import mujoco

    free = [j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError(f"Expected one floating-body joint, found {len(free)}")
    torso = int(model.jnt_bodyid[free[0]])
    center = np.asarray(data.xpos[torso], dtype=float).copy()
    rotation = np.asarray(data.xmat[torso], dtype=float).reshape(3, 3).copy()

    routes = {}
    for leg in LEGS:
        foot_id = int(model.geom(f"{leg}_foot").id)
        foot_body = int(model.geom_bodyid[foot_id])
        joints = []
        visited = set()
        body = foot_body
        while body != 0 and body not in visited:
            visited.add(body)
            first = int(model.body_jntadr[body])
            n = int(model.body_jntnum[body])
            for j in range(first, first + n):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    joints.append(j)
            if body == torso:
                break
            body = int(model.body_parentid[body])
        anchors = [np.asarray(data.xanchor[j], dtype=float).copy()
                   for j in reversed(joints)]
        if len(anchors) < 2:
            raise ValueError(f"Leg {leg} lacks enough hinge anchors: {len(anchors)}")
        path = [center] + anchors + [np.asarray(data.geom_xpos[foot_id], dtype=float).copy()]
        routes[leg] = path
    return center, rotation, routes


def draw_kinematic_frame(model, data, width=960, height=540):
    """Actual 3D MuJoCo FK rendered with CPU Pillow (RGB uint8 frame)."""
    if width < 240 or height < 180:
        raise ValueError("Image is too small for kinematic view")
    center, rotation, legs = _body_and_leg_paths(model, data)
    focus = center.copy()
    focus[2] = 0
    scale = min(width, height) * .67
    def project(point):
        x, y, z = np.asarray(point, dtype=float)-focus
        u = (.74*x-.67*y)*scale+width*.42
        v = (.34*x+.36*y-.88*z)*scale+height*.73
        return (int(round(u)), int(round(v)))

    image = Image.new("RGB", (width, height), (244, 248, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # The world ground grid shows stance and scale without a graphics backend.
    ground = center.copy()
    ground[2] = 0.0
    for t in np.arange(-1.5, 1.6, .25):
        line_x = [ground+np.array([-1.5,t,0]), ground+np.array([1.5,t,0])]
        line_y = [ground+np.array([t,-1.5,0]), ground+np.array([t,1.5,0])]
        draw.line([project(p) for p in line_x],fill=(210,222,233),width=1)
        draw.line([project(p) for p in line_y],fill=(210,222,233),width=1)

    # Torso cuboid oriented by the genuine base quaternion in MuJoCo.
    hx,hy,hz = .20,.105,.065
    local = np.array([[x,y,z] for x in (-hx,hx) for y in (-hy,hy) for z in (-hz,hz)])
    verts = center + local @ rotation.T
    # 0(-x,-y,-z), 1(-x,-y,+z), 2(-x,+y,-z), ...
    faces = ([0,2,6,4],[1,3,7,5],[0,1,5,4],[2,3,7,6],[0,1,3,2],[4,5,7,6])
    shaded = [(np.mean(verts[list(indices),2]),indices) for indices in faces]
    for _, indices in sorted(shaded):
        pts=[project(verts[j]) for j in indices]
        draw.polygon(pts,fill=(114,132,157),outline=(36,54,76),width=2)
    # Highlight the forward (+body X) direction.
    nose = center + rotation[:,0]*.29
    draw.line([project(center),project(nose)],fill=(245,178,60),width=5)
    px,py=project(nose)
    draw.ellipse((px-5,py-5,px+5,py+5),fill=(247,180,59))

    # Draw physical 3D joint anchor chains and the measured foot locations.
    for leg, path in legs.items():
        color=LEG_COLORS[leg]
        for p0,p1 in zip(path[:-1],path[1:]):
            draw.line([project(p0),project(p1)],fill=(31,41,55),width=10)
            draw.line([project(p0),project(p1)],fill=color,width=6)
        for point in path[1:-1]:
            px,py=project(point)
            draw.ellipse((px-5,py-5,px+5,py+5),fill=(245,248,250),outline=color,width=2)
        px,py=project(path[-1])
        draw.ellipse((px-7,py-7,px+7,py+7),fill=color,outline=(15,23,42),width=2)
    draw.rounded_rectangle((11,height-50,435,height-13),radius=6,fill=(20,35,53))
    draw.text((22,height-38),"MuJoCo 3D FK  |  CPU wireframe (not a mesh render)",
              font=font,fill=(255,255,255))
    return np.asarray(image,dtype=np.uint8)
