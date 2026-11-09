"""CPU-friendly visualization overlays for TT vs physical MuJoCo videos.

Green is the replayed MuJoCo path and orange is the TT prediction. Red
circles mark physical slalom poles when present.
"""
import math
import numpy as np


def reference_lines(task, spec, physical_xy):
    """Return geometric reference centerline and corridor boundaries."""
    if task == "circle":
        radius = float(spec.get("radius", 1.0))
        half = float(spec.get("half_width", 0.3))
        th = np.linspace(0, 2*math.pi, 241)
        return [
            np.stack([r*np.sin(th), radius-r*np.cos(th)], 1)
            for r in (radius, radius-half, radius+half)
        ]
    xp = np.linspace(0.0, max(5.0, float(np.nanmax(physical_xy[:, 0]))+1.0), 201)
    half = float(spec.get("half_width", 0.15))
    if task == "zigzag" and "route" in spec:
        route = np.asarray(spec["route"])
        xp = np.unique(np.r_[xp, route[:, 0]])
        center = np.interp(xp, route[:, 0], route[:, 1])
    elif task == "zigzag":
        amp = float(spec.get("amp", 0.3))
        period = float(spec.get("period", 2.0))
        center = amp*(2/math.pi)*np.arcsin(np.clip(np.sin(2*math.pi*xp/period), -1, 1))
    elif task == "straight":
        center = np.zeros_like(xp)
    else:
        raise ValueError("Unknown task "+str(task))
    return [np.stack([xp, center+s], 1) for s in (0.0, -half, half)]


def slalom_markers(spec):
    """Return the real pole footprints and the allowed gate segments."""
    poles = spec.get("poles", [])
    theta = np.linspace(0, 2*math.pi, 40)
    disk = spec.get("pole_radius", 0.06) * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    half = spec.get("half_width", 0.22)
    return ([np.asarray(pole) + disk for pole in poles],
            [np.array([[x, y - half], [x, y + half]]) for x, y in spec.get("gates", [])])


def image_with_overlay(rgb, *, task, task_spec, physical_xy, predicted_xy,
                       frame, fps, candidate, method, fall_step, width=320,
                       display_time=None, termination_reason="fall"):
    """Overlay actual and TT-predicted paths in a fixed mini-map (Pillow)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)), "RGB").convert("RGBA")
    w, h = img.size
    inset_w = min(width, max(140, w//2))
    inset_h = min(250, max(130, h//2))
    x0, y0 = w-inset_w-12, 55
    if y0+inset_h > h:
        y0 = max(6, h-inset_h-6)
    canvas = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()
    d.rounded_rectangle((8, 8, w-8, 43), radius=7, fill=(5, 14, 26, 220))
    timestamp = frame/max(1,fps) if display_time is None else display_time
    header = f"{task.upper()} / {method} / candidate {candidate} / t={timestamp:.2f}s"
    d.text((18, 20), header, font=font, fill=(255,255,255,255))
    d.rounded_rectangle((x0, y0, x0+inset_w, y0+inset_h), radius=9,
                        fill=(246,248,251,242), outline=(30,41,59,255), width=2)
    d.text((x0+12,y0+10),"PATH (top view)",font=font,fill=(15,23,42,255))
    lines = reference_lines(task, task_spec, physical_xy)
    # Predicted trajectory may be inconsistent with actual dynamics; show it
    # clearly as a separate dashed approximation.
    poles, gates = slalom_markers(task_spec)
    arrays = [physical_xy, predicted_xy] + lines + poles + gates
    flat = np.concatenate([a[np.isfinite(a).all(axis=1)] for a in arrays if len(a)], axis=0)
    if len(flat) == 0:
        return img.convert("RGB")
    lo, hi = flat.min(0)-0.15, flat.max(0)+0.15
    span = np.maximum(hi-lo, 0.3)
    pw = inset_w - 24
    ph = inset_h - 57
    scale = min(pw/span[0],ph/span[1])
    start_y = y0+32
    def pt(q):
        return (int(round(x0+12+(q[0]-lo[0])*scale)),
                int(round(start_y+ph-(q[1]-lo[1])*scale)))
    for i, line in enumerate(lines):
        d.line([pt(q) for q in line],fill=(100,116,139,200) if i==0 else (170,180,195,150),width=2)
    for gate in gates:
        d.line([pt(q) for q in gate], fill=(58,92,148,200), width=2)
    for pole in poles:
        d.polygon([pt(q) for q in pole], fill=(172,39,50,255))
    if len(predicted_xy)>1:
        d.line([pt(q) for q in predicted_xy],fill=(231,124,25,115),width=2)
    shown = physical_xy[:min(frame+1,len(physical_xy))]
    if len(shown)>1:
        d.line([pt(q) for q in shown],fill=(0,132,99,255),width=3)
    if len(shown):
        px,py=pt(shown[-1])
        d.ellipse((px-4,py-4,px+4,py+4),fill=(0,132,99,255))
    d.text((x0+12,y0+inset_h-18),"REF gray / TT orange / REAL green",font=font,fill=(23,37,56,255))
    if fall_step is not None and frame >= fall_step:
        d.rounded_rectangle((14,h-53,min(w-14,360),h-14),radius=6,fill=(115,26,34,225))
        d.text((25,h-40),f"{termination_reason.upper()} at {fall_step*0.02:.2f}s (held frame)",font=font,fill=(255,255,255,255))
    return Image.alpha_composite(img,canvas).convert("RGB")
