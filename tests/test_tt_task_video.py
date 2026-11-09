"""Video assets and CPU-only visual overlay regressions (no MuJoCo GL needed)."""
import shutil
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from render_task_videos import encode_frames, select_candidates
from tt_video_overlay import image_with_overlay, reference_lines


def has_ffmpeg_encoder():
    return (shutil.which("ffmpeg") is not None
            or importlib.util.find_spec("imageio_ffmpeg") is not None)


def test_video_selects_best_physical_reward_not_prediction():
    candidates = [
        {"reward": -30.0}, {"reward": -14.0}, {"reward": None},
        {"reward": -20.0}
    ]
    assert select_candidates(candidates, "best") == [1]
    assert select_candidates(candidates, "all") == [0, 1, 2, 3]
    assert select_candidates([{"reward": None}], "best") == []


@pytest.mark.parametrize("task,spec", [
    ("circle", {"radius":1., "half_width":.3}),
    ("zigzag", {"amp":.3, "period":2., "half_width":.15}),
    ("straight", {"half_width":.15}),
])
def test_reference_geometry(task,spec):
    samples = np.array([[0.,0.],[1.,0.],[2.,.1]])
    lines = reference_lines(task,spec,samples)
    assert len(lines)==3
    assert all(a.shape[1]==2 for a in lines)
    assert all(np.isfinite(a).all() for a in lines)
    np.testing.assert_allclose(lines[0][0],np.zeros(2),atol=1e-6)


def test_video_overlay_draws_trajectory_and_fall():
    pytest.importorskip("PIL")
    rgb = np.zeros((240,400,3),dtype=np.uint8)
    physical = np.stack([np.arange(15)*.02,np.arange(15)*0.],1)
    pred = np.stack([np.arange(500)*.02,np.zeros(500)],1)
    img = image_with_overlay(
        rgb, task="straight", task_spec={"half_width":.15},
        physical_xy=physical,predicted_xy=pred,
        frame=14,fps=25,candidate=0,method="gradient",
        fall_step=14,display_time=2.5)
    assert img.size == (400,240)
    assert np.asarray(img).shape==(240,400,3)
    assert np.asarray(img).sum()>100


@pytest.mark.skipif(not has_ffmpeg_encoder(), reason="ffmpeg / imageio-ffmpeg not installed")
def test_ffmpeg_mp4_stream_and_no_overwrite(tmp_path):
    path=tmp_path/"sample.mp4"
    frames=[np.zeros((128,160,3),dtype=np.uint8),
            np.full((128,160,3),64,dtype=np.uint8)]
    encode_frames(path,frames,width=160,height=128,fps=25)
    assert path.stat().st_size>100
    assert not (tmp_path/"sample.partial.mp4").exists()
    with pytest.raises(FileExistsError):
        encode_frames(path,frames,width=160,height=128,fps=25)


@pytest.mark.skipif(not has_ffmpeg_encoder(), reason="ffmpeg / imageio-ffmpeg not installed")
def test_invalid_frame_rolls_back_partial_file(tmp_path):
    path=tmp_path/"bad.mp4"
    with pytest.raises(ValueError):
        encode_frames(path,[np.zeros((20,20,3),dtype=np.uint8)],
                      width=160,height=128,fps=25)
    assert not path.exists()
    assert not (tmp_path/"bad.partial.mp4").exists()


def test_opengl_free_wireframe_projection(monkeypatch):
    pytest.importorskip("PIL")
    import tt_wireframe_render as wire
    center = np.array([0.,0.,.3])
    rotation = np.eye(3)
    limbs = {
        leg: [center, center + np.array([sx*.15, sy*.10, 0.]),
              center + np.array([sx*.22, sy*.16, -.1]),
              center + np.array([sx*.22, sy*.16, -.29])]
        for leg, (sx,sy) in zip(("FR","FL","RR","RL"),
                                ((1,-1),(1,1),(-1,-1),(-1,1)))
    }
    monkeypatch.setattr(wire,"_body_and_leg_paths",
                        lambda model,data: (center,rotation,limbs))
    rgb = wire.draw_kinematic_frame(None,None,width=480,height=300)
    assert rgb.shape==(300,480,3)
    assert rgb.dtype==np.uint8
    assert rgb.max()>rgb.min()
    assert not np.all(rgb==rgb[0,0])
