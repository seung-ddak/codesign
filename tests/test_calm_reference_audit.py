import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'code/motion_tracking/audit_calm_reference.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_tracking_hit_requires_velocity_facing_and_up_together():
    limits=dict(faceLatHitVelAbs=.15,faceLatHitVelRel=.3,faceLatHitYaw=.5,faceLatHitUp=.2)
    q=np.zeros((4,19));q[:,3]=1
    q[2,3]=np.cos(.4);q[2,6]=np.sin(.4)
    q[3,3]=np.cos(.2);q[3,4]=np.sin(.2)
    command=np.tile([.3,0,0],(4,1));velocity=np.tile([.3,0,0],(4,1));velocity[1,0]=.5
    hit,speed,facing,up=m.tracking_hits(q,velocity,command,limits)
    np.testing.assert_array_equal(hit,[True,False,False,False])
    np.testing.assert_allclose(speed,[0,.2,0,0],atol=1e-12)
    assert facing[2]>.5 and up[3]>.2
