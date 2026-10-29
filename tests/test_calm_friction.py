import sys
from pathlib import Path
import mujoco
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'code/motion_tracking'))
from eval_calm_mjx_tracking import set_sliding_friction,check_reference_morphology


def test_friction_override_only_changes_foot_and_plane_sliding_component():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1" friction=".8 .005 .0001"/>
      <geom name="FL_foot" type="sphere" size=".02" pos="0 0 1" friction=".8 .02 .01"/>
      <geom name="body" type="sphere" size=".1" pos="1 0 1" friction=".6 .03 .02"/>
    </worldbody></mujoco>''')
    before=m.geom_friction.copy();set_sliding_friction(m,1.)
    assert m.geom_friction[m.geom('floor').id,0]==1.
    assert m.geom_friction[m.geom('FL_foot').id,0]==1.
    np.testing.assert_array_equal(m.geom_friction[:,1:],before[:,1:])
    np.testing.assert_array_equal(m.geom_friction[m.geom('body').id],before[m.geom('body').id])
    for value in [0,-1,np.nan,np.inf]:
        with pytest.raises(ValueError):set_sliding_friction(m,value)


def test_transfer_requires_reference_for_the_actual_morphology():
    E=np.full(13,.5)
    check_reference_morphology({'uniforms':E.copy()},E)
    other=E.copy();other[3]=.6
    with pytest.raises(ValueError):check_reference_morphology({'uniforms':other},E)
    with pytest.raises(ValueError):check_reference_morphology({},E)
