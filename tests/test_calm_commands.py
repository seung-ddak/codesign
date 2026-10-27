import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('verify_hlc', Path(__file__).parents[1]/'code/motion_tracking/calm/verify_hlc.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_verification_commands_are_bounded_and_smooth():
    t = np.linspace(0, 10, 10001)
    c = np.array([m.command(x) for x in t])
    assert np.isfinite(c).all()
    assert np.linalg.norm(c[:,:2],axis=1).max() <= .35+1e-12
    assert np.max(np.linalg.norm(np.diff(c[:,:2],axis=0),axis=1)) < .001
    np.testing.assert_allclose(m.command(0), [.35,0,0])
    np.testing.assert_allclose(m.command(10), [0,0,np.pi/2])
