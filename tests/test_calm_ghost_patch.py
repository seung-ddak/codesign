"""Check the shipped patch against the original archive, without Isaac Gym."""
import ast
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def test_ghost_patch_is_standalone_and_rotations_match():
    root = Path(__file__).resolve().parents[1]
    patch = root/'code/motion_tracking/calm/patches/standalone_ghost_drawing.patch'
    additions = '\n'.join(line[1:] for line in patch.read_text().splitlines()
                          if line.startswith('+') and not line.startswith('+++'))
    tree = ast.parse(additions)
    assert not any(isinstance(n, ast.ImportFrom) and n.module == 'csv_to_rollout_motion' for n in ast.walk(tree))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['rot_x','rot_y']]
    ns = {'torch': torch}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<patched rotations>', 'exec'), ns)
    angles = np.array([0., np.pi/2, -np.pi/2, .37])
    for axis in ['x', 'y']:
        actual = ns['rot_'+axis](torch.tensor(angles)).numpy()
        np.testing.assert_allclose(actual, Rotation.from_euler(axis, angles[:, None]).as_matrix(), atol=1e-12)
