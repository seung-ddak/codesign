"""Check collector edge cases and saved rollout arrays; pass NPZ paths as arguments."""
import ast
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

source = Path(__file__).with_name("collect_rollouts.py").read_text()
ast.parse(source)
# Exercise the script's real aggregation block without loading GPU policies.
block = source[source.index("pairs, seg_falls ="):source.index("csv_path =")]
for horizon in (1, 500):
    scope = dict(np=np, motions=["a", "b"], B=1, T=horizon, FALL_WINDOW=50,
                 lens=np.array([1]), out=dict(switched=np.array([[True]]),
                 clip=np.array([[0]]), next_clip=np.array([[1]]), fell=np.array([True])))
    exec(block, scope)
    assert scope["pairs"] == {("a", "b"): [1, 1]}
target_line = next(line for line in source.splitlines() if line.startswith('out["target_q"] ='))
scope = dict(np=np, n=2, env=SimpleNamespace(action_offset=np.array([1., 2.]),
             action_scale=0.25, clip_actions=(-10, 10)), out={"action": np.array([[[20., -20.]]])})
exec(target_line, scope)
np.testing.assert_allclose(scope["out"]["target_q"], [[[3.5, -0.5]]])
for filename in sys.argv[1:]:
    with np.load(filename) as z:
        valid = z["valid"]
        b, t = valid.shape
        assert z["qpos"].shape == (b, t + 1, 19)
        assert z["qvel"].shape == (b, t + 1, 18)
        assert z["contact"].shape == (b, t + 1, 4)
        assert z["action"].shape == z["target_q"].shape == (b, t, 12)
        assert not np.any(~valid[:, :-1] & valid[:, 1:])
        for key in ("qpos", "qvel", "action", "target_q", "E"):
            assert np.isfinite(z[key]).all(), key
        np.testing.assert_array_equal(z["switched"], z["clip"] != z["next_clip"])
        np.testing.assert_array_equal(z["next_clip"][:, :-1], z["clip"][:, 1:])
        assert np.all((z["clip"] >= 0) & (z["clip"] < len(z["motions"])))
        if "mode" in z and str(z["mode"]) == "primitive":
            assert not z["switched"].any()
            initial_clip = z["clip"][:, 0]
            np.testing.assert_array_equal(z["ref_frame"][:, 0], z["clip_start"][initial_clip])
            lengths = valid.sum(1)
            expected = (z["clip_end"] - z["clip_start"] - 1)[initial_clip]
            np.testing.assert_array_equal(lengths[~z["fell"]], expected[~z["fell"]])
            assert np.all(lengths <= expected)
        if "action_scale" in z:
            np.testing.assert_allclose(z["target_q"], z["action_offset"] +
                                       np.clip(z["action"], *z["clip_actions"]) * z["action_scale"])
            assert len(z["checkpoint_paths"]) == len(z["motions"])
        print(f"{filename}: {b} episodes, {valid.sum()} valid steps, fell={z['fell'].mean():.1%}")
print("PASS: terminal-switch regression and supplied rollout files")
