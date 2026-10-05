"""legged_gym rough terrain (Q16 a): LeggedRobotCfg.terrain + utils/terrain.py Terrain.curiculum().

Same layout and parameters as legged_gym (10 difficulty rows x 20 type columns of 8 m tiles,
25 m flat border, 0.1 m / 0.005 m grid, proportions [smooth slope, rough slope, stairs up,
stairs down, discrete] = [0.1, 0.1, 0.35, 0.25, 0.2]). The sub-terrain generators are written
here from the isaacgym terrain_utils behaviour (isaacgym itself is not redistributable and does
not import on python 3.12); legged_gym's np.random calls become one seeded RandomState.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HORIZONTAL_SCALE, VERTICAL_SCALE, BORDER_SIZE = 0.1, 0.005, 25.
TERRAIN_LENGTH = TERRAIN_WIDTH = 8.
NUM_ROWS, NUM_COLS, MAX_INIT_TERRAIN_LEVEL = 10, 20, 5
TERRAIN_PROPORTIONS = (0.1, 0.1, 0.35, 0.25, 0.2)
MEASURED_POINTS_X = np.round(np.arange(-0.8, 0.81, 0.1), 2)  # 17 points, 1.6 m
MEASURED_POINTS_Y = np.round(np.arange(-0.5, 0.51, 0.1), 2)  # 11 points, 1.0 m


def pyramid_sloped(h, slope, platform_size):
    n = h.shape[0]
    c = n // 2
    x = (c - np.abs(c - np.arange(n))) / c
    top = int(slope * (HORIZONTAL_SCALE / VERTICAL_SCALE) * (n / 2))
    h += (top * x[:, None] * x[None, :]).astype(h.dtype)
    p = int(platform_size / HORIZONTAL_SCALE / 2)
    edge = h[c - p, c - p]
    return np.clip(h, min(edge, 0), max(edge, 0))


def random_uniform(h, rng, min_height, max_height, step, downsampled_scale):
    lo, hi, st = (int(v / VERTICAL_SCALE) for v in (min_height, max_height, step))
    m = int(h.shape[0] * HORIZONTAL_SCALE / downsampled_scale)
    coarse = rng.choice(np.arange(lo, hi + st, st), (m, m))
    grid = np.linspace(0, h.shape[0] * HORIZONTAL_SCALE, m)
    fine = np.linspace(0, h.shape[0] * HORIZONTAL_SCALE, h.shape[0])
    f = RegularGridInterpolator((grid, grid), coarse)
    xx, yy = np.meshgrid(fine, fine, indexing="ij")
    return h + np.rint(f(np.stack([xx, yy], -1))).astype(h.dtype)


def pyramid_stairs(h, step_width, step_height, platform_size):
    w, s, p = int(step_width / HORIZONTAL_SCALE), int(step_height / VERTICAL_SCALE), int(platform_size / HORIZONTAL_SCALE)
    lo, hi, height = 0, h.shape[0], 0
    while hi - lo > p:
        lo, hi, height = lo + w, hi - w, height + s
        h[lo:hi, lo:hi] = height
    return h


def discrete_obstacles(h, rng, max_height, min_size, max_size, num_rects, platform_size):
    top = int(max_height / VERTICAL_SCALE)
    lo, hi, p = int(min_size / HORIZONTAL_SCALE), int(max_size / HORIZONTAL_SCALE), int(platform_size / HORIZONTAL_SCALE)
    n = h.shape[0]
    for _ in range(num_rects):
        wi, le = rng.choice(range(lo, hi, 4)), rng.choice(range(lo, hi, 4))
        si, sj = rng.choice(range(0, n - wi, 4)), rng.choice(range(0, n - le, 4))
        h[si:si + wi, sj:sj + le] = rng.choice([-top, -top // 2, top // 2, top])
    h[(n - p) // 2:(n + p) // 2, (n - p) // 2:(n + p) // 2] = 0
    return h


def make_terrain(choice, difficulty, rng):
    """legged_gym Terrain.make_terrain for the five default types."""
    n = int(TERRAIN_WIDTH / HORIZONTAL_SCALE)
    h = np.zeros((n, n), dtype=np.int16)
    edges = np.cumsum(TERRAIN_PROPORTIONS)
    slope = difficulty * 0.4
    step_height = 0.05 + 0.18 * difficulty
    if choice < edges[0]:
        h = pyramid_sloped(h, -slope if choice < edges[0] / 2 else slope, 3.)
    elif choice < edges[1]:
        h = pyramid_sloped(h, slope, 3.)
        h = random_uniform(h, rng, -0.05, 0.05, 0.005, 0.2)
    elif choice < edges[3]:
        # legged_gym negates the first stairs type: the robot starts in the pit and climbs out.
        h = pyramid_stairs(h, 0.31, -step_height if choice < edges[2] else step_height, 3.)
    else:
        h = discrete_obstacles(h, rng, 0.05 + difficulty * 0.2, 1., 2., 20, 3.)
    return h


def generate(seed=0):
    """Returns (height_samples int16 (rows, cols), env origins (NUM_ROWS, NUM_COLS, 3) in metres)."""
    rng = np.random.RandomState(seed)
    tile = int(TERRAIN_LENGTH / HORIZONTAL_SCALE)
    border = int(BORDER_SIZE / HORIZONTAL_SCALE)
    hf = np.zeros((NUM_ROWS * tile + 2 * border, NUM_COLS * tile + 2 * border), dtype=np.int16)
    origins = np.zeros((NUM_ROWS, NUM_COLS, 3))
    a, b = int((TERRAIN_LENGTH / 2 - 1) / HORIZONTAL_SCALE), int((TERRAIN_LENGTH / 2 + 1) / HORIZONTAL_SCALE)
    for j in range(NUM_COLS):
        for i in range(NUM_ROWS):
            h = make_terrain(j / NUM_COLS + 0.001, i / NUM_ROWS, rng)
            hf[border + i * tile:border + (i + 1) * tile, border + j * tile:border + (j + 1) * tile] = h
            origins[i, j] = [(i + .5) * TERRAIN_LENGTH, (j + .5) * TERRAIN_WIDTH,
                             h[a:b, a:b].max() * VERTICAL_SCALE]
    return hf, origins


if __name__ == "__main__":
    hf, origins = generate()
    edges = np.cumsum(TERRAIN_PROPORTIONS)
    kind = [int(np.searchsorted(edges, j / NUM_COLS + 0.001, side="right")) for j in range(NUM_COLS)]
    assert hf.shape == (1300, 2100) and origins.shape == (10, 20, 3)
    assert kind.count(2) == 7 and kind.count(3) == 5, kind  # 0.35 / 0.25 of 20 columns
    up, down = kind.index(2), kind.index(3)
    # stairs up starts in the pit (centre lowest), stairs down on the summit; row 9 = difficulty 0.9
    assert origins[9, up, 2] < 0 < origins[9, down, 2]
    tile0 = hf[250 + 9 * 80:250 + 10 * 80, 250 + down * 80:250 + (down + 1) * 80]
    assert np.diff(np.unique(tile0)).max() == int((0.05 + 0.18 * 0.9) / VERTICAL_SCALE)
    print("terrain ok", hf.min() * VERTICAL_SCALE, hf.max() * VERTICAL_SCALE, kind)
