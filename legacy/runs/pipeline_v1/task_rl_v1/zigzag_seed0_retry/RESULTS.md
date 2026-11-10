# zigzag: RL action replacement pilot

E frozen. Guidance and RL use identical CPU physics, S0, and evaluation reward.

| Split | Method | Median reward | Median progress | Survived |
|---|---|---:|---:|---:|
| train | guidance | -2.5387 | -0.1574 | 3/12 |
| train | untrained | -0.2374 | 0.0019 | 12/12 |
| train | rl | -0.1435 | 0.0111 | 12/12 |
| heldout | guidance | -1.8878 | -0.1445 | 1/4 |
| heldout | untrained | -0.2639 | 0.0002 | 4/4 |
| heldout | rl | -0.2125 | -0.0035 | 4/4 |
| all | guidance | -2.4277 | -0.1574 | 4/16 |
| all | untrained | -0.2374 | 0.0013 | 16/16 |
| all | rl | -0.1738 | 0.0045 | 16/16 |

Training transitions: 76,800. Seed: 0.
Training adds -20 on falling; evaluation uses the original reward without that penalty.
Held-out means four generated morphologies excluded from policy training, not a new task or seed.
CPU results must not be interpreted as identical to previous MJX results.
This pilot does not train E or generate S with diffusion; no success conclusion from survival alone.
