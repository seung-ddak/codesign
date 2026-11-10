# circle: RL action replacement pilot

E frozen. Guidance and RL use identical CPU physics, S0, and evaluation reward.

| Split | Method | Median reward | Median progress | Survived |
|---|---|---:|---:|---:|
| train | guidance | -0.8001 | -0.0091 | 7/12 |
| train | untrained | -0.0208 | -0.0000 | 12/12 |
| train | rl | -0.0101 | 0.0140 | 12/12 |
| heldout | guidance | -1.0333 | -0.0308 | 2/4 |
| heldout | untrained | -0.0815 | 0.0002 | 4/4 |
| heldout | rl | 0.0260 | 0.0179 | 4/4 |
| all | guidance | -0.9694 | -0.0091 | 9/16 |
| all | untrained | -0.0296 | -0.0000 | 16/16 |
| all | rl | -0.0081 | 0.0149 | 16/16 |

Training transitions: 768,000. Seed: 0.
Training reward: dense speed/progress/lane/stall/stability/action/torque; -20 fall. Evaluation uses the original task score.
Held-out means four generated morphologies excluded from policy training, not a new task or seed.
CPU results must not be interpreted as identical to previous MJX results.
This pilot does not train E or generate S with diffusion; no success conclusion from survival alone.
