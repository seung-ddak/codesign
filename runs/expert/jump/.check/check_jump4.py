"""D + 착지 순간 자세 보상 확인"""
import os, sys
from pathlib import Path
RL = Path.home() / "RL"
sys.path[:0] = [str(RL / "codesign/code"), str(RL / "codesign/stage2/code"), str(RL / "transformer-transformer")]
os.chdir(RL / "transformer-transformer")
import jax, jax.numpy as jp
import task_env as C

env = C.FlatJumpEnv([0] * 8, 9394, domain_rand=True)
step, a = jax.jit(env.step), jp.zeros(env.action_size)
td, landing = 0., 0.
for seed in range(4):
    s = env.reset(jax.random.PRNGKey(seed))
    for t in range(400):
        s = step(s, a)
        v = float(s.metrics["jump_touchdown"]); td += v; landing += float(s.metrics["jump_landing"])
        assert 0. <= v <= C.JUMP_TOUCHDOWN_REWARD
print(f"touchdown 합 {td:.2f}, landing 합 {landing:.2f}")
assert td > 0
print("OK")
