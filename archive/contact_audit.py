"""정상 보행 중 **무엇이 바닥에 닿는지** 센다 — 낙상 종료 대상 링크를 정하기 위해.

legged_gym 은 `terminate_after_contacts_on` 에 지정한 링크만 보고, TT 는
`penalty_bad_contact_pairs` 로 calf_motor·gripper 를 지목한다. 둘 다 "발 아닌 전부" 가 아니다.
멀쩡히 걷는 로봇도 스치는 부위가 있으므로, 실측해서 목록을 정한다.

  .venv/bin/python tt_codesign/code/contact_audit.py --pkl data/flat_walk_long.pkl --traj 1
"""

import os as _os
_R = _os.environ.get("CODESIGN_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_TT = _os.environ.get("CODESIGN_TT", _os.path.join(_os.path.dirname(_R), "transformer-transformer"))

import argparse
import collections
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import mujoco
import numpy as np

TT = _TT
CODE = _os.path.dirname(_os.path.abspath(__file__))
os.chdir(TT)
sys.path.insert(0, CODE)

from eval_design_d22 import make_runner


def audit(runner, hardware_seed, traj_idx, max_steps):
    env = runner.env
    env.reset_hardware(hardware_seed=hardware_seed)
    runner.post_hardware_reset(hardware_seed=hardware_seed)
    env.traj_idx = traj_idx
    env.reset_no_obs(seed=traj_idx)
    env.p.forward()
    obs = env.get_obs()

    m = env.m
    hits = collections.Counter()
    steps = 0
    for _ in range(max_steps):
        action = runner.default_policy(obs)
        ctrl = np.asarray(action["ctrl/target_qpos"]).reshape(-1)[: env.d.ctrl.shape[0]]
        obs, _, done, _ = env.step(ctrl)
        steps += 1
        d = env.d
        seen = set()
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if m.geom_type[g1] == mujoco.mjtGeom.mjGEOM_PLANE:
                g = g2
            elif m.geom_type[g2] == mujoco.mjtGeom.mjGEOM_PLANE:
                g = g1
            else:
                continue
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "?"
            seen.add(bn.split("/")[-1] or bn)
        for b in seen:
            hits[b] += 1
        if done:
            break
    return steps, hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default=f"{_R}/results/compare_designs.json")
    p.add_argument("--pkl", default="data/flat_walk_long.pkl")
    p.add_argument("--traj", type=int, default=1)
    p.add_argument("--steps", type=int, default=600)
    args = p.parse_args()

    spec = json.load(open(args.spec))
    for tag, s in spec.items():
        runner = make_runner(s["choices"], s["expert"], args.pkl)
        steps, hits = audit(runner, s["seed"], args.traj, args.steps)
        print(f"\n[{tag}] {steps} step — 바닥에 닿은 링크 (step 비율)")
        for body, n in hits.most_common():
            print(f"   {body:32s} {n:4d}  {100 * n / steps:5.1f}%")


if __name__ == "__main__":
    main()
