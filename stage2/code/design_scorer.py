"""Frozen-pi_theta scorer for arbitrary TT designs, compiled once (T7, D39 / D51 / D52).

score(designs) -> episode return per design, averaged over EVAL_COMMANDS (one held command per
episode, no domain randomisation). Every structure group keeps SLOTS design slots; new designs are
written into the slots as jit *inputs*, so the rollout compiles once no matter which designs
pi_phi generates. Unused slots run the group's template design and are ignored. A group with
more than SLOTS designs in one batch is scored over several passes.
"""

import jax
import jax.numpy as jp
import numpy as np

from eval_locomotion_policy import EVAL_COMMANDS, load_policy
from locomotion_env import EPISODE_STEPS, LocomotionEnv
from multi_design_env import PER_DESIGN, MultiDesignEnv, structure_key


class DesignScorer:
    def __init__(self, template_groups, ckpt, slots=8):
        """template_groups: one (choices, seed) list per structure group, e.g. the T5 designs."""
        self.slots, self.episodes = slots, len(EVAL_COMMANDS)
        templates = [[g[i % len(g)] for i in range(slots)] for g in template_groups]
        self.keys = [structure_key(g[0][0]) for g in templates]
        self.env = MultiDesignEnv(templates, EPISODE_STEPS, hold_command=True, domain_rand=False)
        self.groups = [w.env.env for w in self.env.groups]  # AutoReset(Episode(DesignGroupEnv))
        self.base = [g.designs for g in self.groups]
        self.policy = load_policy(ckpt)
        per_group = slots * self.episodes
        # Slot i of a group runs design i % slots with command i // slots.
        self.commands = jp.array([[vx, vy, wz, 0.] for _, vx, vy, wz in EVAL_COMMANDS])[
            np.arange(per_group) // slots]
        self.n = per_group * len(self.groups)
        self._rollout = jax.jit(self._rollout_impl)

    def _rollout_impl(self, designs, rng):
        saved = [g.designs for g in self.groups]
        try:
            for g, d in zip(self.groups, designs):
                g.designs = d
            state = self.env.reset(jax.random.split(rng, self.n))
            state = self.env._merge([s.replace(info={**s.info, "command": self.commands})
                                     for s in state.info["groups"]])

            def body(carry, key):
                state, alive, ret = carry
                state = self.env.step(state, self.policy(state.obs, key)[0])
                ret += state.reward * alive
                alive *= 1. - state.done
                return (state, alive, ret), None

            zero = jp.zeros(self.n)
            (_, _, ret), _ = jax.lax.scan(body, (state, jp.ones(self.n), zero),
                                          jax.random.split(rng, EPISODE_STEPS))
            return ret
        finally:
            for g, d in zip(self.groups, saved):
                g.designs = d

    def _stacked(self, group, designs):
        """Group g's slot arrays with `designs` written into the first slots."""
        base = self.base[group]
        leaves, treedef = jax.tree.flatten(base)
        new = [np.array(x) for x in leaves]
        for slot, (c, s) in enumerate(designs):
            e = LocomotionEnv(list(c), s, hold_command=True, domain_rand=False)
            one = jax.tree.leaves({name: getattr(e, name) for name in PER_DESIGN})
            for arr, x in zip(new, one):
                arr[slot] = np.asarray(x)
        return jax.tree.unflatten(treedef, [jp.asarray(x) for x in new])

    def score(self, designs, seed=0):
        """designs: [(choices, seed)]; returns the mean return over EVAL_COMMANDS for each."""
        by_group = {}
        for i, (c, _) in enumerate(designs):
            by_group.setdefault(self.keys.index(structure_key(c)), []).append(i)
        scores = np.full(len(designs), np.nan)
        per_group = self.slots * self.episodes
        while any(by_group.values()):
            batch = {g: idx[:self.slots] for g, idx in by_group.items() if idx}
            by_group = {g: idx[self.slots:] for g, idx in by_group.items()}
            args = tuple(self._stacked(g, [designs[i] for i in batch[g]]) if g in batch
                         else self.base[g] for g in range(len(self.groups)))
            ret = np.asarray(jax.device_get(self._rollout(args, jax.random.PRNGKey(seed))))
            for g, idx in batch.items():
                block = ret[g * per_group:(g + 1) * per_group].reshape(self.episodes, self.slots)
                for slot, i in enumerate(idx):
                    scores[i] = block[:, slot].mean()
        return scores
