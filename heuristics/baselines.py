"""
Politiques simples pour: valider que le simulateur tourne correctement et 
servir de points de comparaison pour l'agent DRL plus tard 
"""

import random


def random_policy(env, rng=None):
    rng = rng or random
    actions = env.get_valid_actions()
    if not actions:
        return ("wait",)
    return rng.choice(actions)


def greedy_priority_policy(env, rng=None):
    """Priorite aux requetes les plus prioritaires parmi les actions faisables."""
    actions = env.get_valid_actions()
    if not actions:
        return ("wait",)
    return max(actions, key=lambda a: env.requests[a[1]].priority)


def fcfs_policy(env, rng=None):
    """First Come First Served : la fenetre de visibilite qui s'ouvre le plus tot."""
    actions = env.get_valid_actions()
    if not actions:
        return ("wait",)
    def window_start(a):
        _, rid, sid, o_idx = a
        return env.requests[rid].opportunities[o_idx].start
    return min(actions, key=window_start)


def run_episode(env, policy_fn, rng=None, verbose=False):
    env.reset()
    done = False
    while not done:
        action = policy_fn(env, rng)
        _, reward, done, _ = env.step(action)
        if verbose:
            print(f"t={env.t:6.1f}  action={action}  reward={reward:+.2f}")
    return env