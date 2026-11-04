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


#vérification pour récompense négative 
from core.instance_generator import generate_instance
from core.simulator import SatelliteSchedulingEnv

requests, satellites, ground_stations, horizon = generate_instance(
    n_requests=12, n_satellites=2, n_ground_stations=2, horizon=600.0, seed=7)
env = SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon, weights=(1.0, 0.05, 0.3), seed=7)
run_episode(env, greedy_priority_policy)

acquired = sum(1 for r in env.requests.values() if r.status >= 1)
delivered = sum(1 for r in env.requests.values() if r.status == 2)
delays = [r.t_downlink - r.t_end for r in env.requests.values() if r.status == 2]
print(f"acquises={acquired}  livrees={delivered}  delai_moyen={sum(delays)/len(delays):.1f}" if delays else "aucune livraison")
print(f"recompense totale={env.total_reward:.2f}")