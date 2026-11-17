"""
Operateurs d'amelioration locale.

Prennent en entree le plan produit par l'agent entraine (politique gloutonne,
epsilon=0) et tentent de l'ameliorer par trois operateurs complementaires,
chacun accepte seulement s'il ameliore la recompense totale obtenue en
rejouant le plan dans le simulateur :

  1. reorder_within_satellite : reordonnancement des acquisitions d'un
     meme satellite (reduit le depointage/l'attente, meilleure utilisation
     des fenetres de visibilité)
  2. swap_between_satellites : permutation de l'affectation satellite
     de deux requetes, pour mieux equilibrer la charge
  3. greedy_insert_unplanned : insertion gloutonne, par priorite
     decroissante, des requetes laissees non planifiees par l'agent
"""

import copy
import random

from simulator.simulator import SatelliteSchedulingEnv


def replay_plan(requests, satellites, ground_stations, horizon, weights, seed, plan):
    """Rejoue un plan (liste de dicts {request, satellite, opp_index}, dans
    l'ordre fourni) dans un nouvel environnement."""
    env = SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon,
                                  weights=weights, seed=seed)
    env.reset()
    for entry in plan:
        action = ("acquire", entry["request"], entry["satellite"], entry["opp_index"])
        if action in env.get_valid_actions():
            env.step(action)
    env._process_downlinks()
    return env


def get_agent_plan(agent, requests, satellites, ground_stations, horizon, weights, seed):
    """Fait jouer l'agent (politique gloutonne, epsilon=0) sur l'instance et
    enregistre le plan obtenu, dans l'ordre chronologique d'execution."""
    env = SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon,
                                  weights=weights, seed=seed)
    env.reset()
    plan = []
    done = False
    while not done:
        action, _ = agent.select_action(env, epsilon=0.0)
        if action[0] == "acquire":
            plan.append({"request": action[1], "satellite": action[2], "opp_index": action[3]})
        _, _, done, _ = env.step(action)
    return plan, env.total_reward


def reorder_within_satellite(requests, satellites, ground_stations, horizon, weights, seed,
                              plan, rng, n_tries=30):
    """Operateur 1 : echange l'ordre de deux acquisitions d'un meme satellite,
    accepte si la recompense totale s'ameliore."""
    best_plan = copy.deepcopy(plan)
    best_reward = replay_plan(requests, satellites, ground_stations, horizon, weights, seed, best_plan).total_reward

    for _ in range(n_tries):
        by_sat = {}
        for idx, e in enumerate(best_plan):
            by_sat.setdefault(e["satellite"], []).append(idx)
        candidate_sats = [s for s, idxs in by_sat.items() if len(idxs) >= 2]
        if not candidate_sats:
            break
        s = rng.choice(candidate_sats)
        i, j = rng.sample(by_sat[s], 2)

        new_plan = copy.deepcopy(best_plan)
        new_plan[i], new_plan[j] = new_plan[j], new_plan[i]

        reward = replay_plan(requests, satellites, ground_stations, horizon, weights, seed, new_plan).total_reward
        if reward > best_reward:
            best_plan, best_reward = new_plan, reward

    return best_plan, best_reward


def swap_between_satellites(requests, satellites, ground_stations, horizon, weights, seed,
                             plan, rng, n_tries=30):
    """Operateur 2 : echange l'affectation satellite de deux requetes du plan,
    si chacune possede une opportunite sur le satellite de l'autre, accepte
    si la recompense totale s'ameliore (typiquement via un meilleur
    equilibrage de charge)."""
    requests_by_id = {r.id: r for r in requests}
    best_plan = copy.deepcopy(plan)
    best_reward = replay_plan(requests, satellites, ground_stations, horizon, weights, seed, best_plan).total_reward

    for _ in range(n_tries):
        if len(best_plan) < 2:
            break
        i, j = rng.sample(range(len(best_plan)), 2)
        e_i, e_j = best_plan[i], best_plan[j]
        if e_i["satellite"] == e_j["satellite"]:
            continue

        opp_i_on_sj = next((o for o, opp in enumerate(requests_by_id[e_i["request"]].opportunities)
                             if opp.satellite_id == e_j["satellite"]), None)
        opp_j_on_si = next((o for o, opp in enumerate(requests_by_id[e_j["request"]].opportunities)
                             if opp.satellite_id == e_i["satellite"]), None)
        if opp_i_on_sj is None or opp_j_on_si is None:
            continue   # pas d'opportunite croisee disponible, echange impossible

        new_plan = copy.deepcopy(best_plan)
        new_plan[i] = {"request": e_i["request"], "satellite": e_j["satellite"], "opp_index": opp_i_on_sj}
        new_plan[j] = {"request": e_j["request"], "satellite": e_i["satellite"], "opp_index": opp_j_on_si}

        reward = replay_plan(requests, satellites, ground_stations, horizon, weights, seed, new_plan).total_reward
        if reward > best_reward:
            best_plan, best_reward = new_plan, reward

    return best_plan, best_reward


def greedy_insert_unplanned(requests, satellites, ground_stations, horizon, weights, seed, plan):
    """Operateur 3 : tente d'inserer, par priorite decroissante, chaque
    requete laissee non planifiee par l'agent, dans la premiere opportunite
    qui reste realisable et ameliore la recompense."""
    planned_ids = {e["request"] for e in plan}
    leftover = sorted((r for r in requests if r.id not in planned_ids), key=lambda r: -r.priority)

    best_plan = copy.deepcopy(plan)
    best_reward = replay_plan(requests, satellites, ground_stations, horizon, weights, seed, best_plan).total_reward

    for r in leftover:
        for o_idx, opp in enumerate(r.opportunities):
            trial_plan = best_plan + [{"request": r.id, "satellite": opp.satellite_id, "opp_index": o_idx}]
            reward = replay_plan(requests, satellites, ground_stations, horizon, weights, seed, trial_plan).total_reward
            if reward > best_reward:
                best_plan, best_reward = trial_plan, reward
                break   # requete inseree avec succes, passer a la suivante

    return best_plan, best_reward


def improve_plan(agent, requests, satellites, ground_stations, horizon, weights, seed, rng_seed=0):
    """Pipeline complet : plan initial de l'agent, puis application
    sequentielle des trois operateurs. Affiche la progression de la
    recompense apres chaque operateur, pour analyse (chapitre 5)."""
    rng = random.Random(rng_seed)
    plan, reward0 = get_agent_plan(agent, requests, satellites, ground_stations, horizon, weights, seed)
    print(f"Recompense agent seul              : {reward0:.2f}")

    plan, reward1 = reorder_within_satellite(requests, satellites, ground_stations, horizon, weights, seed, plan, rng)
    print(f"+ reordonnancement intra-satellite  : {reward1:.2f}")

    plan, reward2 = swap_between_satellites(requests, satellites, ground_stations, horizon, weights, seed, plan, rng)
    print(f"+ permutation inter-satellites      : {reward2:.2f}")

    plan, reward3 = greedy_insert_unplanned(requests, satellites, ground_stations, horizon, weights, seed, plan)
    print(f"+ insertion gloutonne               : {reward3:.2f}")

    return plan, reward3