"""
Comparaison expérimentale entre :

    1. CP-SAT : solution exacte
    2. Greedy : politique gloutonne par priorité
    3. FCFS    : First Come First Served
    4. Random  : politique aléatoire

Pour chaque instance :
    - une seule instance est générée
    - CP-SAT et les 3 heuristiques travaillent sur cette même instance
    - les rewards sont enregistrés
    - les écarts par rapport à CP-SAT sont calculés
    - les résultats sont sauvegardés dans un fichier CSV
"""

import csv
import random

from core.instance_generator import generate_instance
from core.simulator import SatelliteSchedulingEnv
from heuristics.baselines import (
    greedy_priority_policy,
    fcfs_policy,
    random_policy,
    run_episode
)

from solveur_cp_sat import solve_exact


# PARAMÈTRES DES EXPÉRIENCES
N_INSTANCES = 10

N_REQUESTS = 20
N_SATELLITES = 3
N_GROUND_STATIONS = 2

HORIZON = 900.0

WEIGHTS = (1.0, 0.05, 0.3)

# Une seed différente = une instance différente
SEEDS = [7, 42, 100, 200, 300, 400, 500, 600, 700, 800]


# ÉVALUATION D'UNE POLITIQUE

def evaluate_policy(
    requests,
    satellites,
    ground_stations,
    horizon,
    policy,
    seed,
    is_random=False
):

    env = SatelliteSchedulingEnv(
        requests,
        satellites,
        ground_stations,
        horizon,
        weights=WEIGHTS,
        seed=seed
    )

    if is_random:
        rng = random.Random(seed)
        run_episode(env, policy, rng=rng)
    else:
        run_episode(env, policy)

    # Statistiques
    n = len(env.requests)

    acquired = sum(
        1 for r in env.requests.values()
        if r.status >= 1
    )

    delivered = sum(
        1 for r in env.requests.values()
        if r.status == 2
    )

    delays = [
        r.t_downlink - r.t_end
        for r in env.requests.values()
        if r.status == 2
    ]

    avg_delay = (
        sum(delays) / len(delays)
        if delays
        else None
    )

    # Charge par satellite
    loads = {
        sid: 0.0
        for sid in env.satellites
    }

    for r in env.requests.values():
        if r.status >= 1:
            loads[r.acquired_by] += r.volume

    mean_load = (
        sum(loads.values()) / len(loads)
        if loads
        else 0.0
    )

    E = (
        sum((load - mean_load) ** 2 for load in loads.values())
        / len(loads)
        if loads
        else 0.0
    )

    return {
        "reward": env.total_reward,
        "acquired": acquired,
        "delivered": delivered,
        "avg_delay": avg_delay,
        "E": E
    }


# EXPÉRIMENTATION
def run_experiments():

    results = []

    print("=" * 100)
    print("                 EXPÉRIENCES CP-SAT VS BASELINES")
    print("=" * 100)

    print(
        f"\nParamètres : "
        f"{N_REQUESTS} requêtes | "
        f"{N_SATELLITES} satellite(s) | "
        f"{N_GROUND_STATIONS} station(s) | "
        f"horizon = {HORIZON:.0f} min"
    )

    print(f"Weights = {WEIGHTS}")
    print(f"Nombre d'instances = {N_INSTANCES}")

    # BOUCLE SUR LES INSTANCES
    for instance_id, seed in enumerate(SEEDS[:N_INSTANCES],start=1):

        print("\n" + "-" * 100)
        print(
            f"INSTANCE {instance_id:02d} "
            f"| seed = {seed}"
        )
        print("-" * 100)

        # 1. GÉNÉRER L'INSTANCE
        requests, satellites, ground_stations, horizon = generate_instance(
            n_requests=N_REQUESTS,
            n_satellites=N_SATELLITES,
            n_ground_stations=N_GROUND_STATIONS,
            horizon=HORIZON,
            seed=seed
        )

        # 2. CP-SAT
        cp_result = solve_exact(
            requests,
            satellites,
            ground_stations,
            horizon,
            WEIGHTS
        )

        cp_status = cp_result["status"]
        cp_reward = cp_result["objective_real"]

        print(
            f"CP-SAT : "
            f"{cp_status}"
        )

        if cp_reward is None:

            print(
                "  -> Pas de solution exploitable."
            )

            continue

        print(
            f"  Reward CP-SAT = {cp_reward:.2f}"
        )

        # 3. GREEDY
        greedy = evaluate_policy(
            requests,
            satellites,
            ground_stations,
            horizon,
            greedy_priority_policy,
            seed
        )

        # 4. FCFS
        fcfs = evaluate_policy(
            requests,
            satellites,
            ground_stations,
            horizon,
            fcfs_policy,
            seed
        )

        # 5. RANDOM
        random_result = evaluate_policy(
            requests,
            satellites,
            ground_stations,
            horizon,
            random_policy,
            seed,
            is_random=True
        )

        # 6. CALCUL DES ECARTS
        def calculate_gap(reward):

            ecart = cp_reward - reward

            if cp_reward != 0:
                ecart_pct = 100 * ecart / abs(cp_reward)
            else:
                ecart_pct = 0.0

            return ecart, ecart_pct

        greedy_gap, greedy_gap_pct = calculate_gap(
            greedy["reward"]
        )

        fcfs_gap, fcfs_gap_pct = calculate_gap(
            fcfs["reward"]
        )

        random_gap, random_gap_pct = calculate_gap(
            random_result["reward"]
        )

        # 7. AFFICHAGE
        print(
            f"Greedy : "
            f"reward = {greedy['reward']:.2f} | "
            f"acquis = {greedy['acquired']}/{N_REQUESTS} | "
            f"livrés = {greedy['delivered']}/{N_REQUESTS} | "
            f"écart = {greedy_gap_pct:.2f}%"
        )

        print(
            f"FCFS   : "
            f"reward = {fcfs['reward']:.2f} | "
            f"acquis = {fcfs['acquired']}/{N_REQUESTS} | "
            f"livrés = {fcfs['delivered']}/{N_REQUESTS} | "
            f"écart = {fcfs_gap_pct:.2f}%"
        )

        print(
            f"Random : "
            f"reward = {random_result['reward']:.2f} | "
            f"acquis = {random_result['acquired']}/{N_REQUESTS} | "
            f"livrés = {random_result['delivered']}/{N_REQUESTS} | "
            f"écart = {random_gap_pct:.2f}%"
        )

        # 8. STOCKAGE
        results.append({

            # Instance
            "instance": instance_id,
            "seed": seed,

            # Paramètres
            "n_requests": N_REQUESTS,
            "n_satellites": N_SATELLITES,
            "n_ground_stations": N_GROUND_STATIONS,
            "horizon": HORIZON,

            # CP-SAT
            "cp_sat_status": cp_status,
            "cp_sat_reward": cp_reward,

            # Greedy
            "greedy_reward": greedy["reward"],
            "greedy_acquired": greedy["acquired"],
            "greedy_delivered": greedy["delivered"],
            "greedy_avg_delay": greedy["avg_delay"],
            "greedy_E": greedy["E"],
            "greedy_gap": greedy_gap,
            "greedy_gap_pct": greedy_gap_pct,

            # FCFS
            "fcfs_reward": fcfs["reward"],
            "fcfs_acquired": fcfs["acquired"],
            "fcfs_delivered": fcfs["delivered"],
            "fcfs_avg_delay": fcfs["avg_delay"],
            "fcfs_E": fcfs["E"],
            "fcfs_gap": fcfs_gap,
            "fcfs_gap_pct": fcfs_gap_pct,

            # Random
            "random_reward": random_result["reward"],
            "random_acquired": random_result["acquired"],
            "random_delivered": random_result["delivered"],
            "random_avg_delay": random_result["avg_delay"],
            "random_E": random_result["E"],
            "random_gap": random_gap,
            "random_gap_pct": random_gap_pct,
        })

    # MOYENNES
    avg_cp = sum(
        r["cp_sat_reward"]
        for r in results
    ) / len(results)

    avg_greedy = sum(
        r["greedy_reward"]
        for r in results
    ) / len(results)

    avg_fcfs = sum(
        r["fcfs_reward"]
        for r in results
    ) / len(results)

    avg_random = sum(
        r["random_reward"]
        for r in results
    ) / len(results)

    avg_greedy_gap = sum(
        r["greedy_gap_pct"]
        for r in results
    ) / len(results)

    avg_fcfs_gap = sum(
        r["fcfs_gap_pct"]
        for r in results
    ) / len(results)

    avg_random_gap = sum(
        r["random_gap_pct"]
        for r in results
    ) / len(results)

    print("\n")
    print("=" * 100)
    print("                 MOYENNES SUR LES INSTANCES")
    print("=" * 100)

    print(
        f"\n{'Méthode':<15}"
        f"{'Reward moyen':>18}"
        f"{'Ecart moyen':>18}"
    )

    print("-" * 55)

    print(
        f"{'CP-SAT':<15}"
        f"{avg_cp:>18.2f}"
        f"{'0.00 %':>18}"
    )

    print(
        f"{'Greedy':<15}"
        f"{avg_greedy:>18.2f}"
        f"{avg_greedy_gap:>17.2f} %"
    )

    print(
        f"{'FCFS':<15}"
        f"{avg_fcfs:>18.2f}"
        f"{avg_fcfs_gap:>17.2f} %"
    )

    print(
        f"{'Random':<15}"
        f"{avg_random:>18.2f}"
        f"{avg_random_gap:>17.2f} %"
    )

# PROGRAMME PRINCIPAL
if __name__ == "__main__":
    run_experiments()