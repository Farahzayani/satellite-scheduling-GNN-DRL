"""
Evaluation finale : compare CP-SAT / DQN (poids geles) / Greedy / FCFS /
Random sur les instances de TEST uniquement (jamais vues a l'entrainement).

Produit :
  - detailed_results_<CLASSE>.csv   : detail par instance, analogue a Table A.1
  - un tableau imprime dans la console : statistiques agregees
    (min/Q1/mediane/moyenne/Q3/max) du gap (%) par rapport a CP-SAT,
    analogue a Table 5.4 de Boukamcha (2025).
"""

import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from core.instance_splits import TAILLE_CLASSES, TEST_SEEDS
from core.instance_generator import generate_instance
from simulator.simulator import SatelliteSchedulingEnv
from heuristics.baselines import greedy_priority_policy, fcfs_policy, random_policy, run_episode
from solveur_cp_sat import solve_exact
from models.dqn_agent import DQNAgent

CLASSE = "S"
WEIGHTS = (1.0, 0.05, 0.3)


def make_instance(seed, classe=CLASSE):
    params = TAILLE_CLASSES[classe]
    return generate_instance(
        n_requests=params["n_requests"], n_satellites=params["n_satellites"],
        n_ground_stations=params["n_ground_stations"], horizon=params["horizon"], seed=seed)


def evaluate_dqn(agent, requests, satellites, ground_stations, horizon, seed):
    env = SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon,
                                  weights=WEIGHTS, seed=seed)
    env.reset()
    done = False
    while not done:
        action, _ = agent.select_action(env, epsilon=0.0)   # politique gloutonne, aucune exploration
        _, _, done, _ = env.step(action)
    return env.total_reward


def evaluate_heuristic(requests, satellites, ground_stations, horizon, seed, policy, is_random=False):
    env = SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon,
                                  weights=WEIGHTS, seed=seed)
    if is_random:
        run_episode(env, policy, rng=random.Random(seed))
    else:
        run_episode(env, policy)
    return env.total_reward


def gap_pct(reward, reward_cp_sat):
    """Ecart en % par rapport a CP-SAT (positif = moins bon que l'optimum, negatif = meilleur
    que la solution -- possible seulement si CP-SAT n'a atteint qu'une solution faisable, pas
    prouvee optimale, sur les instances plus grandes)."""
    if reward_cp_sat == 0:
        return 0.0
    return 100.0 * (reward_cp_sat - reward) / abs(reward_cp_sat)


if __name__ == "__main__":
    agent = DQNAgent(seed=0)
    agent.online.load_state_dict(torch.load(f"best_dqn_{CLASSE}.pt"))
    agent.online.eval()

    rows = []   # une ligne par instance de test -- future "Table A.1"

    for instance_id, seed in enumerate(TEST_SEEDS, start=1):
        requests, satellites, ground_stations, horizon = make_instance(seed)

        cp = solve_exact(requests, satellites, ground_stations, horizon, WEIGHTS)
        r_cp = cp["objective_real"]
        if r_cp is None:
            print(f"instance {instance_id} (seed={seed}) : CP-SAT n'a pas trouve de solution, ignoree")
            continue

        r_dqn = evaluate_dqn(agent, requests, satellites, ground_stations, horizon, seed)
        r_greedy = evaluate_heuristic(requests, satellites, ground_stations, horizon, seed, greedy_priority_policy)
        r_fcfs = evaluate_heuristic(requests, satellites, ground_stations, horizon, seed, fcfs_policy)
        r_random = evaluate_heuristic(requests, satellites, ground_stations, horizon, seed, random_policy, is_random=True)

        row = {
            "id": instance_id, "seed": seed,
            "cp_sat_obj": r_cp, "cp_sat_status": cp["status"], "cp_sat_ct": cp["computing_time"],
            "dqn_reward": r_dqn, "dqn_gap_pct": gap_pct(r_dqn, r_cp),
            "greedy_reward": r_greedy, "greedy_gap_pct": gap_pct(r_greedy, r_cp),
            "fcfs_reward": r_fcfs, "fcfs_gap_pct": gap_pct(r_fcfs, r_cp),
            "random_reward": r_random, "random_gap_pct": gap_pct(r_random, r_cp),
        }
        rows.append(row)

        print(f"instance {instance_id:2d} (seed={seed:4d})  CP-SAT={r_cp:7.2f}  "
              f"DQN={r_dqn:7.2f} ({row['dqn_gap_pct']:+6.2f}%)  "
              f"Greedy={r_greedy:7.2f} ({row['greedy_gap_pct']:+6.2f}%)  "
              f"FCFS={r_fcfs:7.2f} ({row['fcfs_gap_pct']:+6.2f}%)")

    # --- sauvegarde du detail par instance (Etape 3, futur Table A.1) ---
    with open(f"detailed_results_{CLASSE}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- statistiques agregees (Etape 4, futur Table 5.4) ---
    print(f"\n=== Statistiques agregees (classe {CLASSE}, {len(rows)} instances de test) ===")
    print(f"{'Methode':<10}{'Min':>10}{'Q1':>10}{'Mediane':>10}{'Moyenne':>10}{'Q3':>10}{'Max':>10}")
    methods = ["dqn", "greedy", "fcfs", "random"]
    gap_values = {}
    for method in methods:
        values = np.array([r[f"{method}_gap_pct"] for r in rows])
        gap_values[method] = values
        stats = [values.min(), np.percentile(values, 25), np.median(values),
                 values.mean(), np.percentile(values, 75), values.max()]
        print(f"{method:<10}" + "".join(f"{v:10.2f}" for v in stats))

    # -----------------------------------------------------------
    # FIGURE 1 : boxplot du gap (%) par methode
    # (representation visuelle directe de min/Q1/mediane/Q3/max)
    # -----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.boxplot([gap_values[m] for m in methods], tick_labels=[m.upper() for m in methods])
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, label="CP-SAT (optimum = 0%)")
    plt.ylabel("Ecart a CP-SAT (%)")
    plt.title(f"Distribution de l'ecart a l'optimum par methode -- classe {CLASSE} ({len(rows)} instances de test)")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"boxplot_gap_{CLASSE}.png", dpi=150)
    plt.show()

    # -----------------------------------------------------------
    # FIGURE 2 : recompense obtenue par methode, instance par instance
    # -----------------------------------------------------------
    ids = [r["id"] for r in rows]
    plt.figure(figsize=(10, 5))
    plt.plot(ids, [r["cp_sat_obj"] for r in rows], "o-", color="black", label="CP-SAT (optimum)")
    plt.plot(ids, [r["dqn_reward"] for r in rows], "o-", label="DQN")
    plt.plot(ids, [r["greedy_reward"] for r in rows], "o-", label="Greedy")
    plt.plot(ids, [r["fcfs_reward"] for r in rows], "o-", label="FCFS")
    plt.plot(ids, [r["random_reward"] for r in rows], "o-", label="Random")
    plt.xlabel("Identifiant de l'instance de test")
    plt.ylabel("Recompense obtenue")
    plt.title(f"Comparaison instance par instance -- classe {CLASSE}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"comparison_per_instance_{CLASSE}.png", dpi=150)
    plt.show()