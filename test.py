from core.instance_splits import TAILLE_CLASSES, VAL_SEEDS
from core.instance_generator import generate_instance
from simulator.simulator import SatelliteSchedulingEnv
from heuristics.baselines import greedy_priority_policy, fcfs_policy, run_episode

CLASSE, WEIGHTS = "S", (1.0, 0.05, 0.3)
params = TAILLE_CLASSES[CLASSE]

for policy, name in [(greedy_priority_policy, "Greedy"), (fcfs_policy, "FCFS")]:
    total = 0.0
    for seed in VAL_SEEDS:
        requests, satellites, ground_stations, horizon = generate_instance(
            n_requests=params["n_requests"], n_satellites=params["n_satellites"],
            n_ground_stations=params["n_ground_stations"], horizon=params["horizon"], seed=seed)
        env = SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon,
                                      weights=WEIGHTS, seed=seed, max_steps=50)
        run_episode(env, policy)
        total += env.total_reward
    print(f"{name} : moyenne sur VAL_SEEDS = {total/len(VAL_SEEDS):.2f}")