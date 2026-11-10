
from core.instance_generator import generate_instance
from core.simulator import SatelliteSchedulingEnv
from double_dqn_agent import DQNAgent
from heuristics.baselines import greedy_priority_policy, fcfs_policy, random_policy, run_episode

import matplotlib.pyplot as plt
import csv

# Parametres de l'instance (a aligner avec vos autres scripts de demo)
N_REQUESTS = 20
N_SATELLITES = 3
N_GROUND_STATIONS = 2
HORIZON = 900.0
WEIGHTS = (1.0, 0.05, 0.3)
TRAIN_SEED = 7

# Parametres d'entrainement
N_EPISODES = 400
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.995
BATCH_SIZE = 32
EVAL_EVERY = 50


def make_env(seed):
    requests, satellites, ground_stations, horizon = generate_instance(
        n_requests=N_REQUESTS, n_satellites=N_SATELLITES,
        n_ground_stations=N_GROUND_STATIONS, horizon=HORIZON, seed=seed)
    return SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon,
                                   weights=WEIGHTS, seed=seed)


def run_policy_reward(policy_fn, seed, rng=None):
    env = make_env(seed)
    run_episode(env, policy_fn, rng=rng)
    return env.total_reward


def evaluate_agent(agent, seed):
    """Evalue l'agent en politique GREEDY (epsilon=0, pas d'exploration)."""
    env = make_env(seed)
    env.reset()
    done = False
    while not done:
        action, _ = agent.select_action(env, epsilon=0.0)
        _, _, done, _ = env.step(action)
    return env.total_reward


if __name__ == "__main__":
    agent = DQNAgent(seed=0)
    epsilon = EPSILON_START

    # baselines de reference, calcules une seule fois sur l'instance d'entrainement
    r_greedy = run_policy_reward(greedy_priority_policy, TRAIN_SEED)
    r_fcfs = run_policy_reward(fcfs_policy, TRAIN_SEED)
    print(f"Instance seed={TRAIN_SEED} : reward Greedy={r_greedy:.2f}  FCFS={r_fcfs:.2f}\n")

    history = []
    for episode in range(1, N_EPISODES + 1):
        env = make_env(TRAIN_SEED)
        env.reset()
        done = False
        while not done:
            action, graph_before = agent.select_action(env, epsilon)
            _, reward, done, _ = env.step(action)
            if action[0] == "acquire":
                agent.remember(graph_before, action, reward, env, done)
            agent.train_step(BATCH_SIZE)

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        #if episode % EVAL_EVERY == 0:
        r_agent = evaluate_agent(agent, TRAIN_SEED)
        history.append((episode, r_agent))
        print(f"episode {episode:5d}  epsilon={epsilon:.3f}  "
                  f"reward(agent)={r_agent:7.2f} ")

    # --- sauvegarde CSV (pour comparer plus tard avec Double DQN) ---
    with open("dqn_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward_agent"])
        writer.writerows(history)

    # --- courbe ---
    episodes = [e for e, _ in history]
    rewards = [r for _, r in history]

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, label="Agent Double DQN (eval, epsilon=0)", color="tab:blue")
    plt.axhline(r_greedy, color="tab:orange", linestyle="--", label=f"Greedy ({r_greedy:.2f})")
    plt.axhline(r_fcfs, color="tab:green", linestyle="--", label=f"FCFS ({r_fcfs:.2f})")
    plt.xlabel("Episode")
    plt.ylabel("Recompense (evaluation, sans exploration)")
    plt.title(f"Courbe d'apprentissage Double DQN (instance seed={TRAIN_SEED})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("double_dqn_training_curve.png", dpi=150)
    plt.show()