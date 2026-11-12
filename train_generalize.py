"""
Entrainement Double_DQN generalise.

Difference cle par rapport a train_double_dqn.py : la seed d'instance change a
CHAQUE episode (tiree aleatoirement dans TRAIN_SEEDS), au lieu d'une
seed fixe. L'agent apprend ainsi une politique qui doit fonctionner sur
des instances variees, pas une sequence d'actions par coeur pour un
seul cas.

Le suivi de progression se fait sur VAL_SEEDS (jamais vues par
l'entrainement), moyenne sur toutes les instances de validation. Le
meilleur modele (au sens de cette moyenne de validation) est sauvegarde
au fil de l'entrainement -- c'est CE fichier de poids qui sera charge
ensuite pour l'evaluation finale sur le jeu de test.
"""

import csv
import random

import matplotlib.pyplot as plt
import torch
import time


from core.instance_splits import TAILLE_CLASSES, TRAIN_SEEDS, VAL_SEEDS

from core.instance_generator import generate_instance
from core.simulator import SatelliteSchedulingEnv
from double_dqn_agent import DQNAgent

CLASSE = "S"                     # classe de taille utilisee pour cette phase d'entrainement
WEIGHTS = (1.0, 0.05, 0.3)

N_EPISODES = 600
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.998
BATCH_SIZE = 32
EVAL_EVERY = 50


def make_env(seed, classe=CLASSE):
    params = TAILLE_CLASSES[classe]
    requests, satellites, ground_stations, horizon = generate_instance(
        n_requests=params["n_requests"], n_satellites=params["n_satellites"],
        n_ground_stations=params["n_ground_stations"], horizon=params["horizon"], seed=seed)
    return SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon,
                                   weights=WEIGHTS, seed=seed)




def evaluate_on_seeds(agent, seeds):
    """Moyenne de la recompense (politique gloutonne, epsilon=0) sur un ensemble de seeds."""
    total = 0.0
    for seed in seeds:
        env = make_env(seed)
        env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(env, epsilon=0.0)
            _, _, done, _ = env.step(action)
        total += env.total_reward
    return total / len(seeds)


if __name__ == "__main__":
    torch.manual_seed(0)
    rng = random.Random(0)
    agent = DQNAgent(seed=0)
    epsilon = EPSILON_START
    best_val = -float("inf")
    val_history = []   # (episode, reward_validation) -- pour la courbe

    for episode in range(1, N_EPISODES + 1):
        t_episode = time.time()
        train_seed = rng.choice(TRAIN_SEEDS)          # <-- coeur de cette etape 

        t = time.time()
        env = make_env(train_seed)
        print("make_env :", time.time() - t)
        env.reset()
        done = False

        n_steps = 0
        time_env = 0
        time_action = 0
        time_train = 0

        while not done:
            t = time.time()
            action, graph_before = agent.select_action(env, epsilon)
            time_action += time.time() - t

            t = time.time()
            _, reward, done, _ = env.step(action)
            time_env += time.time() - t

            if action[0] == "acquire":
                agent.remember(graph_before, action, reward, env, done)

            t = time.time()
            agent.train_step(BATCH_SIZE)
            time_train += time.time() - t

            n_steps += 1

        print(f"episode {episode} termine (seed={train_seed}, reward={env.total_reward:.2f})")  
        print(
            f"Episode {episode} | "
            f"steps={n_steps} | "
            f"action={time_action:.2f}s | "
            f"env={time_env:.2f}s | "
            f"train={time_train:.2f}s | "
            f"total={time.time()-t_episode:.2f}s"
        )

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        #if episode % EVAL_EVERY == 0:

        t_eval = time.time()

        val_reward = evaluate_on_seeds(agent, VAL_SEEDS)

        eval_time = time.time() - t_eval
        val_history.append((episode, val_reward))
        print(f"episode {episode:5d}  epsilon={epsilon:.3f}  "
                  f"reward(val, moyenne sur {len(VAL_SEEDS)} instances)={val_reward:.2f}  "
                  f"best={max(best_val, val_reward):.2f}")
        print(
                f"VALIDATION episode {episode} | "
                f"time={eval_time:.2f}s | "
                f"reward={val_reward:.2f}"
            )
        if val_reward > best_val:
            best_val = val_reward
            torch.save(agent.online.state_dict(), f"best_dqn_{CLASSE}.pt")

    print(f"\nMeilleure recompense de validation obtenue : {best_val:.2f}")
    print(f"Poids sauvegardes dans best_dqn_{CLASSE}.pt")

    # --- sauvegarde CSV (pour comparer plus tard avec d'autres variantes/classes) ---
    with open(f"val_history_{CLASSE}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward_val"])
        writer.writerows(val_history)

    # --- courbe de generalisation ---
    episodes = [e for e, _ in val_history]
    rewards = [r for _, r in val_history]

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, color="tab:blue", label=f"Recompense de validation (moyenne sur {len(VAL_SEEDS)} instances)")
    plt.xlabel("Episode")
    plt.ylabel("Recompense moyenne (validation, epsilon=0)")
    plt.title(f"Courbe de generalisation DQN -- classe {CLASSE}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"dqn_generalization_curve_{CLASSE}.png", dpi=150)
    plt.show()