from heuristics.local_search import improve_plan
from core.instance_generator import generate_instance
from models.double_dqn_agent import DDQNAgent
import torch

agent = DDQNAgent(seed=0)
agent.online.load_state_dict(torch.load("best_DDQN_1_S.pt"))

requests, satellites, ground_stations, horizon = generate_instance(
    n_requests=10, n_satellites=2, n_ground_stations=2, horizon=600.0, seed=200)  # une instance de TEST_SEEDS

plan_final, reward_final = improve_plan(agent, requests, satellites, ground_stations, horizon,
                                          weights=(1.0, 0.05, 0.3), seed=200)