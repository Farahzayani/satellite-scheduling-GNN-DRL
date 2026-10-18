# Integrated Scheduling of Earth Observation Satellite Constellations via Graph Neural Networks and Deep Reinforcement Learning
## Project presentation
This repository accompanies my master's thesis, currently in progress:

> The project investigates the use of Graph Neural Networks (GNNs) and Deep Reinforcement Learning (DRL) to solve the integrated scheduling problem of Earth observation satellite constellations.

## Introduction to the problem solved

Earth observation satellite constellations play a critical role in so many fields: environmental monitoring, disaster management, precision agriculture, urban planning, defense and security... These systems must handle a growing number of image acquisition requests, each subject to restrictive time windows, conflicting priorities, satellite availability, onboard resource limitations, and communication constraints.


<img width="768" height="512" alt="Satellite Earth Observation Infographic" src="https://github.com/user-attachments/assets/3937b35e-3cdf-4bed-a325-945e8fc9a799" />

Beyond determining **which requests should be acquired ?**, **by which satellite ?**, and **at what time ?**, a complete mission plan must also account for the **downlink phase**. Once acquired, data must be transmitted to ground stations through communication links with limited availability and transmission capacity. Coordinating acquisition and downlink activities is therefore essential to producing operationally feasible mission plans.

This project addresses the integrated acquisition–downlink scheduling problem for heterogeneous Earth observation satellite constellations. It explores the combination of mathematical optimization, discrete-event simulation, heterogeneous graph representation, and deep reinforcement learning to construct feasible and high-quality mission plans.

The proposed approach combines several components:

- A **mathematical formulation** of the integrated acquisition–downlink scheduling problem, explicitly considering satellite resources, ground station contact windows, transmission capacity, and scheduling constraints.
- A **discrete-event simulator** designed to reproduce the system dynamics and evaluate scheduling decisions, with validation against an exact solver based on Google OR-Tools CP-SAT.
- A **heterogeneous graph representation** of the system state, modeling **requests, satellites, and ground stations** as distinct node types connected through relevant operational relationships.
- A **Graph Attention Network (GATv2) encoder** designed to learn graph representations that capture the relationships between scheduling entities.
- A **Deep Reinforcement Learning framework** based on Deep Q-Networks (DQN) / Double DQN, where the agent incrementally constructs a mission plan through sequential scheduling decisions.
- A **comparative evaluation framework** using exact optimization and heuristic baselines, including Greedy, First-Come First-Served (FCFS), and Random policies.

The long-term objective is to investigate whether graph-based reinforcement learning can produce high-quality feasible schedules while generalizing across instances of different sizes and characteristics.
