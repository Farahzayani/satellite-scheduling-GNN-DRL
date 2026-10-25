# Core — Problem Modeling, Instance Generation and Simulation

The `core/` folder contains the foundational components of the satellite scheduling project. It defines the data structures used to :
- Represent the Satellite Scheduling Problem
- Generate problem instances
- Implement the simulation environment that models acquisition and downlink operations.

These components form the basis of the entire project: the heterogeneous graph representation, learning agents, and optimization methods rely on the problem entities, generated instances, and simulation logic defined here.

## Files

| File | Role |
|---|---|
| `entities.py` | Defines the core data structures: `Request`, `Satellite`, `GroundStation`, `VisibilityWindow`, and `ContactWindow`. |
| `instance_generator.py` | Generates complete problem instances, including requests, satellites, ground stations, visibility windows, and contact windows, based on configurable parameters and a random seed. |
| `simulator.py` | Implements the discrete-event simulation environment (`SatelliteSchedulingEnv`), including state transitions, reward calculation, acquisition feasibility, and downlink operations. |
| `graph_builder.py` | Constructs the heterogeneous graph representation of a problem instance for the graph neural network. |
| `instance_splits.py` | Defines the training, validation, and test instance pools, including instance size classes and random seeds. |

## `entities.py` — Problem Representation

This module defines five dataclasses representing the main entities and temporal opportunities of the mathematical formulation.

- **`Request`** — Represents an acquisition request \(i \in I\), characterized by its priority, data volume, acquisition duration, and available visibility windows. It also stores the request's execution status and acquisition/downlink timestamps.
- **`Satellite`** — Represents a satellite \(s \in S\), including its onboard memory capacity, rotation speed, stabilization time, and operational state.
- **`GroundStation`** — Represents a ground station \(g \in G\), characterized by its transmission data rate and available contact windows. Ground station availability and downlink capacity are important components of the integrated scheduling problem.
- **`VisibilityWindow`** — Represents an observation opportunity for a request-satellite pair, specifying the start time, end time, and required pointing angle.
- **`ContactWindow`** — Represents an opportunity for a satellite to communicate with a ground station during a specified time interval.


<div align="center">
  <img
    width="718"
    height="417"
    alt="Entities in entities.py Schema"
    src="https://github.com/user-attachments/assets/ba6128bb-16bd-4854-b279-2128ccd57be4"
  />
</div>

These data structures provide the common representation used by the simulator and the other project components.

## `instance_generator.py` — Problem Instance Generation

The `generate_instance(n_requests, n_satellites, n_ground_stations, horizon, seed)` function generates a complete problem instance from a set of parameters.

It creates satellites and ground stations, generates contact windows with randomized timing, and produces acquisition requests with randomized priorities, data volumes, durations, and visibility opportunities.

The orbital dynamics are intentionally simplified. Instead of using a full orbital propagator, the generator creates randomized visibility and contact windows, consistent with the simplified modeling scope of the project.

All downstream components operate on these generated windows rather than on raw orbital parameters.

## `simulator.py` — Scheduling Environment

The `SatelliteSchedulingEnv` class transforms a static problem instance into a sequential decision-making environment. It implements the state transition and reward mechanisms required for reinforcement learning.

The environment follows the standard state-action-transition-reward loop:

- **State:** Represents the current scheduling situation.
- **Actions:** Focus exclusively on acquisition decisions, represented as `("acquire", request_id, satellite_id, opportunity_index)`.
- **Feasibility:** The `get_valid_actions()` method returns acquisition actions that satisfy the implemented constraints, including visibility-window timing, onboard memory capacity, and satellite slewing requirements.
- **Reward:** Evaluates acquisition decisions according to the reward function defined in the mathematical formulation.
- **Downlink management:** Transmission operations are handled automatically by the environment. The simulator processes available contact windows and prioritizes higher-priority requests when transmission capacity is insufficient to serve all pending data.
- **Termination:** An episode ends when no feasible acquisition remains and no data is pending downlink, or when the configured step limit is reached.
- **Plan extraction:** The `get_plan()` method returns a timestamped record of acquisition and downlink operations, which can later be used for schedule analysis and visualization.

The action space is deliberately restricted to acquisition decisions, leaving downlink allocation to a predefined priority-based rule. This design allows the learning agent to focus on the main acquisition scheduling decisions while still accounting for ground station availability and transmission capacity.



As development progresses, this README will be updated to reflect the implementation of additional modules and their integration into the overall scheduling framework.
