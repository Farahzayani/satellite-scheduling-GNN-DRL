"""
ALNS autonome pour le problème de planification satellitaire.

Le plan manipulé par l'ALNS contient uniquement des décisions d'acquisition :
    ("acquire", request_id, satellite_id, opportunity_index)

Les téléchargements restent gérés automatiquement par SatelliteSchedulingEnv.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from simulator.simulator import SatelliteSchedulingEnv


Action = Tuple[str, int, int, int]



class _PlanEvaluationEnv(SatelliteSchedulingEnv):
    """
    Variante utilisée uniquement pour évaluer un plan donné.

    Les requêtes absentes du plan sont masquées des actions valides. Cela
    permet au simulateur de terminer l'évaluation après l'exécution du plan,
    sans réintroduire automatiquement les requêtes retirées par l'ALNS.
    Le fonctionnement des acquisitions et des downlinks reste celui du
    simulateur d'origine.
    """

    def get_valid_actions(self):
        actions = super().get_valid_actions()
        if self.allowed_request_ids is None:
            return actions
        return [
            action for action in actions
            if action[0] == "acquire"
            and action[1] in self.allowed_request_ids
        ]


@dataclass
class _Operator:
    name: str
    weight: float = 1.0
    score: float = 0.0
    uses: int = 0


class ALNS:
    """
    Adaptive Large Neighborhood Search.

    Paramètres principaux
    ---------------------
    requests, satellites, ground_stations, horizon :
        Les quatre éléments retournés par generate_instance().
    iterations :
        Nombre d'itérations de recherche.
    removal_fraction :
        Fraction approximative des acquisitions retirées à chaque itération.
    temperature :
        Température initiale pour accepter occasionnellement une solution
        moins bonne (critère de type simulated annealing).
    cooling_rate :
        Facteur de refroidissement appliqué à chaque itération.
    reaction_factor :
        Vitesse d'adaptation des poids des opérateurs.
    segment_length :
        Nombre d'itérations entre deux mises à jour des poids.
    seed :
        Graine aléatoire pour rendre l'expérience reproductible.
    """

    def __init__(
        self,
        requests,
        satellites,
        ground_stations,
        horizon: float,
        iterations: int = 200,
        removal_fraction: float = 0.25,
        temperature: Optional[float] = None,
        cooling_rate: float = 0.995,
        reaction_factor: float = 0.2,
        segment_length: int = 25,
        seed: Optional[int] = None,
        weights=(1.0, 0.1, 0.05),
        max_steps: int = 1000,
    ):
        self.requests = copy.deepcopy(requests)
        self.satellites = copy.deepcopy(satellites)
        self.ground_stations = copy.deepcopy(ground_stations)
        self.horizon = float(horizon)

        self.iterations = max(1, int(iterations))
        self.removal_fraction = min(0.9, max(0.05, removal_fraction))
        self.cooling_rate = min(1.0, max(0.90, cooling_rate))
        self.reaction_factor = min(1.0, max(0.01, reaction_factor))
        self.segment_length = max(1, int(segment_length))
        self.seed = seed
        self.rng = random.Random(seed)
        self.weights = weights
        self.max_steps = max_steps

        self.request_by_id = {r.id: r for r in self.requests}
        self.request_ids = set(self.request_by_id)

        self.initial_temperature = (
            float(temperature)
            if temperature is not None
            else max(1.0, 0.05 * sum(r.priority for r in self.requests))
        )
        self.temperature = self.initial_temperature

        self.destroy_operators = [
            _Operator("random_removal"),
            _Operator("low_priority_removal"),
            _Operator("satellite_related_removal"),
        ]
        self.repair_operators = [
            _Operator("best_insertion"),
            _Operator("randomized_insertion"),
        ]

        self._evaluation_cache: Dict[Tuple[Action, ...], float] = {}

    # ------------------------------------------------------------------
    # Création et évaluation des environnements
    # ------------------------------------------------------------------

    def _new_env(self, seed_offset: int = 0):
        return SatelliteSchedulingEnv(
            copy.deepcopy(self.requests),
            copy.deepcopy(self.satellites),
            copy.deepcopy(self.ground_stations),
            self.horizon,
            weights=self.weights,
            max_steps=self.max_steps,
            seed=None if self.seed is None else self.seed + seed_offset,
        )

    def _evaluate(self, plan: Sequence[Action]) -> float:
        """Rejoue un plan dans le simulateur et retourne sa récompense totale."""
        plan = tuple(plan)

        # Une requête ne peut être acquise qu'une seule fois.
        request_ids = [a[1] for a in plan]
        if len(request_ids) != len(set(request_ids)):
            return float("-inf")

        # Vérification des identifiants et du format des actions.
        for action in plan:
            if (
                len(action) != 4
                or action[0] != "acquire"
                or action[1] not in self.request_ids
            ):
                return float("-inf")

        if plan in self._evaluation_cache:
            return self._evaluation_cache[plan]

        env = _PlanEvaluationEnv(
            copy.deepcopy(self.requests),
            copy.deepcopy(self.satellites),
            copy.deepcopy(self.ground_stations),
            self.horizon,
            weights=self.weights,
            max_steps=self.max_steps,
            seed=self.seed,
        )
        env.allowed_request_ids = set(request_ids)

        # Un plan vide est une solution valide de récompense nulle.
        if not plan:
            self._evaluation_cache[plan] = 0.0
            return 0.0

        for action in plan:
            if env.done or action not in env.get_valid_actions():
                self._evaluation_cache[plan] = float("-inf")
                return float("-inf")
            env.step(action)

        # Laisser le simulateur terminer les downlinks déjà programmés.
        # Les requêtes absentes du plan sont masquées par allowed_request_ids.
        safety_counter = 0
        while not env.done and safety_counter < self.max_steps:
            env.step(("wait",))
            safety_counter += 1

        if not env.done:
            score = float("-inf")
        else:
            score = float(env.total_reward)

        self._evaluation_cache[plan] = score
        return score

    # ------------------------------------------------------------------
    # Solution initiale indépendante
    # ------------------------------------------------------------------

    def _construct_initial_solution(self) -> List[Action]:
        """
        Construit une solution initiale avec une sélection randomisée parmi
        les actions actuellement faisables.
        """
        env = self._new_env()
        plan: List[Action] = []
        step_guard = 0

        while not env.done and step_guard < self.max_steps:
            actions = env.get_valid_actions()

            if actions:
                # Biais vers les requêtes prioritaires, avec randomisation.
                scores = []
                for action in actions:
                    request = self.request_by_id[action[1]]
                    score = max(0.01, request.priority)
                    score *= self.rng.uniform(0.5, 1.5)
                    scores.append(score)

                selected = self._weighted_choice(actions, scores)
                plan.append(selected)
                env.step(selected)
            else:
                env.step(("wait",))

            step_guard += 1

        # L'évaluation de référence vérifie que le plan est bien rejouable.
        if self._evaluate(plan) == float("-inf"):
            raise RuntimeError(
                "La solution initiale construite n'est pas rejouable "
                "dans le simulateur."
            )
        return plan

    def _weighted_choice(self, items, weights):
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(list(items))
        point = self.rng.random() * total
        cumulative = 0.0
        for item, weight in zip(items, weights):
            cumulative += weight
            if cumulative >= point:
                return item
        return items[-1]

    # ------------------------------------------------------------------
    # Opérateurs de destruction
    # ------------------------------------------------------------------

    def _remove_random(self, plan: List[Action], count: int):
        if not plan:
            return [], []
        indices = sorted(self.rng.sample(range(len(plan)), min(count, len(plan))))
        removed = [plan[i] for i in indices]
        removed_ids = set(indices)
        remaining = [a for i, a in enumerate(plan) if i not in removed_ids]
        return remaining, removed

    def _remove_low_priority(self, plan: List[Action], count: int):
        """Retire d'abord les acquisitions de priorité relativement faible."""
        ranked = sorted(
            plan,
            key=lambda a: (
                self.request_by_id[a[1]].priority,
                -self.request_by_id[a[1]].duration,
            ),
        )
        removed = ranked[: min(count, len(ranked))]
        removed_set = set(removed)
        remaining = [a for a in plan if a not in removed_set]
        return remaining, removed

    def _remove_satellite_related(self, plan: List[Action], count: int):
        """Retire un groupe d'acquisitions affectées au même satellite."""
        if not plan:
            return [], []

        seed_action = self.rng.choice(plan)
        satellite_id = seed_action[2]
        related = [a for a in plan if a[2] == satellite_id]
        unrelated = [a for a in plan if a[2] != satellite_id]
        self.rng.shuffle(related)

        removed = related[: min(count, len(related))]
        remaining = unrelated + related[len(removed):]
        # Préserver autant que possible l'ordre relatif des acquisitions.
        remaining_set = set(remaining)
        remaining = [a for a in plan if a in remaining_set]
        return remaining, removed

    # ------------------------------------------------------------------
    # Opérateurs de réparation
    # ------------------------------------------------------------------

    def _insertion_candidates(self, plan: List[Action], action: Action):
        """Génère toutes les positions possibles d'insertion de l'action."""
        for position in range(len(plan) + 1):
            candidate = plan[:position] + [action] + plan[position:]
            yield candidate

    def _best_position(self, plan: List[Action], request_id: int):
        """Trouve la meilleure opportunité et position d'insertion."""
        request = self.request_by_id[request_id]
        best_plan = None
        best_score = float("-inf")

        for opp_index, opportunity in enumerate(request.opportunities):
            action = (
                "acquire",
                request.id,
                opportunity.satellite_id,
                opp_index,
            )
            for candidate in self._insertion_candidates(plan, action):
                score = self._evaluate(candidate)
                if score > best_score:
                    best_score = score
                    best_plan = candidate

        return best_plan, best_score

    def _repair_best_insertion(self, plan: List[Action], removed: List[Action]):
        """Réinsère les requêtes retirées en privilégiant la meilleure récompense."""
        repaired = list(plan)
        removed_ids = list(dict.fromkeys(a[1] for a in removed))
        removed_ids.sort(
            key=lambda rid: self.request_by_id[rid].priority,
            reverse=True,
        )

        for request_id in removed_ids:
            candidate, score = self._best_position(repaired, request_id)
            if candidate is not None and score != float("-inf"):
                repaired = candidate
        return repaired

    def _repair_randomized_insertion(
        self, plan: List[Action], removed: List[Action]
    ):
        """
        Réparation randomisée : examine les insertions faisables, puis choisit
        aléatoirement parmi les trois meilleures candidates.
        """
        repaired = list(plan)
        removed_ids = list(dict.fromkeys(a[1] for a in removed))
        self.rng.shuffle(removed_ids)

        for request_id in removed_ids:
            request = self.request_by_id[request_id]
            candidates = []

            for opp_index, opportunity in enumerate(request.opportunities):
                action = (
                    "acquire",
                    request.id,
                    opportunity.satellite_id,
                    opp_index,
                )
                for candidate in self._insertion_candidates(repaired, action):
                    score = self._evaluate(candidate)
                    if score != float("-inf"):
                        candidates.append((score, candidate))

            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                top = candidates[: min(3, len(candidates))]
                _, selected_plan = self.rng.choice(top)
                repaired = selected_plan

        return repaired

    # ------------------------------------------------------------------
    # Sélection adaptative des opérateurs
    # ------------------------------------------------------------------

    def _select_operator(self, operators: List[_Operator]) -> _Operator:
        return self._weighted_choice(
            operators, [max(0.001, op.weight) for op in operators]
        )

    def _update_operator_scores(
        self,
        destroy_op: _Operator,
        repair_op: _Operator,
        reward_points: float,
    ):
        destroy_op.score += reward_points
        destroy_op.uses += 1
        repair_op.score += reward_points
        repair_op.uses += 1

    def _adapt_weights(self):
        for operator in self.destroy_operators + self.repair_operators:
            if operator.uses > 0:
                average_score = operator.score / operator.uses
                operator.weight = (
                    (1.0 - self.reaction_factor) * operator.weight
                    + self.reaction_factor * average_score
                )
                operator.weight = max(0.05, operator.weight)
            operator.score = 0.0
            operator.uses = 0

    # ------------------------------------------------------------------
    # Boucle principale ALNS
    # ------------------------------------------------------------------

    def run(self):
        """
        Exécute l'ALNS.

        Retourne
        --------
        best_plan : list[tuple]
            Meilleure séquence d'actions d'acquisition trouvée.
        best_reward : float
            Récompense totale calculée par le simulateur.
        history : list[dict]
            Historique de la meilleure récompense et de la récompense courante.
        """
        current_plan = self._construct_initial_solution()
        current_score = self._evaluate(current_plan)

        best_plan = list(current_plan)
        best_score = current_score

        self.temperature = self.initial_temperature
        history = [{
            "iteration": 0,
            "current_reward": current_score,
            "best_reward": best_score,
            "temperature": self.temperature,
        }]

        for iteration in range(1, self.iterations + 1):
            if not current_plan:
                removal_count = 0
            else:
                removal_count = max(
                    1,
                    int(math.ceil(len(current_plan) * self.removal_fraction)),
                )

            destroy_op = self._select_operator(self.destroy_operators)
            repair_op = self._select_operator(self.repair_operators)

            if destroy_op.name == "random_removal":
                partial_plan, removed = self._remove_random(
                    current_plan, removal_count
                )
            elif destroy_op.name == "low_priority_removal":
                partial_plan, removed = self._remove_low_priority(
                    current_plan, removal_count
                )
            else:
                partial_plan, removed = self._remove_satellite_related(
                    current_plan, removal_count
                )

            if repair_op.name == "best_insertion":
                candidate_plan = self._repair_best_insertion(
                    partial_plan, removed
                )
            else:
                candidate_plan = self._repair_randomized_insertion(
                    partial_plan, removed
                )

            candidate_score = self._evaluate(candidate_plan)
            reward_points = 0.0

            if candidate_score != float("-inf"):
                if candidate_score > best_score:
                    best_plan = list(candidate_plan)
                    best_score = candidate_score
                    current_plan = list(candidate_plan)
                    current_score = candidate_score
                    reward_points = 5.0
                elif candidate_score > current_score:
                    current_plan = list(candidate_plan)
                    current_score = candidate_score
                    reward_points = 3.0
                else:
                    # Accepter parfois une solution moins bonne pour sortir
                    # d'un optimum local.
                    delta = candidate_score - current_score
                    denominator = max(self.temperature, 1e-12)
                    probability = math.exp(min(0.0, delta / denominator))
                    if self.rng.random() < probability:
                        current_plan = list(candidate_plan)
                        current_score = candidate_score
                        reward_points = 1.0

            self._update_operator_scores(
                destroy_op, repair_op, reward_points
            )

            if iteration % self.segment_length == 0:
                self._adapt_weights()

            self.temperature *= self.cooling_rate

            history.append({
                "iteration": iteration,
                "current_reward": current_score,
                "best_reward": best_score,
                "temperature": self.temperature,
            })

        return best_plan, best_score, history


def solve_alns(
    requests,
    satellites,
    ground_stations,
    horizon: float,
    iterations: int = 200,
    seed: Optional[int] = None,
):
    """Fonction pratique : lance l'ALNS et retourne plan, récompense, historique."""
    solver = ALNS(
        requests,
        satellites,
        ground_stations,
        horizon,
        iterations=iterations,
        seed=seed,
    )
    return solver.run()
