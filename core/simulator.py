"""
Implemente la fonction de transition T(s_t, a_t) -> s_{t+1} et la fonction de
recompense r_t = alpha1 * P_i - alpha2 * w_i - alpha3 * E 


Le downlink est traite AUTOMATIQUEMENT par le simulateur des qu'une requete est
acquise, en balayant les fenetres de contact disponibles et en priorisant les
requetes par P_i decroissant quand la capacite de la fenetre est insuffisante

reproduit la dynamique du système: 
state
   ↓
action
   ↓
transition
   ↓
reward
   ↓
new state
"""

import copy
import random
from typing import Dict, List, Tuple

from core.entities import Request, Satellite, GroundStation


def transition_time(sat: Satellite, angle_from: float, angle_to: float) -> float:
    """tau^trans_s(i,j) = |theta_i - theta_j| / v^rot_s + t^sta_s  (depointage)."""
    return abs(angle_to - angle_from) / sat.rotation_speed + sat.stabilization_time 


class SatelliteSchedulingEnv:

    def __init__(self, requests: List[Request], satellites: List[Satellite],
                 ground_stations: List[GroundStation], horizon: float,
                 weights: Tuple[float, float, float] = (1.0, 0.1, 0.05),
                 max_steps: int = 1000, seed: int = None):
        self.requests_init = requests
        self.satellites_init = satellites
        self.ground_stations_init = ground_stations
        self.horizon = horizon
        self.alpha1, self.alpha2, self.alpha3 = weights
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.reset()

    # Initialisation
    def reset(self):
        self.requests: Dict[int, Request] = {r.id: copy.deepcopy(r) for r in self.requests_init}
        self.satellites: Dict[int, Satellite] = {s.id: copy.deepcopy(s) for s in self.satellites_init}
        self.ground_stations: Dict[int, GroundStation] = {g.id: copy.deepcopy(g) for g in self.ground_stations_init}

        # curseur d'occupation (temps deja utilise) pour chaque fenetre de contact,
        # permet un remplissage progressif par plusieurs requetes successives
        self._contact_cursor: Dict[Tuple[int, int, int], float] = {}
        for g in self.ground_stations.values():
            for c in g.contacts:
                self._contact_cursor[(g.id, c.satellite_id, c.contact_index)] = c.start

        self.t = 0.0
        self.step_count = 0
        self.plan: List[dict] = []
        self.total_reward = 0.0
        self.done = False
        return self.get_state()

    # Etat (s_t)
    def get_state(self) -> dict:
        #l'etat s_t = (R_t, Sat_t, G_t)
        return {
            "t": self.t,
            "requests": {
                rid: {"priority": r.priority, "volume": r.volume, "duration": r.duration,
                      "status": r.status}
                for rid, r in self.requests.items()
            },
            "satellites": {
                sid: {"memory_used": s.memory_used, "memory_capacity": s.memory_capacity,
                      "available_time": s.available_time}
                for sid, s in self.satellites.items()
            },
        }

    # Actions valides (= masque d'action pour le futur agent PPO)
    def _earliest_start(self, sat: Satellite, opp) -> float:
        slew = transition_time(sat, sat.current_pointing, opp.pointing_angle)
        return max(opp.start, sat.available_time + slew)

    def get_valid_actions(self) -> List[tuple]:
        actions = []
        for r in self.requests.values():
            if r.status != 0:
                continue
            for o_idx, opp in enumerate(r.opportunities):
                sat = self.satellites[opp.satellite_id]
                t_start = self._earliest_start(sat, opp)
                t_end = t_start + r.duration
                if t_end > opp.end:
                    continue                                     # ne rentre pas dans la fenetre de visibilite
                if sat.memory_used + r.volume > sat.memory_capacity:
                    continue                                      # depasse M_s
                if t_start > self.horizon:
                    continue
                actions.append(("acquire", r.id, opp.satellite_id, o_idx))
        return actions

    # Transition T(s_t, a_t) -> s_{t+1}
    def step(self, action: tuple):
        assert not self.done, "Episode termine : appelle reset()."
        self.step_count += 1
        reward = 0.0

        if action[0] == "wait":
            pass
        elif action[0] == "acquire":
            _, rid, sid, o_idx = action
            reward += self._do_acquire(rid, sid, o_idx)    #alpha1 (obj1)
        else:
            raise ValueError(f"Action inconnue : {action}")

        reward += self._process_downlinks()                #alpha2 (obj2)

        self.done = self._check_done()
        if self.done:
            reward += self._terminal_balance_penalty()     #alpha3 (obj3)

        self.total_reward += reward
        return self.get_state(), reward, self.done, {}

    def _do_acquire(self, rid: int, sid: int, o_idx: int) -> float:
        r = self.requests[rid]
        sat = self.satellites[sid]
        opp = r.opportunities[o_idx]

        t_start = self._earliest_start(sat, opp)
        t_end = t_start + r.duration

        if t_end > opp.end or sat.memory_used + r.volume > sat.memory_capacity:
            return -1.0     # action invalide 
        
        sat.available_time = t_end
        sat.current_pointing = opp.pointing_angle
        self.t = max(self.t, t_end)

        r.status = 1
        r.acquired_by = sid
        r.t_start, r.t_end = t_start, t_end
        sat.memory_used += r.volume
        sat.pending_downlink.append(rid)
        self.plan.append({"type": "acquire", "request": rid, "satellite": sid,
                           "t_start": t_start, "t_end": t_end})
        return self.alpha1 * r.priority

    def _process_downlinks(self) -> float:
        """Ecoule les requetes acquises vers les stations sol des qu'une fenetre
        de contact compatible existe, en priorisant P_i decroissant si la
        capacite manque"""
        reward = 0.0
        for g in self.ground_stations.values():
            for c in g.contacts:
                sat = self.satellites.get(c.satellite_id)
                if sat is None:
                    continue
                key = (g.id, c.satellite_id, c.contact_index)
                cursor = self._contact_cursor[key]
                if cursor >= c.end or not sat.pending_downlink:
                    continue

                candidates = sorted(
                    [rid for rid in sat.pending_downlink if self.requests[rid].t_end <= c.end],
                    key=lambda rid: -self.requests[rid].priority,
                )
                for rid in candidates:
                    r = self.requests[rid]
                    dl_start = max(cursor, r.t_end)
                    dl_duration = r.volume / g.datarate               
                    dl_end = dl_start + dl_duration
                    if dl_end > c.end:
                        continue                                       # ne rentre plus dans cette fenetre -> essaie la suivante

                    r.status = 2
                    r.t_downlink = dl_end
                    r.downlinked_via = g.id
                    sat.memory_used -= r.volume                           
                    sat.pending_downlink.remove(rid)
                    cursor = dl_end

                    w_i = r.t_downlink - r.t_end                             # attente + V_i/B_g
                    reward += -self.alpha2 * w_i
                    self.plan.append({"type": "downlink", "request": rid, "station": g.id,
                                       "t_start": dl_start, "t_end": dl_end, "delay": w_i})
                self._contact_cursor[key] = cursor
        return reward

    def _terminal_balance_penalty(self) -> float:
        """E = (1/S) * sum_s (L_s - L_barre)^2, avec L_s = charge acquise par s."""
        loads = []
        for sid in self.satellites:
            load = sum(r.volume for r in self.requests.values()
                       if r.acquired_by == sid and r.status >= 1)
            loads.append(load)
        if not loads:
            return 0.0
        mean_load = sum(loads) / len(loads)
        E = sum((l - mean_load) ** 2 for l in loads) / len(loads)
        return -self.alpha3 * E

    def _check_done(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if self.get_valid_actions():
            return False
        pending = any(sat.pending_downlink for sat in self.satellites.values())
        return not pending

    def get_plan(self) -> List[dict]:
        """Journal complet des decisions (acquisitions + downlinks), pretes
        pour un affichage type Gantt plus tard."""
        return self.plan