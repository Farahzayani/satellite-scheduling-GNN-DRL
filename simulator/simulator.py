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

        self._satellite_busy = {s.id: [] for s in self.satellites.values()}
        self._station_busy = {g.id: [] for g in self.ground_stations.values()}

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
    def _earliest_slot(self, g_id, earliest, duration, window_end):
    #"""Premier creneau [start, start+duration] avec start >= earliest, libre sur la
    #station g_id (aucun chevauchement avec les telechargements deja planifies, quel
    #que soit le satellite) et contenu dans la fenetre de contact (fin <= window_end).
    #Renvoie start, ou None si aucun creneau ne convient."""
        start = earliest
        for (b0, b1) in sorted(self._station_busy[g_id]):
            if start + duration <= b0:
                break                         # le creneau tient avant cette plage occupee
            if start < b1:
                start = max(start, b1)        # chevauchement : repousser apres la plage
        return start if start + duration <= window_end else None
    def _earliest_satellite_slot(self, sid, earliest, duration, window_end):
    #Recherche le premier créneau libre sur le satellite sid
        start = earliest

        for (b0, b1) in sorted(self._satellite_busy[sid]):

            if start + duration <= b0:
                break

            if start < b1:
                start = max(start, b1)

        return start if start + duration <= window_end + 1e-6 else None

    def _earliest_common_slot(self, sid, gid, earliest, duration, window_end):
        """Recherche un créneau simultanément libre
        sur le satellite et la station sol."""

        start = earliest
        eps = 1e-9

        max_iterations = (
            len(self._station_busy[gid])
            + len(self._satellite_busy[sid])
            + 2
        )

        for _ in range(max_iterations):

            # Disponibilité de la station
            station_start = self._earliest_slot(
                gid, start, duration, window_end
            )

            if station_start is None:
                return None

            # Disponibilité du satellite
            satellite_start = self._earliest_satellite_slot(
                sid, station_start, duration, window_end
            )

            if satellite_start is None:
                return None

            # Les deux ressources sont libres au même instant
            if abs(station_start - satellite_start) < eps:
                return station_start

            # Rechercher un nouveau créneau commun
            start = max(station_start, satellite_start)

        return None

    def _memory_at(self, sat, t):
    #Volume de données présentes à bord à l'instant t
        used = 0.0

        for r in self.requests.values():
            if (
                r.acquired_by == sat.id
                and r.status >= 1
                and r.t_end is not None
                and r.t_end <= t
            ):
                if r.t_downlink is None or r.t_downlink > t:
                    used += r.volume

        return used

    # Actions valides 
    def _earliest_start(self, sat: Satellite, opp,duration: float) -> float:
        slew = transition_time(sat, sat.current_pointing, opp.pointing_angle)
        earliest = max( opp.start, sat.available_time + slew, self.t)
        return self._earliest_satellite_slot( sat.id, earliest, duration, opp.end)

    
    def get_valid_actions(self) -> List[tuple]:
        actions = []
        for r in self.requests.values():
            if r.status != 0:
                continue
            for o_idx, opp in enumerate(r.opportunities):
                sat = self.satellites[opp.satellite_id]
                t_start = self._earliest_start(sat, opp,r.duration)
                if t_start is None:
                    continue
                t_end = t_start + r.duration

                if t_end > opp.end + 1e-6:
                    continue                                     # ne rentre pas dans la fenetre de visibilite
                if self._memory_at(sat, t_start) + r.volume > sat.memory_capacity:
                    continue                                      # depasse M_s
                if t_start > self.horizon:
                    continue
                actions.append(("acquire", r.id, opp.satellite_id, o_idx))
        return actions


    def _refresh_resources(self):
        #met à jour les statuts lorsque le temps atteint réellement la fin des téléchargements.
        for sat in self.satellites.values():

            for rid in list(sat.pending_downlink):

                r = self.requests[rid]

                if (
                    r.t_downlink is not None
                    and r.t_downlink <= self.t
                ):
                    r.status = 2
                    sat.pending_downlink.remove(rid)

        for sat in self.satellites.values():
            sat.memory_used = self._memory_at(sat, self.t)
    # Transition T(s_t, a_t) -> s_{t+1}
    def step(self, action: tuple):
        assert not self.done, "Episode termine : appelle reset()."

        self.step_count += 1
        reward = 0.0
        if action[0] == "wait":
            # Rechercher les fins de telechargements programmees
            future_ends = [
                self.requests[rid].t_downlink
                for sat in self.satellites.values()
                for rid in sat.pending_downlink
                if (
                    self.requests[rid].t_downlink is not None
                    and self.requests[rid].t_downlink > self.t
                )
            ]
           # Si aucun telechargement n'est encore programme,
            # essayer de programmer les telechargements en attente
            if not future_ends:
                reward += self._process_downlinks()
                future_ends = [
                    self.requests[rid].t_downlink
                    for sat in self.satellites.values()
                    for rid in sat.pending_downlink
                    if (
                        self.requests[rid].t_downlink is not None
                        and self.requests[rid].t_downlink > self.t
                    )
                ]
            # Avancer jusqu'a la prochaine fin de telechargement
            if future_ends:
                self.t = min(min(future_ends), self.horizon)
        elif action[0] == "acquire":

            _, rid, sid, o_idx = action
            reward += self._do_acquire(rid, sid, o_idx)

        else:
            raise ValueError(f"Action inconnue : {action}")
        # Actualiser les statuts et la memoire
        self._refresh_resources()
        # Programmer les nouveaux telechargements possibles
        reward += self._process_downlinks()

        # Actualiser une nouvelle fois les ressources
        self._refresh_resources()
        self.done = self._check_done()
        if self.done:
            reward += self._terminal_balance_penalty()
        self.total_reward += reward
        return self.get_state(), reward, self.done, {}

    def _do_acquire(self, rid: int, sid: int, o_idx: int) -> float:
        r = self.requests[rid]
        sat = self.satellites[sid]
        opp = r.opportunities[o_idx]

        t_start = self._earliest_start(sat, opp,r.duration)
        if t_start is None:
            return -1.0
        t_end = t_start + r.duration

        if  t_end > opp.end + 1e-6 or self._memory_at(sat, t_start) + r.volume > sat.memory_capacity:
            return -1.0     # action invalide 
        
        sat.available_time = t_end
        sat.current_pointing = opp.pointing_angle
        self._satellite_busy[sid].append((t_start, t_end))
        self.t = max(self.t, t_end)

        r.status = 1
        r.acquired_by = sid
        r.t_start, r.t_end = t_start, t_end
        #sat.memory_used += r.volume
        sat.pending_downlink.append(rid)
        self.plan.append({"type": "acquire", "request": rid, "satellite": sid,
                           "t_start": t_start, "t_end": t_end})
        return self.alpha1 * r.priority

    def _process_downlinks(self) -> float:
        reward = 0.0
        for g in self.ground_stations.values():
            for c in sorted(g.contacts, key=lambda c: c.start):
                sat = self.satellites.get(c.satellite_id)
                if sat is None or not sat.pending_downlink:
                    continue
    
                candidates = sorted(
                    [rid for rid in sat.pending_downlink if (self.requests[rid].t_end <= c.end 
                                                             and self.requests[rid].t_downlink is None)],
                    key=lambda rid: -self.requests[rid].priority,
                )
                for rid in candidates:
                    r = self.requests[rid]
                    dl_duration = r.volume / g.datarate
                    dl_start = self._earliest_common_slot( sid=sat.id,gid=g.id,earliest=max(self.t, c.start, r.t_end),duration=dl_duration,window_end=c.end)
                    if dl_start is None:
                        continue              # ne tient pas dans cette fenetre -> essaie une suivante
                    dl_end = dl_start + dl_duration
    
                    #r.status = 2
                    r.t_downlink = dl_end
                    r.downlinked_via = g.id
                    self._station_busy[g.id].append((dl_start, dl_end))
                    self._satellite_busy[sat.id].append((dl_start, dl_end))
                    # NB : la memoire n'est PLUS liberee ici ; elle l'est "virtuellement"
                    #      a l'instant dl_end, via _memory_at().
    
                    w_i = r.t_downlink - r.t_end
                    reward += -self.alpha2 * r.priority * w_i    # adapter si Z2 n'est pas encore corrige
                    self.plan.append({"type": "downlink", "request": rid, "station": g.id,
                                    "t_start": dl_start, "t_end": dl_end, "delay": w_i})
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