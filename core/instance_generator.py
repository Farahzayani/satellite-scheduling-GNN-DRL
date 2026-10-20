"""
instance_generator.py
-----------------------
Genere des instances du probleme : requetes, satellites, stations sol, avec
leurs fenetres de visibilite et de contact (dynamique orbitale SIMPLIFIEE,
comme demande dans la tache 3 du sujet -- pas de vraie mecanique orbitale).

generate_toy_instance()  -> petite instance pour valider le simulateur
                             (recommande : 1 satellite, 1 station, 5-10 requetes)
generate_instance()      -> instance parametrable en taille (tache 8 du sujet :
                             generateur d'instances realistes petites/moyennes/grandes)
"""

import random
from core.entities import Request, Satellite, GroundStation, VisibilityWindow, ContactWindow


def _build_ground_stations(n_ground_stations, n_satellites, horizon, rng):
    ground_stations = []
    for g in range(n_ground_stations):
        contacts = []
        t = 0.0
        k = 0
        # passages periodiques simplifies : ~toutes les 80-100 min, duree 6-10 min, par satellite
        while t < horizon:
            for s in range(n_satellites):
                start = t + rng.uniform(0, 5)
                end = start + rng.uniform(6, 10)
                contacts.append(ContactWindow(satellite_id=s, ground_station_id=g,
                                               contact_index=k, start=start, end=end))
                k += 1
            t += rng.uniform(80, 100)
        ground_stations.append(GroundStation(id=g, datarate=rng.uniform(0.5, 1.2), contacts=contacts))
    return ground_stations


def _build_requests(n_requests, n_satellites, horizon, rng, max_opps_per_request=2):
    requests = []
    for i in range(n_requests):
        priority = rng.choice([1, 2, 3, 5, 8])       # echelle de priorite type "client"
        volume = rng.uniform(1.0, 4.0)                  # Go
        duration = rng.uniform(0.5, 2.0)                  # min

        opportunities = []
        n_opps = rng.randint(1, min(max_opps_per_request, n_satellites) if n_satellites > 1 else 1)
        chosen_sats = rng.sample(range(n_satellites), k=min(n_opps, n_satellites))
        for o, s in enumerate(chosen_sats):
            win_start = rng.uniform(0, max(horizon - 30, 1))
            win_len = rng.uniform(duration + 1, duration + 6)
            opportunities.append(VisibilityWindow(
                request_id=i, satellite_id=s, opp_index=o,
                start=win_start, end=win_start + win_len,
                pointing_angle=rng.uniform(-45, 45),
            ))
        requests.append(Request(id=i, priority=priority, volume=volume, duration=duration,
                                 opportunities=opportunities))
    return requests


# tester rapidement en ligne de commande que le simulateur fonctionne.

def generate_instance(n_requests=8, n_satellites=2, n_ground_stations=2,
                           horizon=600.0, seed=None): #valeurs de secours pour un test rapide en ligne de commande

    seed = seed if seed is not None else random.randint(0, 10_000)
    rng = random.Random(seed)
    satellites = [Satellite(id=s, memory_capacity=rng.uniform(30, 60),
                                 rotation_speed=2.0, stabilization_time=0.3)
                      for s in range(n_satellites)]
    ground_stations = _build_ground_stations(n_ground_stations, n_satellites, horizon, rng)
    requests = _build_requests(n_requests, n_satellites, horizon, rng)
    return requests, satellites, ground_stations, horizon