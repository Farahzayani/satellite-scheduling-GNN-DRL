"""
Valide le simulateur face a un solveur MILP EXACT (OR-Tools CP-SAT), en
reprenant les contraintes mathématiques

Deux tests :
  TEST 1 - Coherence : resolution EXACTE (CP-SAT) puis rejeu de la solution
  optimale dans le simulateur Python -> si la recompense correspond a
  l'objectif de CP-SAT, les deux modeles sont coherents entre eux.
  TEST 2 - Ecart des heuristiques (glouton, FCFS) par rapport a cet optimum.

Notes:
    - Equilibrage de charge E : le MILP le definit comme une VARIANCE (terme
    quadratique), difficile a passer directement a CP-SAT sans risquer des
    soucis numeriques. On optimise ici un PROXY LINEAIRE equivalent en
    pratique (ecart max-min de charge entre satellites), puis on recalcule
    la VRAIE variance sur la solution trouvee pour un objectif final
    directement comparable a celui du simulateur.
    """

from ortools.sat.python import cp_model
import math 

from core.instance_generator import generate_instance
from core.simulator import SatelliteSchedulingEnv
from heuristics.baselines import greedy_priority_policy, fcfs_policy, run_episode

# ============ PARAMETRES A MODIFIER LIBREMENT ============
SEED = 7
N_REQUESTS = 12
N_SATELLITES = 2
N_GROUND_STATIONS = 2
HORIZON = 600.0
WEIGHTS = (1.0, 0.05, 0.3)      # (alpha1 profit, alpha2 delai, alpha3 equilibrage)
# ===========================================================

TIME_SCALE = 10     # resolution 1/TIME_SCALE min pour les temps
VOL_SCALE = 10        # resolution 1/VOL_SCALE Go pour les volumes/memoire


def _transition_ticks(sat, opp_i, opp_j):
    minutes = abs(opp_j.pointing_angle - opp_i.pointing_angle) / sat.rotation_speed + sat.stabilization_time
    return max(0, round(minutes * TIME_SCALE))
def _transition_ticks_from_rest(sat, opp):
    """Temps de depointage depuis l'angle de repos initial (0 degre) vers la
    premiere opportunite choisie -- symetrique a _earliest_start() du simulateur."""
    minutes = abs(opp.pointing_angle - 0.0) / sat.rotation_speed + sat.stabilization_time
    return max(0, math.ceil(minutes * TIME_SCALE))


def solve_exact(requests, satellites, ground_stations, horizon, weights):
    alpha1, alpha2, alpha3 = weights
    requests_by_id = {r.id: r for r in requests}
    sats_by_id = {s.id: s for s in satellites}
    BIG_TIME = round(horizon * TIME_SCALE) + 10 * TIME_SCALE

    model = cp_model.CpModel()

    x, start_v, end_v, interval_v = {}, {}, {}, {}
    acq_end, acquired = {}, {}
    intervals_by_satellite = {sid: [] for sid in sats_by_id}

    # --- variables d'acquisition (contrainte 1 : unicite, 2 : fenetres) ---
    for r in requests:
        i = r.id
        acq_end[i] = model.NewIntVar(0, BIG_TIME, f"acq_end_{i}")
        opts = []
        for o_idx, opp in enumerate(r.opportunities):
            if opp.satellite_id not in sats_by_id:
                continue
            sat = sats_by_id[opp.satellite_id]
            vts, vte = math.ceil(opp.start * TIME_SCALE), math.floor(opp.end * TIME_SCALE)
            dur = max(1, math.ceil(r.duration * TIME_SCALE))
            if vte - vts < dur:
                continue
            xv = model.NewBoolVar(f"x_{i}_{o_idx}")
            s_i = model.NewIntVar(vts, vte, f"start_{i}_{o_idx}")
            e_i = model.NewIntVar(vts, vte, f"end_{i}_{o_idx}")
            model.Add(e_i == s_i + dur)
            model.Add(e_i <= vte)
            ivar = model.NewOptionalIntervalVar(s_i, dur, e_i, xv, f"acq_{i}_{o_idx}")
            x[(i, o_idx)] = xv
            start_v[(i, o_idx)], end_v[(i, o_idx)], interval_v[(i, o_idx)] = s_i, e_i, ivar
            intervals_by_satellite[opp.satellite_id].append(((i, o_idx), ivar))
            model.Add(start_v[(i, o_idx)] >= _transition_ticks_from_rest(sat, opp)).OnlyEnforceIf(xv)
            model.Add(acq_end[i] == e_i).OnlyEnforceIf(xv)
            opts.append(xv)
        acquired[i] = model.NewBoolVar(f"acquired_{i}")
        if opts:
            model.Add(sum(opts) <= 1)
            model.Add(acquired[i] == sum(opts))
        else:
            model.Add(acquired[i] == 0)

    # --- contraintes 3/4 (enchainement + depointage) et exclusion mutuelle, PAR SATELLITE ---
    for sid, entries in intervals_by_satellite.items():
        if not entries:
            continue
        model.AddNoOverlap([ivar for _, ivar in entries])
        sat = sats_by_id[sid]
        for a in range(len(entries)):
            (i, oi), _ = entries[a]
            opp_i = requests_by_id[i].opportunities[oi]
            for b in range(a + 1, len(entries)):
                (j, oj), _ = entries[b]
                opp_j = requests_by_id[j].opportunities[oj]
                trans_ij = _transition_ticks(sat, opp_i, opp_j)
                trans_ji = _transition_ticks(sat, opp_j, opp_i)
                order = model.NewBoolVar(f"order_{i}_{oi}_{j}_{oj}")
                model.Add(start_v[(j, oj)] >= end_v[(i, oi)] + trans_ij).OnlyEnforceIf(
                    [order, x[(i, oi)], x[(j, oj)]])
                model.Add(start_v[(i, oi)] >= end_v[(j, oj)] + trans_ji).OnlyEnforceIf(
                    [order.Not(), x[(i, oi)], x[(j, oj)]])

    # --- contrainte 5 : memoire, PAR SATELLITE ---
    for sid, sat in sats_by_id.items():
        terms = [max(1, round(requests_by_id[i].volume * VOL_SCALE)) * xv
                 for (i, o_idx), xv in x.items()
                 if requests_by_id[i].opportunities[o_idx].satellite_id == sid]
        if terms:
            model.Add(sum(terms) <= round(sat.memory_capacity * VOL_SCALE))

    # --- downlink : contraintes 6 (unicite) et 7 (capacite de contact) ---
    delta, dl_start_v, dl_end_v = {}, {}, {}
    by_window = {}
    for r in requests:
        i = r.id
        my_sat_ids = {requests_by_id[i].opportunities[o_idx].satellite_id
                      for (ii, o_idx) in x if ii == i}
        entries_i = []
        for g in ground_stations:
            for k, c in enumerate(g.contacts):
                if c.satellite_id not in my_sat_ids:
                    continue
                a_k, b_k = round(c.start * TIME_SCALE), round(c.end * TIME_SCALE)
                dur_dl = max(1, round((r.volume / g.datarate) * TIME_SCALE))
                if b_k - a_k < dur_dl:
                    continue
                key = (i, g.id, k)
                d_ik = model.NewBoolVar(f"delta_{i}_{g.id}_{k}")
                s_ik = model.NewIntVar(a_k, b_k, f"dls_{i}_{g.id}_{k}")
                e_ik = model.NewIntVar(a_k, b_k, f"dle_{i}_{g.id}_{k}")
                model.Add(e_ik == s_ik + dur_dl)
                model.Add(s_ik >= acq_end[i]).OnlyEnforceIf(d_ik)
                ivar = model.NewOptionalIntervalVar(s_ik, dur_dl, e_ik, d_ik, f"dlint_{i}_{g.id}_{k}")
                # ce downlink n'est valide que si l'opportunite choisie utilise CE satellite
                matching_x = [x[(i, o_idx)] for (ii, o_idx) in x if ii == i
                              and requests_by_id[i].opportunities[o_idx].satellite_id == c.satellite_id]
                if matching_x:
                    model.Add(d_ik <= sum(matching_x))
                delta[key] = d_ik
                dl_start_v[key], dl_end_v[key] = s_ik, e_ik
                entries_i.append(d_ik)
                by_window.setdefault((g.id, k), []).append(ivar)
        model.Add(sum(entries_i) == acquired[i])                 # contrainte 6

    for ivals in by_window.values():
        model.AddNoOverlap(ivals)                                    # contrainte 7 (partage temporel du canal)

    # --- delai de restitution w_i ---
    delay = {}
    for r in requests:
        i = r.id
        delay[i] = model.NewIntVar(0, BIG_TIME, f"delay_{i}")
        my_deltas = [(key, d) for key, d in delta.items() if key[0] == i]
        if my_deltas:
            chosen_end = model.NewIntVar(0, BIG_TIME, f"chosen_end_{i}")
            for key, d in my_deltas:
                model.Add(chosen_end == dl_end_v[key]).OnlyEnforceIf(d)
            model.Add(delay[i] == chosen_end - acq_end[i]).OnlyEnforceIf(acquired[i])
        model.Add(delay[i] == 0).OnlyEnforceIf(acquired[i].Not())

    # --- equilibrage de charge : proxy LINEAIRE (ecart max-min), vraie variance recalculee apres coup ---
    loads = []
    for sid in sats_by_id:
        terms = [max(1, round(requests_by_id[i].volume * VOL_SCALE)) * xv
                 for (i, o_idx), xv in x.items()
                 if requests_by_id[i].opportunities[o_idx].satellite_id == sid]
        load_s = model.NewIntVar(0, 10 ** 7, f"load_{sid}")
        model.Add(load_s == (sum(terms) if terms else 0))
        loads.append(load_s)
    max_load = model.NewIntVar(0, 10 ** 7, "max_load")
    min_load = model.NewIntVar(0, 10 ** 7, "min_load")
    model.AddMaxEquality(max_load, loads)
    model.AddMinEquality(min_load, loads)

    # --- objectif ---
    OBJ_SCALE = TIME_SCALE * 1000
    w1 = round(alpha1 * OBJ_SCALE)
    w2 = round(alpha2 * OBJ_SCALE / TIME_SCALE)
    w3 = round(alpha3 * OBJ_SCALE / VOL_SCALE)
    priorities = {r.id: r.priority for r in requests}
    objective = (sum(w1 * priorities[i] * acquired[i] for i in acquired)
                 - sum(w2 * delay[i] for i in delay)
                 - w3 * (max_load - min_load))
    model.Maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    result = {
        "status": solver.StatusName(status),                                   # "OPTIMAL" / "FEASIBLE" / ...
        "status_short": "O" if status == cp_model.OPTIMAL else ("F" if status == cp_model.FEASIBLE else "-"),
        "computing_time": solver.WallTime(),                                     # CT (secondes)
        "plan": [], "objective_real": None, "true_E": None,
    }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result

    for r in requests:
        i = r.id
        if not solver.Value(acquired[i]):
            continue
        chosen_o = next(o_idx for (ii, o_idx) in x if ii == i and solver.Value(x[(ii, o_idx)]))
        opp = requests_by_id[i].opportunities[chosen_o]
        dl_key = next((key for key in delta if key[0] == i and solver.Value(delta[key])), None)
        result["plan"].append({
            "request": i, "priority": r.priority, "satellite": opp.satellite_id, "opp_index": chosen_o,
            "t_start": solver.Value(start_v[(i, chosen_o)]) / TIME_SCALE,
            "t_end": solver.Value(end_v[(i, chosen_o)]) / TIME_SCALE,
            "downlink_station": dl_key[1] if dl_key else None,
            "downlink_contact": dl_key[2] if dl_key else None,
            "dl_start": solver.Value(dl_start_v[dl_key]) / TIME_SCALE if dl_key else None,
            "dl_end": solver.Value(dl_end_v[dl_key]) / TIME_SCALE if dl_key else None,
            "delay": solver.Value(delay[i]) / TIME_SCALE,
        })

    # vraie variance (quadratique) recalculee sur la solution trouvee -> comparable au simulateur
    loads_real = {sid: 0.0 for sid in sats_by_id}
    for e in result["plan"]:
        loads_real[e["satellite"]] += requests_by_id[e["request"]].volume
    mean_load = sum(loads_real.values()) / len(loads_real) if loads_real else 0.0
    true_E = sum((l - mean_load) ** 2 for l in loads_real.values()) / len(loads_real) if loads_real else 0.0
    result["true_E"] = true_E
    result["loads_by_satellite"] = loads_real

    profit = sum(e["priority"] for e in result["plan"])
    delay_sum = sum(e["delay"] for e in result["plan"])
    result["objective_real"] = alpha1 * profit - alpha2 * delay_sum - alpha3 * true_E

    # --- GAP ---
    if status == cp_model.FEASIBLE:
        bound = solver.BestObjectiveBound()
        obj_scaled = solver.ObjectiveValue()
        result["gap"] = abs(obj_scaled - bound) / (abs(bound) + 1e-8)                       # colonne "Gap"
    else:
        result["gap"] = 0.0
    return result


def replay_plan_in_simulator(env, cp_plan):
    """Rejoue la solution optimale de CP-SAT dans le simulateur, dans l'ordre
    chronologique, avec le MEME couple (requete, satellite, opportunite)."""
    env.reset()
    for entry in sorted(cp_plan, key=lambda e: e["t_start"]):
        action = ("acquire", entry["request"], entry["satellite"], entry["opp_index"])
        if action not in env.get_valid_actions():
            print(f"  [ATTENTION] action {action} non valide au moment de rejouer")
            continue
        env.step(action)
    env._process_downlinks()
    return env


if __name__ == "__main__":
    requests, satellites, ground_stations, horizon = generate_instance(
        n_requests=N_REQUESTS, n_satellites=N_SATELLITES,
        n_ground_stations=N_GROUND_STATIONS, horizon=HORIZON, seed=SEED)


    print(f"Instance : {N_SATELLITES} satellites, {N_GROUND_STATIONS} station(s), "
          f"{N_REQUESTS} requetes, horizon={HORIZON:.0f} min, seed={SEED}\n")

    print("=== TEST 1 : resolution exacte (CP-SAT) puis rejeu dans le simulateur ===")
    result = solve_exact(requests, satellites, ground_stations, horizon, WEIGHTS)
    print(f"Statut CP-SAT : {result['status']}")
    if not result["plan"] and result["status"] not in ("OPTIMAL", "FEASIBLE"):
        print("Pas de solution trouvee -- essaie d'augmenter MAX_SOLVE_SECONDS ou reduire N_REQUESTS.")
    else:
        print("\n--- Math Model ---")
        print(f"S (statut)        : {result['status_short']}  ({result['status']})")
        print(f"CT (temps calcul) : {result['computing_time']:.3f}s")
        print(f"Obj (pondere)     : {result['objective_real']:.2f} ")
        print(f"Gap               : {result['gap']*100:.2f}%  ")
        print(f"Charge par satellite : { {k: round(v,2) for k,v in result['loads_by_satellite'].items()} }")

        print("\nPlan optimal :")
        for e in result["plan"]:
            print(f"   requete {e['request']} (P={e['priority']}) -> satellite {e['satellite']} : "
                  f"acquise [{e['t_start']:.1f}-{e['t_end']:.1f}], "
                  f"downlink station {e['downlink_station']} [{e['dl_start']:.1f}-{e['dl_end']:.1f}], "
                  f"delai={e['delay']:.1f}")

        env = SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon, weights=WEIGHTS, seed=SEED)
        replay_plan_in_simulator(env, result["plan"])
        gap = abs(env.total_reward - result["objective_real"])
        pct = 100 * gap / result["objective_real"] if result["objective_real"] else 0
        print(f"\nRecompense du simulateur en rejouant ce plan : {env.total_reward:.2f}")
        print(f"Objectif CP-SAT (vraie variance)              : {result['objective_real']:.2f}")
        print(f"Ecart : {gap:.2f} ({pct:.1f}%) ")

        print("\n=== TEST 2 : ecart des heuristiques a l'optimum exact ===")
        for name, policy in [("Glouton (priorite)", greedy_priority_policy), ("FCFS", fcfs_policy)]:
            env_h = SatelliteSchedulingEnv(requests, satellites, ground_stations, horizon, weights=WEIGHTS, seed=SEED)
            run_episode(env_h, policy)
            gap_h = result["objective_real"] - env_h.total_reward
            pct_h = 100 * gap_h / result["objective_real"] if result["objective_real"] else 0
            print(f"  {name:22s} : recompense={env_h.total_reward:6.2f}   ecart={gap_h:6.2f}  ({pct_h:5.1f}%)")