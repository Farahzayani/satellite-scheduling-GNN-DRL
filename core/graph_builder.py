"""

Construit une representation heterogene du systeme (torch_geometric.HeteroData)
a partir de l'etat COURANT du simulateur, pour alimenter l'encodeur GNN.


Types de noeuds : 'request', 'satellite', 'station'
Types d'arete :
    ('request', 'visible_by', 'satellite')       -- opportunite d'acquisition encore valide
    ('satellite', 'in_contact_with', 'station')    -- fenetre de contact encore valide
    ('request', 'acquired_by', 'satellite')         -- donnee deja acquise, en attente de downlink
    (+ les 3 aretes inverses, pour un message passing bidirectionnel)

Choix de conception important : les fenetres temporelles (visibilite,
contact) sont des FEATURES D'ARETE, pas des features de noeud
"""

import torch
from torch_geometric.data import HeteroData

# constantes de normalisation (a ajuster si tes instances changent beaucoup d'echelle)
NORM = {
    "priority": 10.0,
    "volume": 5.0,
    "duration": 5.0,
    "memory": 60.0,
    "angle": 90.0,
    "count": 5.0,
}

NODE_FEATURE_DIMS = {"request": 8, "satellite": 4, "station": 3}
EDGE_FEATURE_DIMS = {
    ("request", "visible_by", "satellite"): 3,
    ("satellite", "in_contact_with", "station"): 2,
    ("request", "acquired_by", "satellite"): 2,
}


def _status_onehot(status: int):
    v = [0.0, 0.0, 0.0]
    v[status] = 1.0
    return v 


def build_hetero_graph(env) -> HeteroData:
    """Construit le graphe heterogene de l'etat courant de `env`
    (une instance de SatelliteSchedulingEnv)."""
    data = HeteroData()
    t = env.t
    horizon = env.horizon

    req_ids = list(env.requests.keys())
    sat_ids = list(env.satellites.keys())
    sta_ids = list(env.ground_stations.keys())
    req_idx = {rid: k for k, rid in enumerate(req_ids)}
    sat_idx = {sid: k for k, sid in enumerate(sat_ids)}
    sta_idx = {gid: k for k, gid in enumerate(sta_ids)} 

    # ---------------- NOEUDS ----------------
    req_feats = []
    for rid in req_ids:
        r = env.requests[rid]
        n_opps = sum(1 for o in r.opportunities if r.status == 0 and o.end > t)
        earliest_close = min([o.end for o in r.opportunities if o.end > t], default=horizon)
        urgency = max(0.0, (earliest_close - t) / max(horizon, 1.0))
        '''valeur élevée → la fenêtre est encore assez éloignée ;
            valeur faible → la fenêtre se termine bientôt → requête urgente.
            Elle donne donc au GNN une information temporelle supplémentaire pour l'aider à privilégier les requêtes 
            qui risquent de devenir rapidement impossibles à acquérir.'''
        
        req_feats.append([
            r.priority / NORM["priority"],
            r.volume / NORM["volume"],
            r.duration / NORM["duration"],
            *_status_onehot(r.status),
            min(n_opps, NORM["count"]) / NORM["count"],
            urgency,
        ])
    data["request"].x = (torch.tensor(req_feats, dtype=torch.float) if req_feats
                          else torch.zeros((0, NODE_FEATURE_DIMS["request"])))
    data["request"].request_id = req_ids   # pour retrouver l'id original apres passage dans le GNN

    sat_feats = []
    for sid in sat_ids:
        s = env.satellites[sid]
        util = s.memory_used / s.memory_capacity if s.memory_capacity else 0.0
        busy_for = max(0.0, s.available_time - t) / max(horizon, 1.0) 
        """Indique dans combien de temps le satellite sera libre. Une valeur faible signifie qu'il est bientôt disponible ; 
        une valeur élevée signifie qu'il est encore occupé""" 
        sat_feats.append([
            util,
            s.memory_capacity / NORM["memory"],
            busy_for,
            min(len(s.pending_downlink), NORM["count"]) / NORM["count"],
        ])
    data["satellite"].x = (torch.tensor(sat_feats, dtype=torch.float) if sat_feats
                            else torch.zeros((0, NODE_FEATURE_DIMS["satellite"])))
    data["satellite"].satellite_id = sat_ids

    sta_feats = []
    for gid in sta_ids:
        g = env.ground_stations[gid]
        upcoming = [c for c in g.contacts if c.end > t]
        next_open = min([max(0.0, c.start - t) for c in upcoming], default=horizon) / max(horizon, 1.0)
        sta_feats.append([
            g.datarate,
            next_open,
            min(len(upcoming), NORM["count"]) / NORM["count"],
        ])
    data["station"].x = (torch.tensor(sta_feats, dtype=torch.float) if sta_feats
                          else torch.zeros((0, NODE_FEATURE_DIMS["station"])))
    data["station"].station_id = sta_ids

    # ---------------- ARETES : visibilite (request -> satellite) ----------------
    src, dst, feats = [], [], []
    for rid in req_ids:
        r = env.requests[rid]
        if r.status != 0:
            continue
        for o in r.opportunities:
            if o.end <= t or o.satellite_id not in sat_idx:
                continue
            src.append(req_idx[rid]); dst.append(sat_idx[o.satellite_id])
            feats.append([
                max(0.0, o.start - t) / max(horizon, 1.0),
                max(0.0, o.end - max(t, o.start)) / max(horizon, 1.0),
                o.pointing_angle / NORM["angle"],
                
            ])
    _add_edge_type(data, "request", "visible_by", "satellite", src, dst, feats,
                    EDGE_FEATURE_DIMS[("request", "visible_by", "satellite")])

    # ---------------- ARETES : contact (satellite -> station) ----------------
    src, dst, feats = [], [], []
    for gid in sta_ids:
        g = env.ground_stations[gid]
        for c in g.contacts:
            if c.end <= t or c.satellite_id not in sat_idx:
                continue
            src.append(sat_idx[c.satellite_id]); dst.append(sta_idx[gid])
            feats.append([
                max(0.0, c.start - t) / max(horizon, 1.0),
                max(0.0, c.end - max(t, c.start)) / max(horizon, 1.0),
            ])
    _add_edge_type(data, "satellite", "in_contact_with", "station", src, dst, feats,
                    EDGE_FEATURE_DIMS[("satellite", "in_contact_with", "station")])

    # ---------------- ARETES : deja acquise, en attente de downlink ----------------
    src, dst, feats = [], [], []
    for rid in req_ids:
        r = env.requests[rid]
        if r.status == 1 and r.acquired_by in sat_idx:
            src.append(req_idx[rid]); dst.append(sat_idx[r.acquired_by])
            sat_cap = env.satellites[r.acquired_by].memory_capacity or 1.0
            feats.append([
                (t - r.t_end) / max(horizon, 1.0),
                r.volume / sat_cap,
            ])
    _add_edge_type(data, "request", "acquired_by", "satellite", src, dst, feats,
                    EDGE_FEATURE_DIMS[("request", "acquired_by", "satellite")])

    return data


def _add_edge_type(data, src_type, rel, dst_type, src, dst, feats, n_feat):
    """Ajoute une arete + sa version inverse (message passing bidirectionnel)."""
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(feats, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, n_feat), dtype=torch.float)
    data[src_type, rel, dst_type].edge_index = edge_index
    data[src_type, rel, dst_type].edge_attr = edge_attr
    # sens inverse, memes features (necessaire pour que l'info circule dans les 2 sens)
    data[dst_type, f"rev_{rel}", src_type].edge_index = edge_index.flip(0)
    data[dst_type, f"rev_{rel}", src_type].edge_attr = edge_attr


#NB :
#GNN peut faire circuler l'information dans les deux directions.
#Ce n'est pas parce que l'action physique est bidirectionnelle, C'est uniquement pour permettre au message passing du GNN 
#de propager l'information dans les deux sens.