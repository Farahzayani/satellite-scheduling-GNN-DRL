"""
Encodeur GNN heterogene (GAT) : transforme le graphe construit par
graph_builder.py en embeddings de noeuds + un embedding global du graphe.

Architecture en 3 etapes :

  1. EXTRACTION : projection lineaire par type de noeud (h_i' = W.h_i), puis
     N couches de message passing heterogene. Chaque type d'arete utilise un
     GATv2Conv "edge-aware" (les features d'arete, ex. la fenetre de
     visibilite, influencent directement le score d'attention -- important
     ici puisque ces fenetres sont sur les aretes, pas sur les noeuds).

  2. AGREGATION : embedding global du graphe = concatenation des MOYENNES
     par type de noeud (request/satellite/station).
     le probleme doit generaliser a un nombre VARIABLE de satellites et de stations selon
     l'instance -> on utilise donc un pooling (mean)
     pour LES TROIS types de noeuds, pas seulement les requetes, afin que
     l'embedding du graphe garde une dimension FIXE quelle que soit la
     taille de l'instance. C'est un point a bien justifier dans la memoire.

  3. ESTIMATION (apercu) : ActionScorer illustre comment noter une action
     candidate (requete i, satellite s) a partir de ses embeddings + de
     l'embedding global. La tete complete (Q-value pour DQN, ou logits de
     politique pour PPO) sera batie a l'etape suivante (agent DRL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv

from core.graph_builder import NODE_FEATURE_DIMS, EDGE_FEATURE_DIMS

# construit automatiquement la liste des types d'arete (directs + inverses)
# a partir de graph_builder, pour ne jamais desynchroniser les deux fichiers
EDGE_TYPES = []
EDGE_DIMS = {}
for (s, rel, d), dim in EDGE_FEATURE_DIMS.items():
    EDGE_TYPES += [(s, rel, d), (d, f"rev_{rel}", s)]
    EDGE_DIMS[(s, rel, d)] = dim
    EDGE_DIMS[(d, f"rev_{rel}", s)] = dim


class HeteroGATEncoder(nn.Module):
    def __init__(self, hidden_dim=32, heads=4, n_layers=2):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim doit etre divisible par heads"
        self.hidden_dim = hidden_dim

        # --- etape 1a : projection lineaire par type de noeud ---
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(dim, hidden_dim) for ntype, dim in NODE_FEATURE_DIMS.items()
        })
        #( nn.Linear(8, 32) , nn.Linear(4, 32) , nn.Linear(3, 32) )

        # --- etape 1b : construction des couches de message passing heterogene (GATv2, edge-aware) ---
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = HeteroConv({
                et: GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                               edge_dim=EDGE_DIMS[et], add_self_loops=False)  #edge_dim :permet au GAT de prendre en compte les features d'arête parce que les fenêtres de visibilité/contact sont naturellement des relations entre objets
                for et in EDGE_TYPES
            }, aggr="mean")
            self.convs.append(conv)

    def forward(self, data):   #reçoit le graphe construit par graph_builder.py 
        """Renvoie un dict {type_de_noeud: embeddings [n_noeuds, hidden_dim]}."""
        #transformation des features ( avant: req N*8 ==> aprés : req N*32 )
        x_dict = {ntype: self.input_proj[ntype](data[ntype].x) for ntype in NODE_FEATURE_DIMS}
        edge_index_dict = {et: data[et].edge_index for et in EDGE_TYPES}
        edge_attr_dict = {et: data[et].edge_attr for et in EDGE_TYPES}

        for conv in self.convs:
            out = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            # un noeud qui n'a AUCUNE arete entrante d'un certain type ne recoit rien de
            # cette couche pour ce type -> HeteroConv omet parfois la cle ; on la restaure
            # a partir de x_dict pour ne jamais perdre de noeuds (ex: station isolee).
            for ntype in x_dict:
                if ntype not in out:
                    out[ntype] = x_dict[ntype]
            # CONNEXION RESIDUELLE (Res-GAT, cf. Hou et al. dans ta revue de litterature) :
            # sans ca, avec peu de satellites connectes a beaucoup de requetes, l'identite
            # de chaque requete se dilue en 2 couches de moyennage (over-smoothing) -- verifie
            # empiriquement : l'ecart-type inter-noeuds s'effondrait de 0.097 a 0.0005 sans
            # ce "+ x_dict[k]". Garder cette note pour ta section methodologie.
            x_dict = {k: F.elu(v + x_dict[k]) for k, v in out.items()}

        return x_dict   #dictionnaire avec embeddings de chaque type de noeuds

    def graph_embedding(self, x_dict):
        """Etape 2 : agregation -> embedding global du graphe, taille FIXE
        (3 * hidden_dim) quel que soit le nombre de requetes/satellites/stations."""
        parts = []
        for ntype in ["request", "satellite", "station"]:
            v = x_dict[ntype]
            parts.append(v.mean(dim=0) if v.size(0) > 0 else torch.zeros(self.hidden_dim))
        return torch.cat(parts, dim=-1)


class ActionScorer(nn.Module):
    """Apercu de l'etape 3 : score une action candidate (requete i, satellite s)
    a partir de h_i, h_s et de l'embedding global du graphe."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),   # h_request + h_satellite + h_graph(3*hidden_dim)
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_request, h_satellite, h_graph):
        combined = torch.cat([h_request, h_satellite, h_graph], dim=-1)
        return self.mlp(combined).squeeze(-1)


"""NB:
Pourquoi coller h_graph aussi, et pas seulement h_req et h_sat ? Parce que la qualité d'une action dépend 
du contexte global : acquérir une requête peu prioritaire n'a pas la même valeur si le satellite choisi est presque plein 
(mauvais pour l'équilibrage) ou si la constellation entière est peu chargée. h_graph apporte cette information de contexte 
que h_req et h_sat seuls ne peuvent pas donner."""