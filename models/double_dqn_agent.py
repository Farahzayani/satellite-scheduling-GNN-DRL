import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.graph_builder import build_hetero_graph
from models.gnn_encoder import HeteroGATEncoder, ActionScorer


Transition = namedtuple(
    "Transition",
    ["graph", "rid", "sid", "reward", "next_graph", "next_valid_pairs", "done"],
)


class DQNNet(nn.Module):
    """Regroupe l'encodeur GNN + la tete Q en un seul module. On instancie cette
    classe deux fois : une fois pour le reseau "online" (entraine a chaque pas),
    une fois pour le reseau "cible" (copie periodique, stabilise l'apprentissage)."""

    def __init__(self, hidden_dim=32, heads=4, n_layers=2):
        super().__init__()
        self.encoder = HeteroGATEncoder(hidden_dim=hidden_dim, heads=heads, n_layers=n_layers)
        self.scorer = ActionScorer(hidden_dim=hidden_dim)

    def score_pairs(self, graph, pairs):
        """pairs : liste de (rid, sid). Renvoie un tensor 1D de scores Q(s,a),
        un par paire, dans le meme ordre."""
        if not pairs:
            return torch.zeros(0)
        x_dict = self.encoder(graph)
        h_graph = self.encoder.graph_embedding(x_dict)
        req_id_to_idx = {rid: k for k, rid in enumerate(graph["request"].request_id)}
        sat_id_to_idx = {sid: k for k, sid in enumerate(graph["satellite"].satellite_id)}
        scores = []
        for rid, sid in pairs:
            h_req = x_dict["request"][req_id_to_idx[rid]]
            h_sat = x_dict["satellite"][sat_id_to_idx[sid]]
            scores.append(self.scorer(h_req, h_sat, h_graph))
        return torch.stack(scores)


class ReplayBuffer:
    """Memoire de replay classique : on y stocke les transitions vecues, on en
    tire un echantillon aleatoire a chaque etape d'entrainement pour casser la
    correlation temporelle entre transitions successives."""

    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size, rng):
        return rng.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, hidden_dim=32, heads=4, n_layers=2, lr=1e-3, gamma=0.99,
                 buffer_capacity=20000, target_update_every=200, seed=0):
        self.online = DQNNet(hidden_dim, heads, n_layers)
        self.target = DQNNet(hidden_dim, heads, n_layers)
        self.target.load_state_dict(self.online.state_dict())     

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = gamma                       # facteur d'actualisation
        self.buffer = ReplayBuffer(buffer_capacity)
        self.rng = random.Random(seed)
        self.target_update_every = target_update_every
        self.train_steps = 0

    # ------------------------------------------------------------------
    # SELECTION D'ACTION (epsilon-greedy)
    # ------------------------------------------------------------------
    def select_action(self, env, epsilon):
        """Renvoie (action, graphe_de_l_etat_courant). On construit le graphe
        systematiquement (meme en exploration aleatoire) car on en a besoin
        pour memoriser la transition ensuite."""
        valid_actions = env.get_valid_actions()          # ("acquire", rid, sid, o_idx)
        if not valid_actions:
            return ("wait",), None

        graph = build_hetero_graph(env)

        if self.rng.random() < epsilon:
            return self.rng.choice(valid_actions), graph

        pairs = [(a[1], a[2]) for a in valid_actions]
        with torch.no_grad():
            scores = self.online.score_pairs(graph, pairs)
        best_idx = int(torch.argmax(scores))
        return valid_actions[best_idx], graph

    # ------------------------------------------------------------------
    # MEMORISATION D'UNE TRANSITION
    # ------------------------------------------------------------------
    def remember(self, graph_before, action, reward, env_after, done):
        
        if action[0] != "acquire":
            return
        rid, sid = action[1], action[2]

        if done:
            graph_after, next_pairs = None, []
        else:
            graph_after = build_hetero_graph(env_after)
            next_valid = env_after.get_valid_actions()
            next_pairs = list({(a[1], a[2]) for a in next_valid})   # dedoublonne les (rid,sid)

        self.buffer.push(graph_before, rid, sid, reward, graph_after, next_pairs, done)

    """stocker un tuple "j'étais dans telle situation, j'ai fait ça, voilà ce que ça a donné"."""

    # ------------------------------------------------------------------
    # UNE ETAPE D'ENTRAINEMENT (sur un mini-batch echantillonne du buffer)
    # ------------------------------------------------------------------
    def train_step(self, batch_size=32,min_buffer_size=500):
        if len(self.buffer) < min_buffer_size:
            return None
        batch = self.buffer.sample(batch_size, self.rng)
        self.optimizer.zero_grad()
        losses = []
        for tr in batch:
            q_sa = self.online.score_pairs(tr.graph, [(tr.rid, tr.sid)])[0]

            if tr.done or not tr.next_valid_pairs:
                target = torch.tensor(float(tr.reward))
            else:
                with torch.no_grad():
                    # Double DQN : selection par le reseau online, evaluation par le reseau cible
                    online_next_scores = self.online.score_pairs(tr.next_graph, tr.next_valid_pairs)
                    best_next_idx = int(torch.argmax(online_next_scores))
                    target_next_scores = self.target.score_pairs(tr.next_graph, tr.next_valid_pairs)
                    target = tr.reward + self.gamma * target_next_scores[best_next_idx]

            losses.append(F.smooth_l1_loss(q_sa, target))

        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict()) 

        return loss.item()