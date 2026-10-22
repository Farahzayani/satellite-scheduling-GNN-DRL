"""
Definir les jeux de seeds d'entrainement, de validation et de test.

Definir des classes de taille (S/M/L/XL), pour pouvoir etendre progressivement vers
des instances plus grandes une fois le pipeline S valide.
"""

# --- classes de taille ---
# Commencer par "S" pour valider tout le pipeline (CP-SAT doit rester rapide),

TAILLE_CLASSES = {
    "S": dict(n_requests=10, n_satellites=2, n_ground_stations=2, horizon=600.0),
    "M": dict(n_requests=20, n_satellites=3, n_ground_stations=2, horizon=900.0),
    # "L": dict(n_requests=50, n_satellites=4, n_ground_stations=3, horizon=1440.0),
}

# --- seeds d'entrainement ---
TRAIN_SEEDS = list(range(0, 100))          # 100 instances d'entrainement

# --- seeds de validation ---
VAL_SEEDS = list(range(100, 110))          # 10 instances de validation

# --- seeds de test ---
TEST_SEEDS = list(range(200, 220))         # 20 instances de test

# garde-fou : verifie qu'il n'y a aucun recouvrement entre les trois ensembles
assert not set(TRAIN_SEEDS) & set(VAL_SEEDS), "chevauchement train/val !"
assert not set(TRAIN_SEEDS) & set(TEST_SEEDS), "chevauchement train/test !"
assert not set(VAL_SEEDS) & set(TEST_SEEDS), "chevauchement val/test !"