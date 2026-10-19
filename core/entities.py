"""
Structures de donnees de base du probleme SSP : définition des objets du problème

Correspondance avec la formulation MILP :
    Request        <->  i in I           (priorite P_i, volume V_i, duree d_i)
    Satellite      <->  s in S           (memoire M_s)
    GroundStation  <->  g in G           (debit B_g)
    VisibilityWindow <-> opportunite o in O_i,s  -> [vts_i,s^o , vte_i,s^o]
    ContactWindow  <->  fenetre k in K_s,g       -> [alpha_s,g^k , beta_s,g^k]
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VisibilityWindow:
    """Fenetre de visibilite (opportunite o) d'une requete i par un satellite s."""
    request_id: int
    satellite_id: int
    opp_index: int              # indice o dans O_i,s
    start: float                 # vts_i,s^o (minutes depuis le debut de l'horizon)
    end: float                    # vte_i,s^o
    pointing_angle: float          # theta_i,s^o (deg) -> utilise pour le temps de transition

@dataclass
class ContactWindow:
    """Fenetre de contact (k) satellite - station sol."""
    satellite_id: int
    ground_station_id: int
    contact_index: int          # indice k dans K_s,g
    start: float                 # alpha_s,g^k
    end: float                    # beta_s,g^k


@dataclass
class Request:
    """Requete d'acquisition i."""
    id: int
    priority: float                # P_i
    volume: float                    # V_i (Go)
    duration: float                   # d_i (minutes)
    opportunities: List[VisibilityWindow] = field(default_factory=list)  # union des O_i,s

    #champs mis a jour par le simulateur 
    status: int = 0                   # 0 non planifiee / 1 acquise / 2 telechargee
    acquired_by: Optional[int] = None
    t_start: Optional[float] = None    # t^start_i,s
    t_end: Optional[float] = None       # t^end_i,s
    t_downlink: Optional[float] = None   # instant de fin de transmission (reception sol)
    downlinked_via: Optional[int] = None  # id de la station g


@dataclass
class Satellite:
    """Satellite s de la constellation."""
    id: int
    memory_capacity: float          # M_s
    memory_used: float = 0.0          # m_s(t)
    available_time: float = 0.0        # avail_s : prochain instant libre
    current_pointing: float = 0.0       # angle courant (deg)
    rotation_speed: float = 2.0          # v^rot_s (deg/min)
    stabilization_time: float = 0.3       # t^sta_s (min)
    pending_downlink: List[int] = field(default_factory=list)  # ids requetes acquises en attente de telechargement


@dataclass
class GroundStation:
    """Station sol g."""
    id: int
    datarate: float                 # B_g (Go/min)
    contacts: List[ContactWindow] = field(default_factory=list)