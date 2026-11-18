"""Vessel, voyage legs and prototype thresholds.

Everything marked ASSUMED is a prototype value, not a measured or regulatory one.
"""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Vessel:
    name: str = "MV Vasiliy Golovnin"
    length_m: float = 163.9
    beam_m: float = 22.46
    power_kw: float = 11460.0      # 2 x 5,730 kW (Russian Maritime Register)
    v_open_kn: float = 12.8        # AAD voyage reports
    v_heavy_kn: float = 5.0        # AAD voyage reports
    heavy_ice_conc: float = 0.90   # concentration at which speed bottoms out (ASSUMED)
    hotel_kw: float = 500.0        # ASSUMED
    efficiency: float = 0.92       # ASSUMED
    sfoc_g_kwh: float = 195.0      # ASSUMED
    cruise_load: float = 0.82      # ASSUMED engine load in open water
    max_load: float = 0.95         # ASSUMED engine load when pushing heavy ice
    ice_thickness_m: float = 1.2   # ASSUMED fixed (no thickness data); scales the ice burden


@dataclass(frozen=True)
class Leg:
    id: str
    name: str
    start: tuple
    end: tuple
    t0: str
    t1: str


CAPE_TOWN = (-33.9249, 18.4241)
BHARATI = (-69.4052, 76.1973)   # NCPOR tender anchor proxy: verify current anchorage
MAITRI = (-70.5, 11.8)          # seaward approach (station is ~80 km inland)

LEGS = {
    "leg1": Leg("leg1", "Cape Town -> Bharati", CAPE_TOWN, BHARATI, "2023-12-23", "2024-01-16"),
    "leg2": Leg("leg2", "Bharati -> Maitri", BHARATI, MAITRI, "2024-02-25", "2024-03-07"),
    "leg3": Leg("leg3", "Maitri -> Cape Town", MAITRI, CAPE_TOWN, "2024-03-10", "2024-03-31"),
}

ICE_LIMIT = 0.80        # prototype filter, NOT a Polar Code / ice-class limit
GALE_MS = 17.2          # Beaufort 8
STORM_MS = 24.5         # Beaufort 10
HURRICANE_FORCE_MS = 32.9  # Beaufort 12 (64 kt); hard exclusion in normal routing
VESSEL = Vessel()

# Route labels (single source of truth so the UI palette can never drift from the router).
LABEL_SAFEST = "Safest"
LABEL_FASTEST = "Fastest"
LABEL_FUEL = "Lowest fuel"
LABEL_BALANCED = "Balanced"
LABEL_CORRIDOR = "Feasible corridor"
LABEL_ALT = "Alternative"
