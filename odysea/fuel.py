"""Prototype speed / fuel model. ALL engine constants are ASSUMED (see config.Vessel).

Use the output to COMPARE route options, not as a prediction of tonnes burned.

Model
-----
ice burden b = min(1, (c * thickness/1.2 / heavy_ice_conc)^2)   in [0, 1]
speed        = v_open - (v_open - v_heavy) * b
engine load  = cruise_load + (max_load - cruise_load) * b      (the Master pushes harder in ice)
fuel / hour  = (load * P_rated + hotel) / efficiency * SFOC

The earlier version scaled propulsion power with speed^3 while speed dropped in ice, which made
heavy ice *cheaper* per nautical mile than open water. A ship that slows in ice is resistance-
limited: it holds (or raises) engine load, so fuel per nm rises steeply with concentration.
"""
import numpy as np


def ice_burden(c, V):
    c = np.clip(np.nan_to_num(np.asarray(c, dtype=float), nan=1.0), 0.0, 1.0)
    eff = c * (V.ice_thickness_m / 1.2)
    return np.minimum(1.0, (eff / V.heavy_ice_conc) ** 2)


def speed_kn(c, V):
    b = ice_burden(c, V)
    return V.v_open_kn - (V.v_open_kn - V.v_heavy_kn) * b


def engine_load(c, V):
    b = ice_burden(c, V)
    return V.cruise_load + (V.max_load - V.cruise_load) * b


def fuel_t_per_h(c, V):
    p_kw = engine_load(c, V) * V.power_kw + V.hotel_kw
    return p_kw / V.efficiency * V.sfoc_g_kwh / 1e6


def fuel_t_per_nm(c, V):
    return fuel_t_per_h(c, V) / speed_kn(c, V)
