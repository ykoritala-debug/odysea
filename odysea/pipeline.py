import functools

import numpy as np

from .synthetic import make_scenario, load_netcdf
from .forecast import IceForecaster
from .drift import drift_ensemble, exposure_field
from .routing import plan, RoutingGraph
from .config import LEGS


@functools.lru_cache(maxsize=4)
def _prepare(members, seed, nc_path=None):
    sc = load_netcdf(nc_path) if nc_path else make_scenario(seed)
    n = sc["ice"].shape[0]
    split = int(n * 0.75)

    fc = IceForecaster(members, seed).fit(
        sc["ice"], sc["sst"], sc["u"], sc["v"],
        sc["grid"], sc["land"], range(0, split - 1)
    )
    skill = fc.evaluate(
        sc["ice"], sc["sst"], sc["u"], sc["v"],
        range(split, n - 1)
    )
    return sc, fc, skill, RoutingGraph(sc["grid"]), split


def _get_leg(leg_id):
    if leg_id not in LEGS:
        raise KeyError(f"Unknown leg_id {leg_id!r}; expected one of {sorted(LEGS)}")
    return LEGS[leg_id]


def run_leg(leg_id, horizon=5, members=5, p_block=0.25, seed=0, nc_path=None, extra_bergs=None):
    """Demo / NetCDF pipeline: train -> ensemble ice forecast -> iceberg drift -> route plan.

    Without nc_path the scenario is SYNTHETIC (and labelled so in the result).
    """
    leg = _get_leg(leg_id)
    sc, fc, skill, G, T = _prepare(members, seed, nc_path)
    g = sc["grid"]

    n_days = sc["ice"].shape[0]
    horizon = int(max(1, min(horizon, n_days - 2)))
    T = max(0, min(T, n_days - horizon - 1))
    forcing = [(sc["sst"][T + h], sc["u"][T + h], sc["v"][T + h]) for h in range(horizon)]

    # Worst-day ice over the forecast horizon, per ensemble member (conservative).
    ice_ens = fc.forecast(sc["ice"][T], forcing, horizon).max(1)

    wind = np.hypot(sc["u"][T], sc["v"][T])
    bergs = list(sc["icebergs"]) + list(extra_bergs or [])

    # Daily wind slices, advanced once per 24 h of drift (drift_ensemble forcing_dt_h=24).
    u_days = sc["u"][T:T + 4]
    v_days = sc["v"][T:T + 4]
    trs = [drift_ensemble(g, la, lo, u_days, v_days, hours=72, forcing_dt_h=24.0)
           for _, la, lo in bergs]

    bp = exposure_field(g, trs) if trs else None
    res = plan(g, leg.start, leg.end, ice_ens, wind, bp, graph=G, p_block=p_block)

    return dict(
        leg=leg.name,
        result=res,
        forecast_skill=skill,
        data_mode="SYNTHETIC DEMO" if sc["synthetic"] else "REAL NetCDF",
        scenario=sc,
        ice_ens=ice_ens,
        wind=wind,
        bergp=bp,
        bergs=bergs,
        trajectories=trs,
    )


def run_leg_live(leg_id, members=1, p_block=0.25, ice_path=None, current_path=None, max_snap_nm=100.0):
    """Conservative routing on ONE NOAA/NSIDC ice observation. Not a forecast.

    Missing files raise FileNotFoundError (a data problem, shown as such). An unreachable
    endpoint or an unroutable corridor returns status NO SAFE LIVE ROUTE with the reason in
    result["note"] - it never falls back to demo routes and never raises for safety rejections.
    """
    from .marine import load_live_scenario, snap_live_endpoint, ICE_DEFAULT
    from .geo import haversine_nm

    leg = _get_leg(leg_id)
    sc = load_live_scenario(ice_path or ICE_DEFAULT, current_path)
    g = sc["grid"]

    # Keep the actual observation as one field. No fake ensemble.
    ice_ens = np.asarray(sc["ice"], dtype=float)

    base = {
        "leg": leg.name,
        "forecast_skill": None,
        "data_mode": "LIVE NOAA ICE OBSERVATION + COPERNICUS CURRENTS + ECMWF WIND FORECAST",
        "scenario": sc,
        "ice_ens": ice_ens,
        "wind": sc.get("wind"),
        "bergp": None,
        "bergs": [],
        "trajectories": [],
        "current_u": sc["current_u"],
        "current_v": sc["current_v"],
        "current_coverage": sc["current_coverage"],
        "current_times": sc["current_times"],
        "ice_time": sc["ice_time"],
        "metadata": {
            "ice_time": sc["ice_time"],
            "ice_source": sc.get("ice_source"),
            "current_source": sc.get("current_source"),
            "wind_source": sc.get("wind_source"),
            "wind_valid_until": sc.get("wind_valid_until"),
        },
        "requested_start": leg.start,
        "requested_end": leg.end,
    }

    def rejected(note, **extra):
        base.update(
            result={"routes": [], "snapped": None, "note": note},
            route_status="NO SAFE LIVE ROUTE",
            **extra,
        )
        return base

    try:
        start, start_snap = snap_live_endpoint(g, sc["ice_coverage"], leg.start, "Route start", max_snap_nm)
        end, end_snap = snap_live_endpoint(g, sc["ice_coverage"], leg.end, "Route destination", max_snap_nm)
    except ValueError as exc:
        return rejected(f"Route rejected: {exc}")

    try:
        # Copernicus surface currents are now passed into the route cost and
        # route-time/fuel metrics as a directional along-track contribution.
        # Missing current cells contribute neutral current (0 m/s), not invented flow.
        current_u = np.asarray(sc["current_u"][-1], dtype=float)
        current_v = np.asarray(sc["current_v"][-1], dtype=float)
        result = plan(g, start, end, ice_ens, wind=sc.get("wind"), bergp=None,
                      graph=RoutingGraph(g), p_block=p_block,
                      max_endpoint_snap_nm=max_snap_nm,
                      current_u=current_u, current_v=current_v)
    except ValueError as exc:      # e.g. every cell blocked
        return rejected(f"Route rejected: {exc}")

    if result.get("snapped") is None:
        return rejected(result.get("note") or "Route rejected.")

    actual_start = tuple(float(x) for x in result["snapped"]["start"])
    actual_end = tuple(float(x) for x in result["snapped"]["end"])
    router_start_nm = float(haversine_nm(leg.start[0], leg.start[1], *actual_start))
    router_end_nm = float(haversine_nm(leg.end[0], leg.end[1], *actual_end))

    endpoint_ok = router_start_nm <= max_snap_nm and router_end_nm <= max_snap_nm
    if not endpoint_ok:
        result["routes"] = []
        result["note"] = ("Route rejected: the ice-safety mask moved an endpoint "
                          "beyond the permitted snapping distance.")

    route_ok = endpoint_ok and bool(result.get("routes"))
    base.update(
        result=result,
        route_status="SAFE ROUTE FOUND" if route_ok else "NO SAFE LIVE ROUTE",
        coverage_snapped_start=start,
        coverage_snapped_end=end,
        coverage_start_distance_nm=start_snap,
        coverage_end_distance_nm=end_snap,
        router_snapped_start=actual_start,
        router_snapped_end=actual_end,
        router_start_distance_nm=router_start_nm,
        router_end_distance_nm=router_end_nm,
    )
    return base
