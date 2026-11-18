"""Chance-constrained, multi-objective router.
 - Sparse 8-neighbour graph built ONCE; scipy.sparse.csgraph.dijkstra per lambda (vectorised, fast).
 - Ice is an ENSEMBLE: cells with P(ice>=limit) > p_block are blocked; cost uses E[ice^2].
 - Iceberg drift exposure and wind (gale) fields add cost / blocks.
 - lambda sweep + Pareto filter; every route reports ensemble spread (P10/P90) and P(exceed)."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .config import (VESSEL, ICE_LIMIT, GALE_MS, STORM_MS, HURRICANE_FORCE_MS, LABEL_SAFEST, LABEL_FASTEST,
                     LABEL_FUEL, LABEL_BALANCED, LABEL_CORRIDOR, LABEL_ALT)
from .fuel import speed_kn, fuel_t_per_h
from .geo import haversine_nm

OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
LAMBDAS = [0, 1, 2, 4, 8, 16, 32, 64]

class RoutingGraph:
    def __init__(self, grid):
        H, W = grid.shape; idx = np.arange(H * W).reshape(H, W); s, d = [], []
        for di, dj in OFFS:
            r0, r1 = max(0, -di), H - max(0, di); c0, c1 = max(0, -dj), W - max(0, dj)
            s.append(idx[r0:r1, c0:c1].ravel()); d.append(idx[r0 + di:r1 + di, c0 + dj:c1 + dj].ravel())
        self.grid, self.W, self.N = grid, W, H * W
        self.src, self.dst = np.concatenate(s), np.concatenate(d)
        la, lo = grid.LAT.ravel(), grid.LON.ravel()
        self.dist = haversine_nm(la[self.src], lo[self.src], la[self.dst], lo[self.dst])

    def shortest(self, weights, ok, s, t):
        g = csr_matrix((weights[ok], (self.src[ok], self.dst[ok])), shape=(self.N, self.N))
        dd, pred = dijkstra(g, directed=True, indices=s, return_predecessors=True)
        if not np.isfinite(dd[t]): return None
        p = [t]
        while p[-1] != s: p.append(pred[p[-1]])
        return p[::-1]

def route_metrics(grid, path, ice_ens, wind, bergp, V, limit, current_u=None, current_v=None):
    W = grid.shape[1]; ii, jj = np.divmod(np.array(path), W)
    la, lo = grid.lat[ii], grid.lon[jj]
    seg = haversine_nm(la[:-1], lo[:-1], la[1:], lo[1:])
    ice = np.nan_to_num(ice_ens[:, ii, jj]); c = 0.5 * (ice[:, :-1] + ice[:, 1:])       # (members, segs)
    vessel_speed = speed_kn(c, V)
    along_current_kn = None
    if current_u is not None and current_v is not None and len(path) > 1:
        # Project east/north surface current onto each route segment.
        u = np.asarray(current_u, dtype=float)[ii, jj]
        v = np.asarray(current_v, dtype=float)[ii, jj]
        dlat = np.diff(la)
        dlon = np.diff(lo) * np.cos(np.radians(0.5 * (la[:-1] + la[1:])))
        norm = np.hypot(dlat, dlon)
        east = np.divide(dlon, norm, out=np.zeros_like(dlon), where=norm > 0)
        north = np.divide(dlat, norm, out=np.zeros_like(dlat), where=norm > 0)
        u_mid = np.nan_to_num(0.5 * (u[:-1] + u[1:]))
        v_mid = np.nan_to_num(0.5 * (v[:-1] + v[1:]))
        along_current_kn = (u_mid * east + v_mid * north) * 1.943844
        vessel_speed = np.maximum(1.0, vessel_speed + along_current_kn[None, :])
    tm = seg / vessel_speed; fuel = (tm * fuel_t_per_h(c, V)).sum(1); hours = tm.sum(1)
    cm = c.mean(0); out = dict(
        distance_nm=float(seg.sum()), hours=float(hours.mean()), hours_p10_p90=[float(np.percentile(hours, 10)), float(np.percentile(hours, 90))],
        fuel_t=float(fuel.mean()), fuel_p10_p90=[float(np.percentile(fuel, 10)), float(np.percentile(fuel, 90))],
        max_ice=float(cm.max()), risk_integral=float((seg * cm ** 2).sum()), high_ice_nm=float(seg[cm > 0.5].sum()),
        p_exceed_limit=float((ice.max(1) >= limit).mean()))
    if wind is not None:
        w = np.nan_to_num(wind[ii, jj], nan=0.0); wm = 0.5 * (w[:-1] + w[1:])
        out.update(max_wind_ms=float(w.max()), gale_nm=float(seg[wm >= GALE_MS].sum()),
                   storm_nm=float(seg[wm >= STORM_MS].sum()),
                   hurricane_force_nm=float(seg[wm >= HURRICANE_FORCE_MS].sum()))
    out["berg_exposure"] = float(bergp[ii, jj].max()) if bergp is not None else 0.0
    if along_current_kn is not None:
        out["mean_along_current_kn"] = float(np.mean(along_current_kn))
    return out, la, lo

def _pareto(rs, keys):
    M = np.array([[r["m"][k] for k in keys] for r in rs]); keep = []
    for i in range(len(rs)):
        dom = np.any(np.all(M <= M[i], 1) & np.any(M < M[i], 1))
        if not dom: keep.append(rs[i])
    return keep

def plan(grid, start, end, ice_ens, wind=None, bergp=None, V=VESSEL,
         limit=ICE_LIMIT, p_block=0.25, graph=None, lambdas=LAMBDAS,
         berg_weight=8.0, wind_weight=1.0, max_route_pexceed=0.5,
         max_endpoint_snap_nm=100.0, current_u=None, current_v=None):

    G = graph or RoutingGraph(grid)

    ice_ens = np.asarray(ice_ens, dtype=float)
    if ice_ens.ndim != 3 or ice_ens.shape[1:] != grid.shape:
        raise ValueError(
            f"ice_ens must have shape (members, {grid.shape[0]}, "
            f"{grid.shape[1]}), got {ice_ens.shape}"
        )

    # Missing NOAA observations are blocked, never treated as open water.
    land = ~np.isfinite(ice_ens).all(axis=0)
    ie = np.nan_to_num(ice_ens, nan=1.0, posinf=1.0, neginf=1.0)

    risk = (ie ** 2).mean(0)
    pblk = (ie >= limit).mean(0)
    blocked = land | (pblk > p_block)

    # Extreme wind (Beaufort 12 / >=64 kt) is excluded from normal routing.
    # Gale and storm cells remain traversable but receive progressively larger costs.
    if wind is not None:
        wind = np.asarray(wind, dtype=float)
        if wind.shape != grid.shape:
            raise ValueError(f"wind must match navigation grid {grid.shape}, got {wind.shape}")
        blocked |= ~np.isfinite(wind) | (wind >= HURRICANE_FORCE_MS)

    if bergp is not None:
        blocked |= np.asarray(bergp) > 0.5

    # Find nearest safe cells, then enforce distance from requested points.
    try:
        si, sj = grid.nearest_valid(*start, blocked)
        ti, tj = grid.nearest_valid(*end, blocked)
    except ValueError:
        return {
            "routes": [],
            "snapped": None,
            "note": ("No feasible route: every grid cell is blocked at the "
                     "chosen safety thresholds."),
        }

    snapped_start = (float(grid.lat[si]), float(grid.lon[sj]))
    snapped_end = (float(grid.lat[ti]), float(grid.lon[tj]))

    start_distance = float(
        haversine_nm(start[0], start[1], *snapped_start)
    )
    end_distance = float(
        haversine_nm(end[0], end[1], *snapped_end)
    )

    snap_info = {
        "start": snapped_start,
        "end": snapped_end,
        "start_distance_nm": start_distance,
        "end_distance_nm": end_distance,
    }

    if (start_distance > max_endpoint_snap_nm or
            end_distance > max_endpoint_snap_nm):
        return {
            "routes": [],
            "snapped": snap_info,
            "note": (
                "Route rejected: the ice-safety mask moved an endpoint "
                f"beyond {max_endpoint_snap_nm:.1f} NM."
            ),
        }

    s, t = si * G.W + sj, ti * G.W + tj

    rf = risk.ravel()
    ok = ~blocked.ravel()[G.src] & ~blocked.ravel()[G.dst]
    er = 0.5 * (rf[G.src] + rf[G.dst])
    mult = np.ones_like(er)

    if current_u is not None and current_v is not None:
        # Directional current contribution to edge travel-time cost. Positive
        # along-track flow helps; opposing flow increases cost. Inputs are m/s.
        cu = np.asarray(current_u, dtype=float).ravel()
        cv = np.asarray(current_v, dtype=float).ravel()
        la0, lo0 = grid.LAT.ravel(), grid.LON.ravel()
        dlat = la0[G.dst] - la0[G.src]
        dlon = (lo0[G.dst] - lo0[G.src]) * np.cos(np.radians(0.5 * (la0[G.dst] + la0[G.src])))
        norm = np.hypot(dlat, dlon)
        east = np.divide(dlon, norm, out=np.zeros_like(dlon), where=norm > 0)
        north = np.divide(dlat, norm, out=np.zeros_like(dlat), where=norm > 0)
        ue = np.nan_to_num(0.5 * (cu[G.src] + cu[G.dst]))
        vn = np.nan_to_num(0.5 * (cv[G.src] + cv[G.dst]))
        along_kn = (ue * east + vn * north) * 1.943844
        # Approximate open-water reference speed for the relative time penalty.
        current_factor = np.clip(12.8 / np.maximum(1.0, 12.8 + along_kn), 0.55, 2.5)
        mult *= current_factor

    if wind is not None:
        wflat = np.nan_to_num(wind.ravel(), nan=0.0)
        # Gale penalty from 17.2 m/s; storm penalty is substantially stronger.
        gale_penalty = np.clip((wflat - GALE_MS) / 8.0, 0.0, 1.0)
        storm_penalty = np.clip((wflat - STORM_MS) / 8.4, 0.0, 1.0)
        cell_penalty = 1.0 + 2.0 * (wflat >= GALE_MS) + 4.0 * (wflat >= STORM_MS)
        cell_penalty += gale_penalty + 2.0 * storm_penalty
        mult *= 0.5 * (cell_penalty[G.src] + cell_penalty[G.dst])

    if bergp is not None:
        bp = np.asarray(bergp).ravel()
        mult *= 1 + berg_weight * 0.5 * (
            bp[G.src] + bp[G.dst]
        )

    seen, routes = set(), []

    for lam in lambdas:
        path = G.shortest(
            G.dist * (1 + lam * er) * mult, ok, s, t
        )
        if path is None:
            continue

        sig = tuple(path)
        if sig in seen:
            continue
        seen.add(sig)

        metrics, la, lo = route_metrics(
            grid, path, ice_ens, wind, bergp, V, limit, current_u, current_v
        )
        metrics["lambda"] = lam
        routes.append({
            "m": metrics,
            "lat": la.tolist(),
            "lon": lo.tolist()
        })

    if not routes:
        return {
            "routes": [],
            "snapped": snap_info,
            "note": (
                "No feasible route: all corridors blocked at the "
                "chosen safety thresholds."
            ),
        }

    feasible = [
        r for r in routes
        if r["m"]["p_exceed_limit"] <= max_route_pexceed
    ]

    note = None
    if feasible:
        routes = feasible
    else:
        note = (
            "All candidate routes exceed the ice limit in more than "
            f"{max_route_pexceed:.0%} of ensemble members."
        )

    keys = [
        "distance_nm", "hours", "fuel_t",
        "risk_integral", "berg_exposure"
    ]
    pareto = _pareto(routes, keys) or routes
    pick = lambda f: min(pareto, key=f)

    labels = {
        id(pick(lambda r: r["m"]["risk_integral"]
                + 1e-6 * r["m"]["distance_nm"])): LABEL_SAFEST,
        id(pick(lambda r: r["m"]["hours"])): LABEL_FASTEST,
        id(pick(lambda r: r["m"]["fuel_t"])): LABEL_FUEL,
    }

    maxima = {k: max(r["m"][k] for r in pareto) or 1 for k in keys}
    balanced = pick(
        lambda r: sum(r["m"][k] / maxima[k] for k in keys)
    )
    labels.setdefault(id(balanced), LABEL_BALANCED)

    for route in pareto:
        route["label"] = labels.get(id(route), LABEL_ALT)

    if len(pareto) == 1:
        pareto[0]["label"] = LABEL_CORRIDOR

    return {
        "routes": pareto,
        "note": note,
        "n_candidates": len(routes),
        "snapped": snap_info,
        "wind_policy": {"gale_threshold_ms": GALE_MS, "storm_threshold_ms": STORM_MS,
                        "hurricane_force_block_ms": HURRICANE_FORCE_MS,
                        "policy": "Gales penalized; storms strongly penalized; >=64 kt blocked"},
    }


def alerts(route, bergs, wind_note=True, radius_nm=90.0):
    out = []; la, lo = np.array(route["lat"]), np.array(route["lon"])
    for name, blat, blon in bergs:
        d = haversine_nm(la, lo, blat, blon).min()
        if d < radius_nm: out.append(("DANGER", f"Tracked iceberg {name} within {d:.0f} nm of route"))
    m = route["m"]
    if m.get("gale_nm", 0) > 0: out.append(("WARN", f"{m['gale_nm']:.0f} nm of route in gale winds (>= 17.2 m/s)"))
    if m["p_exceed_limit"] > 0.2: out.append(("WARN", f"{m['p_exceed_limit']:.0%} of ensemble members exceed the ice limit on this route"))
    return out
