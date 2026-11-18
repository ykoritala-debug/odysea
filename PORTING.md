# Porting the routing core into the FastAPI / React app

I only had the Streamlit project, not `src/api/main.py` or `frontend/`, so this is a map, not a patch.
Upload `src/` and `frontend/` and I will do the port against the real code.

## What is portable as-is (numpy/scipy only, no Streamlit)
| This repo | Replaces / feeds in the FastAPI app |
|---|---|
| `odysea/fuel.py` | `FUEL_PROFILE` speed/fuel code in `main.py` |
| `odysea/routing.py` (`RoutingGraph`, `plan`, `route_metrics`, `alerts`) | NetworkX Dijkstra + λ sweep + Pareto in `main.py` |
| `odysea/drift.py` (`drift_ensemble`, `exposure_field`) | `src/models/iceberg_drift.py` (+ adds uncertainty and a routing field) |
| `odysea/nic.py` (`load_csv`, `to_bergs`) | NIC CSV reading in `plan_cache.py` |
| `odysea/forecast.py` | optional: ensemble version of `seaice_forecast_xgb_residual.json` |

## Check first: does your `main.py` fuel model have the same inversion?
If it follows the Judge Briefing (power ∝ speed³ × ice factor, speed capped at 5 kn in ice), it does.
Quick test: fuel per nm at 0 % and at 90 % ice should be **higher** at 90 %:

    fuel_per_nm(c) = fuel_t_per_h(c) / speed_kn(c)

If `fuel_per_nm(0.9) < fuel_per_nm(0.0)` the model rewards ice. After fixing, the numbers in the
Judge Briefing (458 t / 464 t / 271 t / 373 t, total ≈ 1,101 t) must be regenerated and quoted anew.

## Differences you must reconcile
- **Grid**: this repo routes on a 0.5° lat/lon grid; the FastAPI app routes on the NSIDC grid regridded
  to lat/lon and downsampled ×2 (≈ 50 km). `RoutingGraph(grid)` takes any object with `.lat`, `.lon`,
  `.LAT`, `.LON`, `.shape` (`odysea.geo.Grid`).
- **Ice input**: `plan()` takes an ensemble `(members, H, W)`. For your single median map pass
  `ice[None]` (one member): chance-constraint then degenerates to your 0.80 cut-off, as today.
- **Costs**: yours is `distance × (1 + λ·ice²)`; here `distance × (1 + λ·E[ice²]) × wind × berg`.
  Wind and iceberg multipliers switch on only when you pass `wind=` / `bergp=`.
- **North of 60°S** and the **Cape Town sea-exit waypoints** live in `main.py` and are not in this repo.
- **What-if replan** (`replan.py`): pass the user's icebergs/gales as `bergp` (probability field in
  [0, 1], > 0.5 blocks) and `wind` — no separate code path is needed.
- **Labels**: use the constants in `odysea/config.py` so React colours match.
