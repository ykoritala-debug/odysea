# ODySEA — Antarctic voyage decision support (prototype)

Decision support only. **Not certified for navigation. The Master retains authority.**

    python -m venv .venv && .venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/macOS)
    pip install -r requirements.txt
    python scripts/selftest.py                  # trains, forecasts, plans all 3 legs on SYNTHETIC data
    python -m unittest discover -s tests -v     # 17 tests, no pytest needed
    streamlit run app.py                        # dashboard
    uvicorn odysea.api:app                      # API  (docs at /docs)

## Modes
| Mode | Ice source | Notes |
|---|---|---|
| Existing Demo Routes | synthetic scenario, or your own NetCDF (`ice(time,lat,lon)`, optional `u10,v10,sst`) | labelled SYNTHETIC unless a NetCDF is given |
| Live NOAA Routes | `data/live/antarctic_ice_nsidc.nc` (NOAA/NSIDC G10016 V4, Antarctic daily SIC) + `data/live/antarctic_currents*.nc` (Copernicus Marine global daily currents; newest file is used) | one ice observation, **not an ice forecast**; currents are model analysis/forecast fields and affect route cost/time; never falls back to demo routes |

API: `POST /plan`, `POST /plan/live`, `GET /legs`, `GET /health`.

### Download the live routing inputs

1. Install the normal dependencies: `pip install -r requirements.txt`
2. Install the Copernicus downloader: `pip install -r requirements-live.txt`
3. Create a free account at [Copernicus Marine](https://data.marine.copernicus.eu/) and configure the client once with `copernicusmarine login`.
4. From the project root, run: `python scripts/download_live_data.py`

The script searches the previous 15 calendar days for the newest public NOAA/NSIDC G10016 V4 Antarctic daily file and saves it as `data/live/antarctic_ice_nsidc.nc`. It then subsets the Copernicus global daily current dataset to the ODySEA region and saves `data/live/antarctic_currents_route.nc`. If the newest NOAA file has not yet been published, it tries earlier dates. Copernicus Marine account credentials are required for the current subset download; do not put credentials in source code.

The loader reads the NOAA CF projection metadata, concentration, QA and standard-deviation fields; it reads Copernicus eastward/northward currents (`uo`, `vo`) at the shallowest depth. The latest available current time is projected along each route segment and affects route cost, estimated transit time and fuel comparison. Missing current values are treated as neutral flow (0 m/s); missing/invalid NOAA ice remains blocked. These currents are model analysis/forecast fields, not direct current-meter observations. NOAA/NSIDC G10016 V4 is near-real-time and preliminary; replace it with the final CDR when that is published.

## What the router does
1. Bagged residual-XGBoost ensemble + initial-state noise → ice ensemble.
2. Chance-constrained routing: cells with P(ice ≥ limit) > `p_block` are blocked; routes whose ensemble exceedance > 50 % are rejected.
3. Monte-Carlo iceberg drift → exposure probability field **inside** routing.
4. Gale/storm wind cost inside routing. Pareto filter over distance, time, fuel, ice risk, berg exposure.
5. P10–P90 time and fuel per route. Sparse graph + scipy Dijkstra (~0.03 s per leg).

## Honest status
- Fuel/speed constants, the 0.80 ice limit, drift coefficients and buffers are **assumed prototype values**.
- Fuel is for **comparing options**, not a prediction of tonnes burned.
- Forecast horizon is days; legs are weeks: beyond the horizon the router uses worst-day ice.
- Demo mode mixes real USNIC iceberg positions with synthetic ice/wind: fine for a demo, meaningless for a decision.
- USNIC download URL in `odysea/nic.py` is unverified; you can override it with the env var `USNIC_CSV_URL` or load a saved weekly file with `nic.load_csv(path)`.
- Live mode and the Streamlit UI were not executed in the environment these changes were made in (no network / packages) — run `streamlit run app.py` once and check both modes.
- Not validated against a real ship track.

See `CHANGELOG.md` for what changed and `PORTING.md` for moving this routing core into the FastAPI/React app.

## Live wind + iceberg-history integration (added)

Run the live-data downloader after installing the requirements and logging in to Copernicus Marine:

```powershell
pip install -r requirements.txt
pip install -r requirements-live.txt
copernicusmarine login
python scripts\download_live_data.py
```

The downloader now also retrieves:

- `data/live/antarctic_wind_forecast.nc`: ECMWF IFS 0.25 10 m wind speed/direction forecast via the Open-Meteo ECMWF API, sampled on a coarse Antarctic grid and interpolated to the ODySEA route grid.
- `data/live/byu_iceberg_consolidated_v8.zip`: BYU/NIC historical iceberg tracks (1978–22 April 2025).

Live route planning now uses observed NOAA/NSIDC sea ice, Copernicus surface currents, and the ECMWF wind forecast together. Wind policy: gale cells (>=17.2 m/s) receive a route-cost penalty; storm cells (>=24.5 m/s) receive a stronger penalty; hurricane-force cells (>=32.9 m/s, about 64 kt) are excluded from normal routing. Route metrics report gale/storm/hurricane-force distance. This is a prototype decision-support policy, not a certified navigation rule.

The Iceberg Intelligence table joins live USNIC iceberg IDs to the BYU/NIC consolidated historical tracks and shows first/last recorded dates, tracked span, years since first record, and observation count. These are **tracking-history** measures, not the physical age of an iceberg. Historical database coverage is through 22 April 2025; the USNIC live feed supplies current positions separately.

Wind forecasts are time-limited and spatially sampled at 3-degree intervals before interpolation. The app reports the forecast validity time; refresh the downloader regularly. Do not treat this prototype as a substitute for official ice charts, marine forecasts, ice-pilot advice, or vessel-specific safety procedures.
