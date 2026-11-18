# Changelog

## 0.3 — hardening pass

### Model bugs fixed
- **Fuel model was inverted** (`fuel.py`). Power was scaled by speed³ while speed *fell* in ice, so
  heavy ice cost about half the fuel per nautical mile of open water. Now engine load ramps from
  `cruise_load` to `max_load` with ice burden and speed follows the same burden, so fuel/nm rises
  monotonically (0.164 → 0.483 t/nm from open water to ≥0.9 concentration). The unused
  `ice_power_coeff` field was removed. **All fuel numbers change** (ice-heavy routes cost more).
- **Iceberg drift forcing was advanced 8× too fast** (`drift.py`, `pipeline.py`). Daily wind slices
  were indexed once per 3-hour step. New `forcing_dt_h` argument (default 24 h) fixes the indexing.
- **`plan()` raised `ValueError` when every cell was blocked** instead of returning "no route".
- NaN ice is treated as heavy ice (conservative), never as open water, in the fuel model.

### App / pipeline fixes
- Route label `"Lowestfuel"` ≠ palette key `"Lowest fuel"`: that route was drawn white. Labels now
  live in `config.py` and are shared by router and UI.
- The app re-fetched USNIC and replanned on every widget click. Now cached (`st.cache_data` /
  `st.cache_resource`) with a "Refresh data" button.
- Demo mode silently raised the blocking-probability slider to ≥ 0.50. Removed: the slider is honoured.
- Banner said "SYNTHETIC" even when a real NetCDF was loaded. Now reads `data_mode`.
- Live mode: unreachable endpoints / blocked corridors return `NO SAFE LIVE ROUTE` with the reason
  instead of a stack trace; missing data files give a clear message; `metadata` is populated;
  current-file path no longer hard-codes a dated filename (newest `antarctic_currents*.nc` is used).
- `use_container_width` (deprecated in newer Streamlit) replaced by a helper that inspects the
  widget signature.
- `marine.py` imports xarray/pyproj lazily so the module is importable and testable without them.

### Data / API
- `nic.py`: the column matcher treated `"id"` as matching `"width"`; DMS strings in separate
  Latitude/Longitude columns were silently dropped; added `load_csv()` for saved weekly files and a
  skipped-row count.
- `api.py`: validated inputs (`Literal` leg ids, bounded ints/floats), `/legs`, `/plan/live`, 503 for
  missing live data.
- Route de-duplication uses the full path, not every third node.
- `IceForecaster.save/load` close their files.

### Project hygiene
- `requirements.txt` was UTF-16 and missing scipy, scikit-learn, xgboost, xarray, netCDF4, h5netcdf,
  pyproj, matplotlib, fastapi and pydantic. Rewritten (UTF-8); download-only tools in
  `requirements-optional.txt`.
- Removed `.venv/` (≈ 900 MB), `__pycache__/`, `app.py.before_*` and `pipeline.py.before_*` backups; BOMs
  and CRLF line endings stripped; `.gitignore` added.
- Added `tests/test_core.py` (17 tests) and assertions in `scripts/selftest.py`.

### Not changed on purpose
Chance-constrained routing logic, Pareto selection, λ sweep, 0.80 limit, drift coefficients,
XGBoost features. Those are modelling choices, not defects.
