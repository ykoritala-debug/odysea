"""Live NOAA sea ice and Copernicus ocean-current data adapter."""
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .config import DATA_DIR
from .geo import Grid, haversine_nm
from .wind import load_wind, WIND_DEFAULT

# xarray / pyproj are imported lazily inside load_live_scenario so this module (and the
# endpoint-snapping helpers below) can be imported and unit-tested without them.

LIVE_DIR = DATA_DIR / "live"
ICE_DEFAULT = LIVE_DIR / "antarctic_ice_nsidc.nc"


def find_latest(pattern, directory=LIVE_DIR):
    """Newest file in `directory` matching `pattern` (by name, then mtime), or None."""
    files = sorted(Path(directory).glob(pattern), key=lambda p: (p.name, p.stat().st_mtime))
    return files[-1] if files else None


def default_current_path():
    """Latest antarctic_currents_*.nc in data/live (the old code hard-coded one dated filename)."""
    return find_latest("antarctic_currents*.nc") or (LIVE_DIR / "antarctic_currents_route.nc")


CURRENT_DEFAULT = None  # resolved at call time via default_current_path()


def _interp2d(x, y, values, target_x, target_y, method="linear"):
    """Interpolate a regular source grid onto target coordinate arrays."""
    interp = RegularGridInterpolator(
        (y, x),
        values,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack((target_y.ravel(), target_x.ravel()))
    return interp(points).reshape(target_x.shape)


def load_live_scenario(
    ice_path=ICE_DEFAULT,
    current_path=None,
    wind_path=None,
    grid_step=0.5,
):
    """
    Load and regrid live-source data onto the ODySEA lat/lon grid.

    Sea ice is one observation, NOT a multi-day forecast.
    Current fields remain separate from wind fields.
    Missing values remain NaN and are never converted to open water.
    """
    import xarray as xr
    from pyproj import CRS, Transformer

    ice_path = Path(ice_path)
    current_path = Path(current_path) if current_path else default_current_path()
    wind_path = Path(wind_path) if wind_path else WIND_DEFAULT

    if not ice_path.exists():
        raise FileNotFoundError(f"NOAA ice file not found: {ice_path}")
    if not current_path.exists():
        raise FileNotFoundError(f"Copernicus current file not found: {current_path}")

    # ODySEA target grid
    lat = np.arange(-76.0, -32.99, grid_step)
    lon = np.arange(-5.0, 105.01, grid_step)
    grid = Grid(lat, lon)

    # ---------------------------------------------------------
    # 1. NOAA sea ice: polar stereographic EPSG:3412 -> lat/lon
    # ---------------------------------------------------------
    with xr.open_dataset(ice_path) as ds:
        required = [
            "cdr_seaice_conc",
            "cdr_seaice_conc_qa_flag",
            "cdr_seaice_conc_stdev",
        ]
        missing = [name for name in required if name not in ds]
        if missing:
            raise ValueError(f"NOAA file is missing variables: {missing}")

        ice = ds["cdr_seaice_conc"].isel(time=0).values.astype(float)
        qa = ds["cdr_seaice_conc_qa_flag"].isel(time=0).values
        stdev = ds["cdr_seaice_conc_stdev"].isel(time=0).values.astype(float)

        x = ds["x"].values.astype(float)
        y = ds["y"].values.astype(float)

        ice_time = str(ds["time"].values[0])
        # NSIDC V4 supplies CF projection metadata. Use it rather than assuming
        # every release is encoded with the same EPSG label.
        crs_attrs = dict(ds["crs"].attrs) if "crs" in ds else {}
        try:
            source_crs = CRS.from_cf(crs_attrs) if crs_attrs else CRS.from_epsg(3412)
        except Exception:
            source_crs = CRS.from_epsg(3412)

        # Strict QA: reject no-input, invalid-ice and temporal interpolation.
        rejected = (qa.astype(np.uint16) & (8 | 16 | 64)) != 0
        ice[rejected] = np.nan
        stdev[rejected] = np.nan

        # Ensure source axes are ascending for RegularGridInterpolator.
        if y[0] > y[-1]:
            y = y[::-1]
            ice = ice[::-1, :]
            stdev = stdev[::-1, :]

        if x[0] > x[-1]:
            x = x[::-1]
            ice = ice[:, ::-1]
            stdev = stdev[:, ::-1]

        transformer = Transformer.from_crs(
            CRS.from_epsg(4326), source_crs, always_xy=True
        )
        target_x, target_y = transformer.transform(grid.LON, grid.LAT)

        ice_grid = _interp2d(x, y, ice, target_x, target_y)
        stdev_grid = _interp2d(x, y, stdev, target_x, target_y)

        qa_grid = _interp2d(
            x, y, qa.astype(float), target_x, target_y, method="nearest"
        )
        qa_grid = np.where(np.isfinite(qa_grid), qa_grid, 255).astype(np.uint8)

        ice_coverage = np.isfinite(ice_grid)

        # Keep ice concentration in fraction units [0, 1].
        ice_grid = np.clip(ice_grid, 0.0, 1.0)

    # ---------------------------------------------------------
    # 2. Copernicus currents: lat/lon -> ODySEA grid
    # ---------------------------------------------------------
    with xr.open_dataset(current_path) as ds:
        for name in ("uo", "vo", "latitude", "longitude", "time"):
            if name not in ds and name not in ds.coords:
                raise ValueError(f"Copernicus file is missing {name}")

        source_lat = ds["latitude"].values.astype(float)
        source_lon = ds["longitude"].values.astype(float)

        lat_order = np.argsort(source_lat)
        lon_order = np.argsort(source_lon)
        source_lat = source_lat[lat_order]
        source_lon = source_lon[lon_order]

        current_times = ds["time"].values
        current_u = []
        current_v = []
        current_coverage = []

        for ti in range(len(current_times)):
            # Select the available surface depth.
            u_da = ds["uo"].isel(time=ti)
            v_da = ds["vo"].isel(time=ti)
            # Copernicus global products may include a depth axis. Route planning
            # uses the surface layer only; select the shallowest available level.
            if "depth" in u_da.dims:
                u_da = u_da.isel(depth=0)
            if "depth" in v_da.dims:
                v_da = v_da.isel(depth=0)
            u = u_da.squeeze().values.astype(float)
            v = v_da.squeeze().values.astype(float)

            u = u[np.ix_(lat_order, lon_order)]
            v = v[np.ix_(lat_order, lon_order)]

            ui = _interp2d(
                source_lon, source_lat, u, grid.LON, grid.LAT
            )
            vi = _interp2d(
                source_lon, source_lat, v, grid.LON, grid.LAT
            )

            coverage = np.isfinite(ui) & np.isfinite(vi)

            current_u.append(ui)
            current_v.append(vi)
            current_coverage.append(coverage)

        current_u = np.stack(current_u)
        current_v = np.stack(current_v)
        current_coverage = np.stack(current_coverage)

        current_times = current_times.astype("datetime64[ns]")
        current_depth_m = float(ds["depth"].values[0])
        current_source = str(ds.attrs.get("title", "Copernicus Marine"))

    wind_data = load_wind(wind_path, grid)

    return {
        "grid": grid,
        "ice": ice_grid[None, :, :],
        "ice_stdev": stdev_grid,
        "ice_coverage": ice_coverage,
        "ice_qa": qa_grid,
        "ice_time": ice_time,
        "ice_source": "NOAA/NSIDC G10016 V4",
        "ice_is_forecast": False,
        "current_u": current_u,
        "current_v": current_v,
        "current_coverage": current_coverage,
        "current_times": current_times,
        "current_depth_m": current_depth_m,
        "current_source": current_source,
        "wind_u": wind_data["wind_u"],
        "wind_v": wind_data["wind_v"],
        "wind": wind_data["wind"],
        "wind_times": wind_data["wind_times"],
        "wind_source": wind_data["wind_source"],
        "wind_valid_until": wind_data["wind_valid_until"],
        "land": ~ice_coverage,
        "synthetic": False,
    }


def validate_live_endpoint(grid, ice_coverage, point, name="endpoint"):
    """Ensure a route endpoint is actually inside valid NOAA ice coverage."""
    lat, lon = point
    i = int(np.abs(grid.lat - lat).argmin())
    j = int(np.abs(grid.lon - lon).argmin())

    if not ice_coverage[i, j]:
        raise ValueError(
            f"{name} ({lat:.3f}, {lon:.3f}) is outside valid NOAA ice coverage. "
            "Live ice routing is not permitted at this endpoint."
        )
    return i, j


def snap_live_endpoint(grid, ice_coverage, point, name="endpoint",
                       max_distance_nm=100.0):
    """Snap a port/anchorage to nearby valid NOAA ice coverage."""
    lat, lon = point
    distances = haversine_nm(lat, lon, grid.LAT, grid.LON)
    distances = np.where(ice_coverage, distances, np.inf)

    idx = np.unravel_index(np.argmin(distances), distances.shape)
    distance = float(distances[idx])

    if not np.isfinite(distance) or distance > max_distance_nm:
        raise ValueError(
            f"{name}: nearest valid NOAA ice cell is "
            f"{distance:.1f} NM away; maximum allowed is "
            f"{max_distance_nm:.1f} NM. Live route rejected."
        )

    snapped = (
        float(grid.LAT[idx]),
        float(grid.LON[idx]),
    )
    return snapped, distance
