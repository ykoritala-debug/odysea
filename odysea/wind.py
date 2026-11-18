"""Load and regrid the latest ECMWF/Open-Meteo 10 m wind forecast."""
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .geo import Grid
from .config import DATA_DIR

WIND_DEFAULT = DATA_DIR / "live" / "antarctic_wind_forecast.nc"

def load_wind(path, target_grid):
    import xarray as xr
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Wind forecast file not found: {path}")
    with xr.open_dataset(path) as ds:
        required = ("wind_speed_10m", "wind_direction_10m", "latitude", "longitude", "time")
        missing = [v for v in required if v not in ds and v not in ds.coords]
        if missing:
            raise ValueError(f"Wind NetCDF missing variables: {missing}")
        lat = np.asarray(ds.latitude.values, float)
        lon = np.asarray(ds.longitude.values, float)
        lat_order, lon_order = np.argsort(lat), np.argsort(lon)
        lat, lon = lat[lat_order], lon[lon_order]
        speed = np.asarray(ds.wind_speed_10m.values, float)
        direction = np.asarray(ds.wind_direction_10m.values, float)
        # Open-Meteo wind direction is meteorological: direction FROM which wind blows.
        rad = np.deg2rad(direction)
        u = -speed * np.sin(rad)
        v = -speed * np.cos(rad)
        u, v = u[:, lat_order, :][:, :, lon_order], v[:, lat_order, :][:, :, lon_order]
        speed = speed[:, lat_order, :][:, :, lon_order]
        def interp(field):
            out=[]
            for frame in field:
                f=RegularGridInterpolator((lat, lon), frame, bounds_error=False, fill_value=np.nan)
                pts=np.column_stack((target_grid.LAT.ravel(), target_grid.LON.ravel()))
                out.append(f(pts).reshape(target_grid.shape))
            return np.stack(out)
        ui, vi, si = interp(u), interp(v), interp(speed)
        times = np.asarray(ds.time.values).astype("datetime64[ns]")
        # Route against the maximum forecast wind over the next 24 h.
        count = min(len(times), 25)
        if count < 1:
            raise ValueError("Wind forecast contains no time steps")
        wind_speed = np.nanmax(si[:count], axis=0)
        wind_u = np.nanmean(ui[:count], axis=0)
        wind_v = np.nanmean(vi[:count], axis=0)
        source = str(ds.attrs.get("source", "Open-Meteo ECMWF IFS forecast"))
    return {"wind": wind_speed, "wind_u": wind_u, "wind_v": wind_v,
            "wind_times": times, "wind_source": source,
            "wind_valid_until": str(times[count-1])}
