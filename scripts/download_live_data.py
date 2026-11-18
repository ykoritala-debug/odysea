"""Download the live inputs expected by ODySEA.

NOAA/NSIDC G10016 V4 Antarctic daily sea-ice concentration is public HTTPS.
Copernicus Marine current data requires a free Copernicus Marine account and
`copernicusmarine login` before running this script.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import zipfile

import numpy as np
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
LIVE.mkdir(parents=True, exist_ok=True)
ICE_OUT = LIVE / "antarctic_ice_nsidc.nc"
CURRENT_OUT = LIVE / "antarctic_currents_route.nc"
WIND_OUT = LIVE / "antarctic_wind_forecast.nc"
BYU_OUT = LIVE / "byu_iceberg_consolidated_v8.zip"


def download_noaa_ice(days_back=14):
    """Find and download the newest available Antarctic G10016 V4 daily file."""
    today = datetime.now(timezone.utc).date()
    session = requests.Session()
    session.headers.update({"User-Agent": "ODySEA research prototype data downloader"})
    last_error = None
    for offset in range(days_back + 1):
        day = today - timedelta(days=offset)
        name = f"sic_pss25_{day:%Y%m%d}_am2_icdr_v04r00.nc"
        url = (
            "https://noaadata.apps.nsidc.org/NOAA/G10016_V4/"
            f"south/daily/{day:%Y}/{name}"
        )
        try:
            with session.get(url, stream=True, timeout=(20, 180)) as response:
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                tmp = ICE_OUT.with_suffix(".nc.part")
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                if tmp.stat().st_size < 100_000:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("Downloaded NOAA file is unexpectedly small")
                tmp.replace(ICE_OUT)
                print(f"NOAA/NSIDC ice saved: {ICE_OUT} ({day.isoformat()})")
                return
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(
        f"Could not find/download an Antarctic G10016 V4 daily file in the last "
        f"{days_back + 1} days. Last network error: {last_error}"
    )


def download_copernicus_currents():
    """Subset Copernicus global daily surface currents to ODySEA's region."""
    try:
        import copernicusmarine
    except ImportError as exc:
        raise RuntimeError(
            "Install the Copernicus client first: pip install copernicusmarine"
        ) from exc

    # This Copernicus dataset is daily global ocean current analysis/forecast.
    # ODySEA's current loader selects the shallowest depth layer.
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=-5.0,
        maximum_longitude=105.0,
        minimum_latitude=-76.0,
        maximum_latitude=-33.0,
        start_datetime=(datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00"),
        end_datetime=datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00"),
        minimum_depth=0.0,
        maximum_depth=0.5,
        output_filename=CURRENT_OUT.name,
        output_directory=str(LIVE),
        force_download=True,
    )
    if not CURRENT_OUT.exists() or CURRENT_OUT.stat().st_size < 100_000:
        raise RuntimeError(
            "Copernicus client did not create the expected current file: "
            f"{CURRENT_OUT}"
        )
    print(f"Copernicus currents saved: {CURRENT_OUT}")


def download_wind_forecast():
    """Download ECMWF IFS 10 m wind forecasts through Open-Meteo on a 3-degree grid."""
    # Coarse source grid is interpolated to ODySEA's 0.5-degree route grid.
    latitudes = np.arange(-78.0, -29.9, 3.0)
    longitudes = np.arange(-6.0, 108.1, 3.0)
    points = [(float(la), float(lo)) for la in latitudes for lo in longitudes]
    records = {}
    session = requests.Session()
    for start in range(0, len(points), 40):
        batch = points[start:start + 40]
        params = {
            "latitude": ",".join(str(p[0]) for p in batch),
            "longitude": ",".join(str(p[1]) for p in batch),
            "hourly": "wind_speed_10m,wind_direction_10m",
            "forecast_days": 2, "timezone": "UTC",
            "wind_speed_unit": "ms", "models": "ecmwf_ifs025",
            "cell_selection": "sea",
        }
        response = session.get("https://api.open-meteo.com/v1/ecmwf", params=params, timeout=90)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = [payload]
        for item in payload:
            key = (round(float(item["latitude"]), 4), round(float(item["longitude"]), 4))
            records[key] = item
    if not records:
        raise RuntimeError("Open-Meteo returned no ECMWF wind forecast locations")
    first = next(iter(records.values()))
    times = np.asarray(first["hourly"]["time"], dtype="datetime64[ns]")
    lat_out = np.array(sorted({k[0] for k in records}))
    lon_out = np.array(sorted({k[1] for k in records}))
    speed = np.full((len(times), len(lat_out), len(lon_out)), np.nan, dtype=np.float32)
    direction = np.full_like(speed, np.nan)
    lat_i = {v:i for i,v in enumerate(lat_out)}
    lon_i = {v:i for i,v in enumerate(lon_out)}
    for (la, lo), item in records.items():
        h = item["hourly"]
        if len(h["time"]) != len(times):
            continue
        speed[:, lat_i[la], lon_i[lo]] = np.asarray(h["wind_speed_10m"], dtype=np.float32)
        direction[:, lat_i[la], lon_i[lo]] = np.asarray(h["wind_direction_10m"], dtype=np.float32)
    ds = xr.Dataset(
        {"wind_speed_10m": (("time", "latitude", "longitude"), speed),
         "wind_direction_10m": (("time", "latitude", "longitude"), direction)},
        coords={"time": times, "latitude": lat_out, "longitude": lon_out},
        attrs={"source": "ECMWF IFS 0.25 via Open-Meteo ECMWF API",
               "wind_speed_units": "m s-1", "direction_convention": "meteorological FROM"},
    )
    tmp = WIND_OUT.with_suffix(".nc.part")
    ds.to_netcdf(tmp)
    tmp.replace(WIND_OUT)
    print(f"ECMWF wind forecast saved: {WIND_OUT}")


def download_byu_iceberg_history():
    """Download BYU/NIC consolidated iceberg tracks (1978-April 2025)."""
    url = "https://www.scp.byu.edu/data/iceberg/consolidated_database_v8.0.zip"
    response = requests.get(url, timeout=90, headers={"User-Agent": "ODySEA/1.0"})
    response.raise_for_status()
    if len(response.content) < 100_000 or not response.content.startswith(b"PK"):
        raise RuntimeError("BYU iceberg history download was not a valid ZIP archive")
    tmp = BYU_OUT.with_suffix(".zip.part")
    tmp.write_bytes(response.content)
    tmp.replace(BYU_OUT)
    print(f"BYU/NIC iceberg history saved: {BYU_OUT}")


if __name__ == "__main__":
    try:
        download_noaa_ice()
        download_copernicus_currents()
        download_wind_forecast()
        download_byu_iceberg_history()
    except Exception as exc:
        print(f"LIVE DATA DOWNLOAD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
