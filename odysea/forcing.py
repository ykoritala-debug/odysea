"""Optional real forcing. UNTESTED offline in this build: verify on your machine.
ECMWF AIFS is a pretrained ML weather model whose forecasts are published as open data;
we use its 10 m wind as forcing instead of running any weather model ourselves."""
import numpy as np

def fetch_aifs_wind(target="aifs_latest.grib2", steps=(0, 24, 48, 72, 96, 120)):
    from ecmwf.opendata import Client
    Client(source="aws", model="aifs-single").retrieve(
        time=0, stream="oper", type="fc", step=list(steps), param=["10u", "10v"], target=target)
    return target

def read_wind(path):
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib")
    return ds["u10"].values, ds["v10"].values, ds["latitude"].values, ds["longitude"].values

def wind_speed(u, v): return np.hypot(u, v)
