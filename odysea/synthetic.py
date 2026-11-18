"""SYNTHETIC scenario so the whole stack runs with no downloads.
Never present this as real data; the UI labels it."""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from .geo import Grid

def make_scenario(seed=0, days=45, step=0.5):
    rng = np.random.default_rng(seed)
    lat = np.arange(-76, -32.99, step); lon = np.arange(-5, 105.01, step)
    g = Grid(lat, lon); LAT, LON = g.LAT, g.LON
    coast = -71.0 + 1.0 * np.sin(np.radians(LON) * 2)
    land = (LAT < coast) | ((LAT > -34.6) & (LON < 19.5))
    edge0 = -62.0 + 2.0 * np.sin(np.radians(LON) * 3 + 0.5)
    W = len(lon)

    def ar_lon(scale):
        x = np.zeros((days, W)); x[0] = gaussian_filter1d(rng.normal(size=W), 8) * scale * 4
        for t in range(1, days):
            x[t] = 0.85 * x[t-1] + gaussian_filter1d(rng.normal(size=W), 8) * scale * 2
        return x
    vlon, ulon = ar_lon(3.0), ar_lon(3.0)
    westerly = 8 / (1 + np.exp(-(LAT + 58) / 3))
    static = gaussian_filter(rng.normal(size=g.shape), 6) * 0.9

    U = np.empty((days,) + g.shape); V = np.empty_like(U); ICE = np.empty_like(U); SST = np.empty_like(U)
    edge = edge0.copy()
    sst = -1.8 + 21.8 * np.clip((LAT + 70) / 37, 0, 1) ** 1.5
    for t in range(days):
        v = np.broadcast_to(vlon[t], g.shape).copy(); u = ulon[t] + westerly
        storm = 14 * np.exp(-(((LAT + 52) / 4) ** 2 + ((LON - 40 - 0.8 * t) / 9) ** 2))
        u = u + storm; v = v + 0.5 * storm
        edge = edge + 0.12 * vlon[t] + 0.05 * (edge0 - edge) + rng.normal(0, 0.04, W)
        ice = 0.97 / (1 + np.exp(-(edge - LAT) / 0.8))
        lead = np.exp(-((LON - 76 - 3*np.sin(t/6)) / 3.5) ** 2) + 0.9 * np.exp(-((LON - 12) / 3.5) ** 2)   # open leads through the pack
        poly = np.clip(lead * (LAT < -58) , 0, 1)
        ice = np.clip(ice * (1 - 0.7 * poly) + 0.06 * static * (ice > 0.05), 0, 1)
        U[t], V[t], ICE[t], SST[t] = u, v, ice, sst
    ICE[:, land] = np.nan
    bergs = [("A-76B-demo", -66.5, 60.0), ("B-demo", -67.8, 35.0), ("C-demo", -63.5, 85.0)]
    return dict(grid=g, ice=ICE, u=U, v=V, sst=SST, land=land, icebergs=bergs, synthetic=True)

def load_netcdf(path):
    """Expects ice(time,lat,lon) [+ optional u10,v10,sst]. Requires xarray."""
    import xarray as xr
    ds = xr.open_dataset(path)
    lat, lon = ds["lat"].values, ds["lon"].values
    flip = lat[0] > lat[-1]
    def get(k):
        a = ds[k].values
        return a[:, ::-1] if flip else a
    ice = get("ice"); z = np.zeros_like(ice)
    g = Grid(lat[::-1] if flip else lat, lon)
    return dict(grid=g, ice=ice, u=get("u10") if "u10" in ds else z, v=get("v10") if "v10" in ds else z,
                sst=get("sst") if "sst" in ds else z, land=np.isnan(ice).all(0), icebergs=[], synthetic=False)
