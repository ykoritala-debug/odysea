"""Monte-Carlo iceberg drift. Physics: v = a_c*current + a_w*R(theta)*wind + noise,
R = left-hand turn in the Southern Hemisphere (simplified Coriolis/Ekman deflection).
Coefficients are ASSUMED defaults; calibrate on NIC tracks with validate_one_day()."""
import numpy as np
from scipy.ndimage import maximum_filter
from .geo import haversine_nm

def drift_velocity(uc, vc, uw, vw, wind_factor=0.02, turn_deg=20.0, current_scale=1.0):
    th = np.radians(turn_deg); c, s = np.cos(th), np.sin(th)
    return (current_scale * uc + wind_factor * (c * uw - s * vw),
            current_scale * vc + wind_factor * (s * uw + c * vw))

def drift_ensemble(grid, lat0, lon0, u_wind, v_wind, u_cur=None, v_cur=None, hours=72, dt_h=3,
                   n=300, seed=0, forcing_dt_h=24.0):
    """Fields are (H,W) (static) or (T,H,W) (time-varying, one slice per `forcing_dt_h` hours,
    default daily). The slice used at integration step t is floor(t*dt_h/forcing_dt_h): an earlier
    version indexed by step, which advanced daily forcing once every 3 h (8x too fast).
    Returns (n, steps+1, 2) lat/lon."""
    u_wind, v_wind = np.asarray(u_wind), np.asarray(v_wind)
    u_cur = None if u_cur is None else np.asarray(u_cur)
    v_cur = None if v_cur is None else np.asarray(v_cur)
    rng = np.random.default_rng(seed); steps = int(hours / dt_h)
    lat = np.full(n, lat0); lon = np.full(n, lon0)
    wf = 0.02 * rng.lognormal(0, 0.3, n); cs = rng.normal(1.0, 0.2, n); turn = rng.normal(20, 8, n)
    out = np.empty((n, steps + 1, 2)); out[:, 0] = np.c_[lat, lon]
    def gi(f, t):
        if f.ndim == 2:
            return f
        k = int((t * dt_h) // forcing_dt_h)
        return f[min(k, f.shape[0] - 1)]
    for t in range(steps):
        ii = np.abs(grid.lat[None, :] - lat[:, None]).argmin(1); jj = np.abs(grid.lon[None, :] - lon[:, None]).argmin(1)
        uw, vw = gi(u_wind, t)[ii, jj], gi(v_wind, t)[ii, jj]
        uc = gi(u_cur, t)[ii, jj] if u_cur is not None else 0.0
        vc = gi(v_cur, t)[ii, jj] if v_cur is not None else 0.0
        u, v = drift_velocity(uc, vc, uw, vw, wf, turn, cs)
        u = u + rng.normal(0, 0.03, n); v = v + rng.normal(0, 0.03, n)      # unresolved diffusion
        lat = lat + v * dt_h * 3600 / 111_000
        lon = lon + u * dt_h * 3600 / (111_000 * np.cos(np.radians(lat)))
        out[:, t + 1] = np.c_[lat, lon]
    return out

def exposure_field(grid, ensembles, safety_cells=2):
    """P(an iceberg is in/near a cell at some time in the horizon), max over time, dilated."""
    H, W = grid.shape; p = np.zeros((H, W)); dlat = grid.lat[1] - grid.lat[0]; dlon = grid.lon[1] - grid.lon[0]
    for tr in ensembles:
        n, T, _ = tr.shape
        for t in range(T):
            ii = np.round((tr[:, t, 0] - grid.lat[0]) / dlat).astype(int); jj = np.round((tr[:, t, 1] - grid.lon[0]) / dlon).astype(int)
            ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
            h = np.zeros((H, W)); np.add.at(h, (ii[ok], jj[ok]), 1.0 / n)
            p = np.maximum(p, h)
    p = maximum_filter(p, size=2 * safety_cells + 1)
    return np.clip(p * 4, 0, 1)        # x4: a cell holding 25% of particles counts as certain

def validate_one_day(grid, lat0, lon0, lat1, lon1, u_wind, v_wind, u_cur=None, v_cur=None):
    """Error (km) of the ensemble-mean 24 h position vs an observed NIC position."""
    tr = drift_ensemble(grid, lat0, lon0, u_wind, v_wind, u_cur, v_cur, hours=24)
    m = tr[:, -1].mean(0)
    return float(haversine_nm(m[0], m[1], lat1, lon1) * 1.852)
