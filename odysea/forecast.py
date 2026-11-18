"""Ensemble residual sea-ice forecaster (predict daily CHANGE, not the level).
Bagged gradient-boosted trees (XGBoost if installed, else sklearn). Ensemble spread +
initial-state perturbation give a probabilistic ice field the router consumes."""
import pickle
import numpy as np
from scipy.ndimage import uniform_filter

def _features(ice, sst, u, v, lat2d):
    ice = np.nan_to_num(ice)
    i9 = uniform_filter(ice, 9)
    X = np.stack([ice, uniform_filter(ice, 3), i9, ice - i9, sst, u, v, lat2d], -1)
    return X.reshape(-1, X.shape[-1]), i9

def _edge_mask(i9, land):
    return ((i9 > 0.02) & (i9 < 0.98) & ~land).ravel()

def _make_model(seed):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.06, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", random_state=seed, n_jobs=-1)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=250, max_depth=5, learning_rate=0.06, random_state=seed)

class IceForecaster:
    def __init__(self, n_members=5, seed=0):
        self.n_members, self.seed, self.models = n_members, seed, []

    def fit(self, ice, sst, u, v, grid, land, train_days):
        self.lat2d, self.land = grid.LAT, land
        Xd, yd = {}, {}
        for t in train_days:
            X, i9 = _features(ice[t], sst[t], u[t], v[t], self.lat2d)
            m = _edge_mask(i9, land)
            Xd[t] = X[m]; yd[t] = (np.nan_to_num(ice[t + 1]) - np.nan_to_num(ice[t])).ravel()[m]
        rng = np.random.default_rng(self.seed); self.models = []
        for k in range(self.n_members):
            boot = rng.choice(list(train_days), len(train_days))
            self.models.append(_make_model(self.seed + k).fit(np.vstack([Xd[t] for t in boot]),
                                                              np.concatenate([yd[t] for t in boot])))
        return self

    def _step(self, model, s, f):
        X, i9 = _features(s, *f, self.lat2d)
        m = _edge_mask(i9, self.land); d = np.zeros(X.shape[0])
        if m.any(): d[m] = model.predict(X[m])
        return np.clip(s + d.reshape(s.shape), 0, 1)

    def evaluate(self, ice, sst, u, v, test_days):
        mae_m, mae_p = [], []
        for t in test_days:
            X, i9 = _features(ice[t], sst[t], u[t], v[t], self.lat2d); m = _edge_mask(i9, self.land)
            true = (np.nan_to_num(ice[t + 1]) - np.nan_to_num(ice[t])).ravel()[m]
            pred = np.mean([mo.predict(X[m]) for mo in self.models], 0)
            mae_m.append(np.abs(pred - true).mean()); mae_p.append(np.abs(true).mean())
        return dict(model_mae=float(np.mean(mae_m)), persistence_mae=float(np.mean(mae_p)),
                    skill_vs_persistence=float(1 - np.mean(mae_m) / np.mean(mae_p)))

    def forecast(self, ice0, forcing, horizon, init_noise=0.02, seed=1):
        """forcing: list of (sst,u,v) per lead day. Returns (members, horizon, H, W)."""
        rng = np.random.default_rng(seed)
        out = np.empty((self.n_members, horizon) + ice0.shape)
        base = np.nan_to_num(ice0)
        for k, mo in enumerate(self.models):
            s = base.copy()
            edge = (s > 0.02) & (s < 0.98)
            s = np.clip(s + rng.normal(0, init_noise, s.shape) * edge, 0, 1)
            for h in range(horizon):
                s = self._step(mo, s, forcing[min(h, len(forcing) - 1)]); out[k, h] = s
        out[:, :, self.land] = np.nan
        return out

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        """Only load files you created yourself: pickle can execute arbitrary code."""
        with open(path, "rb") as fh:
            return pickle.load(fh)
