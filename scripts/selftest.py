"""End-to-end smoke test on SYNTHETIC data (no downloads).   python scripts/selftest.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from odysea.synthetic import make_scenario
from odysea.forecast import IceForecaster
from odysea.drift import drift_ensemble, exposure_field, validate_one_day
from odysea.routing import plan, RoutingGraph, alerts
from odysea.config import LEGS

sc = make_scenario(); g = sc["grid"]
t0 = time.time()
fc = IceForecaster(n_members=5).fit(sc["ice"], sc["sst"], sc["u"], sc["v"], g, sc["land"], range(0, 34))
print("train s", round(time.time() - t0, 1))
ev = fc.evaluate(sc["ice"], sc["sst"], sc["u"], sc["v"], range(34, 43)); print("eval", ev)
assert np.isfinite(ev["model_mae"]) and np.isfinite(ev["persistence_mae"])

T = 34
ens = fc.forecast(sc["ice"][T], [(sc["sst"][T + h], sc["u"][T + h], sc["v"][T + h]) for h in range(5)], 5)
ice_ens = ens.max(1); print("ens", ice_ens.shape, "spread", float(np.nanmean(ice_ens.std(0))))
wind = np.hypot(sc["u"][T], sc["v"][T])
trs = [drift_ensemble(g, la, lo, sc["u"][T:T + 4], sc["v"][T:T + 4], hours=72) for _, la, lo in sc["icebergs"]]
bp = exposure_field(g, trs); print("berg P max", bp.max())

G = RoutingGraph(g)
for lid, leg in LEGS.items():
    t0 = time.time(); r = plan(g, leg.start, leg.end, ice_ens, wind, bp, graph=G); dt = time.time() - t0
    print(lid, len(r["routes"]), "routes", round(dt, 2), "s")
    assert r["routes"], f"{lid}: no routes"
    for x in r["routes"]:
        m = x["m"]
        print("  ", x["label"], round(m["distance_nm"]), "nm", round(m["hours"]), "h", round(m["fuel_t"]), "t",
              "maxice", round(m["max_ice"], 2), "pexc", m["p_exceed_limit"])
        assert m["fuel_t"] > 0 and m["hours"] > 0
    print("  alerts", alerts(r["routes"][0], sc["icebergs"]))
print("drift err km (self-consistency, synthetic)",
      round(validate_one_day(g, -66.5, 60, -66.5, 60.1, sc["u"][T], sc["v"][T]), 1))
print("SELFTEST OK")
