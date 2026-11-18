"""Run:  python -m unittest discover -s tests -v      (no pytest needed)

Covers the numpy/scipy/sklearn parts. Streamlit UI and real-data loaders (xarray/pyproj) need
their packages and data files and are exercised by running the app.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odysea import nic                                           # noqa: E402
from odysea.config import (LEGS, VESSEL, ICE_LIMIT, LABEL_SAFEST, LABEL_FASTEST, LABEL_FUEL,   # noqa: E402
                           LABEL_BALANCED, LABEL_CORRIDOR, LABEL_ALT)
from odysea.drift import drift_ensemble                          # noqa: E402
from odysea.fuel import speed_kn, fuel_t_per_h, fuel_t_per_nm    # noqa: E402
from odysea.geo import haversine_nm                              # noqa: E402
from odysea.routing import plan, RoutingGraph, alerts           # noqa: E402
from odysea.synthetic import make_scenario                       # noqa: E402

KNOWN_LABELS = {LABEL_SAFEST, LABEL_FASTEST, LABEL_FUEL, LABEL_BALANCED, LABEL_CORRIDOR, LABEL_ALT}


class FuelModel(unittest.TestCase):
    def test_ice_is_never_cheaper_than_open_water(self):
        c = np.linspace(0, 1, 21)
        self.assertTrue(np.all(np.diff(speed_kn(c, VESSEL)) <= 1e-12), "speed must not rise with ice")
        self.assertTrue(np.all(np.diff(fuel_t_per_h(c, VESSEL)) >= -1e-12), "fuel/h must not fall with ice")
        self.assertTrue(np.all(np.diff(fuel_t_per_nm(c, VESSEL)) >= -1e-12), "fuel/nm must not fall with ice")
        self.assertGreater(fuel_t_per_nm(0.9, VESSEL), 2.0 * fuel_t_per_nm(0.0, VESSEL))

    def test_speed_bounds(self):
        self.assertAlmostEqual(float(speed_kn(0.0, VESSEL)), VESSEL.v_open_kn)
        self.assertAlmostEqual(float(speed_kn(1.0, VESSEL)), VESSEL.v_heavy_kn)

    def test_nan_ice_is_treated_as_heavy_not_open_water(self):
        self.assertAlmostEqual(float(speed_kn(np.nan, VESSEL)), VESSEL.v_heavy_kn)


class DriftTiming(unittest.TestCase):
    def setUp(self):
        self.g = make_scenario(days=3)["grid"]
        shape = self.g.shape
        self.u = np.stack([np.zeros(shape), np.full(shape, 20.0)])   # day 0 calm, day 1 gale
        self.v = np.zeros_like(self.u)

    def _disp_km(self, hours):
        tr = drift_ensemble(self.g, -60.0, 50.0, self.u, self.v, hours=hours, n=50, seed=1)
        m = tr[:, -1].mean(0)
        return float(haversine_nm(-60.0, 50.0, m[0], m[1]) * 1.852)

    def test_daily_forcing_is_not_advanced_every_3h_step(self):
        self.assertLess(self._disp_km(12), 5.0)      # all 12 h are still inside day 0 (calm)

    def test_second_day_forcing_is_used_after_24h(self):
        self.assertGreater(self._disp_km(48), 10.0)


class Routing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = make_scenario()
        cls.g = cls.sc["grid"]
        ice = np.nan_to_num(cls.sc["ice"][10], nan=np.nan)
        cls.ice_ens = np.stack([ice, ice, ice])
        cls.G = RoutingGraph(cls.g)

    def test_routes_found_labelled_and_never_enter_blocked_cells(self):
        for leg in LEGS.values():
            res = plan(self.g, leg.start, leg.end, self.ice_ens, graph=self.G)
            self.assertTrue(res["routes"], f"no route for {leg.name}")
            for r in res["routes"]:
                self.assertIn(r["label"], KNOWN_LABELS)
                ii = np.abs(self.g.lat[:, None] - np.array(r["lat"])[None, :]).argmin(0)
                jj = np.abs(self.g.lon[:, None] - np.array(r["lon"])[None, :]).argmin(0)
                cells = np.nan_to_num(self.ice_ens[0][ii, jj], nan=1.0)
                self.assertTrue((cells < ICE_LIMIT).all(), "route crosses ice at/above the limit")

    def test_ice_costs_fuel_along_the_safest_route_pair(self):
        leg = LEGS["leg1"]
        res = plan(self.g, leg.start, leg.end, self.ice_ens, graph=self.G)
        for r in res["routes"]:
            self.assertGreater(r["m"]["fuel_t"], 0)
            self.assertGreater(r["m"]["hours"], 0)

    def test_unreachable_when_everything_is_ice(self):
        full = np.ones_like(self.ice_ens)
        leg = LEGS["leg2"]
        res = plan(self.g, leg.start, leg.end, full, graph=self.G)
        self.assertEqual(res["routes"], [])

    def test_bad_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            plan(self.g, LEGS["leg1"].start, LEGS["leg1"].end, self.ice_ens[0], graph=self.G)

    def test_gale_cells_are_penalized_and_reported(self):
        wind = np.full(self.g.shape, 18.0)
        res = plan(self.g, LEGS["leg2"].start, LEGS["leg2"].end, self.ice_ens,
                   wind=wind, graph=self.G)
        self.assertTrue(res["routes"])
        self.assertGreater(res["routes"][0]["m"]["gale_nm"], 0)

    def test_hurricane_force_cells_are_blocked(self):
        wind = np.full(self.g.shape, 33.0)
        res = plan(self.g, LEGS["leg2"].start, LEGS["leg2"].end, self.ice_ens,
                   wind=wind, graph=self.G)
        self.assertEqual(res["routes"], [])

    def test_alerts_flag_nearby_iceberg(self):
        leg = LEGS["leg3"]
        res = plan(self.g, leg.start, leg.end, self.ice_ens, graph=self.G)
        r = res["routes"][0]
        berg = ("TEST-BERG", r["lat"][len(r["lat"]) // 2], r["lon"][len(r["lon"]) // 2])
        self.assertTrue(any(level == "DANGER" for level, _ in alerts(r, [berg])))


class NicParsing(unittest.TestCase):
    def test_dms_and_decimal(self):
        self.assertAlmostEqual(nic.parse_one("69° 24.31' S", "lat"), -(69 + 24.31 / 60), places=6)
        self.assertAlmostEqual(nic.parse_one("76 11.84 E", "lon"), 76 + 11.84 / 60, places=6)
        self.assertEqual(nic.parse_one("-66.5", "lat"), -66.5)
        self.assertIsNone(nic.parse_one("95 S", "lat"))
        self.assertIsNone(nic.parse_one("garbage", "lat"))

    def test_width_column_is_not_mistaken_for_the_name_column(self):
        df = pd.DataFrame({"Width": [10], "Iceberg": ["A-76B"], "Latitude": ["69° 24.31' S"],
                           "Longitude": ["76° 11.84' E"]})
        (name, lat, lon), = nic.to_bergs(df)
        self.assertEqual(name, "A-76B")
        self.assertAlmostEqual(lat, -69.4052, places=3)

    def test_byu_history_dates_are_parsed_as_tracking_history(self):
        import tempfile, zipfile
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "history.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("a23a.csv", "date,nic_1,nic_2\n2001001,-65,20\n2001365,-66,21\n")
            index = nic.load_byu_history_index(archive)
            item = nic.history_for_berg("A23A", index)
            self.assertEqual(item["first_observed"], "2001-01-01")
            self.assertEqual(item["last_observed"], "2001-12-31")
            self.assertEqual(item["observations"], 2)

    def test_combined_location_column_and_skip_count(self):
        df = pd.DataFrame({"Iceberg": ["X", "Y"], "Location": ["70° 0' S 10° 0' E", "n/a"]})
        bergs, skipped = nic.to_bergs(df, return_skipped=True)
        self.assertEqual(len(bergs), 1)
        self.assertEqual(skipped, 1)


class LivePipeline(unittest.TestCase):
    """run_leg_live with the xarray/pyproj loader replaced by a fake NOAA-like scenario."""

    @staticmethod
    def _fake_loader(cover_mask=None):
        sc0 = make_scenario()
        g = sc0["grid"]
        ice = sc0["ice"][10]
        cover = np.isfinite(ice)
        if cover_mask is not None:
            cover &= cover_mask(g)
        ice = np.where(cover, ice, np.nan)
        T = 2
        return {
            "grid": g, "ice": ice[None], "ice_stdev": np.zeros(g.shape), "ice_coverage": cover,
            "ice_qa": np.zeros(g.shape, np.uint8), "ice_time": "2026-11-16T00:00:00",
            "ice_source": "FAKE", "ice_is_forecast": False,
            "current_u": np.zeros((T,) + g.shape), "current_v": np.zeros((T,) + g.shape),
            "current_coverage": np.ones((T,) + g.shape, bool), "current_times": np.arange(T),
            "current_depth_m": 0.5, "current_source": "FAKE", "wind_u": None, "wind_v": None,
            "land": ~cover, "synthetic": False,
        }

    def test_safe_route_found(self):
        from odysea import pipeline
        with mock.patch("odysea.marine.load_live_scenario", lambda *a, **k: self._fake_loader()):
            out = pipeline.run_leg_live("leg3")
        self.assertEqual(out["route_status"], "NO SAFE LIVE ROUTE" if not out["result"]["routes"] else "SAFE ROUTE FOUND")
        self.assertTrue(out["result"]["routes"])
        self.assertIn("ice_time", out["metadata"])

    def test_unreachable_endpoint_returns_rejection_not_exception(self):
        from odysea import pipeline
        far = lambda g: g.LAT < -50          # no coverage anywhere near Cape Town
        with mock.patch("odysea.marine.load_live_scenario", lambda *a, **k: self._fake_loader(far)):
            out = pipeline.run_leg_live("leg1")
        self.assertEqual(out["route_status"], "NO SAFE LIVE ROUTE")
        self.assertEqual(out["result"]["routes"], [])
        self.assertIn("rejected", out["result"]["note"].lower())

    def test_unknown_leg(self):
        from odysea import pipeline
        with self.assertRaises(KeyError):
            pipeline.run_leg_live("nope")


class DemoPipeline(unittest.TestCase):
    def test_end_to_end_with_extra_real_bergs(self):
        from odysea.pipeline import run_leg
        out = run_leg("leg3", horizon=3, members=3, p_block=0.25,
                      extra_bergs=[("REAL-1", -66.0, 30.0)])
        self.assertEqual(out["data_mode"], "SYNTHETIC DEMO")
        self.assertEqual(len(out["bergs"]), len(out["trajectories"]))
        self.assertTrue(np.isfinite(out["forecast_skill"]["model_mae"]))
        self.assertTrue(out["result"]["routes"])
        for r in out["result"]["routes"]:
            self.assertIn(r["label"], KNOWN_LABELS)


if __name__ == "__main__":
    unittest.main()
