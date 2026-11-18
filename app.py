
import inspect
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import folium
import matplotlib.cm as cm

from datetime import date, timedelta
from pathlib import Path
from html import escape
from streamlit_folium import st_folium

from odysea.pipeline import run_leg, run_leg_live
from odysea.config import (
    LEGS, LABEL_SAFEST, LABEL_FASTEST, LABEL_FUEL, LABEL_BALANCED, LABEL_CORRIDOR,
)
from odysea.routing import alerts
from odysea import nic
from odysea.copernicus import load_copernicus_observation


def stretch(fn):
    """Full-width kwargs that work on old AND new Streamlit.

    `use_container_width` is deprecated in newer Streamlit in favour of width="stretch".
    We inspect the callable instead of guessing, so a Streamlit upgrade can't break the app.
    """
    params = inspect.signature(fn).parameters
    if fn is st_folium:                       # st_folium's `width` is pixels, never "stretch"
        return {"use_container_width": True} if "use_container_width" in params else {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


# Cache network / disk / model work so ordinary widget clicks don't refetch or replan.
@st.cache_data(ttl=3600, show_spinner=False)
def cached_nic_bergs():
    return nic.to_bergs(nic.fetch_current())


@st.cache_data(show_spinner=False)
def cached_copernicus():
    return load_copernicus_observation()


@st.cache_data(ttl=86400, show_spinner=False)
def cached_berg_history():
    history_file = Path(__file__).resolve().parent / "data" / "live" / "byu_iceberg_consolidated_v8.zip"
    if not history_file.exists():
        return {}
    return nic.load_byu_history_index(history_file)


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_demo_run(leg_id, horizon, members, p_block, nc_path, bergs):
    return run_leg(leg_id, horizon, members, p_block, nc_path=nc_path, extra_bergs=list(bergs))


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_live_run(leg_id, p_block):
    return run_leg_live(leg_id, p_block=p_block)


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ODySEA | Antarctic Command Centre",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM DARK THEME
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #07111f;
    color: #e5edf7;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 100%;
}
section[data-testid="stSidebar"] {
    background-color: #0c1b2b;
    border-right: 1px solid #203b55;
}
section[data-testid="stSidebar"] * {
    color: #dce8f5;
}
h1, h2, h3 {
    color: #f1f7ff !important;
}
p, label {
    color: #c3d2e3;
}
.command-header {
    background: linear-gradient(110deg, #10263d, #0b1929);
    border: 1px solid #24415c;
    border-radius: 14px;
    padding: 22px 25px;
    margin-bottom: 20px;
}
.command-title {
    color: #f0f7ff;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: 2px;
}
.command-subtitle {
    color: #8daac5;
    font-size: 12px;
    margin-top: 6px;
    letter-spacing: 1px;
}
.status-pill {
    display: inline-block;
    background: #103b35;
    border: 1px solid #176b5b;
    color: #55e2bc;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    margin-top: 13px;
}
div[data-testid="stMetric"] {
    background: #102238;
    border: 1px solid #23415e;
    border-radius: 12px;
    padding: 15px;
    min-height: 105px;
}
div[data-testid="stMetricLabel"] {
    color: #91acc7 !important;
}
div[data-testid="stMetricValue"] {
    color: #f1f7ff !important;
    font-size: 25px;
    font-weight: 700;
}
.panel {
    background: #0d1c2d;
    border: 1px solid #203b55;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 13px;
}
.panel-heading {
    color: #e8f2ff;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 12px;
}
.small-label {
    color: #7f9bb6;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.small-value {
    color: #f0f7ff;
    font-size: 22px;
    font-weight: 700;
    margin-top: 5px;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0c1b2b;
    border-bottom: 1px solid #24415c;
    gap: 12px;
}
.stTabs [data-baseweb="tab"] {
    color: #9cb2c9;
    padding: 12px 15px;
}
.stTabs [aria-selected="true"] {
    color: #4dc9f6 !important;
    border-bottom-color: #4dc9f6 !important;
}
.stButton button {
    background: #087eaa;
    color: white;
    border: 1px solid #1599c5;
    border-radius: 8px;
    font-weight: 650;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #23415e;
    border-radius: 10px;
}
hr {
    border-color: #203b55;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. HEADER
# =========================================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown("""
    <div class="command-header">
        <div class="command-title">🧊 ODYSEA</div>
        <div class="command-subtitle">
            ANTARCTIC MARITIME INTELLIGENCE &
            PROBABILISTIC ROUTE DECISION SUPPORT
        </div>
        <div>
            <span class="status-pill">● COMMAND SYSTEM ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_right:
    st.markdown("""
    <div style="text-align:right;padding-top:15px;">
        <div style="color:#4dc9f6;font-size:13px;font-weight:700;">
            ANTARCTIC OPERATIONS
        </div>
        <div style="color:#7f9bb6;font-size:11px;">
            MARITIME INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 4. SIDEBAR CONFIGURATION
# =========================================================

with st.sidebar:
    st.markdown("## ⚙️ Mission Control")
    st.caption("Antarctic voyage configuration")
    st.divider()

    st.markdown("### 🚢 Route Selection")

    leg_id = st.selectbox(
        "Select voyage",
        list(LEGS),
        format_func=lambda k: LEGS[k].name
    )

    route_mode = st.radio(
        "Routing data mode",
        ["Existing Demo Routes", "Live NOAA Routes"],
        index=0,
        help=(
            "Demo mode uses the existing synthetic route pipeline. "
            "Live mode uses the downloaded NOAA/NSIDC sea-ice "
            "observation and does not substitute demo routes."
        )
    )

    st.divider()
    st.markdown("### 🧊 Ice Prediction")

    horizon = st.slider("Forecast horizon (days)", 1, 7, 5)
    members = st.slider("Ensemble members", 3, 10, 5)

    p_block = st.slider(
        "Ice blocking probability",
        0.05, 0.90, 0.25, 0.05
    )

    st.divider()
    st.markdown("### 📡 Data Sources")

    nc = st.text_input(
        "Processed NetCDF path",
        placeholder="Blank = synthetic demo"
    )

    use_nic = st.checkbox(
        "Enable live USNIC iceberg data",
        value=True,
        help="Fetches current iceberg records from the USNIC connector."
    )

    st.divider()
    st.markdown("### 🛰️ Satellite Imagery")

    satellite_date = st.date_input(
        "NASA MODIS observation date",
        value=date.today() - timedelta(days=2),
        min_value=date(2000, 2, 24),
        max_value=date.today()
    )

    st.caption("NASA GIBS • MODIS Terra Corrected Reflectance True Color")

    st.divider()
    st.markdown("### 🗺️ Map Layers")

    show_ice = st.checkbox("ML / NOAA sea-ice layer", True)
    show_copernicus = st.checkbox("Copernicus observed sea ice", True)
    show_wind = st.checkbox("Gale zones", True)
    show_bergs = st.checkbox("Iceberg trajectories", True)
    show_routes = st.checkbox("Route alternatives", True)

    st.divider()
    if st.button("🔄 Refresh data (clear caches)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# =========================================================
# 5. FETCH LIVE USNIC DATA
# =========================================================

extra = []
nic_error = None

if use_nic:
    try:
        extra = cached_nic_bergs()
        st.sidebar.success(f"✓ {len(extra)} USNIC iceberg records loaded")
    except Exception as e:
        nic_error = str(e)
        st.sidebar.error(f"USNIC fetch failed: {e}")
else:
    st.sidebar.info("USNIC live iceberg feed is disabled.")


berg_history_index = {}
try:
    berg_history_index = cached_berg_history()
except Exception as history_exc:
    st.sidebar.warning(f"Historical iceberg tracks unavailable: {history_exc}")



# =========================================================
# 6. LOAD COPERNICUS SATELLITE OBSERVATION
# =========================================================

cop_obs = None
cop_error = None

if show_copernicus:
    try:
        cop_obs = cached_copernicus()
        st.sidebar.success(
            f"✓ Copernicus observation loaded: {cop_obs['date']}"
        )
    except Exception as e:
        cop_error = str(e)
        st.sidebar.warning(f"Copernicus unavailable: {e}")


# =========================================================
# 7. RUN SELECTED ROUTING PIPELINE
# =========================================================

try:
    with st.spinner("🧠 Initializing navigation and planning routes..."):

        if route_mode == "Live NOAA Routes":
            out = dict(cached_live_run(leg_id, float(p_block)))
        else:
            # The slider value is used as-is (an earlier version silently raised it to 0.50).
            out = dict(cached_demo_run(
                leg_id,
                int(horizon),
                int(members),
                float(p_block),
                nc or None,
                tuple(extra),
            ))

    sc = out.get("scenario", {})
    g = sc.get("grid")

    if g is None:
        raise ValueError("The selected pipeline did not return a navigation grid.")

    # Support both pipeline result formats.
    res = out.get("result")

    if res is None:
        res = {"routes": out.get("routes", [])}

    routes = res.get("routes", [])

    # Normalize fields shared by the dashboard.
    out.setdefault("forecast_skill", None)
    out.setdefault("wind", sc.get("wind"))
    out.setdefault("bergs", [])
    out.setdefault("trajectories", [])

    if "ice_ens" not in out:
        source_ice = out.get("ice", sc.get("ice"))

        if source_ice is None:
            out["ice_ens"] = np.full((1,) + g.shape, np.nan)
        else:
            source_ice = np.asarray(source_ice, dtype=float)

            if source_ice.ndim == 2:
                source_ice = source_ice[np.newaxis, ...]

            out["ice_ens"] = source_ice

    if route_mode == "Existing Demo Routes" and not out["bergs"]:
        out["bergs"] = extra

    # Only use trajectories when they correspond to iceberg records.
    if len(out["trajectories"]) != len(out["bergs"]):
        out["trajectories"] = []

except FileNotFoundError as e:
    st.error(
        f"Required data file is missing: {e}. "
        "Live mode needs NOAA sea ice, Copernicus currents, ECMWF wind forecast, "
        "and BYU iceberg history files in data/live (see README)."
    )
    st.stop()
except Exception as e:
    st.error(f"Navigation pipeline failed: {e}")
    st.exception(e)
    st.stop()


# =========================================================
# 8. DATA SOURCE STATUS
# =========================================================

if route_mode == "Existing Demo Routes":
    if out.get("data_mode") != "SYNTHETIC DEMO":
        st.info(
            f"DATA MODE: {out.get('data_mode')} — routed from the NetCDF you "
            "supplied. Iceberg positions below are USNIC records drifted with "
            "that file's wind; no ocean-current forcing is applied."
        )

else:
    live_status = out.get("route_status", "LIVE DATA LOADED")

    st.info(
        f"DATA MODE: {out.get('data_mode', 'LIVE MARINE DATA')} "
        " | NOAA sea ice: observed field • ECMWF wind: forecast"
    )

    if live_status == "SAFE ROUTE FOUND" and routes:
        st.success("LIVE ROUTING STATUS: SAFE ROUTE FOUND")
    else:
        st.error(
            "NO SAFE LIVE ROUTE — no route passed the live "
            "observation and routing safety checks. "
            "No synthetic route has been substituted."
        )

    metadata = out.get("metadata", {})

    if metadata:
        if metadata.get("ice_time"):
            st.caption(f"NOAA/NSIDC observation time: {metadata['ice_time']}")
        if metadata.get("ice_source"):
            st.caption(f"Sea-ice source: {metadata['ice_source']}")
        if metadata.get("current_source"):
            st.caption(f"Ocean-current source: {metadata['current_source']}")
        if metadata.get("wind_source"):
            st.caption(f"Wind source: {metadata['wind_source']} • forecast through {metadata.get('wind_valid_until', 'unknown')}")


if res.get("note"):
    st.warning(res["note"])


# =========================================================
# 9. DATA FEED INTEGRITY
# =========================================================

st.markdown("### 📡 Data Feed Integrity")

feed1, feed2, feed3, feed4 = st.columns(4)

feed1.metric("Satellite imagery", "NASA GIBS")
feed1.caption(f"Selected imagery date: {satellite_date.isoformat()}")

if use_nic and extra:
    feed2.metric("Iceberg observations", "USNIC LIVE")
    feed2.caption(f"{len(extra)} records returned by connector")
elif use_nic:
    feed2.metric("Iceberg observations", "UNAVAILABLE")
    feed2.caption(nic_error or "No records returned.")
else:
    feed2.metric("Iceberg observations", "DISABLED")
    feed2.caption("Live USNIC fetch is switched off.")

if route_mode == "Live NOAA Routes":
    feed3.metric("Navigation input", "NOAA/NSIDC")
    feed3.caption("Observed sea ice; not a forecast.")
else:
    feed3.metric("Navigation input", "SYNTHETIC DEMO")
    feed3.caption("Existing demo pipeline is selected.")

if cop_obs is not None:
    feed4.metric("Copernicus sea ice", "HISTORICAL")
    feed4.caption(f"Observation: {cop_obs['date']}")
elif show_copernicus:
    feed4.metric("Copernicus sea ice", "UNAVAILABLE")
    feed4.caption(cop_error or "Observation could not be loaded.")
else:
    feed4.metric("Copernicus sea ice", "DISABLED")
    feed4.caption("Copernicus layer is switched off.")

if cop_obs is not None:
    st.info(
        "Copernicus data is displayed as an independent historical "
        "satellite observation. It is not automatically substituted "
        "for the navigation pipeline's input."
    )


# =========================================================
# 10. PREPARE DASHBOARD METRICS
# =========================================================

best_route = routes[0] if routes else None
iceberg_count = len(out["bergs"])

if best_route:
    metrics = best_route.get("m", {})
    distance_text = f"{metrics.get('distance_nm', 0):,.0f} NM"
    hours_text = f"{metrics.get('hours', 0):.1f} hrs"
    fuel_text = f"{metrics.get('fuel_t', 0):.1f} t"
    risk_text = f"{metrics.get('p_exceed_limit', 0):.1%}"
else:
    distance_text = "N/A"
    hours_text = "N/A"
    fuel_text = "N/A"
    risk_text = "N/A"


# =========================================================
# 11. TOP KPI CARDS
# =========================================================

st.markdown("### 📊 Mission Overview")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("🧊 Icebergs", iceberg_count)
k2.metric("📍 Distance", distance_text)
k3.metric("⏱️ Travel Time", hours_text)
k4.metric("⛽ Estimated Fuel", fuel_text)
k5.metric("⚠️ Ice Risk", risk_text)

st.divider()


# =========================================================
# 12. ANTARCTIC OPERATIONS MAP
# =========================================================

st.markdown("### 🗺️ Antarctic Operations Map")

m = folium.Map(
    location=[-70, 20],
    zoom_start=3,
    tiles=None,
    control_scale=True
)

gibs_date = satellite_date.isoformat()

folium.TileLayer(
    tiles=(
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "MODIS_Terra_CorrectedReflectance_TrueColor/default/"
        f"{gibs_date}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
    ),
    attr="NASA GIBS / MODIS Terra",
    name=f"🛰️ NASA MODIS True Color ({gibs_date})",
    overlay=False,
    control=True,
    show=True,
    max_zoom=9
).add_to(m)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri World Imagery",
    name="🛰️ Esri Satellite Imagery",
    overlay=False,
    control=True,
    show=False,
    max_zoom=19
).add_to(m)

folium.TileLayer(
    tiles="CartoDB dark_matter",
    name="🌑 Dark Map",
    overlay=False,
    control=True,
    show=False
).add_to(m)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="🗺️ Standard Map",
    overlay=False,
    control=True,
    show=False
).add_to(m)


# =========================================================
# 13. SEA-ICE OVERLAY
# =========================================================

bounds = [
    [float(g.lat[0]), float(g.lon[0])],
    [float(g.lat[-1]), float(g.lon[-1])]
]

if show_ice:
    ice_array = np.asarray(out["ice_ens"], dtype=float)

    if ice_array.ndim == 3:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # land cells are all-NaN
            ice = np.nanmean(ice_array, axis=0)
    else:
        ice = ice_array

    ice = np.clip(np.nan_to_num(ice, nan=0.0), 0.0, 1.0)

    if ice.shape == g.shape:
        rgba = cm.Blues(ice).copy()
        rgba[..., 3] = np.where(ice > 0.05, 0.6, 0.0)

        layer_name = (
            "🧊 NOAA Observed Sea Ice"
            if route_mode == "Live NOAA Routes"
            else "🧊 ML Forecast Sea Ice"
        )

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=bounds,
            origin="lower",
            name=layer_name,
            opacity=0.7,
            interactive=True,
            zindex=5
        ).add_to(m)


# =========================================================
# 14. COPERNICUS OBSERVED SEA-ICE OVERLAY
# =========================================================

if show_copernicus and cop_obs is not None:
    observed_ice = np.asarray(cop_obs["ice"])
    observed_confidence = np.asarray(cop_obs["confidence"])
    observed_valid = np.asarray(cop_obs["valid_mask"], dtype=bool)

    cop_rgba = cm.Blues(
        np.nan_to_num(observed_ice, nan=0.0)
    ).copy()

    cop_rgba[..., 3] = np.where(
        observed_valid & (observed_ice > 0.01),
        0.85 * observed_confidence,
        0.0
    )

    cop_bounds = [
        [
            float(cop_obs["latitude"][0]),
            float(cop_obs["longitude"][0])
        ],
        [
            float(cop_obs["latitude"][-1]),
            float(cop_obs["longitude"][-1])
        ]
    ]

    folium.raster_layers.ImageOverlay(
        image=cop_rgba,
        bounds=cop_bounds,
        origin="lower",
        name="🛰️ Copernicus Observed Sea Ice",
        opacity=0.9,
        interactive=True,
        cross_origin=False,
        zindex=10
    ).add_to(m)

    cop_valid_count = int(observed_valid.sum())

    cop_mean = (
        float(np.nanmean(observed_ice[observed_valid]) * 100)
        if cop_valid_count else 0.0
    )

    st.caption(
        f"Copernicus historical observation: {cop_obs['date']} | "
        f"Valid pixels: {cop_valid_count:,} | "
        f"Mean concentration: {cop_mean:.2f}%"
    )


# =========================================================
# 15. GALE ZONE OVERLAY
# =========================================================

wind_data = out.get("wind")

if show_wind and wind_data is not None:
    wind_array = np.asarray(wind_data, dtype=float)

    # Display only if wind grid matches the navigation grid.
    if wind_array.shape == g.shape:
        # Separate severity overlays so users can distinguish gale, storm and
        # hurricane-force exclusion cells on the map.
        for layer_name, mask, rgba in (
            ("🌬️ Gale (17.2–24.5 m/s)", (wind_array >= 17.2) & (wind_array < 24.5), (1.0, 0.55, 0.0, 0.30)),
            ("🔴 Storm (24.5–32.9 m/s)", (wind_array >= 24.5) & (wind_array < 32.9), (1.0, 0.0, 0.0, 0.38)),
            ("⛔ Hurricane force (≥32.9 m/s)", wind_array >= 32.9, (0.35, 0.0, 0.0, 0.55)),
        ):
            overlay = np.zeros(wind_array.shape + (4,))
            overlay[mask] = rgba
            folium.raster_layers.ImageOverlay(
                image=overlay, bounds=bounds, origin="lower",
                name=layer_name, opacity=0.8, zindex=6
            ).add_to(m)


# =========================================================
# 16. ICEBERG MARKERS AND TRAJECTORIES
# =========================================================

if show_bergs:
    trajectories = out.get("trajectories", [])

    for idx, berg in enumerate(out["bergs"]):
        try:
            name, la, lo = berg
        except (ValueError, TypeError):
            continue

        folium.CircleMarker(
            location=[float(la), float(lo)],
            radius=7,
            color="#ff4545",
            weight=2,
            fill=True,
            fill_color="#ff4545",
            fill_opacity=0.9,
            tooltip=f"Iceberg: {name}"
        ).add_to(m)

        if idx >= len(trajectories):
            continue

        tr = np.asarray(trajectories[idx])

        if tr.ndim != 3 or tr.shape[-1] != 2:
            continue

        mean_track = tr.mean(axis=0)

        folium.PolyLine(
            mean_track.tolist(),
            color="#ff6666",
            weight=2,
            dash_array="5",
            tooltip=f"{name} - Mean predicted drift"
        ).add_to(m)

        final_lat_min = float(tr[:, -1, 0].min())
        final_lat_max = float(tr[:, -1, 0].max())
        final_lon_min = float(tr[:, -1, 1].min())
        final_lon_max = float(tr[:, -1, 1].max())

        folium.Rectangle(
            bounds=[
                [final_lat_min, final_lon_min],
                [final_lat_max, final_lon_max]
            ],
            color="#ff8888",
            weight=1,
            opacity=0.5,
            fill=False,
            tooltip=f"{name} - Final drift uncertainty extent"
        ).add_to(m)


# =========================================================
# 17. ROUTE ALTERNATIVES
# =========================================================

palette = {
    LABEL_SAFEST: "#00d68f",
    LABEL_FASTEST: "#32aaff",
    LABEL_FUEL: "#b38cff",
    LABEL_BALANCED: "#ffad42",
    LABEL_CORRIDOR: "#00d68f",
}

if show_routes:
    for route in routes:
        folium.PolyLine(
            list(zip(route["lat"], route["lon"])),
            color=palette.get(route.get("label"), "#ffffff"),
            weight=5,
            opacity=0.9,
            tooltip=f"Route: {route.get('label', 'Alternative')}"
        ).add_to(m)


# =========================================================
# 18. MAP LAYER CONTROL
# =========================================================

folium.LayerControl(
    collapsed=False,
    position="topright"
).add_to(m)


# =========================================================
# 19. COMMAND CENTRE MAP LAYOUT
# =========================================================

map_col, info_col = st.columns([3.2, 1], gap="medium")

with map_col:
    st.markdown(
        '<div class="small-label">GEOSPATIAL SITUATIONAL AWARENESS</div>',
        unsafe_allow_html=True
    )

    st_folium(
        m,
        height=590,
        returned_objects=[],
        **stretch(st_folium)
    )

    st.caption(
        f"NASA GIBS MODIS Terra true-colour imagery • "
        f"selected date {gibs_date}. "
        "If tiles are blank, try Esri imagery using the layer control."
    )

    if cop_obs is not None:
        st.caption(
            f"Copernicus observation date: {cop_obs['date']}. "
            "Coverage is limited to the downloaded regional footprint."
        )

with info_col:
    st.markdown(
        '<div class="panel-heading">MISSION STATUS</div>',
        unsafe_allow_html=True
    )

    selected_voyage = escape(LEGS[leg_id].name)

    st.markdown(
        f"""
        <div class="panel">
            <div class="small-label">Selected Voyage</div>
            <div class="small-value" style="font-size:17px;">
                {selected_voyage}
            </div>
        </div>
        <div class="panel">
            <div class="small-label">Iceberg Records</div>
            <div class="small-value">{iceberg_count}</div>
        </div>
        <div class="panel">
            <div class="small-label">Routing Mode</div>
            <div class="small-value" style="font-size:17px;">
                {escape(route_mode)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if route_mode == "Existing Demo Routes":
        st.markdown(
            f"""
            <div class="panel">
                <div class="small-label">Forecast Horizon</div>
                <div class="small-value">{horizon} Days</div>
            </div>
            <div class="panel">
                <div class="small-label">Ensemble Members</div>
                <div class="small-value">{members}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if best_route:
        selected_route_label = escape(str(best_route.get("label", "")))
        route_risk = best_route.get("m", {}).get("p_exceed_limit")

        risk_display = (
            f"{route_risk:.1%}"
            if route_risk is not None else "N/A"
        )

        st.markdown(
            f"""
            <div class="panel">
                <div class="small-label">Selected First Route</div>
                <div class="small-value" style="font-size:18px;">
                    {selected_route_label}
                </div>
                <div style="color:#8daac5;font-size:11px;margin-top:8px;">
                    Ice threshold probability
                </div>
                <div style="color:#4dc9f6;font-size:22px;font-weight:700;">
                    {risk_display}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif route_mode == "Live NOAA Routes":
        st.error("NO SAFE LIVE ROUTE")


# =========================================================
# 20. NAVIGATION INTELLIGENCE
# =========================================================

st.divider()
st.markdown("## 🧠 Navigation Intelligence")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧊 Ice Intelligence",
    "🌬️ Weather Intelligence",
    "🚢 Route Comparison",
    "🤖 AI Forecast",
    "🛰️ Satellite Observations"
])


# =========================================================
# TAB 1: ICE INTELLIGENCE
# =========================================================

with tab1:
    st.markdown("### Sea Ice & Iceberg Intelligence")

    ice_col1, ice_col2 = st.columns(2)

    ice_values = np.asarray(out["ice_ens"], dtype=float)
    mean_ice = float(np.nanmean(ice_values)) if np.isfinite(ice_values).any() else float("nan")

    ice_col1.metric(
        "Mean ice concentration",
        f"{mean_ice:.1%}" if np.isfinite(mean_ice) else "N/A"
    )

    ice_col2.metric("Tracked iceberg records", iceberg_count)

    if iceberg_count:
        iceberg_rows = []

        for berg in out["bergs"]:
            try:
                name, la, lo = berg
                history = nic.history_for_berg(name, berg_history_index) if berg_history_index else None
                iceberg_rows.append({
                    "Iceberg": name,
                    "Latitude": round(float(la), 4),
                    "Longitude": round(float(lo), 4),
                    "First recorded": history["first_observed"] if history else "No matching historical track",
                    "Last recorded": history["last_observed"] if history else "—",
                    "Tracked span (years)": history["tracking_span_years"] if history else "—",
                    "Years since first record": history["tracked_years"] if history else "—",
                    "Historical observations": history["observations"] if history else "—",
                })
            except (ValueError, TypeError):
                continue

        if iceberg_rows:
            st.dataframe(
                pd.DataFrame(iceberg_rows),
                hide_index=True,
                **stretch(st.dataframe)
            )
    else:
        st.info(
            "No iceberg records are currently available. "
            "This does not mean Antarctica has zero icebergs."
        )
    if berg_history_index:
        st.caption("Historical dates and tracked duration come from the BYU/NIC consolidated scatterometer + NIC track archive. Tracked duration is not the iceberg's physical age.")
    else:
        st.warning("Historical iceberg-age lookup is unavailable. Run scripts\\download_live_data.py to download the BYU/NIC tracking archive.")

    if route_mode == "Live NOAA Routes":
        st.caption(
            "The live navigation ice layer is an observed NOAA/NSIDC "
            "sea-ice field. It is not a multi-day forecast. Wind-aware routing uses the ECMWF forecast field: gales are penalized, storms strongly penalized, and hurricane-force cells are excluded."
        )
    else:
        st.caption(
            "The modeled ice concentration is produced by the "
            "existing demo ensemble pipeline."
        )


# =========================================================
# TAB 2: WEATHER INTELLIGENCE
# =========================================================

with tab2:
    st.markdown("### Wind & Gale Zone Analysis")

    if wind_data is not None:
        wind_array = np.asarray(wind_data, dtype=float)
        finite_wind = wind_array[np.isfinite(wind_array)]

        if finite_wind.size:
            wind_col1, wind_col2, wind_col3 = st.columns(3)

            wind_col1.metric(
                "Average Wind Speed",
                f"{np.mean(finite_wind):.2f} m/s"
            )
            wind_col2.metric(
                "Maximum Wind Speed",
                f"{np.max(finite_wind):.2f} m/s"
            )
            wind_col3.metric("Gale Threshold", "17.2 m/s")
            st.write({
                "Gale cells (17.2–24.5 m/s)": int(np.sum((wind_array >= 17.2) & (wind_array < 24.5))),
                "Storm cells (24.5–32.9 m/s)": int(np.sum((wind_array >= 24.5) & (wind_array < 32.9))),
                "Hurricane-force cells (>=32.9 m/s)": int(np.sum(wind_array >= 32.9)),
            })
            if out.get("metadata", {}).get("wind_valid_until"):
                st.caption(f"ECMWF forecast valid through: {out['metadata']['wind_valid_until']}")
        else:
            st.info("Wind values are unavailable for this mode.")
    else:
        st.info(
            "No wind field was returned by the selected pipeline. "
            "Gale overlay is therefore not displayed."
        )

    st.caption(
        "Wind source: ECMWF IFS 10 m forecast via Open-Meteo. Gales are penalized, "
        "storms receive a stronger cost penalty, and hurricane-force cells (>=32.9 m/s) "
        "are excluded from normal routing. Forecast is sampled and interpolated to the route grid."
    )


# =========================================================
# TAB 3: ROUTE COMPARISON
# =========================================================

with tab3:
    st.markdown("### Compare Route Alternatives")

    if routes:
        rows = []

        for route in routes:
            metrics = route.get("m", {})

            rows.append({
                "Route": route.get("label", "Alternative"),
                "Distance (NM)": round(metrics.get("distance_nm", 0)),
                "Travel Hours": round(metrics.get("hours", 0), 1),
                "Fuel (tonnes)": round(metrics.get("fuel_t", 0), 1),
                "Peak Ice": (
                    f"{metrics['max_ice']:.1%}"
                    if metrics.get("max_ice") is not None else "N/A"
                ),
                "Ice Risk": (
                    f"{metrics['p_exceed_limit']:.1%}"
                    if metrics.get("p_exceed_limit") is not None else "N/A"
                ),
                "Berg Exposure": round(metrics.get("berg_exposure", 0), 2),
                "Gale Distance (NM)": round(metrics.get("gale_nm", 0))
            })

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            **stretch(st.dataframe)
        )
    else:
        if route_mode == "Live NOAA Routes":
            st.error(
                "NO SAFE LIVE ROUTE. The live observation did not "
                "produce a route that passed the configured safety checks."
            )
        else:
            st.warning("No feasible demo routes were returned by the pipeline.")


# =========================================================
# TAB 4: AI FORECAST
# =========================================================

with tab4:
    st.markdown("### AI Forecast Performance")

    skill = out.get("forecast_skill")

    if route_mode == "Live NOAA Routes":
        st.info(
            "Live NOAA mode uses a sea-ice observation, not an AI forecast. "
            "Forecast skill metrics are not applicable to this observation."
        )
    elif skill:
        f1, f2, f3 = st.columns(3)

        f1.metric(
            "Model MAE",
            f"{skill.get('model_mae', float('nan')):.4f}"
        )
        f2.metric(
            "Persistence MAE",
            f"{skill.get('persistence_mae', float('nan')):.4f}"
        )
        f3.metric(
            "Skill vs Persistence",
            f"{skill.get('skill_vs_persistence', float('nan')):.1%}"
        )

        st.info(
            "Forecast skill should be interpreted only after confirming "
            "that evaluation data is independent of training data."
        )
    else:
        st.info("Forecast skill metrics were not returned by the pipeline.")

    st.markdown("#### Configuration")

    st.write(f"Forecast horizon: **{horizon} days**")
    st.write(f"Ensemble members: **{members}**")
    st.write(f"Selected ice blocking probability: **{p_block:.2f}**")


# =========================================================
# TAB 5: COPERNICUS SATELLITE OBSERVATIONS
# =========================================================

with tab5:
    st.markdown("### Copernicus Marine Satellite Observations")

    if cop_obs is not None:
        observed_ice = np.asarray(cop_obs["ice"])
        valid_mask = np.asarray(cop_obs["valid_mask"], dtype=bool)

        valid_count = int(valid_mask.sum())

        if valid_count:
            observed_mean = float(np.nanmean(observed_ice[valid_mask]) * 100)
            observed_max = float(np.nanmax(observed_ice[valid_mask]) * 100)
        else:
            observed_mean = 0.0
            observed_max = 0.0

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Observation Date", cop_obs["date"])
        c2.metric("Valid Pixels", f"{valid_count:,}")
        c3.metric("Mean Ice Concentration", f"{observed_mean:.2f}%")
        c4.metric("Maximum Concentration", f"{observed_max:.2f}%")

        st.success("Copernicus observation successfully loaded.")

        st.write(
            "**Source:** Copernicus Marine — Antarctic Ocean "
            "High Resolution Sea Ice Information"
        )
        st.write("**Satellite instruments:** Sentinel-1 SAR and AMSR2")
        st.write("**Variables:** Ice concentration and confidence")

        st.write(
            "**Observation coverage:** "
            f"Latitude {cop_obs['latitude'][0]:.4f}° to "
            f"{cop_obs['latitude'][-1]:.4f}°; "
            f"longitude {cop_obs['longitude'][0]:.4f}° to "
            f"{cop_obs['longitude'][-1]:.4f}°."
        )

        st.caption(
            "This is a historical observation from the downloaded "
            "NetCDF file, not a current live observation or forecast."
        )

        st.warning(
            "This regional dataset covers only a limited area near "
            "Bharati. It does not provide complete sea-ice coverage "
            "for the Cape Town–Bharati–Maitri voyage."
        )
    else:
        st.warning(
            "Copernicus observations are unavailable. "
            "Check the downloaded NetCDF file and the sidebar."
        )

        if cop_error:
            st.code(cop_error)


# =========================================================
# 21. OPERATIONAL ALERTS
# =========================================================

st.divider()
st.markdown("## 🚨 Operational Alerts")

if routes:
    try:
        for level, message in alerts(routes[0], out["bergs"]):
            if level == "DANGER":
                st.error(f"🔴 {message}")
            else:
                st.warning(f"🟠 {message}")
    except Exception as e:
        st.warning(f"Alert assessment unavailable: {e}")
else:
    st.warning("No route is available for alert assessment.")


# =========================================================
# 22. FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:15px;
    color:#68849e;
    font-size:11px;
    line-height:2;
">
    <b style="color:#4dc9f6;">ODYSEA</b>
    | ANTARCTIC MARITIME DECISION SUPPORT SYSTEM
    <br>
    AI • Geospatial Intelligence • Probabilistic Route Planning
    <br>
</div>
""", unsafe_allow_html=True)
