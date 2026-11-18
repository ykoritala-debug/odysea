"""US National Ice Center (USNIC) iceberg CSV loader.

Two sources:
  * fetch_current()  - download the latest CSV (URL below is UNVERIFIED; override with env USNIC_CSV_URL)
  * load_csv(path)   - read a weekly archive file, e.g. AntarcticIcebergs_20240104.csv

Only large named icebergs are tracked and positions are weekly: an empty result is NOT proof of
clear water.
"""
import io
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests

USNIC_CSV_URL = os.environ.get("USNIC_CSV_URL", "https://usicecenter.gov/File/DownloadCurrent?pId=134")

_DMS = re.compile(r"(\d{1,3})(?:\s*[°º ]\s*(\d{1,2}(?:\.\d+)?))?\s*['′’]?\s*([NSEW])", re.I)


def parse_one(value, axis=None):
    """Parse ONE coordinate: -69.4, '69.4 S', "69° 24.31' S", "69 24.31 S". Returns float or None.

    `axis` is 'lat' or 'lon' and is only used to range-check.
    """
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        text = str(value).strip().replace("\u2212", "-")
        try:
            out = float(text)
        except ValueError:
            m = _DMS.search(text.upper())
            if not m:
                return None
            deg, minutes, hemi = m.groups()
            out = float(deg) + float(minutes or 0) / 60.0
            if hemi.upper() in "SW":
                out = -out
    limit = 90 if axis == "lat" else 180
    return out if abs(out) <= limit else None


def parse_coordinate(value):
    """Parse a combined 'lat, lon' string such as "69° 24.31' S 76° 11.84' E" -> (lat, lon)."""
    if not isinstance(value, str):
        return None, None
    parts = _DMS.findall(value.upper())
    if len(parts) < 2:
        return None, None
    out = []
    for deg, minutes, hemi in parts[:2]:
        v = float(deg) + float(minutes or 0) / 60.0
        out.append(-v if hemi in "SW" else v)
    lat, lon = out
    if abs(lat) > 90 or abs(lon) > 180:
        return None, None
    return lat, lon


def _frame_from_text(text):
    lines = text.lstrip("\ufeff").splitlines()
    header = 0
    for i, line in enumerate(lines[:20]):
        low = line.lower()
        if ("iceberg" in low or "name" in low) and ("location" in low or "latitude" in low or "lat" in low):
            header = i
            break
    return pd.read_csv(io.StringIO("\n".join(lines[header:])))


def fetch_current():
    """Retrieve the latest USNIC iceberg CSV."""
    response = requests.get(USNIC_CSV_URL, timeout=30, headers={"User-Agent": "ODySEA/1.0"})
    response.raise_for_status()
    df = _frame_from_text(response.text)
    df.attrs["source"] = "USNIC"
    df.attrs["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    return df


def load_csv(path):
    """Load a saved USNIC weekly archive CSV."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        df = _frame_from_text(fh.read())
    df.attrs["source"] = f"USNIC archive file {os.path.basename(str(path))}"
    return df


def _pick(cols, exact, contains=()):
    """Choose a column by EXACT name first, then by whole-word match. No bare substring tests
    (the old matcher treated 'id' as a match for 'width')."""
    for name in exact:
        if name in cols:
            return cols[name]
    for key, original in cols.items():
        words = set(re.split(r"[^a-z0-9]+", key))
        if any(c in words for c in contains):
            return original
    return None


def to_bergs(df, return_skipped=False):
    """Convert USNIC records into [(name, latitude, longitude), ...]."""
    cols = {str(c).lower().strip(): c for c in df.columns}
    name_col = _pick(cols, ["iceberg", "name", "iceberg name", "berg"], ["iceberg", "name", "berg"])
    location_col = _pick(cols, ["location", "position"], ["location", "position"])
    lat_col = _pick(cols, ["latitude", "lat"], ["latitude", "lat"])
    lon_col = _pick(cols, ["longitude", "lon", "long"], ["longitude", "lon", "long"])

    out, skipped = [], 0
    for _, row in df.iterrows():
        lat = lon = None
        if lat_col is not None and lon_col is not None:
            lat = parse_one(row[lat_col], "lat")
            lon = parse_one(row[lon_col], "lon")
        if (lat is None or lon is None) and location_col is not None:
            lat, lon = parse_coordinate(str(row[location_col]))
        if lat is None or lon is None:
            skipped += 1
            continue
        name = str(row[name_col]) if name_col is not None and pd.notna(row[name_col]) else "Unknown iceberg"
        out.append((name, float(lat), float(lon)))
    return (out, skipped) if return_skipped else out


def load_byu_history_index(zip_path):
    """Read BYU consolidated iceberg CSV tracks; returns first/last observed dates.

    BYU track dates are encoded as YYYYDDD (year + day-of-year). These are
    tracking dates, not iceberg formation dates.
    """
    import zipfile
    from io import BytesIO
    from datetime import date, timedelta
    index = {}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.lower().endswith('.csv'):
                continue
            stem = os.path.splitext(os.path.basename(member))[0].upper().replace(' ', '')
            try:
                frame = pd.read_csv(BytesIO(zf.read(member)))
                date_col = next((c for c in frame.columns if str(c).strip().lower() == 'date'), None)
                if date_col is None:
                    continue
                raw = pd.to_numeric(frame[date_col], errors='coerce').dropna().astype(int)
                dates=[]
                for value in raw:
                    year, doy = divmod(int(value), 1000)
                    if 1 <= doy <= 366:
                        try:
                            dates.append(date(year, 1, 1) + timedelta(days=doy - 1))
                        except ValueError:
                            pass
                if dates:
                    index[stem] = {"first_observed": min(dates), "last_observed": max(dates),
                                   "observations": len(dates), "source": "BYU/NIC consolidated tracking database"}
            except Exception:
                continue
    return index


def history_for_berg(name, history_index):
    key = str(name).strip().upper().replace(' ', '').replace('-', '')
    item = history_index.get(key)
    if not item:
        return None
    from datetime import date
    first, last = item["first_observed"], item["last_observed"]
    today = date.today()
    years = (today - first).days / 365.2425
    result = dict(item)
    result.update({"first_observed": first.isoformat(), "last_observed": last.isoformat(),
                   "tracked_years": round(years, 1),
                   "tracking_span_years": round((last-first).days/365.2425, 1)})
    return result
