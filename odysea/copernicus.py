"""Load historical Copernicus Antarctic sea-ice observations."""

from pathlib import Path

import numpy as np


DEFAULT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "copernicus"
    / "bharati_sea_ice_feb_mar_2024.nc"
)


def load_copernicus_observation(path=DEFAULT_FILE):
    """Load the latest observation in a Copernicus NetCDF file.

    Returns sea-ice concentration as a fraction (0-1).
    Missing observations remain NaN.
    """

    import xarray as xr  # lazy: demo mode must not require the NetCDF stack

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Copernicus observation file not found: {path}"
        )

    with xr.open_dataset(path) as ds:

        required = [
            "latitude",
            "longitude",
            "ice_concentration",
            "confidence",
            "time",
        ]

        missing = [name for name in required if name not in ds]

        if missing:
            raise ValueError(
                f"Missing Copernicus variables or coordinates: {missing}"
            )

        # Select the latest available observation.
        ice = np.asarray(
            ds["ice_concentration"].isel(time=-1).values,
            dtype=float
        )

        confidence = np.asarray(
            ds["confidence"].isel(time=-1).values,
            dtype=float
        )

        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)

        observation_date = str(
            ds["time"].values[-1]
        )[:10]

    # Grid coordinates must be ascending.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        ice = ice[::-1, :]
        confidence = confidence[::-1, :]

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        ice = ice[:, ::-1]
        confidence = confidence[:, ::-1]

    if ice.shape != (len(lat), len(lon)):
        raise ValueError("Sea-ice data shape does not match coordinates.")

    if confidence.shape != ice.shape:
        raise ValueError("Confidence data shape does not match sea-ice data.")

    # Convert percentages (0-100) into fractions (0-1).
    ice_fraction = ice / 100.0

    # Missing pixels remain NaN. Never interpret them as open water.
    valid = (
        np.isfinite(ice_fraction)
        & np.isfinite(confidence)
    )

    ice_fraction[~valid] = np.nan

    confidence_fraction = np.clip(
        confidence / 100.0,
        0.0,
        1.0
    )

    confidence_fraction[~valid] = 0.0

    return {
        "latitude": lat,
        "longitude": lon,
        "ice": ice_fraction,
        "confidence": confidence_fraction,
        "valid_mask": valid,
        "date": observation_date,
        "source": "Copernicus Marine - Sentinel-1 / AMSR2",
        "path": str(path),
    }
