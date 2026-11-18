import numpy as np

R_NM = 3440.065


def haversine_nm(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2)
        * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * R_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class Grid:
    """Regular lat/lon grid, ascending latitude and longitude."""

    def __init__(self, lat, lon):
        self.lat = np.asarray(lat, dtype=float)
        self.lon = np.asarray(lon, dtype=float)
        self.LAT, self.LON = np.meshgrid(
            self.lat, self.lon, indexing="ij"
        )
        self.shape = self.LAT.shape

    def nearest_valid(self, lat, lon, blocked):
        blocked = np.asarray(blocked, dtype=bool)

        if blocked.ndim == 3:
            if blocked.shape[0] != 1:
                raise ValueError(
                    f"Expected a 2D blocked mask or singleton time "
                    f"dimension; got {blocked.shape}"
                )
            blocked = blocked[0]

        if blocked.shape != self.shape:
            raise ValueError(
                f"Blocked mask shape {blocked.shape} does not match "
                f"grid shape {self.shape}"
            )

        distances = haversine_nm(
            lat, lon, self.LAT, self.LON
        )
        distances = np.where(blocked, np.inf, distances)

        if not np.isfinite(distances).any():
            raise ValueError("No valid unblocked grid cells available.")

        i, j = np.unravel_index(
            np.argmin(distances), self.shape
        )
        return int(i), int(j)

    def sample(self, field, lat, lon):
        field = np.asarray(field)
        if field.ndim != 2:
            raise ValueError(
                f"sample() expects a 2D field; got {field.shape}"
            )
        i = int(np.abs(self.lat - lat).argmin())
        j = int(np.abs(self.lon - lon).argmin())
        return field[i, j]
