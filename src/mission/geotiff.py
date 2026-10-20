"""Optional GeoTIFF pixel-to-WGS84 reference for mission export.

The reference raster must have the same original pixel dimensions as the
original input image. Routing still uses a single local metres-per-pixel estimate;
mission coordinates use the complete affine transform and CRS.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def _to_wgs84(crs):
    from pyproj import Transformer
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


@dataclass(frozen=True)
class GeoTiffReference:
    path: str
    width: int
    height: int
    crs: str
    transform: object
    centre: tuple[float, float]
    local_mpp: float

    @classmethod
    def from_file(cls, path):
        try:
            import rasterio
            from pyproj import Geod
        except ImportError as exc:
            raise ValueError("GeoTIFF support requires rasterio and pyproj") from exc
        path = str(Path(path).expanduser().resolve())
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError("GeoTIFF has no CRS")
            width, height, transform, crs = src.width, src.height, src.transform, src.crs
        if width < 2 or height < 2 or abs(transform.a * transform.e - transform.b * transform.d) < 1e-15:
            raise ValueError("GeoTIFF dimensions or affine transform are invalid")
        to_wgs = _to_wgs84(str(crs))
        def at(x, y):
            east, north = transform * (x + 0.5, y + 0.5)
            lon, lat = to_wgs.transform(east, north)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("GeoTIFF maps outside valid WGS84 coordinates")
            return lat, lon
        cx, cy = (width - 1) / 2, (height - 1) / 2
        centre = at(cx, cy)
        east, south = at(cx + 1, cy), at(cx, cy + 1)
        geod = Geod(ellps="WGS84")
        ew = geod.inv(centre[1], centre[0], east[1], east[0])[2]
        ns = geod.inv(centre[1], centre[0], south[1], south[0])[2]
        local_mpp = (abs(ew) + abs(ns)) / 2
        if not 0 < local_mpp < 100000:
            raise ValueError("GeoTIFF has an implausible local ground scale")
        return cls(path, width, height, str(crs), transform, centre, local_mpp)

    def pixel_to_latlon(self, x, y):
        east, north = self.transform * (float(x) + 0.5, float(y) + 0.5)
        lon, lat = _to_wgs84(self.crs).transform(east, north)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Pixel maps outside valid WGS84 coordinates")
        return lat, lon

    def display_to_latlon(self, x, y, scale_factor):
        if scale_factor <= 0:
            raise ValueError("Display scale must be positive")
        return self.pixel_to_latlon((float(x) + 0.5) / scale_factor - 0.5,
                                    (float(y) + 0.5) / scale_factor - 0.5)
