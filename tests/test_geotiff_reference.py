"""GeoTIFF export checks with a rotated projected raster."""
import tempfile
from pathlib import Path

import numpy as np
import unittest
try:
    import rasterio
    import pyproj
    from affine import Affine
except ImportError:
    raise unittest.SkipTest("rasterio, pyproj or affine not installed")

from src.mission.geotiff import GeoTiffReference
from src.mission.mission_output import generate_mission_file


def test_rotated_geotiff_mission_matches_affine_crs():
    with tempfile.TemporaryDirectory() as tmp:
        tif = Path(tmp) / "rotated.tif"
        transform = Affine(12, 2, 500000, -1, -9, 1000000)
        with rasterio.open(tif, "w", driver="GTiff", width=40, height=30,
                           count=1, dtype="uint8", crs="EPSG:32630", transform=transform) as ds:
            ds.write(np.zeros((30, 40), np.uint8), 1)
        ref = GeoTiffReference.from_file(tif)
        to_wgs = pyproj.Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True)
        for x, y in ((0, 0), (10.25, 7.5), (39, 29)):
            east, north = transform * (x + 0.5, y + 0.5)
            lon, lat = to_wgs.transform(east, north)
            assert ref.pixel_to_latlon(x, y) == pytest.approx((lat, lon), abs=1e-10)
            assert ref.display_to_latlon((x + 0.5) * 0.5 - 0.5,
                                          (y + 0.5) * 0.5 - 0.5, 0.5) == pytest.approx((lat, lon), abs=1e-10)
        wpl = Path(tmp) / "route.waypoints"
        route = [(0, 0), (10, 7), (0, 0)]
        generate_mission_file(route, [1], route[0], (19.5, 14.5), ref.centre,
                              ref.local_mpp, filename=wpl, coordinate_fn=ref.pixel_to_latlon)
        rows = [line.split() for line in wpl.read_text().splitlines()[1:]]
        point_rows = [r for r in rows if r[3] == "16"]
        # The drop has both descent and post-servo ascent navigation rows.
        expected = [ref.pixel_to_latlon(x, y) for x, y in
                    (route[0], route[1], route[1], route[1], route[2])]
        for row, (lat, lon) in zip(point_rows, expected):
            assert (float(row[8]), float(row[9])) == pytest.approx((lat, lon), abs=1e-10)
