"""End-to-end checks for the faculty-requested map guard and GeoTIFF input."""
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import main as main_module
from src.vision.image_processing import ImageProcessor
from tests.test_multi_uav import make_map, run_main


OUTPUT = Path("data/output/enriched_drone_mission.waypoints")


def test_dense_flood_mask_stops_before_planning_or_export(tmp_path):
    image = tmp_path / "dense.png"
    make_map(str(image), [(0, 0, 199, 199)], size=200)
    with mock.patch.object(main_module, "predict_spread", side_effect=AssertionError("planning began")):
        status = run_main([str(image), "--max-mask-fraction", "0.8"], str(tmp_path))
    assert status == 2
    assert not (tmp_path / OUTPUT).exists()


def test_sparse_flood_mask_passes_review_gate(tmp_path):
    image = tmp_path / "sparse.png"
    make_map(str(image), [(40, 40, 100, 100)], size=200)
    status = run_main([str(image), "--max-mask-fraction", "0.8"], str(tmp_path))
    assert status == 0
    mission = tmp_path / OUTPUT
    assert mission.read_text().startswith("QGC WPL 110\n")


def test_geotiff_dimension_mismatch_rejected_before_operator(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    image = tmp_path / "input.png"
    make_map(str(image), [(40, 40, 100, 100)], size=200)
    tif = tmp_path / "wrong_size.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=100, height=100,
                       count=1, dtype="uint8", crs="EPSG:4326",
                       transform=Affine(.0001, 0, -.8, 0, -.0001, 9.7)) as ds:
        ds.write(np.zeros((100, 100), np.uint8), 1)
    no_click = mock.patch.object(ImageProcessor, "select_sample_points",
                                 side_effect=AssertionError("operator prompted"))
    status = run_main([str(image), "--geotiff-reference", str(tif)], str(tmp_path),
                      extra=(no_click,))
    assert status == 2
    assert not (tmp_path / OUTPUT).exists()


def test_geotiff_cli_uses_affine_after_display_resize(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    from src.mission.geotiff import GeoTiffReference

    image = tmp_path / "input.png"
    make_map(str(image), [(40, 40, 100, 100)], size=200)
    tif = tmp_path / "reference.tif"
    transform = Affine(.0001, .00002, -.8, -.00001, -.0001, 9.7)
    with rasterio.open(tif, "w", driver="GTiff", width=200, height=200,
                       count=1, dtype="uint8", crs="EPSG:4326", transform=transform) as ds:
        ds.write(np.zeros((200, 200), np.uint8), 1)
    original_generate = main_module.generate_mission_file
    with mock.patch.object(main_module, "generate_mission_file", wraps=original_generate) as generate:
        status = run_main([str(image), "--display-width", "100",
                           "--geotiff-reference", str(tif)], str(tmp_path))
    assert status == 0
    home = generate.call_args.args[2]
    expected = GeoTiffReference.from_file(tif).display_to_latlon(*home, scale_factor=.5)
    first = (tmp_path / OUTPUT).read_text().splitlines()[1].split("\t")
    assert (float(first[8]), float(first[9])) == pytest.approx(expected, abs=1e-8)
