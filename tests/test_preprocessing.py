# Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
# Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""
tests/test_preprocessing.py
=============================
Unit tests for the canonical grid and preprocessing utilities.
No external data files required.
"""

from __future__ import annotations

from typing import Any
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.preprocessing import (
    CanonicalGrid,
    compute_topographic_roughness,
    compute_terrain_diversity,
    normalise_to_0_1,
)
from configs.pipeline_config import TITAN_RADIUS_M


# ---------------------------------------------------------------------------
# Test: CanonicalGrid
# ---------------------------------------------------------------------------

class TestCanonicalGrid:
    def test_default_resolution(self) -> None:
        g = CanonicalGrid(4490.0)
        assert g.pixel_size_m == 4490.0

    def test_ncols_positive_integer(self) -> None:
        for res in (4490, 1000, 350):
            g = CanonicalGrid(float(res))
            assert g.ncols > 0
            assert g.nrows > 0

    def test_aspect_ratio_approx_2_to_1(self) -> None:
        """
        SimpleCylindrical global grid: ncols =~ 2 x nrows.
        (360 deg wide, 180 deg tall.)
        """
        g = CanonicalGrid(4490.0)
        ratio = g.ncols / g.nrows
        assert 1.9 < ratio < 2.1, f"Expected ~2:1 aspect ratio, got {ratio:.3f}"

    def test_pixel_size_consistent(self) -> None:
        """dx x ncols should equal 360 deg in metres."""
        g = CanonicalGrid(4490.0)
        circumference_m = 2 * math.pi * TITAN_RADIUS_M
        m_per_deg = circumference_m / 360.0
        expected_width_m = 360.0 * m_per_deg
        actual_width_m   = g.ncols * g.dx_m
        assert abs(actual_width_m - expected_width_m) / expected_width_m < 0.01

    def test_lat_centres_shape(self) -> None:
        g = CanonicalGrid(4490.0)
        lats = g.lat_centres_deg()
        assert len(lats) == g.nrows

    def test_lon_centres_shape(self) -> None:
        g = CanonicalGrid(4490.0)
        lons = g.lon_centres_deg()
        assert len(lons) == g.ncols

    def test_lat_centres_range(self) -> None:
        g = CanonicalGrid(4490.0)
        lats = g.lat_centres_deg()
        assert lats[0] > 0, "First row should be north (positive lat)"
        assert lats[-1] < 0, "Last row should be south (negative lat)"
        assert abs(lats[0]) < 90, "Lat centres should be inside +/-90 deg"
        assert abs(lats[-1]) < 90

    def test_empty_returns_nan_array(self) -> None:
        g = CanonicalGrid(4490.0)
        arr = g.empty()
        assert arr.shape == (g.nrows, g.ncols)
        assert np.all(np.isnan(arr))

    def test_transform_is_affine(self) -> None:
        """The transform should be a rasterio Affine object."""
        pytest.importorskip("rasterio")
        from rasterio.transform import Affine
        g = CanonicalGrid(4490.0)
        assert isinstance(g.transform, Affine)

    def test_repr_contains_resolution(self) -> None:
        g = CanonicalGrid(4490.0)
        assert "4490" in repr(g)


# ---------------------------------------------------------------------------
# Test: normalise_to_0_1
# ---------------------------------------------------------------------------

class TestNormaliseTo01:
    def test_uniform_input(self) -> None:
        """All-same values -> all zeros (no dynamic range)."""
        arr = np.full((5, 5), 42.0, dtype=np.float32)
        out = normalise_to_0_1(arr)
        assert np.all(out == 0.0)

    def test_range_is_0_to_1(self) -> None:
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        out = normalise_to_0_1(arr)
        finite = out[np.isfinite(out)]
        assert finite.min() >= 0.0
        assert finite.max() <= 1.0

    def test_nan_preserved(self) -> None:
        arr = np.array([0.0, 1.0, np.nan, 2.0, 3.0], dtype=np.float32)
        out = normalise_to_0_1(arr)
        assert np.isnan(out[2])

    def test_all_nan_returns_nan(self) -> None:
        arr = np.full((3, 3), np.nan, dtype=np.float32)
        out = normalise_to_0_1(arr)
        assert np.all(np.isnan(out))

    def test_dtype_float32(self) -> None:
        arr = np.arange(10, dtype=np.float64)
        out = normalise_to_0_1(arr)
        assert out.dtype == np.float32

    def test_clipping_applied(self) -> None:
        arr = np.array([-10.0, 50.0, 150.0], dtype=np.float32)
        out = normalise_to_0_1(arr, clip=True)
        assert np.all(out[np.isfinite(out)] <= 1.0)
        assert np.all(out[np.isfinite(out)] >= 0.0)


# ---------------------------------------------------------------------------
# Test: compute_topographic_roughness
# ---------------------------------------------------------------------------

class TestTopographicRoughness:
    def test_flat_dem_low_roughness(self) -> None:
        """Flat DEM -> near-zero roughness everywhere."""
        dem = np.zeros((20, 20), dtype=np.float32)
        rough = compute_topographic_roughness(dem, window_radius=2)
        assert rough.max() < 0.1, f"Flat DEM should have low roughness, max={rough.max()}"

    def test_rough_dem_higher_values(self) -> None:
        """Random DEM -> higher roughness than flat DEM."""
        rng = np.random.default_rng(0)
        dem_flat  = np.zeros((20, 20), dtype=np.float32)
        dem_rough = rng.normal(0, 100, (20, 20)).astype(np.float32)
        r_flat  = compute_topographic_roughness(dem_flat)
        r_rough = compute_topographic_roughness(dem_rough)
        assert r_rough.mean() > r_flat.mean()

    def test_output_range(self) -> None:
        rng = np.random.default_rng(1)
        dem = rng.uniform(-500, 500, (30, 30)).astype(np.float32)
        out = compute_topographic_roughness(dem)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_nan_in_dem(self) -> None:
        """NaN pixels in DEM should not crash the computation."""
        dem = np.zeros((10, 10), dtype=np.float32)
        dem[3:5, 3:5] = np.nan
        out = compute_topographic_roughness(dem, window_radius=1)
        assert out.shape == dem.shape


# ---------------------------------------------------------------------------
# Test: compute_terrain_diversity
# ---------------------------------------------------------------------------

class TestTerrainDiversity:
    def test_uniform_class_zero_diversity(self) -> None:
        """All pixels same class -> zero Shannon diversity."""
        class_map = np.ones((20, 20), dtype=np.int32)
        out = compute_terrain_diversity(class_map, n_classes=7, window_radius=2)
        # With all same class, H = 0 everywhere -> normalised to 0
        assert out.max() < 0.1

    def test_equal_classes_high_diversity(self) -> None:
        """
        Checkerboard of two classes -> higher diversity than uniform.
        """
        class_map = np.zeros((20, 20), dtype=np.int32)
        class_map[::2, ::2]  = 1
        class_map[1::2, 1::2] = 2
        class_map[1::2, ::2]  = 3
        class_map[::2, 1::2]  = 4
        out = compute_terrain_diversity(class_map, n_classes=7, window_radius=3)
        assert out.mean() > 0.2, "Diverse terrain should yield higher diversity scores."



# ---------------------------------------------------------------------------
# Test: _synthesise_cirs_temperature
# ---------------------------------------------------------------------------

class TestSynthesiseCirsTemperature:
    """
    Tests for the Jennings-formula-based CIRS temperature synthesis.
    No external files required -- the formula is fully embedded.
    """

    @pytest.fixture
    def tmp_pipeline(self, tmp_path: Path) -> Any:
        """Minimal pipeline config and grid for temp synthesis tests."""
        from configs.pipeline_config import PipelineConfig
        processed = tmp_path / "processed"
        processed.mkdir()
        config = PipelineConfig(data_dir=tmp_path, processed_dir=processed)
        grid   = CanonicalGrid(pixel_size_m=500_000)  # coarse -- fast
        return config, grid

    def test_synthesise_produces_geotiff(self, tmp_pipeline: Any) -> None:
        """Output file should be created."""
        rasterio = pytest.importorskip("rasterio")
        from titan.preprocessing import DataPreprocessor
        config, grid = tmp_pipeline
        dp = DataPreprocessor(config, grid)
        result = dp._synthesise_cirs_temperature(overwrite=True)
        assert "cirs_temperature" in result
        assert result["cirs_temperature"].exists()

    def test_synthesised_temperature_physically_plausible(self, tmp_pipeline: Any) -> None:
        """All synthesised temperatures should be in 87-96 K."""
        rasterio = pytest.importorskip("rasterio")
        from titan.preprocessing import DataPreprocessor
        config, grid = tmp_pipeline
        dp = DataPreprocessor(config, grid)
        result = dp._synthesise_cirs_temperature(overwrite=True)
        with rasterio.open(result["cirs_temperature"]) as ds:
            arr = ds.read(1)
        finite = arr[np.isfinite(arr)]
        assert len(finite) > 0
        assert float(finite.min()) > 87.0, f"min={float(finite.min()):.2f}"
        assert float(finite.max()) < 96.0, f"max={float(finite.max()):.2f}"

    def test_synthesised_temperature_equatorial_near_93K(self, tmp_pipeline: Any) -> None:
        """Near-equatorial pixels should be close to 93.5 K at epoch 2011."""
        rasterio = pytest.importorskip("rasterio")
        from titan.preprocessing import DataPreprocessor
        config, grid = tmp_pipeline
        dp = DataPreprocessor(config, grid)
        result = dp._synthesise_cirs_temperature(overwrite=True)
        with rasterio.open(result["cirs_temperature"]) as ds:
            arr = ds.read(1)
        # Find equatorial rows (lat =~ 0)
        lats = grid.lat_centres_deg()
        eq_row = int(np.argmin(np.abs(lats)))
        T_eq = float(arr[eq_row, arr.shape[1] // 2])
        assert 91.0 < T_eq < 95.0, (
            f"Equatorial T at epoch 2011 expected ~93K, got {T_eq:.2f}"
        )

    def test_synthesised_poles_cooler_than_equator(self, tmp_pipeline: Any) -> None:
        """Polar pixels should be cooler than equatorial pixels."""
        rasterio = pytest.importorskip("rasterio")
        from titan.preprocessing import DataPreprocessor
        config, grid = tmp_pipeline
        dp = DataPreprocessor(config, grid)
        result = dp._synthesise_cirs_temperature(overwrite=True)
        with rasterio.open(result["cirs_temperature"]) as ds:
            arr = ds.read(1).astype(float)
        T_np_mean = float(arr[:3,  :].mean())   # top rows = north pole
        T_sp_mean = float(arr[-3:, :].mean())   # bottom rows = south pole
        T_eq_mean = float(arr[arr.shape[0]//2-1:arr.shape[0]//2+2, :].mean())
        assert T_eq_mean > T_np_mean + 0.5, (
            f"Equator ({T_eq_mean:.2f}) should be warmer than NP ({T_np_mean:.2f})"
        )
        assert T_eq_mean > T_sp_mean + 0.5, (
            f"Equator ({T_eq_mean:.2f}) should be warmer than SP ({T_sp_mean:.2f})"
        )

    def test_synthesise_respects_overwrite_false(self, tmp_pipeline: Any) -> None:
        """Second call with overwrite=False should not regenerate the file."""
        rasterio = pytest.importorskip("rasterio")
        from titan.preprocessing import DataPreprocessor
        config, grid = tmp_pipeline
        dp = DataPreprocessor(config, grid)
        result1 = dp._synthesise_cirs_temperature(overwrite=True)
        mtime1  = result1["cirs_temperature"].stat().st_mtime

        result2 = dp._synthesise_cirs_temperature(overwrite=False)
        mtime2  = result2["cirs_temperature"].stat().st_mtime
        assert mtime1 == mtime2, "overwrite=False should not touch existing file"

    def test_cirs_appears_in_stack_all_names(self) -> None:
        """cirs_temperature must be in the default CanonicalDataStack name list."""
        import inspect
        from titan.preprocessing import CanonicalDataStack
        src = inspect.getsource(CanonicalDataStack.load)
        assert "cirs_temperature" in src


# ---------------------------------------------------------------------------
# Test: _preprocess_channels (dependency-free parts)
# ---------------------------------------------------------------------------

class TestPreprocessChannels:
    """Dependency-free tests for the channel preprocessing method."""

    def test_channel_density_in_stack_default_names(self) -> None:
        """channel_density must be in the default layer list."""
        import inspect
        from titan.preprocessing import CanonicalDataStack
        src = inspect.getsource(CanonicalDataStack.load)
        assert "channel_density" in src

    def test_preprocess_channels_returns_empty_when_shp_missing(self, tmp_path: Path) -> None:
        """Missing global_channels.shp -> graceful empty return, no exception."""
        from configs.pipeline_config import PipelineConfig
        from titan.preprocessing import DataPreprocessor
        config = PipelineConfig(
            data_dir=tmp_path,
            processed_dir=tmp_path / "processed",
            shapefile_dir=tmp_path / "nonexistent_shapefiles",
        )
        (tmp_path / "processed").mkdir()
        dp = DataPreprocessor(config, CanonicalGrid(500_000))
        result = dp._preprocess_channels(overwrite=True)
        assert result == {}

    def test_rasterise_channels_function_exists(self) -> None:
        """_rasterise_channels should be importable from titan.preprocessing."""
        from titan.preprocessing import _rasterise_channels
        assert callable(_rasterise_channels)

    def test_preprocess_channels_function_exists(self) -> None:
        """DataPreprocessor should have _preprocess_channels method."""
        from titan.preprocessing import DataPreprocessor
        assert hasattr(DataPreprocessor, "_preprocess_channels")

    def test_titan_channels_spec_in_catalogue(self) -> None:
        from configs.pipeline_config import default_dataset_catalogue
        cat = default_dataset_catalogue()
        assert "titan_channels" in cat
        spec = cat["titan_channels"]
        assert spec.local_filename.endswith("global_channels.shp")
        assert "Miller" in spec.citation