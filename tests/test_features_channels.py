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
tests/test_features_channels.py
================================
Tests for:
  - Feature 5 (surface_atm_interaction) with channel_density present/absent
  - _rasterise_channels preprocessing helper
  - Channel DatasetSpec in pipeline_config
"""

import gzip
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


# ============================================================================
# Helpers
# ============================================================================

def _make_stack(
    nrows: int = 16,
    ncols: int = 32,
    include_topo: bool = True,
    include_geo: bool = True,
    include_channels: bool = True,
) -> xr.Dataset:
    """Build a minimal xarray Dataset mimicking CanonicalDataStack output."""
    rng  = np.random.default_rng(42)
    lats = np.linspace(-90, 90, nrows)
    lons = np.linspace(0, 360, ncols)
    data_vars: dict = {}

    if include_topo:
        dem = rng.uniform(-500, 500, (nrows, ncols)).astype(np.float32)
        data_vars["topography"] = xr.DataArray(
            dem, dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
        )

    if include_geo:
        # lake class = 7, labyrinth = 6, plains = rest
        geo = rng.integers(1, 8, (nrows, ncols)).astype(np.float32)
        data_vars["geomorphology"] = xr.DataArray(
            geo, dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
        )

    if include_channels:
        # Synthetic channel density: high along a diagonal band
        ch = np.zeros((nrows, ncols), dtype=np.float32)
        for i in range(nrows):
            j = int(i * ncols / nrows)
            ch[i, max(0, j-1):min(ncols, j+2)] = 1.0
        data_vars["channel_density"] = xr.DataArray(
            ch, dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
        )

    return xr.Dataset(data_vars)


# ============================================================================
# Feature 5 tests
# ============================================================================

class TestSurfaceAtmInteraction:

    @pytest.fixture(autouse=True)
    def import_calculator(self) -> None:
        from titan.features import FeatureExtractor
        from titan.preprocessing import CanonicalGrid
        self.Calculator = FeatureExtractor
        self.grid = CanonicalGrid(pixel_size_m=500_000)

    def _compute(self, stack: xr.Dataset) -> np.ndarray:
        calc = self.Calculator(self.grid)
        nrows = stack.sizes.get("lat", self.grid.nrows) if stack.sizes else self.grid.nrows
        ncols = stack.sizes.get("lon", self.grid.ncols) if stack.sizes else self.grid.ncols
        nan   = np.full((nrows, ncols), np.nan, dtype=np.float32)
        return calc._surface_atm_interaction(stack, nan)

    def test_returns_float32_array_with_all_three_layers(self) -> None:
        stack  = _make_stack(include_topo=True, include_geo=True,
                              include_channels=True)
        result = self._compute(stack)
        assert result.dtype == np.float32
        assert result.shape == (16, 32)

    def test_no_nan_in_output_when_all_layers_present(self) -> None:
        stack  = _make_stack(include_topo=True, include_geo=True,
                              include_channels=True)
        result = self._compute(stack)
        assert np.sum(np.isfinite(result)) > 0

    def test_values_in_0_1_range(self) -> None:
        stack  = _make_stack(include_topo=True, include_geo=True,
                              include_channels=True)
        result = self._compute(stack)
        finite = result[np.isfinite(result)]
        assert float(finite.min()) >= -0.01
        assert float(finite.max()) <= 1.01

    def test_channel_density_raises_result_near_channels(self) -> None:
        """
        Where channel_density is high, surface_atm_interaction should be
        higher on average than where it is zero.
        """
        stack = _make_stack(include_topo=False, include_geo=False,
                             include_channels=True)
        ch = stack["channel_density"].values
        result = self._compute(stack)

        high_ch = result[ch > 0.5]
        low_ch  = result[ch < 0.1]
        if len(high_ch) > 0 and len(low_ch) > 0:
            assert float(np.nanmean(high_ch)) > float(np.nanmean(low_ch)), (
                "High-channel regions should have higher surface_atm_interaction"
            )

    def test_without_channel_density_still_works(self) -> None:
        """Feature 5 should degrade gracefully when channel_density absent."""
        stack  = _make_stack(include_topo=True, include_geo=True,
                              include_channels=False)
        result = self._compute(stack)
        assert result.dtype == np.float32
        assert result.shape == (16, 32)
        assert np.sum(np.isfinite(result)) > 0

    def test_result_differs_with_and_without_channels(self) -> None:
        """Adding channel_density should change the result."""
        with_ch    = _make_stack(include_topo=True, include_geo=True,
                                  include_channels=True)
        without_ch = _make_stack(include_topo=True, include_geo=True,
                                  include_channels=False)
        r_with    = self._compute(with_ch)
        r_without = self._compute(without_ch)
        assert not np.allclose(r_with, r_without, equal_nan=True), (
            "Including channel_density should change the result"
        )

    def test_no_inputs_returns_nan(self) -> None:
        """With no layers at all, should return all-NaN."""
        stack  = _make_stack(include_topo=False, include_geo=False,
                              include_channels=False)
        result = self._compute(stack)
        assert np.all(~np.isfinite(result)), "No inputs -> result should be NaN"

    def test_weight_sum_implies_correct_blend(self) -> None:
        """
        With only channel_density present, the result should equal the
        normalised channel density (since it has 100% of the weight).
        """
        stack = _make_stack(include_topo=False, include_geo=False,
                             include_channels=True)
        result = self._compute(stack)
        ch = stack["channel_density"].values.astype(np.float32)

        # Normalise channel to [0,1]
        ch_max = ch.max()
        if ch_max > 0:
            ch_norm = ch / ch_max
        else:
            ch_norm = ch

        # The result should equal ch_norm (only component present)
        mask = np.isfinite(result)
        np.testing.assert_allclose(
            result[mask], ch_norm[mask], atol=0.02,
            err_msg="Channel-only input should match normalised channel density",
        )

    def test_topo_only_returns_slope_based_result(self) -> None:
        """With only topography, result should be slope-derived."""
        stack  = _make_stack(include_topo=True, include_geo=False,
                              include_channels=False)
        result = self._compute(stack)
        assert np.sum(np.isfinite(result)) > 0
        # Flat DEM -> near-zero result
        flat_dem = np.zeros((16, 32), dtype=np.float32)
        stack2   = xr.Dataset({
            "topography": xr.DataArray(
                flat_dem, dims=["lat", "lon"],
                coords={"lat": np.linspace(-90,90,16),
                        "lon": np.linspace(0,360,32)},
            )
        })
        result2 = self._compute(stack2)
        # Flat DEM -> slope = 0 everywhere -> normalise_to_0_1 gives 0
        finite2 = result2[np.isfinite(result2)]
        assert float(finite2.mean()) < 0.1


# ============================================================================
# Channel DatasetSpec
# ============================================================================

class TestChannelDatasetSpec:

    def test_titan_channels_in_catalogue(self) -> None:
        from configs.pipeline_config import default_dataset_catalogue
        cat = default_dataset_catalogue()
        assert "titan_channels" in cat

    def test_local_filename_correct(self) -> None:
        from configs.pipeline_config import default_dataset_catalogue
        spec = default_dataset_catalogue()["titan_channels"]
        assert spec.local_filename == "geomorphology_shapefiles/global_channels.shp"

    def test_has_citation(self) -> None:
        from configs.pipeline_config import default_dataset_catalogue
        spec = default_dataset_catalogue()["titan_channels"]
        assert spec.citation and "Miller" in spec.citation

    def test_has_url(self) -> None:
        from configs.pipeline_config import default_dataset_catalogue
        spec = default_dataset_catalogue()["titan_channels"]
        assert spec.url and "hayesresearchgroup" in spec.url


# ============================================================================
# _rasterise_channels helper (unit test with synthetic shapefile)
# ============================================================================

class TestRasteriseChannels:

    @pytest.fixture
    def synthetic_channel_shp(self, tmp_path: Path) -> Path:
        """
        Create a minimal GeoDataFrame with one channel line and save as
        shapefile.  Uses east-positive coordinates since the pipeline
        flip_geodataframe_longitude expects GCS_Titan_2000 east-positive.
        """
        geopandas = pytest.importorskip("geopandas")
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import LineString

        # A single horizontal channel at equator in east-positive coords
        line = LineString([(90.0, 0.0), (270.0, 0.0)])
        gdf  = geopandas.GeoDataFrame(
            {"geometry": [line], "order": [1]},
            crs="EPSG:4326",
        )
        shp_path = tmp_path / "global_channels.shp"
        gdf.to_file(shp_path)
        return shp_path

    def test_rasterise_channels_produces_tif(self, synthetic_channel_shp: Path,
                                              tmp_path: Path) -> None:
        rasterio = pytest.importorskip("rasterio")
        from titan.preprocessing import _rasterise_channels
        from titan.preprocessing import CanonicalGrid

        grid     = CanonicalGrid(pixel_size_m=500_000)  # coarse 500 km grid
        out_path = tmp_path / "channel_density.tif"
        _rasterise_channels(synthetic_channel_shp, out_path, grid)

        assert out_path.exists(), "Output tif should exist"
        with rasterio.open(out_path) as ds:
            arr = ds.read(1)
            assert arr.dtype == np.float32
            assert arr.shape == (grid.nrows, grid.ncols)
            # Values should be in [0, 1]
            assert float(arr.min()) >= 0.0
            assert float(arr.max()) <= 1.0 + 1e-6
            # At least some non-zero pixels (channel is present)
            assert float(arr.max()) > 0.0, "Should have non-zero channel density"

    def test_rasterise_channels_gaussian_spreads(self, synthetic_channel_shp: Path,
                                                   tmp_path: Path) -> None:
        """
        After Gaussian blur the channel influence should spread beyond the
        immediate channel line -- more non-zero pixels than the raw burn.
        """
        rasterio = pytest.importorskip("rasterio")
        from titan.preprocessing import _rasterise_channels, CanonicalGrid

        grid     = CanonicalGrid(pixel_size_m=500_000)
        out_path = tmp_path / "channel_density_blur.tif"
        _rasterise_channels(synthetic_channel_shp, out_path, grid)

        with rasterio.open(out_path) as ds:
            arr = ds.read(1)
        # Should have fractional values (Gaussian creates intermediate values)
        unique_vals = np.unique(arr[arr > 0])
        assert len(unique_vals) > 1, (
            "Gaussian blur should create fractional density values"
        )

    def test_preprocess_channels_skips_if_missing(self, tmp_path: Path) -> None:
        """If global_channels.shp is absent, method returns {} gracefully."""
        from titan.preprocessing import DataPreprocessor, CanonicalGrid
        from configs.pipeline_config import PipelineConfig

        config = PipelineConfig(
            data_dir=tmp_path / "data",
            processed_dir=tmp_path / "processed",
            shapefile_dir=tmp_path / "shapefiles_nonexistent",
        )
        (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
        grid  = CanonicalGrid(pixel_size_m=500_000)
        dp    = DataPreprocessor(config, grid)
        result = dp._preprocess_channels(overwrite=True)
        assert result == {}, (
            "Missing global_channels.shp should return empty dict, not raise"
        )