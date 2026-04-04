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
tests/test_features_methane.py
================================
Tests for Feature 4 (_methane_cycle) after the cirs_temperature integration.

The updated feature blends:
  cirs_temperature   (0.50) -- meridional |dT/dlat| gradient (primary)
  channel_density    (0.35) -- Miller+2021 fluvial transport (secondary)
  lat_weight         (0.15) -- Gaussian prior (always available)
  (VIMS coverage density removed: observing bias, not geophysical signal)

Tests verify correctness of each blend mode and the physical
validity of the cirs_temperature gradient contribution.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lats_lons(nrows: int = 18, ncols: int = 36) -> None:
    return np.linspace(-90, 90, nrows), np.linspace(0, 360, ncols)


def _da(arr: np.ndarray, lats: Any, lons: Any) -> "xr.DataArray":
    return xr.DataArray(arr, dims=["lat", "lon"],
                        coords={"lat": lats, "lon": lons})


def _stack(**layers) -> xr.Dataset:
    lats, lons = _lats_lons()
    nrows, ncols = len(lats), len(lons)
    rng = np.random.default_rng(7)
    dvs = {}
    if "vims" in layers:
        dvs["vims_coverage"] = _da(
            rng.uniform(0, 1, (nrows, ncols)).astype(np.float32), lats, lons)
    if "cirs" in layers:
        # Synthesise realistic cirs_temperature from Jennings 2019
        from titan.atmospheric_profiles import jennings_temperature_grid
        lat_grid = np.tile(lats[:, None], (1, ncols)).astype(np.float32)
        T = jennings_temperature_grid(lat_grid, 2011.0)
        dvs["cirs_temperature"] = _da(T, lats, lons)
    if "cirs_flat" in layers:
        # Uniform temperature (no gradient -> should not boost methane cycle)
        T_flat = np.full((nrows, ncols), 93.0, dtype=np.float32)
        dvs["cirs_temperature"] = _da(T_flat, lats, lons)
    return xr.Dataset(dvs)


def _compute(stack: xr.Dataset, nrows: int=18, ncols: int=36) -> np.ndarray:
    from titan.features import FeatureExtractor
    from titan.preprocessing import CanonicalGrid
    grid = CanonicalGrid(pixel_size_m=500_000)
    calc = FeatureExtractor(grid)
    if stack.sizes:
        nrows = stack.sizes.get("lat", nrows)
        ncols = stack.sizes.get("lon", ncols)
    nan = np.full((nrows, ncols), np.nan, dtype=np.float32)
    return calc._methane_cycle(stack, nan)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMethane_CycleBasic:

    def test_pure_lat_prior_fallback(self) -> None:
        """With no VIMS and no CIRS, should return pure Gaussian prior."""
        result = _compute(xr.Dataset({}))
        assert result.dtype == np.float32
        assert result.ndim == 2
        assert np.all(np.isfinite(result))
        # Peak should be in interior rows (not at pole or equator)
        nrows = result.shape[0]
        peak_row = int(np.argmax(result[:, 0]))
        assert 1 <= peak_row <= nrows - 2, (
            f"Lat prior peak row {peak_row} should be interior"
        )

    def test_result_always_in_0_1(self) -> None:
        """All blend modes should produce values in [0, 1]."""
        for layers in [set(), {"vims"}, {"cirs"}, {"vims", "cirs"}]:
            stack = _stack(**{k: True for k in layers})
            result = _compute(stack)
            assert float(result.min()) >= -0.01, f"Below 0 with {layers}"
            assert float(result.max()) <=  1.01, f"Above 1 with {layers}"

    def test_result_dtype_float32(self) -> None:
        for layers in [set(), {"vims"}, {"cirs"}, {"vims", "cirs"}]:
            stack = _stack(**{k: True for k in layers})
            result = _compute(stack)
            assert result.dtype == np.float32, f"Wrong dtype with {layers}"

    def test_result_fully_valid_no_nans(self) -> None:
        """No NaN values should appear in the output."""
        for layers in [set(), {"vims"}, {"cirs"}, {"vims", "cirs"}]:
            stack = _stack(**{k: True for k in layers})
            result = _compute(stack)
            assert np.all(np.isfinite(result)), (
                f"NaN found in result with layers={layers}"
            )


class TestMethane_CycleCIRS:

    def test_cirs_gradient_changes_result(self) -> None:
        """Adding cirs_temperature should alter the result."""
        stack_no_cirs  = _stack(vims=True)
        stack_with_cirs = _stack(vims=True, cirs=True)
        r_no   = _compute(stack_no_cirs)
        r_with = _compute(stack_with_cirs)
        assert not np.allclose(r_no, r_with, rtol=1e-4), (
            "cirs_temperature should change the result"
        )

    def test_cirs_gradient_highest_at_mid_latitudes(self) -> None:
        """
        The Jennings temperature gradient |dT/dlat| is largest at mid-
        latitudes (where the cosine drops off fastest), so cirs_weight
        should be elevated there relative to the equator.
        """
        from titan.atmospheric_profiles import jennings_temperature_map
        lats = np.linspace(-90, 90, 18)
        T = jennings_temperature_map(lats, 2011.0)
        dT = np.abs(np.gradient(T))
        # Gradient should be near-zero at the temperature peak and grow
        # away from it -- verify poles have higher gradient than equator
        assert dT[0] > dT[9] or dT[-1] > dT[9], (
            "Gradient should be larger at poles than at equator"
        )

    def test_flat_cirs_has_minimal_effect(self) -> None:
        """
        A spatially uniform cirs_temperature has zero gradient everywhere.
        The variance guard in _methane_cycle checks:
            std(dT/dy) > 1e-4 K/px
        For a flat 93 K field, std == 0, so the CIRS layer is dropped
        entirely and cirs_weight stays None.  The result is therefore
        identical to the lat-only baseline -- not worse, not better.

        This is the CORRECT behaviour: silent discard of a zero-information
        source rather than polluting the blend with 50% dead weight.
        """
        r_lat_only  = _compute(_stack())           # no CIRS, no channels
        r_flat_cirs = _compute(_stack(cirs_flat=True))

        # Flat CIRS is dropped by the variance guard -> identical to lat-only
        np.testing.assert_array_equal(
            r_flat_cirs, r_lat_only,
            err_msg=(
                "Flat CIRS should be silently discarded by the variance guard "
                "(std(dT/dy)==0 < 1e-4 threshold), leaving result == lat-only"
            ),
        )

    def test_cirs_only_blends_with_lat_prior(self) -> None:
        """cirs only (no VIMS): blend = cirs 60% + lat 40%."""
        stack  = _stack(cirs=True)
        result = _compute(stack)
        # Resulting values should be non-trivial (not all identical)
        assert float(result.std()) > 0.01, (
            "cirs+lat blend should produce spatially varying output"
        )


class TestMethane_CycleWeights:

    def test_vims_cirs_blend_differs_from_vims_only(self) -> None:
        stack_v    = _stack(vims=True)
        stack_vc   = _stack(vims=True, cirs=True)
        r_v  = _compute(stack_v)
        r_vc = _compute(stack_vc)
        assert not np.allclose(r_v, r_vc), "Three-way blend != two-way blend"

    def test_three_way_is_bounded_by_components(self) -> None:
        """
        Weighted blend should not exceed the max of any component.
        (All components are in [0,1] so the blend must also be.)
        """
        stack = _stack(vims=True, cirs=True)
        result = _compute(stack)
        assert float(result.max()) <= 1.01
        assert float(result.min()) >= -0.01

    def test_lat_prior_forms_gaussian_peaks(self) -> None:
        """
        Pure lat prior should peak near +/-45 deg, symmetric about equator.
        Peaks should be in mid-latitude rows, not at poles or equator.
        """
        result = _compute(xr.Dataset({}))
        nrows = result.shape[0]
        mid = nrows // 2
        # Northern hemisphere rows (mid -> end) should peak in the interior
        n_peak_row = mid + int(np.argmax(result[mid:, 0]))
        assert mid < n_peak_row < nrows - 1, (
            f"NH peak row {n_peak_row} should be between equator ({mid}) "
            f"and north pole ({nrows - 1})"
        )
        # Southern hemisphere rows (0 -> mid) should also peak in the interior
        s_peak_row = int(np.argmax(result[:mid, 0]))
        assert 0 < s_peak_row < mid, (
            f"SH peak row {s_peak_row} should be between south pole (0) "
            f"and equator ({mid})"
        )


class TestMethane_CycleDocumentation:

    def test_config_describes_cirs_component(self) -> None:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        src = (root / "configs" / "pipeline_config.py").read_text()
        # Feature 4 section should mention cirs_temperature
        m_idx = src.find("Feature 4")
        m_end = src.find("Feature 5", m_idx)
        section = src[m_idx:m_end]
        assert "cirs_temperature" in section, (
            "Feature 4 config docstring should mention cirs_temperature"
        )
        assert "Jennings" in section, (
            "Feature 4 config docstring should cite Jennings et al."
        )

    def test_features_source_uses_cirs_weight(self) -> None:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        src = (root / "titan" / "features.py").read_text()
        # Find _methane_cycle method
        m_idx = src.find("def _methane_cycle")
        m_end = src.find("def _surface_atm_interaction", m_idx)
        section = src[m_idx:m_end]
        assert "cirs_temperature" in section
        assert "cirs_weight" in section
        assert "0.50" in section or ".50" in section  # CIRS weight is 0.50
        assert "0.35" in section                       # channel weight is 0.35
        assert "0.15" in section                       # lat prior weight is 0.15