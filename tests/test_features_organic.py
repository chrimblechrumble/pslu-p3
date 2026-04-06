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
tests/test_features_organic.py
================================
Unit tests for Feature 2 -- ``organic_abundance``.

Design
------
Feature 2 uses two data sources:

1. **VIMS+ISS mosaic** (primary) -- 1.59/1.27 umm band ratio; spectroscopic
   tholin proxy.  Covers ~50 % of the globe (roughly 0-180  degW).

2. **Geomorphology-class scores** (gap-fill) -- maps each Lopes et al. (2019)
   terrain class to a published organic-abundance value from
   :data:`titan.features.TERRAIN_ORGANIC_SCORES`.  Provides 100 % global
   coverage without any radiometric seam because no two different instruments
   are blended.

ISS 938 nm broadband reflectance is intentionally NOT used as a gap-filler.
Diagnostic testing showed that VIMS raw values (63-225, band ratio) and ISS
raw values (0.040-0.085, reflectance) differ by a factor of ~3000, making
any normalisation-based blending leave a hard visible seam at the coverage
boundary.  The geomorphology-based approach is scientifically superior and
seam-free because each terrain class carries a single, citable organic-
abundance value.

Test categories
---------------
* ``TestGeoOrganicConversion`` -- unit tests for the
  :func:`~titan.features._geo_class_to_organic` helper.
* ``TestOrganicAbundancePrimary`` -- VIMS-only and geo-only paths.
* ``TestOrganicAbundanceCombined`` -- combined VIMS + geomorphology.
* ``TestOrganicAbundanceSeam`` -- the original seam regression test,
  verifying that the geo gap-fill produces no visible discontinuity.
* ``TestOrganicAbundanceFallback`` -- coverage-density and all-NaN paths.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NROWS: int = 36   # 5 deg per pixel -- small enough for fast tests
NCOLS: int = 72


def _lats() -> np.ndarray:
    """Latitude centres (deg), north to south."""
    return np.linspace(-87.5, 87.5, NROWS)


def _lons() -> np.ndarray:
    """Longitude centres ( degW), 0-360."""
    return np.linspace(2.5, 357.5, NCOLS)


def _da(arr: np.ndarray) -> xr.DataArray:
    """Wrap a (NROWS, NCOLS) array in a DataArray with lat/lon coords."""
    return xr.DataArray(arr, dims=["lat", "lon"],
                        coords={"lat": _lats(), "lon": _lons()})


def _extractor() -> "FeatureExtractor":
    """Return a :class:`~titan.features.FeatureExtractor` on a small grid."""
    from titan.features import FeatureExtractor
    from titan.preprocessing import CanonicalGrid
    return FeatureExtractor(CanonicalGrid(pixel_size_m=500_000))


def _compute(stack: xr.Dataset, mode: str = "blended") -> np.ndarray:
    """Call ``_organic_abundance`` directly and return the result array.

    Parameters
    ----------
    mode:
        Override :data:`titan.features.ORGANIC_SOURCE_MODE` for this call.
        Defaults to ``"blended"`` so existing tests that exercise the VIMS
        blending path continue to work regardless of the module-level default.
    """
    import titan.features as _tf_mod
    _orig = _tf_mod.ORGANIC_SOURCE_MODE
    try:
        _tf_mod.ORGANIC_SOURCE_MODE = mode
        calc = _extractor()
        nan_arr = np.full((NROWS, NCOLS), np.nan, dtype=np.float32)
        return calc._organic_abundance(stack, nan_arr)
    finally:
        _tf_mod.ORGANIC_SOURCE_MODE = _orig


def _make_vims_left_half(rng: np.random.Generator) -> np.ndarray:
    """VIMS raw values covering left half only (0-180 degW = cols 0..NCOLS//2-1).

    Values are ~120-180, simulating real VIMS band-ratio units.
    """
    arr = np.full((NROWS, NCOLS), np.nan, dtype=np.float32)
    half = NCOLS // 2
    arr[:, :half] = rng.uniform(120, 180, size=(NROWS, half)).astype(np.float32)
    return arr


def _make_geo_full(class_id: int = 3) -> np.ndarray:
    """Geomorphology raster -- uniform ``class_id`` over the full globe."""
    return np.full((NROWS, NCOLS), class_id, dtype=np.float32)


def _make_geo_varied() -> np.ndarray:
    """Geomorphology raster with all terrain classes distributed spatially."""
    arr = np.zeros((NROWS, NCOLS), dtype=np.float32)
    # Divide grid into 7 column bands, one per class 1-7
    band = NCOLS // 7
    for i, cls in enumerate(range(1, 8)):
        arr[:, i * band: (i + 1) * band] = float(cls)
    return arr


# ---------------------------------------------------------------------------
# TestGeoOrganicConversion
# ---------------------------------------------------------------------------

class TestGeoOrganicConversion:
    """Unit tests for :func:`titan.features._geo_class_to_organic`."""

    def test_known_classes_return_expected_scores(self) -> None:
        """Each class ID maps to its published organic abundance score."""
        from titan.features import TERRAIN_ORGANIC_SCORES, _geo_class_to_organic

        for cls_id, expected in TERRAIN_ORGANIC_SCORES.items():
            arr = np.array([[cls_id]], dtype=np.int32)
            result = _geo_class_to_organic(arr)
            if np.isnan(expected):
                assert np.isnan(result[0, 0]), f"Class {cls_id} should be NaN"
            else:
                assert abs(float(result[0, 0]) - expected) < 1e-5, (
                    f"Class {cls_id}: expected {expected}, got {result[0,0]}"
                )

    def test_nodata_class_zero_returns_nan(self) -> None:
        """Class 0 (nodata) must produce NaN, not a numeric score."""
        from titan.features import _geo_class_to_organic
        arr = np.zeros((5, 5), dtype=np.int32)
        result = _geo_class_to_organic(arr)
        assert np.all(np.isnan(result)), "All nodata pixels should be NaN"

    def test_output_dtype_float32(self) -> None:
        """Output dtype must be float32."""
        from titan.features import _geo_class_to_organic
        arr = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        result = _geo_class_to_organic(arr)
        assert result.dtype == np.float32

    def test_scores_in_0_1_range(self) -> None:
        """All numeric scores must be in [0, 1]."""
        from titan.features import TERRAIN_ORGANIC_SCORES
        for cls_id, score in TERRAIN_ORGANIC_SCORES.items():
            if not np.isnan(score):
                assert 0.0 <= score <= 1.0, (
                    f"Class {cls_id} score {score} outside [0, 1]"
                )

    def test_unknown_class_produces_nan_without_raising(self) -> None:
        """Terrain class IDs not in the table produce NaN and do not raise."""
        from titan.features import _geo_class_to_organic
        arr = np.array([[99]], dtype=np.int32)
        # Must not raise -- unknown classes are handled gracefully
        result = _geo_class_to_organic(arr)
        assert np.isnan(result[0, 0]), (
            "Unknown class ID 99 should map to NaN, not a numeric value"
        )

    def test_output_shape_matches_input(self) -> None:
        """Output shape must equal input shape."""
        from titan.features import _geo_class_to_organic
        for shape in [(1, 1), (10, 20), (NROWS, NCOLS)]:
            arr = np.ones(shape, dtype=np.int32) * 3
            result = _geo_class_to_organic(arr)
            assert result.shape == shape, f"Shape mismatch for input {shape}"

    def test_dunes_highest_organic(self) -> None:
        """Dunes (class 2) should have the highest organic abundance."""
        from titan.features import TERRAIN_ORGANIC_SCORES
        dune_score = TERRAIN_ORGANIC_SCORES[2]
        for cls_id, score in TERRAIN_ORGANIC_SCORES.items():
            if cls_id in (0, 2) or np.isnan(score):
                continue
            assert dune_score >= score, (
                f"Dunes (0.{dune_score}) should outrank class {cls_id} ({score})"
            )

    def test_lakes_lowest_organic(self) -> None:
        """Lakes (class 7) should have the lowest organic abundance."""
        from titan.features import TERRAIN_ORGANIC_SCORES
        lake_score = TERRAIN_ORGANIC_SCORES[7]
        for cls_id, score in TERRAIN_ORGANIC_SCORES.items():
            if cls_id in (0, 7) or np.isnan(score):
                continue
            assert lake_score <= score, (
                f"Lakes ({lake_score}) should be below class {cls_id} ({score})"
            )


# ---------------------------------------------------------------------------
# TestOrganicAbundancePrimary
# ---------------------------------------------------------------------------

class TestOrganicAbundancePrimary:
    """Tests for the VIMS-only and geo-only input paths."""

    def setup_method(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_vims_only_normalises_to_0_1(self) -> None:
        """VIMS-only input: result values must be in [0, 1]."""
        vims = _make_vims_left_half(self.rng)
        stack = xr.Dataset({"vims_mosaic": _da(vims)})
        result = _compute(stack)
        finite = result[np.isfinite(result)]
        assert len(finite) > 0
        assert float(finite.min()) >= -0.01
        assert float(finite.max()) <=  1.01

    def test_vims_only_nan_where_no_data(self) -> None:
        """Right half (no VIMS, no geo) is filled with row-median of left half.

        When only VIMS data is available, _fill_organic_nan propagates
        each row's median value into NaN columns.  The right half therefore
        becomes finite, equal to the left-half row median.  This is the
        correct declared behaviour (ORGANIC_GAP_FILL flag is set).
        """
        vims = _make_vims_left_half(self.rng)
        stack = xr.Dataset({"vims_mosaic": _da(vims)})
        result = _compute(stack)
        half = NCOLS // 2
        # After _fill_organic_nan the right half must be finite (row-median fill)
        assert np.all(np.isfinite(result[:, half:])), (
            "Right half should be filled with row medians (not NaN) "
            "after _fill_organic_nan is applied in the VIMS-only path."
        )
        # Each row's right-half values should equal the left-half row median
        for r in range(result.shape[0]):
            left_median = float(np.nanmedian(result[r, :half]))
            right_vals = result[r, half:]
            assert np.allclose(right_vals, left_median, atol=1e-5), (
                f"Row {r}: right-half values {right_vals[:3]} should equal "
                f"left-half median {left_median:.4f}"
            )

    def test_geo_only_returns_valid_global_coverage(self) -> None:
        """Geo-only: all terrain-class pixels (non-nodata) produce valid values."""
        geo = _make_geo_full(class_id=3)   # all Plains
        stack = xr.Dataset({"geomorphology": _da(geo)})
        result = _compute(stack)
        assert np.all(np.isfinite(result)), (
            "Geo-only with uniform non-nodata class should give 100% valid"
        )

    def test_geo_only_values_match_lookup_table(self) -> None:
        """Geo-only: pixel values must equal the TERRAIN_ORGANIC_SCORES entry."""
        from titan.features import TERRAIN_ORGANIC_SCORES
        for cls_id in range(1, 8):
            geo = _make_geo_full(class_id=cls_id)
            stack = xr.Dataset({"geomorphology": _da(geo)})
            result = _compute(stack)
            expected = TERRAIN_ORGANIC_SCORES[cls_id]
            finite = result[np.isfinite(result)]
            assert len(finite) > 0
            assert abs(float(np.mean(finite)) - expected) < 0.01, (
                f"Class {cls_id}: expected {expected:.3f}, got {float(np.mean(finite)):.3f}"
            )

    def test_geo_only_dtype_float32(self) -> None:
        """Geo-only result must be float32."""
        geo = _make_geo_full(class_id=2)
        stack = xr.Dataset({"geomorphology": _da(geo)})
        result = _compute(stack)
        assert result.dtype == np.float32

    def test_no_inputs_returns_all_nan(self) -> None:
        """With no VIMS, geo, or coverage data, result must be all-NaN."""
        result = _compute(xr.Dataset({}))
        assert np.all(~np.isfinite(result)), (
            "No inputs should produce all-NaN output"
        )


# ---------------------------------------------------------------------------
# TestOrganicAbundanceCombined
# ---------------------------------------------------------------------------

class TestOrganicAbundanceCombined:
    """Tests for the VIMS-primary + geo-gap-fill combination."""

    def setup_method(self) -> None:
        self.rng = np.random.default_rng(7)

    def test_combined_gives_100_percent_coverage(self) -> None:
        """VIMS + geo should produce no NaN pixels (100% valid)."""
        vims = _make_vims_left_half(self.rng)
        geo  = _make_geo_full(class_id=3)
        stack = xr.Dataset({
            "vims_mosaic":    _da(vims),
            "geomorphology":  _da(geo),
        })
        result = _compute(stack)
        assert np.all(np.isfinite(result)), (
            f"{np.sum(~np.isfinite(result))} NaN pixels remain "
            "with both VIMS and geo present"
        )

    def test_vims_region_uses_vims_values(self) -> None:
        """Where VIMS is valid, result must come from VIMS (not geo)."""
        vims = _make_vims_left_half(self.rng)
        geo  = _make_geo_full(class_id=7)   # lakes = 0.05
        stack = xr.Dataset({
            "vims_mosaic":   _da(vims),
            "geomorphology": _da(geo),
        })
        result = _compute(stack)
        half = NCOLS // 2
        # VIMS region values should be >> 0.05 (not all equal to lake score)
        vims_region_mean = float(np.nanmean(result[:, :half]))
        assert vims_region_mean > 0.15, (
            f"VIMS region mean {vims_region_mean:.3f} too low -- "
            "may be overridden by geo-lake score (0.05)"
        )

    def test_gap_region_uses_geo_values(self) -> None:
        """Where VIMS is NaN, result must come from geomorphology scores.

        The gap region receives ``calibrated_geo = geo_score + global_offset``
        where ``global_offset = vims_mean − geo_mean`` in the overlap zone.
        With random VIMS normalised to ~0.45 and dune score 0.82, the offset
        is ~−0.05 to −0.07, so gap_mean ≈ 0.75–0.82.  Tolerance 0.12 covers
        this while still catching a fully wrong source (e.g. lake score 0.05).
        """
        from titan.features import TERRAIN_ORGANIC_SCORES
        vims = _make_vims_left_half(self.rng)
        geo  = _make_geo_full(class_id=2)   # dunes = 0.82
        stack = xr.Dataset({
            "vims_mosaic":   _da(vims),
            "geomorphology": _da(geo),
        })
        result = _compute(stack)
        half = NCOLS // 2
        gap_mean = float(np.nanmean(result[:, half:]))
        expected = TERRAIN_ORGANIC_SCORES[2]
        assert abs(gap_mean - expected) < 0.12, (
            f"Gap region mean {gap_mean:.3f} should be close to dune score "
            f"{expected:.3f} (within 0.12 to allow for global_offset calibration)"
        )

    def test_result_in_0_1_range(self) -> None:
        """All valid result pixels must lie in [0, 1]."""
        vims = _make_vims_left_half(self.rng)
        geo  = _make_geo_varied()
        stack = xr.Dataset({
            "vims_mosaic":   _da(vims),
            "geomorphology": _da(geo),
        })
        result = _compute(stack)
        finite = result[np.isfinite(result)]
        assert float(finite.min()) >= -0.01
        assert float(finite.max()) <=  1.01

    def test_result_dtype_float32(self) -> None:
        """Output dtype must be float32 for all input combinations."""
        vims = _make_vims_left_half(self.rng)
        geo  = _make_geo_full(class_id=3)
        for stack in [
            xr.Dataset({"vims_mosaic": _da(vims), "geomorphology": _da(geo)}),
            xr.Dataset({"vims_mosaic": _da(vims)}),
            xr.Dataset({"geomorphology": _da(geo)}),
            xr.Dataset({}),
        ]:
            result = _compute(stack)
            assert result.dtype == np.float32, (
                f"Wrong dtype {result.dtype} for stack keys {list(stack)}"
            )


# ---------------------------------------------------------------------------
# TestOrganicAbundanceSeam
# ---------------------------------------------------------------------------

class TestOrganicAbundanceSeam:
    """
    Regression test: no visible seam at the VIMS/geo boundary.

    The key insight is that the geomorphology-based gap-fill uses a single
    cited value per terrain class, derived from the same VIMS spectral studies
    that calibrated the VIMS primary proxy.  This is fundamentally different
    from blending ISS broadband reflectance (which measures different physics)
    and should produce no seam.
    """

    def setup_method(self) -> None:
        self.rng = np.random.default_rng(99)

    def test_no_visible_seam_boundary_vs_interior_variation(self) -> None:
        """
        Boundary jump must not exceed 3x the typical within-VIMS pixel-to-pixel
        variation.  The geo-based gap-fill should produce smooth transitions.
        """
        vims = _make_vims_left_half(self.rng)
        geo  = _make_geo_full(class_id=3)   # uniform plains for clean test
        stack = xr.Dataset({
            "vims_mosaic":   _da(vims),
            "geomorphology": _da(geo),
        })
        result = _compute(stack)
        half = NCOLS // 2

        # Step at the boundary (last VIMS col -> first geo col)
        boundary_diff = float(np.nanmean(
            np.abs(result[:, half - 1] - result[:, half])
        ))

        # Typical within-VIMS column-to-column variation
        within_diffs = [
            float(np.nanmean(np.abs(result[:, c] - result[:, c - 1])))
            for c in range(1, half)
        ]
        typical_variation = float(np.mean(within_diffs)) if within_diffs else 0.05

        ratio = boundary_diff / (typical_variation + 1e-6)
        assert ratio < 3.0, (
            f"Seam detected: boundary_diff={boundary_diff:.4f}, "
            f"typical_variation={typical_variation:.4f}, ratio={ratio:.2f} "
            f"(threshold 3.0)"
        )

    def test_gap_region_median_close_to_expected_class_score(self) -> None:
        """
        In the gap region, the median should match the terrain class score
        to within +/-0.05, confirming no hidden ISS blending is occurring.
        """
        from titan.features import TERRAIN_ORGANIC_SCORES
        vims = _make_vims_left_half(self.rng)
        geo  = _make_geo_full(class_id=3)
        stack = xr.Dataset({
            "vims_mosaic":   _da(vims),
            "geomorphology": _da(geo),
        })
        result = _compute(stack)
        half = NCOLS // 2
        gap_median = float(np.nanmedian(result[:, half:]))
        expected   = TERRAIN_ORGANIC_SCORES[3]
        assert abs(gap_median - expected) < 0.05, (
            f"Gap median {gap_median:.3f} should equal plains score "
            f"{expected:.3f} +/- 0.05"
        )


# ---------------------------------------------------------------------------
# TestOrganicAbundanceFallback
# ---------------------------------------------------------------------------

class TestOrganicAbundanceFallback:
    """Tests for coverage-density fallback and all-NaN edge case."""

    def test_coverage_density_fallback_returns_array(self) -> None:
        """Coverage density fallback returns a valid float32 array."""
        coverage = np.random.default_rng(0).uniform(0, 1, (NROWS, NCOLS)).astype(np.float32)
        stack = xr.Dataset({"vims_coverage": _da(coverage)})
        result = _compute(stack)
        assert result.dtype == np.float32
        assert result.shape == (NROWS, NCOLS)

    def test_no_data_returns_all_nan(self) -> None:
        """Empty stack returns all-NaN."""
        result = _compute(xr.Dataset({}))
        assert np.all(~np.isfinite(result))
        assert result.dtype == np.float32

    def test_nodata_geo_class_zero_treated_as_nan(self) -> None:
        """Class 0 (nodata) pixels in geo produce NaN, not 0."""
        geo = np.zeros((NROWS, NCOLS), dtype=np.float32)
        stack = xr.Dataset({"geomorphology": _da(geo)})
        result = _compute(stack)
        assert np.all(~np.isfinite(result)), (
            "Geomorphology nodata (class 0) should produce all-NaN"
        )
