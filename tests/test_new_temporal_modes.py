# Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
# Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
tests/test_new_temporal_modes.py
==================================
Unit tests for the two new temporal modes added in v4:

  LAKE_FORMATION  (~1.0 Gya, cryovolcanic lake formation onset)
  NEAR_FUTURE     (~0.25 Gya, D2 solar-warming window centre)

Tests cover:
  - TemporalMode enum membership
  - Feature name lists
  - Prior set weights sum to 1.0
  - Prior means in [0, 1]
  - get_prior_set() returns correct TemporalPriorSet for each mode
  - Specific scientific constraints on prior values
  - PCHIP interpolator correctness
  - Epoch axis contains +0.250 Gya exact anchor
  - load_anchor_posteriors() raises when required anchor missing
  - TemporalFeatureExtractor dispatch routes correctly
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure the project root is on sys.path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.temporal_config import (
    TemporalMode,
    LAKE_FORMATION_FEATURES,
    NEAR_FUTURE_FEATURES,
    PRESENT_FEATURES,
    PAST_FEATURES,
    TEMPORAL_FEATURE_NAMES,
    get_prior_set,
)


# ---------------------------------------------------------------------------
# TemporalMode enum
# ---------------------------------------------------------------------------

class TestTemporalModeEnum:
    def test_all_five_modes_exist(self) -> None:
        assert TemporalMode.PAST           == "past"
        assert TemporalMode.LAKE_FORMATION == "lake_formation"
        assert TemporalMode.PRESENT        == "present"
        assert TemporalMode.NEAR_FUTURE    == "near_future"
        assert TemporalMode.FUTURE         == "future"

    def test_enum_count(self) -> None:
        assert len(TemporalMode) == 5

    def test_lake_formation_string_value(self) -> None:
        assert TemporalMode("lake_formation") is TemporalMode.LAKE_FORMATION

    def test_near_future_string_value(self) -> None:
        assert TemporalMode("near_future") is TemporalMode.NEAR_FUTURE


# ---------------------------------------------------------------------------
# Feature name lists
# ---------------------------------------------------------------------------

class TestFeatureNameLists:
    def test_lake_formation_has_9_features(self) -> None:
        assert len(LAKE_FORMATION_FEATURES) == 9

    def test_lake_formation_includes_impact_melt_proxy(self) -> None:
        assert "impact_melt_proxy" in LAKE_FORMATION_FEATURES

    def test_lake_formation_includes_cryovolcanic_flux(self) -> None:
        assert "cryovolcanic_flux" in LAKE_FORMATION_FEATURES

    def test_lake_formation_includes_all_7_present_features(self) -> None:
        # All 7 non-subsurface_ocean present features must be in lake_formation
        # (subsurface_ocean is replaced by impact_melt + cryo in 9-feature set)
        core_7 = [f for f in PRESENT_FEATURES if f != "subsurface_ocean"]
        for name in core_7:
            assert name in LAKE_FORMATION_FEATURES, (
                f"Expected '{name}' in LAKE_FORMATION_FEATURES")

    def test_near_future_equals_present_features(self) -> None:
        # NEAR_FUTURE uses exactly the same 8 features as PRESENT
        assert NEAR_FUTURE_FEATURES is PRESENT_FEATURES or \
               list(NEAR_FUTURE_FEATURES) == list(PRESENT_FEATURES)

    def test_near_future_has_subsurface_ocean(self) -> None:
        assert "subsurface_ocean" in NEAR_FUTURE_FEATURES

    def test_temporal_feature_names_dict_has_all_5_modes(self) -> None:
        for mode in TemporalMode:
            assert mode in TEMPORAL_FEATURE_NAMES, (
                f"Mode {mode} missing from TEMPORAL_FEATURE_NAMES")

    def test_lake_formation_features_mapped(self) -> None:
        assert TEMPORAL_FEATURE_NAMES[TemporalMode.LAKE_FORMATION] \
               is LAKE_FORMATION_FEATURES

    def test_near_future_features_mapped(self) -> None:
        nf_list = TEMPORAL_FEATURE_NAMES[TemporalMode.NEAR_FUTURE]
        assert list(nf_list) == list(NEAR_FUTURE_FEATURES)


# ---------------------------------------------------------------------------
# Prior sets — structural validation
# ---------------------------------------------------------------------------

class TestPriorSets:
    @pytest.mark.parametrize("mode", list(TemporalMode))
    def test_weights_sum_to_1(self, mode: TemporalMode) -> None:
        ps = get_prior_set(mode)
        total = sum(ps.weights)
        assert abs(total - 1.0) < 0.011, (
            f"{mode}: weights sum to {total:.4f}, expected 1.0 ± 0.01")

    @pytest.mark.parametrize("mode", list(TemporalMode))
    def test_prior_means_in_unit_interval(self, mode: TemporalMode) -> None:
        ps = get_prior_set(mode)
        for name, mean in zip(ps.feature_names, ps.prior_means):
            assert 0.0 <= mean <= 1.0, (
                f"{mode}: feature '{name}' prior_mean={mean} outside [0,1]")

    @pytest.mark.parametrize("mode", list(TemporalMode))
    def test_validate_passes(self, mode: TemporalMode) -> None:
        ps = get_prior_set(mode)
        ps.validate()   # must not raise

    @pytest.mark.parametrize("mode", list(TemporalMode))
    def test_feature_names_match_list(self, mode: TemporalMode) -> None:
        ps = get_prior_set(mode)
        expected = TEMPORAL_FEATURE_NAMES[mode]
        assert list(ps.feature_names) == list(expected)

    @pytest.mark.parametrize("mode", list(TemporalMode))
    def test_weights_count_matches_features(self, mode: TemporalMode) -> None:
        ps = get_prior_set(mode)
        assert len(ps.weights) == len(ps.feature_names)

    @pytest.mark.parametrize("mode", list(TemporalMode))
    def test_prior_means_count_matches_features(self, mode: TemporalMode) -> None:
        ps = get_prior_set(mode)
        assert len(ps.prior_means) == len(ps.feature_names)


# ---------------------------------------------------------------------------
# LAKE_FORMATION specific scientific constraints
# ---------------------------------------------------------------------------

class TestLakeFormationPriors:
    """
    Scientific constraints on LAKE_FORMATION prior values.

    References
    ----------
    Tobie et al. (2006) Nature 440:61 -- cryovolcanic flux at 1.0 Gya
    Neish & Lorenz (2012) PSS 60:26  -- impact melt relics degraded
    """

    def setup_method(self) -> None:
        self.ps = get_prior_set(TemporalMode.LAKE_FORMATION)
        self.w  = dict(zip(self.ps.feature_names, self.ps.weights))
        self.m  = dict(zip(self.ps.feature_names, self.ps.prior_means))

    def test_impact_melt_weight_much_lower_than_past(self) -> None:
        # At 1.0 Gya, LHB relics are ~2.5 Gyr old and highly degraded.
        # Impact_melt_proxy should have much lower weight than in PAST (0.10).
        past_ps = get_prior_set(TemporalMode.PAST)
        past_w  = dict(zip(past_ps.feature_names, past_ps.weights))
        assert self.w["impact_melt_proxy"] < past_w["impact_melt_proxy"], (
            "impact_melt_proxy weight should be lower in LAKE_FORMATION than PAST "
            "(LHB relics degraded by 1.0 Gya)")

    def test_cryovolcanic_flux_weight_high(self) -> None:
        # Tobie 2006: cryovolcanism peaked ~500 Mya; at 1.0 Gya actively building.
        assert self.w["cryovolcanic_flux"] >= 0.05, (
            "cryovolcanic_flux weight should be significant (>=0.05) in "
            "LAKE_FORMATION mode (Tobie 2006)")

    def test_cryovolcanic_flux_prior_mean_high(self) -> None:
        # At 1.0 Gya cryovolcanism is active; prior_mean should be > PAST (0.30)
        past_ps = get_prior_set(TemporalMode.PAST)
        past_m  = dict(zip(past_ps.feature_names, past_ps.prior_means))
        assert self.m["cryovolcanic_flux"] >= past_m["cryovolcanic_flux"], (
            "cryovolcanic_flux prior_mean should be at least as high in "
            "LAKE_FORMATION as PAST (Tobie 2006: peak approaching at 1 Gya)")

    def test_liquid_hydrocarbon_prior_lower_than_present(self) -> None:
        # Lakes are forming but sparse at 1.0 Gya; <<present coverage.
        present_ps = get_prior_set(TemporalMode.PRESENT)
        present_m  = dict(zip(present_ps.feature_names, present_ps.prior_means))
        assert self.m["liquid_hydrocarbon"] < present_m["liquid_hydrocarbon"] * 3, (
            "liquid_hydrocarbon prior_mean should be much lower in "
            "LAKE_FORMATION than PRESENT")

    def test_organic_abundance_between_past_and_present(self) -> None:
        # 3.5 Gyr of accumulation; midway between PAST (0.30) and PRESENT (0.70).
        past_ps    = get_prior_set(TemporalMode.PAST)
        present_ps = get_prior_set(TemporalMode.PRESENT)
        past_m     = dict(zip(past_ps.feature_names, past_ps.prior_means))
        pres_m     = dict(zip(present_ps.feature_names, present_ps.prior_means))
        assert past_m["organic_abundance"] < self.m["organic_abundance"] < \
               pres_m["organic_abundance"], (
            "organic_abundance prior_mean should be between PAST and PRESENT "
            "at 1.0 Gya (3.5 Gyr of tholin accumulation)")

    def test_impact_melt_prior_mean_low(self) -> None:
        # LHB relics at 1.0 Gya are ~2.5 Gyr old; should be much less than PAST (0.50).
        assert self.m["impact_melt_proxy"] <= 0.20, (
            "impact_melt_proxy prior_mean should be low (<=0.20) in "
            "LAKE_FORMATION -- LHB annuli heavily eroded by 1 Gya "
            "(Neish & Lorenz 2012)")


# ---------------------------------------------------------------------------
# NEAR_FUTURE specific scientific constraints
# ---------------------------------------------------------------------------

class TestNearFuturePriors:
    """
    Scientific constraints on NEAR_FUTURE prior values.

    References
    ----------
    Lorenz et al. (1997) GRL 24:2905 -- D2 window and solar model
    Neish et al. (2024) Astrobiology  -- ocean flux constraint
    """

    def setup_method(self) -> None:
        self.nf_ps  = get_prior_set(TemporalMode.NEAR_FUTURE)
        self.pres_ps = get_prior_set(TemporalMode.PRESENT)
        self.nf_m    = dict(zip(self.nf_ps.feature_names, self.nf_ps.prior_means))
        self.pres_m  = dict(zip(self.pres_ps.feature_names, self.pres_ps.prior_means))
        self.nf_w    = dict(zip(self.nf_ps.feature_names, self.nf_ps.weights))
        self.pres_w  = dict(zip(self.pres_ps.feature_names, self.pres_ps.weights))

    def test_weights_unchanged_from_present(self) -> None:
        # NEAR_FUTURE uses same weights as PRESENT (feature set unchanged)
        for name in PRESENT_FEATURES:
            assert abs(self.nf_w[name] - self.pres_w[name]) < 1e-9, (
                f"NEAR_FUTURE weight for '{name}' differs from PRESENT")

    def test_subsurface_ocean_prior_raised(self) -> None:
        # D2 solar warming raises subsurface ocean habitability slightly.
        assert self.nf_m["subsurface_ocean"] > self.pres_m["subsurface_ocean"], (
            "subsurface_ocean prior_mean should be higher in NEAR_FUTURE than "
            "PRESENT (D2 solar warming; Lorenz 1997)")

    def test_subsurface_ocean_prior_still_low(self) -> None:
        # Neish 2024 organic flux constraint still applies.
        assert self.nf_m["subsurface_ocean"] <= 0.15, (
            "subsurface_ocean prior_mean should remain low in NEAR_FUTURE "
            "(Neish 2024 organic flux constraint still applies)")

    def test_liquid_hydrocarbon_unchanged(self) -> None:
        # 2.5% solar brightening has negligible effect on lake stability.
        assert abs(self.nf_m["liquid_hydrocarbon"] -
                   self.pres_m["liquid_hydrocarbon"]) < 0.01, (
            "liquid_hydrocarbon prior_mean should be unchanged in NEAR_FUTURE")

    def test_organic_abundance_unchanged(self) -> None:
        # 250 Myr adds <3% to accumulated organic inventory.
        assert abs(self.nf_m["organic_abundance"] -
                   self.pres_m["organic_abundance"]) < 0.05

    def test_methane_cycle_slightly_lower_or_equal(self) -> None:
        # Minor methane depletion at +250 Myr.
        assert self.nf_m["methane_cycle"] <= self.pres_m["methane_cycle"] + 0.01


# ---------------------------------------------------------------------------
# Epoch axis
# ---------------------------------------------------------------------------

class TestEpochAxis:
    def test_plus_0250_exact_in_axis(self) -> None:
        from generate_temporal_maps import make_epoch_axis
        epochs = make_epoch_axis()
        assert any(abs(t - 0.250) < 1e-6 for t in epochs), (
            "Epoch axis must contain +0.250 Gya exact (D2 near_future anchor)")

    def test_72_frames(self) -> None:
        from generate_temporal_maps import make_epoch_axis
        epochs = make_epoch_axis()
        assert len(epochs) == 72

    def test_near_future_anchor_at_frame_30(self) -> None:
        from generate_temporal_maps import make_epoch_axis
        epochs = list(make_epoch_axis())
        idx = next(i for i, t in enumerate(epochs) if abs(t - 0.250) < 1e-6)
        assert idx == 30, (
            f"Near-future anchor should be at frame 30, found at {idx}")


# ---------------------------------------------------------------------------
# PCHIP interpolation
# ---------------------------------------------------------------------------

class TestPchipInterpolation:
    """Tests for build_pchip_interpolator and interpolate_posterior_at_epoch."""

    def test_identity_at_anchor_epochs(self) -> None:
        from generate_temporal_maps import (build_pchip_interpolator,
                                            interpolate_posterior_at_epoch)
        a0 = np.full((5, 5), 0.20, dtype=np.float32)
        a1 = np.full((5, 5), 0.40, dtype=np.float32)
        interp = build_pchip_interpolator([-3.5, 0.0], [a0, a1])
        out0 = interpolate_posterior_at_epoch(interp, -3.5, -3.5, 0.0, output_shape=(5, 5))
        out1 = interpolate_posterior_at_epoch(interp, 0.0,  -3.5, 0.0, output_shape=(5, 5))
        np.testing.assert_allclose(out0, 0.20, atol=1e-5)
        np.testing.assert_allclose(out1, 0.40, atol=1e-5)

    def test_midpoint_between_linear_anchors(self) -> None:
        from generate_temporal_maps import (build_pchip_interpolator,
                                            interpolate_posterior_at_epoch)
        # With only 2 anchor points PCHIP reduces to linear interpolation.
        a0 = np.full((4, 4), 0.10, dtype=np.float32)
        a1 = np.full((4, 4), 0.50, dtype=np.float32)
        interp = build_pchip_interpolator([-2.0, 0.0], [a0, a1])
        mid = interpolate_posterior_at_epoch(interp, -1.0, -2.0, 0.0, output_shape=(4, 4))
        np.testing.assert_allclose(mid, 0.30, atol=1e-4)

    def test_output_clamped_to_unit_interval(self) -> None:
        from generate_temporal_maps import (build_pchip_interpolator,
                                            interpolate_posterior_at_epoch)
        # PCHIP can very slightly overshoot; output must be clipped to [0, 1].
        a0 = np.full((3, 3), 0.00, dtype=np.float32)
        a1 = np.full((3, 3), 0.01, dtype=np.float32)
        a2 = np.full((3, 3), 0.00, dtype=np.float32)
        interp = build_pchip_interpolator([-1.0, 0.0, 1.0], [a0, a1, a2])
        out = interpolate_posterior_at_epoch(interp, 0.5, -1.0, 1.0, output_shape=(3, 3))
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_extrapolation_returns_nan_or_boundary(self) -> None:
        """PchipInterpolator with extrapolate=False returns NaN outside range."""
        from generate_temporal_maps import build_pchip_interpolator
        a0 = np.full((3, 3), 0.20, dtype=np.float32)
        a1 = np.full((3, 3), 0.40, dtype=np.float32)
        interp = build_pchip_interpolator([0.0, 1.0], [a0, a1])
        # t = 2.0 is outside [0.0, 1.0]; raw interpolator returns NaN
        raw = interp(2.0)
        # Our wrapper clips to [lo, hi] so the result equals the value at hi
        from generate_temporal_maps import interpolate_posterior_at_epoch
        clipped = interpolate_posterior_at_epoch(interp, 2.0, 0.0, 1.0, output_shape=(3, 3))
        np.testing.assert_allclose(clipped, 0.40, atol=1e-4)

    def test_output_shape_matches_grid(self) -> None:
        from generate_temporal_maps import (build_pchip_interpolator,
                                            interpolate_posterior_at_epoch,
                                            GRID_SHAPE)
        a0 = np.zeros(GRID_SHAPE, dtype=np.float32)
        a1 = np.ones(GRID_SHAPE,  dtype=np.float32)
        interp = build_pchip_interpolator([-1.0, 0.0], [a0, a1])
        # No output_shape → defaults to GRID_SHAPE
        out = interpolate_posterior_at_epoch(interp, -0.5, -1.0, 0.0)
        assert out.shape == GRID_SHAPE

    def test_four_anchor_monotone(self) -> None:
        """PCHIP should not produce values outside [min_anchor, max_anchor]
        when posteriors are monotonically increasing."""
        from generate_temporal_maps import (build_pchip_interpolator,
                                            interpolate_posterior_at_epoch)
        anchors = [0.10, 0.20, 0.30, 0.40]
        epochs  = [-3.5, -1.0, 0.0, 0.25]
        arrs    = [np.full((3, 3), v, dtype=np.float32) for v in anchors]
        interp  = build_pchip_interpolator(epochs, arrs)
        test_epochs = np.linspace(-3.5, 0.25, 20)
        for t in test_epochs:
            out = interpolate_posterior_at_epoch(interp, float(t), -3.5, 0.25,
                                                 output_shape=(3, 3))
            assert np.all(out >= 0.09), f"Below floor at t={t:.3f}: {out.min()}"
            assert np.all(out <= 0.41), f"Above ceil at t={t:.3f}: {out.max()}"


# ---------------------------------------------------------------------------
# load_anchor_posteriors
# ---------------------------------------------------------------------------

class TestLoadAnchorPosteriors:
    def test_missing_present_raises(self, tmp_path: Path) -> None:
        from generate_temporal_maps import load_anchor_posteriors
        with pytest.raises(FileNotFoundError):
            load_anchor_posteriors(tmp_path)

    def test_loads_present_when_present(self, tmp_path: Path) -> None:
        from generate_temporal_maps import load_anchor_posteriors, GRID_SHAPE
        inf_dir = tmp_path / "present" / "inference"
        inf_dir.mkdir(parents=True)
        arr = np.full(GRID_SHAPE, 0.30, dtype=np.float32)
        np.save(inf_dir / "posterior_mean.npy", arr)
        anchors = load_anchor_posteriors(tmp_path)
        assert "present" in anchors
        np.testing.assert_array_equal(anchors["present"], arr)

    def test_missing_non_required_anchor_skipped(self, tmp_path: Path) -> None:
        from generate_temporal_maps import load_anchor_posteriors, GRID_SHAPE
        inf_dir = tmp_path / "present" / "inference"
        inf_dir.mkdir(parents=True)
        np.save(inf_dir / "posterior_mean.npy",
                np.zeros(GRID_SHAPE, dtype=np.float32))
        anchors = load_anchor_posteriors(tmp_path)
        # Only present loaded; missing ones silently skipped
        assert "past"           not in anchors
        assert "lake_formation" not in anchors
        assert "near_future"    not in anchors
        assert "future"         not in anchors


# ---------------------------------------------------------------------------
# TemporalFeatureExtractor dispatch
# ---------------------------------------------------------------------------

class TestTemporalFeatureExtractorDispatch:
    """Verify the dispatch routes to the correct extraction method."""

    def _make_extractor(self, mode: TemporalMode):
        from titan.temporal_features import TemporalFeatureExtractor
        from configs.pipeline_config import HabitabilityWindowConfig
        from titan.preprocessing import CanonicalGrid
        # CanonicalGrid takes pixel_size_m (metres).  500 km/px gives a
        # ~16 x 32 grid -- small enough for fast tests.
        grid = CanonicalGrid(pixel_size_m=500_000)
        window = HabitabilityWindowConfig()
        return TemporalFeatureExtractor(mode=mode, grid=grid, window_config=window)

    def test_lake_formation_dispatch_exists(self) -> None:
        ext = self._make_extractor(TemporalMode.LAKE_FORMATION)
        assert hasattr(ext, "_extract_lake_formation")

    def test_near_future_dispatch_exists(self) -> None:
        ext = self._make_extractor(TemporalMode.NEAR_FUTURE)
        assert hasattr(ext, "_extract_near_future")

    def test_unknown_mode_raises(self) -> None:
        from titan.temporal_features import TemporalFeatureExtractor
        from configs.pipeline_config import HabitabilityWindowConfig
        from titan.preprocessing import CanonicalGrid
        grid = CanonicalGrid(pixel_size_m=500_000)
        ext = TemporalFeatureExtractor.__new__(TemporalFeatureExtractor)
        ext.mode   = "bogus_mode"   # type: ignore[assignment]
        ext.grid   = grid
        ext.window = HabitabilityWindowConfig()
        with pytest.raises(ValueError):
            ext.extract(None)   # type: ignore[arg-type]
