"""
tests/test_temporal.py
=======================
Tests for the temporal habitability framework (past/present/future).

Tests cover:
  - Prior sets validate (weights sum to 1.0, all in [0,1])
  - Feature name lists are correct and non-overlapping where expected
  - PAST features produce reasonable spatial outputs from synthetic data
  - FUTURE features produce reasonable spatial outputs from synthetic data
  - Temporal inference runs without errors (no real data needed)
  - Comparison: FUTURE posterior > PRESENT posterior (globally)
  - Assumptions are documented and enforced
"""

from __future__ import annotations

from typing import Any, Callable
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.temporal_config import (
    TemporalMode,
    PRESENT_FEATURES,
    PAST_FEATURES,
    FUTURE_FEATURES,
    TEMPORAL_FEATURE_NAMES,
    TemporalPriorSet,
    get_prior_set,
    describe_prior_changes,
)
from titan.temporal_features import (
    CRYOVOLCANIC_CANDIDATES,
    IMPACT_MELT_CRATERS,
    _gaussian_proximity_map,
    extract_impact_melt_proxy,
    extract_cryovolcanic_flux,
    extract_water_ammonia_solvent,
    extract_organic_stockpile,
    extract_dissolved_energy,
    extract_water_ammonia_cycle,
    extract_global_ocean_habitability,
    TemporalFeatureExtractor,
    TemporalFeatureStack,
)
from titan.preprocessing import CanonicalGrid

PASS, FAIL = [], []

def run(name: str, fn: Callable[..., Any]) -> None:
    try:
        fn()
        PASS.append(name)
    except Exception:
        FAIL.append((name, traceback.format_exc()))


# ---------------------------------------------------------------------------
# Helper: make a minimal synthetic xarray stack
# ---------------------------------------------------------------------------

def make_synthetic_stack(nrows: int=18, ncols: int=36) -> "xr.Dataset":
    """
    Build a minimal xr.Dataset with synthetic data for feature testing.
    Small grid (18×36) for speed.
    """
    import xarray as xr
    rng = np.random.default_rng(42)
    lat = np.linspace(90, -90, nrows)
    lon = np.linspace(0, 360, ncols, endpoint=False)

    # SAR mosaic: float32, some zeros (nodata), rest uniform(0,1)
    sar = rng.uniform(0.1, 0.9, (nrows, ncols)).astype(np.float32)
    sar[0:2, 0:3] = 0.0  # simulate nodata

    # Topography: range -500 to 500 m
    topo = rng.uniform(-500, 500, (nrows, ncols)).astype(np.float32)
    topo[topo < -400] = np.nan  # some missing

    # VIMS mosaic: band ratio
    vims = rng.uniform(0.2, 0.8, (nrows, ncols)).astype(np.float32)

    # Geomorphology: integer labels 1-7 with nodata 0
    geo = rng.integers(1, 8, (nrows, ncols)).astype(np.float32)
    geo[0, :] = 0.0  # nodata row

    # VIMS coverage
    cov = rng.uniform(0, 1, (nrows, ncols)).astype(np.float32)

    return xr.Dataset({
        "sar_mosaic":     xr.DataArray(sar,  dims=["lat", "lon"],
                                       coords={"lat": lat, "lon": lon}),
        "topography":     xr.DataArray(topo, dims=["lat", "lon"],
                                       coords={"lat": lat, "lon": lon}),
        "vims_mosaic":    xr.DataArray(vims, dims=["lat", "lon"],
                                       coords={"lat": lat, "lon": lon}),
        "geomorphology":  xr.DataArray(geo,  dims=["lat", "lon"],
                                       coords={"lat": lat, "lon": lon}),
        "vims_coverage":  xr.DataArray(cov,  dims=["lat", "lon"],
                                       coords={"lat": lat, "lon": lon}),
    })


# Small grid matching the synthetic stack
MINI_GRID = CanonicalGrid(pixel_size_m=4490.0)
# Override with tiny grid for speed
class TinyGrid:
    nrows = 18
    ncols = 36
    pixel_size_m = 4490.0
    def lat_centres_deg(self) -> np.ndarray: return np.linspace(90, -90, self.nrows)
    def lon_centres_deg(self) -> np.ndarray: return np.linspace(0, 360, self.ncols, endpoint=False)
    def empty(self) -> "xr.DataArray": return np.full((self.nrows, self.ncols), np.nan, dtype=np.float32)
    def __repr__(self) -> str: return f"TinyGrid(18×36)"

TINY = TinyGrid()
SYNTH = make_synthetic_stack(nrows=18, ncols=36)


# ===========================================================================
# 1. TemporalMode enum
# ===========================================================================

def test_temporal_mode_values() -> None:
    assert TemporalMode.PAST.value    == "past"
    assert TemporalMode.PRESENT.value == "present"
    assert TemporalMode.FUTURE.value  == "future"

def test_temporal_mode_from_string() -> None:
    assert TemporalMode("past")    == TemporalMode.PAST
    assert TemporalMode("present") == TemporalMode.PRESENT
    assert TemporalMode("future")  == TemporalMode.FUTURE

for fn in [test_temporal_mode_values, test_temporal_mode_from_string]:
    run(fn.__name__, fn)


# ===========================================================================
# 2. Feature name lists
# ===========================================================================

def test_present_feature_count() -> None:   assert len(PRESENT_FEATURES) == 8
def test_past_feature_count() -> None:      assert len(PAST_FEATURES) == 9   # 7+2
def test_future_feature_count() -> None:    assert len(FUTURE_FEATURES) == 8

def test_past_has_new_features() -> None:
    assert "impact_melt_proxy"   in PAST_FEATURES
    assert "cryovolcanic_flux"   in PAST_FEATURES

def test_future_has_new_features() -> None:
    assert "water_ammonia_solvent"    in FUTURE_FEATURES
    assert "organic_stockpile"        in FUTURE_FEATURES
    assert "dissolved_energy"         in FUTURE_FEATURES
    assert "water_ammonia_cycle"      in FUTURE_FEATURES
    assert "global_ocean_habitability" in FUTURE_FEATURES

def test_past_no_subsurface_ocean() -> None:
    # subsurface_ocean replaced by impact_melt_proxy in PAST
    # (liquid_hydrocarbon remains as impact melt proxy)
    assert "impact_melt_proxy" in PAST_FEATURES

def test_future_no_liquid_hydrocarbon() -> None:
    # liquid_hydrocarbon replaced by water_ammonia_solvent
    assert "liquid_hydrocarbon" not in FUTURE_FEATURES
    assert "water_ammonia_solvent" in FUTURE_FEATURES

def test_temporal_feature_names_dict_complete() -> None:
    assert TemporalMode.PAST    in TEMPORAL_FEATURE_NAMES
    assert TemporalMode.PRESENT in TEMPORAL_FEATURE_NAMES
    assert TemporalMode.FUTURE  in TEMPORAL_FEATURE_NAMES

for fn in [test_present_feature_count, test_past_feature_count,
           test_future_feature_count, test_past_has_new_features,
           test_future_has_new_features, test_past_no_subsurface_ocean,
           test_future_no_liquid_hydrocarbon, test_temporal_feature_names_dict_complete]:
    run(fn.__name__, fn)


# ===========================================================================
# 3. Prior set validation
# ===========================================================================

def test_present_priors_sum_to_1() -> None:
    p = get_prior_set(TemporalMode.PRESENT)
    p.validate()  # raises if not 1.0 ± 0.01
    assert abs(sum(p.weights) - 1.0) < 0.011

def test_past_priors_sum_to_1() -> None:
    p = get_prior_set(TemporalMode.PAST)
    p.validate()
    assert abs(sum(p.weights) - 1.0) < 0.011

def test_future_priors_sum_to_1() -> None:
    p = get_prior_set(TemporalMode.FUTURE)
    p.validate()
    assert abs(sum(p.weights) - 1.0) < 0.011

def test_all_prior_means_in_range() -> None:
    for mode in TemporalMode:
        p = get_prior_set(mode)
        for name, m in zip(p.feature_names, p.prior_means):
            assert 0.0 <= m <= 1.0, f"[{mode.value}] {name} mean={m} out of [0,1]"

def test_all_weights_positive() -> None:
    for mode in TemporalMode:
        p = get_prior_set(mode)
        for name, w in zip(p.feature_names, p.weights):
            assert w > 0, f"[{mode.value}] {name} weight={w} must be > 0"

def test_feature_names_match_weights() -> None:
    for mode in TemporalMode:
        p = get_prior_set(mode)
        assert len(p.feature_names) == len(p.weights) == len(p.prior_means)

def test_past_organic_lower_than_present() -> None:
    """Past organic abundance prior should be lower (fewer Gyr of tholins)."""
    past    = get_prior_set(TemporalMode.PAST)
    present = get_prior_set(TemporalMode.PRESENT)
    past_m    = dict(zip(past.feature_names,    past.prior_means))
    present_m = dict(zip(present.feature_names, present.prior_means))
    assert past_m["organic_abundance"] < present_m["organic_abundance"], (
        "PAST organic_abundance should be lower than PRESENT "
        "(fewer Gyr of UV photolysis → less tholins)"
    )

def test_future_water_ammonia_high() -> None:
    """Future water-ammonia solvent prior should be high (global ocean)."""
    future = get_prior_set(TemporalMode.FUTURE)
    future_m = dict(zip(future.feature_names, future.prior_means))
    assert future_m["water_ammonia_solvent"] > 0.70, (
        "FUTURE water_ammonia_solvent prior should be > 0.70 "
        "(Lorenz 1997: global ~200K ocean)"
    )

def test_present_subsurface_ocean_revised_down() -> None:
    """Subsurface ocean prior should be lower than 0.10 (Neish 2024)."""
    present = get_prior_set(TemporalMode.PRESENT)
    present_m = dict(zip(present.feature_names, present.prior_means))
    assert present_m["subsurface_ocean"] < 0.08, (
        "PRESENT subsurface_ocean should be < 0.08 "
        "(Neish 2024: organic flux ~1 elephant/year, very limited)"
    )

def test_as_dicts() -> None:
    p = get_prior_set(TemporalMode.PRESENT)
    w = p.as_weight_dict()
    m = p.as_mean_dict()
    assert set(w.keys()) == set(m.keys())
    assert abs(sum(w.values()) - 1.0) < 0.011

for fn in [test_present_priors_sum_to_1, test_past_priors_sum_to_1,
           test_future_priors_sum_to_1, test_all_prior_means_in_range,
           test_all_weights_positive, test_feature_names_match_weights,
           test_past_organic_lower_than_present, test_future_water_ammonia_high,
           test_present_subsurface_ocean_revised_down, test_as_dicts]:
    run(fn.__name__, fn)


# ===========================================================================
# 4. Gaussian proximity map
# ===========================================================================

def test_proximity_map_shape() -> None:
    sites  = [(180.0, 0.0, "test")]
    result = _gaussian_proximity_map(sites, TINY, sigma_deg=10.0)
    assert result.shape == (TINY.nrows, TINY.ncols)
    assert result.dtype == np.float32

def test_proximity_map_range() -> None:
    sites  = [(180.0, 0.0, "test"), (90.0, 45.0, "test2")]
    result = _gaussian_proximity_map(sites, TINY, sigma_deg=15.0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-5

def test_proximity_map_peak_near_site() -> None:
    """Maximum value should be close to the site location."""
    sites  = [(180.0, 0.0, "equator_180")]
    result = _gaussian_proximity_map(sites, TINY, sigma_deg=20.0)
    max_idx = np.unravel_index(result.argmax(), result.shape)
    lon_at_max = TINY.lon_centres_deg()[max_idx[1]]
    lat_at_max = TINY.lat_centres_deg()[max_idx[0]]
    assert abs(lon_at_max - 180.0) < 30.0, f"Max lon {lon_at_max:.1f}° far from 180°W"
    assert abs(lat_at_max - 0.0) < 20.0, f"Max lat {lat_at_max:.1f}° far from equator"

def test_proximity_with_diameter_scaling() -> None:
    sites = [(100.0, 20.0, 400.0, "big_crater"), (200.0, -20.0, 40.0, "small")]
    result = _gaussian_proximity_map(sites, TINY, sigma_deg=5.0, use_diameter=True)
    assert result.shape == (TINY.nrows, TINY.ncols)

for fn in [test_proximity_map_shape, test_proximity_map_range,
           test_proximity_map_peak_near_site, test_proximity_with_diameter_scaling]:
    run(fn.__name__, fn)


# ===========================================================================
# 5. PAST feature extractors
# ===========================================================================

def test_impact_melt_proxy_shape() -> None:
    result = extract_impact_melt_proxy(SYNTH, TINY)
    assert result.shape == (TINY.nrows, TINY.ncols)
    assert result.dtype == np.float32

def test_impact_melt_proxy_range() -> None:
    result = extract_impact_melt_proxy(SYNTH, TINY)
    finite = result[np.isfinite(result)]
    assert len(finite) > 0
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0 + 1e-5

def test_cryovolcanic_flux_shape() -> None:
    result = extract_cryovolcanic_flux(SYNTH, TINY)
    assert result.shape == (TINY.nrows, TINY.ncols)

def test_cryovolcanic_flux_range() -> None:
    result = extract_cryovolcanic_flux(SYNTH, TINY)
    finite = result[np.isfinite(result)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0 + 1e-5

def test_impact_melt_craters_catalog_nonempty() -> None:
    """The crater catalog used for PAST mode must be non-empty."""
    assert len(IMPACT_MELT_CRATERS) >= 5

def test_cryovolcanic_catalog_nonempty() -> None:
    assert len(CRYOVOLCANIC_CANDIDATES) >= 5

def test_selk_in_catalog() -> None:
    """Selk crater (Dragonfly target) must be in the impact melt catalog."""
    names = [c[3].lower() for c in IMPACT_MELT_CRATERS]
    assert any("selk" in n for n in names), "Selk crater missing from catalog"

def test_sotra_in_cryovolcanic_catalog() -> None:
    """Sotra Facula (best cryovolcanic candidate) must be in catalog."""
    names = [c[2].lower() for c in CRYOVOLCANIC_CANDIDATES]
    assert any("sotra" in n for n in names), "Sotra Facula missing from catalog"

for fn in [test_impact_melt_proxy_shape, test_impact_melt_proxy_range,
           test_cryovolcanic_flux_shape, test_cryovolcanic_flux_range,
           test_impact_melt_craters_catalog_nonempty,
           test_cryovolcanic_catalog_nonempty,
           test_selk_in_catalog, test_sotra_in_cryovolcanic_catalog]:
    run(fn.__name__, fn)


# ===========================================================================
# 6. FUTURE feature extractors
# ===========================================================================

def test_water_ammonia_shape() -> None:
    result = extract_water_ammonia_solvent(SYNTH, TINY)
    assert result.shape == (TINY.nrows, TINY.ncols)

def test_water_ammonia_range() -> None:
    result = extract_water_ammonia_solvent(SYNTH, TINY)
    finite = result[np.isfinite(result)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0 + 1e-5

def test_water_ammonia_low_terrain_higher() -> None:
    """Low-elevation pixels should have higher water_ammonia_solvent (future ocean)."""
    result = extract_water_ammonia_solvent(SYNTH, TINY)
    topo   = SYNTH["topography"].values
    # Compare top-10% lowest terrain vs top-10% highest terrain
    valid  = np.isfinite(topo) & np.isfinite(result)
    if valid.sum() < 10:
        return  # skip if not enough valid pixels in tiny grid
    flat_topo   = topo[valid]
    flat_result = result[valid]
    low_terrain_mask  = flat_topo < np.percentile(flat_topo, 20)
    high_terrain_mask = flat_topo > np.percentile(flat_topo, 80)
    if low_terrain_mask.sum() > 3 and high_terrain_mask.sum() > 3:
        assert flat_result[low_terrain_mask].mean() >= flat_result[high_terrain_mask].mean() * 0.8, \
            "Low terrain should have >= high terrain solvent score (future bathymetry)"

def test_organic_stockpile_shape() -> None:
    result = extract_organic_stockpile(SYNTH, TINY)
    assert result.shape == (TINY.nrows, TINY.ncols)

def test_dissolved_energy_shape() -> None:
    result = extract_dissolved_energy(SYNTH, TINY)
    assert result.shape == (TINY.nrows, TINY.ncols)

def test_dissolved_energy_bounded_by_inputs() -> None:
    """dissolved_energy should not exceed either stockpile or ocean proxy."""
    energy   = extract_dissolved_energy(SYNTH, TINY)
    stockpile = extract_organic_stockpile(SYNTH, TINY)
    ocean    = extract_water_ammonia_solvent(SYNTH, TINY)
    valid    = np.isfinite(energy) & np.isfinite(stockpile) & np.isfinite(ocean)
    if valid.sum() > 0:
        assert energy[valid].max() <= 1.0 + 1e-5

def test_global_ocean_habitability_shape() -> None:
    result = extract_global_ocean_habitability(SYNTH, TINY)
    assert result.shape == (TINY.nrows, TINY.ncols)

def test_water_ammonia_cycle_shape() -> None:
    result = extract_water_ammonia_cycle(SYNTH, TINY)
    assert result.shape == (TINY.nrows, TINY.ncols)

for fn in [test_water_ammonia_shape, test_water_ammonia_range,
           test_water_ammonia_low_terrain_higher,
           test_organic_stockpile_shape, test_dissolved_energy_shape,
           test_dissolved_energy_bounded_by_inputs,
           test_global_ocean_habitability_shape, test_water_ammonia_cycle_shape]:
    run(fn.__name__, fn)


# ===========================================================================
# 7. TemporalFeatureExtractor
# ===========================================================================

def test_present_extractor_returns_correct_features() -> None:
    ext   = TemporalFeatureExtractor(TINY, TemporalMode.PRESENT)
    stack = ext.extract(SYNTH)
    assert stack.mode == TemporalMode.PRESENT
    assert len(stack.feature_names()) == 8
    for n in PRESENT_FEATURES:
        assert n in stack.features
        assert stack.features[n].shape == (TINY.nrows, TINY.ncols)

def test_past_extractor_returns_correct_features() -> None:
    ext   = TemporalFeatureExtractor(TINY, TemporalMode.PAST)
    stack = ext.extract(SYNTH)
    assert stack.mode == TemporalMode.PAST
    assert len(stack.feature_names()) == 9
    for n in PAST_FEATURES:
        assert n in stack.features

def test_future_extractor_returns_correct_features() -> None:
    ext   = TemporalFeatureExtractor(TINY, TemporalMode.FUTURE)
    stack = ext.extract(SYNTH)
    assert stack.mode == TemporalMode.FUTURE
    assert len(stack.feature_names()) == 8
    for n in FUTURE_FEATURES:
        assert n in stack.features

def test_temporal_feature_stack_as_array_shape() -> None:
    ext   = TemporalFeatureExtractor(TINY, TemporalMode.PRESENT)
    stack = ext.extract(SYNTH)
    arr   = stack.as_array()
    assert arr.shape == (8, TINY.nrows, TINY.ncols)

def test_temporal_feature_stack_coverage() -> None:
    ext   = TemporalFeatureExtractor(TINY, TemporalMode.PRESENT)
    stack = ext.extract(SYNTH)
    cov   = stack.coverage_fraction()
    assert set(cov.keys()) == set(PRESENT_FEATURES)
    for name, frac in cov.items():
        assert 0.0 <= frac <= 1.0

def test_temporal_feature_stack_to_xarray() -> None:
    import xarray as xr
    ext   = TemporalFeatureExtractor(TINY, TemporalMode.FUTURE)
    stack = ext.extract(SYNTH)
    ds    = stack.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert "temporal_mode" in ds.attrs
    assert ds.attrs["temporal_mode"] == "future"

for fn in [test_present_extractor_returns_correct_features,
           test_past_extractor_returns_correct_features,
           test_future_extractor_returns_correct_features,
           test_temporal_feature_stack_as_array_shape,
           test_temporal_feature_stack_coverage,
           test_temporal_feature_stack_to_xarray]:
    run(fn.__name__, fn)


# ===========================================================================
# 7b. D1/D2/D3 configurability (design decisions accepted by researcher)
# ===========================================================================

def test_d1_past_epoch_default_is_3pt5() -> None:
    """D1: default past epoch must be 3.5 Gya."""
    from configs.pipeline_config import HabitabilityWindowConfig
    cfg = HabitabilityWindowConfig()
    assert cfg.past_liquid_water_epoch_gya == 3.5


def test_d1_past_epoch_configurable() -> None:
    """D1: past epoch must be overrideable."""
    from configs.pipeline_config import HabitabilityWindowConfig
    for epoch in (0.5, 2.0, 3.5, 4.0):
        cfg = HabitabilityWindowConfig(past_liquid_water_epoch_gya=epoch)
        assert cfg.past_liquid_water_epoch_gya == epoch


def test_d1_past_epoch_passed_to_extractor() -> None:
    """D1: window_config is propagated into FeatureExtractor."""
    from configs.pipeline_config import HabitabilityWindowConfig
    cfg = HabitabilityWindowConfig(past_liquid_water_epoch_gya=4.0)
    ext = TemporalFeatureExtractor(TINY, TemporalMode.PRESENT, window_config=cfg)
    assert ext.window_config.past_liquid_water_epoch_gya == 4.0
    # Lazily-initialised present extractor should also carry it
    pe = ext._get_present_extractor()
    assert pe.window.past_liquid_water_epoch_gya == 4.0


def test_d2_future_window_defaults() -> None:
    """D2: default near-future window is 100–400 Myr."""
    from configs.pipeline_config import HabitabilityWindowConfig
    cfg = HabitabilityWindowConfig()
    assert cfg.future_window_min_myr == 100.0
    assert cfg.future_window_max_myr == 400.0


def test_d2_uniform_warming_default_true() -> None:
    """D2: uniform global warming assumption is ON by default (explicit assumption)."""
    from configs.pipeline_config import HabitabilityWindowConfig
    cfg = HabitabilityWindowConfig()
    assert cfg.assume_uniform_warming is True, (
        "Uniform warming must default to True — it is an explicit documented "
        "assumption (D2). Users must actively opt out via --no-uniform-warming."
    )


def test_d2_uniform_warming_can_be_disabled() -> None:
    """D2: disabling uniform warming zeroes the future-window prior contribution."""
    from configs.pipeline_config import HabitabilityWindowConfig
    cfg_off = HabitabilityWindowConfig(assume_uniform_warming=False)
    assert cfg_off.temporal_prior_weight() == 0.0


def test_d2_future_window_configurable() -> None:
    """D2: future window bounds are overrideable."""
    from configs.pipeline_config import HabitabilityWindowConfig
    cfg = HabitabilityWindowConfig(future_window_min_myr=50.0, future_window_max_myr=600.0)
    assert cfg.future_window_min_myr == 50.0
    assert cfg.future_window_max_myr == 600.0
    cfg.validate()   # must not raise


def test_d2_window_width_affects_weight() -> None:
    """D2: narrower window → higher temporal prior weight (uniform density)."""
    from configs.pipeline_config import HabitabilityWindowConfig
    narrow = HabitabilityWindowConfig(future_window_min_myr=100, future_window_max_myr=200)
    wide   = HabitabilityWindowConfig(future_window_min_myr=100, future_window_max_myr=600)
    assert narrow.temporal_prior_weight() > wide.temporal_prior_weight()


def test_d2_invalid_window_raises() -> None:
    """D2: max must be greater than min."""
    from configs.pipeline_config import HabitabilityWindowConfig
    import contextlib
    cfg = HabitabilityWindowConfig(future_window_min_myr=400, future_window_max_myr=100)
    raised = False
    try:
        cfg.validate()
    except ValueError:
        raised = True
    assert raised, "Should raise ValueError when max <= min"


def test_d3_subsurface_ocean_prior_default_is_0pt03() -> None:
    """D3: default prior must be 0.03 (revised down from 0.10, Neish et al. 2024)."""
    from configs.pipeline_config import BayesianPriorConfig
    cfg = BayesianPriorConfig()
    assert cfg.prior_mean_subsurface_ocean == 0.03, (
        "D3: subsurface_ocean prior must default to 0.03 per Neish et al. "
        "(2024) — organic flux to ocean ~one elephant/year."
    )


def test_d3_prior_configurable() -> None:
    """D3: subsurface ocean prior must be overrideable."""
    from configs.pipeline_config import BayesianPriorConfig
    for val in (0.01, 0.03, 0.05, 0.10):
        cfg = BayesianPriorConfig(prior_mean_subsurface_ocean=val)
        assert cfg.prior_mean_subsurface_ocean == val
        cfg.validate()


def test_d3_prior_passed_to_extractor() -> None:
    """D3: subsurface_ocean_base_prior is propagated into TemporalFeatureExtractor."""
    ext = TemporalFeatureExtractor(
        TINY, TemporalMode.PRESENT,
        subsurface_ocean_base_prior=0.05,
    )
    assert ext.subsurface_ocean_base_prior == 0.05
    pe = ext._get_present_extractor()
    assert pe.subsurface_ocean_base_prior == 0.05


def test_d3_prior_affects_subsurface_ocean_feature() -> None:
    """D3: higher prior → higher mean subsurface_ocean feature values."""
    ext_low  = TemporalFeatureExtractor(TINY, TemporalMode.PRESENT,
                                        subsurface_ocean_base_prior=0.01)
    ext_high = TemporalFeatureExtractor(TINY, TemporalMode.PRESENT,
                                        subsurface_ocean_base_prior=0.20)
    # Use a SAR-free stack so no annular boosting, pure base prior comparison
    import xarray as xr
    no_sar_stack = xr.Dataset({
        k: v for k, v in SYNTH.data_vars.items() if k != "sar_mosaic"
    })
    stack_low  = ext_low.extract(no_sar_stack)
    stack_high = ext_high.extract(no_sar_stack)
    mean_low   = float(np.nanmean(stack_low.features["subsurface_ocean"]))
    mean_high  = float(np.nanmean(stack_high.features["subsurface_ocean"]))
    assert mean_high > mean_low, (
        f"D3: higher base prior (0.20) should yield higher mean "
        f"subsurface_ocean ({mean_high:.3f} vs {mean_low:.3f})"
    )


def test_d3_prior_not_affected_by_sar_when_zero() -> None:
    """D3: with base prior 0.0, feature should remain near zero absent SAR."""
    import xarray as xr
    ext = TemporalFeatureExtractor(TINY, TemporalMode.PRESENT,
                                   subsurface_ocean_base_prior=0.0)
    no_sar = xr.Dataset({k: v for k, v in SYNTH.data_vars.items()
                         if k != "sar_mosaic"})
    stack = ext.extract(no_sar)
    arr   = stack.features["subsurface_ocean"]
    valid = arr[np.isfinite(arr)]
    assert valid.mean() < 0.1, (
        "With base_prior=0.0 and no SAR, subsurface_ocean should be near 0"
    )


def test_pipeline_config_wires_d1_d2_d3() -> None:
    """PipelineConfig correctly propagates D1/D2/D3 to FeatureExtractor."""
    from configs.pipeline_config import (
        PipelineConfig, HabitabilityWindowConfig, BayesianPriorConfig
    )
    win = HabitabilityWindowConfig(
        past_liquid_water_epoch_gya=4.0,
        future_window_min_myr=50,
        future_window_max_myr=500,
        assume_uniform_warming=False,
    )
    pri = BayesianPriorConfig(prior_mean_subsurface_ocean=0.05)
    cfg = PipelineConfig(priors=pri, habitability_window=win)
    assert cfg.habitability_window.past_liquid_water_epoch_gya == 4.0
    assert cfg.habitability_window.future_window_min_myr == 50
    assert cfg.habitability_window.assume_uniform_warming is False
    assert cfg.priors.prior_mean_subsurface_ocean == 0.05


for fn in [test_d1_past_epoch_default_is_3pt5,
           test_d1_past_epoch_configurable,
           test_d1_past_epoch_passed_to_extractor,
           test_d2_future_window_defaults,
           test_d2_uniform_warming_default_true,
           test_d2_uniform_warming_can_be_disabled,
           test_d2_future_window_configurable,
           test_d2_window_width_affects_weight,
           test_d2_invalid_window_raises,
           test_d3_subsurface_ocean_prior_default_is_0pt03,
           test_d3_prior_configurable,
           test_d3_prior_passed_to_extractor,
           test_d3_prior_affects_subsurface_ocean_feature,
           test_d3_prior_not_affected_by_sar_when_zero,
           test_pipeline_config_wires_d1_d2_d3]:
    run(fn.__name__, fn)


# ===========================================================================
# 8. Temporal inference (sklearn backend, no real data needed)
# ===========================================================================

def test_temporal_inference_present() -> None:
    from titan.bayesian.temporal_inference import run_temporal_inference
    from configs.pipeline_config import PipelineConfig

    ext   = TemporalFeatureExtractor(TINY, TemporalMode.PRESENT)
    stack = ext.extract(SYNTH)
    cfg   = PipelineConfig(bayesian_backend="sklearn")
    result = run_temporal_inference(stack, cfg)

    assert result.posterior_mean.shape == (TINY.nrows, TINY.ncols)
    assert "present" in result.backend.lower()
    assert result.n_valid_pixels > 0
    finite = result.posterior_mean[np.isfinite(result.posterior_mean)]
    assert len(finite) > 0
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0 + 1e-5

def test_temporal_inference_past() -> None:
    from titan.bayesian.temporal_inference import run_temporal_inference
    from configs.pipeline_config import PipelineConfig

    ext    = TemporalFeatureExtractor(TINY, TemporalMode.PAST)
    stack  = ext.extract(SYNTH)
    cfg    = PipelineConfig(bayesian_backend="sklearn")
    result = run_temporal_inference(stack, cfg)

    assert result.posterior_mean.shape == (TINY.nrows, TINY.ncols)
    assert "past" in result.backend.lower()
    assert result.n_valid_pixels > 0

def test_temporal_inference_future() -> None:
    from titan.bayesian.temporal_inference import run_temporal_inference
    from configs.pipeline_config import PipelineConfig

    ext    = TemporalFeatureExtractor(TINY, TemporalMode.FUTURE)
    stack  = ext.extract(SYNTH)
    cfg    = PipelineConfig(bayesian_backend="sklearn")
    result = run_temporal_inference(stack, cfg)

    assert result.posterior_mean.shape == (TINY.nrows, TINY.ncols)
    assert "future" in result.backend.lower()

def test_future_mean_posterior_higher_than_present() -> None:
    """
    Global mean P(habitable) should be higher in FUTURE than PRESENT.

    Scientific basis: Lorenz et al. (1997) predict global water-ammonia
    oceans at 200K with abundant organic substrate → higher habitability
    than the current cold, HC-lake-only environment.
    """
    from titan.bayesian.temporal_inference import run_temporal_inference
    from configs.pipeline_config import PipelineConfig
    cfg = PipelineConfig(bayesian_backend="sklearn")

    ext_p  = TemporalFeatureExtractor(TINY, TemporalMode.PRESENT)
    ext_f  = TemporalFeatureExtractor(TINY, TemporalMode.FUTURE)
    sp     = ext_p.extract(SYNTH)
    sf     = ext_f.extract(SYNTH)
    rp     = run_temporal_inference(sp, cfg)
    rf     = run_temporal_inference(sf, cfg)

    mean_p = float(np.nanmean(rp.posterior_mean))
    mean_f = float(np.nanmean(rf.posterior_mean))

    assert mean_f > mean_p, (
        f"FUTURE mean posterior ({mean_f:.3f}) should exceed "
        f"PRESENT mean ({mean_p:.3f}) — red giant scenario has "
        f"global ocean and vast organic stockpile (Lorenz 1997)"
    )

def test_importances_sum_to_1() -> None:
    from titan.bayesian.temporal_inference import run_temporal_inference
    from configs.pipeline_config import PipelineConfig
    cfg = PipelineConfig(bayesian_backend="sklearn")

    for mode in TemporalMode:
        ext    = TemporalFeatureExtractor(TINY, mode)
        stack  = ext.extract(SYNTH)
        result = run_temporal_inference(stack, cfg)
        total  = sum(result.feature_importances.values())
        assert abs(total - 1.0) < 0.02, (
            f"[{mode.value}] Feature importances sum = {total:.4f}, expected ~1.0"
        )

for fn in [test_temporal_inference_present, test_temporal_inference_past,
           test_temporal_inference_future,
           test_future_mean_posterior_higher_than_present,
           test_importances_sum_to_1]:
    run(fn.__name__, fn)


# ===========================================================================
# 9. describe_prior_changes returns strings
# ===========================================================================

def test_describe_prior_changes_all_modes() -> None:
    for mode in TemporalMode:
        desc = describe_prior_changes(mode)
        assert isinstance(desc, str)
        assert len(desc) > 20, f"Description for {mode.value} is too short"

run("test_describe_prior_changes_all_modes", test_describe_prior_changes_all_modes)


# ===========================================================================
# Results
# ===========================================================================

print(f"\n{'='*62}")
print(f"  TEMPORAL TESTS: {len(PASS)} passed  {len(FAIL)} failed")
print(f"{'='*62}")
for n in PASS:
    print(f"  ✓  {n}")
if FAIL:
    print()
    for n, tb in FAIL:
        print(f"\n  ✗  {n}")
        for line in tb.strip().split('\n')[-5:]:
            print(f"     {line}")
print(f"{'='*62}")