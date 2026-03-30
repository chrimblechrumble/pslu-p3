"""
titan/temporal_features.py
===========================
Feature extraction for PAST and FUTURE temporal habitability modes.

These supplement the core 8 features in titan/features.py with:

  PAST mode — two new features derived from existing Cassini data:
    impact_melt_proxy   — SAR-bright annuli around impact craters
                          proxy for past liquid water exposure
                          Source: SAR + Hedgepeth et al. (2020) crater catalog
    cryovolcanic_flux   — proximity to cryovolcanic candidate features
                          proxy for past subsurface heat/ocean pathway
                          Source: SAR + Lopes et al. (2007/2013) catalog

  FUTURE mode — 8 transformed features:
    water_ammonia_solvent  — liquid water-ammonia proxy (global ocean)
    organic_stockpile      — accumulated organic inventory
    dissolved_energy       — chemical energy from organics in warm water
    water_ammonia_cycle    — analog to methane cycle under 200K conditions
    (plus 4 retained/adapted features from PRESENT)

All features are normalised to [0, 1].
All datasets used are from existing Cassini observations — NO future-state
data exists.  FUTURE features are derived from present-day observations
used as proxies for future conditions.

References
----------
Hedgepeth et al. (2020)  doi:10.1016/j.icarus.2020.113664  (crater catalog)
Lopes et al. (2007)      doi:10.1016/j.icarus.2006.09.006  (cryovolcanism)
Lopes et al. (2013)      doi:10.1002/jgre.20062             (cryovolcanism)
Lorenz et al. (1997)     doi:10.1029/97GL52843              (future habitable window)
Neish et al. (2018)      doi:10.1089/ast.2017.1758          (crater melt / prebiotic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from titan.features import FeatureStack, FeatureExtractor, FEATURE_NAMES
from titan.preprocessing import CanonicalGrid, normalise_to_0_1
from configs.temporal_config import TemporalMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known cryovolcanic candidate locations from Lopes et al. (2007, 2013)
# ---------------------------------------------------------------------------

# Coordinates of major cryovolcanic candidate features from:
#   Lopes et al. (2007) Icarus 186:395 — Table 1
#   Lopes et al. (2013) JGR-Planets 118:416 — Table 2
# (lon_west_deg, lat_deg)
CRYOVOLCANIC_CANDIDATES = [
    (144.5,  9.8,  "Sotra Facula"),      # most convincing (Lopes 2013)
    (179.7, 40.3,  "Doom Mons"),
    (145.6,  9.8,  "Mohini Fluctus"),
    (143.2,  9.7,  "Rohe Fluctus"),
    (211.3, 19.1,  "Winia Fluctus"),
    (181.3, 28.9,  "Erebor Mons"),
    ( 78.0, 24.5,  "Ara Fluctus"),
    (174.3, 37.5,  "Emakong Patera"),
    (219.0, 27.1,  "Hotei Regio candidate"),
    (162.0, 26.0,  "Tui Regio candidate"),
]

# Known large impact craters (fresh, with melt signatures) from
#   Hedgepeth et al. (2020) Icarus 344:113664 — Table 1 selected entries
#   Neish et al. (2018) — best-candidate craters for prebiotic chemistry
# (lon_west_deg, lat_deg, diameter_km)
IMPACT_MELT_CRATERS = [
    (199.0,  5.0,  82.0,  "Selk"),      # Dragonfly target; fresh, melt signatures
    (149.5, 11.6,  39.0,  "Ksa"),       # smallest confirmed with melt
    ( 28.1, 11.7,  80.0,  "Sinlap"),    # fresh, bright floor
    (183.2, -1.0, 110.0,  "Hano"),      # central uplift visible
    (200.5, -1.4, 115.0,  "Afekan"),    # well-preserved
    ( 35.6, 48.2, 425.0,  "Menrva"),    # largest; may have breached ocean interface
    (254.5, -5.0,  78.0,  "Forseti"),   # large, degraded
    (  0.0,  7.0,  28.0,  "Momoy"),     # small, fresh
]


# ---------------------------------------------------------------------------
# PAST features
# ---------------------------------------------------------------------------

def _gaussian_proximity_map(
    sites: List[tuple],   # list of (lon_west, lat, ...) tuples
    grid:  CanonicalGrid,
    sigma_deg: float = 10.0,
    use_diameter: bool = False,
) -> np.ndarray:
    """
    Build a Gaussian proximity map from a list of point locations.

    Each site contributes a Gaussian blob centred at its lon/lat.
    Larger craters (if use_diameter=True) have proportionally wider Gaussians.

    Parameters
    ----------
    sites:
        List of (lon_west_deg, lat_deg, [optionally: diameter_km or label, ...])
    grid:
        Canonical grid.
    sigma_deg:
        Gaussian width in degrees.
    use_diameter:
        If True, expects sites to be (lon, lat, diameter_km, ...).
        Gaussian width scales with sqrt(diameter_km).

    Returns
    -------
    np.ndarray
        float32 proximity map, shape (nrows, ncols), range [0, 1].
    """
    lats = grid.lat_centres_deg()
    lons = grid.lon_centres_deg()
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    result = np.zeros((grid.nrows, grid.ncols), dtype=np.float32)

    for site in sites:
        lon_w   = float(site[0])
        lat_s   = float(site[1])
        # Wrap-safe longitude difference
        dlon = lon_grid - lon_w
        dlon = ((dlon + 180) % 360) - 180   # wrap to ±180
        dlat = lat_grid - lat_s

        if use_diameter and len(site) > 2:
            diam_km = float(site[2])
            # Scale sigma by sqrt(diameter_km / reference_km)
            # Reference: 100 km crater → sigma_deg
            sigma_eff = sigma_deg * np.sqrt(max(diam_km, 10.0) / 100.0)
        else:
            sigma_eff = sigma_deg

        gauss = np.exp(-(dlon**2 + dlat**2) / (2 * sigma_eff**2))
        result = np.maximum(result, gauss.astype(np.float32))

    return result


def extract_impact_melt_proxy(
    stack: xr.Dataset,
    grid:  CanonicalGrid,
) -> np.ndarray:
    """
    Compute the impact melt proxy for PAST habitability.

    Combines two signals:
    1. SAR-bright annuli around known crater locations (fresh water-ice exposure)
       → Gaussian proximity to crater locations, scaled by crater diameter
    2. SAR backscatter elevated at crater margins relative to surroundings
       → identifies water-ice-like bright spots near craters

    Scientific basis
    ----------------
    Fresh craters on Titan show SAR-bright annuli interpreted as water-ice
    exposed by impact melting (Neish et al. 2018 Astrobiology).
    This ice was in liquid form (water or water-ammonia) for 10²–10⁴ years
    after impact (O'Brien et al. 2005; Hedgepeth et al. 2022).
    Larger, fresher craters = longer liquid water duration = more habitability.

    Parameters
    ----------
    stack:
        Canonical data stack (uses 'sar_mosaic').
    grid:
        Canonical grid.

    Returns
    -------
    np.ndarray
        float32 impact melt proxy, [0, 1].
    """
    # Gaussian proximity to all known impact craters with melt potential
    proximity = _gaussian_proximity_map(
        IMPACT_MELT_CRATERS, grid,
        sigma_deg=8.0, use_diameter=True,
    )

    # Boost by SAR backscatter (bright annuli around craters)
    if "sar_mosaic" in stack:
        sar  = stack["sar_mosaic"].values.astype(np.float32)
        sar_norm = normalise_to_0_1(sar, percentile_lo=2, percentile_hi=98)
        # High SAR backscatter near crater sites = water-ice exposure
        # Use combined: proximity × SAR brightness
        boosted = np.where(
            np.isfinite(sar_norm),
            np.clip(proximity * 0.5 + sar_norm * proximity * 0.5, 0, 1),
            proximity,
        )
        return boosted.astype(np.float32)

    logger.warning("SAR mosaic not available; impact_melt_proxy uses crater proximity only.")
    return proximity


def extract_cryovolcanic_flux(
    stack: xr.Dataset,
    grid:  CanonicalGrid,
) -> np.ndarray:
    """
    Compute the cryovolcanic flux proxy for PAST habitability.

    Encodes proximity to cryovolcanic candidate features identified by
    Lopes et al. (2007, 2013) in Cassini RADAR SAR data.

    Scientific basis
    ----------------
    Cryovolcanism on Titan is debated but several features show flow
    morphologies consistent with water-ammonia eruptions (Lopes 2007/2013).
    These would have provided: (a) surface-ocean chemical exchange,
    (b) temporary liquid water environments, (c) energy gradients.
    In early Titan (higher internal heat from radioactive decay),
    cryovolcanism was likely more frequent (Tobie et al. 2006 Nature 440:61).

    Parameters
    ----------
    stack:
        Canonical data stack (uses 'sar_mosaic').
    grid:
        Canonical grid.

    Returns
    -------
    np.ndarray
        float32 cryovolcanic proximity map, [0, 1].
    """
    # Gaussian proximity to known cryovolcanic candidates
    proximity = _gaussian_proximity_map(
        CRYOVOLCANIC_CANDIDATES, grid, sigma_deg=12.0, use_diameter=False,
    )

    # SAR-bright flow features can reinforce the signal
    if "sar_mosaic" in stack:
        sar = stack["sar_mosaic"].values.astype(np.float32)
        sar_norm = normalise_to_0_1(sar, percentile_lo=2, percentile_hi=98)
        # High-SAR near candidate sites suggests flow features
        boosted = np.where(
            np.isfinite(sar_norm),
            np.clip(proximity * 0.6 + sar_norm * proximity * 0.4, 0, 1),
            proximity,
        )
        return boosted.astype(np.float32)

    return proximity


# ---------------------------------------------------------------------------
# FUTURE features
# ---------------------------------------------------------------------------

def extract_water_ammonia_solvent(
    stack: xr.Dataset,
    grid:  CanonicalGrid,
) -> np.ndarray:
    """
    Water-ammonia solvent availability proxy for FUTURE habitability.

    In the red giant scenario (Lorenz et al. 1997), the entire surface of
    Titan is covered by liquid water-ammonia at ~200 K during a window
    of ~several hundred Myr.  The spatial distribution of this solvent
    depends on topography: lower terrain accumulates deeper oceans.

    Derived from PRESENT topography (GTDR) as a future bathymetry proxy:
      - Low terrain → deeper future ocean → higher habitability proxy
      - High terrain → shallower or absent ocean → lower proxy

    Scientific basis
    ----------------
    Lorenz et al. (1997) GRL 24:2905: global surface T ~200K allows
    eutectic water-ammonia mixtures (freezing point ~176 K) to remain
    liquid.  The surface is not uniformly covered; topography determines
    ocean depth.  This analogy maps directly to current DEM.

    ⚠️ ASSUMPTION: Current topography approximates future ocean bathymetry.

    Parameters
    ----------
    stack, grid: standard.

    Returns
    -------
    np.ndarray float32 [0, 1].
    """
    if "topography" in stack:
        dem = stack["topography"].values.astype(np.float64)
        # Low terrain → likely future ocean (invert DEM)
        # Clip to reasonable elevation range: −2000 to +1000 m
        dem_clipped = np.clip(dem, -2000, 1000)
        # Invert: low elevation → high future solvent availability
        inverted = normalise_to_0_1(-dem_clipped, percentile_lo=2, percentile_hi=98)
        return inverted.astype(np.float32)

    # Fallback: global constant (Lorenz 1997 says whole surface warm enough)
    # Use 0.85 as the flat prior from TemporalPriorSet
    logger.warning("No topography for water_ammonia_solvent; using flat prior.")
    return np.full((grid.nrows, grid.ncols), 0.85, dtype=np.float32)


def extract_organic_stockpile(
    stack: xr.Dataset,
    grid:  CanonicalGrid,
) -> np.ndarray:
    """
    Accumulated organic stockpile proxy for FUTURE habitability.

    In the future epoch (red giant, ~6 Gya), all surface organics dissolve
    into the global water-ammonia ocean and become available as chemical
    substrate.  The current surface organic distribution is the best proxy
    for the future stockpile.

    Data sources and priority (mirrors :meth:`~titan.features.FeatureExtractor._organic_abundance`)
    -------------------------------------------------------------------------------------------------
    1. **VIMS+ISS mosaic — primary (where available).**
       1.59/1.27 µm band ratio; the spectroscopically established tholin
       proxy.  Coverage ~50% (0–180°W).

    2. **Geomorphology-class scores — gap-fill (global, 100% coverage).**
       Uses :data:`~titan.features.TERRAIN_ORGANIC_SCORES`: each Lopes 2019
       terrain class carries a published VIMS-derived organic abundance value.
       Provides seamless global coverage without any radiometric seam.

       ISS 938nm broadband is intentionally NOT used — raw values differ from
       VIMS by a factor ~3000 (different physical quantities), producing an
       irremovable seam at the coverage boundary regardless of normalisation.

    Scientific basis
    ----------------
    Neish et al. (2009) Icarus 201:412: tholins produce amino acids when
    dissolved in ammonia-water.  Billions of years of UV photolysis and
    atmospheric chemistry will have deposited vast tholin layers across the
    entire surface.  The present distribution (dominated by dune fields at
    ~0.82 and plains at ~0.68) is a lower bound on the future stockpile.

    ⚠️ ASSUMPTION: Present organic distribution → future chemical substrate.

    Parameters
    ----------
    stack:
        xarray Dataset with layers ``vims_mosaic`` and/or ``geomorphology``.
    grid:
        Canonical grid (provides fallback shape).

    Returns
    -------
    np.ndarray
        float32 array of shape (nrows, ncols), values in [0, 1].
        100% global coverage when geomorphology is available.
    """
    from titan.features import TERRAIN_ORGANIC_SCORES, _geo_class_to_organic

    # ── Step 1: build geomorphology-based organic map (global, 100% cov) ──
    # Same approach as Option B in _organic_abundance: each terrain class
    # maps to a cited VIMS-derived organic abundance score.
    geo_organic: Optional[np.ndarray] = None
    if "geomorphology" in stack:
        geo_raw: np.ndarray = stack["geomorphology"].values
        geo_int32: np.ndarray = np.where(
            np.isfinite(geo_raw.astype(np.float32)), geo_raw, 0
        ).astype(np.int32)
        geo_organic = _geo_class_to_organic(geo_int32)
        logger.debug(
            "extract_organic_stockpile: geo-based map built (valid=%.1f%%)",
            100.0 * np.isfinite(geo_organic).mean(),
        )

    # ── Step 2: VIMS primary (where available) ────────────────────────────
    vims_norm: Optional[np.ndarray] = None
    if "vims_mosaic" in stack:
        vims_raw: np.ndarray = stack["vims_mosaic"].values.astype(np.float32)
        vims_norm = normalise_to_0_1(vims_raw, percentile_lo=2, percentile_hi=98)

    # ── Step 3: combine — VIMS primary, geo gap-fill ──────────────────────
    if vims_norm is not None and geo_organic is not None:
        from titan.features import TERRAIN_ORGANIC_SCORES, _geo_class_to_organic
        geo_raw2: np.ndarray = stack["geomorphology"].values
        geo_int: np.ndarray = np.where(
            np.isfinite(geo_raw2.astype(np.float32)), geo_raw2, 0
        ).astype(np.int32)

        # Published scores + single global offset (preserves relative ranking)
        geo_published = _geo_class_to_organic(geo_int)
        overlap_mask: np.ndarray = np.isfinite(vims_norm) & np.isfinite(geo_published)
        global_offset = 0.0
        if int(overlap_mask.sum()) >= 5000:
            global_offset = float(np.mean(vims_norm[overlap_mask])) -                             float(np.mean(geo_published[overlap_mask]))
        calibrated_geo: np.ndarray = np.clip(
            geo_published.astype(np.float64) + global_offset, 0.0, 1.0
        )
        calibrated_geo[~np.isfinite(geo_published)] = np.nan

        # Row-wise local offset: ensures continuity at the exact VIMS edge per row
        nrows, ncols = vims_norm.shape
        deg_per_col = 360.0 / ncols
        decay_cols  = max(5, int(10.0 / deg_per_col))
        vims_valid  = np.isfinite(vims_norm)

        last_valid_col = np.full(nrows, -1, dtype=np.int32)
        for r in range(nrows):
            rv = np.where(vims_valid[r, :])[0]
            if len(rv):
                last_valid_col[r] = int(rv.max())

        edge_offsets = np.zeros(nrows, dtype=np.float64)
        for r in np.where(last_valid_col >= 0)[0]:
            ce = last_valid_col[r]
            if np.isfinite(calibrated_geo[r, ce]) and np.isfinite(vims_norm[r, ce]):
                edge_offsets[r] = float(vims_norm[r, ce]) - float(calibrated_geo[r, ce])

        col_idx       = np.arange(ncols, dtype=np.float64)[np.newaxis, :]
        c_edge_2d     = last_valid_col[:, np.newaxis].astype(np.float64)
        dist          = col_idx - c_edge_2d
        decay_factor  = np.where(dist > 0, np.exp(-dist / max(decay_cols, 1)), 0.0)
        offset_map    = edge_offsets[:, np.newaxis] * decay_factor
        geo_fill_mask = ~vims_valid & np.isfinite(calibrated_geo)
        adjusted_geo  = calibrated_geo.copy()
        adjusted_geo[geo_fill_mask] = (
            calibrated_geo[geo_fill_mask] + offset_map[geo_fill_mask]
        )

        result: np.ndarray = np.clip(
            np.where(np.isfinite(vims_norm), vims_norm, adjusted_geo),
            0.0, 1.0,
        )
        logger.info(
            "extract_organic_stockpile: VIMS (%d px) + row-offset geo fill "
            "(%d px), valid=%.1f%%",
            int(np.sum(vims_valid)),
            int(geo_fill_mask.sum()),
            100.0 * np.isfinite(result).mean(),
        )
        return result.astype(np.float32)

    if vims_norm is not None:
        return vims_norm

    if geo_organic is not None:
        logger.info(
            "extract_organic_stockpile: geomorphology-only (no VIMS), "
            "valid=%.1f%%", 100.0 * np.isfinite(geo_organic).mean(),
        )
        return geo_organic

    # ── ISS gap-fill fallback (no geomorphology available) ──────────────────
    # Mirrors the same fallback logic in FeatureExtractor._organic_abundance().
    if "iss_mosaic_450m" in stack:
        iss_raw: np.ndarray = stack["iss_mosaic_450m"].values.astype(np.float64)
        iss_finite = iss_raw[np.isfinite(iss_raw)]
        if len(iss_finite) > 0:
            i2  = float(np.percentile(iss_finite,  2))
            i98 = float(np.percentile(iss_finite, 98))
            iss_inv = np.clip(
                1.0 - (iss_raw - i2) / (i98 - i2 + 1e-12), 0.0, 1.0
            ).astype(np.float32)
            iss_inv[~np.isfinite(iss_raw)] = np.nan

            if vims_norm is not None:
                overlap = np.isfinite(vims_norm) & np.isfinite(iss_inv)
                n_ov = int(overlap.sum())
                if n_ov > 1000:
                    offset = float(np.median(vims_norm[overlap])) -                              float(np.median(iss_inv[overlap]))
                    iss_adj = np.clip(iss_inv + offset, 0.0, 1.0)
                    result: np.ndarray = np.where(
                        np.isfinite(vims_norm), vims_norm, iss_adj
                    ).astype(np.float32)
                    logger.warning(
                        "extract_organic_stockpile: geomorphology not found; "
                        "using ISS gap-fill (offset=%.3f, overlap=%d px).",
                        offset, n_ov,
                    )
                    return result
            else:
                return iss_inv

    if vims_norm is not None:
        return vims_norm

    logger.warning(
        "extract_organic_stockpile: no VIMS, geomorphology, or ISS available; "
        "returning flat prior 0.85."
    )
    return np.full((grid.nrows, grid.ncols), 0.85, dtype=np.float32)


def extract_dissolved_energy(
    stack: xr.Dataset,
    grid:  CanonicalGrid,
) -> np.ndarray:
    """
    Chemical energy from organics dissolving in future water-ammonia.

    When the surface ocean forms at ~200 K, accumulated tholins, HCN,
    and acetylene dissolve in water-ammonia and undergo hydrolysis,
    providing chemical energy and building blocks (Neish et al. 2009).

    This is the FUTURE analog of acetylene_energy (PRESENT mode).
    The spatial distribution is proportional to:
    - Organic stockpile (more organics → more energy on dissolution)
    - Low terrain (ocean forms first in depressions → more dissolution)

    ⚠️ ASSUMPTION: Organic dissolution rate is proportional to current
    organic abundance × ocean depth proxy.

    Parameters
    ----------
    stack, grid: standard.

    Returns
    -------
    np.ndarray float32 [0, 1].
    """
    stockpile = extract_organic_stockpile(stack, grid)
    ocean_proxy = extract_water_ammonia_solvent(stack, grid)

    # Energy where both organics and ocean exist
    combined = np.where(
        np.isfinite(stockpile) & np.isfinite(ocean_proxy),
        np.sqrt(stockpile * ocean_proxy),  # geometric mean: need both
        np.where(np.isfinite(stockpile), stockpile * 0.5, ocean_proxy * 0.5),
    )
    return normalise_to_0_1(combined).astype(np.float32)


def extract_water_ammonia_cycle(
    stack: xr.Dataset,
    grid:  CanonicalGrid,
) -> np.ndarray:
    """
    Water-ammonia meteorological cycle proxy for FUTURE habitability.

    Lorenz et al. (1997) predict that once the surface warms to ~200 K,
    a water-ammonia evaporation/precipitation cycle begins — analogous to
    Titan's current methane cycle.  This would transport dissolved organics,
    maintain chemical gradients, and provide energy flux.

    Derived from topographic slope: steep terrain → stronger runoff →
    more vigorous evaporation/condensation cycle (same logic as
    surface_atm_interaction in PRESENT mode, but for water-ammonia).

    ⚠️ ASSUMPTION: Topographic slope drives the future water-ammonia cycle
    analogously to how it drives the current methane cycle.

    Parameters
    ----------
    stack, grid: standard.

    Returns
    -------
    np.ndarray float32 [0, 1].
    """
    if "topography" in stack:
        dem = stack["topography"].values.astype(np.float64)
        dy, dx = np.gradient(np.where(np.isfinite(dem), dem, 0.0))
        slope  = np.sqrt(dx**2 + dy**2)
        slope_norm = normalise_to_0_1(slope, percentile_lo=2, percentile_hi=98)
        return slope_norm.astype(np.float32)

    # Fallback: methane cycle feature is a reasonable proxy
    # (active methane-cycle regions likely correlate with active water-ammonia cycle)
    from titan.features import FeatureExtractor
    extractor = FeatureExtractor(grid)
    nan = np.full((grid.nrows, grid.ncols), np.nan, dtype=np.float32)
    return extractor._methane_cycle(stack, nan)


def extract_global_ocean_habitability(
    stack: xr.Dataset,
    grid:  CanonicalGrid,
) -> np.ndarray:
    """
    Global ocean habitability proxy for FUTURE habitability.

    In FUTURE mode, the subsurface water-ammonia ocean expands to the
    surface (Lorenz 1997).  The entire surface becomes an ocean during
    the ~200K window.  This feature replaces subsurface_ocean from
    PRESENT mode and represents the habitability of the global surface
    ocean.

    Key factors:
    - Ocean chemical energy (from Affholder 2025 glycine fermentation)
    - Water-rock interaction at seafloor → redox gradients
    - Organic availability (from billions of years of surface accumulation)

    Derived from:
    - Organic stockpile (chemical substrate)
    - Future ocean depth proxy (inverse DEM)
    - Topographic complexity (seafloor habitat diversity)

    ⚠️ ASSUMPTION: Surface ocean habitability scales with organic availability
    × ocean depth × topographic complexity of seafloor.

    Parameters
    ----------
    stack, grid: standard.

    Returns
    -------
    np.ndarray float32 [0, 1].
    """
    stockpile  = extract_organic_stockpile(stack, grid)
    ocean_prox = extract_water_ammonia_solvent(stack, grid)

    from titan.preprocessing import compute_topographic_roughness
    if "topography" in stack:
        dem      = stack["topography"].values.astype(np.float64)
        roughness = compute_topographic_roughness(dem, window_radius=3)
    else:
        roughness = np.full((grid.nrows, grid.ncols), 0.4, dtype=np.float32)

    # Ocean habitability = product of three factors
    combined = np.where(
        np.isfinite(stockpile) & np.isfinite(ocean_prox) & np.isfinite(roughness),
        (stockpile + ocean_prox + roughness) / 3.0,
        np.full_like(stockpile, np.nan),
    )
    return normalise_to_0_1(combined).astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal FeatureStack builders
# ---------------------------------------------------------------------------

@dataclass
class TemporalFeatureStack:
    """
    Feature stack for a specific temporal mode.

    Contains the feature arrays for the selected mode, along with
    metadata about which features are active and their scientific
    justification.

    Attributes
    ----------
    mode:
        Temporal mode used to compute these features.
    features:
        Dict mapping feature name → 2-D float32 array.
    grid:
        Canonical grid.
    """
    mode:     TemporalMode
    features: Dict[str, np.ndarray]
    grid:     CanonicalGrid

    def as_array(self) -> np.ndarray:
        """Stack features into (n_features, nrows, ncols) float32 array."""
        names = list(self.features.keys())
        return np.stack([self.features[n] for n in names], axis=0).astype(np.float32)

    def feature_names(self) -> List[str]:
        """Return list of feature names in this stack."""
        return list(self.features.keys())

    def get_feature(self, name: str) -> Optional[np.ndarray]:
        """Return the feature array for the given name, or None if absent."""
        return self.features.get(name)

    def coverage_fraction(self) -> Dict[str, float]:
        return {
            name: float(np.sum(np.isfinite(arr)) / arr.size)
            for name, arr in self.features.items()
        }

    def to_xarray(self) -> xr.Dataset:
        lats = self.grid.lat_centres_deg()
        lons = self.grid.lon_centres_deg()
        data_vars = {
            name: xr.DataArray(
                arr, dims=["lat", "lon"],
                coords={"lat": lats, "lon": lons},
                attrs={"units": "dimensionless [0,1]",
                       "temporal_mode": self.mode.value},
            )
            for name, arr in self.features.items()
        }
        return xr.Dataset(
            data_vars,
            attrs={
                "title": f"Titan habitability features [{self.mode.value}]",
                "temporal_mode": self.mode.value,
            },
        )


class TemporalFeatureExtractor:
    """
    Extracts habitability features for any temporal mode.

    Delegates to FeatureExtractor for shared PRESENT features, and
    implements new features for PAST and FUTURE modes.

    Parameters
    ----------
    grid:
        Canonical spatial grid.
    mode:
        Temporal mode.
    window_config:
        Temporal habitability window parameters (D1: past epoch,
        D2: future window + uniform warming assumption).
        If None, uses HabitabilityWindowConfig defaults
        (3.5 Gya past epoch, 100–400 Myr near-future window).
    subsurface_ocean_base_prior:
        Base prior probability for the subsurface_ocean feature (D3).
        Default 0.03 (Neish et al. 2024: organic flux ~1 elephant/yr).
        Passed through to FeatureExtractor unchanged.
    """

    def __init__(
        self,
        grid: CanonicalGrid,
        mode: TemporalMode,
        window_config: Optional["HabitabilityWindowConfig"] = None,
        subsurface_ocean_base_prior: float = 0.03,
    ) -> None:
        self.grid = grid
        self.mode = mode
        self.subsurface_ocean_base_prior = subsurface_ocean_base_prior

        # Import here to avoid circular at module level
        from configs.pipeline_config import HabitabilityWindowConfig
        self.window_config = window_config or HabitabilityWindowConfig()

        self._present_extractor: Optional[FeatureExtractor] = None  # lazy init

    def _get_present_extractor(self) -> FeatureExtractor:
        if self._present_extractor is None:
            from titan.features import FeatureExtractor
            self._present_extractor = FeatureExtractor(
                self.grid,
                window_config               = self.window_config,
                subsurface_ocean_base_prior = self.subsurface_ocean_base_prior,
            )
        return self._present_extractor

    def extract(self, stack: xr.Dataset) -> TemporalFeatureStack:
        """
        Extract all features for the configured temporal mode.

        Parameters
        ----------
        stack:
            xarray Dataset from CanonicalDataStack.load().

        Returns
        -------
        TemporalFeatureStack
        """
        if self.mode == TemporalMode.PRESENT:
            return self._extract_present(stack)
        elif self.mode == TemporalMode.PAST:
            return self._extract_past(stack)
        elif self.mode == TemporalMode.FUTURE:
            return self._extract_future(stack)
        else:
            raise ValueError(f"Unknown temporal mode: {self.mode}")

    def _extract_present(self, stack: xr.Dataset) -> TemporalFeatureStack:
        """Extract present features using the standard FeatureExtractor."""
        extractor = self._get_present_extractor()
        feature_stack = extractor.extract(stack)
        features = {n: getattr(feature_stack, n) for n in FEATURE_NAMES}
        return TemporalFeatureStack(
            mode=TemporalMode.PRESENT,
            features=features,
            grid=self.grid,
        )

    def _extract_past(self, stack: xr.Dataset) -> TemporalFeatureStack:
        """Extract PAST features: 7 adapted PRESENT features + 2 new."""
        extractor = self._get_present_extractor()
        nan = np.full((self.grid.nrows, self.grid.ncols), np.nan, dtype=np.float32)

        features = {
            "liquid_hydrocarbon":      extractor._liquid_hydrocarbon(stack, nan),
            "organic_abundance":       extractor._organic_abundance(stack, nan),
            "acetylene_energy":        extractor._acetylene_energy(stack, nan),
            "methane_cycle":           extractor._methane_cycle(stack, nan),
            "surface_atm_interaction": extractor._surface_atm_interaction(stack, nan),
            "topographic_complexity":  extractor._topographic_complexity(stack, nan),
            "geomorphologic_diversity":extractor._geomorphologic_diversity(stack, nan),
            # New PAST-specific features
            "impact_melt_proxy":       extract_impact_melt_proxy(stack, self.grid),
            "cryovolcanic_flux":       extract_cryovolcanic_flux(stack, self.grid),
        }

        for name, arr in features.items():
            pct = 100.0 * np.sum(np.isfinite(arr)) / arr.size
            logger.info("[PAST] Feature %-32s  valid=%.1f%%", name, pct)

        return TemporalFeatureStack(
            mode=TemporalMode.PAST,
            features=features,
            grid=self.grid,
        )

    def _extract_future(self, stack: xr.Dataset) -> TemporalFeatureStack:
        """Extract FUTURE features: 4 transformed + 4 shared/adapted."""
        extractor = self._get_present_extractor()
        nan = np.full((self.grid.nrows, self.grid.ncols), np.nan, dtype=np.float32)

        features = {
            # Transformed features
            "water_ammonia_solvent":   extract_water_ammonia_solvent(stack, self.grid),
            "organic_stockpile":       extract_organic_stockpile(stack, self.grid),
            "dissolved_energy":        extract_dissolved_energy(stack, self.grid),
            "water_ammonia_cycle":     extract_water_ammonia_cycle(stack, self.grid),
            # Retained/adapted features
            "surface_atm_interaction": extractor._surface_atm_interaction(stack, nan),
            "topographic_complexity":  extractor._topographic_complexity(stack, nan),
            "geomorphologic_diversity":extractor._geomorphologic_diversity(stack, nan),
            "global_ocean_habitability": extract_global_ocean_habitability(stack, self.grid),
        }

        for name, arr in features.items():
            pct = 100.0 * np.sum(np.isfinite(arr)) / arr.size
            logger.info("[FUTURE] Feature %-32s  valid=%.1f%%", name, pct)

        return TemporalFeatureStack(
            mode=TemporalMode.FUTURE,
            features=features,
            grid=self.grid,
        )
