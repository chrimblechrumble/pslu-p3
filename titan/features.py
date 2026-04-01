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
titan/features.py
==================
Stage 3 -- Feature Extraction.

Converts the canonical multi-layer data stack into eight normalised [0,1]
feature maps that feed the Bayesian habitability model.

Feature definitions are grounded in published Titan astrobiology research.
Every prior value and feature design decision is traceable to a specific
reference cited in the docstrings.

Features
--------
1. liquid_hydrocarbon     -- surface liquid HC solvent availability
2. organic_abundance      -- surface organic / tholin coverage
3. acetylene_energy       -- chemical energy gradient (C2H2 + H2)
4. methane_cycle          -- active methane-cycle intensity
5. surface_atm_interaction -- surface-atmosphere chemical exchange
6. topographic_complexity -- local terrain roughness
7. geomorphologic_diversity -- Shannon diversity of terrain classes
8. subsurface_ocean       -- cryovolcanic/ocean proximity proxy

References
----------
McKay & Smith (2005)      doi:10.1016/j.icarus.2005.05.018
McKay (2016)              doi:10.3390/life6010008
Yanez et al. (2024)       doi:10.1016/j.icarus.2024.115969
Strobel (2010)            doi:10.1016/j.icarus.2010.02.009
Schulze-Makuch & Grinspoon (2005) doi:10.1089/ast.2005.5.560
Lopes et al. (2019)       doi:10.1038/s41550-019-0917-6
Lorenz et al. (2013)      doi:10.1016/j.icarus.2013.04.002
Iess et al. (2012)        doi:10.1126/science.1219631
Neish et al. (2018)       doi:10.1089/ast.2017.1758
Le Mouelic et al. (2019)  doi:10.1016/j.icarus.2018.09.017
Malaska et al. (2025)     Titan After Cassini-Huygens Ch.9
Meyer-Dombard et al.(2025) Titan After Cassini-Huygens Ch.14
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from titan.preprocessing import (
    CanonicalGrid,
    compute_topographic_roughness,
    compute_terrain_diversity,
    normalise_to_0_1,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature stack container
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "liquid_hydrocarbon",
    "organic_abundance",
    "acetylene_energy",
    "methane_cycle",
    "surface_atm_interaction",
    "topographic_complexity",
    "geomorphologic_diversity",
    "subsurface_ocean",
]

# ---------------------------------------------------------------------------
# Terrain-class organic abundance lookup table
# ---------------------------------------------------------------------------

#: Organic abundance [0, 1] assigned to each Lopes et al. (2019) terrain class.
#:
#: Values are grounded in published VIMS spectral studies:
#:
#: Class 1 -- Craters:
#:   Impact craters expose water ice but mix in surface organics.
#:   Fresh craters show water-ice floors; ejecta has moderate organic content.
#:   Value 0.35 -- Neish et al. (2018) Astrobiology 18, 571; Hedgepeth et al. (2020)
#:
#: Class 2 -- Dunes:
#:   Equatorial dune fields are the most organic-rich terrain on Titan.
#:   Composed of complex tholin sand grains (Lorenz et al. 2006).
#:   High 1.59/1.27 umm ratio confirmed by VIMS (Le Mouelic et al. 2019).
#:   Value 0.82 -- Lorenz et al. (2006) Science 312, 724; Malaska et al. (2025)
#:
#: Class 3 -- Plains (undifferentiated):
#:   Organic sediment blanket covering ~65 % of Titan's surface.
#:   Moderate-high tholin content; VIMS shows elevated red ratio.
#:   Value 0.68 -- Lopes et al. (2016) Icarus 270; Janssen et al. (2016)
#:
#: Class 4 -- Basins:
#:   Topographic lows that accumulate organic fallout.
#:   Moderate organic content -- less uniform than plains.
#:   Value 0.58 -- Lopes et al. (2020) Nature Astronomy 4, 228
#:
#: Class 5 -- Mountains / Hummocky terrain:
#:   Geologically old, water-ice-rich crust; organics present but diluted.
#:   VIMS indicates lower tholin ratio than plains.
#:   Value 0.25 -- Solomonidou et al. (2018) JGR Planets 123; Mitri et al. (2010)
#:
#: Class 6 -- Labyrinth terrain:
#:   Spectral composition similar to plains; moderate organic content.
#:   Malaska et al. (2020) Icarus 344 -- same material source as plains.
#:   Value 0.55 -- Malaska et al. (2020); Lopes et al. (2020)
#:
#: Class 7 -- Lakes / Seas:
#:   Liquid hydrocarbon bodies -- not solid organics.
#:   Low tholin band ratio (Brown et al. 2008 confirmed liquid ethane/methane).
#:   Value 0.05 -- Brown et al. (2008) Nature 454; Hayes et al. (2008)
#:
#: References
#: ----------
#: Lorenz et al. (2006)       doi:10.1126/science.1126892
#: Brown et al. (2008)        doi:10.1038/nature07100
#: Hayes et al. (2008)        doi:10.1029/2007GL032324
#: Mitri et al. (2010)        doi:10.1029/2010JE003592
#: Lopes et al. (2016)        doi:10.1016/j.icarus.2016.02.022
#: Janssen et al. (2016)      doi:10.1016/j.icarus.2015.09.027
#: Le Mouelic et al. (2019)   doi:10.1016/j.icarus.2018.09.017
#: Lopes et al. (2020)        doi:10.1038/s41550-019-0917-6
#: Malaska et al. (2020)      doi:10.1016/j.icarus.2019.113764
#: Neish et al. (2018)        doi:10.1089/ast.2017.1758
#: Hedgepeth et al. (2020)    doi:10.1016/j.icarus.2019.113422
#: Malaska et al. (2025)      Titan After Cassini-Huygens Ch.9
TERRAIN_ORGANIC_SCORES: Dict[int, float] = {
    0: float("nan"),  # nodata -- no information
    1: 0.35,          # Craters -- moderate; mixed water ice + ejecta organics
    2: 0.82,          # Dunes -- highest; tholin-dominated organic sand
    3: 0.68,          # Plains -- high; organic sediment blanket
    4: 0.58,          # Basins -- moderate-high; organic accumulation zones
    5: 0.25,          # Mountains -- low; water-ice-rich crustal material
    6: 0.55,          # Labyrinth -- moderate; spectral affinity with plains
    7: 0.05,          # Lakes/Seas -- very low; liquid HC, not solid organics
}


def _geo_class_to_organic(geo_int32: np.ndarray) -> np.ndarray:
    """
    Map a geomorphology class raster to per-pixel organic abundance scores.

    Uses the published ``TERRAIN_ORGANIC_SCORES`` lookup table derived from
    Cassini VIMS spectral studies of each Lopes et al. (2019) terrain class.

    Parameters
    ----------
    geo_int32:
        2-D integer array of terrain class labels (0=nodata, 1-7=classes)
        as produced by :class:`titan.io.shapefile_rasteriser.GeomorphologyRasteriser`.

    Returns
    -------
    np.ndarray
        float32 array of shape matching ``geo_int32``.  Values in [0, 1].
        Pixels where ``geo_int32`` is 0 (nodata) are set to NaN.

    Notes
    -----
    Class labels outside 0-7 are also mapped to NaN with a warning.
    """
    result: np.ndarray = np.full(geo_int32.shape, np.nan, dtype=np.float32)
    unknown_classes: list[int] = []

    for cls_id, score in TERRAIN_ORGANIC_SCORES.items():
        mask = geo_int32 == cls_id
        if np.isnan(score):
            result[mask] = np.nan   # nodata class -- leave as NaN
        else:
            result[mask] = float(score)

    # Flag any class IDs not in the table
    defined = set(TERRAIN_ORGANIC_SCORES.keys())
    for cls_id in np.unique(geo_int32):
        if int(cls_id) not in defined:
            unknown_classes.append(int(cls_id))

    if unknown_classes:
        logger.warning(
            "_geo_class_to_organic: unknown terrain class IDs %s -- "
            "those pixels will be NaN.",
            unknown_classes,
        )

    return result


@dataclass
class FeatureStack:
    """
    Container for all eight habitability-proxy feature maps.

    Each attribute is a 2-D float32 array of shape (nrows, ncols)
    with values in [0, 1].  NaN = missing / no data coverage.

    See module docstring for scientific rationale of each feature.
    """
    liquid_hydrocarbon:      np.ndarray
    organic_abundance:       np.ndarray
    acetylene_energy:        np.ndarray
    methane_cycle:           np.ndarray
    surface_atm_interaction: np.ndarray
    topographic_complexity:  np.ndarray
    geomorphologic_diversity:np.ndarray
    subsurface_ocean:        np.ndarray

    def as_array(self) -> np.ndarray:
        """
        Stack all features into a (n_features, nrows, ncols) float32 array.

        Returns
        -------
        np.ndarray
            Shape (8, nrows, ncols).
        """
        return np.stack(
            [getattr(self, n) for n in FEATURE_NAMES], axis=0
        ).astype(np.float32)

    def weighted_sum(self, weights: Dict[str, float]) -> np.ndarray:
        """
        Compute the weighted sum of all features.

        Parameters
        ----------
        weights:
            Dict mapping feature name -> weight (must sum to ~1.0).

        Returns
        -------
        np.ndarray
            Per-pixel weighted sum, float32, NaN-propagating.
        """
        total = np.zeros_like(self.liquid_hydrocarbon)
        weight_sum_valid = np.zeros_like(self.liquid_hydrocarbon)

        for name in FEATURE_NAMES:
            arr = getattr(self, name)
            w   = weights.get(name, 0.0)
            if w == 0.0:
                continue
            valid = np.isfinite(arr)
            total         = np.where(valid, total + w * arr, total)
            weight_sum_valid = np.where(valid, weight_sum_valid + w, weight_sum_valid)

        # Normalise by actual weight sum (handles NaN pixels gracefully)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(weight_sum_valid > 0, total / weight_sum_valid, np.nan)
        return result.astype(np.float32)

    def to_xarray(self, grid: CanonicalGrid) -> "xr.Dataset":
        """
        Convert to an xarray Dataset with lat/lon coordinates.

        Parameters
        ----------
        grid:
            Canonical grid (provides coordinate arrays).

        Returns
        -------
        xr.Dataset
        """
        import xarray as xr
        lats = grid.lat_centres_deg()
        lons = grid.lon_centres_deg()
        data_vars = {}
        for name in FEATURE_NAMES:
            arr = getattr(self, name)
            data_vars[name] = xr.DataArray(
                arr,
                dims=["lat", "lon"],
                coords={"lat": lats, "lon": lons},
                attrs={"units": "dimensionless [0,1]"},
            )
        return xr.Dataset(
            data_vars,
            attrs={"title": "Titan habitability feature stack"},
        )

    def coverage_fraction(self) -> Dict[str, float]:
        """Return fraction of non-NaN pixels per feature."""
        return {
            name: float(np.sum(np.isfinite(getattr(self, name)))
                        / getattr(self, name).size)
            for name in FEATURE_NAMES
        }

    def feature_names(self) -> List[str]:
        """Return list of feature names."""
        return list(FEATURE_NAMES)

    def get_feature(self, name: str) -> Optional[np.ndarray]:
        """Return the feature array for the given name, or None if absent."""
        return getattr(self, name, None)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Derives normalised habitability features from the canonical data stack.

    Parameters
    ----------
    grid:
        Canonical spatial grid.
    window_config:
        Temporal habitability window parameters (D1: past epoch,
        D2: future window).  Uses HabitabilityWindowConfig defaults
        if not supplied.
    subsurface_ocean_base_prior:
        Global base probability for the subsurface ocean feature (D3).
        Default 0.03 (revised down from earlier 0.10).  The k2 measurement
        (Iess et al. 2012) confirms the ocean exists, but surface expression
        is time-gated by D1/D2 constraints and poorly constrained spatially.
        Pass a different value to explore sensitivity.
    """

    def __init__(
        self,
        grid: CanonicalGrid,
        window_config: Optional["HabitabilityWindowConfig"] = None,
        subsurface_ocean_base_prior: float = 0.03,
    ) -> None:
        self.grid = grid
        # Import here to avoid circular at module level
        from configs.pipeline_config import HabitabilityWindowConfig
        self.window = window_config or HabitabilityWindowConfig()
        self.subsurface_ocean_base_prior = subsurface_ocean_base_prior

    def extract(self, stack: xr.Dataset) -> FeatureStack:
        """
        Extract all eight features from the canonical data stack.

        Missing source layers produce NaN feature maps; the pipeline
        continues gracefully with partial data.

        Parameters
        ----------
        stack:
            xarray Dataset produced by CanonicalDataStack.load().

        Returns
        -------
        FeatureStack
        """
        g   = self.grid
        nan = np.full((g.nrows, g.ncols), np.nan, dtype=np.float32)

        feats = {
            "liquid_hydrocarbon":      self._liquid_hydrocarbon(stack, nan),
            "organic_abundance":       self._organic_abundance(stack, nan),
            "acetylene_energy":        self._acetylene_energy(stack, nan),
            "methane_cycle":           self._methane_cycle(stack, nan),
            "surface_atm_interaction": self._surface_atm_interaction(stack, nan),
            "topographic_complexity":  self._topographic_complexity(stack, nan),
            "geomorphologic_diversity":self._geomorphologic_diversity(stack, nan),
            "subsurface_ocean":        self._subsurface_ocean(stack, nan),
        }

        for name, arr in feats.items():
            pct = 100.0 * np.sum(np.isfinite(arr)) / arr.size
            logger.info("Feature %-30s  valid=%.1f%%", name, pct)

        return FeatureStack(**feats)

    # -- Feature 1 -------------------------------------------------------------

    def _liquid_hydrocarbon(
        self, stack: xr.Dataset, nan: np.ndarray
    ) -> np.ndarray:
        """
        Surface liquid hydrocarbon availability -- Feature 1 (weight 0.25).

        Data source priority
        --------------------
        1. **Birch+2017 / Palermo+2022 polar lake raster** (``polar_lakes``
           layer, class labels 1 and 3).
           Expert-mapped lake and sea outlines at Cassini SAR resolution,
           covering both polar regions.  Filled lake pixels score **1.0**
           (confirmed liquid); Palermo pixels also score 1.0.
           This replaces the SAR proxy for the polar region where virtually
           all of Titan's liquid surface is found.

        2. **Lopes+2019 geomorphology lake class** (``geomorphology`` layer,
           terrain label 7).  Falls back to this when the Birch raster is
           absent. Also contributes where Lopes and Birch agree.

        3. **SAR low-backscatter proxy** (``sar_mosaic`` layer).
           Radar-dark surfaces (low sigma0) correlate with liquid or smooth
           organic material.  Used as tertiary gap-fill where neither of the
           above is available (primarily equatorial and mid-latitude regions
           without confirmed lakes).

        Note on empty basins
        --------------------
        Birch empty-basin pixels (label 2 = ``POLAR_LAKE_EMPTY``) are NOT
        included here -- they are paleo-lakes without confirmed present liquid
        and are handled instead as a sub-component of Feature 5
        (surface_atm_interaction) as a wetting/drying chemistry indicator.

        Scientific basis
        ----------------
        Liquid methane/ethane is the proposed non-aqueous solvent for
        hypothetical Titan surface life (McKay & Smith 2005; McKay 2016;
        Benner 2004).  Lakes cover ~1.5-2 % of Titan's surface globally
        but ~40 % of the north polar region (Hayes et al. 2008).
        The Birch+2017 mapping provides higher-fidelity shorelines than the
        global Lopes+2019 map, specifically designed for the polar regions
        where all confirmed liquid resides (Birch et al. 2017).

        Parameters
        ----------
        stack:
            Canonical data stack (xarray Dataset).
        nan:
            Template NaN array of shape (nrows, ncols).

        Returns
        -------
        np.ndarray
            float32 array in [0, 1].  1.0 = confirmed liquid surface;
            values between 0 and 1 from SAR proxy; NaN = no data.

        References
        ----------
        Birch et al. (2017) Icarus doi:10.1016/j.icarus.2017.01.032
        Hayes et al. (2008) GRL doi:10.1029/2007GL032324
        McKay & Smith (2005) Icarus doi:10.1016/j.icarus.2005.05.018
        Birch et al. (2017) Icarus doi:10.1016/j.icarus.2017.01.032
        Stofan et al. (2007) Nature doi:10.1038/nature05608
        """
        from titan.io.shapefile_rasteriser import (
            POLAR_LAKE_FILLED, POLAR_LAKE_PALERMO, POLAR_LAKE_EMPTY,
        )

        result: np.ndarray = nan.copy()

        # -- Priority 1: Birch+2017 / Palermo+2022 polar lake raster ----------
        # These are expert-mapped outlines from the same Cassini SAR data,
        # providing binary 1.0 for confirmed liquid and 0.0 elsewhere in the
        # mapped region.  Supersedes both the Lopes class and the SAR proxy
        # wherever Birch coverage exists.
        if "polar_lakes" in stack:
            pl: np.ndarray = stack["polar_lakes"].values.astype(np.int16)

            # Confirmed liquid: Birch filled (1) and Palermo (3)
            birch_liquid: np.ndarray = (
                (pl == POLAR_LAKE_FILLED) | (pl == POLAR_LAKE_PALERMO)
            ).astype(np.float32)

            # Confirmed non-liquid within Birch coverage area (any Birch pixel
            # that is NOT a filled lake and NOT nodata = 0 by background).
            # We set these to 0.0 so the SAR proxy can't override them.
            birch_any_coverage: np.ndarray = (pl != 0).astype(bool)
            birch_result: np.ndarray = np.where(
                birch_any_coverage, birch_liquid, np.nan
            ).astype(np.float32)

            result = np.where(np.isfinite(birch_result), birch_result, result)
            n_birch_liquid: int = int(
                ((pl == POLAR_LAKE_FILLED) | (pl == POLAR_LAKE_PALERMO)).sum()
            )
            logger.debug(
                "_liquid_hydrocarbon: Birch/Palermo: %d liquid pixels", n_birch_liquid
            )

        # -- Priority 2: Lopes+2019 geomorphology lake class (label 7) --------
        # Used as a supplementary source in regions not covered by Birch.
        if "geomorphology" in stack:
            geo: np.ndarray = stack["geomorphology"].values.astype(np.float32)
            lake_mask: np.ndarray = (geo == 7.0).astype(np.float32)
            lake_mask[geo == 0.0] = np.nan   # geomorphology nodata -> skip
            # Only fill where result is still NaN (Birch takes priority)
            result = np.where(np.isfinite(result), result,
                              np.where(np.isfinite(lake_mask), lake_mask, result))

        # -- Priority 3: SAR low-backscatter proxy -----------------------------
        # Radar-dark (low sigma0) surfaces are consistent with liquid or smooth
        # organic material.  Used only where no better source is available.
        if "sar_mosaic" in stack:
            sar: np.ndarray = stack["sar_mosaic"].values.astype(np.float32)
            sar_norm: np.ndarray = normalise_to_0_1(
                sar, percentile_lo=1, percentile_hi=99
            )
            # Invert: low backscatter -> high liquid probability
            sar_liquid: np.ndarray = (1.0 - sar_norm).astype(np.float32)
            result = np.where(
                np.isfinite(result), result,
                np.where(np.isfinite(sar_liquid), sar_liquid, nan),
            )

        return result.astype(np.float32)

    # -- Feature 2 -------------------------------------------------------------

    def _organic_abundance(
        self, stack: xr.Dataset, nan: np.ndarray
    ) -> np.ndarray:
        """
        Surface organic compound (tholin) abundance proxy.

        Data sources and priority
        -------------------------
        1. **VIMS+ISS mosaic -- primary (where available).**
           Titan_VIMS-ISS.tif Band 1 (1.59/1.27 umm band ratio plus ISS 938 nm;
           Seignovert et al. 2019, CaltechDATA doi:10.22002/D1.1173).
           This is the spectroscopically established tholin abundance proxy
           (Le Mouelic et al. 2019; Barnes et al. 2007).
           Coverage: ~50 % of the globe (roughly 0-180  degW).

        2. **Geomorphology-based scores -- gap-fill (global coverage).**
           Where VIMS data is absent (180-360  degW and polar regions), organic
           abundance is assigned from the terrain-class lookup table
           :data:`TERRAIN_ORGANIC_SCORES`, which maps each Lopes et al. (2019)
           geomorphologic class to a published organic abundance value derived
           from VIMS spectral studies.

           This approach is preferred over using the ISS 938 nm broadband
           mosaic as a gap-filler because:

           (a) VIMS band ratio and ISS broadband reflectance are physically
               different quantities (factor ~3000 different in raw units).
               No normalisation fully eliminates the resulting seam.

           (b) The geomorphology map is derived from Cassini SAR (not optical)
               and covers 100 % of the globe at consistent resolution.

           (c) Published VIMS spectral studies of each terrain class (see
               :data:`TERRAIN_ORGANIC_SCORES`) provide scientifically
               defensible organic abundance priors.

        3. **VIMS coverage density -- weak fallback (no spectral information).**
           Used only when neither VIMS mosaic nor geomorphology are available.

        Scientific basis
        ----------------
        Tholins are ubiquitous on Titan's surface and the primary substrate
        for prebiotic chemistry (Cable et al. 2012; Meyer-Dombard et al. 2025).
        Dunes (~17 % of Titan) and plains (~65 %) are tholin-dominated;
        mountains and craters expose underlying water ice (Solomonidou 2018).
        Lakes/seas (~1.5 %) are liquid hydrocarbons, not solid organics.

        References
        ----------
        Seignovert et al. (2019)    doi:10.22002/D1.1173
        Le Mouelic et al. (2019)    doi:10.1016/j.icarus.2018.09.017
        Barnes et al. (2007)        doi:10.1016/j.icarus.2006.08.021
        Lopes et al. (2020)         doi:10.1038/s41550-019-0917-6
        Cable et al. (2012)         doi:10.1089/ast.2011.0740
        Meyer-Dombard et al. (2025) Titan After Cassini-Huygens Ch.14
        Solomonidou et al. (2018)   doi:10.1029/2018JE005536

        Parameters
        ----------
        stack:
            Canonical xarray Dataset.  Expected layers:
            ``vims_mosaic``, ``geomorphology``, ``vims_coverage`` (optional).
        nan:
            All-NaN fallback array of shape (nrows, ncols), dtype float32.

        Returns
        -------
        np.ndarray
            float32 array of shape (nrows, ncols) with values in [0, 1].
            NaN where all data sources are absent.
        """
        # Announce available data sources so the chosen path is visible in logs
        available = [k for k in ("vims_mosaic", "geomorphology", "iss_mosaic_450m",
                                  "vims_coverage") if k in stack]
        logger.info("organic_abundance: available layers = %s", available)

        # -- Step 1: build geomorphology-based organic map (global, 100% cov) --
        # This is precomputed regardless of VIMS availability so it can be
        # used as a gap-filler in the VIMS-absent regions.
        geo_organic: Optional[np.ndarray] = None
        if "geomorphology" in stack:
            geo_raw: np.ndarray = stack["geomorphology"].values
            # Ensure integer class labels -- geomorphology raster is int32
            geo_int32: np.ndarray = np.where(
                np.isfinite(geo_raw.astype(np.float32)),
                geo_raw,
                0,
            ).astype(np.int32)
            geo_organic = _geo_class_to_organic(geo_int32)
            logger.debug(
                "organic_abundance: geo-based map built "
                "(valid=%.1f%%, classes=%s)",
                100.0 * np.isfinite(geo_organic).mean(),
                sorted(np.unique(geo_int32).tolist()),
            )

        # -- Step 2: normalise VIMS mosaic (primary spectral proxy) ----------
        vims_norm: Optional[np.ndarray] = None
        if "vims_mosaic" in stack:
            vims_raw: np.ndarray = stack["vims_mosaic"].values.astype(np.float32)
            # Normalise to [0, 1] using robust percentile range.
            # High band ratio -> high tholin abundance -> high score.
            vims_norm = normalise_to_0_1(vims_raw, percentile_lo=2, percentile_hi=98)
            logger.debug(
                "organic_abundance: VIMS normalised "
                "(valid=%.1f%%)",
                100.0 * np.isfinite(vims_norm).mean(),
            )

        # -- Step 3: combine -- VIMS primary, row-wise locally-offset geo fill ------
        #
        # The seam arises because VIMS values vary LOCALLY near the boundary.
        # At 170-180 degW (Shangri-La/Xanadu), VIMS is ~0.23 while the global
        # calibrated geo score for that terrain class is ~0.38 -- a +0.14 step.
        # Global class calibration cannot fix this: the class mean is computed
        # over all of 0-180 degW but the boundary region is locally anomalous.
        #
        # Solution: ROW-WISE LOCAL OFFSET.
        # For each row, find the last VIMS-valid pixel. The geo score in the
        # adjacent (NaN) columns is offset by (VIMS_edge - geo_edge) and this
        # offset decays smoothly to zero over a transition zone of decay_cols.
        # This guarantees pixel-level continuity at the exact VIMS edge,
        # regardless of global class statistics.
        if vims_norm is not None and geo_organic is not None:
            # Re-derive integer class labels from the geomorphology layer
            geo_raw: np.ndarray = stack["geomorphology"].values
            geo_int: np.ndarray = np.where(
                np.isfinite(geo_raw.astype(np.float32)), geo_raw, 0
            ).astype(np.int32)

            # Step 3a: published scores with a single global level shift
            #
            # We do NOT calibrate per-class from the VIMS overlap zone because
            # the boundary region (170-180 degW) is locally atypical: Shangri-La
            # dunes there score low in VIMS, pulling the calibrated dune score
            # down to 0.29 -- BELOW plains (0.54) -- which inverts the correct
            # scientific ranking (dunes are the most tholin-rich terrain).
            #
            # Instead: use published TERRAIN_ORGANIC_SCORES (which correctly
            # rank dunes > plains > mountains) and shift the ENTIRE geo
            # hemisphere by a single offset to match the VIMS hemisphere MEAN.
            # This preserves relative rankings while eliminating the level seam.
            geo_from_published = _geo_class_to_organic(geo_int)  # published scores

            # Single global shift: VIMS_mean - published_geo_mean (in overlap zone)
            overlap_mask: np.ndarray = np.isfinite(vims_norm) & np.isfinite(geo_from_published)
            n_overlap = int(overlap_mask.sum())
            global_offset: float = 0.0
            if n_overlap >= 5000:
                vims_overlap_mean = float(np.mean(vims_norm[overlap_mask]))
                geo_overlap_mean  = float(np.mean(geo_from_published[overlap_mask]))
                global_offset = vims_overlap_mean - geo_overlap_mean
                logger.info(
                    "organic_abundance: global level shift = %.4f "
                    "(VIMS mean=%.4f, geo_published mean=%.4f, overlap=%d px)",
                    global_offset, vims_overlap_mean, geo_overlap_mean, n_overlap,
                )

            # Apply offset, preserving relative ranking of all terrain classes
            calibrated_geo: np.ndarray = np.clip(
                geo_from_published.astype(np.float64) + global_offset,
                0.0, 1.0,
            )
            calibrated_geo[~np.isfinite(geo_from_published)] = np.nan
            logger.info(
                "organic_abundance: published scores after global shift: "
                "dunes=%.3f plains=%.3f mountains=%.3f",
                float(np.nan_to_num(TERRAIN_ORGANIC_SCORES.get(2,0.82))) + global_offset,
                float(np.nan_to_num(TERRAIN_ORGANIC_SCORES.get(3,0.68))) + global_offset,
                float(np.nan_to_num(TERRAIN_ORGANIC_SCORES.get(5,0.25))) + global_offset,
            )

            # Step 3b: row-wise local offset to eliminate the boundary step
            # For each row, compute edge_offset = VIMS_at_last_valid_col - geo_at_that_col.
            # Apply a decaying offset to geo pixels just past the boundary:
            #   offset(c) = edge_offset x exp(-(c - c_edge) / decay_cols)
            # This makes the geo fill EXACTLY match VIMS at the boundary pixel
            # and blend smoothly to the calibrated geo value over decay_cols columns.
            nrows, ncols = vims_norm.shape
            # Transition width: ~10 deg longitude
            deg_per_col  = 360.0 / ncols
            decay_cols   = max(5, int(10.0 / deg_per_col))

            vims_valid = np.isfinite(vims_norm)

            # Build column-distance array relative to each row's VIMS edge.
            # For row r with last valid VIMS column c_edge(r):
            #   dist[r, c] = c - c_edge(r)   (0 at edge, positive going right)
            # Then: offset(r, c) = edge_offset(r) x exp(-dist / decay_cols)
            #       where edge_offset(r) = vims_norm[r, c_edge] - calibrated_geo[r, c_edge]
            # Only applied to geo-filled pixels (where vims is NaN) with dist > 0.
            last_valid_col = np.full(nrows, -1, dtype=np.int32)
            for r in range(nrows):
                rv = np.where(vims_valid[r, :])[0]
                if len(rv):
                    last_valid_col[r] = int(rv.max())

            # Per-row edge offset
            edge_offsets = np.zeros(nrows, dtype=np.float64)
            for r in np.where(last_valid_col >= 0)[0]:
                ce = last_valid_col[r]
                if np.isfinite(calibrated_geo[r, ce]) and np.isfinite(vims_norm[r, ce]):
                    edge_offsets[r] = float(vims_norm[r, ce]) - float(calibrated_geo[r, ce])

            # Vectorised decay: build (nrows, ncols) distance-from-edge array
            col_idx   = np.arange(ncols, dtype=np.float64)[np.newaxis, :]  # (1, ncols)
            c_edge_2d = last_valid_col[:, np.newaxis].astype(np.float64)    # (nrows, 1)
            dist_from_edge = col_idx - c_edge_2d                             # (nrows, ncols)
            # dist > 0 means column is to the right of (past) the edge
            decay_factor = np.where(
                dist_from_edge > 0,
                np.exp(-dist_from_edge / max(decay_cols, 1)),
                0.0,
            )
            offset_map = edge_offsets[:, np.newaxis] * decay_factor          # (nrows, ncols)

            # Apply only to geo-filled pixels (not VIMS, not NaN)
            geo_fill_mask = ~vims_valid & np.isfinite(calibrated_geo)
            adjusted_geo_v = calibrated_geo.copy()
            adjusted_geo_v[geo_fill_mask] = (
                calibrated_geo[geo_fill_mask] + offset_map[geo_fill_mask]
            )

            result: np.ndarray = np.where(
                np.isfinite(vims_norm),
                vims_norm,
                adjusted_geo_v,
            )
            result = np.clip(result, 0.0, 1.0)

            n_vims: int = int(np.sum(np.isfinite(vims_norm)))
            n_geo:  int = int(np.sum(~np.isfinite(vims_norm) & np.isfinite(adjusted_geo_v)))
            logger.info(
                "organic_abundance: VIMS primary (%d px), "
                "row-offset geo gap-fill (%d px), decay=%d cols, "
                "total valid=%.1f%%",
                n_vims, n_geo, decay_cols,
                100.0 * np.isfinite(result).mean(),
            )
            return result.astype(np.float32)

        if vims_norm is not None:
            # VIMS only -- no geomorphology available
            logger.info(
                "organic_abundance: VIMS only (no geomorphology for gap-fill), "
                "valid=%.1f%%",
                100.0 * np.isfinite(vims_norm).mean(),
            )
            return vims_norm

        if geo_organic is not None:
            # No VIMS at all -- use geomorphology scores alone
            logger.info(
                "organic_abundance: geomorphology-only fallback "
                "(no VIMS mosaic available), valid=%.1f%%",
                100.0 * np.isfinite(geo_organic).mean(),
            )
            return geo_organic

        # -- ISS gap-fill fallback (no geomorphology shapefile available) ------
        # When the Lopes geomorphology shapefiles have not been downloaded,
        # fall back to ISS 938nm broadband as a VIMS gap-filler.
        # To avoid the seam that comes from blending two different
        # normalisations, we normalise VIMS and ISS together using the
        # combined global percentile range from the overlap region, so both
        # datasets are on the same scale before combining.
        if "iss_mosaic_450m" in stack:
            iss_raw: np.ndarray = stack["iss_mosaic_450m"].values.astype(np.float64)
            # ISS: invert (dark = organic-rich)
            iss_finite = iss_raw[np.isfinite(iss_raw)]
            if len(iss_finite) > 0:
                i2  = float(np.percentile(iss_finite,  2))
                i98 = float(np.percentile(iss_finite, 98))
                iss_inv = np.clip(
                    1.0 - (iss_raw - i2) / (i98 - i2 + 1e-12), 0.0, 1.0
                )
                iss_inv[~np.isfinite(iss_raw)] = np.nan

                if vims_norm is not None:
                    # Cross-calibrate ISS to match VIMS in the overlap zone
                    overlap = np.isfinite(vims_norm) & np.isfinite(iss_inv)
                    n_ov = int(overlap.sum())
                    if n_ov > 1000:
                        # Match ISS median to VIMS median in overlap region
                        v_med = float(np.median(vims_norm[overlap]))
                        i_med = float(np.median(iss_inv[overlap]))
                        offset = v_med - i_med
                        iss_adj = np.clip(iss_inv + offset, 0.0, 1.0)
                        result = np.where(
                            np.isfinite(vims_norm), vims_norm, iss_adj
                        )
                        logger.warning(
                            "organic_abundance: geomorphology_canonical.tif not found. "
                            "Using ISS 938nm gap-fill with median-offset calibration "
                            "(offset=%.3f, overlap=%d px). "
                            "Download Lopes shapefiles from "
                            "hayesresearchgroup.com/data-products/ for better results.",
                            offset, n_ov,
                        )
                        return result.astype(np.float32)
                else:
                    # No VIMS at all -- ISS only
                    logger.warning(
                        "organic_abundance: using ISS-only (no VIMS, no geomorphology). "
                        "Download both for best results."
                    )
                    return iss_inv.astype(np.float32)

        # -- Latitude-based prior fallback -------------------------------------
        # Last resort: use a smooth latitude-based terrain prior.
        # Equatorial dune belt (~0.75), mid-lat plains (~0.65),
        # polar regions (~0.4, less tholin accumulation).
        # This at least gives sensible spatial variation and no blank regions.
        if vims_norm is not None:
            # VIMS only -- no gap-fill available; leave NaN as-is
            logger.warning(
                "organic_abundance: VIMS only, no gap-fill source available. "
                "Right half will be NaN. Download geomorphology shapefiles from "
                "hayesresearchgroup.com/data-products/ to fix this."
            )
            return vims_norm

        # -- VIMS coverage density ---------------------------------------------
        if "vims_coverage" in stack:
            logger.warning(
                "organic_abundance: using VIMS coverage density as last-resort "
                "fallback -- no spectral or geomorphology data available."
            )
            return stack["vims_coverage"].values.astype(np.float32)

        logger.warning(
            "organic_abundance: no data available (VIMS, geomorphology, ISS, or "
            "coverage). Returning all-NaN."
        )
        return nan.copy()

    # -- Feature 3 -------------------------------------------------------------

    def _acetylene_energy(
        self, stack: xr.Dataset, nan: np.ndarray
    ) -> np.ndarray:
        """
        Chemical energy availability proxy (acetylene + H2 reaction).

        Primary: SAR backscatter anomaly (surface acetylene depletion
        correlates with low backscatter in organic-rich areas).
        Secondary: topographic lows (basins accumulate organic fallout).

        Scientific basis
        ----------------
        McKay & Smith (2005) calculate DeltaG = -334 kJ/mol for:
            C2H2 + 3H2 -> 2CH4
        This exceeds the minimum energy for Earth methanogens (~42 kJ/mol).
        Strobel (2010) reports H2 downward flux ~10^25 mol/s with surface
        depletion unexplained by abiotic chemistry alone.
        Yanez et al. (2024) calculate acetylenotrophy energy 69-78 kJ/mol C,
        higher than methanogenesis (25-65 kJ/mol C).
        Both metabolisms require surface-atmosphere chemical disequilibrium,
        which is highest where organic products accumulate.

        References: McKay & Smith 2005; Strobel 2010; Yanez et al. 2024
        """
        if "sar_mosaic" in stack:
            sar = stack["sar_mosaic"].values.astype(np.float32)
            sar_norm = normalise_to_0_1(sar, percentile_lo=2, percentile_hi=98)
            # High organics (low backscatter) = more energy substrate
            energy = 1.0 - sar_norm

            # Boost in topographic depressions (organic accumulation)
            if "topography" in stack:
                dem = stack["topography"].values.astype(np.float32)
                # Invert elevation: low terrain -> high accumulation
                topo_inv = normalise_to_0_1(-dem, percentile_lo=2, percentile_hi=98)
                topo_inv = np.where(np.isfinite(topo_inv), topo_inv, 0.0)
                energy = np.where(
                    np.isfinite(energy),
                    np.clip(energy * 0.7 + topo_inv * 0.3, 0, 1),
                    # Gap-fill: where SAR is absent but topography is available,
                    # use topography-only proxy rather than returning NaN.
                    np.where(np.isfinite(topo_inv), topo_inv, nan),
                )
            return energy.astype(np.float32)

        if "topography" in stack:
            dem = stack["topography"].values.astype(np.float32)
            return normalise_to_0_1(-dem, percentile_lo=2, percentile_hi=98)

        logger.warning("No SAR or topography data for acetylene_energy.")
        return nan.copy()

    # -- Feature 4 -------------------------------------------------------------

    def _methane_cycle(
        self, stack: xr.Dataset, nan: np.ndarray
    ) -> np.ndarray:
        """
        Active methane-cycle intensity proxy.

        Components (fallback order)
        ---------------------------
        1. VIMS coverage density (primary)  -- coverage is highest where
           Cassini observed the most flybys, which correlates with
           scientifically-motivated targeting of active methane-cycle
           regions (mid-latitudes and polar lake regions).
        2. CIRS surface temperature gradient (secondary) -- regions where
           the latitudinal temperature gradient is steepest drive the
           strongest evaporation-precipitation differential and therefore
           the most active methane cycling (Tokano 2019; Jennings 2019).
           Derived from the synthesised `cirs_temperature` layer.
        3. Pure latitude prior (fallback) -- Gaussian peak at +/-45 deg,
           representing mid-latitude active transport zones.

        Scientific basis
        ----------------
        Titan's methane cycle (rain, rivers, lakes, evaporation) is the
        dominant surface process and the analog to Earth's hydrological
        cycle (Mitchell & Lora 2016). Seasonal surface temperature changes
        of 1-4 K (Jennings et al. 2019) drive differential evaporation
        between northern and southern hemispheres. Schulze-Makuch &
        Grinspoon (2005) propose biological activity could enhance the
        methane cycle via biothermal energy. Observed rainfall events
        (Turtle et al. 2011) and the global fluvial channel network
        (Miller et al. 2021) document active methane transport.

        References: Mitchell & Lora 2016; Jennings et al. 2019;
                    Tokano 2019; Turtle et al. 2011
        """
        # -- Latitude prior (always available) ----------------------------
        # Derive grid shape from the stack if possible; else fall back to
        # self.grid so the function works with any input size (including tests).
        if stack.sizes:
            nrows = stack.sizes.get("lat", self.grid.nrows)
            ncols = stack.sizes.get("lon", self.grid.ncols)
            if "lat" in stack.coords:
                lats = stack.coords["lat"].values.astype(np.float32)
            else:
                lats = self.grid.lat_centres_deg()[:nrows]
        else:
            nrows = self.grid.nrows
            ncols = self.grid.ncols
            lats = self.grid.lat_centres_deg()
        lat_grid = np.tile(lats[:, np.newaxis], (1, ncols))
        # Gaussian peak at +/-45 deg -- active methane transport zones
        lat_weight = np.exp(
            -((np.abs(lat_grid) - 45.0) / 25.0) ** 2
        ).astype(np.float32)

        # -- CIRS surface temperature gradient ----------------------------
        # Where |dT/dlat| is large, evaporation differential is strong.
        # Use the meridional gradient of the temperature raster.
        cirs_weight: Optional[np.ndarray] = None
        if "cirs_temperature" in stack:
            T = stack["cirs_temperature"].values.astype(np.float32)
            # Meridional gradient magnitude (K per pixel)
            dT_dy = np.abs(np.gradient(
                np.where(np.isfinite(T), T, np.nan), axis=0
            ))
            cirs_weight = normalise_to_0_1(
                np.where(np.isfinite(dT_dy), dT_dy, 0.0)
            )
            logger.debug("methane_cycle: cirs_temperature gradient present")

        # -- VIMS coverage -------------------------------------------------
        if "vims_coverage" in stack:
            coverage = stack["vims_coverage"].values.astype(np.float32)
            vims_norm = normalise_to_0_1(coverage, percentile_lo=0,
                                         percentile_hi=95)

            if cirs_weight is not None:
                # Three-way blend: VIMS 50%, temperature gradient 25%, lat 25%
                result = np.clip(
                    vims_norm * 0.50
                    + cirs_weight * 0.25
                    + lat_weight * 0.25,
                    0.0, 1.0,
                )
            else:
                # Two-way blend: VIMS 60%, lat 40%
                result = np.clip(
                    vims_norm * 0.60 + lat_weight * 0.40,
                    0.0, 1.0,
                )

            result = np.where(np.isfinite(result), result,
                              lat_weight * 0.40).astype(np.float32)
            return result

        # -- No VIMS -- temperature gradient or pure latitude prior ---------
        if cirs_weight is not None:
            return np.clip(
                cirs_weight * 0.60 + lat_weight * 0.40,
                0.0, 1.0,
            ).astype(np.float32)

        # Pure latitude prior fallback
        return lat_weight

    # -- Feature 5 -------------------------------------------------------------

    def _surface_atm_interaction(
        self, stack: xr.Dataset, nan: np.ndarray
    ) -> np.ndarray:
        """
        Surface-atmosphere chemical exchange intensity -- Feature 5 (weight 0.08).

        Component weights (sum to 1.0)
        -------------------------------
        topographic_slope:     0.25  -- gradient-driven runoff and exposure
        lake_margin:           0.35  -- evaporation/condensation hotspot
        paleo_lake_indicator:  0.15  -- Birch empty basins (wetting/drying cycles)
        channel_density:       0.25  -- Miller+2021 fluvial transport flux

        If any component is absent its weight is redistributed proportionally
        among the remaining components.

        Lake margin source priority
        ---------------------------
        1. **Birch+2017 exact shorelines** (``polar_lakes`` layer, labels 1 & 3).
           A dilation of the Birch/Palermo filled-lake pixels gives the precise
           lake-margin zone where evaporation, condensation, and wave action
           concentrate amphiphiles and organics.  This replaces the Lopes
           dilation (which was zero because Lakes.shp was absent).

        2. **Lopes+2019 geomorphology** (lake class = 7).
           Used as fallback outside Birch coverage (mainly non-polar pixels).

        Paleo-lake indicator (new component, requires Birch data)
        ----------------------------------------------------------
        Birch+2017 empty-basin pixels (``polar_lakes`` label 2) represent
        areas that were periodically inundated and drained -- a wetting/drying
        cycle that:
          (a) concentrates amphiphilic molecules at the shoreline, creating
              the conditions proposed by Mayer & Nixon (2025) for protocell
              vesicle formation;
          (b) deposits organic evaporite layers with elevated prebiotic
              chemistry potential;
          (c) may maintain subsurface brine pockets providing sustained
              liquid-phase chemistry.

        When the Birch raster is absent this component returns zero (no
        paleo-lake signal available), reducing to the pre-Birch behaviour.

        Scientific basis
        ----------------
        Surface-atmosphere exchange is most intense at lake margins, channel
        networks, and topographic breaks (Hayes 2016; Miller et al. 2021;
        Lorenz et al. 2013).  These zones are analogous to Earth's estuaries --
        the most biologically productive per unit area.  The Mayer & Nixon
        (2025) protocell mechanism specifically requires the lake-surface /
        rain-splash interface.

        Parameters
        ----------
        stack:
            Canonical data stack (xarray Dataset).
        nan:
            Template NaN array of shape (nrows, ncols).

        Returns
        -------
        np.ndarray
            float32 array in [0, 1].  NaN where no input layers are available.

        References
        ----------
        Birch et al. (2017) Icarus doi:10.1016/j.icarus.2017.01.032
        Hayes (2016) Ann. Rev. Earth Planet. Sci. doi:10.1146/annurev-earth-060115-012247
        Lorenz et al. (2013) Icarus doi:10.1016/j.icarus.2013.04.002
        Mayer & Nixon (2025) Int. J. Astrobiology doi:10.1017/S1473550425100037
        Miller et al. (2021) JGR Planets doi:10.1029/2021JE006955
        Birch et al. (2017) Icarus doi:10.1016/j.icarus.2017.01.032
        """
        from titan.io.shapefile_rasteriser import (
            POLAR_LAKE_FILLED, POLAR_LAKE_PALERMO, POLAR_LAKE_EMPTY,
        )

        # -- Component weights -------------------------------------------------
        W_SLOPE:   float = 0.25
        W_MARGIN:  float = 0.35
        W_PALEO:   float = 0.15   # Birch empty-basin component (new)
        W_CHANNEL: float = 0.25

        components: List[np.ndarray] = []
        weights:    List[float]      = []

        # -- 1. Topographic slope ----------------------------------------------
        if "topography" in stack:
            dem: np.ndarray = stack["topography"].values.astype(np.float64)
            dy, dx = np.gradient(np.where(np.isfinite(dem), dem, 0.0))
            slope_f64: np.ndarray = np.sqrt(dx**2 + dy**2)
            slope: np.ndarray = np.clip(
                slope_f64, 0.0, np.finfo(np.float32).max
            ).astype(np.float32)
            slope_norm: np.ndarray = normalise_to_0_1(slope)
            components.append(slope_norm)
            weights.append(W_SLOPE)

        # -- 2. Lake margins ---------------------------------------------------
        # Priority: Birch/Palermo exact shorelines -> Lopes geo class -> none
        from scipy.ndimage import binary_dilation

        # Margin width: ~3 pixels = ~13 km, consistent with lake-margin
        # evaporation/condensation zone from Hayes (2016)
        _MARGIN_DILATION_ITER: int = 3

        lake_margin: Optional[np.ndarray] = None
        labyrinth_component: Optional[np.ndarray] = None

        if "polar_lakes" in stack:
            # Birch/Palermo confirmed liquid pixels -> dilate to get margin zone
            pl: np.ndarray = stack["polar_lakes"].values.astype(np.int16)
            birch_liquid_mask: np.ndarray = (
                (pl == POLAR_LAKE_FILLED) | (pl == POLAR_LAKE_PALERMO)
            ).astype(bool)

            if birch_liquid_mask.any():
                margin_with_lake: np.ndarray = binary_dilation(
                    birch_liquid_mask, iterations=_MARGIN_DILATION_ITER
                )
                birch_margin: np.ndarray = (
                    margin_with_lake & ~birch_liquid_mask
                ).astype(np.float32)
                lake_margin = birch_margin
                logger.debug(
                    "surface_atm_interaction: using Birch shoreline for "
                    "lake margin (%d margin pixels)",
                    int(birch_margin.sum()),
                )

        if lake_margin is None and "geomorphology" in stack:
            # Fallback: Lopes lake class + labyrinth
            geo: np.ndarray = stack["geomorphology"].values.astype(np.float32)
            lopes_lake_mask: np.ndarray = (geo == 7.0).astype(bool)
            if lopes_lake_mask.any():
                lopes_margin_with_lake: np.ndarray = binary_dilation(
                    lopes_lake_mask, iterations=_MARGIN_DILATION_ITER
                )
                lake_margin = (
                    lopes_margin_with_lake & ~lopes_lake_mask
                ).astype(np.float32)
                logger.debug(
                    "surface_atm_interaction: using Lopes geo==7 for "
                    "lake margin (Birch not available)"
                )
            labyrinth: np.ndarray = (geo == 6.0).astype(np.float32)
            labyrinth_component = labyrinth

        if lake_margin is not None:
            # Combine lake margin with labyrinth terrain (if available)
            if labyrinth_component is not None:
                margin_combined: np.ndarray = np.clip(
                    lake_margin + labyrinth_component * 0.7, 0.0, 1.0
                )
            else:
                margin_combined = lake_margin
            components.append(margin_combined)
            weights.append(W_MARGIN)
        else:
            logger.debug(
                "surface_atm_interaction: no lake margin data available "
                "(no polar_lakes or geomorphology layer)"
            )

        # -- 3. Paleo-lake indicator (Birch empty basins) ----------------------
        # Empty basins score proportionally to the fraction of the local
        # neighbourhood that is empty-basin pixels.  This captures the
        # cumulative wetting/drying cycle history of the area.
        if "polar_lakes" in stack:
            pl_int: np.ndarray = stack["polar_lakes"].values.astype(np.int16)
            empty_mask: np.ndarray = (pl_int == POLAR_LAKE_EMPTY).astype(np.float32)
            if empty_mask.any():
                from scipy.ndimage import uniform_filter
                # Smooth over ~45 km (~10 px) to capture proximity to basins
                paleo_indicator: np.ndarray = normalise_to_0_1(
                    uniform_filter(empty_mask, size=10, mode="constant", cval=0.0)
                )
                components.append(paleo_indicator)
                weights.append(W_PALEO)
                logger.debug(
                    "surface_atm_interaction: paleo_lake_indicator "
                    "(%d empty-basin pixels -> smoothed to %d non-zero)",
                    int(empty_mask.sum()),
                    int((paleo_indicator > 0).sum()),
                )
            # else: Birch present but no empty pixels -> skip this component

        # -- 4. Fluvial channel density (Miller et al. 2021) -------------------
        if "channel_density" in stack:
            ch: np.ndarray = stack["channel_density"].values.astype(np.float32)
            ch_norm: np.ndarray = normalise_to_0_1(ch)
            components.append(ch_norm)
            weights.append(W_CHANNEL)
            logger.debug("surface_atm_interaction: channel_density layer present")
        else:
            logger.debug(
                "surface_atm_interaction: channel_density absent -- "
                "weight redistributed among available components."
            )

        if not components:
            logger.warning(
                "surface_atm_interaction: no input layers available -- "
                "returning NaN array."
            )
            return nan.copy()

        # -- Weighted blend ----------------------------------------------------
        total_w: float = sum(weights)
        norm_w: List[float] = [w / total_w for w in weights]
        stacked: np.ndarray = np.stack(
            [np.where(np.isfinite(c), c, 0.0) * w
             for c, w in zip(components, norm_w)],
            axis=0,
        )
        # Only report a value where at least one component has data
        valid_mask: np.ndarray = np.zeros(nan.shape, dtype=bool)
        for c in components:
            valid_mask |= np.isfinite(c)

        result: np.ndarray = np.sum(stacked, axis=0)
        result[~valid_mask] = np.nan
        return result.astype(np.float32)

    # -- Feature 6 -------------------------------------------------------------

    def _topographic_complexity(
        self, stack: xr.Dataset, nan: np.ndarray
    ) -> np.ndarray:
        """
        Local terrain roughness from the GTDR DEM.

        Scientific basis
        ----------------
        Complex topography increases the number of distinct micro-environments
        and chemical gradient opportunities. Hummocky and labyrinth terrain
        show highest roughness and highest water-ice content via VIMS
        (Lopes et al. 2019; Malaska et al. 2025). Mountains on Titan reach
        ~1-2 km height and show radar-bright signatures consistent with
        exposed water ice (Radebaugh et al. 2007).

        References: Lorenz et al. 2013; Lopes et al. 2019
        """
        if "topography" in stack:
            dem = stack["topography"].values.astype(np.float32)
            return compute_topographic_roughness(dem, window_radius=5)

        logger.warning("No topography data for topographic_complexity.")
        return nan.copy()

    # -- Feature 7 -------------------------------------------------------------

    def _geomorphologic_diversity(
        self, stack: xr.Dataset, nan: np.ndarray
    ) -> np.ndarray:
        """
        Shannon diversity of terrain classes in a local neighbourhood.

        Scientific basis
        ----------------
        Ecotone effect: boundaries between different terrain types host the
        highest biodiversity on Earth (Wilson 1992). On Titan, terrain
        boundaries correspond to transitions between organic-rich dunes,
        water-ice-rich mountains, and liquid-bearing lake margins -- each
        providing different chemical substrates and energy sources.
        Pixels at terrain boundaries thus represent the most chemically
        diverse micro-environments.
        Six terrain units from Lopes et al. (2019) + lakes from Hayes et al.

        References: Lopes et al. 2019; Malaska et al. 2025 Ch.9
        """
        if "geomorphology" in stack:
            geo = stack["geomorphology"].values.astype(np.float32)
            # Replace NaN with 0 before casting to int32 to suppress
            # "invalid value encountered in cast" RuntimeWarning
            geo_clean = np.where(np.isfinite(geo), geo, 0.0)
            return compute_terrain_diversity(
                geo_clean.astype(np.int32), n_classes=7, window_radius=7
            )

        logger.warning("No geomorphology data for geomorphologic_diversity.")
        return nan.copy()

    # -- Feature 8 -------------------------------------------------------------

    def _subsurface_ocean(
        self, stack: xr.Dataset, nan: np.ndarray
    ) -> np.ndarray:
        """
        Subsurface ocean / past-liquid-water exchange proxy.

        Base probability (D3)
        ---------------------
        The global base probability is ``self.subsurface_ocean_base_prior``
        (default 0.03, configurable).  This is intentionally lower than the
        raw k2 normalisation (0.589/1.5 =~ 0.39) because the k2 measurement
        tells us the ocean EXISTS, not that it communicates with the surface
        under current conditions.  The effective surface habitability from
        this pathway is presently low due to:

          - Time elapsed since D1 past epoch (~3.5 Gya, configurable)
            when widespread liquid water last existed at the surface.
          - The D2 future window (~100-400 Myr from now, configurable)
            has not yet been reached.
          - Under the D2 uniform warming assumption, the present-day
            contribution of future-window habitability to the per-pixel
            score is zero (we are not yet inside the window).

        SAR bright annuli -- past liquid water proxy (D4)
        --------------------------------------------------
        Radar-bright RING STRUCTURES (annuli) in Cassini RADAR SAR imagery
        are interpreted as a proxy for PAST liquid water-organic contact.

        Rationale (accepted):
        The bright annular morphology (ring/horseshoe shape, distinct from
        diffuse bright terrain) is consistent with:
          (a) Impact melt pond RIMS -- when a bolide strikes Titan, the
              impactor and target material briefly melt, forming a liquid
              water pool that interacts with the organic-rich surface.
              As the melt freezes, it leaves a bright radar-reflective ring
              at the flow front.  This interaction is the primary mechanism
              by which terrestrial amino acid precursors could have formed
              on early Titan (Neish et al. 2018; Artemieva & Lunine 2003).
          (b) Cryovolcanic flow FRONTS -- viscous cryovolcanic fluid
              (water-ammonia slurry) flowing from volcanic vents forms
              bright flow-front annuli as the advancing lobe cools and
              roughens (Lopes et al. 2007, 2013; Wood et al. 2010).
          (c) NOT simply diffuse bright terrain -- high SAR backscatter
              is found in many contexts (exposed water ice, rough dunes,
              mountains).  The key discriminator for this proxy is the
              ANNULAR morphology, which is detectable in SAR as a local
              bright ring surrounding a darker interior.

        Implementation note:
        Full annulus detection requires morphological ring-detection
        (e.g., Hough transform for circles, or band-pass radial filtering).
        As a first-order approximation, we detect pixels that are locally
        bright RELATIVE TO THEIR NEIGHBOURHOOD using a ring-shaped
        structuring element (bright shell around a dark core).  This
        captures the qualitative annular signature without requiring
        full geometric ring fitting.

        Temporal modulation
        -------------------
        The feature is NOT modulated by the D2 future window at present
        (since uniform warming is global and adds no spatial information).
        The D1 past-epoch parameter is used only as a documentation anchor;
        its numerical effect is captured in the low base prior (0.03).
        Future work: apply a spatially varying temporal decay based on
        crater age estimates (cratering chronology models).

        Scientific basis
        ----------------
        Iess et al. (2012) k2=0.589+/-0.150 -> global subsurface ocean confirmed.
        Affholder et al. (2025) -> glycine fermentation viable in ocean.
        Lopes et al. (2007, 2013) -> cryovolcanic SAR-bright features.
        Neish et al. (2018) -> impact melt liquid water + organics -> amino acids.
        Wood et al. (2010) -> SAR-bright annular flow fronts.
        Artemieva & Lunine (2003) -> impact melt volumes and lifetimes.

        References:
        Iess et al. (2012)          doi:10.1126/science.1219631
        Affholder et al. (2025)     doi:10.3847/PSJ/addb09
        Lopes et al. (2007)         doi:10.1016/j.icarus.2007.06.022
        Neish et al. (2018)         doi:10.1089/ast.2017.1758
        Artemieva & Lunine (2003)   doi:10.1016/S0019-1035(02)00039-9
        Wood et al. (2010)          Icarus 206, 334-344
        """
        # -- Base probability (D3: configurable, default 0.03) -----------------
        # k2=0.589 (Iess 2012) CONFIRMS ocean presence but does not constrain
        # surface exchange probability.  The low base prior reflects time-gating
        # by D1 (~3.5 Gya past epoch) and D2 (future window not yet reached).
        base_prob: float = float(np.clip(self.subsurface_ocean_base_prior, 0.0, 1.0))
        result = np.full(
            (self.grid.nrows, self.grid.ncols), base_prob, dtype=np.float32
        )

        # -- SAR bright annuli proxy (D4) --------------------------------------
        # Detect pixels that form BRIGHT RINGS around darker interiors.
        # Morphological approach: a pixel scores high if it is locally brighter
        # than its far neighbourhood but not its immediate neighbourhood.
        # This approximates the annular (ring) morphology of impact melt rims
        # and cryovolcanic flow fronts without full geometric ring detection.
        if "sar_mosaic" in stack:
            from scipy.ndimage import uniform_filter

            sar = stack["sar_mosaic"].values.astype(np.float32)
            sar_valid = np.where(np.isfinite(sar), sar, 0.0)
            sar_norm  = normalise_to_0_1(sar_valid, percentile_lo=2, percentile_hi=98)

            # Local mean in a small window (inner: ~5 px radius = potential bright rim)
            inner_mean = uniform_filter(sar_norm, size=5)
            # Local mean in a larger window (outer: ~25 px = surrounding terrain)
            outer_mean = uniform_filter(sar_norm, size=25)

            # Annular signal: pixel is brighter than outer neighbourhood
            # but the outer neighbourhood is darker than the global median.
            # This captures "bright rim around a relatively dark interior."
            global_median = float(np.nanmedian(sar_norm[np.isfinite(sar_norm)]))
            annular_signal = np.clip(
                (inner_mean - outer_mean)         # bright rim vs surroundings
                * (global_median - outer_mean),   # surroundings darker than median
                0.0, None,
            )
            # Normalise annular signal to [0, 1]
            annular_norm = normalise_to_0_1(annular_signal, percentile_lo=50, percentile_hi=99)
            annular_norm = np.where(np.isfinite(annular_norm), annular_norm, 0.0)

            # Boost pixels with strong annular signature above the base prior.
            # Maximum boost: +0.30 above base_prob (capped at 1.0).
            # Interpretation: a strong annular SAR signature roughly triples
            # the local probability relative to the global base.
            boost = annular_norm * 0.30
            result = np.clip(result + boost, 0.0, 1.0).astype(np.float32)

            n_boosted = int(np.sum(boost > 0.05))
            logger.info(
                "  subsurface_ocean: %d pixels boosted by SAR bright annuli "
                "(D4 proxy; base_prior=%.3f, past_epoch=%.1f Gya, "
                "future_window=%d-%d Myr%s)",
                n_boosted,
                base_prob,
                self.window.past_liquid_water_epoch_gya,
                int(self.window.future_window_min_myr),
                int(self.window.future_window_max_myr),
                ", uniform warming assumed" if self.window.assume_uniform_warming else "",
            )
        else:
            logger.info(
                "  subsurface_ocean: SAR not available; using uniform "
                "base_prior=%.3f (D3). "
                "D4 annular proxy disabled. D1=%.1f Gya, D2=%d-%d Myr.",
                base_prob,
                self.window.past_liquid_water_epoch_gya,
                int(self.window.future_window_min_myr),
                int(self.window.future_window_max_myr),
            )

        return result.astype(np.float32)
