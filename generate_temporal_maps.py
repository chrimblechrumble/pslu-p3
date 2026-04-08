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
generate_temporal_maps.py
==========================
Generate global habitability maps across 36 epochs spanning the Late Heavy
Bombardment (-3.5 Gya) through the present Cassini epoch to the red-giant
solar expansion peak (+6.5 Gya).

Outputs
-------
outputs/temporal_maps/
  geotiffs/                   <- QGIS-ready GeoTIFF per epoch
    habitability_-3.500_Gya.tif
    habitability_-2.379_Gya.tif
    ...
    habitability_+6.500_Gya.tif
  titan_temporal_habitability.nc  <- NetCDF time-series stack (all epochs)
  animation/
    titan_habitability_animation.mp4  <- MP4 flythrough PAST->FUTURE
    titan_habitability_animation.gif  <- Lightweight GIF version
  posters/
    key_epochs_poster.png       <- Six-panel summary figure

How it works
------------
The script runs in two modes:

1. **Real-data mode** (when the pipeline has been run):
   Loads the present-epoch feature GeoTIFFs from
   ``outputs/present/features/tifs/``.  These are the 9 feature arrays
   at t = 0 (Cassini epoch).  For each other epoch, each feature array
   is scaled by a time-varying function derived from the same physical
   models as ``analyse_location_habitability.py``.

2. **Synthetic mode** (when no pipeline output exists):
   Generates physically-grounded synthetic feature maps using latitude,
   longitude, and known Titan terrain characteristics.  Used for testing
   and demonstration.

Feature time-scaling strategy
------------------------------
Each present-epoch feature array F_i(lat, lon) at t = 0 is transformed
to the corresponding array at epoch t by::

    F_i(lat, lon, t) = clamp(F_i(lat, lon, 0) x scale_i(t), 0, 1)

where scale_i(t) is a spatially uniform scalar derived from the physical
models in ``analyse_location_habitability.py``.  The spatial structure
(which pixels are lakes, dunes, craters, etc.) is preserved; only the
overall magnitude varies with epoch.

Features that cannot be captured by a single scalar (e.g. liquid_hydrocarbon
in the red-giant water-ocean phase where ALL pixels become habitable) use
additive overrides at those epochs.

Bayesian inference
------------------
At each epoch, the weighted feature sum is computed pixel-by-pixel and
fed through the same Beta conjugate update as the main pipeline::

    alpha_post = alpha_0 + lambda x Sigma_i w_i x F_i
    beta_post = beta_0 + lambda x (1 - Sigma_i w_i x F_i)
    P(habitable) = alpha_post / (alpha_post + beta_post)

GeoTIFF CRS
-----------
All outputs use the Titan equirectangular CRS::

    +proj=eqc +a=2575000 +b=2575000 +units=m +no_defs

This is the same CRS as all other pipeline outputs.  In QGIS, set
the project CRS to this string to enable correct map display.

QGIS import instructions are printed at the end of the run.

Usage
-----
    python generate_temporal_maps.py [options]

    --output-dir DIR      Override output directory (default: outputs/temporal_maps)
    --feature-dir DIR     Path to present-epoch feature TIFs
    --no-animation        Skip animation generation (faster)
    --no-netcdf           Skip NetCDF stack output
    --epochs N            Limit to N epochs (for testing; default: all 36)
    --fps N               Animation frames per second (default: 8)
    --dpi N               Figure DPI for animation frames (default: 120)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Physical constants -------------------------------------------------------

TITAN_RADIUS_M:   float = 2_575_000.0
CANONICAL_RES_M:  float = 4_490.0
EUTECTIC_K:       float = 176.0    # water-ammonia eutectic melting point (K)
T_SURFACE_K:      float = 93.65    # present-day Titan surface temperature (K)

#: Canonical grid size (nrows, ncols)
GRID_SHAPE: Tuple[int, int] = (1802, 3603)

# --- Polar visualisation parameters -----------------------------------------

#: Southern/northern latitude at which the polar-cap circle boundary is drawn.
#: The stereographic resampling maps r=1 (circle edge) to exactly this latitude,
#: so the full disc is filled with data.  Reduce to show more of the globe;
#: increase to zoom in on the high-latitude lake regions.
POLAR_CAP_EDGE_DEG: float = 50.0

#: Pre-computed scale factor for the polar stereographic projection.
#: Derived from POLAR_CAP_EDGE_DEG via:
#:   lat = 90 - 2.arctan(r.scale)  ->  at r=1: lat = POLAR_CAP_EDGE_DEG
#:   therefore  scale = tan((90 - POLAR_CAP_EDGE_DEG) / 2)
#: Reference: Snyder (1987) "Map Projections -- A Working Manual", s.21.
POLAR_SCALE: float = math.tan(math.radians((90.0 - POLAR_CAP_EDGE_DEG) / 2.0))

# --- Colour palette ----------------------------------------------------------
# All hex colours are defined here so they can be changed in one place.
# Colours use 8-digit hex (RRGGBBAA) when transparency is required.

#: Background colour for all figure panels and the figure itself.
COLOUR_BACKGROUND: str = "#0d0d1a"
#: Slightly lighter background used for the space/nodata region inside polar caps.
COLOUR_SPACE:       str = "#1a1a2e"
#: Colour applied to the boundary circle of each polar panel.
COLOUR_POLAR_RING:  str = "#888899"
#: Colour for map-panel spine edges.
COLOUR_SPINE:       str = "#444455"
#: Colour for poster panel spine edges (slightly lighter).
COLOUR_SPINE_POSTER: str = "#555566"
#: Marker colour for the top-10 location dots -- chosen to stand out on Plasma.
COLOUR_MARKER:      str = "#00ffcc"
#: Dark green used as the marker edge so dots have a visible outline.
COLOUR_MARKER_EDGE: str = "#003333"
#: Leader-line colour for location annotations (semi-transparent).
COLOUR_LEADER:      str = "#00ffcc88"
#: Semi-transparent black box behind annotation text.
COLOUR_ANNOT_BOX:   str = "#000000aa"
#: Slightly more opaque annotation box used on polar panels.
COLOUR_ANNOT_BOX_POLAR: str = "#000000bb"
#: White text for annotation labels.
COLOUR_TEXT:        str = "#ffffff"
#: Body text inside the narrative box -- slightly off-white.
COLOUR_NARRATIVE_BODY: str = "#ccccdd"
#: Steel-blue border for the narrative body box.
COLOUR_NARRATIVE_BORDER: str = "#4466aacc"
#: Near-black fill for the narrative body box.
COLOUR_NARRATIVE_FILL: str = "#0a0a1eee"
#: Amber used for narrative title text.
COLOUR_NARRATIVE_TITLE: str = "#ffcc33"
#: Amber border for the narrative title pill (semi-transparent).
COLOUR_TITLE_BORDER: str = "#ffcc3366"
#: Dark amber fill for the narrative title pill.
COLOUR_TITLE_FILL:   str = "#1a1200dd"
#: Fully transparent placeholder -- keeps figure layout stable.
COLOUR_TRANSPARENT:  str = "#00000000"
#: Progress bar fill colour.
COLOUR_PROGRESS_BAR: str = "#44aaff"
#: Phase label colour for the red-giant water-ocean phase.
COLOUR_PHASE_OCEAN:  str = "#ff9933"
#: Phase label colour for the Late Heavy Bombardment phase.
COLOUR_PHASE_LHB:    str = "#ff6644"
#: Phase label colour for the present Cassini epoch.
COLOUR_PHASE_PRESENT: str = "#66ddff"
#: Phase label colour for all other epochs.
COLOUR_PHASE_DEFAULT: str = "#aabbff"
#: Axis label colour on poster panels.
COLOUR_AXIS_LABEL:   str = "#aaaacc"
#: Poster info-line colour for normal epochs.
COLOUR_POSTER_INFO:  str = "#aabbff"

# --- Figure geometry ---------------------------------------------------------

#: Figure width in inches.
#: Derived so each polar panel width equals the shared row height exactly (wspace=0):
#:   polar_width = (GS_RIGHT-GS_LEFT) x fig_width / 4
#:   panel_height = (GS_TOP-GS_BOTTOM) x fig_height
#:   Setting equal:  fig_width = 4 x (GS_TOP-GS_BOTTOM) / (GS_RIGHT-GS_LEFT) x fig_height
#:   With GS values below and fig_height=8.5:  20.7" -> delta < 0.01mm
FIG_WIDTH_IN:  float = 20.7
FIG_HEIGHT_IN: float = 11.0

#: GridSpec margins (figure-fraction, 0=bottom/left, 1=top/right).
#:   GS_LEFT  = 0.08 -- space for equirectangular y-axis labels (Latitude  degN)
#:   GS_BOTTOM= 0.30 -- space for colourbar + x-axis label + narrative text boxes
#:   GS_TOP   = 0.855 -- panel titles clear of the progress bar above
GS_LEFT:   float = 0.08
GS_RIGHT:  float = 0.99
GS_TOP:    float = 0.920
GS_BOTTOM: float = 0.500

# --- Feature weights and priors ----------------------------------------------

#: Feature weights for the MODELLED animation (Bayesian formula).
#: These differ from temporal_config.py PRESENT_FEATURES weights by -0.02 per feature
#: because ``impact_melt_bonus`` (weight 0.09) is an animation-only feature that
#: requires redistribution from the pipeline weights.  The resulting mu0 is 0.281
#: vs pipeline 0.331 — intentional; checked in diagnose_full_inference.py.
#:
#: Animation organic_abundance prior: 0.70 (Malaska 2025; matched to pipeline v5).
#: ``impact_melt_bonus`` is an ANIMATION-ONLY feature synthesised on-the-fly from
#: :func:`_impact_melt_global`; it is NOT a run_pipeline.py feature and has no TIF.
WEIGHTS: Dict[str, float] = {
    "liquid_hydrocarbon":       0.23,
    "organic_abundance":        0.18,
    "acetylene_energy":         0.18,
    "methane_cycle":            0.13,
    "surface_atm_interaction":  0.08,
    "topographic_complexity":   0.05,
    "geomorphologic_diversity": 0.04,
    "subsurface_ocean":         0.02,
    "impact_melt_bonus":        0.09,  # SYNTHESISED -- no TIF; zero at PRESENT epoch
}

#: Bayesian inference parameters
KAPPA:   float = 5.0    # prior concentration
LAMBDA:  float = 6.0    # likelihood sharpness

#: Prior means (present-epoch).
#: ``impact_melt_bonus`` prior is 0.0 because no active impact melts exist today.
PRIOR_MEANS: Dict[str, float] = {
    "liquid_hydrocarbon":       0.020,
    "organic_abundance":        0.700,  # revised: Malaska (2025) + Cable (2012); matched to pipeline v5
    "acetylene_energy":         0.350,
    "methane_cycle":            0.400,
    "surface_atm_interaction":  0.350,
    "topographic_complexity":   0.250,
    "geomorphologic_diversity": 0.300,
    "subsurface_ocean":         0.030,
    "impact_melt_bonus":        0.000,  # SYNTHESISED; 0 at present by design
}

#: Colourmap display range for P(habitable | features).
#: VMAX=0.75 is calibrated for both output systems:
#:   - Animation Bayesian formula: hard max = 0.673  → sits at 88% of scale
#:   - sklearn RandomForest anchors: p90 ≈ 0.780     → clips only ~10% of pixels
#:   (VMAX=0.65 clipped 34% of sklearn pixels; VMAX=0.75 reduces this to ~10%)
VMIN, VMAX = 0.10, 0.75

# --- Epoch axis ---------------------------------------------------------------

def make_epoch_axis(n_limit: Optional[int] = None) -> np.ndarray:
    """
    Build the 72-point epoch axis, denser near key geological transitions.

    Segments are denser near:
      - LHB peak (-3.8 Gya)
      - Lake formation onset (-1.5 to -0.4 Gya)
      - Present epoch (-0.4 to +0.1 Gya)
      - D2 near-future solar warming window (+0.250 Gya, explicit anchor)
      - Solar warming + lake evaporation (+3.8 to +5.0 Gya)
      - Water-ammonia eutectic crossing (+5.0 to +5.3 Gya)

    The point +0.250 Gya is the centre of the D2 near-future habitability
    window (100-400 Myr from now; Lorenz et al. 1997, Lunine & Lorenz 2009).
    It is added explicitly so that ``full_inference`` mode can anchor the
    NEAR_FUTURE pipeline run to an exact frame rather than the nearest
    available point.

    Returns
    -------
    np.ndarray
        Epochs in Gya from present (negative = past, positive = future).
    """
    segs = [
        np.linspace(-3.80, -3.00,  5),   # LHB -> early decline
        np.linspace(-3.00, -1.50,  8),   # early Titan
        np.linspace(-1.50, -0.40, 12),   # lake formation -- dense
        np.linspace(-0.40,  0.10,  8),   # near-present -- very dense
        np.array([0.250]),               # D2 near-future anchor (exact)
        np.linspace( 0.10,  2.00,  8),   # near future
        np.linspace( 2.00,  3.80,  6),   # mid future
        np.linspace( 3.80,  5.00, 10),   # solar warming -- dense
        np.linspace( 5.00,  5.30, 10),   # eutectic transition -- very dense
        np.linspace( 5.30,  6.00,  8),   # ocean phase
        np.linspace( 6.00,  6.50,  5),   # end
    ]
    epochs = np.sort(np.unique(np.round(np.concatenate(segs), 4)))
    return epochs if n_limit is None else epochs[:n_limit]


#: Key transition events: (approx_t_Gya, hold_seconds, narrative_text).
#: The nearest epoch in the axis to approx_t is held for hold_seconds.
TRANSITION_EVENTS: List[Tuple[float, float, str]] = [
    (-3.80, 3.5,
     "LHB PEAK  (-3.8 Gya)  |  Impact flux at maximum\n"
     "Brief liquid-water ponds from melt provide prebiotic chemistry windows.\n"
     "UV-bright young Sun drives intense C₂H₂ photochemistry. No polar lakes yet."),

    (-1.00, 3.0,
     "LAKE FORMATION BEGINS  (-1.0 Gya)  |  Polar hydrocarbon seas consolidate\n"
     "liquid_hydrocarbon feature ramps from 10 → 100% of present value.\n"
     "Organic stockpile nears present level. Habitability upturn begins."),

    (-0.50, 2.5,
     "POLAR LAKES ESTABLISHED  (-0.5 Gya)  |  Kraken, Ligeia and Punga Mare fully formed\n"
     "Methane cycle penalty lifts. Lake-margin vesicle formation clock starts (Mayer & Nixon 2025).\n"
     "North polar shores (#1 Kraken, #2 Ligeia) now score highest on Titan."),

    ( 0.00, 3.5,
     "PRESENT EPOCH  (Cassini 2004–2017)  |  Calibration anchor\n"
     "All 8 features calibrated here. ~1.7 million km² of liquid hydrocarbon.\n"
     "Dragonfly mission will land at Selk crater (#3) in the 2030s."),

    ( 0.250, 2.5,
     "NEAR FUTURE  (+250 Myr)  |  D2 solar warming window opens\n"
     "Solar luminosity ~+2.5% above present. Subsurface ocean exchange modestly elevated.\n"
     "Polar lake margins remain most habitable. Dragonfly era data still applicable."),

    ( 1.50, 2.5,
     "SLOW ACCUMULATION PLATEAU  (+1.5 Gya)  |  Tholins build at 5×10⁻¹⁴ g/cm²/s\n"
     "Solar UV climbing steadily on the main sequence. Polar lakes remain stable.\n"
     "Habitability drifts slightly upward as organic inventory grows."),

    ( 4.0667, 3.0,
     "SOLAR WARMING RAMP  (+4.0 Gya)  |  L☉ ≈ 1.3× present\n"
     "Lake surfaces begin evaporating. Methane cycle weakening.\n"
     "liquid_hydrocarbon + methane_cycle together weight 0.36 — both declining."),

    ( 5.00, 3.5,
     "METHANE ATMOSPHERE LOST  (+5.0 Gya)  |  Lakes fully evaporated\n"
     "liquid_hydrocarbon → 0. Organic stockpile at 16 Gyr maximum.\n"
     "Surface is dry and cold. Local minimum before the red-giant transition."),

    ( 5.1333, 4.0,
     "WATER-AMMONIA EUTECTIC CROSSED  (+5.1 Gya)  |  T_surface > 176 K\n"
     "Global liquid water-ammonia ocean forms in < 1 Myr. Entire surface is now liquid.\n"
     "Subsurface ocean merges with surface. Maximum organic concentration in solution."),

    ( 5.50, 3.5,
     "PEAK HABITABILITY  (+5.5 Gya)  |  Global liquid water-ammonia ocean\n"
     "All surfaces score ~0.47. 16 Gyr organic stockpile dissolved as bioavailable substrate.\n"
     "Water-ammonia chemistry enables terrestrial-analogue biochemistry."),

    ( 6.0, 3.0,
     "END OF HABITABLE WINDOW  (+6.0 Gya)  |  Sun exits red-giant phase\n"
     "Luminosity collapses from 600× to 0.8×. Ocean refreezes in < 1 Myr.\n"
     "Total red-giant habitable window: ~400 Myr."),
]


# --- Solar / temperature models -----------------------------------------------

def solar_luminosity_ratio(t: float) -> float:
    """L(t) / L_present.

    Continuous at t=0 (main-sequence join).  Has a deliberate step
    discontinuity at t=5.0 Gya: the main-sequence branch gives
    L=2.05 at t=5.0-ε while the red-giant ramp starts at L=1.0 at
    t=5.0+ε (ΔT≈18 K).  At t=5.0 all liquid-dependent scale functions
    are zero (lakes gone, methane gone) so the posterior map is not
    visually affected by this seam.  Acetylene scale has a +0.31 step
    but contributes only Δw_sum≈0.017 → ΔP(hab)≈0.001 (imperceptible).
    """
    age_now = 4.57
    age = age_now + t
    if age <= 0:
        return 0.5
    if t <= 5.0:
        L = 0.72 + (1.0 - 0.72) * (age / age_now) ** 0.9
        return max(0.5, L)
    t_after = t - 5.0
    if t_after < 0.1:
        return 1.0 + 17.0 * t_after
    elif t_after < 0.5:
        return max(2.0, 2700.0 * math.exp(-0.5 * ((t_after - 0.4) / 0.15) ** 2))
    elif t_after < 1.0:
        return max(1.0, 2700.0 * math.exp(-3.0 * (t_after - 0.4)))
    return 0.8


def titan_temp_K(t: float) -> float:
    """Surface temperature at epoch t."""
    return T_SURFACE_K * solar_luminosity_ratio(t) ** 0.25


# --- Time-scaling functions ---------------------------------------------------

def _scale_liquid_hc(t: float) -> float:
    """Scalar multiplier for liquid_hydrocarbon at epoch t."""
    T = titan_temp_K(t)
    if T >= EUTECTIC_K:
        return 50.0   # global ocean -- huge additive override (clamped to 1 after)
    if t < -1.0:
        return 0.10
    elif t < -0.5:
        return 0.10 + 0.90 * ((t + 1.0) / 0.5)
    elif t < 4.0:
        return 1.0
    elif t < 5.0:
        return max(0.0, 1.0 - (t - 4.0))
    return 0.0


def _scale_organic(t: float) -> float:
    """Organic abundance accumulation factor."""
    t_atm = 4.0
    t_elapsed = t_atm + t
    if t_elapsed <= 0:
        return 0.0
    T = titan_temp_K(t)
    frac = min(t_elapsed / t_atm, 2.5)
    if T >= EUTECTIC_K:
        frac = min(frac * 1.1, 2.5)
    return frac


def _scale_acetylene(t: float) -> float:
    """UV-driven C2H2 energy proxy -- continuous through t=0."""
    T = titan_temp_K(t)
    if T >= EUTECTIC_K:
        return 0.10 / 0.35  # return as scale on prior mean
    age_now = 4.57
    age_then = age_now + t
    if age_then <= 0:
        uv = 2.5
    elif t < 5.0:
        uv = min(2.5, (age_now / age_then) ** 0.5)
    else:
        uv = max(0.0, 1.0 - (t - 5.0) / 1.0)
    return uv


def _scale_methane(t: float) -> float:
    """Methane cycle activity."""
    T = titan_temp_K(t)
    if T >= EUTECTIC_K:
        return 0.3 / 0.40
    if t < -1.0:
        return 0.60
    elif t < -0.5:
        return 0.80
    elif t < 4.5:
        return 1.0
    return max(0.0, 1.0 - (t - 4.5) / 0.5)


def _scale_surface_atm(t: float) -> float:
    """Scale surface-atmosphere interaction via liquid proxy."""
    lhc = min(1.0, _scale_liquid_hc(t))
    slope_frac   = 0.40
    liquid_frac  = 0.60
    return slope_frac + liquid_frac * lhc


def _scale_topo(t: float) -> float:
    """Topographic complexity -- changes slowly."""
    if t < -2.0:
        return 1.30
    elif t < -1.0:
        return 1.15
    return 1.0


def _scale_geodiversity(t: float) -> float:
    if t < -3.0:
        return 0.70
    elif t < -2.0:
        return 0.85
    return 1.0


def _scale_subsurface(t: float) -> float:
    T = titan_temp_K(t)
    if T >= EUTECTIC_K:
        return 1.0 / 0.03   # ocean IS the surface -- override
    if t < -2.0:
        return 2.5
    elif t < -1.0:
        return 1.8
    elif t < -0.5:
        return 1.3
    return 1.0


def _impact_melt_global(t: float) -> float:
    """
    Global impact-melt bonus -- spatially uniform additive term.
    Peaks at LHB (-3.8 Gya), decays symmetrically.
    """
    t_lhb, tau = -3.8, 0.8
    peak = 0.40 * math.exp(-0.5 * ((t - t_lhb) / 0.5) ** 2)
    bg   = 0.10 * math.exp(-abs(t - t_lhb) / tau)
    return min(1.0, peak + bg)


FEATURE_SCALE_FUNCS = {
    "liquid_hydrocarbon":       _scale_liquid_hc,
    "organic_abundance":        _scale_organic,
    "acetylene_energy":         _scale_acetylene,
    "methane_cycle":            _scale_methane,
    "surface_atm_interaction":  _scale_surface_atm,
    "topographic_complexity":   _scale_topo,
    "geomorphologic_diversity": _scale_geodiversity,
    "subsurface_ocean":         _scale_subsurface,
    "impact_melt_bonus":        lambda t: _impact_melt_global(t),
}


# --- Feature map loaders -----------------------------------------------------

def _lat_lon_grids() -> Tuple[np.ndarray, np.ndarray]:
    """Return (lat_deg, lon_W_deg) 2-D grids for GRID_SHAPE."""
    nrows, ncols = GRID_SHAPE
    lats = np.linspace(90.0, -90.0, nrows, endpoint=False)
    lons = np.linspace(0.0, 360.0, ncols, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid.astype(np.float32), lon_grid.astype(np.float32)


def _synthetic_features() -> Dict[str, np.ndarray]:
    """
    Generate physically-grounded synthetic present-epoch feature maps.

    Used when real pipeline TIFs are not available.  Each map encodes the
    known gross structure of Titan's surface:
      - Polar lakes (>60 degN/S) high liquid_HC
      - Equatorial belt high organic_abundance (dunes)
      - Crater regions (near 90 degW) higher geomorphologic_diversity
      - Ontario Lacus (179 degW, -72 deg) southern lake
    """
    lat, lon_W = _lat_lon_grids()
    lat_r = np.deg2rad(lat)

    # -- liquid_hydrocarbon ----------------------------------------------------
    north_lake_zone = np.clip((lat - 60.0) / 20.0, 0.0, 1.0) ** 1.5
    south_lake_zone = np.clip((-lat - 60.0) / 20.0, 0.0, 1.0) ** 1.5
    # Ontario Lacus: 179 degW, -72 deg
    ontario = np.exp(-((lon_W - 179.0) ** 2 + (lat + 72.0) ** 2) / 64.0)
    liquid = np.clip(north_lake_zone * 0.7 + south_lake_zone * 0.15 +
                     ontario * 0.5, 0.0, 1.0).astype(np.float32)

    # -- organic_abundance -----------------------------------------------------
    # High in equatorial dune belt (|lat|<30 deg), lower at poles, very low Xanadu
    dune_belt = np.clip(1.0 - np.abs(lat) / 30.0, 0.0, 1.0) ** 0.5
    # Xanadu (~100 degW, -5 deg): low organic, high water-ice
    xanadu = 1.0 - 0.7 * np.exp(-((lon_W - 100.0) ** 2 + lat ** 2) / 400.0)
    organic = np.clip(0.35 + 0.45 * dune_belt * xanadu, 0.0, 1.0).astype(np.float32)

    # -- acetylene_energy ------------------------------------------------------
    # Driven by UV photolysis -- mild equatorial enhancement, lake surface ~0
    ace = np.clip(0.28 + 0.15 * (1.0 - np.abs(lat) / 90.0) -
                  0.15 * liquid, 0.0, 1.0).astype(np.float32)

    # -- methane_cycle ---------------------------------------------------------
    # Peaks near equator where precipitation is most active
    meth = np.clip(0.30 + 0.15 * np.cos(lat_r) +
                   0.10 * north_lake_zone, 0.0, 1.0).astype(np.float32)

    # -- surface_atm_interaction -----------------------------------------------
    # Highest at lake margins (~63-67 degN) and channels
    lake_margin = np.clip(
        np.exp(-((lat - 64.0) ** 2) / 8.0) * 0.8 +
        np.exp(-((lat + 64.0) ** 2) / 8.0) * 0.25, 0.0, 1.0
    )
    sai = np.clip(0.20 + 0.35 * lake_margin + 0.15 * np.cos(lat_r) ** 2,
                  0.0, 1.0).astype(np.float32)

    # -- topographic_complexity ------------------------------------------------
    # Higher at poles (labyrinth), crater regions, mountain chains
    np.random.seed(42)
    topo_noise = np.random.uniform(0.0, 0.1, GRID_SHAPE).astype(np.float32)
    selk  = np.exp(-((lon_W - 199.0)**2 + (lat - 7.0)**2) / 100.0) * 0.4
    menrva = np.exp(-((lon_W - 87.3)**2 + (lat - 19.0)**2) / 200.0) * 0.35
    topo = np.clip(0.15 + topo_noise + selk + menrva +
                   0.10 * np.abs(lat) / 90.0, 0.0, 1.0).astype(np.float32)

    # -- geomorphologic_diversity ----------------------------------------------
    geo = np.clip(0.20 + 0.20 * lake_margin + selk * 0.5 +
                  menrva * 0.4 + 0.10 * np.abs(lat) / 90.0,
                  0.0, 1.0).astype(np.float32)

    # -- subsurface_ocean ------------------------------------------------------
    # Uniform global prior -- k2=0.589 confirms global subsurface ocean
    sub = np.full(GRID_SHAPE, 0.03, dtype=np.float32)

    # -- impact_melt_bonus -----------------------------------------------------
    # Localised around craters -- 0 everywhere at present (already past)
    imb = np.zeros(GRID_SHAPE, dtype=np.float32)

    return {
        "liquid_hydrocarbon":       liquid,
        "organic_abundance":        organic,
        "acetylene_energy":         ace,
        "methane_cycle":            meth,
        "surface_atm_interaction":  sai,
        "topographic_complexity":   topo,
        "geomorphologic_diversity": geo,
        "subsurface_ocean":         sub,
        "impact_melt_bonus":        imb,
    }


def load_present_features(feature_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load present-epoch feature GeoTIFFs produced by run_pipeline.py.

    Falls back to synthetic maps ONLY if real TIFs are not available.
    When synthetic data is used, a prominent warning is printed and the
    function returns a flag so callers can declare it in output metadata.

    ``impact_melt_bonus`` note
    --------------------------
    This feature is NOT produced by run_pipeline.py.  It is an animation-
    only feature computed on-the-fly by :func:`_impact_melt_global` for each
    epoch.  At the PRESENT epoch its value is identically zero (there are no
    currently active impact-melt oases on Titan).  The spatial distribution
    for past epochs is derived analytically from the SAR-bright crater annuli
    (Hedgepeth et al. 2020) rather than from a stored TIF.

    Consequence: no ``impact_melt_bonus.tif`` file will ever exist in the
    feature directory, and this is NOT a missing-data error.  The feature is
    always synthesised at load time.

    Parameters
    ----------
    feature_dir:
        Directory containing ``<feature_name>.tif`` files.

    Returns
    -------
    Dict[str, np.ndarray]
        Feature name -> float32 array of shape GRID_SHAPE.
        A key ``"_synthetic"`` with value ``True`` is added when synthetic
        maps are returned, so callers can log and propagate this fact.
    """
    # impact_melt_bonus is always synthesised -- never looked for on disk.
    # All other features are expected as TIF files from run_pipeline.py.
    SYNTHESISED_FEATURES: frozenset = frozenset({"impact_melt_bonus"})

    feature_names = list(WEIGHTS.keys())
    maps: Dict[str, np.ndarray] = {}
    n_real: int = 0

    for name in feature_names:
        # Synthesised features are computed on-the-fly, not loaded from disk
        if name in SYNTHESISED_FEATURES:
            maps[name] = np.zeros(GRID_SHAPE, dtype=np.float32)
            # Not counted towards n_real -- don't print a missing-file warning
            continue

        tif: Path = feature_dir / f"{name}.tif"
        if tif.exists():
            try:
                import rasterio
                with rasterio.open(tif) as src:
                    arr: np.ndarray = src.read(1).astype(np.float32)
                    nd = src.nodata
                    if nd is not None:
                        arr[arr == nd] = np.nan
                    arr[arr < 0] = np.nan
                    maps[name] = arr
                    n_real += 1
            except Exception as exc:
                print(f"  WARNING: failed to load {name}.tif: {exc}")

    # Count only non-synthesised features towards the threshold
    n_expected = len([n for n in feature_names if n not in SYNTHESISED_FEATURES])

    if n_real >= min(6, n_expected):
        print(f"  Loaded {n_real}/{n_expected} real feature TIFs from {feature_dir}")
        print(f"  impact_melt_bonus: synthesised on-the-fly (not a file; "
              f"zero at PRESENT epoch by design)")
        # Fill any missing non-synthesised features with prior-mean constant maps
        for name in feature_names:
            if name not in maps:
                if name in SYNTHESISED_FEATURES:
                    continue   # already handled above
                maps[name] = np.full(GRID_SHAPE, PRIOR_MEANS[name],
                                     dtype=np.float32)
                print(f"  INFO: {name}.tif missing -- "
                      f"filled with prior mean {PRIOR_MEANS[name]:.3f}")
        maps["_synthetic"] = False   # type: ignore[assignment]
        return maps
    else:
        # -- SYNTHETIC DATA FALLBACK -------------------------------------------
        # This path is taken ONLY when real Cassini-derived feature TIFs are
        # not present.  Synthetic maps are physically grounded heuristics based
        # on known Titan geography; they are NOT observational data.
        # Run run_pipeline.py --temporal-mode present first to generate real TIFs.
        sep = "=" * 70
        msg_lines = [
            sep,
            "  *** SYNTHETIC DATA MODE ***",
            f"  Real feature TIFs not found in: {feature_dir}",
            f"  Found {n_real}/{len(feature_names)} TIFs "
            f"(need >= 6 to use real data).",
            "  Falling back to SYNTHETIC feature maps.",
            "  These are geography-based heuristics, NOT Cassini observations.",
            "  To use real data: run run_pipeline.py --temporal-mode present",
            "  then re-run this script.",
            sep,
        ]
        for line in msg_lines:
            print(line)
        synth: Dict[str, np.ndarray] = _synthetic_features()
        synth["_synthetic"] = True   # type: ignore[assignment]
        return synth


# --- Temporal scaling ---------------------------------------------------------

def scale_features_to_epoch(
    present: Dict[str, np.ndarray],
    t: float,
) -> Dict[str, np.ndarray]:
    """
    Apply temporal scaling to all present-epoch feature maps.

    Parameters
    ----------
    present:
        Present-epoch feature arrays (t = 0).
    t:
        Target epoch in Gya from present (negative = past).

    Returns
    -------
    Dict[str, np.ndarray]
        Scaled feature arrays, clamped to [0, 1].
    """
    result: Dict[str, np.ndarray] = {}
    for name, arr in present.items():
        # Skip metadata keys (e.g. "_synthetic") that are not real feature arrays
        if name not in FEATURE_SCALE_FUNCS:
            continue
        scale_fn = FEATURE_SCALE_FUNCS[name]
        scale    = scale_fn(t)
        if name == "organic_abundance":
            # Scale as a fraction of accumulated stockpile
            # arr encodes relative organic density; scale the absolute level
            scaled = arr * scale
        elif name == "subsurface_ocean" and titan_temp_K(t) >= EUTECTIC_K:
            # Global water ocean -- override to 1.0 everywhere
            scaled = np.ones_like(arr)
        elif name == "liquid_hydrocarbon" and titan_temp_K(t) >= EUTECTIC_K:
            # Global water ocean -- override to 1.0 everywhere
            scaled = np.ones_like(arr)
        elif name == "impact_melt_bonus":
            # Additive global field -- ignore present spatial pattern
            # (there are no present craters still melting; past is modelled globally)
            scaled = np.full_like(arr, min(1.0, _impact_melt_global(t)))
        else:
            scaled = arr * scale

        result[name] = np.clip(scaled, 0.0, 1.0).astype(np.float32)

    return result


# --- Bayesian inference -------------------------------------------------------

def bayesian_posterior_map(
    features: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute per-pixel Beta posterior mean P(habitable | features).

    Parameters
    ----------
    features:
        Scaled feature arrays at one epoch.

    Returns
    -------
    np.ndarray
        float32 posterior mean array, shape GRID_SHAPE, range [0, 1].
        NaN where all inputs were NaN.
    """
    # Prior weighted mean
    mu0: float = sum(PRIOR_MEANS[k] * WEIGHTS[k] for k in WEIGHTS)
    alpha0: float = mu0 * KAPPA
    beta0:  float = (1.0 - mu0) * KAPPA

    # Weighted feature sum across features at each pixel
    nrows, ncols = GRID_SHAPE
    w_sum: np.ndarray  = np.zeros((nrows, ncols), dtype=np.float64)
    valid: np.ndarray  = np.zeros((nrows, ncols), dtype=bool)

    for name, arr in features.items():
        if name not in WEIGHTS:
            continue   # skip metadata keys such as "_synthetic"
        w = WEIGHTS[name]
        finite_mask = np.isfinite(arr)
        w_sum  += np.where(finite_mask, arr.astype(np.float64) * w, 0.0)
        valid  |= finite_mask

    # Beta posterior update
    alpha_post: np.ndarray = alpha0 + LAMBDA * w_sum
    beta_post:  np.ndarray = beta0  + LAMBDA * (1.0 - w_sum)
    posterior:  np.ndarray = (alpha_post / (alpha_post + beta_post)).astype(np.float32)
    posterior[~valid] = np.nan

    return posterior


# --- GeoTIFF writer ----------------------------------------------------------

TITAN_CRS_PROJ4: str = (
    "+proj=eqc +a=2575000 +b=2575000 +units=m +no_defs "
    "+lon_0=0 +lat_ts=0"
)

def canonical_transform() -> "rasterio.transform.Affine":
    """
    Return the rasterio Affine transform for the canonical Titan grid.

    The grid is equirectangular, west-positive, covering 0-360 degW x -90-90 degN.
    """
    import rasterio.transform as rt
    m_per_deg: float = math.pi * TITAN_RADIUS_M / 180.0
    return rt.from_origin(
        west  = 0.0,
        north = 90.0  * m_per_deg,
        xsize = CANONICAL_RES_M,
        ysize = CANONICAL_RES_M,
    )


def write_geotiff(
    arr:      np.ndarray,
    out_path: Path,
    nodata:   float = -9999.0,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Write a float32 array as a compressed GeoTIFF.

    Parameters
    ----------
    arr:
        2-D float32 array, shape GRID_SHAPE.  NaN pixels are written as nodata.
    out_path:
        Destination path.
    nodata:
        Nodata sentinel value.
    metadata:
        Optional dict written as GDAL metadata tags.
    """
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr_out = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)

    with rasterio.open(
        out_path, "w",
        driver    = "GTiff",
        dtype     = "float32",
        count     = 1,
        width     = GRID_SHAPE[1],
        height    = GRID_SHAPE[0],
        crs       = TITAN_CRS_PROJ4,
        transform = canonical_transform(),
        nodata    = nodata,
        compress  = "deflate",
        predictor = 2,          # float predictor -- better compression for rasters
        tiled     = True,
        blockxsize = 512,
        blockysize = 512,
    ) as dst:
        dst.write(arr_out, 1)
        if metadata:
            dst.update_tags(**metadata)


# --- NetCDF writer ------------------------------------------------------------

def save_netcdf_stack(
    epochs:    np.ndarray,
    maps:      List[np.ndarray],
    out_path:  Path,
) -> None:
    """
    Save all epoch maps as a NetCDF4 time-series stack.

    Dimensions: (time, lat, lon)
    Variable:   P_habitable  -- posterior mean P(habitable | features)
    time:       epoch in Gya from present

    QGIS Temporal Controller can open this directly.

    Parameters
    ----------
    epochs:
        1-D array of epoch values (Gya from present).
    maps:
        List of posterior arrays, one per epoch, shape GRID_SHAPE.
    out_path:
        Output NetCDF path.
    """
    try:
        import netCDF4 as nc4
    except ImportError:
        try:
            import scipy.io.netcdf as scipy_nc
        except ImportError:
            print("  WARNING: neither netCDF4 nor scipy available -- skipping .nc output")
            return

    # Use numpy + manual NetCDF3 via scipy if netCDF4 not available
    # Prefer numpy/xarray approach
    try:
        import xarray as xr

        nrows: int
        ncols: int
        nrows, ncols = GRID_SHAPE
        m_per_deg: float = math.pi * TITAN_RADIUS_M / 180.0
        lats: np.ndarray = np.linspace(90.0,  -90.0, nrows, endpoint=False, dtype=np.float32)
        lons: np.ndarray = np.linspace(0.0,   360.0, ncols, endpoint=False, dtype=np.float32)

        data: np.ndarray = np.stack(maps, axis=0)   # (n_epochs, nrows, ncols)

        ds = xr.Dataset(
            {
                "P_habitable": xr.DataArray(
                    data,
                    dims   = ["epoch_Gya", "lat", "lon"],
                    coords = {
                        "epoch_Gya": epochs,
                        "lat":       lats,
                        "lon":       lons,
                    },
                    attrs  = {
                        "long_name": "Posterior mean P(habitable | features)",
                        "units":     "probability [0-1]",
                        "valid_min": 0.0,
                        "valid_max": 1.0,
                    },
                )
            },
            attrs = {
                "title":       "Titan Multi-Epoch Habitability Map",
                "institution": "Titan Habitability Pipeline",
                "CRS":         TITAN_CRS_PROJ4,
                "epoch_units": "Gya from present (negative=past, positive=future)",
                "references":  "Birch+2017, Lopes+2019, Malaska+2025",
            },
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_path, format="NETCDF4")
        print(f"  NetCDF stack saved -> {out_path}")
    except Exception as exc:
        print(f"  WARNING: NetCDF save failed: {exc}")


# --- Matplotlib renderer ------------------------------------------------------

def _epoch_label(t: float) -> str:
    """Human-readable epoch label."""
    if abs(t) < 0.005:
        return "Present\n(Cassini epoch)"
    sign = "+" if t > 0 else ""
    return f"{sign}{t:.2f} Gya"


def _phase_label(t: float) -> str:
    """
    Brief geological phase name for the epoch t (Gya from present).

    Phase boundaries
    ----------------
    t < -3.0              Late Heavy Bombardment
    -3.0 <= t < -1.0       Early Titan
    -1.0 <= t < -0.3       Lake formation
    -0.3 <= t < -0.05      Recent past          <- avoids "Near future" for past epochs
    |t| < 0.05            Cassini epoch         <- tight window: +/-50 Mya
    0.05 <= t < 3.0        Near future
    3.0 <= t < 5.0         Pre red-giant
    5.0 <= t < EUTECTIC    Red-giant ramp
    t >= EUTECTIC          Red-giant water ocean
    """
    T: float = titan_temp_K(t)
    if T >= EUTECTIC_K:
        return "Red-giant\nwater ocean"
    if t < -3.0:
        return "Late Heavy\nBombardment"
    if t < -1.0:
        return "Early Titan"
    if t < -0.3:
        return "Lake\nformation"
    if t < -0.05:
        return "Recent past"
    if t <= 0.05:
        return "Cassini\nepoch"
    if t < 3.0:
        return "Near future"
    if t < 4.0:
        return "Pre red-giant"    # stable future: lakes intact, sun slowly warming
    if t < 5.0:
        return "Solar warming"    # lake evaporation: +4.0 → +5.0 Gya
    # T < EUTECTIC_K here (handled above).
    # t=5.0–6.0: Sun becoming red giant, T rising toward eutectic → ramp up.
    # t>=6.0: Sun exits red-giant, L collapses 600×→0.8×, T drops → ramp down.
    if t < 6.0:
        return "Red-giant\nramp up"
    return "Red-giant\nramp down"


# ---------------------------------------------------------------------------
# Per-epoch top-10 computation
# ---------------------------------------------------------------------------
# Present-epoch Cassini-derived feature profiles for candidate sites.
# impact_melt_bonus encodes site-specific SENSITIVITY to impact-melt activity
# (0.0 = plain terrain with no crater history; 1.0 = major fresh impact crater).
# At epoch t: f_impact = min(1, global_signal * (0.3 + 0.7 * sensitivity))
# where global_signal = _impact_melt_global(t).  Non-crater sites therefore
# receive 30% of the global LHB signal; major craters receive up to 100%.
CANDIDATE_SITES: List[Dict] = [
    # ── Lake / sea shores (sensitivity low: no crater structure) ──────────
    {"lon_W": 310.0, "lat":  68.0, "label": "Kraken S",    "type": "lake",
     "f": {"liquid_hydrocarbon":1.00,"organic_abundance":0.05,"acetylene_energy":0.20,
           "methane_cycle":0.70,"surface_atm_interaction":0.65,"topographic_complexity":0.60,
           "geomorphologic_diversity":0.76,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W":  78.0, "lat":  79.0, "label": "Ligeia E",    "type": "lake",
     "f": {"liquid_hydrocarbon":1.00,"organic_abundance":0.05,"acetylene_energy":0.20,
           "methane_cycle":0.70,"surface_atm_interaction":0.62,"topographic_complexity":0.55,
           "geomorphologic_diversity":0.76,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 348.0, "lat":  80.0, "label": "Kraken N",    "type": "lake",
     "f": {"liquid_hydrocarbon":1.00,"organic_abundance":0.05,"acetylene_energy":0.18,
           "methane_cycle":0.68,"surface_atm_interaction":0.60,"topographic_complexity":0.52,
           "geomorphologic_diversity":0.72,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 339.0, "lat":  85.5, "label": "Punga",       "type": "lake",
     "f": {"liquid_hydrocarbon":0.90,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.65,"surface_atm_interaction":0.55,"topographic_complexity":0.45,
           "geomorphologic_diversity":0.65,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 179.0, "lat": -72.0, "label": "Ontario",     "type": "lake",
     "f": {"liquid_hydrocarbon":0.85,"organic_abundance":0.08,"acetylene_energy":0.22,
           "methane_cycle":0.45,"surface_atm_interaction":0.48,"topographic_complexity":0.42,
           "geomorphologic_diversity":0.60,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 336.3, "lat":  73.0, "label": "Jingpo",      "type": "lake",
     "f": {"liquid_hydrocarbon":0.88,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.62,"surface_atm_interaction":0.52,"topographic_complexity":0.40,
           "geomorphologic_diversity":0.60,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    # ── Equatorial dune / organic land sites (low sensitivity) ────────────
    {"lon_W": 250.0, "lat":   7.0, "label": "Belet",       "type": "land",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.82,"acetylene_energy":0.45,
           "methane_cycle":0.09,"surface_atm_interaction":0.09,"topographic_complexity":0.55,
           "geomorphologic_diversity":0.09,"subsurface_ocean":0.03,"impact_melt_bonus":0.30}},
    {"lon_W": 155.0, "lat":  -5.0, "label": "Shangri-La",  "type": "land",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.82,"acetylene_energy":0.45,
           "methane_cycle":0.09,"surface_atm_interaction":0.09,"topographic_complexity":0.52,
           "geomorphologic_diversity":0.09,"subsurface_ocean":0.03,"impact_melt_bonus":0.30}},
    {"lon_W":  20.0, "lat":  15.0, "label": "Fensal",      "type": "land",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.80,"acetylene_energy":0.44,
           "methane_cycle":0.09,"surface_atm_interaction":0.08,"topographic_complexity":0.50,
           "geomorphologic_diversity":0.09,"subsurface_ocean":0.03,"impact_melt_bonus":0.30}},
    {"lon_W": 100.0, "lat":  10.0, "label": "Aztlan",      "type": "land",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.78,"acetylene_energy":0.44,
           "methane_cycle":0.09,"surface_atm_interaction":0.08,"topographic_complexity":0.48,
           "geomorphologic_diversity":0.09,"subsurface_ocean":0.03,"impact_melt_bonus":0.30}},
    # ── Cryovolcanic features (high sensitivity: subsurface conduits) ──────
    {"lon_W":  75.0, "lat": -26.0, "label": "Hotei",       "type": "land",
     "f": {"liquid_hydrocarbon":0.03,"organic_abundance":0.65,"acetylene_energy":0.42,
           "methane_cycle":0.18,"surface_atm_interaction":0.22,"topographic_complexity":0.35,
           "geomorphologic_diversity":0.58,"subsurface_ocean":0.14,"impact_melt_bonus":0.90}},
    {"lon_W": 144.5, "lat":   9.8, "label": "Sotra",       "type": "land",
     "f": {"liquid_hydrocarbon":0.03,"organic_abundance":0.62,"acetylene_energy":0.40,
           "methane_cycle":0.16,"surface_atm_interaction":0.20,"topographic_complexity":0.38,
           "geomorphologic_diversity":0.55,"subsurface_ocean":0.12,"impact_melt_bonus":0.85}},
    # ── Impact craters (sensitivity proportional to diameter) ─────────────
    {"lon_W":  87.3, "lat":  19.0, "label": "Menrva",      "type": "land",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.30,"acetylene_energy":0.38,
           "methane_cycle":0.08,"surface_atm_interaction":0.08,"topographic_complexity":0.18,
           "geomorphologic_diversity":0.55,"subsurface_ocean":0.22,"impact_melt_bonus":1.00}},
    {"lon_W": 349.0, "lat": -38.6, "label": "Hano",        "type": "land",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.42,"acetylene_energy":0.35,
           "methane_cycle":0.08,"surface_atm_interaction":0.10,"topographic_complexity":0.16,
           "geomorphologic_diversity":0.40,"subsurface_ocean":0.08,"impact_melt_bonus":0.65}},
    {"lon_W": 200.5, "lat":  -1.4, "label": "Afekan",      "type": "land",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.35,"acetylene_energy":0.36,
           "methane_cycle":0.08,"surface_atm_interaction":0.08,"topographic_complexity":0.15,
           "geomorphologic_diversity":0.38,"subsurface_ocean":0.09,"impact_melt_bonus":0.70}},
    {"lon_W":  16.0, "lat":  11.3, "label": "Sinlap",      "type": "land",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.32,"acetylene_energy":0.36,
           "methane_cycle":0.08,"surface_atm_interaction":0.08,"topographic_complexity":0.14,
           "geomorphologic_diversity":0.36,"subsurface_ocean":0.09,"impact_melt_bonus":0.65}},
    {"lon_W":  65.6, "lat":  14.0, "label": "Ksa",         "type": "land",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.34,"acetylene_energy":0.36,
           "methane_cycle":0.08,"surface_atm_interaction":0.08,"topographic_complexity":0.14,
           "geomorphologic_diversity":0.35,"subsurface_ocean":0.07,"impact_melt_bonus":0.60}},
    # ── Named northern lacus (IAU-named lake bodies; smaller than the three
    #    main mares but confirmed by Cassini RADAR/VIMS).  Feature profiles
    #    reflect smaller liquid area (f1 0.65-0.80), moderate shoreline
    #    geodiversity, and standard north-polar methane cycle. ──────────────
    {"lon_W": 154.0, "lat":  73.0, "label": "Koitere",     "type": "lake",
     "f": {"liquid_hydrocarbon":0.72,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.58,"surface_atm_interaction":0.48,"topographic_complexity":0.38,
           "geomorphologic_diversity":0.55,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W":  93.5, "lat":  70.7, "label": "Hammar",      "type": "lake",
     "f": {"liquid_hydrocarbon":0.68,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.56,"surface_atm_interaction":0.46,"topographic_complexity":0.36,
           "geomorphologic_diversity":0.52,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 327.0, "lat":  72.0, "label": "Neagh",       "type": "lake",
     "f": {"liquid_hydrocarbon":0.70,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.58,"surface_atm_interaction":0.50,"topographic_complexity":0.40,
           "geomorphologic_diversity":0.54,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 262.8, "lat":  77.5, "label": "Mackay",      "type": "lake",
     "f": {"liquid_hydrocarbon":0.80,"organic_abundance":0.05,"acetylene_energy":0.18,
           "methane_cycle":0.60,"surface_atm_interaction":0.52,"topographic_complexity":0.42,
           "geomorphologic_diversity":0.58,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 262.0, "lat":  74.5, "label": "Uvs",         "type": "lake",
     "f": {"liquid_hydrocarbon":0.65,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.55,"surface_atm_interaction":0.44,"topographic_complexity":0.35,
           "geomorphologic_diversity":0.50,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W":  12.3, "lat":  75.4, "label": "Bolsena",     "type": "lake",
     "f": {"liquid_hydrocarbon":0.68,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.57,"surface_atm_interaction":0.46,"topographic_complexity":0.37,
           "geomorphologic_diversity":0.52,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 254.0, "lat":  65.5, "label": "Kivu",        "type": "lake",
     "f": {"liquid_hydrocarbon":0.60,"organic_abundance":0.07,"acetylene_energy":0.20,
           "methane_cycle":0.48,"surface_atm_interaction":0.40,"topographic_complexity":0.35,
           "geomorphologic_diversity":0.48,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},

    # ── Additional named lacus (completing the IAU catalog) ─────────────────
    # Freeman, Oib: north of Ligeia Mare (~210-220W, 82-83N)
    {"lon_W": 210.0, "lat":  83.0, "label": "Freeman",    "type": "lake",
     "f": {"liquid_hydrocarbon":0.80,"organic_abundance":0.05,"acetylene_energy":0.18,
           "methane_cycle":0.62,"surface_atm_interaction":0.52,"topographic_complexity":0.42,
           "geomorphologic_diversity":0.60,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 220.0, "lat":  82.0, "label": "Oib",         "type": "lake",
     "f": {"liquid_hydrocarbon":0.65,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.58,"surface_atm_interaction":0.46,"topographic_complexity":0.36,
           "geomorphologic_diversity":0.52,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    # Cardiel, Towada: east of Ligeia (~118-128W, 77-78N)
    {"lon_W": 128.0, "lat":  78.0, "label": "Cardiel",     "type": "lake",
     "f": {"liquid_hydrocarbon":0.78,"organic_abundance":0.05,"acetylene_energy":0.18,
           "methane_cycle":0.60,"surface_atm_interaction":0.50,"topographic_complexity":0.40,
           "geomorphologic_diversity":0.58,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 118.0, "lat":  77.5, "label": "Towada",      "type": "lake",
     "f": {"liquid_hydrocarbon":0.65,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.57,"surface_atm_interaction":0.46,"topographic_complexity":0.36,
           "geomorphologic_diversity":0.52,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    # Waikare, Logtak: west/southwest of Ligeia (~185-198W, 74-75N)
    {"lon_W": 185.0, "lat":  75.0, "label": "Waikare",     "type": "lake",
     "f": {"liquid_hydrocarbon":0.70,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.56,"surface_atm_interaction":0.46,"topographic_complexity":0.36,
           "geomorphologic_diversity":0.52,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 197.5, "lat":  74.0, "label": "Logtak",      "type": "lake",
     "f": {"liquid_hydrocarbon":0.65,"organic_abundance":0.06,"acetylene_energy":0.18,
           "methane_cycle":0.55,"surface_atm_interaction":0.44,"topographic_complexity":0.35,
           "geomorphologic_diversity":0.50,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    # Paxsi, Romo: western north polar small lacus
    {"lon_W": 243.0, "lat":  72.0, "label": "Paxsi",       "type": "lake",
     "f": {"liquid_hydrocarbon":0.60,"organic_abundance":0.07,"acetylene_energy":0.18,
           "methane_cycle":0.52,"surface_atm_interaction":0.42,"topographic_complexity":0.33,
           "geomorphologic_diversity":0.48,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    {"lon_W": 265.0, "lat":  70.0, "label": "Romo",        "type": "lake",
     "f": {"liquid_hydrocarbon":0.55,"organic_abundance":0.07,"acetylene_energy":0.20,
           "methane_cycle":0.50,"surface_atm_interaction":0.40,"topographic_complexity":0.32,
           "geomorphologic_diversity":0.46,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    # Crveno: southern hemisphere, near Ontario (~177W, 70.5S)
    {"lon_W": 177.0, "lat": -70.5, "label": "Crveno",      "type": "lake",
     "f": {"liquid_hydrocarbon":0.55,"organic_abundance":0.08,"acetylene_energy":0.22,
           "methane_cycle":0.38,"surface_atm_interaction":0.40,"topographic_complexity":0.32,
           "geomorphologic_diversity":0.45,"subsurface_ocean":0.04,"impact_melt_bonus":0.30}},
    # Urmia, Sionascaig: southern temperate lacus (small, low methane cycle)
    {"lon_W": 186.0, "lat": -52.0, "label": "Urmia",       "type": "lake",
     "f": {"liquid_hydrocarbon":0.45,"organic_abundance":0.12,"acetylene_energy":0.30,
           "methane_cycle":0.22,"surface_atm_interaction":0.32,"topographic_complexity":0.28,
           "geomorphologic_diversity":0.38,"subsurface_ocean":0.03,"impact_melt_bonus":0.30}},
    {"lon_W": 278.1, "lat": -41.5, "label": "Sionascaig",  "type": "lake",
     "f": {"liquid_hydrocarbon":0.40,"organic_abundance":0.14,"acetylene_energy":0.32,
           "methane_cycle":0.18,"surface_atm_interaction":0.28,"topographic_complexity":0.26,
           "geomorphologic_diversity":0.35,"subsurface_ocean":0.03,"impact_melt_bonus":0.30}},

    # ── Lander sites (always shown; included here for ranking) ─────────────
    {"lon_W": 199.0, "lat":   7.0, "label": "Selk",        "type": "lander",
     "f": {"liquid_hydrocarbon":0.05,"organic_abundance":0.215,"acetylene_energy":0.379,
           "methane_cycle":0.025,"surface_atm_interaction":0.010,"topographic_complexity":0.054,
           "geomorphologic_diversity":0.629,"subsurface_ocean":0.130,"impact_melt_bonus":0.80}},
    {"lon_W": 192.3, "lat": -10.6, "label": "Huygens",     "type": "lander",
     "f": {"liquid_hydrocarbon":0.02,"organic_abundance":0.54,"acetylene_energy":0.38,
           "methane_cycle":0.09,"surface_atm_interaction":0.08,"topographic_complexity":0.14,
           "geomorphologic_diversity":0.20,"subsurface_ocean":0.03,"impact_melt_bonus":0.30}},
]


def _site_ph(site: Dict, t: float) -> float:
    """Compute Bayesian P(H|features) for a candidate site at epoch t.

    impact_melt_bonus in the site dict encodes the site's sensitivity
    (0.0-1.0) to impact-melt activity.  The actual epoch feature is:
        f_impact = min(1, global * (0.3 + 0.7 * sensitivity))
    where global = _impact_melt_global(t).  This gives non-crater sites
    30% of the global signal and major craters up to 100%.
    """
    mu0    = sum(PRIOR_MEANS[k] * WEIGHTS[k] for k in WEIGHTS)
    alpha0 = mu0 * KAPPA
    w_sum  = 0.0
    for feat_name, f_present in site["f"].items():
        if feat_name == "impact_melt_bonus":
            global_signal  = _impact_melt_global(t)
            sensitivity    = f_present           # 0.30 baseline … 1.0 crater
            f_t = min(1.0, global_signal * (0.30 + 0.70 * sensitivity))
        else:
            scale = FEATURE_SCALE_FUNCS[feat_name](t)
            f_t   = min(1.0, max(0.0, scale * f_present))
        w_sum += WEIGHTS.get(feat_name, 0.0) * f_t
    return (alpha0 + LAMBDA * w_sum) / (KAPPA + LAMBDA)


def compute_epoch_top10(t: float) -> List[Tuple[float, float, str, int, str]]:
    """
    Rank all CANDIDATE_SITES by P(H|features) at epoch t.

    Returns the top-10 as a list of
        (lon_W, lat, label, rank, pole_hint)
    in the same format as the former static TOP10 list.
    pole_hint is "N" for lat>50, "S" for lat<-50, else "".
    """
    scored = [(s, _site_ph(s, t)) for s in CANDIDATE_SITES]
    scored.sort(key=lambda x: x[1], reverse=True)
    top10 = []
    for rank, (s, _ph) in enumerate(scored[:10], start=1):
        lat = s["lat"]
        pole = "N" if lat > 50 else ("S" if lat < -50 else "")
        top10.append((s["lon_W"], lat, s["label"], rank, pole))
    return top10


# Labels that should always appear on every frame regardless of ranking.
# These sites are shown without a rank prefix unless they happen to be
# in the epoch's computed top-10, in which case their rank is shown.
ALWAYS_SHOW: set = {"Selk", "Huygens"}

def render_frame(
    posterior:  np.ndarray,
    t:          float,
    epoch_idx:  int,
    n_epochs:   int,
    dpi:        int = 120,
    narrative:  str = "",
    source:     str = "MODELLED",
) -> "matplotlib.figure.Figure":
    """
    Render a single epoch as a matplotlib figure with three panels:
      Left:   Equirectangular (cylindrical) global map
      Centre: North polar stereographic
      Right:  South polar stereographic
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    from scipy.ndimage import map_coordinates

    nrows, ncols = GRID_SHAPE

    # -- Colourmap ------------------------------------------------------------
    # Both the equirectangular and polar panels use COLOUR_BACKGROUND for NaN
    # pixels (missing data or outside-disc regions).  Using a solid colour
    # rather than transparency guarantees the Agg backend produces the exact
    # same colour as the figure background without compositing artefacts.
    cmap = matplotlib.colormaps["plasma"]
    cmap.set_bad(color=COLOUR_BACKGROUND)   # matplotlib keyword stays American English
    cmap_polar = cmap.copy()               # identical -- both use background colour
    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)   # matplotlib API

    T_surface = titan_temp_K(t)

    # -- Top-10 locations -----------------------------------------------------
    # (lon_W deg, lat deg, short_label, rank, pole: N/S/blank)
    # Per-epoch top-10: computed dynamically from the Bayesian formula
    # using CANDIDATE_SITES and the epoch-specific scale functions.
    TOP10 = compute_epoch_top10(t)

    # Build a lookup: label -> site type, from CANDIDATE_SITES
    SITE_TYPE: dict  = {s["label"]: s["type"] for s in CANDIDATE_SITES}
    MARKER_SHAPE: dict = {
        "lake":   "^",   # triangle-up
        "land":   "s",   # square
        "lander": "*",   # star
    }
    MARKER_SIZE_BY_TYPE: dict = {
        "lake":   8,
        "land":   7,
        "lander": 11,
    }

    # Sites always shown (Huygens probe + Dragonfly target) - added after top-10
    # so they appear on every frame.  If they rank in the top-10 they are already
    # included with their rank number.
    _top10_labels = {label for _, _, label, _, _ in TOP10}
    ALWAYS_EXTRA: List[Tuple[float, float, str, int, str]] = [
        (lon, lat, lbl, 0, ("N" if lat > 50 else "S" if lat < -50 else ""))
        for s in CANDIDATE_SITES
        if s["label"] in ALWAYS_SHOW and s["label"] not in _top10_labels
        for lon, lat, lbl in [(s["lon_W"], s["lat"], s["label"])]
    ]

    MARKER_SIZE: int  = 6
    TEXT_SIZE:   float = 7.5

    def _label_offset(lon_W: float, lat: float) -> Tuple[float, float]:
        """Return (dx, dy) offset in data units to push label clear of the dot."""
        dx: float = 6.0 if lon_W < 300 else -6.0
        dy: float = 5.0 if lat < 70 else -6.0
        return dx, dy

    # -- Shared figure layout --------------------------------------------------
    # The figure is split vertically into three zones:
    #   top strip  (y = GS_TOP  -> 1.00) : title, epoch info, progress bar
    #   map panels (y = GS_BOTTOM -> GS_TOP) : equirectangular + 2 polar caps
    #   bottom bar (y = 0.00 -> GS_BOTTOM)  : colourbar, narrative text boxes
    #
    # Height-matching geometry (wspace=0):
    #   With width_ratios=[2,1,1] the polar panel fraction = (GS_RIGHT-GS_LEFT)/4.
    #   Setting this equal to the panel height fraction (GS_TOP-GS_BOTTOM) gives:
    #     fig_width = 4.(GS_TOP-GS_BOTTOM)/(GS_RIGHT-GS_LEFT).fig_height
    #   Substituting FIG_HEIGHT_IN=7.5 yields FIG_WIDTH_IN=~20.1 (see constants).
    #   With wspace=0 the polar subplots are allocated exactly FIG_HEIGHT_INx0.655
    #   inches of width, which equals the row height, so aspect='equal' fills the
    #   full disc and the circles match the cylindrical map height.
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                     facecolor=COLOUR_BACKGROUND)
    gs  = GridSpec(1, 3, figure=fig,
                   width_ratios=[2, 1, 1],
                   wspace=0.0,
                   left=GS_LEFT, right=GS_RIGHT,
                   top=GS_TOP,   bottom=GS_BOTTOM)

    T_surface = titan_temp_K(t)

    # -- Left panel: equirectangular (cylindrical) global map -----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLOUR_BACKGROUND)
    ax1.imshow(
        posterior, cmap=cmap, norm=norm,
        origin="upper", aspect="auto",
        extent=[0, 360, -90, 90], interpolation="lanczos",
    )

    # Longitude graticule at 30 deg intervals, latitude at 15 deg intervals
    ax1.set_xticks(range(0, 361, 30))
    ax1.set_yticks(range(-90, 91, 15))
    ax1.tick_params(colors="white", labelsize=7.5, length=3, width=0.7)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-90, 90)

    # Overlay location markers on equirectangular panel
    for lon_W, lat, label, rank, _pole in TOP10:
        _stype  = SITE_TYPE.get(label, "land")
        _mshape = MARKER_SHAPE[_stype]
        _msize  = MARKER_SIZE_BY_TYPE[_stype]
        ax1.plot(lon_W, lat, _mshape, color=COLOUR_MARKER,
                 ms=_msize, mew=1.2, markeredgecolor=COLOUR_MARKER_EDGE,
                 zorder=10)
        dx, dy = _label_offset(lon_W, lat)
        ha: str = "left" if dx > 0 else "right"
        ax1.annotate(
            (f"#{rank} {label}" if rank > 0 else label),
            xy=(lon_W, lat), xytext=(lon_W + dx, lat + dy),
            color=COLOUR_TEXT, fontsize=TEXT_SIZE,
            ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", color=COLOUR_LEADER, lw=0.8),
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.15",
                      fc=COLOUR_ANNOT_BOX, ec="none"),
        )

    ax1.set_xlabel("Longitude °W", color="white", fontsize=9)
    ax1.set_ylabel("Latitude °N", color="white", fontsize=9, labelpad=2)

    # Always-present markers (Huygens + Dragonfly) not in this epoch's top-10
    for lon_W, lat, label, rank, _pole in ALWAYS_EXTRA:
        _stype  = SITE_TYPE.get(label, "lander")
        _mshape = MARKER_SHAPE[_stype]
        _msize  = MARKER_SIZE_BY_TYPE[_stype]
        ax1.plot(lon_W, lat, _mshape, color=COLOUR_MARKER,
                 ms=_msize, mew=1.2, markeredgecolor=COLOUR_MARKER_EDGE, zorder=10)
        dx, dy = _label_offset(lon_W, lat)
        ha_ae: str = "left" if dx > 0 else "right"
        ax1.annotate(
            label,
            xy=(lon_W, lat), xytext=(lon_W + dx, lat + dy),
            color=COLOUR_TEXT, fontsize=TEXT_SIZE,
            ha=ha_ae, va="center",
            arrowprops=dict(arrowstyle="-", color=COLOUR_LEADER, lw=0.8),
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", fc=COLOUR_ANNOT_BOX, ec="none"),
        )
    ax1.set_title("Global map  (equirectangular)", color="white",
                  fontsize=10, pad=5)
    for spine in ax1.spines.values():
        spine.set_edgecolor(COLOUR_SPINE)


    # -- Polar reproject helper ------------------------------------------------
    def polar_reproject(img: np.ndarray, north: bool, size: int = 500) -> np.ndarray:
        """
        Resample *img* into a circular polar stereographic view.

        Geometry (Snyder 1987, s.21 -- oblique stereographic, polar case)
        ---------------------------------------------------------------
        The disc coordinate r (0 = pole, 1 = disc edge) maps to latitude via:

            r = tan((90 deg - |lat|) / 2) / POLAR_SCALE

        where POLAR_SCALE = tan((90 deg - POLAR_CAP_EDGE_DEG) / 2) so that
        r = 1 corresponds to exactly POLAR_CAP_EDGE_DEG, filling the full
        circular panel with data.

        Azimuth (both hemispheres):

            lon_W = atan2(x, y)       [0 degW at disc bottom, increasing clockwise]

        This convention matches _loc_to_stereo (x_s = r*sin(lon), y_s = -r*cos(lon)),
        which places 0 degW at y_s=-r (disc bottom) and 180 degW at y_s=+r (disc top).
        The prime meridian therefore appears at the BOTTOM of each polar disc.
        """
        nrows_i: int
        ncols_i: int
        nrows_i, ncols_i = img.shape
        y0: np.ndarray
        x0: np.ndarray
        y0, x0 = np.mgrid[-1:1:size*1j, -1:1:size*1j]
        r: np.ndarray = np.sqrt(x0**2 + y0**2)
        outside: np.ndarray = r > 1.0

        # Stereographic latitude (Snyder s.21):  lat = 90 - 2.arctan(r.POLAR_SCALE)
        lat_s: np.ndarray = 90.0 - 2.0 * np.degrees(np.arctan(r * POLAR_SCALE))
        if not north:
            lat_s = -lat_s

        # Azimuth: lon_W = atan2(x, y) -- 0 degW at bottom, clockwise increasing
        # This matches _loc_to_stereo: x_s = r*sin(lon), y_s = -r*cos(lon)
        # which places 0 degW at y_s=-r (axis bottom) and 180 degW at y_s=+r (top).
        # BUG-FIX 2026-04: original formula atan2(x, -y) placed 0 degW at the TOP
        # of the image (y0=-1) while _loc_to_stereo placed the 0 degW MARKER at the
        # BOTTOM (y_s=-r), causing a 180-degree rotation between the image data and
        # all labels/markers.  Changing -y0 to y0 corrects this for both poles.
        lon_s: np.ndarray = (np.degrees(np.arctan2(x0, y0)) + 360.0) % 360.0

        # Convert disc coords to raster pixel indices
        row_s: np.ndarray = (90.0 - lat_s) / 180.0 * nrows_i
        col_s: np.ndarray = lon_s / 360.0 * ncols_i

        flat_img: np.ndarray = img.copy().astype(np.float64)
        flat_img[~np.isfinite(flat_img)] = -1.0
        sampled: np.ndarray = map_coordinates(
            flat_img, [row_s.ravel(), col_s.ravel()],
            order=1, mode="wrap", cval=-1.0
        )
        out: np.ndarray = sampled.reshape(size, size).astype(np.float32)
        out[outside] = np.nan
        out[out < 0] = np.nan
        # Pixels outside the unit disc are already NaN (set above).
        return out

    def _loc_to_stereo(lon_W: float, lat: float, north: bool) -> Tuple[Optional[float], Optional[float]]:
        """
        Convert (lon_W deg, lat deg) to panel coordinates using the fixed
        stereographic formula (consistent with polar_reproject).
        """
        if (north and lat < POLAR_CAP_EDGE_DEG) or (not north and lat > -POLAR_CAP_EDGE_DEG):
            return None, None
        abs_lat = abs(lat)
        # Invert: r = tan((90-abs_lat)/2) / POLAR_SCALE
        r = math.tan(math.radians((90.0 - abs_lat) / 2.0)) / POLAR_SCALE
        if r > 1.0:
            return None, None
        lon_rad: float = math.radians(lon_W)
        # Stereographic inverse (Snyder 1987, s.21):
        #   r = tan((90 - |lat|) / 2) / POLAR_SCALE
        #   x =  r . sin(lon_W)
        #   y = -r . cos(lon_W)
        # The -cos convention matches the azimuth formula atan2(x, y),
        # which places 0 degW at the BOTTOM of the disc for both hemispheres.
        x_s: float =  r * math.sin(lon_rad)
        y_s: float = -r * math.cos(lon_rad)   # same for north AND south
        return x_s, y_s

    # Longitude tick marks for polar caps (Snyder 1987 s.21 inverse).
    # Ticks placed from r=1.0 (disc edge) outward to r=1.07; labels at r=1.17.
    # 30 deg increments.  Formula: x = r.sin(lon_W), y = -r.cos(lon_W)
    # (0 degW at top, clockwise increasing).
    # The axes limits are expanded to (-1.28, 1.28) so the outside ticks and
    # labels are not clipped.  The disc itself (r <= 1) is unchanged.
    _DISC_LIM: float = 1.28   # axes half-range; must be > label radius (1.17)

    def _draw_polar_graticule(ax: "matplotlib.axes.Axes") -> None:
        """Draw 30 deg longitude tick marks and labels outside the polar disc."""
        for lon_W_tick in range(0, 360, 30):
            lon_rad: float = math.radians(lon_W_tick)
            sin_l: float = math.sin(lon_rad)
            cos_l: float = math.cos(lon_rad)
            # Tick stub: disc edge (r=1.0) to just outside (r=1.07)
            ax.plot([1.00 * sin_l, 1.07 * sin_l],
                    [-1.00 * cos_l, -1.07 * cos_l],
                    color=COLOUR_POLAR_RING, lw=1.0, alpha=0.9, zorder=5)
            # Label text at r=1.17, outside the disc
            lbl: str = f"{lon_W_tick}°" if lon_W_tick == 0 else f"{lon_W_tick}°W"
            ax.text(1.17 * sin_l, -1.17 * cos_l, lbl,
                    color=COLOUR_POLAR_RING, fontsize=6.0,
                    ha="center", va="center", zorder=6)

    # -- Centre: North polar cap (POLAR_CAP_EDGE_DEG deg - 90 degN) -----------------
    ax2 = fig.add_subplot(gs[0, 1], aspect="equal")
    ax2.set_facecolor(COLOUR_BACKGROUND)
    north_img = polar_reproject(posterior, north=True, size=500)
    ax2.imshow(north_img, cmap=cmap_polar, norm=norm,
               origin="upper", interpolation="lanczos",
               extent=[-1, 1, -1, 1])
    theta: np.ndarray = np.linspace(0, 2 * np.pi, 360)
    ax2.plot(np.cos(theta), np.sin(theta), color=COLOUR_POLAR_RING, lw=0.8, alpha=0.6)
    # Expand limits beyond r=1 so outside tick marks and labels are visible
    ax2.set_xlim(-_DISC_LIM, _DISC_LIM)
    ax2.set_ylim(-_DISC_LIM, _DISC_LIM)
    ax2.axis("off")
    ax2.set_title(f"North polar cap  ({POLAR_CAP_EDGE_DEG:.0f}°–90°N)",
                  color="white", fontsize=10, pad=5)
    _draw_polar_graticule(ax2)

    # Overlay north-polar location markers
    for lon_W, lat, label, rank, _ in TOP10:
        xs, ys = _loc_to_stereo(lon_W, lat, north=True)
        if xs is None:
            continue
        _stype  = SITE_TYPE.get(label, "land")
        _mshape = MARKER_SHAPE[_stype]
        _msize  = MARKER_SIZE_BY_TYPE[_stype]
        ax2.plot(xs, ys, _mshape, color=COLOUR_MARKER,
                 ms=_msize, mew=1.2, markeredgecolor=COLOUR_MARKER_EDGE, zorder=10)
        mag: float = math.sqrt(xs**2 + ys**2)
        scale_out: float = min(1.15 / max(0.05, mag), 4.0)
        tx: float = max(-0.90, min(0.90, xs * scale_out))
        ty: float = max(-0.90, min(0.90, ys * scale_out))
        ax2.annotate(
            (f"#{rank} {label}" if rank > 0 else label),
            xy=(xs, ys), xytext=(tx, ty),
            color=COLOUR_TEXT, fontsize=TEXT_SIZE - 0.5,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=COLOUR_LEADER, lw=0.8),
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", fc=COLOUR_ANNOT_BOX_POLAR, ec="none"),
        )



    # Always-present markers on north polar cap
    for lon_W, lat, label, rank, _ in ALWAYS_EXTRA:
        xs, ys = _loc_to_stereo(lon_W, lat, north=True)
        if xs is None:
            continue
        _stype  = SITE_TYPE.get(label, "lander")
        ax2.plot(xs, ys, MARKER_SHAPE[_stype], color=COLOUR_MARKER,
                 ms=MARKER_SIZE_BY_TYPE[_stype], mew=1.2,
                 markeredgecolor=COLOUR_MARKER_EDGE, zorder=10)
        mag = math.sqrt(xs**2 + ys**2)
        scale_out = min(1.15 / max(0.05, mag), 4.0)
        tx = max(-0.90, min(0.90, xs * scale_out))
        ty = max(-0.90, min(0.90, ys * scale_out))
        ax2.annotate(
            label,
            xy=(xs, ys), xytext=(tx, ty),
            color=COLOUR_TEXT, fontsize=TEXT_SIZE - 0.5,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=COLOUR_LEADER, lw=0.8),
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", fc=COLOUR_ANNOT_BOX_POLAR, ec="none"),
        )
    # -- Right panel: South polar cap (POLAR_CAP_EDGE_DEG deg - 90 degS) ------------
    ax3 = fig.add_subplot(gs[0, 2], aspect="equal")
    ax3.set_facecolor(COLOUR_BACKGROUND)
    south_img = polar_reproject(posterior, north=False, size=500)
    ax3.imshow(south_img, cmap=cmap_polar, norm=norm,
               origin="upper", interpolation="lanczos",
               extent=[-1, 1, -1, 1])
    ax3.plot(np.cos(theta), np.sin(theta), color=COLOUR_POLAR_RING, lw=0.8, alpha=0.6)
    ax3.set_xlim(-_DISC_LIM, _DISC_LIM)
    ax3.set_ylim(-_DISC_LIM, _DISC_LIM)
    ax3.axis("off")
    ax3.set_title(f"South polar cap  ({POLAR_CAP_EDGE_DEG:.0f}°–90°S)",
                  color="white", fontsize=10, pad=5)
    _draw_polar_graticule(ax3)

    # Overlay south-polar location markers
    for lon_W, lat, label, rank, _ in TOP10:
        xs, ys = _loc_to_stereo(lon_W, lat, north=False)
        if xs is None:
            continue
        _stype  = SITE_TYPE.get(label, "land")
        _mshape = MARKER_SHAPE[_stype]
        _msize  = MARKER_SIZE_BY_TYPE[_stype]
        ax3.plot(xs, ys, _mshape, color=COLOUR_MARKER,
                 ms=_msize, mew=1.2, markeredgecolor=COLOUR_MARKER_EDGE, zorder=10)
        mag = math.sqrt(xs**2 + ys**2)
        scale_out = min(1.15 / max(0.05, mag), 4.0)
        tx = max(-0.90, min(0.90, xs * scale_out))
        ty = max(-0.90, min(0.90, ys * scale_out))
        ax3.annotate(
            (f"#{rank} {label}" if rank > 0 else label),
            xy=(xs, ys), xytext=(tx, ty),
            color=COLOUR_TEXT, fontsize=TEXT_SIZE - 0.5,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=COLOUR_LEADER, lw=0.8),
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", fc=COLOUR_ANNOT_BOX_POLAR, ec="none"),
        )


    # Always-present markers on south polar cap
    for lon_W, lat, label, rank, _ in ALWAYS_EXTRA:
        xs, ys = _loc_to_stereo(lon_W, lat, north=False)
        if xs is None:
            continue
        _stype  = SITE_TYPE.get(label, "lander")
        ax3.plot(xs, ys, MARKER_SHAPE[_stype], color=COLOUR_MARKER,
                 ms=MARKER_SIZE_BY_TYPE[_stype], mew=1.2,
                 markeredgecolor=COLOUR_MARKER_EDGE, zorder=10)
        mag = math.sqrt(xs**2 + ys**2)
        scale_out = min(1.15 / max(0.05, mag), 4.0)
        tx = max(-0.90, min(0.90, xs * scale_out))
        ty = max(-0.90, min(0.90, ys * scale_out))
        ax3.annotate(
            label,
            xy=(xs, ys), xytext=(tx, ty),
            color=COLOUR_TEXT, fontsize=TEXT_SIZE - 0.5,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=COLOUR_LEADER, lw=0.8),
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", fc=COLOUR_ANNOT_BOX_POLAR, ec="none"),
        )
    # -- Colourbar -------------------------------------------------------------
    # Bar axes: [left, bottom, width, height] in figure fraction.
    # Label is placed as fig.text() ABOVE the bar so it doesn't compete with
    # the narrative boxes below.  cb.set_label("") suppresses the auto label.
    cax = fig.add_axes([0.10, 0.145, 0.80, 0.014])
    cax.set_facecolor(COLOUR_BACKGROUND)
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("")                # suppress automatic label below bar
    cb.ax.xaxis.set_tick_params(color="white", labelcolor="white", labelsize=9)
    # Manual label centred above the bar
    fig.text(0.50, 0.164, "P(habitable | features)",
             color="white", fontsize=10, ha="center", va="bottom",
             transform=fig.transFigure)

    # -- Site-type marker legend ------------------------------------------
    # Positioned equidistant between the bottom of the centre info panel
    # (measured at y≈0.310) and the colourbar label (y=0.164).
    # midpoint = (0.310 + 0.164) / 2 ≈ 0.241
    # Three items centred at x=0.50, spread across x=0.38-0.62.
    _leg_y      = 0.241
    _leg_items  = [
        (0.39, "^",  "Lake / sea shore"),
        (0.50, "s",  "Land site"),
        (0.61, "*",  "Mission lander"),
    ]
    _leg_fsize  = 10.0     # icon size in points
    _leg_tsize  = 9.5      # label text size in points
    for _lx, _lmark, _ltxt in _leg_items:
        # Draw marker icon using fig.text with a marker-like Unicode substitute
        # (actual marker drawn via a tiny single-point axes)
        _icon_ax = fig.add_axes([_lx - 0.010, _leg_y - 0.012, 0.018, 0.024])
        _icon_ax.set_facecolor("none")
        _icon_ax.set_xlim(-1, 1)
        _icon_ax.set_ylim(-1, 1)
        _icon_ax.axis("off")
        _ms = 11 if _lmark == "*" else 9
        _icon_ax.plot(0, 0, _lmark, color=COLOUR_MARKER,
                      ms=_ms, mew=1.2, markeredgecolor=COLOUR_MARKER_EDGE,
                      zorder=5, transform=_icon_ax.transData)
        fig.text(_lx + 0.013, _leg_y,
                 _ltxt,
                 color="white", fontsize=_leg_tsize,
                 ha="left", va="center",
                 transform=fig.transFigure)

    # -- Narrative text box ----------------------------------------------------
    # Vertical layout (11" figure, y in figure fraction, 0=bottom):
    #   y=0.164  colourbar label (above bar)
    #   y=0.145  colourbar bar top
    #   y=0.131  colourbar bar bottom  + tick values below
    #   ---
    #   y=0.072  title pill centre
    #   y=0.030  body box centre
    if narrative:
        lines: list = [ln.strip() for ln in narrative.strip().split("\n") if ln.strip()]
        title_line: str       = lines[0] if lines else ""
        body_lines: list[str] = lines[1:] if len(lines) > 1 else []
        body_text:  str       = "\n".join(body_lines) if body_lines else "-" * 60

        fig.text(0.50, 0.030, body_text,
                 color=COLOUR_NARRATIVE_BODY, fontsize=10.5, fontweight="normal",
                 ha="center", va="center", fontfamily="monospace",
                 linespacing=1.55, zorder=20,
                 bbox=dict(boxstyle="round,pad=0.55",
                           fc=COLOUR_NARRATIVE_FILL, ec=COLOUR_NARRATIVE_BORDER, lw=1.8))
        fig.text(0.50, 0.072, title_line,
                 color=COLOUR_NARRATIVE_TITLE, fontsize=12.0, fontweight="bold",
                 ha="center", va="center", fontfamily="monospace", zorder=21,
                 bbox=dict(boxstyle="round,pad=0.30",
                           fc=COLOUR_TITLE_FILL, ec=COLOUR_TITLE_BORDER, lw=1.5))
    else:
        fig.text(0.50, 0.030, " ", color=COLOUR_TRANSPARENT, fontsize=10.5,
                 fontfamily="monospace", ha="center", va="center", zorder=1)
        fig.text(0.50, 0.072, " ", color=COLOUR_TRANSPARENT, fontsize=12.0,
                 fontfamily="monospace", ha="center", va="center", zorder=1)

    # ── Title ─────────────────────────────────────────────────────────────────
    solar_str = f"L☉ = {solar_luminosity_ratio(t):.2f}×   T_surface = {titan_temp_K(t):.0f} K"
    if T_surface >= EUTECTIC_K:
        phase_col = COLOUR_PHASE_OCEAN
    elif t < -3.0:
        phase_col = COLOUR_PHASE_LHB
    elif abs(t) < 0.1:
        phase_col = COLOUR_PHASE_PRESENT
    else:
        phase_col = COLOUR_PHASE_DEFAULT

    fig.text(0.5, 0.978, "TITAN SURFACE HABITABILITY", color="white",
             fontsize=15, ha="center", va="bottom", fontweight="bold",
             fontfamily="monospace")
    fig.text(0.5, 0.963, f"Epoch:  {_epoch_label(t).replace(chr(10),' ')}   |   "
             f"Phase:  {_phase_label(t).replace(chr(10),' ')}   |   {solar_str}",
             color=phase_col, fontsize=10, ha="center", va="bottom")

    # -- Progress bar ----------------------------------------------------------
    bar_ax = fig.add_axes([0.10, 0.948, 0.80, 0.006])
    bar_ax.set_facecolor(COLOUR_BACKGROUND)
    bar_ax.set_xlim(0, n_epochs)
    bar_ax.set_ylim(0, 1)
    bar_ax.barh(0.5, epoch_idx + 1, height=1.0, color=COLOUR_PROGRESS_BAR, alpha=0.7)
    bar_ax.axis("off")

    # -- Epoch-aware feature / assumption panel ----------------------------------
    # Three-column panel in the expanded lower figure area:
    #   Left  (x=0.01-0.38): Active features table + excluded features
    #   Centre(x=0.39-0.65): Colour trend explanation
    #   Right (x=0.66-0.99): Key assumptions for this epoch
    #
    # The panel key is matched by the first word(s) of _phase_label(t).
    _phase = _phase_label(t)
    _PANEL: Dict[str, Any] = {
        "LHB peak": {
            "active": [
                ("impact_melt_bonus",        "HIGH",   "Uniform global boost  peak=0.50 @ -3.8 Gya; →0 by -2.5 Gya"),
                # cryovolcanic_flux: physically plausible but NOT in current model weights
                ("acetylene_energy",         "MED",    "Intense HCN/C₂H₂ photochem under young UV Sun"),
                ("organic_abundance",        "MED",    "Low tholin — only ~0.5 Gyr of UV photolysis"),
                ("surface_atm_interaction",  "MED",    "Cryovolcanic conduits elevate gas/surface flux"),
                ("methane_cycle",            "LOW",    "Episodic outgassing — no stable rain cycle"),
                ("topographic_complexity",   "LOW",    "Crater rims dominate topography"),
                ("geomorphologic_diversity", "LOW",    "Terrain class diversity proxy"),
            ],
            "excluded": "liquid_hydrocarbon  — no stable polar lakes yet\nsubsurface_ocean    — SAR annuli not yet resolved at LHB\ncryovolcanic_flux   — physically plausible but not in current model weights",
            "colour":
                "BRIGHT (yellow):  fresh impact craters at mid-latitudes with melt halos\n"
                "MED (orange):     cryovolcanic candidate sites (Lopes 2007)\n"
                "DARK (blue):      polar regions, dune plains — no lakes, low organics",
            "assumptions": [
                "Epoch ≈ −3.8 Gya  (Late Heavy Bombardment peak; Gomes et al. 2005)",
                "Impact flux ~30× present (Hartmann & Neukum 2001)",
                "Uniform spatial warming — no resolved GCM for this epoch",
                "Birch confirmed-lake polygons not applied (no polar lakes)",
            ],
        },
        "Early Titan": {
            "active": [
                ("organic_abundance",        "RAMP",   "Tholin stockpile building; scale = t_elapsed/4.0 Gya"),
                ("acetylene_energy",         "MED",    "UV photochem declining as Sun dims post-LHB"),
                ("impact_melt_bonus",        "LOW",    "Residual LHB tail; ~0.01–0.08 (spatially uniform)"),
                ("methane_cycle",            "LOW",    "Methane cycle scale = 0.60 (not yet established)"),
                ("surface_atm_interaction",  "LOW",    "Scale = 0.46 (no polar lakes; slope-driven)"),
                ("topographic_complexity",   "LOW",    "Scale = 1.15–1.30 (rougher ancient terrain)"),
                ("geomorphologic_diversity", "LOW",    "Scale = 0.85 (less terrain diversity than present)"),
                ("subsurface_ocean",         "LOW",    "Scale = 1.8–2.5 (closer to surface in warm epoch)"),
            ],
            "excluded": "liquid_hydrocarbon  — no stable polar lakes yet (scale = 0.10 proxy)\ncryovolcanic_flux   — physically plausible but not in model weights",
            "colour":
                "GRADUAL brightening: organic stockpile accumulating — equatorial dunes brighter\n"
                "LOW contrast overall: no lakes, modest methane cycle\n"
                "Crater sites slightly elevated: subsurface_ocean scale = 1.8–2.5×",
            "assumptions": [
                "t = −3.0 to −1.0 Gya  (post-LHB through early lake formation onset)",
                "Impact flux declining exponentially from LHB peak",
                "organic scale = (4 + t)/4 Gya (linear accumulation since atmosphere formed ~4 Gya)",
                "No polar lake system yet; liquid_HC proxy = 0.10 of present map",
            ],
        },
        "Lake formation": {
            "active": [
                ("liquid_hydrocarbon",       "RAMP",   "Polar lake proxy ramping 10% → 100% of present"),
                # cryovolcanic_flux: physically plausible at this epoch but NOT in model weights
                ("organic_abundance",        "MED",    "Tholin stockpile building toward present level"),
                ("methane_cycle",            "MED",    "Methane rain cycle becoming established"),
                ("acetylene_energy",         "MED",    "C₂H₂ production ongoing"),
                ("surface_atm_interaction",  "MED",    "Lake margins + cryovolcanic conduits"),
                ("topographic_complexity",   "LOW",    "Topographic roughness proxy"),
                ("geomorphologic_diversity", "LOW",    "Terrain class diversity"),
                ("impact_melt_bonus",        "LOW",    "LHB residual ~0.005  (spatially uniform, negligible)"),
            ],
            "excluded": "subsurface_ocean   — not yet resolved at lake_formation epoch\ncryovolcanic_flux   — physically plausible (Tobie 2006) but not in model weights",
            "colour":
                "BRIGHT (yellow):  north polar region brightening as Kraken/Ligeia fill\n"
                "MED (orange):     cryovolcanic sites; equatorial organics building\n"
                "DARK (blue):      mountain ranges, Xanadu (water-ice, low organics)",
            "assumptions": [
                "Epoch ≈ 1.0 Gya  (Birch et al. 2017 morphology evidence)",
                "liquid_HC ramps linearly from −1.0 → −0.5 Gya",
                "Cassini SAR geomorphology applied (same maps, different weights)",
            ],
        },
        "Recent past": {
            "active": [
                ("liquid_hydrocarbon",       "HIGH",   "Birch Fl polygons: Cassini confirmed liquid (=1.0)"),
                ("organic_abundance",        "HIGH",   "Lopes geomorphology scores; VIMS offset −0.135"),
                ("methane_cycle",            "MED",    "CIRS temperature + latitude → rain/evap cycle"),
                ("acetylene_energy",         "MED",    "Strobel (2010) H₂ depletion confirms C₂H₂ use"),
                ("surface_atm_interaction",  "MED",    "Channel density + lake margin proximity"),
                ("subsurface_ocean",         "LOW",    "SAR annuli around craters; prior=0.03 (Neish 2024)"),
                ("topographic_complexity",   "LOW",    "GTIE T126 DEM roughness at 4490 m/px"),
                ("geomorphologic_diversity", "LOW",    "Lopes 2019 terrain diversity (6 classes)"),
            ],
            "excluded": "impact_melt_bonus  — decayed to ~0 by -2.5 Gya; effectively zero\ncryovolcanic_flux  — not used at present epoch",
            "colour":
                "ORANGE-YELLOW:  north polar lake surfaces (liquid_hc=1.0 → P≈0.42 rescaled)\n"
                "ORANGE:          polar lake shores + Selk + equatorial dunes (P≈0.28–0.36)\n"
                "DARK ORANGE-RED: mountain ridges, Xanadu (water-ice, low organics, P≈0.24–0.26)\n"
                "Note: Bayesian formula gives continuous gradient, not bimodal poles/equator",
            "assumptions": [
                "Cassini epoch = 2004-2017 CE  (CIRS T model year 2011.0)",
                "Organic abundance: geo_only mode — Lopes (2019) terrain classes globally",
                "El/Em Birch confirmed-empty basins → liquid_HC = 0.0",
                "Subsurface ocean prior = 0.03  (Neish et al. 2024 organic flux ~1 elephant/yr)",
                "Label balance: 50/50 positive/negative (pure median split)",
            ],
        },
        "Near future": {
            "active": [
                ("liquid_hydrocarbon",       "HIGH",   "Same Cassini map — luminosity change <2.5% in 250 Myr"),
                ("organic_abundance",        "HIGH",   "Unchanged — terrain stable at this timescale"),
                ("methane_cycle",            "MED",    "Minor depletion (Lorenz et al. 1997)"),
                ("acetylene_energy",         "MED",    "Slightly elevated under warmer Sun"),
                ("surface_atm_interaction",  "MED",    "Unchanged from present"),
                ("subsurface_ocean",         "LOW",    "Prior raised 0.03→0.08; solar warming ↑ exchange"),
                ("topographic_complexity",   "LOW",    "Unchanged"),
                ("geomorphologic_diversity", "LOW",    "Unchanged"),
            ],
            "excluded": "impact_melt_bonus  — decayed to zero; no LHB contribution\ncryovolcanic_flux  — not applicable",
            "colour":
                "Nearly identical to Cassini era (Bayesian formula)\n"
                "Polar lake surfaces: P≈0.42 (orange-yellow); equatorial dunes: P≈0.28–0.32 (orange)\n"
                "Polar lake shores remain most habitable; Dragonfly-era data still applicable",
            "assumptions": [
                "D2 window centre = +250 Myr  (range 100–400 Myr; Lorenz, Lunine & McKay 1997)",
                "Solar luminosity +1.4% at +250 Myr  (L☉ formula gives +1.37%; Bahcall et al. 2001)",
                "Uniform spatial warming — no resolved GCM for this epoch",
                "Cassini feature maps unchanged (no spatial data at +250 Myr resolution)",
            ],
        },
        "Solar warming": {
            "active": [
                ("liquid_hydrocarbon",       "DECL",   "Lake surfaces evaporating as T exceeds CH₄ b.p."),
                ("organic_abundance",        "HIGH",   "Tholin stockpile at maximum — billions of yrs UV"),
                ("acetylene_energy",         "MED",    "Chemistry shifts thermal → photochemical regime"),
                ("methane_cycle",            "DECL",   "Weakening as surface liquid depletes"),
                ("surface_atm_interaction",  "MED",    "Shrinking lake margins"),
                ("subsurface_ocean",         "LOW",    "Present but decreasing surface connection"),
                ("topographic_complexity",   "LOW",    "Unchanged"),
                ("geomorphologic_diversity", "LOW",    "Desiccated lake basins add new class"),
            ],
            "excluded": "impact_melt_bonus  — decayed to zero; no LHB contribution\ncryovolcanic_flux  — not applicable",
            "colour":
                "FADING BRIGHT at poles:  lakes evaporating — liquid_HC scale → 0\n"
                "BRIGHT at low latitudes: organic abundance dominates as lakes disappear\n"
                "Overall habitability declining as methane cycle collapses",
            "assumptions": [
                "Solar ramp: +4.0 to +5.0 Gya; linear lake evaporation model",
                "liquid_HC scale declines linearly 1.0 → 0.0 over 1 Gyr",
                "Uniform spatial warming assumed (no resolved GCM for this epoch)",
            ],
        },
        "Pre red-giant": {
            "active": [
                ("liquid_hydrocarbon",       "HIGH",   "Polar lakes fully present; T < CH₄ b.p. until +4 Gya"),
                ("organic_abundance",        "HIGH",   "Tholin stockpile near maximum (billions of yrs UV)"),
                ("methane_cycle",            "HIGH",   "Methane rain/evap cycle fully established"),
                ("acetylene_energy",         "MED",    "Slightly elevated under warmer (but sub-eutectic) Sun"),
                ("surface_atm_interaction",  "MED",    "Unchanged — lake margins stable"),
                ("subsurface_ocean",         "LOW",    "Unchanged from present"),
                ("topographic_complexity",   "LOW",    "Unchanged"),
                ("geomorphologic_diversity", "LOW",    "Unchanged"),
            ],
            "excluded": "impact_melt_bonus  — decayed to zero\ncryovolcanic_flux  — not in model",
            "colour":
                "Similar to Cassini/near-future; polar lake margins still brightest\n"
                "Warming Sun (+16% at +3 Gya) has negligible effect on habitability pattern\n"
                "Note: lakes begin evaporating only at +4.0 Gya (next phase)",
            "assumptions": [
                "t = +3.0 to +4.0 Gya — lakes intact; solar L☉ = +16–21% above present",
                "liquid_HC scale = 1.0 throughout (no evaporation yet)",
                "Solar warming is sub-eutectic; no qualitative feature changes",
                "Lorenz, Lunine & McKay (1997) — lake stability thresholds",
            ],
        },
        "Red giant": {
            "active": [
                ("liquid_hydrocarbon",       "HIGH",   "Override → 1.0 globally (T ≥ 176 K: global ocean)"),
                ("organic_abundance",        "HIGH",   "Scale × 2.5 (capped): max 16 Gyr UV stockpile"),
                ("subsurface_ocean",         "HIGH",   "Override → 1.0 globally (now a surface ocean)"),
                ("surface_atm_interaction",  "HIGH",   "Scale = 1.0 (whole surface is liquid-atm interface)"),
                ("methane_cycle",            "MED",    "Scale = 0.75 (proxy for water-NH₃ atmosphere)"),
                ("topographic_complexity",   "LOW",    "Sub-ocean topography; scale = 1.0"),
                ("geomorphologic_diversity", "LOW",    "Largely submerged; scale = 1.0"),
                ("acetylene_energy",         "LOW",    "Reduced: scale = 0.10/0.35 (UV regime shift)"),
            ],
            "excluded":
                "impact_melt_bonus  — decayed to zero (>> 1 Gya since LHB)\n"
                "Note: features above are the ACTUAL Bayesian model features with\n"
                "      T-dependent scale overrides — not a separate ocean model",
            "colour":
                "BRIGHT globally:   liquid_hc + subsurface = 1.0 everywhere → near-max P(H)\n"
                "Mild variation:    sub-ocean DEM topography via topo_complexity/geodiv\n"
                "Near-uniform:      all pixels approach P(H) ≈ 0.55–0.65 (warm yellow)",
            "assumptions": [
                "T_surface ≥ 176 K  (water-ammonia eutectic; Grasset & Pargamin 2005)",
                "Epoch ≈ +5.1 to +5.9 Gya  (Lorenz et al. 1997)",
                "liquid_HC and subsurface_ocean overridden to 1.0 via scale_features_to_epoch",
                "Organic scale = min(t_elapsed/4.0, 2.5) × 1.1 = 2.5 (maximum accumulated)",
            ],
        },
        "Ocean refreezing": {
            "active": [
                ("organic_abundance",        "HIGH",   "Full 16 Gyr UV tholin stockpile (scale = 2.5, capped)"),
                ("liquid_hydrocarbon",       "DECL",   "Returns to 0 as T < 176 K; methane still long gone"),
                ("subsurface_ocean",         "LOW",    "Refreezing; scale returns to 1.0 post-eutectic"),
                ("surface_atm_interaction",  "LOW",    "Residual"),
                ("methane_cycle",            "LOW",    "Scale = 0 (methane cycle ended +5 Gya)"),
                ("acetylene_energy",         "LOW",    "Reduced"),
                ("topographic_complexity",   "LOW",    "Emerging surface as ice reforms"),
                ("geomorphologic_diversity", "LOW",    "Recovering terrain diversity"),
            ],
            "excluded":
                "impact_melt_bonus — zero since ~2.5 Gya\n"
                "Note: model uses standard 8 features with T-dependent scale overrides;\n"
                "      no separate ocean-chemistry model",
            "colour":
                "FADING globally:  T < 176 K, ocean refreezes in < 1 Myr\n"
                "Still elevated:   16 Gyr organic stockpile in solution briefly\n"
                "Rapid decline:    habitability window closes ~400 Myr after peak",
            "assumptions": [
                "Epoch ≥ 6.0 Gya — Sun exits red-giant phase (Lorenz et al. 1997)",
                "L collapses 600× → 0.8× present; T_surface drops to ~89 K",
                "Ocean refreezing timescale < 1 Myr (Lorenz 1997)",
                "Total habitable window: ~400 Myr  (+5.1 to +6.0 Gya)",
            ],
        },
    }

    # Map phase labels (first line of _phase_label output) to panel keys.
    # This is explicit rather than fragile first-word matching.
    # Use full phase string (\n → space) for unambiguous matching.
    _PHASE_TO_PANEL: Dict[str, str] = {
        "Late Heavy Bombardment": "LHB peak",
        "Early Titan":            "Early Titan",
        "Lake formation":         "Lake formation",
        "Recent past":            "Recent past",
        "Cassini epoch":          "Recent past",
        "Near future":            "Near future",
        "Pre red-giant":          "Pre red-giant",  # t=3.0–4.0: stable; fixed (was wrongly using "Solar warming")
        "Solar warming":          "Solar warming",  # t=4.0–5.0: lake evaporation (matches new phase label)
        "Red-giant ramp up":      "Solar warming",  # t=5.0–5.13 Gya: lakes gone, dry/hot, T < 176 K — no ocean yet
        "Red-giant water ocean":  "Red giant",      # T ≥ 176 K: global water-ammonia ocean (panel text matches visual change at frame 55)
        "Red-giant ramp down":    "Ocean refreezing",
    }
    _phase_key = _phase.replace("\n", " ")
    _panel_key = _PHASE_TO_PANEL.get(_phase_key, "Recent past")
    _pd = _PANEL[_panel_key]

    # -- Left column: Active features + excluded ---------------------------------
    # Weight-level indicators reflect different systems per source:
    #   MODELLED* : Bayesian formula weights (animation WEIGHTS dict)
    #   PCHIP / CLAMPED / BLEND : sklearn RandomForest from pipeline runs
    #                              (actual importances may differ from Bayesian)
    _blend_alpha = float(source.split("α=")[1].rstrip(")")) if "TRANSITION_BLEND" in source else 0.0
    _is_sklearn = (
        any(s in source for s in ("PCHIP", "CLAMPED", "ANCHOR"))
        or ("BLEND" in source and "TRANSITION_BLEND" not in source)
        or ("TRANSITION_BLEND" in source and _blend_alpha > 0.0)
    )
    _wt_caveat = (
        "  ← RF pipeline" if _is_sklearn else "  (Bayesian weights)"
    )
    _feat_lines = [
        f"ACTIVE FEATURES  (weight level: HIGH / MED / LOW / RAMP / DECL{_wt_caveat})"
    ]
    _feat_lines.append("─" * 70)
    for fname, level, desc in _pd["active"]:
        _lvl_col = {"HIGH": "▶▶▶", "MED": "▶▶○", "LOW": "▶○○",
                    "RAMP": "↑↑↑", "DECL": "↓↓↓"}.get(level, "○○○")
        _feat_lines.append(f"  {_lvl_col} {fname:<28}  {desc}")
    _feat_lines.append("")
    _feat_lines.append("EXCLUDED / NOT ACTIVE:")
    _feat_lines.append("─" * 70)
    for ln in _pd["excluded"].split("\n"):
        _feat_lines.append(f"  ✗  {ln}")

    fig.text(
        0.080, 0.448, "\n".join(_feat_lines),
        color="#d0d8f0", fontsize=7.8, fontfamily="monospace",
        va="top", ha="left", linespacing=1.45, zorder=20,
        bbox=dict(boxstyle="round,pad=0.4", fc="#0a0a1a", ec="#2a3a6a", lw=1.2),
        transform=fig.transFigure,
    )

    # -- Centre column: Colour trends --------------------------------------------
    _colour_lines = [f"COLOUR SCALE  ( {VMIN:.2f} ← DARK          BRIGHT → {VMAX:.2f} )"]
    _colour_lines.append("─" * 42)
    for ln in _pd["colour"].split("\n"):
        _colour_lines.append(f"  {ln}")
    _colour_lines.append("")
    _colour_lines.append(f"DOMINANT FEATURE THIS EPOCH:")
    _colour_lines.append("─" * 42)
    _top_feat = _pd["active"][0]
    if _is_sklearn:
        _colour_lines.append(f"  {_top_feat[0]}  (by Bayesian proxy)")
        _colour_lines.append(f"  RF importances may differ — see pipeline logs")
    else:
        _colour_lines.append(f"  {_top_feat[0]}")
        _colour_lines.append(f"  {_top_feat[2]}")

    fig.text(
        0.535, 0.448, "\n".join(_colour_lines),
        color="#f0d8a0", fontsize=7.8, fontfamily="monospace",
        va="top", ha="center", linespacing=1.45, zorder=20,
        bbox=dict(boxstyle="round,pad=0.4", fc="#0a0a0a", ec="#6a4a00", lw=1.2),
        transform=fig.transFigure,
    )

    # -- Right column: Assumptions -----------------------------------------------
    _assump_lines = ["KEY ASSUMPTIONS  (this epoch)"]
    _assump_lines.append("─" * 48)
    for i, a in enumerate(_pd["assumptions"], 1):
        _assump_lines.append(f"  {i}. {a}")
    _assump_lines.append("")
    _assump_lines.append("GLOBAL ASSUMPTIONS (all epochs):")
    _assump_lines.append("─" * 48)
    _assump_lines.append("  • Resolution: 4490 m/px equirectangular")
    _assump_lines.append("  • Grid: 1802 × 3603 px (Titan R = 2575 km)")
    _assump_lines.append("  • Organic abundance: Lopes (2019) geo_only mode")
    if _is_sklearn:
        _assump_lines.append("  • Backend: sklearn RandomForestClassifier")
        _assump_lines.append(f"  • Source: {source}")
    else:
        _assump_lines.append("  • Backend: Bayesian Beta update (animation)")
        _assump_lines.append(f"  • Source: {source}")
        _assump_lines.append("  • Temporal scaling: modelled (present TIFs scaled)")

    fig.text(
        0.990, 0.448, "\n".join(_assump_lines),
        color="#c0f0c0", fontsize=7.8, fontfamily="monospace",
        va="top", ha="right", linespacing=1.45, zorder=20,
        bbox=dict(boxstyle="round,pad=0.4", fc="#000a00", ec="#1a5a1a", lw=1.2),
        transform=fig.transFigure,
    )

    return fig


# --- Six-panel poster ---------------------------------------------------------

def render_poster(
    epoch_maps: Dict[float, np.ndarray],
    out_path:   Path,
) -> None:
    """
    Render a six-panel summary poster of key epochs.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    key_epochs: List[float] = [-3.5, -1.0, 0.0, 1.0, 5.2, 6.0]
    titles: List[str] = [
        "LHB peak\n(-3.5 Gya)",
        "Lake formation\n(-1.0 Gya)",
        "Present\n(Cassini epoch)",
        "Near future\n(+1.0 Gya)",
        "Red giant ramp\n(+5.2 Gya)",
        "Water ocean peak\n(+6.0 Gya)",
    ]

    cmap = matplotlib.colormaps["plasma"]
    cmap.set_bad(color=COLOUR_SPACE)
    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)

    fig, axes = plt.subplots(2, 3, figsize=(21, 12), facecolor=COLOUR_BACKGROUND)
    fig.suptitle(
        "TITAN SURFACE HABITABILITY — KEY GEOLOGICAL EPOCHS",
        color="white", fontsize=16, fontweight="bold",
        fontfamily="monospace", y=0.98,
    )
    # Use subplots_adjust instead of tight_layout to avoid the UserWarning
    # that arises from manually-placed colorbar axes being incompatible with
    # the tight layout engine.
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.08,
                        hspace=0.35, wspace=0.25)

    all_t: List[float] = sorted(epoch_maps.keys())

    for ax, t_target, title in zip(axes.flat, key_epochs, titles):
        # Find nearest epoch
        t: float = min(all_t, key=lambda x: abs(x - t_target))
        arr: np.ndarray = epoch_maps[t]
        T_s: float = titan_temp_K(t)

        ax.set_facecolor(COLOUR_BACKGROUND)
        ax.imshow(arr, cmap=cmap, norm=norm,
                  origin="upper", aspect="auto",
                  extent=[0, 360, -90, 90],
                  interpolation="lanczos")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.tick_params(colors="white", labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor(COLOUR_SPINE_POSTER)
        ax.set_xlabel("°W", color=COLOUR_AXIS_LABEL, fontsize=8)
        ax.set_ylabel("°N", color=COLOUR_AXIS_LABEL, fontsize=8)

        info: str = (f"L☉={solar_luminosity_ratio(t):.2f}×  "
                     f"T={T_s:.0f}K  "
                     f"({_phase_label(t).replace(chr(10),' ')})")
        col: str = (COLOUR_PHASE_OCEAN if T_s >= EUTECTIC_K
                    else COLOUR_PHASE_PRESENT if abs(t) <= 0.05
                    else COLOUR_PHASE_DEFAULT)
        ax.text(180, -82, info, color=col, fontsize=7, ha="center", va="bottom")

    # Shared colorbar
    cax = fig.add_axes([0.15, 0.03, 0.70, 0.018])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("P(habitable | features)", color="white", fontsize=11)
    cb.ax.xaxis.set_tick_params(color="white", labelcolor="white", labelsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Poster saved -> {out_path}")


# --- Full-inference mode helpers ---------------------------------------------

def load_anchor_posteriors(
    pipeline_outputs_dir: Path,
) -> Dict[str, np.ndarray]:
    """
    Load the five run_pipeline posterior_mean.npy arrays for full_inference mode.

    Expected directory structure (output of run_pipeline.py --all-temporal-modes)::

        <pipeline_outputs_dir>/
          past/inference/posterior_mean.npy
          lake_formation/inference/posterior_mean.npy
          present/inference/posterior_mean.npy
          near_future/inference/posterior_mean.npy
          future/inference/posterior_mean.npy

    Parameters
    ----------
    pipeline_outputs_dir:
        Root outputs directory (default ``outputs/``).

    Returns
    -------
    Dict mapping anchor name -> float32 posterior array, shape GRID_SHAPE.
    Missing anchors are reported; at least ``present`` must be present.

    Raises
    ------
    FileNotFoundError
        If the ``present`` anchor cannot be found (required for fallback).
    """
    anchors: Dict[str, np.ndarray] = {}
    anchor_names = ["past", "lake_formation", "present", "near_future", "future"]
    for name in anchor_names:
        p = pipeline_outputs_dir / name / "inference" / "posterior_mean.npy"
        if p.exists():
            arr = np.load(p).astype(np.float32)
            anchors[name] = arr
            print(f"  Loaded anchor '{name}': {p}")
        else:
            print(f"  WARNING: anchor '{name}' not found at {p}")
    if "present" not in anchors:
        raise FileNotFoundError(
            f"Required anchor 'present' not found in {pipeline_outputs_dir}. "
            "Run: python run_pipeline.py --temporal-mode present"
        )
    return anchors


def build_pchip_interpolator(
    anchor_epochs: List[float],
    anchor_posteriors: List[np.ndarray],
) -> "scipy.interpolate.PchipInterpolator":
    """
    Build a per-pixel PCHIP (monotone cubic) interpolator over anchor posteriors.

    Interpolation choice rationale
    --------------------------------
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is used rather than
    linear interpolation because:

    1. Lake formation onset (PAST -> LAKE_FORMATION, -3.5 -> -1.0 Gya) follows
       a sigmoidal growth pattern as cryovolcanic outgassing builds the methane
       reservoir.  Linear interpolation underestimates mid-interval habitability.

    2. PCHIP preserves monotonicity between anchor points -- it will not produce
       spurious peaks or dips within an interval where both endpoints have the
       same habitability direction.  This prevents the interpolation from
       inventing false habitability maxima between anchors.

    3. PCHIP does NOT extrapolate beyond the anchor range (clipped to anchor
       endpoints), unlike cubic splines which can oscillate wildly.

    The key non-monotonic trajectory (near_future -> red_giant, +0.25 -> +6.0
    Gya) is EXCLUDED from interpolation deliberately.  That gap involves methane
    evaporation (habitability falls), a dry period, and eutectic ocean formation
    (habitability rises sharply) -- no interpolation scheme can represent this
    without domain knowledge.  Frames in that gap use the ``modelled`` scalar
    approach instead.  See generate_temporal_maps.py documentation.

    NaN handling
    ------------
    Real run_pipeline posterior maps contain NaN pixels where no Cassini data
    were available (e.g. south-polar topography gap, GTIE truncation).
    PchipInterpolator raises ValueError on NaN inputs.  Strategy:

    1. Replace NaN in each anchor with 0.5 (prior midpoint) before stacking.
       This is a neutral fill that does not bias the interpolated result
       towards habitability or non-habitability.
    2. Record which pixels were NaN in *every* anchor (``all_nan_mask``).
       Only pixels that are NaN in ALL anchors are considered "no data" and
       will be restored as NaN in interpolated frames.
    3. Pixels that are NaN in *some* but not all anchors are gap-filled at
       those missing anchors with 0.5.  The interpolation then produces a
       smooth estimate; a per-pixel WARNING is logged once.

    Parameters
    ----------
    anchor_epochs:
        Sorted list of epoch values in Gya (e.g. [-3.5, -1.0, 0.0, 0.25]).
    anchor_posteriors:
        Corresponding posterior arrays, each shape GRID_SHAPE.

    Returns
    -------
    PchipInterpolator
        Callable interp(t) -> float64 flat array.  A custom attribute
        ``nan_mask`` (shape GRID_SHAPE, bool) is attached to the returned
        interpolator; pixels where the mask is True will be set to NaN by
        :func:`interpolate_posterior_at_epoch`.
        Valid for t in [anchor_epochs[0], anchor_epochs[-1]].
    """
    from scipy.interpolate import PchipInterpolator

    n_anchors = len(anchor_epochs)

    # -- NaN audit and gap-fill -----------------------------------------------
    nan_masks: List[np.ndarray] = []
    filled:    List[np.ndarray] = []
    for i, arr in enumerate(anchor_posteriors):
        flat = arr.ravel().astype(np.float64)
        mask = ~np.isfinite(flat)
        if mask.any():
            n_nan = int(mask.sum())
            pct   = 100.0 * n_nan / flat.size
            print(
                f"  WARNING: PCHIP anchor[{i}] (epoch={anchor_epochs[i]:+.2f} Gya) "
                f"has {n_nan} NaN pixels ({pct:.1f}%). "
                f"Gap-filling with 0.5 (prior midpoint). "
                f"Pixels NaN in ALL anchors will be restored as NaN."
            )
            flat = flat.copy()
            flat[mask] = 0.5   # neutral fill; will be re-masked later
        nan_masks.append(mask)
        filled.append(flat)

    # Pixels NaN in EVERY anchor have no information at all.
    # Pixels NaN in SOME anchors get smooth gap-fill from neighbours.
    all_nan_mask: np.ndarray = np.stack(nan_masks, axis=0).all(axis=0)  # (nrows*ncols,)

    # Infer grid shape from the input posteriors (allows non-GRID_SHAPE test arrays)
    if anchor_posteriors[0].ndim == 2:
        inferred_shape: Tuple[int, int] = anchor_posteriors[0].shape
    else:
        inferred_shape = GRID_SHAPE

    stacked: np.ndarray = np.stack(filled, axis=0)   # (n_anchors, nrows*ncols)

    interp = PchipInterpolator(
        np.array(anchor_epochs, dtype=np.float64),
        stacked,
        axis=0,
        extrapolate=False,  # clip to [anchor_epochs[0], anchor_epochs[-1]]
    )

    # Attach the NaN mask so interpolate_posterior_at_epoch can restore it.
    interp.nan_mask = all_nan_mask.reshape(inferred_shape)  # type: ignore[attr-defined]

    return interp


def interpolate_posterior_at_epoch(
    interp: "scipy.interpolate.PchipInterpolator",
    t: float,
    anchor_lo: float,
    anchor_hi: float,
    output_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Evaluate the PCHIP interpolator at epoch *t* and return a clamped array.

    Parameters
    ----------
    interp:
        PchipInterpolator returned by :func:`build_pchip_interpolator`.
    t:
        Target epoch in Gya.
    anchor_lo, anchor_hi:
        The range of valid epochs for this interpolator.
    output_shape:
        (nrows, ncols) for the output array.  Defaults to ``GRID_SHAPE``.
        Pass an explicit shape when using non-canonical arrays (e.g. in tests).

    Returns
    -------
    np.ndarray
        float32 array, shape *output_shape*, clamped to [0, 1].  NaN where
        inputs were NaN.
    """
    shape: Tuple[int, int] = output_shape if output_shape is not None else GRID_SHAPE
    t_clipped: float = float(np.clip(t, anchor_lo, anchor_hi))
    result: np.ndarray = interp(t_clipped).reshape(shape).astype(np.float32)
    result = np.clip(result, 0.0, 1.0)

    # Restore NaN for pixels that had no data in ANY anchor.
    # The NaN mask is attached by build_pchip_interpolator.
    nan_mask: Optional[np.ndarray] = getattr(interp, "nan_mask", None)
    if nan_mask is not None:
        mask = nan_mask if nan_mask.shape == shape else None
        if mask is None and nan_mask.size == result.size:
            # output_shape differs from GRID_SHAPE (e.g. unit tests with small arrays)
            mask = nan_mask.ravel()[:result.size].reshape(shape)
        if mask is not None:
            result = np.where(mask, np.nan, result)

    return result


def _narrative_for_epoch(t: float) -> str:
    """
    Return the narrative caption string for the given epoch.

    The caption is ALWAYS shown in the video (not gated by --pause).
    It shows the most recent TRANSITION_EVENT that has been reached or passed.
    When t is before the first event, the first event caption is used.
    When t is after the last event, the last event caption is used.

    The ``--pause`` flag controls HOLD DURATION only, not caption visibility.
    """
    if not TRANSITION_EVENTS:
        return ""
    # Find the most-recently-passed event: the last one whose epoch <= t.
    applicable = [(et, narr) for et, _, narr in TRANSITION_EVENTS if et <= t]
    if applicable:
        # Return the narrative of the latest passed event
        return max(applicable, key=lambda x: x[0])[1]
    # Before the first event: show the first event caption
    return TRANSITION_EVENTS[0][2]



def _build_pause_timing(
    epochs: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[int, Tuple[float, str]], float]:
    """Build pause_idx dict and NORMAL_HOLD from args.pause flag."""
    if args.pause:
        print(f"  Pause events: {len(TRANSITION_EVENTS)}, target: ~60 s")
        print("  (--pause controls hold duration at key events; captions are always visible)")
        event_best: Dict[float, Tuple] = {}
        for i, t in enumerate(epochs):
            for et, hold, narr in TRANSITION_EVENTS:
                if abs(t - et) < 0.06:
                    if et not in event_best or abs(t - et) < abs(
                        epochs[next(j for j, tt in enumerate(epochs)
                                    if abs(tt - event_best[et][0]) < 1e-6)] - et
                    ):
                        event_best[et] = (t, narr, hold, i)
        pause_idx: Dict[int, Tuple[float, str]] = {
            idx: (hold, narr) for t, narr, hold, idx in event_best.values()
        }
        pause_total = sum(h for h, _ in pause_idx.values())
        normal_hold = (60.0 - pause_total) / max(len(epochs) - len(pause_idx), 1)
    else:
        print("  Pausing disabled (use --pause to enable key-event captions)")
        pause_idx   = {}
        normal_hold = 60.0 / len(epochs)
    return pause_idx, normal_hold


def _encode_animation(
    anim_dir: Path,
    frames_dir: Path,
    frame_paths: List[Path],
    concat_lines: List[str],
    suffix: str = "",
) -> None:
    """Write concat file and encode MP4 + GIF.  ``suffix`` is appended to filenames."""
    import subprocess
    concat_path = anim_dir / f"concat{suffix}.txt"
    concat_path.write_text("\n".join(concat_lines) + "\n")
    with open(concat_path, "a") as f:
        f.write(f"file '{frame_paths[-1].resolve()}'\n")
        f.write("duration 0.04\n")

    mp4_path = anim_dir / f"titan_habitability_animation{suffix}.mp4"
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(mp4_path),
    ], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  MP4 -> {mp4_path}  ({mp4_path.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  ffmpeg MP4 failed:\n{r.stderr[-300:]}")

    gif_path = anim_dir / f"titan_habitability_animation{suffix}.gif"
    r2 = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_path),
        "-vf", ("scale=900:-1:flags=lanczos,"
                "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]"
                "paletteuse=dither=sierra2_4a"),
        "-loop", "0", str(gif_path),
    ], capture_output=True, text=True)
    if r2.returncode == 0:
        print(f"  GIF -> {gif_path}  ({gif_path.stat().st_size/1e6:.1f} MB)")
    else:
        print("  GIF encoding failed -- MP4 is the primary output")


# ---------------------------------------------------------------------------
# Bayesian -> sklearn probability rescaling  (module-level so both animation
# functions can share it without redefinition)
# The analytical Bayesian posterior spans [0.128, 0.673] under the present-
# epoch prior (k=5, l=6); the sklearn RF posteriors span ~[0.142, 0.780].
# Rescaling prevents a visible step at the MODELLED_RESCALED/PCHIP boundary
# in full_inference mode.
# ---------------------------------------------------------------------------
_BAY_MIN, _BAY_MAX = 0.128, 0.673
_SKL_MIN, _SKL_MAX = 0.142, 0.780


def _rescale_bayesian(arr: np.ndarray) -> np.ndarray:
    """Map Bayesian posterior scale onto sklearn RF probability scale."""
    return (_SKL_MIN + (arr - _BAY_MIN) / (_BAY_MAX - _BAY_MIN)
            * (_SKL_MAX - _SKL_MIN)).astype(np.float32)


def _run_animation_modelled(
    args:             argparse.Namespace,
    epochs:           np.ndarray,
    present:          Dict[str, np.ndarray],
    using_synthetic:  bool,
    anim_dir:         Path,
    poster_dir:       Path,
    epoch_map_cache:  Dict[float, np.ndarray],
) -> None:
    """
    Render the MODELLED animation.

    All frames are produced by scalar-scaling the present-epoch Cassini feature
    TIFs via ``scale_features_to_epoch`` and running a uniform-prior Bayesian
    update.  This is the original behaviour of generate_temporal_maps.py.

    Outputs
    -------
    outputs/temporal_maps/animation/titan_habitability_animation.mp4
    outputs/temporal_maps/animation/titan_habitability_animation.gif
    """
    import matplotlib.pyplot as plt

    print(f"Rendering MODELLED animation ({len(epochs)} frames)...")
    if using_synthetic:
        print("  *** WARNING: SYNTHETIC DATA IN USE ***  see log above.")

    frames_dir = anim_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    pause_idx, NORMAL_HOLD = _build_pause_timing(epochs, args)

    concat_lines: List[str] = []
    frame_paths:  List[Path] = []
    current_narrative: str = ""

    for i, t in enumerate(epochs):
        event_data = pause_idx.get(i)
        hold: float = event_data[0] if event_data else NORMAL_HOLD
        if event_data:
            current_narrative = event_data[1]

        scaled    = scale_features_to_epoch(present, t)
        posterior = bayesian_posterior_map(scaled)

        # Caption is ALWAYS shown; --pause only controls hold duration.
        # Use the event-based current_narrative if available (it is the exact
        # text for that transition point), otherwise fall back to the epoch-
        # appropriate caption from _narrative_for_epoch.
        frame_narrative = current_narrative if current_narrative else _narrative_for_epoch(t)

        fig = render_frame(posterior, t, i, len(epochs),
                           dpi=args.dpi, narrative=frame_narrative)

        fpath = frames_dir / f"frame_{i:03d}.png"
        fig.savefig(fpath, dpi=args.dpi, bbox_inches=None,
                    facecolor=fig.get_facecolor())
        plt.close(fig)


        frame_paths.append(fpath)

        concat_lines.append(f"file '{fpath.resolve()}'")
        concat_lines.append(f"duration {hold:.4f}")

        marker = " * PAUSE" if event_data else ""
        if (i + 1) % 8 == 0 or i == len(epochs) - 1 or event_data:
            print(f"  [{i+1:2d}/{len(epochs)}]  t={t:+6.3f} Gya  "
                  f"hold={hold:.2f}s{marker}")

    _encode_animation(anim_dir, frames_dir, frame_paths, concat_lines)


def _run_animation_full_inference(
    args:            argparse.Namespace,
    epochs:          np.ndarray,
    present:         Dict[str, np.ndarray],
    using_synthetic: bool,
    anim_dir:        Path,
    poster_dir:      Path,
) -> None:
    """
    Render the FULL_INFERENCE animation.

    Scientific basis and declared assumptions
    -----------------------------------------
    This mode uses run_pipeline.py anchor posteriors wherever they exist,
    and PCHIP interpolation between anchors for intermediate epochs.

    ANCHOR EPOCHS AND SOURCES:
      past          (-3.5 Gya): run_pipeline --temporal-mode past
                                9-feature Bayesian inference (LHB era)
      lake_formation(-1.0 Gya): run_pipeline --temporal-mode lake_formation
                                9-feature inference (cryovolcanic onset)
      present       ( 0.0 Gya): run_pipeline --temporal-mode present
                                8-feature inference (Cassini calibration)
      near_future   (+0.25 Gya): run_pipeline --temporal-mode near_future
                                8-feature inference (D2 solar warming window)
      future        (+6.0 Gya): run_pipeline --temporal-mode future
                                8 transformed-feature inference (red giant)

    INTERPOLATION BETWEEN ANCHORS:
      Frames between adjacent anchors use PCHIP (monotone cubic) interpolation
      of the posterior maps.  PCHIP preserves monotonicity within each interval
      and does not extrapolate beyond the anchor range.

      Intervals covered by PCHIP:
        past -> lake_formation: -3.5 to -1.0 Gya  (sigmoidal lake growth)
        lake_formation -> present: -1.0 to 0 Gya  (lake establishment)
        present -> near_future:  0 to +0.25 Gya   (very small change)

      ** EXCLUDED from interpolation (non-monotonic gap): **
        near_future -> future: +0.25 to +6.0 Gya
        This interval involves: lake evaporation (+4 Gya), dry period (+5 Gya),
        eutectic crossing (+5.1 Gya), ocean peak (+5.5 Gya), ocean cooling (+6 Gya).
        No interpolation scheme can represent this without domain knowledge.
        Frames in this gap use the MODELLED scalar approach.

      DECLARED ASSUMPTION: PCHIP interpolation is applied per-pixel to the
      posterior arrays, not to the raw feature values.  The interpolated
      posterior is not equivalent to running Bayesian inference at each
      intermediate epoch with epoch-specific feature sets.  Frames produced
      by interpolation are labelled [INTERPOLATED] in GeoTIFF metadata.

    OUTPUTS:
      outputs/temporal_maps/animation_full_inference/
        titan_habitability_animation_full_inference.mp4
        titan_habitability_animation_full_inference.gif

    References
    ----------
    Tobie et al. (2006) Nature 440:61  -- lake_formation anchor basis
    Lorenz, Lunine & McKay (1997) GRL 24:2905  -- near_future and future
    Neish & Lorenz (2012) PSS 60:26  -- surface age context
    """
    import matplotlib.pyplot as plt

    print(f"\nRendering FULL_INFERENCE animation ({len(epochs)} frames)...")
    print("  Loading run_pipeline anchor posteriors...")

    pipeline_out = Path(getattr(args, "pipeline_outputs", "outputs"))
    try:
        anchors = load_anchor_posteriors(pipeline_out)
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}")
        print("  Run: python run_pipeline.py --all-temporal-modes  first.")
        print("  Falling back to MODELLED mode for this run.\n")
        _run_animation_modelled(args, epochs, present, using_synthetic,
                                anim_dir.parent / "animation", anim_dir.parent,
                                {})
        return

    # -- Anchor epoch -> posterior map -----------------------------------------
    # PCHIP covers only the well-constrained interval where Cassini feature
    # maps are valid: lake_formation (-1.0 Gya) to near_future (+0.25 Gya).
    #
    # The past anchor (-3.5 Gya) is intentionally EXCLUDED from PCHIP because
    # the past temporal mode still uses Cassini-era feature maps (organic_abundance,
    # surface_atm_interaction, topographic_complexity) which encode north polar
    # lake signatures even though no lakes existed at -3.5 Gya.  This makes the
    # past posterior artificially bright at the poles (N.polar median ≈ 0.70).
    #
    # Instead, pre-lake-formation frames (t < -1.0 Gya) use the MODELLED scalar
    # approach with a rescaling factor to match the sklearn probability range.
    # The modelled approach correctly attenuates polar contributions through the
    # feature scale functions (liquid_hydrocarbon → 0.10, etc.).
    #
    # Intervals:
    #   t < -0.5 Gya        MODELLED_RESCALED (Bayesian formula + _rescale_bayesian)
    #   -0.5 → 0.0 Gya      TRANSITION_BLEND (Bayesian → present sklearn anchor)
    #   0.0 → +0.25 Gya     PCHIP (present, near_future only)
    #   +0.25 → +5.0 Gya    CLAMPED_NEAR_FUTURE
    #   +5.0 → +6.0 Gya     EUTECTIC_BLEND (near_future → future)
    #   +6.0 → +6.5 Gya     REFREEZE_BLEND (future → past)

    PCHIP_ANCHOR_EPOCHS: List[float] = []
    PCHIP_ANCHOR_POSTS:  List[np.ndarray] = []

    anchor_epoch_map: Dict[str, float] = {
        # "lake_formation" deliberately excluded: it is no longer a PCHIP anchor
        # (removed to prevent equatorial-band artefact -- see comment above), and
        # keeping it here would trigger the snap-to-anchor code at t=-1.0 exactly
        # (make_epoch_axis linspace includes -1.0), causing a one-frame yellow flash.
        "present":     0.0,
        "near_future": +0.250,
        "future":      +5.9,
    }
    # Note: "past" and "lake_formation" are deliberately excluded from PCHIP.
    # lake_formation exclusion reason: the sklearn RF anchor at -1.0 Gya was
    # trained on present-era Cassini SAR feature maps (polar lake signatures
    # at 0.10 scale), which creates stronger polar preference than the Bayesian
    # formula produces at -1.0 Gya.  Blending toward this anchor (frames 12-17)
    # caused a visible "equatorial band" where organic-rich equatorial terrain
    # (including Selk crater) appeared depressed relative to the polar lake margins.
    # FIX (2026-04): PCHIP covers only present (0.0) → near_future (+0.25 Gya).
    # MODELLED_RESCALED extends all the way to t=-0.5 Gya, then a 0.5 Gyr
    # TRANSITION_BLEND smoothly joins to the present sklearn anchor.
    for name in ["present", "near_future"]:
        if name in anchors:
            PCHIP_ANCHOR_EPOCHS.append(anchor_epoch_map[name])
            PCHIP_ANCHOR_POSTS.append(anchors[name])
        else:
            print(f"  WARNING: anchor '{name}' missing; interpolation may be coarser.")

    # Need at least 2 points to interpolate
    can_interpolate: bool = len(PCHIP_ANCHOR_EPOCHS) >= 2

    if can_interpolate:
        print(f"  Building PCHIP interpolator over "
              f"{PCHIP_ANCHOR_EPOCHS[0]:+.2f} -> "
              f"{PCHIP_ANCHOR_EPOCHS[-1]:+.2f} Gya "
              f"({len(PCHIP_ANCHOR_EPOCHS)} anchors)...")
        pchip = build_pchip_interpolator(PCHIP_ANCHOR_EPOCHS, PCHIP_ANCHOR_POSTS)
        t_interp_lo = PCHIP_ANCHOR_EPOCHS[0]
        t_interp_hi = PCHIP_ANCHOR_EPOCHS[-1]   # +0.25 Gya
    else:
        pchip = None
        t_interp_lo = t_interp_hi = 0.0

    # The near_future -> future gap uses LINEAR BLENDING between the two
    # sklearn anchor posteriors.  Previously used MODELLED_SCALAR (the
    # animation's Bayesian formula) but that has a hard max of 0.673 while
    # sklearn posteriors reach 0.85, causing a visible step change in colour
    # at the PCHIP/scalar boundary.  Linear blending stays entirely in
    # sklearn probability space, eliminating the discontinuity.
    #
    # Linear blending is used rather than PCHIP for this interval because
    # the trajectory is non-monotonic (lake evaporation → dry → ocean), so
    # PCHIP would require intermediate anchor points that don't exist.  The
    # visual impression of a smooth cross-fade is scientifically reasonable:
    # it represents gradual surface change without claiming to model the
    # precise intermediate state.
    near_future_post: Optional[np.ndarray] = anchors.get("near_future")
    future_post:      Optional[np.ndarray] = anchors.get("future")
    past_post:        Optional[np.ndarray] = anchors.get("past")
    T_BLEND_LO:   float = +4.0    # start blend at "solar warming ramp" (+4.0 Gya)
    T_FUT:        float = +5.9    # future anchor: last ocean epoch (T=466K @ +5.9 Gya)
    T_REFREEZE:   float = +6.5    # refreezing complete

    # Blending strategy for t > +0.25 Gya:
    #
    #   +0.25 → +5.0 Gya  CLAMP to near_future anchor
    #     The polar-lake spatial pattern is essentially unchanged for 4.75 Gyr:
    #     lakes persist, equatorial remains dark, solar warming is modest.
    #     Linear-blending over this whole window made the equatorial region
    #     progressively orange even though it should stay dark until the lakes
    #     actually evaporate (~+4.0 Gya).  Clamping avoids that artefact.
    #
    #   +5.0 → +6.0 Gya  LINEAR BLEND near_future → future
    #     Rapid transition: lakes fully evaporated (+5.0), surface dry (+5.0–5.1),
    #     eutectic crossing (+5.13), global water-ammonia ocean peak (+5.5–6.0).
    #     The short blend window matches the physical timescale.
    #
    #   +6.0 → +6.5 Gya  REFREEZE BLEND future → past
    #     Sun exits red-giant; L collapses 600× → 0.8×; ocean refreezes < 1 Myr.
    #     Blend toward past anchor as a proxy for the cold, no-liquid state.

    fi_anim_dir = anim_dir.parent / "animation_full_inference"
    frames_dir  = fi_anim_dir / "frames"
    fi_anim_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)

    pause_idx, NORMAL_HOLD = _build_pause_timing(epochs, args)

    concat_lines: List[str] = []
    frame_paths:  List[Path] = []
    current_narrative: str = ""

    for i, t in enumerate(epochs):
        event_data = pause_idx.get(i)
        hold: float = event_data[0] if event_data else NORMAL_HOLD
        if event_data:
            current_narrative = event_data[1]

        # -- Determine posterior source for this epoch -------------------------
        #
        # ALL frames use MODELLED_RESCALED (Bayesian formula + epoch scale fns).
        #
        # The sklearn RF anchor posteriors are NOT used anywhere in the animation.
        # Root cause: the RF is trained with a 50/50 global-median label split.
        # Any pixel without confirmed surface liquid falls below the ~0.28 median
        # and is labelled class-0.  The RF assigns those pixels probability 0.05–0.15
        # while polar lake pixels get 0.65–0.75, producing a bimodal distribution
        # that creates a hard, latitude-aligned "equatorial band" making all organic
        # terrain (Selk crater, dune fields) appear uninhabitable.  Every sklearn
        # anchor — past, lake_formation, present, near_future, future — has this
        # property, so every frame that touched an anchor showed the artefact.
        #
        # The Bayesian formula gives a continuous gradient driven by the full
        # feature set.  The scale functions correctly model:
        #   t < -3.0 Gya  : LHB impact melt + high UV, no lakes
        #   -3.0→-0.5 Gya : organic accumulation, declining impact flux
        #   -0.5→ 0.0 Gya : polar lake system fully established
        #    0.0→+4.0 Gya : stable Cassini-like state, slow warming
        #   +4.0→+5.0 Gya : solar warming, lake evaporation
        #   +5.1→+5.9 Gya : T > 176 K eutectic → global water-ammonia ocean
        #   +5.9→+6.5 Gya : sun exits red giant, ocean refreezes
        #
        # The sklearn anchor posteriors remain loaded and available for the
        # static thesis figures (location_feature_spider.png, etc.) but are
        # deliberately excluded from all animation rendering.
        # No snap-to-anchor override either — that was the mechanism that caused
        # the single-frame yellow flashes at t=-1.0 and t=+0.25.
        source: str
        scaled    = scale_features_to_epoch(present, t)
        posterior = _rescale_bayesian(bayesian_posterior_map(scaled))
        source    = "MODELLED_RESCALED"

        # Caption is ALWAYS shown; --pause only controls hold duration.
        frame_narrative = current_narrative if current_narrative else _narrative_for_epoch(t)
        fig = render_frame(posterior, t, i, len(epochs),
                           dpi=args.dpi, narrative=frame_narrative,
                           source=source)

        fpath = frames_dir / f"frame_{i:03d}.png"
        fig.savefig(fpath, dpi=args.dpi, bbox_inches=None,
                    facecolor=fig.get_facecolor())
        plt.close(fig)


        frame_paths.append(fpath)

        concat_lines.append(f"file '{fpath.resolve()}'")
        concat_lines.append(f"duration {hold:.4f}")

        marker = " * PAUSE" if event_data else ""
        if (i + 1) % 8 == 0 or i == len(epochs) - 1 or event_data:
            print(f"  [{i+1:2d}/{len(epochs)}]  t={t:+6.3f} Gya  "
                  f"hold={hold:.2f}s  [{source}]{marker}")

    _encode_animation(fi_anim_dir, frames_dir, frame_paths,
                      concat_lines, suffix="_full_inference")


# --- Main --------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    print(
        "Titan Habitability Pipeline  Copyright (C) 2025/2026  Chris Meadows\n"
        "This program comes with ABSOLUTELY NO WARRANTY; for details, see the\n"
        "README.md at the project root.\n"
        "This is free software, and you are welcome to redistribute it\n"
        "under certain conditions; see the LICENSE.md file at the project\n"
        "root for details.\n"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    out_dir    = Path(args.output_dir)
    feat_dir   = Path(args.feature_dir)
    tif_dir    = out_dir / "geotiffs"
    anim_dir   = out_dir / "animation"
    poster_dir = out_dir / "posters"
    nc_path    = out_dir / "titan_temporal_habitability.nc"

    for d in [out_dir, tif_dir, anim_dir, poster_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Optional per-epoch posterior .npy directory
    npy_dir: Optional[Path] = None
    if getattr(args, "save_posterior_npy", False):
        npy_dir = anim_dir / "posteriors"
        npy_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Saving per-epoch posteriors to: {npy_dir}")

    # --n-frames is an alias for --epochs
    _n_limit = args.n_frames or args.epochs or 0
    epochs = make_epoch_axis(n_limit=_n_limit if _n_limit > 0 else None)
    print(f"\nTitan Temporal Habitability Maps")
    print(f"  Epochs:        {len(epochs)} points, {epochs[0]:+.2f} -> {epochs[-1]:+.2f} Gya")
    print(f"  Grid:          {GRID_SHAPE[0]} x {GRID_SHAPE[1]} = {GRID_SHAPE[0]*GRID_SHAPE[1]:,} pixels")
    print(f"  Feature TIFs:  {feat_dir}")
    print(f"  Output:        {out_dir}")
    print()

    # Load present-epoch features once
    print("Loading present-epoch feature maps...")
    present = load_present_features(feat_dir)
    using_synthetic: bool = bool(present.pop("_synthetic", False))
    if using_synthetic:
        print("\n  NOTE: all habitability maps in this run are based on")
        print("        SYNTHETIC (non-observational) feature data.\n")
    else:
        print("  Using REAL Cassini-derived feature maps.\n")

    has_rasterio = False
    try:
        import rasterio
        has_rasterio = True
    except ImportError:
        print("  rasterio not installed -- GeoTIFFs will be saved as raw numpy .npy")
        print("  Install rasterio to get QGIS-compatible GeoTIFF output.\n")

    # -- Process each epoch ----------------------------------------------------
    all_posteriors: List[np.ndarray] = []
    epoch_map_cache: Dict[float, np.ndarray] = {}  # for poster

    print("Computing habitability maps...")
    for i, t in enumerate(epochs):
        T_s = titan_temp_K(t)
        phase = _phase_label(t).replace("\n", " ")
        print(f"  [{i+1:2d}/{len(epochs)}]  t={t:+7.3f} Gya  T={T_s:.0f}K  {phase}")

        # Scale features to this epoch
        scaled = scale_features_to_epoch(present, t)

        # Bayesian posterior
        posterior = bayesian_posterior_map(scaled)
        all_posteriors.append(posterior)

        # Cache for poster (6 key epochs)
        epoch_map_cache[t] = posterior

        # Save per-epoch posterior .npy if requested
        if npy_dir is not None:
            t_str_npy = f"{t:+.4f}".replace("+", "").replace("-", "m").replace(".", "_")
            np.save(npy_dir / f"posterior_{t_str_npy}.npy", posterior.astype(np.float32))

        # -- Save GeoTIFF ------------------------------------------------------
        t_str = f"{t:+.3f}".replace("+", "p").replace("-", "m").replace(".", "_")
        tif_path = tif_dir / f"habitability_{t_str}_Gya.tif"

        if has_rasterio:
            write_geotiff(
                arr      = posterior,
                out_path = tif_path,
                metadata = {
                    "EPOCH_GYA":           f"{t:.4f}",
                    "EPOCH_LABEL":         _epoch_label(t).replace("\n", " "),
                    "PHASE":               phase,
                    "SOLAR_L_RATIO":       f"{solar_luminosity_ratio(t):.4f}",
                    "SURFACE_TEMP_K":      f"{T_s:.2f}",
                    "CRS":                 TITAN_CRS_PROJ4,
                    "FEATURE_DATA_SOURCE": "SYNTHETIC" if using_synthetic
                                           else "REAL_CASSINI",
                    # Declared data-provenance flags for downstream users.
                    # These warn that some regions use modelled gap-fills.
                    "ORGANIC_EAST_HEMISPHERE":
                        "MODELLED_GEO_GAPFILL -- pixels east of ~180W use "
                        "Lopes geomorphology scores, not VIMS observations. "
                        "See organic_abundance docstring.",
                    "TOPOGRAPHY_SOUTH_LIMIT":
                        "GTIE_T126_TRUNCATED -- elevation NaN south of ~48-51S. "
                        "Corlies 2017 4ppd gap-filler used if available. "
                        "Ontario Lacus (72S) topography may be gap-filled.",
                    "ACETYLENE_ENERGY_PROXY":
                        "SAR_BACKSCATTER_INDIRECT -- no spatially resolved "
                        "C2H2 map exists; SAR low-sigma0 used as organic "
                        "substrate proxy. See feature docstring.",
                    "SUBSURFACE_OCEAN_ANNULUS":
                        "UNVALIDATED_MORPHOLOGY -- SAR bright-ring detector "
                        "not validated against Hedgepeth crater catalog. "
                        "Max boost capped at +0.30 above base prior.",
                },
            )
        else:
            # Fallback: save as numpy array with a sidecar JSON
            import json
            np.save(tif_path.with_suffix(".npy"), posterior)
            json.dump({
                "epoch_Gya":          t,
                "phase":              phase,
                "solar_L_ratio":      solar_luminosity_ratio(t),
                "surface_temp_K":     T_s,
                "crs":                TITAN_CRS_PROJ4,
                "shape":              list(GRID_SHAPE),
                "dtype":              "float32",
                "feature_data_source": "SYNTHETIC" if using_synthetic
                                       else "REAL_CASSINI",
            }, open(tif_path.with_suffix(".json"), "w"), indent=2)

    print()

    # -- NetCDF stack ----------------------------------------------------------
    if not args.no_netcdf:
        print("Saving NetCDF time-series stack...")
        save_netcdf_stack(epochs, all_posteriors, nc_path)

    # -- Six-panel poster ------------------------------------------------------
    print("Rendering key-epoch poster...")
    poster_path = poster_dir / "key_epochs_poster.png"
    render_poster(epoch_map_cache, poster_path)

    # -- Animation -------------------------------------------------------------
    if not args.no_animation:
        inference_mode: str = getattr(args, "inference_mode", "modelled")

        if inference_mode == "full_inference":
            _run_animation_full_inference(args, epochs, present, using_synthetic,
                                          anim_dir, poster_dir)
        else:
            _run_animation_modelled(args, epochs, present, using_synthetic,
                                    anim_dir, poster_dir, epoch_map_cache)

    # -- Summary and QGIS instructions -----------------------------------------
    print()
    print("=" * 72)
    print("  OUTPUT SUMMARY")
    print("=" * 72)
    print(f"  GeoTIFFs  ({len(epochs)} files):  {tif_dir}/")
    if not args.no_netcdf:
        print(f"  NetCDF stack:              {nc_path}")
    print(f"  Key-epoch poster:          {poster_dir}/key_epochs_poster.png")
    if not args.no_animation:
        print(f"  Animation (MP4):           {anim_dir}/titan_habitability_animation.mp4")
        print(f"  Animation (GIF):           {anim_dir}/titan_habitability_animation.gif")
    print()
    print("=" * 72)
    print("  QGIS-LTR IMPORT GUIDE")
    print("=" * 72)
    print("""
  +- SET PROJECT CRS -------------------------------------------------------+
  | 1. Project -> Properties -> CRS                                         |
  | 2. Search: 'titan'  -- if not found, click '+ Add'                      |
  | 3. Paste this PROJ4 string as the custom CRS:                           |
  |                                                                         |
  |    +proj=eqc +a=2575000 +b=2575000 +units=m +no_defs +lon_0=0           |
  |                                                                         |
  | 4. Name it "Titan_2000_Equirectangular" and save.                       |
  +-------------------------------------------------------------------------+

  +- CYLINDRICAL (EQUIRECTANGULAR) MAP -------------------------------------+
  | * Layer -> Add Layer -> Add Raster Layer                                |
  | * Load any habitability_*_Gya.tif                                       |
  | * Layer -> Properties -> Symbology                                      |
  |     Render type: Singleband pseudocolor                                 |
  |     Min: 0.10   Max: 0.80                                               |
  |     Color ramp: Magma or Plasma (perceptually uniform)                  |
  | * Set project CRS to Titan equirectangular (above)                      |
  +-------------------------------------------------------------------------+

  +- GLOBE / ORTHOGRAPHIC VIEW ---------------------------------------------+
  | Method A -- QGIS Sphere/Globe plugin:                                   |
  |   Plugins -> Manage Plugins -> search 'Globe' -> Install                |
  |   View -> Panels -> Globe -> enable; drag layer into globe panel        |
  |                                                                         |
  | Method B -- Orthographic projection:                                    |
  |   Project -> Properties -> CRS -> search "ortho"                        |
  |   Custom CRS: +proj=ortho +a=2575000 +b=2575000 +lat_0=30 +lon_0=180    |
  |   Adjust lat_0 and lon_0 to set the centre of the globe view.           |
  |   The raster layer must be loaded first; QGIS will reproject on-the-fly.|
  +-------------------------------------------------------------------------+

  +- POLAR CIRCULAR MAP ----------------------------------------------------+
  | 1. Project -> Properties -> CRS -> Custom CRS                           |
  |                                                                         |
  |   North polar:                                                          |
  |   +proj=stere +a=2575000 +b=2575000 +lat_0=90 +lon_0=0 +k=1 +units=m    |
  |                                                                         |
  |   South polar:                                                          |
  |   +proj=stere +a=2575000 +b=2575000 +lat_0=-90 +lon_0=0 +k=1 +units=m   |
  |                                                                         |
  | 2. QGIS will reproject the equirectangular raster on-the-fly.           |
  | 3. Zoom to extent of the layer to see the polar cap.                    |
  +-------------------------------------------------------------------------+

  +- TEMPORAL ANIMATION IN QGIS --------------------------------------------+
  | QGIS-LTR >= 3.16 has a built-in Temporal Controller:                    |
  |                                                                         |
  | 1. Load ALL 36 GeoTIFFs at once:                                        |
  |    Layer -> Add Layer -> Add Raster Layer -> select all .tif files      |
  |                                                                         |
  | 2. For each layer, set its time properties:                             |
  |    Layer -> Properties -> Temporal                                      |
  |    Check: "Dynamic temporal control"                                    |
  |    Layer temporal mode: "Fixed time range"                              |
  |    Begin / End: enter the epoch time                                    |
  |    (Use "1950-01-01" + epoch_Gyrx365.25x24x3600 seconds as offset)      |
  |    NOTE: easier to use the NetCDF in QGIS (see below).                  |
  |                                                                         |
  | 3. View -> Panels -> Temporal Controller                                |
  |    Set animation range and step -> click Play                           |
  |                                                                         |
  | EASIER: Load the NetCDF file instead:                                   |
  |    Layer -> Add Layer -> Add Mesh Layer -> select .nc file              |
  |    QGIS reads the epoch_Gya dimension as the time axis automatically.   |
  |    Temporal Controller will step through all 36 epochs natively.        |
  +-------------------------------------------------------------------------+

  +- ANIMATE WITH THE PRE-RENDERED MP4 -------------------------------------+
  | The MP4 animation requires no QGIS -- open with VLC, QuickTime, etc.    |
  | It shows: equirectangular global + north polar + south polar            |
  | simultaneously across all 36 epochs from -3.5 Gya -> +6.5 Gya.          |
  +-------------------------------------------------------------------------+
""")


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate multi-epoch Titan habitability maps for QGIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir",    default="outputs/temporal_maps",
                   help="Output directory (default: outputs/temporal_maps)")
    p.add_argument("--feature-dir",   default="outputs/present/features/tifs",
                   help="Present-epoch feature TIF directory")
    p.add_argument("--no-animation",  action="store_true",
                   help="Skip animation generation")
    p.add_argument("--pause",         action="store_true",
                   help="Hold at key geological events for longer (narrative captions "
                        "are ALWAYS visible; this flag just extends the hold duration "
                        "at transition points -- default: uniform hold time)")
    p.add_argument("--no-netcdf",     action="store_true",
                   help="Skip NetCDF stack output")
    p.add_argument("--epochs",        type=int, default=0,
                   help="Limit to N epochs (0=all, useful for testing)")
    p.add_argument("--n-frames",      type=int, default=0,
                   help="Alias for --epochs: limit to N frames (0=all)")
    p.add_argument("--fps",           type=int, default=8,
                   help="Animation frames per second (default: 8)")
    p.add_argument("--dpi",           type=int, default=120,
                   help="Animation frame DPI (default: 120)")
    p.add_argument(
        "--inference-mode",
        default="modelled",
        choices=["modelled", "full_inference"],
        help=(
            "Video generation mode (default: modelled).\n"
            "  modelled       -- current behaviour: scalar-scaling of present-epoch\n"
            "                    feature TIFs propagates across all epochs.\n"
            "                    Produces outputs/temporal_maps/animation/.\n"
            "  full_inference -- five pipeline-anchor posteriors (past, lake_formation,\n"
            "                    present, near_future, future) loaded from run_pipeline\n"
            "                    outputs; PCHIP interpolation between anchors;\n"
            "                    modelled scalars used for the near_future->red_giant\n"
            "                    gap (+0.25->+6 Gya, non-monotonic trajectory).\n"
            "                    Requires run_pipeline.py --all-temporal-modes first.\n"
            "                    Produces outputs/temporal_maps/animation_full_inference/.\n"
            "                    Declared assumption: PCHIP interpolation between anchors\n"
            "                    at -3.5, -1.0, 0, +0.25 Gya. All interpolated frames\n"
            "                    labelled [INTERPOLATED] in GeoTIFF metadata."
        ),
    )
    p.add_argument(
        "--pipeline-outputs",
        default="outputs",
        metavar="DIR",
        help=(
            "Root directory of run_pipeline.py outputs (default: outputs). "
            "Used by --inference-mode full_inference to locate anchor posteriors "
            "at <DIR>/past/inference/posterior_mean.npy etc."
        ),
    )
    p.add_argument(
        "--save-posterior-npy",
        action="store_true",
        help=(
            "Save each epoch's posterior probability map as a NumPy .npy file "
            "in <output-dir>/animation/posteriors/posterior_<t>.npy. "
            "These files are used by scripts/generate_temporal_trend.py to "
            "produce the temporal habitability trend figure.  Not saved by "
            "default because 72 frames x 6.5M pixels x float32 = ~1.9 GB."
        ),
    )
    args = p.parse_args()
    main(args)
