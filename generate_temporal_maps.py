"""
generate_temporal_maps.py
==========================
Generate global habitability maps across 36 epochs spanning the Late Heavy
Bombardment (−3.5 Gya) through the present Cassini epoch to the red-giant
solar expansion peak (+6.5 Gya).

Outputs
-------
outputs/temporal_maps/
  geotiffs/                   ← QGIS-ready GeoTIFF per epoch
    habitability_-3.500_Gya.tif
    habitability_-2.379_Gya.tif
    ...
    habitability_+6.500_Gya.tif
  titan_temporal_habitability.nc  ← NetCDF time-series stack (all epochs)
  animation/
    titan_habitability_animation.mp4  ← MP4 flythrough PAST→FUTURE
    titan_habitability_animation.gif  ← Lightweight GIF version
  posters/
    key_epochs_poster.png       ← Six-panel summary figure

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

    F_i(lat, lon, t) = clamp(F_i(lat, lon, 0) × scale_i(t), 0, 1)

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

    α_post = α_0 + λ × Σ_i w_i × F_i
    β_post = β_0 + λ × (1 − Σ_i w_i × F_i)
    P(habitable) = α_post / (α_post + β_post)

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

# ─── Physical constants ───────────────────────────────────────────────────────

TITAN_RADIUS_M:   float = 2_575_000.0
CANONICAL_RES_M:  float = 4_490.0
EUTECTIC_K:       float = 176.0    # water–ammonia eutectic melting point (K)
T_SURFACE_K:      float = 93.65    # present-day Titan surface temperature (K)

#: Canonical grid size (nrows, ncols)
GRID_SHAPE: Tuple[int, int] = (1802, 3603)

# ─── Polar visualisation parameters ─────────────────────────────────────────

#: Southern/northern latitude at which the polar-cap circle boundary is drawn.
#: The stereographic resampling maps r=1 (circle edge) to exactly this latitude,
#: so the full disc is filled with data.  Reduce to show more of the globe;
#: increase to zoom in on the high-latitude lake regions.
POLAR_CAP_EDGE_DEG: float = 50.0

#: Pre-computed scale factor for the polar stereographic projection.
#: Derived from POLAR_CAP_EDGE_DEG via:
#:   lat = 90 - 2·arctan(r·scale)  →  at r=1: lat = POLAR_CAP_EDGE_DEG
#:   ∴  scale = tan((90 - POLAR_CAP_EDGE_DEG) / 2)
#: Reference: Snyder (1987) "Map Projections — A Working Manual", §21.
POLAR_SCALE: float = math.tan(math.radians((90.0 - POLAR_CAP_EDGE_DEG) / 2.0))

# ─── Colour palette ──────────────────────────────────────────────────────────
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
#: Marker colour for the top-10 location dots — chosen to stand out on Plasma.
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
#: Body text inside the narrative box — slightly off-white.
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
#: Fully transparent placeholder — keeps figure layout stable.
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

# ─── Figure geometry ─────────────────────────────────────────────────────────

#: Figure width in inches.
#: Derived so each polar panel width equals the shared row height exactly (wspace=0):
#:   polar_width = (GS_RIGHT−GS_LEFT) × fig_width / 4
#:   panel_height = (GS_TOP−GS_BOTTOM) × fig_height
#:   Setting equal:  fig_width = 4 × (GS_TOP−GS_BOTTOM) / (GS_RIGHT−GS_LEFT) × fig_height
#:   With GS values below and fig_height=8.5:  20.7" → delta < 0.01mm
FIG_WIDTH_IN:  float = 20.7
FIG_HEIGHT_IN: float = 8.5

#: GridSpec margins (figure-fraction, 0=bottom/left, 1=top/right).
#:   GS_LEFT  = 0.08 — space for equirectangular y-axis labels (Latitude °N)
#:   GS_BOTTOM= 0.30 — space for colourbar + x-axis label + narrative text boxes
#:   GS_TOP   = 0.855 — panel titles clear of the progress bar above
GS_LEFT:   float = 0.08
GS_RIGHT:  float = 0.99
GS_TOP:    float = 0.855
GS_BOTTOM: float = 0.300

# ─── Feature weights and priors ──────────────────────────────────────────────

#: Feature weights (must match pipeline run_pipeline.py).
WEIGHTS: Dict[str, float] = {
    "liquid_hydrocarbon":       0.23,
    "organic_abundance":        0.18,
    "acetylene_energy":         0.18,
    "methane_cycle":            0.13,
    "surface_atm_interaction":  0.08,
    "topographic_complexity":   0.05,
    "geomorphologic_diversity": 0.04,
    "subsurface_ocean":         0.02,
    "impact_melt_bonus":        0.09,
}

#: Bayesian inference parameters
KAPPA:   float = 5.0    # prior concentration
LAMBDA:  float = 6.0    # likelihood sharpness

#: Prior means (present-epoch)
PRIOR_MEANS: Dict[str, float] = {
    "liquid_hydrocarbon":       0.020,
    "organic_abundance":        0.600,
    "acetylene_energy":         0.350,
    "methane_cycle":            0.400,
    "surface_atm_interaction":  0.350,
    "topographic_complexity":   0.250,
    "geomorphologic_diversity": 0.300,
    "subsurface_ocean":         0.030,
    "impact_melt_bonus":        0.000,
}

#: Colourmap display range for P(habitable | features).
VMIN, VMAX = 0.10, 0.65

# ─── Epoch axis ───────────────────────────────────────────────────────────────

def make_epoch_axis(n_limit: Optional[int] = None) -> np.ndarray:
    """
    Build the ~71-point epoch axis, denser near key geological transitions.

    Segments are denser near:
      - LHB peak (-3.8 Gya)
      - Lake formation onset (-1.5 to -0.4 Gya)
      - Present epoch (-0.4 to +0.1 Gya)
      - Solar warming + lake evaporation (+3.8 to +5.0 Gya)
      - Water-ammonia eutectic crossing (+5.0 to +5.3 Gya)

    Returns
    -------
    np.ndarray
        Epochs in Gya from present (negative = past, positive = future).
    """
    segs = [
        np.linspace(-3.80, -3.00,  5),   # LHB → early decline
        np.linspace(-3.00, -1.50,  8),   # early Titan
        np.linspace(-1.50, -0.40, 12),   # lake formation — dense
        np.linspace(-0.40,  0.10,  8),   # near-present — very dense
        np.linspace( 0.10,  2.00,  8),   # near future
        np.linspace( 2.00,  3.80,  6),   # mid future
        np.linspace( 3.80,  5.00, 10),   # solar warming — dense
        np.linspace( 5.00,  5.30, 10),   # eutectic transition — very dense
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

    ( 1.50, 2.5,
     "SLOW ACCUMULATION PLATEAU  (+1.5 Gya)  |  Tholins build at 5×10⁻¹⁴ g/cm²/s\n"
     "Solar UV climbing steadily on the main sequence. Polar lakes remain stable.\n"
     "Habitability drifts slightly upward as organic inventory grows."),

    ( 4.00, 3.0,
     "SOLAR WARMING RAMP  (+4.0 Gya)  |  L☉ ≈ 1.3× present\n"
     "Lake surfaces begin evaporating. Methane cycle weakening.\n"
     "liquid_hydrocarbon + methane_cycle together weight 0.36 — both declining."),

    ( 5.00, 3.5,
     "METHANE ATMOSPHERE LOST  (+5.0 Gya)  |  Lakes fully evaporated\n"
     "liquid_hydrocarbon → 0. Organic stockpile at 16 Gyr maximum.\n"
     "Surface is dry and cold. Local minimum before the red-giant transition."),

    ( 5.15, 4.0,
     "EUTECTIC THRESHOLD CROSSED  (+5.1 Gya)  |  T_surface > 176 K\n"
     "Global water-ammonia ocean forms in < 1 Myr. Entire surface becomes liquid.\n"
     "Subsurface ocean merges with surface. Maximum organic concentration in solution."),

    ( 5.50, 3.5,
     "PEAK HABITABILITY  (+5.5 Gya)  |  Global ocean phase\n"
     "All surfaces score ~0.47. 16 Gyr organic stockpile dissolved as bioavailable substrate.\n"
     "Water-ammonia chemistry enables terrestrial-analogue biochemistry."),

    ( 6.50, 3.0,
     "END OF HABITABLE WINDOW  (+6.5 Gya)  |  Sun enters AGB phase\n"
     "Luminosity collapses post-RGB tip. Ocean may refreeze.\n"
     "Total red-giant habitable window: ~400 Myr."),
]


# ─── Solar / temperature models ───────────────────────────────────────────────

def solar_luminosity_ratio(t: float) -> float:
    """L(t) / L_present. Continuous through t=0."""
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


# ─── Time-scaling functions ───────────────────────────────────────────────────

def _scale_liquid_hc(t: float) -> float:
    """Scalar multiplier for liquid_hydrocarbon at epoch t."""
    T = titan_temp_K(t)
    if T >= EUTECTIC_K:
        return 50.0   # global ocean — huge additive override (clamped to 1 after)
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
    """UV-driven C2H2 energy proxy — continuous through t=0."""
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
    """Scale surface–atmosphere interaction via liquid proxy."""
    lhc = min(1.0, _scale_liquid_hc(t))
    slope_frac   = 0.40
    liquid_frac  = 0.60
    return slope_frac + liquid_frac * lhc


def _scale_topo(t: float) -> float:
    """Topographic complexity — changes slowly."""
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
        return 1.0 / 0.03   # ocean IS the surface — override
    if t < -2.0:
        return 2.5
    elif t < -1.0:
        return 1.8
    elif t < -0.5:
        return 1.3
    return 1.0


def _impact_melt_global(t: float) -> float:
    """
    Global impact-melt bonus — spatially uniform additive term.
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


# ─── Feature map loaders ─────────────────────────────────────────────────────

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
      - Polar lakes (>60°N/S) high liquid_HC
      - Equatorial belt high organic_abundance (dunes)
      - Crater regions (near 90°W) higher geomorphologic_diversity
      - Ontario Lacus (179°W, -72°) southern lake
    """
    lat, lon_W = _lat_lon_grids()
    lat_r = np.deg2rad(lat)

    # ── liquid_hydrocarbon ────────────────────────────────────────────────────
    north_lake_zone = np.clip((lat - 60.0) / 20.0, 0.0, 1.0) ** 1.5
    south_lake_zone = np.clip((-lat - 60.0) / 20.0, 0.0, 1.0) ** 1.5
    # Ontario Lacus: 179°W, -72°
    ontario = np.exp(-((lon_W - 179.0) ** 2 + (lat + 72.0) ** 2) / 64.0)
    liquid = np.clip(north_lake_zone * 0.7 + south_lake_zone * 0.15 +
                     ontario * 0.5, 0.0, 1.0).astype(np.float32)

    # ── organic_abundance ─────────────────────────────────────────────────────
    # High in equatorial dune belt (|lat|<30°), lower at poles, very low Xanadu
    dune_belt = np.clip(1.0 - np.abs(lat) / 30.0, 0.0, 1.0) ** 0.5
    # Xanadu (~100°W, -5°): low organic, high water-ice
    xanadu = 1.0 - 0.7 * np.exp(-((lon_W - 100.0) ** 2 + lat ** 2) / 400.0)
    organic = np.clip(0.35 + 0.45 * dune_belt * xanadu, 0.0, 1.0).astype(np.float32)

    # ── acetylene_energy ──────────────────────────────────────────────────────
    # Driven by UV photolysis — mild equatorial enhancement, lake surface ~0
    ace = np.clip(0.28 + 0.15 * (1.0 - np.abs(lat) / 90.0) -
                  0.15 * liquid, 0.0, 1.0).astype(np.float32)

    # ── methane_cycle ─────────────────────────────────────────────────────────
    # Peaks near equator where precipitation is most active
    meth = np.clip(0.30 + 0.15 * np.cos(lat_r) +
                   0.10 * north_lake_zone, 0.0, 1.0).astype(np.float32)

    # ── surface_atm_interaction ───────────────────────────────────────────────
    # Highest at lake margins (~63–67°N) and channels
    lake_margin = np.clip(
        np.exp(-((lat - 64.0) ** 2) / 8.0) * 0.8 +
        np.exp(-((lat + 64.0) ** 2) / 8.0) * 0.25, 0.0, 1.0
    )
    sai = np.clip(0.20 + 0.35 * lake_margin + 0.15 * np.cos(lat_r) ** 2,
                  0.0, 1.0).astype(np.float32)

    # ── topographic_complexity ────────────────────────────────────────────────
    # Higher at poles (labyrinth), crater regions, mountain chains
    np.random.seed(42)
    topo_noise = np.random.uniform(0.0, 0.1, GRID_SHAPE).astype(np.float32)
    selk  = np.exp(-((lon_W - 199.0)**2 + (lat - 7.0)**2) / 100.0) * 0.4
    menrva = np.exp(-((lon_W - 87.3)**2 + (lat - 19.0)**2) / 200.0) * 0.35
    topo = np.clip(0.15 + topo_noise + selk + menrva +
                   0.10 * np.abs(lat) / 90.0, 0.0, 1.0).astype(np.float32)

    # ── geomorphologic_diversity ──────────────────────────────────────────────
    geo = np.clip(0.20 + 0.20 * lake_margin + selk * 0.5 +
                  menrva * 0.4 + 0.10 * np.abs(lat) / 90.0,
                  0.0, 1.0).astype(np.float32)

    # ── subsurface_ocean ──────────────────────────────────────────────────────
    # Uniform global prior — k2=0.589 confirms global subsurface ocean
    sub = np.full(GRID_SHAPE, 0.03, dtype=np.float32)

    # ── impact_melt_bonus ─────────────────────────────────────────────────────
    # Localised around craters — 0 everywhere at present (already past)
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
    Load present-epoch feature GeoTIFFs.

    Falls back to synthetic maps if real TIFs are not available.

    Parameters
    ----------
    feature_dir:
        Directory containing ``<feature_name>.tif`` files.

    Returns
    -------
    Dict[str, np.ndarray]
        Feature name → float32 array of shape GRID_SHAPE.
    """
    feature_names = list(WEIGHTS.keys())
    maps: Dict[str, np.ndarray] = {}
    n_real = 0

    for name in feature_names:
        tif = feature_dir / f"{name}.tif"
        if tif.exists():
            try:
                import rasterio
                with rasterio.open(tif) as src:
                    arr = src.read(1).astype(np.float32)
                    nd = src.nodata
                    if nd is not None:
                        arr[arr == nd] = np.nan
                    arr[arr < 0] = np.nan
                    maps[name] = arr
                    n_real += 1
            except Exception as exc:
                print(f"  WARNING: failed to load {name}.tif: {exc}")

    if n_real >= 6:
        print(f"  Loaded {n_real}/{len(feature_names)} feature TIFs from {feature_dir}")
        # Fill any missing with priors
        for name in feature_names:
            if name not in maps:
                maps[name] = np.full(GRID_SHAPE, PRIOR_MEANS[name], dtype=np.float32)
        return maps
    else:
        if n_real > 0:
            print(f"  Only {n_real} TIFs found — falling back to synthetic maps")
        else:
            print(f"  No feature TIFs found in {feature_dir} — using synthetic maps")
        return _synthetic_features()


# ─── Temporal scaling ─────────────────────────────────────────────────────────

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
        scale_fn = FEATURE_SCALE_FUNCS[name]
        scale    = scale_fn(t)
        if name == "organic_abundance":
            # Scale as a fraction of accumulated stockpile
            # arr encodes relative organic density; scale the absolute level
            scaled = arr * scale
        elif name == "subsurface_ocean" and titan_temp_K(t) >= EUTECTIC_K:
            # Global water ocean — override to 1.0 everywhere
            scaled = np.ones_like(arr)
        elif name == "liquid_hydrocarbon" and titan_temp_K(t) >= EUTECTIC_K:
            # Global water ocean — override to 1.0 everywhere
            scaled = np.ones_like(arr)
        elif name == "impact_melt_bonus":
            # Additive global field — ignore present spatial pattern
            # (there are no present craters still melting; past is modelled globally)
            scaled = np.full_like(arr, min(1.0, _impact_melt_global(t)))
        else:
            scaled = arr * scale

        result[name] = np.clip(scaled, 0.0, 1.0).astype(np.float32)

    return result


# ─── Bayesian inference ───────────────────────────────────────────────────────

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


# ─── GeoTIFF writer ──────────────────────────────────────────────────────────

TITAN_CRS_PROJ4: str = (
    "+proj=eqc +a=2575000 +b=2575000 +units=m +no_defs "
    "+lon_0=0 +lat_ts=0"
)

def canonical_transform() -> "rasterio.transform.Affine":
    """
    Return the rasterio Affine transform for the canonical Titan grid.

    The grid is equirectangular, west-positive, covering 0–360°W × −90–90°N.
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
        predictor = 2,          # float predictor — better compression for rasters
        tiled     = True,
        blockxsize = 512,
        blockysize = 512,
    ) as dst:
        dst.write(arr_out, 1)
        if metadata:
            dst.update_tags(**metadata)


# ─── NetCDF writer ────────────────────────────────────────────────────────────

def save_netcdf_stack(
    epochs:    np.ndarray,
    maps:      List[np.ndarray],
    out_path:  Path,
) -> None:
    """
    Save all epoch maps as a NetCDF4 time-series stack.

    Dimensions: (time, lat, lon)
    Variable:   P_habitable  — posterior mean P(habitable | features)
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
            print("  WARNING: neither netCDF4 nor scipy available — skipping .nc output")
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
        print(f"  NetCDF stack saved → {out_path}")
    except Exception as exc:
        print(f"  WARNING: NetCDF save failed: {exc}")


# ─── Matplotlib renderer ──────────────────────────────────────────────────────

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
    -3.0 ≤ t < -1.0       Early Titan
    -1.0 ≤ t < -0.3       Lake formation
    -0.3 ≤ t < -0.05      Recent past          ← avoids "Near future" for past epochs
    |t| < 0.05            Cassini epoch         ← tight window: ±50 Mya
    0.05 ≤ t < 3.0        Near future
    3.0 ≤ t < 5.0         Pre red-giant
    5.0 ≤ t < EUTECTIC    Red-giant ramp
    t ≥ EUTECTIC          Red-giant water ocean
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
    if t < 5.0:
        return "Pre red-giant"
    return "Red-giant\nramp"


def render_frame(
    posterior:  np.ndarray,
    t:          float,
    epoch_idx:  int,
    n_epochs:   int,
    dpi:        int = 120,
    narrative:  str = "",
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

    # ── Colourmap ────────────────────────────────────────────────────────────
    cmap = matplotlib.colormaps["plasma"]
    cmap.set_bad(color=COLOUR_SPACE)   # matplotlib keyword stays American English
    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)   # matplotlib API

    T_surface = titan_temp_K(t)

    # ── Top-10 locations ─────────────────────────────────────────────────────
    # (lon_W°, lat°, short_label, rank, pole: N/S/blank)
    TOP10: List[Tuple[float, float, str, int, str]] = [
        (310.0,  68.0, "Kraken",     1, "N"),
        ( 78.0,  79.0, "Ligeia",     2, "N"),
        (199.0,   7.0, "Selk",       3, ""),
        ( 87.3,  19.0, "Menrva",     4, ""),
        (155.0,  -5.0, "Shangri-La", 5, ""),
        (250.0,   5.0, "Belet",      6, ""),
        (192.3, -10.6, "Huygens",    7, ""),
        (100.0,  -5.0, "Xanadu",     8, ""),
        (179.0, -72.0, "Ontario",    9, "S"),
        ( 78.0, -20.0, "Hotei",     10, ""),
    ]

    MARKER_SIZE: int  = 6
    TEXT_SIZE:   float = 7.5

    def _label_offset(lon_W: float, lat: float) -> Tuple[float, float]:
        """Return (dx, dy) offset in data units to push label clear of the dot."""
        dx: float = 6.0 if lon_W < 300 else -6.0
        dy: float = 5.0 if lat < 70 else -6.0
        return dx, dy

    # ── Shared figure layout ──────────────────────────────────────────────────
    # The figure is split vertically into three zones:
    #   top strip  (y = GS_TOP  → 1.00) : title, epoch info, progress bar
    #   map panels (y = GS_BOTTOM → GS_TOP) : equirectangular + 2 polar caps
    #   bottom bar (y = 0.00 → GS_BOTTOM)  : colourbar, narrative text boxes
    #
    # Height-matching geometry (wspace=0):
    #   With width_ratios=[2,1,1] the polar panel fraction = (GS_RIGHT−GS_LEFT)/4.
    #   Setting this equal to the panel height fraction (GS_TOP−GS_BOTTOM) gives:
    #     fig_width = 4·(GS_TOP−GS_BOTTOM)/(GS_RIGHT−GS_LEFT)·fig_height
    #   Substituting FIG_HEIGHT_IN=7.5 yields FIG_WIDTH_IN≈20.1 (see constants).
    #   With wspace=0 the polar subplots are allocated exactly FIG_HEIGHT_IN×0.655
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

    # ── Left panel: equirectangular (cylindrical) global map ─────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLOUR_BACKGROUND)
    ax1.imshow(
        posterior, cmap=cmap, norm=norm,
        origin="upper", aspect="auto",
        extent=[0, 360, -90, 90], interpolation="lanczos",
    )

    # Longitude graticule at 30° intervals, latitude at 15° intervals
    ax1.set_xticks(range(0, 361, 30))
    ax1.set_yticks(range(-90, 91, 15))
    ax1.tick_params(colors="white", labelsize=7.5, length=3, width=0.7)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-90, 90)

    # Overlay location markers on equirectangular panel
    for lon_W, lat, label, rank, _pole in TOP10:
        ax1.plot(lon_W, lat, "o", color=COLOUR_MARKER,
                 ms=MARKER_SIZE, mew=1.2, markeredgecolor=COLOUR_MARKER_EDGE,
                 zorder=10)
        dx, dy = _label_offset(lon_W, lat)
        ha: str = "left" if dx > 0 else "right"
        ax1.annotate(
            f"#{rank} {label}",
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
    ax1.set_title("Global map  (equirectangular)", color="white",
                  fontsize=10, pad=5)
    for spine in ax1.spines.values():
        spine.set_edgecolor(COLOUR_SPINE)

    # ── Polar reproject helper ────────────────────────────────────────────────
    def polar_reproject(img: np.ndarray, north: bool, size: int = 500) -> np.ndarray:
        """
        Resample *img* into a circular polar stereographic view.

        The circle boundary (r=1) corresponds exactly to POLAR_CAP_EDGE_DEG (50°),
        so the filled disc occupies the entire circular panel.

        Fix for original bug:
          Original formula: lat = 90 - 2*arctan(r)
            → r=1 maps to lat=0° (equator), so only r<0.364 was visible
            → filled only 13% of the circle area.
          Fixed formula: lat = 90 - 2*arctan(r * POLAR_SCALE)
            → r=1 maps to lat=POLAR_CAP_EDGE_DEG (50°), filling the full circle.
        """
        nrows_i, ncols_i = img.shape
        y0, x0 = np.mgrid[-1:1:size*1j, -1:1:size*1j]
        r = np.sqrt(x0**2 + y0**2)
        outside = r > 1.0

        # Fixed stereographic formula
        lat_s = 90.0 - 2.0 * np.degrees(
            np.arctan(r * POLAR_SCALE)
        )
        if not north:
            lat_s = -lat_s

        # Azimuth: lon_W = atan2(x, -y)
        # This places 0°W at the disc top (y = -1 in display coords)
        # with longitude increasing clockwise — consistent with standard
        # polar maps of both hemispheres.
        lon_s = (np.degrees(np.arctan2(x0, -y0)) + 360.0) % 360.0

        # Convert to pixel coords
        row_s = (90.0 - lat_s) / 180.0 * nrows_i
        col_s = lon_s / 360.0 * ncols_i

        flat_img = img.copy().astype(np.float64)
        flat_img[~np.isfinite(flat_img)] = -1.0
        sampled = map_coordinates(flat_img, [row_s.ravel(), col_s.ravel()],
                                  order=1, mode="wrap", cval=-1.0)
        out = sampled.reshape(size, size).astype(np.float32)
        out[outside] = np.nan
        out[out < 0] = np.nan
        # Pixels outside the unit disc are already NaN (set above).
        return out

    def _loc_to_stereo(lon_W: float, lat: float, north: bool) -> Tuple[Optional[float], Optional[float]]:
        """
        Convert (lon_W°, lat°) to panel coordinates using the fixed
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
        # Stereographic inverse (Snyder 1987, §21):
        #   r = tan((90 - |lat|) / 2) / POLAR_SCALE
        #   x =  r · sin(lon_W)
        #   y = -r · cos(lon_W)
        # The -cos convention matches the azimuth formula atan2(x, -y),
        # which places 0°W at the top of the disc for both hemispheres.
        x_s: float =  r * math.sin(lon_rad)
        y_s: float = -r * math.cos(lon_rad)   # same for north AND south
        return x_s, y_s

    # Longitude tick marks for polar caps (Snyder 1987 §21 inverse).
    # Ticks placed from r=0.93 to r=1.0; labels at r=0.82, 30° increments.
    # Formula: x = r·sin(lon_W),  y = -r·cos(lon_W)  (0°W at top, CW increasing)
    def _draw_polar_graticule(ax: "matplotlib.axes.Axes") -> None:
        """Draw 30° longitude tick marks and labels around the polar disc."""
        for lon_W_tick in range(0, 360, 30):
            lon_rad: float = math.radians(lon_W_tick)
            sin_l: float = math.sin(lon_rad)
            cos_l: float = math.cos(lon_rad)
            # Tick line: inner r=0.93 to outer r=1.0
            ax.plot([0.93 * sin_l, 1.0 * sin_l],
                    [-0.93 * cos_l, -1.0 * cos_l],
                    color=COLOUR_POLAR_RING, lw=1.0, alpha=0.8, zorder=5)
            # Label text at r=0.82
            lbl: str = f"{lon_W_tick}°" if lon_W_tick == 0 else f"{lon_W_tick}°W"
            ax.text(0.82 * sin_l, -0.82 * cos_l, lbl,
                    color=COLOUR_POLAR_RING, fontsize=6.0,
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.05",
                              fc=COLOUR_BACKGROUND, ec="none", alpha=0.7))

    # ── Centre: North polar cap (POLAR_CAP_EDGE_DEG° – 90°N) ─────────────────
    ax2 = fig.add_subplot(gs[0, 1], aspect="equal")
    ax2.set_facecolor(COLOUR_BACKGROUND)
    north_img = polar_reproject(posterior, north=True, size=500)
    ax2.imshow(north_img, cmap=cmap, norm=norm,
               origin="upper", interpolation="lanczos",
               extent=[-1, 1, -1, 1])
    theta: np.ndarray = np.linspace(0, 2 * np.pi, 360)
    ax2.plot(np.cos(theta), np.sin(theta), color=COLOUR_POLAR_RING, lw=0.8, alpha=0.6)
    # Use exact limits so the disc fills the subplot height (matching equirectangular)
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.axis("off")
    ax2.set_title(f"North polar cap  ({POLAR_CAP_EDGE_DEG:.0f}°–90°N)",
                  color="white", fontsize=10, pad=5)
    _draw_polar_graticule(ax2)

    # Overlay north-polar location markers
    for lon_W, lat, label, rank, _ in TOP10:
        xs, ys = _loc_to_stereo(lon_W, lat, north=True)
        if xs is None:
            continue
        ax2.plot(xs, ys, "o", color=COLOUR_MARKER,
                 ms=MARKER_SIZE, mew=1.2, markeredgecolor=COLOUR_MARKER_EDGE, zorder=10)
        mag: float = math.sqrt(xs**2 + ys**2)
        scale_out: float = min(1.15 / max(0.05, mag), 4.0)
        tx: float = max(-0.90, min(0.90, xs * scale_out))
        ty: float = max(-0.90, min(0.90, ys * scale_out))
        ax2.annotate(
            f"#{rank} {label}",
            xy=(xs, ys), xytext=(tx, ty),
            color=COLOUR_TEXT, fontsize=TEXT_SIZE - 0.5,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=COLOUR_LEADER, lw=0.8),
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", fc=COLOUR_ANNOT_BOX_POLAR, ec="none"),
        )

    # ── Right panel: South polar cap (POLAR_CAP_EDGE_DEG° – 90°S) ────────────
    ax3 = fig.add_subplot(gs[0, 2], aspect="equal")
    ax3.set_facecolor(COLOUR_BACKGROUND)
    south_img = polar_reproject(posterior, north=False, size=500)
    ax3.imshow(south_img, cmap=cmap, norm=norm,
               origin="upper", interpolation="lanczos",
               extent=[-1, 1, -1, 1])
    ax3.plot(np.cos(theta), np.sin(theta), color=COLOUR_POLAR_RING, lw=0.8, alpha=0.6)
    ax3.set_xlim(-1.0, 1.0)
    ax3.set_ylim(-1.0, 1.0)
    ax3.axis("off")
    ax3.set_title(f"South polar cap  ({POLAR_CAP_EDGE_DEG:.0f}°–90°S)",
                  color="white", fontsize=10, pad=5)
    _draw_polar_graticule(ax3)

    # Overlay south-polar location markers
    for lon_W, lat, label, rank, _ in TOP10:
        xs, ys = _loc_to_stereo(lon_W, lat, north=False)
        if xs is None:
            continue
        ax3.plot(xs, ys, "o", color=COLOUR_MARKER,
                 ms=MARKER_SIZE, mew=1.2, markeredgecolor=COLOUR_MARKER_EDGE, zorder=10)
        mag = math.sqrt(xs**2 + ys**2)
        scale_out = min(1.15 / max(0.05, mag), 4.0)
        tx = max(-0.90, min(0.90, xs * scale_out))
        ty = max(-0.90, min(0.90, ys * scale_out))
        ax3.annotate(
            f"#{rank} {label}",
            xy=(xs, ys), xytext=(tx, ty),
            color=COLOUR_TEXT, fontsize=TEXT_SIZE - 0.5,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=COLOUR_LEADER, lw=0.8),
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", fc=COLOUR_ANNOT_BOX_POLAR, ec="none"),
        )

    # ── Colourbar ─────────────────────────────────────────────────────────────
    # Placed well below GS_BOTTOM so the equirectangular x-axis label (at
    # ~y=0.268) and the colourbar top (y=0.170) have ~0.8" of clear space.
    cax = fig.add_axes([0.10, 0.178, 0.80, 0.022])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("P(habitable | features)", color="white", fontsize=11)
    cb.ax.xaxis.set_tick_params(color="white", labelcolor="white", labelsize=9)

    # ── Narrative text box ────────────────────────────────────────────────────
    # Both frames (narrative and blank) render identical sized elements so that
    # bbox_inches=None produces a consistent frame height throughout the video.
    #
    # Vertical layout from bottom of figure (FIG_HEIGHT_IN = 8.5"):
    #   y=0.170  colourbar bar top
    #   y=0.148  colourbar bar bottom
    #   y=0.120  colourbar label centre
    #   ── 0.5" gap ──────────────────────────────────────
    #   y=0.090  title pill centre
    #   ── gap ────────────────────────────────────────────
    #   y=0.038  body box centre
    if narrative:
        lines: list = [ln.strip() for ln in narrative.strip().split("\n") if ln.strip()]
        title_line: str       = lines[0] if lines else ""
        body_lines: list[str] = lines[1:] if len(lines) > 1 else []
        body_text:  str       = "\n".join(body_lines) if body_lines else "─" * 60

        fig.text(0.50, 0.038, body_text,
                 color=COLOUR_NARRATIVE_BODY, fontsize=10.5, fontweight="normal",
                 ha="center", va="center", fontfamily="monospace",
                 linespacing=1.55, zorder=20,
                 bbox=dict(boxstyle="round,pad=0.55",
                           fc=COLOUR_NARRATIVE_FILL, ec=COLOUR_NARRATIVE_BORDER, lw=1.8))
        fig.text(0.50, 0.090, title_line,
                 color=COLOUR_NARRATIVE_TITLE, fontsize=12.0, fontweight="bold",
                 ha="center", va="center", fontfamily="monospace", zorder=21,
                 bbox=dict(boxstyle="round,pad=0.30",
                           fc=COLOUR_TITLE_FILL, ec=COLOUR_TITLE_BORDER, lw=1.5))
    else:
        fig.text(0.50, 0.038, " ", color=COLOUR_TRANSPARENT, fontsize=10.5,
                 fontfamily="monospace", ha="center", va="center", zorder=1)
        fig.text(0.50, 0.090, " ", color=COLOUR_TRANSPARENT, fontsize=12.0,
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

    fig.text(0.5, 0.973, "TITAN SURFACE HABITABILITY", color="white",
             fontsize=15, ha="center", va="bottom", fontweight="bold",
             fontfamily="monospace")
    fig.text(0.5, 0.940, f"Epoch:  {_epoch_label(t).replace(chr(10),' ')}   |   "
             f"Phase:  {_phase_label(t).replace(chr(10),' ')}   |   {solar_str}",
             color=phase_col, fontsize=10, ha="center", va="bottom")

    # ── Progress bar ──────────────────────────────────────────────────────────
    bar_ax = fig.add_axes([0.10, 0.910, 0.80, 0.006])
    bar_ax.set_xlim(0, n_epochs)
    bar_ax.set_ylim(0, 1)
    bar_ax.barh(0.5, epoch_idx + 1, height=1.0, color=COLOUR_PROGRESS_BAR, alpha=0.7)
    bar_ax.axis("off")

    return fig


# ─── Six-panel poster ─────────────────────────────────────────────────────────

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
        "LHB peak\n(−3.5 Gya)",
        "Lake formation\n(−1.0 Gya)",
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
    print(f"  Poster saved → {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Main entry point."""
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

    epochs = make_epoch_axis(n_limit=args.epochs if args.epochs > 0 else None)
    print(f"\nTitan Temporal Habitability Maps")
    print(f"  Epochs:        {len(epochs)} points, {epochs[0]:+.2f} → {epochs[-1]:+.2f} Gya")
    print(f"  Grid:          {GRID_SHAPE[0]} × {GRID_SHAPE[1]} = {GRID_SHAPE[0]*GRID_SHAPE[1]:,} pixels")
    print(f"  Feature TIFs:  {feat_dir}")
    print(f"  Output:        {out_dir}")
    print()

    # Load present-epoch features once
    print("Loading present-epoch feature maps…")
    present = load_present_features(feat_dir)
    print()

    has_rasterio = False
    try:
        import rasterio
        has_rasterio = True
    except ImportError:
        print("  rasterio not installed — GeoTIFFs will be saved as raw numpy .npy")
        print("  Install rasterio to get QGIS-compatible GeoTIFF output.\n")

    # ── Process each epoch ────────────────────────────────────────────────────
    all_posteriors: List[np.ndarray] = []
    epoch_map_cache: Dict[float, np.ndarray] = {}  # for poster

    print("Computing habitability maps…")
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

        # ── Save GeoTIFF ──────────────────────────────────────────────────────
        t_str = f"{t:+.3f}".replace("+", "p").replace("-", "m").replace(".", "_")
        tif_path = tif_dir / f"habitability_{t_str}_Gya.tif"

        if has_rasterio:
            write_geotiff(
                arr      = posterior,
                out_path = tif_path,
                metadata = {
                    "EPOCH_GYA":     f"{t:.4f}",
                    "EPOCH_LABEL":   _epoch_label(t).replace("\n", " "),
                    "PHASE":         phase,
                    "SOLAR_L_RATIO": f"{solar_luminosity_ratio(t):.4f}",
                    "SURFACE_TEMP_K":f"{T_s:.2f}",
                    "CRS":           TITAN_CRS_PROJ4,
                },
            )
        else:
            # Fallback: save as numpy array with a sidecar JSON
            import json
            np.save(tif_path.with_suffix(".npy"), posterior)
            json.dump({
                "epoch_Gya": t, "phase": phase,
                "solar_L_ratio": solar_luminosity_ratio(t),
                "surface_temp_K": T_s,
                "crs": TITAN_CRS_PROJ4,
                "shape": list(GRID_SHAPE),
                "dtype": "float32",
            }, open(tif_path.with_suffix(".json"), "w"), indent=2)

    print()

    # ── NetCDF stack ──────────────────────────────────────────────────────────
    if not args.no_netcdf:
        print("Saving NetCDF time-series stack…")
        save_netcdf_stack(epochs, all_posteriors, nc_path)

    # ── Six-panel poster ──────────────────────────────────────────────────────
    print("Rendering key-epoch poster…")
    render_poster(epoch_map_cache, poster_dir / "key_epochs_poster.png")

    # ── Animation ─────────────────────────────────────────────────────────────
    if not args.no_animation:
        print(f"Rendering animation ({len(epochs)} unique frames)…")
        print(f"  Pause events: {len(TRANSITION_EVENTS)}, target: ~60 s")

        frames_dir = anim_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Build per-epoch narrative and hold-time tables
        NORMAL_HOLD: float = (60.0 - sum(h for _, h, _ in TRANSITION_EVENTS)) / max(
            len(epochs) - len(TRANSITION_EVENTS), 1
        )

        # Map each event to the single closest epoch — avoids pausing multiple
        # adjacent frames when the dense axis has two epochs near the same event.
        event_best: Dict[float, Tuple[float, str]] = {}
        for i, t in enumerate(epochs):
            for et, hold, narr in TRANSITION_EVENTS:
                if abs(t - et) < 0.06:
                    if et not in event_best or abs(t - et) < abs(
                        epochs[next(j for j,tt in enumerate(epochs) if abs(tt - event_best[et][0]) < 1e-6)] - et
                    ):
                        event_best[et] = (t, narr, hold, i)
        pause_idx: Dict[int, Tuple[float, str]] = {
            idx: (hold, narr) for t, narr, hold, idx in event_best.values()
        }
        pause_total_s  = sum(h for h, _ in pause_idx.values())
        n_normal_frames = len(epochs) - len(pause_idx)
        NORMAL_HOLD = (60.0 - pause_total_s) / max(n_normal_frames, 1)

        concat_lines: List[str] = []
        frame_paths: List[Path] = []
        current_narrative: str = ""   # persists across frames until next event

        for i, t in enumerate(epochs):
            event_data = pause_idx.get(i)
            hold  = event_data[0] if event_data else NORMAL_HOLD
            # Update the persistent narrative when a new event fires
            if event_data:
                current_narrative = event_data[1]

            scaled    = scale_features_to_epoch(present, t)
            posterior = bayesian_posterior_map(scaled)
            # Pass current_narrative to every frame so text persists between events
            fig       = render_frame(posterior, t, i, len(epochs),
                                     dpi=args.dpi, narrative=current_narrative)

            fpath = frames_dir / f"frame_{i:03d}.png"
            fig.savefig(fpath, dpi=args.dpi,
                        bbox_inches=None,   # fixed size — NEVER tight-crop frames
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            frame_paths.append(fpath)

            # ffmpeg concat entry
            concat_lines.append(f"file '{fpath.resolve()}'")
            concat_lines.append(f"duration {hold:.4f}")

            marker: str = " ◆ PAUSE" if event_data else ""
            if (i + 1) % 8 == 0 or i == len(epochs) - 1 or event_data:
                print(f"  [{i+1:2d}/{len(epochs)}]  t={t:+6.3f} Gya  "
                      f"hold={hold:.2f}s{marker}")

        # Write concat file
        concat_path = anim_dir / "concat.txt"
        concat_path.write_text("\n".join(concat_lines) + "\n")

        # Repeat last frame entry (ffmpeg concat requires final duration workaround)
        with open(concat_path, "a") as f:
            f.write(f"file '{frame_paths[-1].resolve()}'\n")
            f.write("duration 0.04\n")

        import subprocess

        mp4_path = anim_dir / "titan_habitability_animation.mp4"
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_path),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "slow",
            "-crf", "17", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(mp4_path),
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            sz = mp4_path.stat().st_size / 1e6
            print(f"  MP4 saved → {mp4_path}  ({sz:.1f} MB)")
        else:
            print(f"  ffmpeg MP4 failed:\n{result.stderr[-400:]}")

        # GIF from the same concat (lower resolution)
        gif_path = anim_dir / "titan_habitability_animation.gif"
        ffmpeg_gif = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_path),
            "-vf", (
                "scale=900:-1:flags=lanczos,"
                "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]"
                "paletteuse=dither=sierra2_4a"
            ),
            "-loop", "0",
            str(gif_path),
        ]
        result2 = subprocess.run(ffmpeg_gif, capture_output=True, text=True)
        if result2.returncode == 0:
            sz2 = gif_path.stat().st_size / 1e6
            print(f"  GIF saved → {gif_path}  ({sz2:.1f} MB)")
        else:
            print(f"  GIF encoding failed — MP4 is the primary output")

    # ── Summary and QGIS instructions ─────────────────────────────────────────
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
  ┌─ SET PROJECT CRS ──────────────────────────────────────────────────────┐
  │ 1. Project → Properties → CRS                                          │
  │ 2. Search: 'titan'  — if not found, click '+ Add'                      │
  │ 3. Paste this PROJ4 string as the custom CRS:                          │
  │                                                                         │
  │    +proj=eqc +a=2575000 +b=2575000 +units=m +no_defs +lon_0=0         │
  │                                                                         │
  │ 4. Name it "Titan_2000_Equirectangular" and save.                      │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─ CYLINDRICAL (EQUIRECTANGULAR) MAP ────────────────────────────────────┐
  │ • Layer → Add Layer → Add Raster Layer                                  │
  │ • Load any habitability_*_Gya.tif                                       │
  │ • Layer → Properties → Symbology                                        │
  │     Render type: Singleband pseudocolor                                 │
  │     Min: 0.10   Max: 0.65                                               │
  │     Color ramp: Magma or Plasma (perceptually uniform)                  │
  │ • Set project CRS to Titan equirectangular (above)                      │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─ GLOBE / ORTHOGRAPHIC VIEW ─────────────────────────────────────────────┐
  │ Method A — QGIS Sphere/Globe plugin:                                     │
  │   Plugins → Manage Plugins → search 'Globe' → Install                   │
  │   View → Panels → Globe → enable; drag layer into globe panel            │
  │                                                                          │
  │ Method B — Orthographic projection:                                      │
  │   Project → Properties → CRS → search "ortho"                           │
  │   Custom CRS: +proj=ortho +a=2575000 +b=2575000 +lat_0=30 +lon_0=180   │
  │   Adjust lat_0 and lon_0 to set the centre of the globe view.           │
  │   The raster layer must be loaded first; QGIS will reproject on-the-fly.│
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ POLAR CIRCULAR MAP ────────────────────────────────────────────────────┐
  │ 1. Project → Properties → CRS → Custom CRS                              │
  │                                                                          │
  │   North polar:                                                           │
  │   +proj=stere +a=2575000 +b=2575000 +lat_0=90 +lon_0=0 +k=1 +units=m  │
  │                                                                          │
  │   South polar:                                                           │
  │   +proj=stere +a=2575000 +b=2575000 +lat_0=-90 +lon_0=0 +k=1 +units=m │
  │                                                                          │
  │ 2. QGIS will reproject the equirectangular raster on-the-fly.           │
  │ 3. Zoom to extent of the layer to see the polar cap.                    │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─ TEMPORAL ANIMATION IN QGIS ────────────────────────────────────────────┐
  │ QGIS-LTR ≥ 3.16 has a built-in Temporal Controller:                    │
  │                                                                          │
  │ 1. Load ALL 36 GeoTIFFs at once:                                        │
  │    Layer → Add Layer → Add Raster Layer → select all .tif files         │
  │                                                                          │
  │ 2. For each layer, set its time properties:                              │
  │    Layer → Properties → Temporal                                         │
  │    Check: "Dynamic temporal control"                                     │
  │    Layer temporal mode: "Fixed time range"                               │
  │    Begin / End: enter the epoch time                                     │
  │    (Use "1950-01-01" + epoch_Gyr×365.25×24×3600 seconds as offset)     │
  │    NOTE: easier to use the NetCDF in QGIS (see below).                  │
  │                                                                          │
  │ 3. View → Panels → Temporal Controller                                  │
  │    Set animation range and step → click Play                            │
  │                                                                          │
  │ EASIER: Load the NetCDF file instead:                                    │
  │    Layer → Add Layer → Add Mesh Layer → select .nc file                │
  │    QGIS reads the epoch_Gya dimension as the time axis automatically.   │
  │    Temporal Controller will step through all 36 epochs natively.        │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─ ANIMATE WITH THE PRE-RENDERED MP4 ────────────────────────────────────┐
  │ The MP4 animation requires no QGIS — open with VLC, QuickTime, etc.    │
  │ It shows: equirectangular global + north polar + south polar             │
  │ simultaneously across all 36 epochs from −3.5 Gya → +6.5 Gya.         │
  └────────────────────────────────────────────────────────────────────────┘
""")


# ─── Entry point ─────────────────────────────────────────────────────────────

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
    p.add_argument("--no-netcdf",     action="store_true",
                   help="Skip NetCDF stack output")
    p.add_argument("--epochs",        type=int, default=0,
                   help="Limit to N epochs (0=all, useful for testing)")
    p.add_argument("--fps",           type=int, default=8,
                   help="Animation frames per second (default: 8)")
    p.add_argument("--dpi",           type=int, default=120,
                   help="Animation frame DPI (default: 120)")
    args = p.parse_args()
    main(args)
