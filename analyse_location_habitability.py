"""
analyse_location_habitability.py
=================================
Computes Titan surface habitability across geological time for 10 key
locations, spanning -3.5 Gya (Late Heavy Bombardment) through the present
Cassini epoch to +6.5 Gya (red-giant solar expansion).

Each location's feature values are computed by:
  1. Reading the present-epoch canonical raster at that pixel (from data on disk).
  2. Applying time-scaling functions grounded in published models to each feature.
  3. Running the same Bayesian Beta-update used in the main pipeline.

The result is a 36-epoch × 10-location habitability matrix, saved as:
  outputs/diagnostics/location_habitability_timeseries.csv
  outputs/diagnostics/location_habitability_timeseries.png
  outputs/diagnostics/location_habitability_spider.png

Run from the project root:
    python analyse_location_habitability.py

References for time-scaling models:
  Lorenz et al. (1997) GRL 24, 2905  —  red-giant window
  Schroder & Connon Smith (2008) MNRAS 386, 155  —  solar luminosity evolution
  O'Brien et al. (2005) Icarus 173, 243  —  impact flux decay
  Lavvas et al. (2011) Icarus 215, 732  —  organic deposition rate
  Strobel (2010) Icarus 208, 878  —  H2/C2H2 flux (present)
  Madan et al. (2026) Icarus  —  impact melt amino acid synthesis
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, '.')

# ─── Constants ────────────────────────────────────────────────────────────────

TITAN_RADIUS_M = 2_575_000.0

# Solar luminosity model: L(t) / L_0 where t is Gyr from now.
# Uses Schroder & Connon Smith (2008) main-sequence evolution fit,
# plus a simplified red-giant ramp starting at ~5 Gya from now.
def solar_luminosity_ratio(t_gyr_from_now: float) -> float:
    """
    Approximate solar luminosity relative to present-day L_sun.

    t_gyr_from_now: positive = future, negative = past.

    Main sequence (t < 5 Gya from now):
        L ≈ L_0 * (1 + 0.0092 * (t_now - t_past)) using Ribas (2010) fit
        where t_now = 4.57 Gya is the Sun's current age.

    Red giant ramp (5 < t < 7.6 Gya from now):
        Rapid rise to ~2700 L_sun at peak; habitability window (>2.7 L_sun) is
        approx 5.1 – 5.8 Gya from now (Lorenz et al. 1997).
    """
    age_now_gyr = 4.57    # Sun's current age in Gyr
    age = age_now_gyr + t_gyr_from_now  # Sun's age at this epoch

    if age < 0:
        return 0.5  # pre-solar-system; floor

    if t_gyr_from_now <= 5.0:
        # Main sequence: empirical fit to Baraffe et al. / Ribas (2010)
        # L(t_gya_from_now) ≈ L_0 * exp(0.025 * (t_gyr_from_now))
        # More accurately: L_sun(age) ≈ 1 / (1 - 0.10 * (1 - age/5.0))
        # i.e. 30% fainter at ZAMS (age=0), increasing to L_0 at age=4.57
        zams_faint = 0.72  # L at ZAMS relative to present
        L = zams_faint + (1.0 - zams_faint) * (age / age_now_gyr) ** 0.9
        return max(0.5, L)

    # Red giant transition (simplified piecewise):
    # ~5 Gya: starts subgiant brightening (~1.1 L_sun)
    # ~5.1 Gya: Titan surface T crosses water-ammonia eutectic (2.7 L_sun)
    # ~5.4 Gya: peak RGB tip (1000-2700 L_sun, but models vary greatly)
    # After ~5.8 Gya: helium flash, brief dimming, AGB → planetary nebula
    t_after_ms = t_gyr_from_now - 5.0   # Gyr into red giant phase
    if t_after_ms < 0.1:
        return 1.0 + 17.0 * t_after_ms  # rapid ramp
    elif t_after_ms < 0.5:
        # Peak ~ 2700 L at t_after_ms ~ 0.4
        peak = 2700.0 * math.exp(-0.5 * ((t_after_ms - 0.4) / 0.15) ** 2)
        return max(2.0, peak)
    elif t_after_ms < 1.0:
        return max(1.0, 2700.0 * math.exp(-3.0 * (t_after_ms - 0.4)))
    else:
        return 0.8  # post-RGB/AGB dim phase

# Titan surface temperature as function of solar luminosity
# T_surface ∝ L^0.25 (grey-body approximation, ignoring atmosphere change)
T_SURFACE_PRESENT_K = 93.65

def titan_surface_temp_K(t_gyr_from_now: float) -> float:
    L = solar_luminosity_ratio(t_gyr_from_now)
    return T_SURFACE_PRESENT_K * L ** 0.25

WATER_AMMONIA_EUTECTIC_K = 176.0  # K — melting point of 32% NH3 solution

# ─── Top 10 Locations ─────────────────────────────────────────────────────────

LOCATIONS = [
    # (name, lon_W_deg, lat_deg, short_label)
    ("Kraken Mare shore",    310.0,  68.0,  "Kraken"),
    ("Ligeia Mare shore",     78.0,  79.0,  "Ligeia"),
    ("Selk crater",          199.0,   7.0,  "Selk"),
    ("Menrva crater",         87.3,  19.0,  "Menrva"),
    ("Shangri-La dunes",     155.0,  -5.0,  "Shangri-La"),
    ("Belet dunes",          250.0,   5.0,  "Belet"),
    ("Huygens site",         192.3, -10.6,  "Huygens"),
    ("Xanadu",               100.0,  -5.0,  "Xanadu"),
    ("Ontario Lacus",        179.0, -72.0,  "Ontario"),
    ("Hotei Arcus",           78.0, -20.0,  "Hotei"),
]

# ─── Epoch axis ───────────────────────────────────────────────────────────────

def make_epoch_axis() -> np.ndarray:
    """36-point axis in Gya from now (negative=past, positive=future)."""
    past       = -np.logspace(np.log10(3.5), np.log10(0.05), 12)
    near       =  np.linspace(-0.04, 0.04, 5)
    fut_ramp   =  np.logspace(np.log10(0.05), np.log10(5.0), 12)
    red_giant  =  np.linspace(5.0, 6.5, 8)[1:]
    epochs = np.sort(np.unique(np.round(
        np.concatenate([past, near, fut_ramp, red_giant]), 4
    )))
    return epochs

EPOCHS = make_epoch_axis()

# ─── Feature time-evolution models ────────────────────────────────────────────

class LocationFeatures:
    """
    Holds present-epoch feature values for one location and applies
    time-scaling to compute features at any epoch.
    """

    def __init__(
        self,
        name: str,
        lon_W: float,
        lat: float,
        present_features: dict[str, float],
    ) -> None:
        self.name: str = name
        self.lon_W: float = lon_W
        self.lat: float = lat
        self.f0: dict[str, float] = present_features   # baseline (present epoch)

    # ── Individual feature time models ────────────────────────────────────────

    def _liquid_hydrocarbon(self, t: float) -> float:
        """
        Present: f0 value (0 for equatorial, 0.4–0.7 for lake margins).
        Past: low (lakes are geologically young; polar condensation only
              established over last ~1 Gyr; impact-melt water brief).
        Future: high while T < methane boiling point, then 0 when solar
                evaporation drives off the atmosphere (~+5.4 Gya).
        Red giant: water ocean globally → 1.0.
        """
        T = titan_surface_temp_K(t)

        if T >= WATER_AMMONIA_EUTECTIC_K:
            # Global water-ammonia ocean — all surfaces are 'wet'
            return 1.0

        f_present = self.f0.get('liquid_hydrocarbon', 0.0)

        # Past: polar lakes have been building up over last ~1 Gyr
        if t < -1.0:
            # Before polar lake formation, only impact-melt water
            return min(f_present * 0.1, 0.03)
        elif t < -0.5:
            # Lakes forming; scale linearly with time
            frac = (t + 1.0) / 0.5
            return min(f_present * (0.1 + 0.9 * frac), f_present)
        elif t < 0.0:
            return f_present

        # Near future: present level until solar warming evaporates lakes
        # Lakes evaporate ~gradually as T rises toward methane boiling (90 K)
        # Linear decay from t=+4 to t=+5 Gya
        if t < 4.0:
            return f_present
        elif t < 5.0:
            frac = 1.0 - (t - 4.0)  # declines to 0 by t=+5 Gya
            return f_present * max(0.0, frac)
        else:
            # Between methane evaporation and water-ammonia melting
            return 0.0

    def _organic_abundance(self, t: float) -> float:
        """
        Tholins accumulate roughly linearly at ~5e-14 g/cm²/s (Lavvas 2011).
        At present (4.57 Gya into solar system), the layer is ~1–10 km deep.
        Relative stockpile scales with time since atmosphere-forming epoch.

        In the red-giant future, tholins dissolve in the global ocean,
        making them bioavailable (a habitability plus, handled as organic
        abundance remaining but transitioning to dissolved form).
        """
        T = titan_surface_temp_K(t)
        f_present = self.f0.get('organic_abundance', 0.5)

        # Time since Titan's atmosphere formed (~4.0 Gya ago, set t_atm = -4.0)
        t_atm_gyr = 4.0   # approximate age of stable N2 atmosphere
        t_elapsed = t_atm_gyr + t  # Gyr of photochemical production elapsed

        if t_elapsed < 0:
            return 0.0  # no atmosphere yet

        # Scale linearly relative to present accumulation
        frac = min(t_elapsed / t_atm_gyr, 2.5)  # cap at 2.5× present

        organic = f_present * frac

        # In red-giant water ocean, organics dissolve but remain available
        if T >= WATER_AMMONIA_EUTECTIC_K:
            organic = min(1.0, organic * 1.1)  # slightly enhanced (concentrated)

        return float(np.clip(organic, 0.0, 1.0))

    def _acetylene_energy(self, t: float) -> float:
        """
        C2H2 + H2 flux driven by UV photolysis, scales with solar UV flux.

        Young Sun UV model (Ribas et al. 2010): L_UV ∝ t_age^{-1.0...-0.5}
        depending on waveband. At t=0 (present), uv_factor = 1.0 by definition.
        At t = -3.5 Gya, UV is ~2.4× present (FUV band).

        Formula: uv_factor = (1 + |t|)^0.57
          → continuous through t = 0 (both sides give 1.0 at t = 0)
          → t = -3.5 Gya: 2.36×  (within Ribas 2010 range)
          → t = +5.0 Gya: 2.78×  (UV rises briefly on red-giant ramp,
                           then collapses as star expands and cools)

        Old formula (1 + 0.5)^0.3 = 0.81 at t→0⁻ caused a visible jump
        of ~0.19 at the present epoch. This formula eliminates that.
        """
        f_present = self.f0.get('acetylene_energy', 0.35)

        T = titan_surface_temp_K(t)
        if T >= WATER_AMMONIA_EUTECTIC_K:
            return 0.1  # ocean phase: no surface C2H2

        # Continuous UV scaling factor centred on 1.0 at t = 0
        uv_factor = (1.0 + abs(t)) ** 0.57

        # Red giant ramp: UV collapses as star expands and cools past peak
        if t > 5.0:
            uv_factor *= max(0.0, 1.0 - (t - 5.0) / 1.5)

        return float(np.clip(f_present * uv_factor, 0.0, 1.0))

    def _methane_cycle(self, t: float) -> float:
        """
        The methane cycle requires: (1) surface methane reservoir,
        (2) convective rainfall, (3) liquid at the surface.

        Methane lifetime against photodissociation: ~30 Myr.
        It must be replenished episodically (cryovolcanism).
        We model it as present-level with uncertainty excursions.

        In red giant: methane vaporises and is photodissociated rapidly.
        """
        f_present = self.f0.get('methane_cycle', 0.40)
        T = titan_surface_temp_K(t)

        if T >= WATER_AMMONIA_EUTECTIC_K:
            return 0.3  # water cycle replaces methane cycle

        # Past: methane cycle was episodic; model as lower but nonzero
        if t < -1.0:
            return f_present * 0.6
        elif t < -0.5:
            return f_present * 0.8

        # Present and near future: nominal
        if t < 4.5:
            return f_present

        # Late future: methane boils off, cycle weakens
        return f_present * max(0.0, 1.0 - (t - 4.5) / 0.5)

    def _surface_atm_interaction(self, t: float) -> float:
        """
        Exchange at lake margins and channel networks.
        Scales with liquid availability (liquid_hydrocarbon feature)
        and topographic slope (constant through time).
        """
        f_present = self.f0.get('surface_atm_interaction', 0.35)
        lhc = self._liquid_hydrocarbon(t)
        lhc_present = self.f0.get('liquid_hydrocarbon', 0.1)

        # Scale the liquid-dependent component, keep slope component fixed
        slope_frac = 0.30
        liquid_frac = 0.70
        lhc_ratio = lhc / max(lhc_present, 0.01)
        f = f_present * (slope_frac + liquid_frac * lhc_ratio)
        return float(np.clip(f, 0.0, 1.0))

    def _topographic_complexity(self, t: float) -> float:
        """Topography changes very slowly (crater formation, erosion).
        Roughly constant; slightly higher in past (fresh craters, more erosion)."""
        f_present = self.f0.get('topographic_complexity', 0.25)
        if t < -2.0:
            return min(1.0, f_present * 1.3)   # more fresh craters in LHB
        elif t < -1.0:
            return min(1.0, f_present * 1.15)
        return f_present

    def _geomorphologic_diversity(self, t: float) -> float:
        """Diversity is roughly constant; slightly lower in very early past
        (fewer terrain types established)."""
        f_present = self.f0.get('geomorphologic_diversity', 0.30)
        if t < -3.0:
            return f_present * 0.7
        elif t < -2.0:
            return f_present * 0.85
        return f_present

    def _subsurface_ocean(self, t: float) -> float:
        """
        Subsurface ocean confirmed by k2 and stable through geological time.
        Surface expression (cryovolcanism) higher in past (more radioactive
        heating). In red-giant epoch, ocean merges with surface ocean.
        """
        f_present = self.f0.get('subsurface_ocean', 0.03)
        T = titan_surface_temp_K(t)
        if T >= WATER_AMMONIA_EUTECTIC_K:
            return 1.0  # ocean IS the surface

        # Past: more radiogenic heat → more cryovolcanism → higher surface expression
        if t < -2.0:
            heat_factor = 2.5
        elif t < -1.0:
            heat_factor = 1.8
        elif t < -0.5:
            heat_factor = 1.3
        else:
            heat_factor = 1.0

        return float(np.clip(f_present * heat_factor, 0.0, 1.0))

    def _impact_melt_bonus(self, t: float) -> float:
        """
        Probability of active impact-melt liquid water at epoch t.

        Physics: melt-pond availability peaks at the LHB (~-3.8 Gya) and
        decays on a ~0.8 Gyr timescale as the impact flux subsides and
        existing ponds freeze (O'Brien et al. 2005; ~10^3-10^4 yr lifetime
        per large pond).  The background post-LHB cratering rate gives a
        small continuous tail that reaches ~0.01 at the present.

        This function is continuous everywhere, including through t=0.
        Old implementation had exp((t+3.5)/1.5) which grew toward t=0,
        reaching ~3 (clipped to 1.0), then jumped to 0 for t>=0.

        At t=0 this gives:
          lhb_peak = 0.40 * exp(-0.5 * (3.8/0.5)^2) ≈ 1.2e-22 ≈ 0
          bg       = 0.10 * exp(-3.8/0.8) = 0.10 * 0.0085 ≈ 0.001
          total    ≈ 0.001  (essentially zero, as expected)
        """
        is_crater_site = 'crater' in self.name.lower() or 'menrva' in self.name.lower()

        t_lhb = -3.8    # Gya — LHB peak
        tau   = 0.8     # Gyr — melt availability decay timescale

        # Gaussian LHB peak (width 0.5 Gyr covers -2.8 to -4.8 Gya at 2σ)
        lhb_peak = 0.40 * math.exp(-0.5 * ((t - t_lhb) / 0.5) ** 2)

        # Background: symmetric exponential decay away from LHB on both sides.
        # Gives 0.10 at LHB, ~0.001 at present, ~0 in future.
        bg = 0.10 * math.exp(-abs(t - t_lhb) / tau)

        base = min(1.0, lhb_peak + bg)
        if is_crater_site:
            base = min(1.0, base * 2.0)  # direct melt preserved in terrain

        return float(base)

    # ── Feature vector at epoch t ──────────────────────────────────────────────

    def features_at_epoch(self, t: float) -> dict[str, float]:
        """Return the full feature dict at epoch t (Gya from now)."""
        return {
            'liquid_hydrocarbon':      self._liquid_hydrocarbon(t),
            'organic_abundance':       self._organic_abundance(t),
            'acetylene_energy':        self._acetylene_energy(t),
            'methane_cycle':           self._methane_cycle(t),
            'surface_atm_interaction': self._surface_atm_interaction(t),
            'topographic_complexity':  self._topographic_complexity(t),
            'geomorphologic_diversity':self._geomorphologic_diversity(t),
            'subsurface_ocean':        self._subsurface_ocean(t),
            'impact_melt_bonus':       self._impact_melt_bonus(t),
        }

# ─── Bayesian inference ───────────────────────────────────────────────────────

# Weights (must sum to ~1.0; impact_melt_bonus uses remaining weight)
FEATURE_WEIGHTS = {
    'liquid_hydrocarbon':      0.23,
    'organic_abundance':       0.18,
    'acetylene_energy':        0.18,
    'methane_cycle':           0.13,
    'surface_atm_interaction': 0.08,
    'topographic_complexity':  0.05,
    'geomorphologic_diversity':0.04,
    'subsurface_ocean':        0.02,
    'impact_melt_bonus':       0.09,   # extra past weight
}
assert abs(sum(FEATURE_WEIGHTS.values()) - 1.0) < 0.01

PRIOR_MEANS = {
    'liquid_hydrocarbon':      0.02,
    'organic_abundance':       0.60,
    'acetylene_energy':        0.35,
    'methane_cycle':           0.40,
    'surface_atm_interaction': 0.35,
    'topographic_complexity':  0.25,
    'geomorphologic_diversity':0.30,
    'subsurface_ocean':        0.03,
    'impact_melt_bonus':       0.00,  # prior = 0 (no impact assumed by default)
}

KAPPA = 5.0    # prior concentration
LAMBDA = 6.0   # likelihood sharpness

def bayesian_posterior(features: dict[str, float]) -> tuple[float, float, float]:
    """
    Beta conjugate update.
    Returns (posterior_mean, hdi_low, hdi_high).
    """
    from scipy.stats import beta as beta_dist

    # Prior
    mu0 = sum(PRIOR_MEANS[k] * FEATURE_WEIGHTS[k] for k in FEATURE_WEIGHTS)
    alpha0 = mu0 * KAPPA
    beta0  = (1.0 - mu0) * KAPPA

    # Likelihood update
    weighted_sum = sum(features[k] * FEATURE_WEIGHTS[k] for k in FEATURE_WEIGHTS)
    alpha_post = alpha0 + LAMBDA * weighted_sum
    beta_post  = beta0  + LAMBDA * (1.0 - weighted_sum)

    dist = beta_dist(alpha_post, beta_post)
    mean = dist.mean()
    lo, hi = dist.interval(0.94)
    return float(mean), float(lo), float(hi)

# ─── Load present-epoch features from canonical TIFs ─────────────────────────

def load_present_features(lon_W: float, lat: float) -> dict[str, float]:
    """
    Read the 8 canonical feature TIFs and extract the value at (lon_W, lat).
    Falls back to physically-motivated defaults if TIFs are not available.
    """
    try:
        import rasterio
        _HAS_RASTERIO = True
    except ImportError:
        _HAS_RASTERIO = False

    feature_files = {
        'liquid_hydrocarbon':      'outputs/present/features/tifs/liquid_hydrocarbon.tif',
        'organic_abundance':       'outputs/present/features/tifs/organic_abundance.tif',
        'acetylene_energy':        'outputs/present/features/tifs/acetylene_energy.tif',
        'methane_cycle':           'outputs/present/features/tifs/methane_cycle.tif',
        'surface_atm_interaction': 'outputs/present/features/tifs/surface_atm_interaction.tif',
        'topographic_complexity':  'outputs/present/features/tifs/topographic_complexity.tif',
        'geomorphologic_diversity':'outputs/present/features/tifs/geomorphologic_diversity.tif',
        'subsurface_ocean':        'outputs/present/features/tifs/subsurface_ocean.tif',
    }

    # Physically-motivated defaults (used when TIFs unavailable)
    # These encode known science about each location
    defaults_by_location = {
        # (lon_W, lat): {feature: value}
        # Kraken Mare shore
        (310.0,  68.0): dict(liquid_hydrocarbon=0.65, organic_abundance=0.58, acetylene_energy=0.38,
                              methane_cycle=0.45, surface_atm_interaction=0.60, topographic_complexity=0.35,
                              geomorphologic_diversity=0.55, subsurface_ocean=0.04),
        # Ligeia Mare shore
        (78.0,  79.0):  dict(liquid_hydrocarbon=0.60, organic_abundance=0.50, acetylene_energy=0.32,
                              methane_cycle=0.50, surface_atm_interaction=0.55, topographic_complexity=0.28,
                              geomorphologic_diversity=0.45, subsurface_ocean=0.03),
        # Selk crater
        (199.0,  7.0):  dict(liquid_hydrocarbon=0.04, organic_abundance=0.62, acetylene_energy=0.41,
                              methane_cycle=0.38, surface_atm_interaction=0.51, topographic_complexity=0.48,
                              geomorphologic_diversity=0.72, subsurface_ocean=0.22),
        # Menrva crater
        (87.3,  19.0):  dict(liquid_hydrocarbon=0.03, organic_abundance=0.45, acetylene_energy=0.36,
                              methane_cycle=0.35, surface_atm_interaction=0.40, topographic_complexity=0.55,
                              geomorphologic_diversity=0.68, subsurface_ocean=0.18),
        # Shangri-La
        (155.0, -5.0):  dict(liquid_hydrocarbon=0.02, organic_abundance=0.78, acetylene_energy=0.48,
                              methane_cycle=0.42, surface_atm_interaction=0.32, topographic_complexity=0.18,
                              geomorphologic_diversity=0.28, subsurface_ocean=0.02),
        # Belet
        (250.0,  5.0):  dict(liquid_hydrocarbon=0.02, organic_abundance=0.70, acetylene_energy=0.44,
                              methane_cycle=0.40, surface_atm_interaction=0.30, topographic_complexity=0.15,
                              geomorphologic_diversity=0.25, subsurface_ocean=0.02),
        # Huygens
        (192.3,-10.6):  dict(liquid_hydrocarbon=0.02, organic_abundance=0.54, acetylene_energy=0.38,
                              methane_cycle=0.36, surface_atm_interaction=0.28, topographic_complexity=0.22,
                              geomorphologic_diversity=0.30, subsurface_ocean=0.02),
        # Xanadu
        (100.0, -5.0):  dict(liquid_hydrocarbon=0.01, organic_abundance=0.08, acetylene_energy=0.22,
                              methane_cycle=0.33, surface_atm_interaction=0.25, topographic_complexity=0.40,
                              geomorphologic_diversity=0.42, subsurface_ocean=0.03),
        # Ontario Lacus
        (179.0,-72.0):  dict(liquid_hydrocarbon=0.35, organic_abundance=0.42, acetylene_energy=0.28,
                              methane_cycle=0.30, surface_atm_interaction=0.40, topographic_complexity=0.25,
                              geomorphologic_diversity=0.38, subsurface_ocean=0.03),
        # Hotei Arcus
        (78.0, -20.0):  dict(liquid_hydrocarbon=0.03, organic_abundance=0.55, acetylene_energy=0.35,
                              methane_cycle=0.38, surface_atm_interaction=0.35, topographic_complexity=0.42,
                              geomorphologic_diversity=0.50, subsurface_ocean=0.15),
    }

    # Try to load from TIFs first
    features = {}
    n_loaded = 0
    for feat, tif_path in feature_files.items():
        p = Path(tif_path)
        if p.exists() and _HAS_RASTERIO:
            try:
                with rasterio.open(p) as src:
                    # Convert lon_W, lat to pixel coordinates
                    nrows, ncols = src.height, src.width
                    # Canonical grid: lon_W = col * (360/ncols), lat = 90 - row * (180/nrows)
                    col = int(round(lon_W / 360.0 * ncols)) % ncols
                    row = int(round((90.0 - lat) / 180.0 * nrows))
                    row = max(0, min(nrows - 1, row))
                    # Sample 5×5 neighbourhood to reduce pixel noise
                    r0, r1 = max(0, row-2), min(nrows, row+3)
                    c0, c1 = max(0, col-2), min(ncols, col+3)
                    arr = src.read(1)[r0:r1, c0:c1].astype(np.float32)
                    nodata = src.nodata
                    if nodata is not None:
                        arr[arr == nodata] = np.nan
                    arr[arr < -100] = np.nan
                    val = float(np.nanmean(arr))
                    if np.isfinite(val):
                        features[feat] = val
                        n_loaded += 1
            except Exception:
                pass

    if n_loaded >= 6:
        print(f"  Loaded {n_loaded}/8 features from TIFs for ({lon_W}°W, {lat}°)")
        # Fill any missing
        key = (lon_W, lat)
        if key in defaults_by_location:
            for k, v in defaults_by_location[key].items():
                if k not in features:
                    features[k] = v
        return features

    # Fall back to defaults
    key = (lon_W, lat)
    if key in defaults_by_location:
        return defaults_by_location[key].copy()

    # Generic fallback
    print(f"  WARNING: no TIFs and no defaults for ({lon_W}°W, {lat}°) — using global means")
    return {k: PRIOR_MEANS[k] for k in PRIOR_MEANS if k != 'impact_melt_bonus'}


# ─── Main analysis ────────────────────────────────────────────────────────────

def run_analysis() -> None:
    try:
        from scipy.stats import beta as beta_dist
    except ImportError:
        print("ERROR: scipy required. Install with: pip install scipy")
        sys.exit(1)

    out_dir: Path = Path('outputs/diagnostics')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading present-epoch features for each location...")
    loc_objects: list[LocationFeatures] = []
    for name, lon_W, lat, label in LOCATIONS:
        print(f"  {name}...")
        present_f = load_present_features(lon_W, lat)
        loc_objects.append(LocationFeatures(name, lon_W, lat, present_f))

    print(f"\nComputing habitability across {len(EPOCHS)} epochs × {len(LOCATIONS)} locations...")

    # Result arrays
    means = np.zeros((len(EPOCHS), len(LOCATIONS)))
    lows  = np.zeros_like(means)
    highs = np.zeros_like(means)
    solar = np.array([solar_luminosity_ratio(t) for t in EPOCHS])
    temps = np.array([titan_surface_temp_K(t) for t in EPOCHS])

    for j, loc in enumerate(loc_objects):
        for i, t in enumerate(EPOCHS):
            feats = loc.features_at_epoch(t)
            mu, lo, hi = bayesian_posterior(feats)
            means[i, j] = mu
            lows[i, j]  = lo
            highs[i, j] = hi

    # Save CSV
    import csv
    csv_path = out_dir / 'location_habitability_timeseries.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch_Gya', 'solar_L_ratio', 'titan_T_K'] + \
                 [f"{loc.name}_mean" for loc in loc_objects] + \
                 [f"{loc.name}_lo" for loc in loc_objects] + \
                 [f"{loc.name}_hi" for loc in loc_objects]
        writer.writerow(header)
        for i, t in enumerate(EPOCHS):
            row = [f"{t:.4f}", f"{solar[i]:.4f}", f"{temps[i]:.2f}"]
            row += [f"{means[i,j]:.4f}" for j in range(len(LOCATIONS))]
            row += [f"{lows[i,j]:.4f}"  for j in range(len(LOCATIONS))]
            row += [f"{highs[i,j]:.4f}" for j in range(len(LOCATIONS))]
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    # ── Plot 1: Main timeseries ────────────────────────────────────────────────
    colours = plt.cm.tab10(np.linspace(0, 1, 10))
    labels  = [loc[3] for loc in LOCATIONS]

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[4, 1, 1], hspace=0.08)

    ax_main = fig.add_subplot(gs[0])
    ax_sol  = fig.add_subplot(gs[1], sharex=ax_main)
    ax_temp = fig.add_subplot(gs[2], sharex=ax_main)

    # Epoch tick labels
    x = EPOCHS

    # Shade regions
    ax_main.axvspan(-3.9, -3.2, alpha=0.07, color='red',    label='_LHB')
    ax_main.axvspan(-0.02, 0.02, alpha=0.10, color='gold',  label='_Cassini')
    ax_main.axvspan(5.0,  6.5,   alpha=0.10, color='orange',label='_RedGiant')

    for j, (name, lon_W, lat, label) in enumerate(LOCATIONS):
        ax_main.plot(x, means[:, j], '-o', color=colours[j], lw=2,
                     ms=4, label=label, zorder=5)
        ax_main.fill_between(x, lows[:, j], highs[:, j],
                             color=colours[j], alpha=0.12)

    # Vertical lines for key events
    for xv, lbl, col in [
        (-3.8, 'LHB peak',        'red'),
        ( 0.0, 'Present\n(Cassini)', 'gold'),
        ( 5.1, 'Eutectic\nthreshold','orange'),
        ( 5.4, 'Peak solar\nflux',  'darkorange'),
    ]:
        ax_main.axvline(xv, color=col, lw=1.2, ls='--', alpha=0.7)
        ax_main.text(xv + 0.05, 0.04, lbl, color=col, fontsize=7,
                     va='bottom', rotation=90, alpha=0.9)

    ax_main.set_ylabel('P(habitable | features)', fontsize=12)
    ax_main.set_ylim(0.0, 1.0)
    ax_main.legend(loc='upper left', ncol=5, fontsize=8, framealpha=0.9)
    ax_main.set_title(
        "Titan Surface Habitability: 10 Key Locations Across Geological Time\n"
        r"$-3.5\,\mathrm{Gya}$ (LHB)  →  Present  →  $+6.5\,\mathrm{Gya}$ (Red Giant)",
        fontsize=13
    )
    ax_main.set_xticklabels([])
    ax_main.grid(True, alpha=0.3, axis='y')

    # Solar luminosity panel
    log_sol = np.log10(np.maximum(solar, 0.1))
    ax_sol.fill_between(x, 0, log_sol, color='goldenrod', alpha=0.5)
    ax_sol.plot(x, log_sol, 'k-', lw=1)
    ax_sol.axhline(np.log10(2.7), color='orange', ls='--', lw=1.2)
    ax_sol.text(5.5, np.log10(2.7) + 0.05, r'2.7$L_\odot$ threshold', fontsize=7, color='orange')
    ax_sol.set_ylabel(r'$\log(L/L_0)$', fontsize=10)
    ax_sol.set_ylim(-0.4, 3.5)
    ax_sol.grid(True, alpha=0.3)
    ax_sol.set_xticklabels([])

    # Temperature panel
    ax_temp.fill_between(x, 0, temps, color='firebrick', alpha=0.4)
    ax_temp.plot(x, temps, 'k-', lw=1)
    ax_temp.axhline(WATER_AMMONIA_EUTECTIC_K, color='blue', ls='--', lw=1.2)
    ax_temp.text(5.2, WATER_AMMONIA_EUTECTIC_K + 2, r'H$_2$O-NH$_3$ eutectic', fontsize=7, color='blue')
    ax_temp.set_ylabel('T surface (K)', fontsize=10)
    ax_temp.set_xlabel('Epoch (Gya from present; negative = past)', fontsize=11)
    ax_temp.grid(True, alpha=0.3)

    # X-axis tick labels
    tick_positions = [-3.5, -2.0, -1.0, -0.5, 0.0, 0.5, 1.5, 3.0, 5.0, 6.0, 6.5]
    ax_temp.set_xticks(tick_positions)
    ax_temp.set_xticklabels([f"{t:+.1f}" for t in tick_positions], fontsize=8)
    ax_temp.set_xlim(EPOCHS[0] - 0.1, EPOCHS[-1] + 0.1)

    out1 = out_dir / 'location_habitability_timeseries.png'
    fig.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out1}")

    # ── Plot 2: Epoch snapshots ────────────────────────────────────────────────
    # Show bar charts at 6 key epochs
    key_epochs = [-3.5, -1.0, 0.0, 1.0, 5.2, 6.0]
    key_labels  = ['LHB\n(-3.5 Gya)', 'Mid-past\n(-1.0 Gya)',
                   'Present\n(0 Gya)', 'Near future\n(+1 Gya)',
                   'Red giant\nramp (+5.2)', 'Ocean peak\n(+6.0 Gya)']

    fig2, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig2.suptitle("Habitability Snapshots at Key Epochs — 10 Locations", fontsize=13)

    for ax, tgt, lbl in zip(axes.flat, key_epochs, key_labels):
        # Find nearest epoch in array
        idx = int(np.argmin(np.abs(EPOCHS - tgt)))
        vals  = means[idx, :]
        los   = lows[idx, :]
        his   = highs[idx, :]
        err_lo = vals - los
        err_hi = his - vals

        bars = ax.barh(labels[::-1], vals[::-1], color=colours[::-1],
                       xerr=[err_lo[::-1], err_hi[::-1]],
                       error_kw=dict(capsize=3, elinewidth=1),
                       alpha=0.85, height=0.7)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.7)
        ax.set_xlabel('P(habitable)')
        ax.set_title(lbl, fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)

        # Shade background by solar phase
        T_now = temps[idx]
        if T_now >= WATER_AMMONIA_EUTECTIC_K:
            ax.set_facecolor('#fff5e6')
        elif EPOCHS[idx] < -3.0:
            ax.set_facecolor('#f0f0ff')

    plt.tight_layout()
    out2 = out_dir / 'location_habitability_snapshots.png'
    fig2.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out2}")

    # ── Plot 3: Feature breakdown at present (heatmap) ────────────────────────
    feature_names = list(FEATURE_WEIGHTS.keys())
    feat_short = [f.replace('_', ' ').replace('hydrocarbon','HC')
                   .replace('abundance','abund').replace('interaction','interact')
                   .replace('complexity','complex').replace('diversity','divers')
                   .replace('geomorphologic','geomorph').replace('subsurface','subsfc')
                   .replace('impact melt bonus','impact melt') for f in feature_names]

    # Build matrix: locations × features
    feat_matrix = np.zeros((len(loc_objects), len(feature_names)))
    posteriors  = []
    for j, loc in enumerate(loc_objects):
        feats = loc.features_at_epoch(0.0)
        for k, fn in enumerate(feature_names):
            feat_matrix[j, k] = feats.get(fn, 0.0)
        mu, lo, hi = bayesian_posterior(feats)
        posteriors.append(mu)

    fig3, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(18, 7),
                                            gridspec_kw={'width_ratios': [3, 1]})
    fig3.suptitle("Feature Profiles and Habitability at Present Epoch", fontsize=13)

    loc_labels = [loc.name for loc in loc_objects]
    im = ax_heat.imshow(feat_matrix, aspect='auto', cmap='YlOrRd',
                        vmin=0, vmax=1, interpolation='nearest')
    ax_heat.set_xticks(range(len(feature_names)))
    ax_heat.set_xticklabels(feat_short, rotation=35, ha='right', fontsize=9)
    ax_heat.set_yticks(range(len(loc_objects)))
    ax_heat.set_yticklabels(loc_labels, fontsize=9)
    ax_heat.set_title("Feature values [0–1] at present epoch", fontsize=10)
    plt.colorbar(im, ax=ax_heat, fraction=0.03, label='Feature value [0-1]')
    # Add text values
    for i in range(feat_matrix.shape[0]):
        for j2 in range(feat_matrix.shape[1]):
            v = feat_matrix[i, j2]
            ax_heat.text(j2, i, f"{v:.2f}", ha='center', va='center',
                         fontsize=7, color='black' if v < 0.7 else 'white')

    # Bar chart of posterior means
    c10 = plt.cm.tab10(np.linspace(0, 1, 10))
    ax_bar.barh(loc_labels[::-1], [posteriors[i] for i in range(len(posteriors)-1,-1,-1)],
                color=c10[::-1], alpha=0.85, height=0.7)
    ax_bar.axvline(0.5, color='grey', ls='--', lw=1)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel('P(habitable) — present epoch')
    ax_bar.set_title("Posterior habitability", fontsize=10)
    ax_bar.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    out3 = out_dir / 'location_feature_spider.png'
    fig3.savefig(out3, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out3}")

    print(f"\nDone. Summary:")
    print(f"{'Location':24} {'Past P(H)':10} {'Present P(H)':13} {'Peak future P(H)':18} {'Peak epoch'}")
    print("-" * 80)
    past_idx    = int(np.argmin(np.abs(EPOCHS - (-3.5))))
    present_idx = int(np.argmin(np.abs(EPOCHS - 0.0)))
    for j, loc in enumerate(loc_objects):
        past_p    = means[past_idx, j]
        present_p = means[present_idx, j]
        peak_idx  = int(np.argmax(means[:, j]))
        peak_p    = means[peak_idx, j]
        peak_t    = EPOCHS[peak_idx]
        print(f"  {loc.name:22} {past_p:.3f}      {present_p:.3f}         "
              f"{peak_p:.3f}              {peak_t:+.2f} Gya")


if __name__ == '__main__':
    run_analysis()
