"""
configs/temporal_config.py
===========================
Temporal habitability configurations for the Titan pipeline.

Three temporal modes are supported:

  PAST    (~3.5 Gya — Early Titan / Late Heavy Bombardment era)
  PRESENT (Cassini epoch, ~2004–2017; this is what the original 8 features model)
  FUTURE  (~6 Gya from now — Sun as red giant, Lorenz et al. 1997)

═══════════════════════════════════════════════════════════════════════════════
SECTION 1 — WHERE EACH ORIGINAL PRIOR CAME FROM AND ITS QUALITY
═══════════════════════════════════════════════════════════════════════════════

Feature 1 — liquid_hydrocarbon  (weight 0.25, prior_mean 0.02)
  Source: Stofan et al. (2007), Nature 445:61; Hayes et al. (2008),
          GRL 35:L09204.
  Basis:  Lakes cover ~1.5–2% of total surface (directly measured by SAR).
          North polar coverage ~40%; global mean ~2%.
  Quality: STRONG — directly observed global fraction.
  Alternative: Hayes (2016) Ann.Rev. gives total lake/sea area ~0.65×10⁶ km²
               = 0.013 of 8.3×10⁷ km² surface = 1.3%.  Alternative prior: 0.015.
               This is a minor revision; 0.02 is acceptable.

Feature 2 — organic_abundance  (weight 0.20, prior_mean 0.60)
  Source: Barnes et al. (2007), Icarus 186:242; Le Mouélic et al. (2019),
          Icarus 319:121.
  Basis:  "Near-ubiquitous" is the qualitative description; Malaska et al.
          (2025) Ch.9 gives terrain fractions: Plains 65%, Dunes 17%, which
          are organic-rich.  Together ~82% of surface is organic-dominated.
  Quality: MODERATE — "near-ubiquitous" is a qualitative claim;
           quantitative organic fraction per pixel is not directly measured.
  Alternative: If 65%+17% = 82% of surface is organic-rich terrain and the
               remainder (mountains, labyrinth, craters) shows water-ice
               exposure, a better prior is 0.70–0.75.
               Cable et al. (2012) Astrobiology 12:163 notes tholins coat
               essentially all non-lake surfaces.  Updated prior: 0.70.

Feature 3 — acetylene_energy  (weight 0.20, prior_mean 0.30)
  Source: McKay & Smith (2005) Icarus 178:274; Strobel (2010) Icarus 208:878;
          Yanez et al. (2024) Icarus 408:115969.
  Basis:  H₂ downward flux 10²⁵ mol/s (Strobel 2010) — this is OBSERVED and
          confirmed.  The chemical energy IS available.  The question is only
          whether biology or abiotic processes explain the depletion.
  Quality: MODERATE — energy source confirmed; prior_mean of 0.30 is
           conservative because it conflates "energy available" with "life
           consuming it".  These should be separated.  The energy availability
           should have a higher prior (~0.50) since Strobel's observation is
           confirmed.  The life probability is a separate inference.
  Alternative: Split into energy_available (prior 0.50, well-constrained by
               Strobel 2010) and biological_signal (prior 0.15, speculative).
               In the combined single feature, 0.35 is more defensible.

Feature 4 — methane_cycle  (weight 0.15, prior_mean 0.40)
  Source: Turtle et al. (2011) Science 331:1414; Mitchell & Lora (2016)
          Ann.Rev. 44:353; Hayes et al. (2018) Nature Geoscience 11:306.
  Basis:  Active methane cycle confirmed by direct ISS/VIMS observations
          (rain events, channel networks).  Mid-latitude activity observed.
  Quality: STRONG — directly observed.  0.40 is reasonable (not all surface
           equally active).
  Alternative: Birch et al. (2025) Titan After C-H Ch.10 notes cycle is
               concentrated at 45°–60°N/S; a latitude-weighted global mean
               of 0.35–0.40 is appropriate.  No major revision needed.

Feature 5 — surface_atm_interaction  (weight 0.08, prior_mean 0.35)
  Source: Hayes (2016) Ann.Rev. 44:57.
  Basis:  Lake margins + channel heads = zones of evaporation/condensation.
          This is a derived feature, not directly observed.
  Quality: WEAK — the underlying concept is sound but the prior is not
           directly constrained by a measurement.  It is derived from the
           lake and fluvial feature distribution.
  Alternative: No better constraint exists.  Flag as "model-derived" prior.

Feature 6 — topographic_complexity  (weight 0.06, prior_mean 0.25)
  Source: Lorenz et al. (2013) Icarus 225:367; Lopes et al. (2019) NatAstron.
  Basis:  Hummocky and labyrinthine terrains show highest roughness and
          water-ice content.  Global roughness distribution is estimated.
  Quality: MODERATE — DEM roughness is measured (GTDR) but its link to
           habitability is inferred.

Feature 7 — geomorphologic_diversity  (weight 0.04, prior_mean 0.30)
  Source: Malaska et al. (2025) Titan After C-H Ch.9.
  Basis:  Ecotone diversity argument (Earth analogy).  Shannon diversity
          of terrain classes.
  Quality: WEAK — terrestrial ecotone analogy applied to alien world.
           This is the most speculative feature.  Prior is essentially a
           Bayesian regulariser rather than a measurement-grounded constraint.

Feature 8 — subsurface_ocean  (weight 0.02, prior_mean 0.10)
  Source: Iess et al. (2012) Science 337:457; Affholder et al. (2025) PSJ 6:86.
  Basis:  k₂ = 0.589 ± 0.150 confirms global ocean.  Affholder (2025) shows
          glycine fermentation viable.  BUT Neish et al. (2024) Astrobiology
          shows organic flux to the ocean is extremely limited (~7,500 kg/yr
          glycine — equivalent to one elephant), making the ocean poorly
          connected to surface organics.
  Quality: STRONG that ocean exists; WEAK that it is habitable.
  Alternative: Given Neish et al. (2024), the prior for ocean habitability
               should be LOWER than 0.10.  A value of 0.05 is more defensible
               for the current epoch.  For the past (higher impactor flux,
               more Menrva-sized craters), it may be higher (~0.15).

═══════════════════════════════════════════════════════════════════════════════
SECTION 2 — TEMPORAL MODES: SCIENTIFIC BASIS
═══════════════════════════════════════════════════════════════════════════════

PAST mode  (~3.5 Gya — Late Heavy Bombardment / Early Titan)
─────────────────────────────────────────────────────────────
Scientific basis:
  - Higher impactor flux during LHB created more and larger impact melt oases.
    O'Brien et al. (2005) Icarus 173:243 show 150 km craters sustain liquid
    water for 10³–10⁴ yr.  Earlier Titan had more of these.
  - Higher internal heat (radioactive decay, residual accretion) drove more
    cryovolcanism → more surface-ocean exchange pathways.
    Tobie et al. (2006) Nature 440:61 propose episodic cryovolcanic outgassing
    maintained atmospheric methane.
  - Less accumulated organic surface material (fewer Gyr of UV photolysis /
    tholins).  Early Titan surface had fewer tholins.
  - Liquid water exposure was episodic but potentially frequent near large craters.
  - Surface age 200 Myr–1 Gyr (Neish & Lorenz 2012; Hedgepeth et al. 2020)
    means most of the current surface is ALREADY past-epoch material.

Key proxy available in existing Cassini data:
  - Impact melt exposure at craters: SAR-bright annuli around craters
    (Neish et al. 2018 Astrobiology 18:571 shows water-ice + organic mixing)
  - Crater degradation state: fresh (recent melt) vs degraded (old)
    (Neish et al. 2013 Icarus 223:82; Hedgepeth et al. 2020 Icarus 344:113664)
  - Cryovolcanic candidate features: Lopes et al. (2007) Icarus 186:395;
    Lopes et al. (2013) JGR-Planets 118:416

New feature for PAST mode:
  "impact_melt_proxy" = SAR-bright annuli at crater margins
                       (water-ice + organic mixing zones)
  Source dataset: existing SAR + Hedgepeth crater catalog
  Prior mean: 0.25 (10% of surface within crater melt zones historically;
              elevated for PAST because more/larger ancient craters)

  "cryovolcanic_flux" = proximity to cryovolcanic candidate features
  Source: Lopes et al. 2007/2013 feature catalog (from SAR raster)
  Prior mean: 0.20 (past: higher internal heat → more cryovolcanism)

Assumption required from user:
  ⚠️ WHICH epoch? "Past" spans 4.5 Gya (accretion) to ~500 Mya (recent).
  The scientific literature clusters around two sub-modes:
    (a) Early Titan ~3.5–4.0 Gya: heavy bombardment, possibly liquid surface
        water during accretion warming, high cryovolcanic flux.
    (b) "Young surface" epoch ~200–500 Mya: current surface was emplaced.
        Relevant to interpreting observed terrain ages.
  RECOMMENDATION: Use (a) as PAST default; add sub-mode PAST_RECENT for (b).
  For this implementation, PAST = ~3.5 Gya unless user specifies otherwise.

PRESENT mode  (Cassini epoch ~2004–2017)
─────────────────────────────────────────
No changes from original 8 features.  Subsurface ocean prior lowered from
0.10 to 0.05 to reflect Neish et al. (2024) finding of very limited organic
flux to ocean.  organic_abundance raised from 0.60 to 0.70 based on terrain
fractions (Cable et al. 2012; Malaska et al. 2025).

FUTURE mode  (~6 Gya — Sun as red giant)
─────────────────────────────────────────
Scientific basis:
  Lorenz, Lunine & McKay (1997) Geophys.Res.Lett. 24:2905:
  - ~6 Gyr from now, UV flux from red giant drops → haze production stops
  - Anti-greenhouse weakens → surface T rises from 94 K to ~200 K
  - Window of "several hundred Myr" of water-ammonia surface oceans
  - Duration EXCEEDS time for life to begin on Earth (~200–800 Myr)
  - Surface resembles "warm little pond" chemistry with abundant organics

Key changes relative to PRESENT:
  1. liquid_hydrocarbon REPLACED by water_ammonia_solvent
     (liquid water-ammonia oceans now cover most of surface → prior ~0.85)
  2. organic_abundance HIGHER (centuries of tholin accumulation → prior 0.85)
  3. methane_cycle REPLACED by water_ammonia_cycle
     (methane evaporated/photolyzed; water-ammonia takes over as working fluid)
  4. acetylene_energy modified: UV photolysis stops → no new C2H2.
     BUT accumulated C2H2 and organics dissolve in water-ammonia → rich chemistry.
     Prior lowered for ongoing energy flux, but raised for total organic substrate.
  5. subsurface_ocean NOW CONNECTED to surface (ocean expands to surface)
     Prior raised from 0.05 to 0.90 (it IS the surface).

New features for FUTURE mode:
  "organic_stockpile" = accumulated organic inventory (proxy: terrain organic coverage)
  Prior mean: 0.85 (all of Titan's surface organics available as substrate)

  "water_ammonia_solvent" = liquid water-ammonia solvent availability
  Prior mean: 0.85 (global surface oceans during ~200 Myr window)

  "thermal_window" = temperature within water-ammonia eutectic range (176–200 K)
  Source: Solar evolution models (not a Cassini dataset — derived from
          Lorenz 1997 thermal models applied to current topographic data)
  Prior mean: 0.70 (model uncertainty in exactly when/how long window opens)

Limitations and assumptions for FUTURE mode:
  ⚠️ No Cassini dataset directly constrains future habitability.
     All future features are DERIVED from present-day data used as proxies:
     - Organic inventory → future chemical substrate
     - Topography → future ocean bathymetry and shoreline distribution
     - Current atmospheric composition → future atmosphere via models
  ⚠️ The red giant scenario (Lorenz 1997) assumes:
     (a) Titan's atmosphere persists for 6 Gyr
     (b) Methane does not fully escape before the red giant phase
     (c) Saturn's tidal capture of Titan does not change substantially
     (d) Titan's internal heat is sufficient to prevent complete freezing
  ⚠️ The "several hundred Myr" window is highly uncertain:
     The duration depends on the rate of UV flux drop in the red giant transition.
     Range in literature: 100 Myr to 1 Gyr.

═══════════════════════════════════════════════════════════════════════════════
SECTION 3 — MISSING DATASETS BY TEMPORAL MODE
═══════════════════════════════════════════════════════════════════════════════

PAST — missing datasets:
  1. Cryovolcanic feature catalog (Lopes et al. 2007/2013) as a raster
     → Can be derived from existing SAR by identifying bright flow features.
     Source: Lopes et al. (2013) JGR-Planets 118:416 (Table 1, feature coordinates)
     Action: Add as new DatasetSpec "cryovolcanic_features"

  2. Impact melt exposure index
     → Fresh craters show SAR-bright annuli = water-ice exposed by impact melt.
     Source: Neish et al. (2018) Astrobiology; Hedgepeth et al. (2020) catalog
     Action: Derive from SAR + Hedgepeth crater positions (Table 1 coordinates)
     Available: Hedgepeth crater positions on PDS/supplementary material

  3. Titan internal heat flow model output
     → Needed to constrain cryovolcanic flux. Not a Cassini dataset.
     Source: Tobie et al. (2006) Nature 440:61 models; Mitri et al. (2008)
     Action: Use as a scalar constant (global, not spatially resolved)
     ⚠️ ASSUMPTION: Internal heat flow is not spatially mapped. Use scalar.

PRESENT — missing datasets:
  4. Atmospheric H₂ depletion map (spatially resolved)
     → Strobel (2010) gives a global flux; no spatially resolved map exists.
     Source: CIRS/INMS data; no spatial map published
     ⚠️ Applied as global constant in feature 3 (acetylene_energy).
     Clarification: The acetylene_energy feature uses SAR backscatter as a
     SURFACE proxy for organic depletion, not the atmospheric H₂ flux directly.
     This is an ASSUMPTION that should be acknowledged.

  5. RADAR emissivity (passive radiometry) for subsurface properties
     → Cassini RADAR radiometer data gives emissivity ∝ subsurface composition
     Source: Janssen et al. (2016) Icarus 270:443 (2.18 cm emissivity map)
     Available: PDS Radar node; USGS Astropedia
     Action: Add as "sar_emissivity" dataset — useful for detecting water-ice
     and ammonia-hydrate subsurface (past cryovolcanic products)

FUTURE — missing datasets:
  6. Solar evolution model (insolation timeline)
     → Not observational data; physics models of solar luminosity evolution
     Source: Bressan et al. (2012) MNRAS 427:127 (PARSEC isochrones);
             Lorenz et al. (1997) GRL 24:2905 (Titan-specific)
     Action: Encode as a temporal_scaling_factor per mode, not a raster.

  7. Titan atmospheric escape model
     → How much methane/nitrogen remains in 6 Gyr?
     Source: Johnson et al. (2009); Strobel (2009)
     ⚠️ ASSUMPTION: Atmosphere persists. This is theoretically supported but
     not confirmed. Flag as a major uncertainty.

═══════════════════════════════════════════════════════════════════════════════
SECTION 4 — ACCEPTED DESIGN DECISIONS (D1–D4)
═══════════════════════════════════════════════════════════════════════════════

These decisions have been confirmed by the researcher and are now the
canonical defaults of the pipeline.  Each parameter is configurable via
CLI flags (see run_pipeline.py --help for exact argument names).

D1: PAST epoch — ACCEPTED: 3.5 Gya default, configurable
    ────────────────────────────────────────────────────────
    The Late Heavy Bombardment / early cryovolcanic era (~3.5 Gya) is accepted
    as the default past epoch. During this period, higher impactor flux created
    more and larger impact melt oases; residual internal heat drove more
    cryovolcanism. SAR-bright crater annuli observed today (D4) are interpreted
    as relics from this epoch.

    Default: 3.5 Gya
    CLI:     --past-epoch-gya FLOAT
    Range:   Sensible values 0.5–4.5 Gya.  Lower = more recent episode (e.g.
             young-surface epoch ~0.5 Gya); higher = accretion era (~4.0 Gya).
    Ref:     Neish & Lorenz (2012) doi:10.1016/j.pss.2011.02.016
             Artemieva & Lunine (2003) doi:10.1016/S0019-1035(02)00039-9

D2: Near-future solar-warming window — ACCEPTED: 100–400 Myr, uniform warming
    ────────────────────────────────────────────────────────────────────────────
    More recent radiative-transfer estimates narrow the onset of the near-future
    habitability window to 100–400 Myr from now (as solar luminosity gradually
    increases ~10%/Gyr). This is MUCH sooner than the classical red-giant window
    (Lorenz et al. 1997, ~6 Gya) and affects a different feature (the
    subsurface_ocean temporal prior in PRESENT mode, via HabitabilityWindowConfig).
    The FUTURE temporal mode (red giant, ~6 Gya) remains separate and unchanged.

    ⚠️  EXPLICIT ASSUMPTION — uniform global warming:
    Solar brightening is applied as a spatially uniform temperature increment
    across Titan's entire surface. A full GCM would show differential warming
    by latitude and altitude, with equatorial regions warming faster. No GCM
    output at the resolution of the SAR mosaic is currently available, so the
    uniform assumption is used. This likely UNDERESTIMATES habitability at low
    latitudes and OVERESTIMATES it at high latitudes. Users with GCM output
    can override by disabling via --no-uniform-warming and implementing a
    spatially varying warming raster.

    Default: 100–400 Myr, uniform warming = True
    CLI:     --future-window-min MYR  (default 100)
             --future-window-max MYR  (default 400)
             --no-uniform-warming     (disables future-window prior entirely)
    Ref:     Lorenz, Lunine & McKay (1997) doi:10.1029/97GL52843
             Lunine & Lorenz (2009) Annual Rev. Earth Planet. Sci.
             Hörst (2017) doi:10.1002/2016JE005240

D3: Subsurface ocean prior — ACCEPTED: 0.03 default, configurable
    ────────────────────────────────────────────────────────────────
    Revised down from an earlier working value of 0.10 based on:
    Neish et al. (2024) Astrobiology: the organic flux from Titan's surface to
    its subsurface ocean is ~7,500 kg/yr glycine (~one elephant equivalent).
    This severely limits present-epoch ocean habitability by demonstrating that
    the ocean is effectively isolated from the organic-rich surface inventory.
    The k2=0.589 measurement (Iess 2012) confirms the ocean EXISTS but places
    no constraint on its habitability.

    Default: 0.03
    CLI:     --subsurface-ocean-prior FLOAT  (range [0, 1])
    Note:    The SAR bright-annuli proxy (D4) boosts this locally by up to +0.30
             at pixels with strong annular morphology.
    Ref:     Neish et al. (2024) doi:10.1089/ast.2023.0055
             Iess et al. (2012)  doi:10.1126/science.1219631

D4: SAR bright annuli as past-liquid-water proxy — ACCEPTED
    ──────────────────────────────────────────────────────────
    Radar-bright ring (annular) structures in Cassini SAR data are accepted as
    a proxy for past episodes of liquid water–organic contact. The rationale:

      (a) Impact melt rims: bolide impacts briefly melt the target material,
          forming liquid water pools in contact with the organic surface.
          As the melt refreezes, it leaves a radar-bright ring at the flow
          front (high surface roughness, water-ice exposure).
          Ref: Neish et al. (2018) doi:10.1089/ast.2017.1758
               Artemieva & Lunine (2003) doi:10.1016/S0019-1035(02)00039-9

      (b) Cryovolcanic flow fronts: water-ammonia slurry flowing from volcanic
          vents forms bright annular lobes as the leading edge cools and
          roughens against the organic substrate.
          Ref: Lopes et al. (2007) doi:10.1016/j.icarus.2006.09.006
               Wood et al. (2010) Icarus 206:334–344

    DISCRIMINATOR: The key morphological indicator is ANNULAR shape (bright ring
    around a darker interior), NOT simply high backscatter. High SAR backscatter
    alone is non-specific (water-ice, rough mountains, bright dunes all qualify).
    Implementation uses a ring-shaped spatial filter (bright relative to far
    neighbourhood, not immediate neighbourhood) as a first-order morphological
    discriminant. Full Hough-transform circle detection would be more precise
    but requires the full-resolution SAR at 351 m/px.

    See: titan/features.py Feature 8 (_subsurface_ocean) for full implementation.

References for this file
─────────────────────────
Barnes et al. (2007)          Icarus 186:242     doi:10.1016/j.icarus.2006.08.021
Cable et al. (2012)           Astrobiology 12:163 doi:10.1089/ast.2011.0751
Hayes (2016)                  Ann.Rev. 44:57     doi:10.1146/annurev-earth-060115-012247
Hayes et al. (2008)           GRL 35:L09204      doi:10.1029/2007GL032324
Hedgepeth et al. (2020)       Icarus 344:113664  doi:10.1016/j.icarus.2020.113664
Iess et al. (2012)            Science 337:457    doi:10.1126/science.1219631
Le Mouélic et al. (2019)      Icarus 319:121     doi:10.1016/j.icarus.2018.09.017
Lopes et al. (2007)           Icarus 186:395     doi:10.1016/j.icarus.2006.09.006
Lopes et al. (2013)           JGR-Planets 118:416 doi:10.1002/jgre.20062
Lorenz, Lunine & McKay (1997) GRL 24:2905        doi:10.1029/97GL52843
Malaska et al. (2025)         Titan After C-H Ch.9
McKay & Smith (2005)          Icarus 178:274     doi:10.1016/j.icarus.2005.05.018
Mitchell & Lora (2016)        Ann.Rev. 44:353    doi:10.1146/annurev-earth-060115-012054
Neish & Lorenz (2012)         Plan.Sp.Sci. 60:26 doi:10.1016/j.pss.2011.02.016
Neish et al. (2013)           Icarus 223:82      doi:10.1016/j.icarus.2012.11.030
Neish et al. (2018)           Astrobiology 18:571 doi:10.1089/ast.2017.1758
Neish et al. (2024)           Astrobiology        doi:10.1089/ast.2023.0055
O'Brien et al. (2005)         Icarus 173:243     doi:10.1016/j.icarus.2004.07.034
Stofan et al. (2007)          Nature 445:61      doi:10.1038/nature05608
Strobel (2010)                Icarus 208:878     doi:10.1016/j.icarus.2010.02.009
Tobie et al. (2006)           Nature 440:61      doi:10.1038/nature04497
Turtle et al. (2011)          Science 331:1414   doi:10.1126/science.1201063
Yanez et al. (2024)           Icarus 408:115969  doi:10.1016/j.icarus.2024.115969
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class TemporalMode(str, Enum):
    """
    Temporal mode for the habitability analysis.

    PAST    : Early Titan, ~3.5–4.0 Gya.
              High impactor flux, higher internal heat, episodic liquid water
              from impact melt oases and cryovolcanism.
              Lower accumulated organics than present day.

    PRESENT : Cassini epoch (~2004–2017).
              Liquid methane/ethane lakes.  Active methane cycle.
              Subsurface water-ammonia ocean confirmed but poorly connected
              to surface organics (Neish et al. 2024).

    FUTURE  : Red giant epoch, ~6 Gya from now (Lorenz et al. 1997).
              Global water-ammonia oceans.  No UV → no new haze.
              Window of ~several hundred Myr with surface T ~200 K.
              Abundant organic substrate from billions of years of tholin
              accumulation.
    """
    PAST    = "past"
    PRESENT = "present"
    FUTURE  = "future"


# ---------------------------------------------------------------------------
# Feature name sets per temporal mode
# ---------------------------------------------------------------------------

#: Feature names for PRESENT mode (original 8).
PRESENT_FEATURES: List[str] = [
    "liquid_hydrocarbon",
    "organic_abundance",
    "acetylene_energy",
    "methane_cycle",
    "surface_atm_interaction",
    "topographic_complexity",
    "geomorphologic_diversity",
    "subsurface_ocean",
]

#: Feature names for PAST mode.
#: Replaces subsurface_ocean with cryovolcanic_flux and adds impact_melt_proxy.
PAST_FEATURES: List[str] = [
    "liquid_hydrocarbon",        # impact melt oases (proxy for past liquid water)
    "organic_abundance",         # lower than present — fewer Gyr of tholin production
    "acetylene_energy",          # chemical energy from impact-driven chemistry
    "methane_cycle",             # uncertain in early Titan; modelled as reduced
    "surface_atm_interaction",   # higher in past due to more cryovolcanism
    "topographic_complexity",    # unchanged — DEM is inherited from PRESENT
    "geomorphologic_diversity",  # unchanged — terrain map is current epoch
    "impact_melt_proxy",         # NEW: SAR-bright crater annuli = past liquid water
    "cryovolcanic_flux",         # NEW: proximity to cryovolcanic candidate sites
]

#: Feature names for FUTURE mode.
#: Methane-cycle features replaced by water-ammonia analogs.
FUTURE_FEATURES: List[str] = [
    "water_ammonia_solvent",     # REPLACES liquid_hydrocarbon (global ocean)
    "organic_stockpile",         # REPLACES organic_abundance (accumulated organics)
    "dissolved_energy",          # REPLACES acetylene_energy (C₂H₂ in warm water)
    "water_ammonia_cycle",       # REPLACES methane_cycle
    "surface_atm_interaction",   # retained — now ocean-atmosphere exchange
    "topographic_complexity",    # retained — future ocean bathymetry proxy
    "geomorphologic_diversity",  # retained — future habitat diversity proxy
    "global_ocean_habitability", # REPLACES subsurface_ocean (ocean IS the surface)
]

# Map from temporal mode → feature name list
TEMPORAL_FEATURE_NAMES: Dict[TemporalMode, List[str]] = {
    TemporalMode.PAST:    PAST_FEATURES,
    TemporalMode.PRESENT: PRESENT_FEATURES,
    TemporalMode.FUTURE:  FUTURE_FEATURES,
}


# ---------------------------------------------------------------------------
# Priors per temporal mode
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalPriorSet:
    """
    Bayesian prior weights and prior means for one temporal mode.

    All weights must sum to 1.0 (enforced by validate()).
    All prior_means must be in [0, 1].
    """
    mode: TemporalMode
    feature_names: Tuple[str, ...]
    weights: Tuple[float, ...]
    prior_means: Tuple[float, ...]
    citations: Tuple[str, ...]  # one citation per feature

    def as_weight_dict(self) -> Dict[str, float]:
        return dict(zip(self.feature_names, self.weights))

    def as_mean_dict(self) -> Dict[str, float]:
        return dict(zip(self.feature_names, self.prior_means))

    def validate(self) -> None:
        total = sum(self.weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"[{self.mode.value}] Weights sum to {total:.4f}, must be 1.0 ± 0.01"
            )
        for n, w in zip(self.feature_names, self.weights):
            if not 0 <= w <= 1:
                raise ValueError(f"[{self.mode.value}] Weight '{n}' = {w} out of [0,1]")
        for n, m in zip(self.feature_names, self.prior_means):
            if not 0 <= m <= 1:
                raise ValueError(f"[{self.mode.value}] Prior mean '{n}' = {m} out of [0,1]")


def get_prior_set(mode: TemporalMode) -> TemporalPriorSet:
    """
    Return the research-grounded prior set for the given temporal mode.

    All values are documented with source citations.  See module docstring
    for detailed justification of each number.
    """
    if mode == TemporalMode.PRESENT:
        return TemporalPriorSet(
            mode=TemporalMode.PRESENT,
            feature_names=tuple(PRESENT_FEATURES),
            weights=(
                0.25,  # liquid_hydrocarbon     — highest weight: solvent availability
                0.20,  # organic_abundance       — substrate for chemistry
                0.20,  # acetylene_energy        — confirmed H₂ depletion (Strobel 2010)
                0.15,  # methane_cycle           — confirmed active cycle
                0.08,  # surface_atm_interaction — lake/channel margins
                0.06,  # topographic_complexity  — micro-environment diversity
                0.04,  # geomorphologic_diversity— terrain ecotone analogy
                0.02,  # subsurface_ocean        — ocean confirmed; surface connection weak
            ),
            prior_means=(
                0.020,  # liquid_hydrocarbon: 1.5–2% surface (Hayes 2008; Stofan 2007)
                0.700,  # organic_abundance:  ~82% organic terrain (Cable 2012; Malaska 2025)
                         #                    revised UP from 0.60
                0.350,  # acetylene_energy:   H₂ depletion confirmed (Strobel 2010)
                         #                    revised UP from 0.30 — energy IS there
                0.400,  # methane_cycle:      active at mid-latitudes (Turtle 2011)
                0.350,  # surface_atm_interaction: lake margins (Hayes 2016)
                0.250,  # topographic_complexity: DEM roughness (Lorenz 2013)
                0.300,  # geomorphologic_diversity: Shannon terrain (Malaska 2025)
                0.030,  # subsurface_ocean:   ocean confirmed; organic flux ~1 elephant/yr
                         #                    revised DOWN from 0.10 (Neish et al. 2024)
            ),
            citations=(
                "Stofan2007; Hayes2008",
                "Cable2012; Malaska2025; LeMouelic2019",
                "McKay&Smith2005; Strobel2010; Yanez2024",
                "Turtle2011; Mitchell&Lora2016",
                "Hayes2016",
                "Lorenz2013; Lopes2019",
                "Malaska2025",
                "Iess2012; Neish2024; Affholder2025",
            ),
        )

    elif mode == TemporalMode.PAST:
        return TemporalPriorSet(
            mode=TemporalMode.PAST,
            feature_names=tuple(PAST_FEATURES),
            weights=(
                0.22,  # liquid_hydrocarbon    — impact melt oases more common in LHB
                0.16,  # organic_abundance      — LOWER: fewer Gyr of tholin production
                0.18,  # acetylene_energy       — HIGHER: impact-driven chemistry
                0.10,  # methane_cycle          — LOWER: uncertain in early Titan
                0.10,  # surface_atm_interaction— HIGHER: more cryovolcanism
                0.06,  # topographic_complexity — same (using current DEM as proxy)
                0.03,  # geomorphologic_diversity — lower (using current map as proxy)
                0.10,  # impact_melt_proxy      — NEW: key past-habitability feature
                0.05,  # cryovolcanic_flux      — NEW: internal heat → cryovolcanism
            ),
            prior_means=(
                0.025,  # liquid_hydrocarbon: slightly elevated vs present from higher LHB impact flux
                         # Artemieva & Lunine (2005): global melt layer improbable; only local
                         # transient melting near newly formed craters. O'Brien et al. (2005)
                         # gives melt LONGEVITY (10^3-10^4 yr for 150 km craters), not surface
                         # fraction. Present-day ~1.5% lakes → PAST ~2.5% with transient melt oases.
                0.300,  # organic_abundance: lower than present; fewer Gyr of production
                         # Early Titan had ~2 Gyr less UV photolysis time
                0.450,  # acetylene_energy: impact-driven HCN/C2H2 chemistry very active
                         # Madan et al. (2026) PSJ: amino acid synthesis in Selk-type craters
                0.200,  # methane_cycle: uncertain; reduced if cryovolcanism, not rain
                         # Tobie et al. (2006): episodic outgassing, not continuous cycle
                0.500,  # surface_atm_interaction: higher — more cryovolcanic conduits
                         # Lopes et al. (2007, 2013): cryovolcanic features observed
                0.250,  # topographic_complexity: same as PRESENT (using current DEM)
                0.200,  # geomorphologic_diversity: somewhat lower (using current map)
                0.500,  # impact_melt_proxy: where craters exist, impact-melt habitability
                         # potential is high — O'Brien et al. (2005) shows melt longevity
                         # 10^3-10^4 yr for large (150 km) craters; Artemieva & Lunine (2003)
                         # modelled transient melting at newly formed craters across history.
                         # Neish et al. (2018) confirms water-ice exposure at fresh crater floors.
                         # Prior 0.50 = high habitability WHERE craters occur, not surface %.
                0.300,  # cryovolcanic_flux: strong past activity implied by k2, methane
                         # Tobie et al. (2006); Lopes et al. (2013)
            ),
            citations=(
                "OBrien2005; Artemieva&Lunine2003",
                "Cable2012; Malaska2025 [reduced for early epoch]",
                "Madan2026; McKay&Smith2005 [impact-driven]",
                "Tobie2006 [episodic methane]",
                "Lopes2007; Lopes2013 [cryovolcanism]",
                "Lorenz2013 [current DEM proxy]",
                "Lopes2019 [current map proxy]",
                "Neish2018; Hedgepeth2020 [SAR crater annuli]",
                "Tobie2006; Lopes2007 [internal heat]",
            ),
        )

    elif mode == TemporalMode.FUTURE:
        return TemporalPriorSet(
            mode=TemporalMode.FUTURE,
            feature_names=tuple(FUTURE_FEATURES),
            weights=(
                0.30,  # water_ammonia_solvent    — dominant: global ocean is the key
                0.20,  # organic_stockpile        — vast accumulated substrate
                0.15,  # dissolved_energy         — organics dissolving in warm water
                0.10,  # water_ammonia_cycle      — analog to methane cycle
                0.08,  # surface_atm_interaction  — ocean-atmosphere exchange
                0.07,  # topographic_complexity   — ocean depth/bathymetry proxy
                0.05,  # geomorphologic_diversity — habitat diversity (shallow vs deep)
                0.05,  # global_ocean_habitability— ocean IS the surface
            ),
            prior_means=(
                0.850,  # water_ammonia_solvent: ~200K surface → global liquid
                         # Lorenz et al. (1997): surface T reaches ~200K during window
                         # Uncertainty: when exactly / how long → prior not 1.0
                0.850,  # organic_stockpile: billions of years of tholin accumulation
                         # Malaska et al. (2025): organics on ~82% of current surface
                         # Future: all surface organics available as substrate → high
                0.600,  # dissolved_energy: C2H2, HCN dissolve in 200K water-ammonia
                         # Neish et al. (2009) Icarus 201:412: tholins produce amino acids
                         # when dissolved in ammonia-water; abundant substrate available
                0.600,  # water_ammonia_cycle: analog to methane cycle under warm conditions
                         # Lorenz et al. (1997): evaporation, rainfall of water-ammonia
                0.400,  # surface_atm_interaction: ocean-atmosphere boundary exchange
                         # Similar to lake margins but global
                0.400,  # topographic_complexity: future ocean bathymetry proxy
                         # Low terrain = deeper ocean = potentially richer environment
                0.400,  # geomorphologic_diversity: submerged terrain type diversity
                0.800,  # global_ocean_habitability: ocean IS the surface during window
                         # Lorenz 1997; Affholder 2025: glycine fermentation viable
                         # High prior because the question becomes WHEN not WHETHER
            ),
            citations=(
                "Lorenz1997 GRL; duration ~100-400 Myr",
                "Malaska2025; Cable2012 [accumulated organics]",
                "Neish2009; Madan2026 [amino acids from tholins in water]",
                "Lorenz1997 [water-ammonia meteorology predicted]",
                "Lorenz1997; Hayes2016 [ocean-atmosphere exchange]",
                "Lorenz2013 [topography → future bathymetry proxy]",
                "Lopes2019 [terrain map → future habitat diversity]",
                "Lorenz1997; Affholder2025 [surface ocean = habitable]",
            ),
        )

    else:
        raise ValueError(f"Unknown temporal mode: {mode}")


# ---------------------------------------------------------------------------
# Differences table (for reporting)
# ---------------------------------------------------------------------------

def describe_prior_changes(
    mode: TemporalMode,
) -> str:
    """
    Return a human-readable summary of how priors differ from PRESENT for
    the given temporal mode.
    """
    if mode == TemporalMode.PRESENT:
        return (
            "PRESENT mode — original 8 features with two corrections:\n"
            "  • organic_abundance prior_mean:   0.60 → 0.70  "
            "(Cable 2012; ~82% organic terrain)\n"
            "  • acetylene_energy prior_mean:    0.30 → 0.35  "
            "(Strobel 2010 H₂ depletion is confirmed)\n"
            "  • subsurface_ocean prior_mean:    0.10 → 0.03  "
            "(Neish 2024: organic flux to ocean ~1 elephant/year)"
        )

    elif mode == TemporalMode.PAST:
        return (
            "PAST mode (~3.5 Gya) — 9 features:\n"
            "  • liquid_hydrocarbon prior_mean:  0.02 → 0.06  "
            "(more/larger impact craters → more melt oases)\n"
            "  • organic_abundance prior_mean:   0.70 → 0.30  "
            "(fewer Gyr of UV photolysis → less tholin production)\n"
            "  • acetylene_energy prior_mean:    0.35 → 0.45  "
            "(impact-driven HCN/C2H2 chemistry more active)\n"
            "  • methane_cycle prior_mean:       0.40 → 0.20  "
            "(episodic cryovolcanic outgassing, not rain cycle)\n"
            "  • surface_atm_interaction:        0.35 → 0.50  "
            "(more cryovolcanic conduits to surface)\n"
            "  + impact_melt_proxy (NEW):        prior_mean 0.50\n"
            "  + cryovolcanic_flux (NEW):        prior_mean 0.30\n"
            "  Weights redistributed to sum to 1.0.\n"
            "  ⚠️  ASSUMPTION: 'Past' = ~3.5 Gya (LHB epoch). Confirm?"
        )

    elif mode == TemporalMode.FUTURE:
        return (
            "FUTURE mode (~6 Gya, red giant window) — 8 transformed features:\n"
            "  • liquid_hydrocarbon → water_ammonia_solvent:  0.02 → 0.85\n"
            "    (global surface ocean during ~200K window; Lorenz 1997)\n"
            "  • organic_abundance → organic_stockpile:       0.70 → 0.85\n"
            "    (billions of years of tholin accumulation)\n"
            "  • acetylene_energy  → dissolved_energy:        0.35 → 0.60\n"
            "    (organics dissolving in warm water; Neish 2009)\n"
            "  • methane_cycle     → water_ammonia_cycle:     0.40 → 0.60\n"
            "    (Lorenz 1997 predicts water-ammonia meteorology)\n"
            "  • subsurface_ocean  → global_ocean_habitability: 0.03 → 0.80\n"
            "    (ocean IS the surface; Lorenz 1997; Affholder 2025)\n"
            "  • surface_atm_interaction: retained, 0.35 → 0.40\n"
            "  • topographic_complexity:  retained, 0.25 → 0.40\n"
            "  • geomorphologic_diversity: retained, 0.30 → 0.40\n"
            "  ⚠️  ASSUMPTION: All future features derived from CURRENT data.\n"
            "  ⚠️  ASSUMPTION: Red giant window ~6 Gya; 'several hundred Myr'.\n"
            "  ⚠️  No Cassini dataset directly constrains future state."
        )

    return ""
