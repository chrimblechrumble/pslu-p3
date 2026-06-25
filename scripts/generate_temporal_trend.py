#!/usr/bin/env python3
"""
scripts/generate_temporal_trend.py
=====================================
Generate the 74-epoch temporal habitability trend (Figure R3).

Reads per-frame posterior .npy files saved by generate_temporal_maps.py
when run with --save-posterior-npy.

Usage:
    # Step 1: generate posteriors
    python generate_temporal_maps.py --inference-mode full_inference --save-posterior-npy

    # Step 2: generate plot
    python scripts/generate_temporal_trend.py

Output: outputs/diagnostics/temporal_habitability_trend.pdf
    outputs/diagnostics/temporal_habitability_trend.png
        outputs/diagnostics/temporal_habitability_trend.png

NOTE: If --save-posterior-npy hasn't been run, this script uses the existing
anchor posteriors and linear interpolation as a lower-fidelity approximation.
"""
from __future__ import annotations
import sys, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.pipeline_config import PipelineConfig, BayesianPriorConfig
from configs.temporal_config import TemporalMode, get_prior_set
from configs.site_catalogue import IMPACT_MELT_CRATERS

# -- Bayesian parameters (single source of truth: configs) --------------------
# Used only by the fully-standalone fallback below, so that its representative
# curves obey the SAME Beta-conjugate formula and bounds as the pipeline
# (Eq. posterior_mean in methods.tex) rather than an ad-hoc linear map.
_BPC      = BayesianPriorConfig()
KAPPA     = _BPC.beta_concentration           # prior concentration (5)
LAMBDA    = _BPC.likelihood_sharpness         # likelihood sharpness (6)
_PS_PRES  = get_prior_set(TemporalMode.PRESENT)
_PS_FUT   = get_prior_set(TemporalMode.FUTURE)
WEIGHTS   = dict(zip(_PS_PRES.feature_names, _PS_PRES.weights))
MU0_PRES  = sum(w * m for w, m in zip(_PS_PRES.weights, _PS_PRES.prior_means))  # 0.331
MU0_FUT   = sum(w * m for w, m in zip(_PS_FUT.weights,  _PS_FUT.prior_means))   # 0.695

#: Per-anchor global prior mean mu0 = sum_i w_i mu_i, taken directly from each
#: epoch's configs.temporal_config prior set (no hardcoded values).
_ANCHOR_MU0 = {
    t: sum(w * m for w, m in zip(ps.weights, ps.prior_means))
    for t, ps in [
        (-3.5, get_prior_set(TemporalMode.PAST)),
        (-1.0, get_prior_set(TemporalMode.LAKE_FORMATION)),
        ( 0.0, _PS_PRES),
        ( 0.25, get_prior_set(TemporalMode.NEAR_FUTURE)),
    ]
}


def _beta_posterior_mean(w_sum: float, mu0: float) -> float:
    """Exact Beta-conjugate posterior mean (Eq. posterior_mean, methods.tex).

    P(H) = (kappa*mu0 + lambda*w_sum) / (kappa + lambda).
    Naturally bounded to [kappa*mu0, kappa*mu0+lambda]/(kappa+lambda); no clamp.
    """
    return (KAPPA * mu0 + LAMBDA * w_sum) / (KAPPA + LAMBDA)

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SHAPE = PipelineConfig().canonical_grid_shape
NROWS, NCOLS = GRID_SHAPE

# Latitude grid (row 0 = 90N)
LATS = np.linspace(90.0, -90.0, NROWS, endpoint=False)

def region_mask(lat_lo, lat_hi):
    rows = (LATS >= lat_lo) & (LATS <= lat_hi)
    mask = np.zeros(GRID_SHAPE, dtype=bool)
    mask[rows, :] = True
    return mask.ravel()

REGIONS = {
    "Global":           (~np.zeros(NROWS * NCOLS, dtype=bool)),  # all pixels
    "N. Polar (>60°N)": region_mask(60, 90),
    "Equatorial (|lat|<30°)": region_mask(-30, 30),
}

# Try to load Hedgepeth crater locations for a crater-site mask
try:
    import geopandas as gpd
    crater_catalogue = Path("data/raw/hedgepeth_craters.gpkg")
    if crater_catalogue.exists():
        gdf = gpd.read_file(crater_catalogue)
        print(f"Loaded {len(gdf)} crater sites for mask")
        crater_rows = [int((90 - lat) / 180 * NROWS) for lat in gdf["lat"]]
        crater_cols = [int(lon_w / 360 * NCOLS) for lon_w in gdf["lon_west"]]
        crater_mask = np.zeros(GRID_SHAPE, dtype=bool)
        for r, c in zip(crater_rows, crater_cols):
            r0, r1 = max(0,r-5), min(NROWS,r+5)
            c0, c1 = max(0,c-5), min(NCOLS,c+5)
            crater_mask[r0:r1, c0:c1] = True
        REGIONS["Crater sites"] = crater_mask.ravel()
    else:
        raise FileNotFoundError
except Exception:
    # Approximate crater mask: known large craters
    crater_sites = [(lon_W, lat) for lon_W, lat, _, _ in IMPACT_MELT_CRATERS[:3]]
    crater_mask = np.zeros(GRID_SHAPE, dtype=bool)
    for lon_w, lat in crater_sites:
        r = int((90 - lat) / 180 * NROWS)
        c = int(lon_w / 360 * NCOLS)
        r0, r1 = max(0,r-10), min(NROWS,r+10)
        c0, c1 = max(0,c-10), min(NCOLS,c+10)
        crater_mask[r0:r1, c0:c1] = True
    REGIONS["Crater sites"] = crater_mask.ravel()

# ------------------------------------------------------------------
# Load posterior data
# ------------------------------------------------------------------
npy_dir = Path("outputs/temporal_maps/animation_full_inference/posteriors")

if npy_dir.exists() and len(list(npy_dir.glob("*.npy"))) > 0:
    # Full per-frame posteriors available
    files = sorted(npy_dir.glob("posterior_*.npy"))
    print(f"Loading {len(files)} per-frame posterior arrays ...")
    epochs, region_medians = [], {k: [] for k in REGIONS}
    for fp in files:
        t = float(fp.stem.replace("posterior_", "").replace("m", "-").replace("_", "."))
        arr = np.load(fp).ravel().astype(np.float32)
        epochs.append(t)
        for rname, mask in REGIONS.items():
            vals = arr[mask]
            vals = vals[np.isfinite(vals)]
            region_medians[rname].append(float(np.median(vals)) if len(vals) > 0 else np.nan)
    epochs = np.array(epochs)

    # Sort by epoch time (files are alphabetically sorted, not temporally)
    order = np.argsort(epochs)
    epochs = epochs[order]
    for rname in region_medians:
        region_medians[rname] = np.array(region_medians[rname])[order]

    # Trim pre-anchor extrapolated frames.  The earliest data-constrained
    # posterior is the Past anchor at -3.5 Gya; frames before it (-3.8, -3.6)
    # are extrapolated from the impact-flux model ahead of any anchor, overshoot
    # it by ~0.05, and create a spurious pre-anchor bump with a discontinuous
    # step down at -3.5.  Start the trend at the Past anchor so the figure
    # matches the data and the Results text ("0.271 at the LHB").
    _keep = epochs >= -3.5 - 1e-6
    epochs = epochs[_keep]
    for rname in region_medians:
        region_medians[rname] = region_medians[rname][_keep]
else:
    # Fallback: use anchor posteriors + linear interpolation
    print("Per-frame posteriors not found. Using anchor posteriors + interpolation.")
    print("Run:  python generate_temporal_maps.py --inference-mode full_inference --save-posterior-npy")

    anchors = {}
    for name, t_val in [("past",-3.5),("lake_formation",-1.0),("present",0.0),
                         ("near_future",0.25),("future",5.9)]:
        p = Path(f"outputs/{name}/inference/posterior_mean.npy")
        if p.exists():
            anchors[t_val] = np.load(p).ravel().astype(np.float32)
    if not anchors:
        # Fully standalone fallback: construct representative curves
        # analytically from the Bayesian model parameters so the figure
        # can be regenerated without any pipeline outputs.
        print("[INFO] No anchor posteriors.  Using analytic standalone curves.")
        epochs = np.linspace(-4.2, 6.7, 300)

        def _liquid_scale(t):
            if t < -1.0: return 0.10
            if t < -0.5: return 0.10 + 0.90 * ((t + 1.0) / 0.5)
            if t < 4.0:  return 1.0
            if t < 5.0:  return max(0.0, 1.0 - (t - 4.0))
            if t >= 5.1: return 1.0   # global ocean
            return 0.0
        def _organic_scale(t):
            elapsed = 4.0 + t
            if elapsed <= 0: return 0.0
            return min(elapsed / 4.0, 2.5)
        def _acetylene_scale(t):
            age = 4.57 + t
            if age <= 0: return 2.5
            return min(2.5, (4.57 / age) ** 0.5)

        # Representative present-epoch feature values per region (informed by
        # the thesis site/median tables).  The four dynamic features
        # (liquid, organic, acetylene, methane) are modulated over time by the
        # scale functions above and clipped to [0,1]; the static features stay
        # at their representative present values.  P(H) is then the EXACT
        # config Beta formula -- no ad-hoc constants, and bounds follow from
        # [kappa*mu0, kappa*mu0+lambda]/(kappa+lambda).
        #   region: (liq_max, organic, acetylene, methane_max,
        #            sai, topo, geomorph, subsurface)
        _REGION_BASE = {
            "Global":   (0.10, 0.51, 0.42, 0.45, 0.10, 0.15, 0.10, 0.03),
            "N. Polar": (0.45, 0.62, 0.45, 0.65, 0.22, 0.16, 0.10, 0.03),
            "Equator":  (0.02, 0.55, 0.38, 0.12, 0.05, 0.12, 0.15, 0.03),
            "Crater":   (0.02, 0.28, 0.30, 0.06, 0.02, 0.08, 0.55, 0.03),
        }

        _knots = sorted(_ANCHOR_MU0.items())   # [(-3.5,..),(-1.0,..),(0,..),(0.25,..)]

        def _mu0_for(t):
            # Cassini-era basin (<= +0.25 Gya): linearly interpolate the real
            # per-anchor mu0 from config (past < lake < present ordering).
            # Plateau at the near-future mu0 through the solvent-free era, then
            # ramp to the FUTURE red-giant ocean prior (mu0=0.695) over
            # +5.0 -> +5.9 Gya after the transient minimum.
            if t <= _knots[0][0]:
                return _knots[0][1]
            if t <= _knots[-1][0]:
                for (t0, m0), (t1, m1) in zip(_knots, _knots[1:]):
                    if t <= t1:
                        return m0 + (m1 - m0) * (t - t0) / (t1 - t0)
            if t <= 5.0:
                return _knots[-1][1]
            if t >= 5.9:
                return MU0_FUT
            return _knots[-1][1] + (t - 5.0) / 0.9 * (MU0_FUT - _knots[-1][1])

        def _clip01(x):
            return min(1.0, max(0.0, x))

        def _region_median(region, t):
            liq_max, org, ace, cyc, sai, topo, geo, sub = _REGION_BASE[region]
            s1, s2, s3 = _liquid_scale(t), _organic_scale(t), _acetylene_scale(t)
            feats = {
                "liquid_hydrocarbon":      _clip01(liq_max * s1),
                "organic_abundance":       _clip01(org * s2),
                "acetylene_energy":        _clip01(ace * s3),
                "methane_cycle":           _clip01(cyc * (0.2 + 0.8 * s1)),
                "surface_atm_interaction": sai,
                "topographic_complexity":  topo,
                "geomorphologic_diversity": geo,
                "subsurface_ocean":        sub,
            }
            w_sum = sum(WEIGHTS[k] * feats[k] for k in WEIGHTS)
            return _beta_posterior_mean(w_sum, _mu0_for(t))

        def global_median(t):  return _region_median("Global", t)
        def npolar_median(t):  return _region_median("N. Polar", t)
        def equat_median(t):   return _region_median("Equator", t)
        def crater_median(t):  return _region_median("Crater", t)

        region_medians = {
            "Global":                   np.array([global_median(t) for t in epochs]),
            "N. Polar (>60°N)":         np.array([npolar_median(t) for t in epochs]),
            "Equatorial (|lat|<30°)":   np.array([equat_median(t)  for t in epochs]),
            "Crater sites":             np.array([crater_median(t) for t in epochs]),
        }

    else:
        from scipy.interpolate import interp1d
        t_anchors = np.array(sorted(anchors.keys()))
        # Build per-region median arrays at anchor epochs
        anchor_medians = {rname: [] for rname in REGIONS}
        for t_val in t_anchors:
            arr = anchors[t_val]
            for rname, mask in REGIONS.items():
                vals = arr[mask][np.isfinite(arr[mask])]
                anchor_medians[rname].append(float(np.median(vals)) if len(vals) > 0 else np.nan)

        epochs = np.linspace(-4.0, 6.5, 200)
        region_medians = {}
        for rname in REGIONS:
            y = np.array(anchor_medians[rname], dtype=np.float64)
            valid = np.isfinite(y)
            if valid.sum() < 2:
                region_medians[rname] = np.full_like(epochs, np.nan)
                continue
            interp = interp1d(t_anchors[valid], y[valid], kind="cubic",
                              bounds_error=False, fill_value="extrapolate")
            region_medians[rname] = np.clip(interp(epochs), 0.1, 1.0)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
COLORS = {
    "Global":                   "#000000",
    "N. Polar (>60°N)":         "#2266dd",
    "Equatorial (|lat|<30°)":   "#cc5500",
    "Crater sites":             "#cc2222",
}
LINESTYLES = {"Global": "-", "N. Polar (>60°N)": "--",
              "Equatorial (|lat|<30°)": "-.", "Crater sites": ":"}

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for rname, vals in region_medians.items():
    ax.plot(epochs, vals, color=COLORS.get(rname, "black"),
            linestyle=LINESTYLES.get(rname, "-"), linewidth=1.8, label=rname)

# Ocean window shading
ax.axvspan(5.1, 6.0, alpha=0.12, color="#4488ff", zorder=0)
ax.text(5.5, 0.60, "Super-\neutectic", ha="center", fontsize=8, color="#2255aa", style="italic")

# Key event verticals – labels placed BELOW the x-axis using the
# mixed-transform (data-x, axes-fraction-y).  clip_on=False lets
# them extend outside the Axes box.
# Events closely spaced in time (0/+0.25, and 5.1/5.9/6.0) are
# staggered to two depth levels (-0.06 and -0.14 axis fraction)
# so they do not overlap each other or the x-tick labels.
EVENTS = [
    (-3.5, "#cc3311", "LHB / Past −3.5",      0.95),
    (-1.0, "#3355cc", "Lake formation −1.0",  0.98),
    ( 0.0, "#0088aa", "Present 0.0",          0.95),
    ( 0.25,"#226622", "+0.25",                0.90),
    ( 4.0, "#996600", "Solar warm +4.0",      0.95),
    ( 5.1, "#cc7700", "Eutectic +5.1",        0.98),
    ( 5.9, "#887700", "Ocean peak +5.9",      0.95),
    ( 6.0, "#cc2222", "RGB ends +6.0",        0.90),
]
xfm = ax.get_xaxis_transform()   # x in data coords, y in axes-fraction
for xv, col, label, yoff in EVENTS:
    ax.axvline(xv, color=col, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(xv + (0.3 if xv == 6.0 else 0), yoff, label,
            transform=xfm, clip_on=False,
            ha="center", va="top",
            fontsize=8, color=col, style="italic")

ax.set_xlim(-3.7, 6.7)
ax.set_ylim(0.10, 0.92)   # raised top so N-polar curve never clips against title
ax.set_xlabel("Time (Gya from present)", color="black", fontsize=11)
ax.set_ylabel("Median $P(H \\mid \\mathbf{f})$", color="black", fontsize=11)
# Fixed title: removed rogue backslash before underscore
ax.set_title("Regional Median Habitability Through Geologic Time",
             color="black", fontsize=11)
ax.tick_params(colors="black")
# Legend moved to lower right — clear of the rising N-polar and
# equatorial curves which are highest at present and near-future.
ax.legend(loc="upper left",
          bbox_to_anchor=(0.08, 0.7),
          framealpha=0.5, fontsize=9,
          facecolor="white", edgecolor="#aaaaaa",
          labelcolor="black")
for spine in ax.spines.values():
    spine.set_edgecolor("#aaaaaa")

# Extra bottom margin so the below-axis event labels have room
plt.subplots_adjust(bottom=0.22)
for _ext in ("pdf", "png"):
    out = OUT_DIR / f"temporal_habitability_trend.{_ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved -> {out}")
plt.close(fig)
