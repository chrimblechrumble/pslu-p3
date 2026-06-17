#!/usr/bin/env python3
"""
scripts/generate_hdi_comparison.py
====================================
Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk

Generates Figure: 95% CI comparison across representative Titan sites.

All feature values are read at runtime from the canonical TIF files in
outputs/present/features/tifs/ using the same pixel-sampling method as
verify_thesis_values.py and generate_rankings.py.  No feature values are
hardcoded — running this script after run_pipeline.py guarantees that the
figure is consistent with all other pipeline outputs.

Usage:   python scripts/generate_hdi_comparison.py
Output:  outputs/diagnostics/fig_hdi_comparison.pdf / .png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.temporal_config import TemporalMode, get_prior_set
from configs.pipeline_config import BayesianPriorConfig
from configs.site_catalogue import get_coords as _get_coords

OUT_DIR  = Path("outputs/diagnostics")
TIF_DIR  = Path("outputs/present/features/tifs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_bpc = BayesianPriorConfig()
KAPPA  = _bpc.beta_concentration_default
LAMBDA = _bpc.likelihood_sharpness

_priors = get_prior_set(TemporalMode.PRESENT)

# Feature names must match TIF filenames exactly (without .tif)
FEATURE_NAMES = list(_priors.feature_names)

# Short keys used internally (same order as FEATURE_NAMES)
FEAT_KEYS = [
    "liquid_hc", "organic", "acetylene",
    "methane",   "sai",     "topo",
    "geodiv",    "ocean",
]

_w = list(_priors.weights)
_m = list(_priors.prior_means)
WEIGHTS = dict(zip(FEAT_KEYS, _w))
PRIOR_MEANS = dict(zip(FEAT_KEYS, _m))

# Sites: (display_name, site_type, lon_W_deg, lat_deg)
# site_type: "lake" | "land" | "lander"
# Coordinates from configs.site_catalogue.
def _sdef(catalogue_name, display_name, site_type):
    lon_W, lat = _get_coords(catalogue_name)
    return (display_name, site_type, lon_W, lat)

SITES_DEF = [
    _sdef("Towada",     "Towada Lacus",   "lake"),
    _sdef("Muggel",     "Müggel Lacus",   "lake"),
    _sdef("Koitere",    "Koitere Lacus",  "lake"),
    _sdef("Ontario",    "Ontario Lacus",  "lake"),
    _sdef("Belet",      "Belet Dunes",    "land"),
    _sdef("Hotei",      "Hotei Regio",    "land"),
    _sdef("Huygens",    "Huygens Site",   "lander"),
    _sdef("Selk",       "Selk Crater",    "lander"),
    _sdef("Menrva",     "Menrva Crater",  "land"),
    _sdef("Xanadu",     "Xanadu Centre",  "land"),
    _sdef("Mithrim",    "Mithrim Montes", "land"),
]

TYPE_COLOURS = {"lake": "#0075a3", "land": "#8B5000", "lander": "#EC407A"}
TYPE_LABELS  = {"lake": "Lake/sea shore", "land": "Land site",
                "lander": "Mission lander"}


def sample_tif(arr: np.ndarray, lon_W: float, lat: float,
               radius: int = 2) -> float:
    """
    Sample a TIF array at (lon_W, lat) using a (2r+1)x(2r+1) neighbourhood mean.
    Grid: equirectangular, full globe.
    """
    nrows, ncols = arr.shape
    col = int(round(lon_W / 360.0 * ncols)) % ncols
    row = int(round((90.0 - lat) / 180.0 * nrows))
    row = max(0, min(nrows - 1, row))
    r0, r1 = max(0, row - radius), min(nrows, row + radius + 1)
    c0, c1 = max(0, col - radius), min(ncols, col + radius + 1)
    patch = arr[r0:r1, c0:c1].astype(np.float64)
    patch[~np.isfinite(patch)] = np.nan
    v = float(np.nanmean(patch))
    return float(np.clip(v, 0.0, 1.0))


def load_tifs() -> dict[str, np.ndarray]:
    """Load all 8 feature TIFs. Falls back to prior mean array if file missing."""
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required to read feature TIFs.\n"
            "Install with: pip install rasterio"
        )

    arrays: dict[str, np.ndarray] = {}
    for feat_name, feat_key in zip(FEATURE_NAMES, FEAT_KEYS):
        p = TIF_DIR / f"{feat_name}.tif"
        if not p.exists():
            raise FileNotFoundError(
                f"TIF not found: {p}\n"
                f"Run: python run_pipeline.py --temporal-mode present"
            )
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)
            nd  = src.nodata
            if nd is not None:
                arr[arr == nd] = np.nan
        arrays[feat_key] = arr
        print(f"  Loaded {feat_key:<12} <- {p.name}")
    return arrays


def sample_sites(arrays: dict[str, np.ndarray]) -> list[tuple]:
    """
    Sample all feature TIFs at each site coordinate.
    Returns list of (name, site_type, feature_dict).
    """
    results = []
    for name, stype, lon_W, lat in SITES_DEF:
        feats = {}
        for feat_key, arr in arrays.items():
            feats[feat_key] = sample_tif(arr, lon_W, lat)
        results.append((name, stype, feats))
        ws = sum(WEIGHTS[k] * feats[k] for k in WEIGHTS)
        print(f"  {name:<18} lon={lon_W:6.1f} lat={lat:+6.1f}  ws={ws:.3f}")
    return results


def ph_hdi(features: dict, ci: float = 0.95) -> tuple[float, float, float]:
    mu0    = sum(WEIGHTS[k] * PRIOR_MEANS[k] for k in WEIGHTS)
    alpha0 = mu0 * KAPPA
    beta0  = (1.0 - mu0) * KAPPA
    w_sum  = sum(WEIGHTS[k] * features[k] for k in WEIGHTS)
    a      = alpha0 + LAMBDA * w_sum
    b      = beta0  + LAMBDA * (1.0 - w_sum)
    lo = beta_dist.ppf((1.0 - ci) / 2.0, a, b)
    hi = beta_dist.ppf((1.0 + ci) / 2.0, a, b)
    return a / (a + b), lo, hi


def make_figure(sites: list[tuple]) -> plt.Figure:
    dark_bg  = "white"
    grid_col = "#cccccc"
    txt_col  = "#222222"

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)

    results = [(name, stype, *ph_hdi(feats)) for name, stype, feats in sites]
    results.sort(key=lambda r: r[2])   # ascending so highest is at top

    y_pos = np.arange(len(results))

    mu0    = sum(WEIGHTS[k] * PRIOR_MEANS[k] for k in WEIGHTS)
    alpha0 = mu0 * KAPPA
    p_min  = alpha0 / (KAPPA + LAMBDA)
    p_max  = (alpha0 + LAMBDA) / (KAPPA + LAMBDA)

    ax.axvspan(p_min - 0.002, p_min + 0.010, color="#FF1744", alpha=0.20, zorder=1)

    legend_seen: set = set()
    for i, (name, stype, mean, lo, hi) in enumerate(results):
        col = TYPE_COLOURS[stype]
        y   = y_pos[i]
        lbl = TYPE_LABELS[stype] if stype not in legend_seen else "_nolegend_"
        legend_seen.add(stype)

        ax.barh(y, hi - lo, left=lo, height=0.55,
                color=col, alpha=0.22, zorder=2, label=lbl)
        ax.plot([lo, hi], [y, y], color=col, lw=2.0,
                solid_capstyle="round", zorder=3)
        ax.plot([lo, lo], [y - 0.20, y + 0.20], color=col, lw=1.5, zorder=3)
        ax.plot([hi, hi], [y - 0.20, y + 0.20], color=col, lw=1.5, zorder=3)
        ax.scatter([mean], [y], color=col, s=65, zorder=5,
                   edgecolors="black", linewidths=0.7)
        ax.text(hi + 0.012, y, f"{mean:.3f}", va="center", ha="left",
                color=col, fontsize=8.5, fontweight="bold")

    ax.axvline(mu0, color="#666666", lw=1.4, ls="--", alpha=0.8,
               label=f"Prior mean ({mu0:.3f})")
    ax.axvline(p_max, color="#666666", lw=0.7, ls=":", alpha=0.4)

    ax.text(p_min, len(results) - 0.2,
            f"P_min={p_min:.3f}", color="#FF5252",
            fontsize=7.5, va="bottom", ha="center")
    ax.text(p_max, len(results) - 0.2,
            f"P_max={p_max:.3f}", color="#666666",
            fontsize=7.5, va="bottom", ha="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in results], color=txt_col, fontsize=9.5)
    ax.set_xlabel("P(H | features) with 95% CI", color=txt_col, fontsize=10)
    ax.set_xlim(0.04, 0.82)
    ax.set_ylim(-0.7, len(results) - 0.2)
    ax.tick_params(colors=txt_col, labelsize=9)
    ax.spines[:].set_color(grid_col)
    ax.xaxis.grid(True, color=grid_col, alpha=0.5, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Present-Epoch Bayesian Posterior Means with 95% CI",
                 color=txt_col, fontsize=11, fontweight="bold", pad=10)
    ax.legend(
        fontsize=8.5, framealpha=0.35, facecolor=dark_bg,
        edgecolor=grid_col, labelcolor=txt_col,
        loc="upper right",
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Titan Habitability Pipeline  Copyright (C) 2025/2026  Chris Meadows")
    print("Generating CI comparison figure from TIF pipeline outputs...\n")

    print("Loading feature TIFs...")
    arrays = load_tifs()

    print("\nSampling sites...")
    sites = sample_sites(arrays)

    print("\nComputing posteriors and rendering figure...")
    fig = make_figure(sites)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_hdi_comparison.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved -> {out}")
    plt.close(fig)

    print("\nSite P(H) summary (from TIF-sampled features):")
    results = []
    for name, stype, feats in sites:
        mean, lo, hi = ph_hdi(feats)
        results.append((name, mean, lo, hi))
    for name, mean, lo, hi in sorted(results, key=lambda x: -x[1]):
        print(f"  {name:<20} P(H)={mean:.3f}  95%CI=[{lo:.2f},{hi:.2f}]")
    print("\nDone.")