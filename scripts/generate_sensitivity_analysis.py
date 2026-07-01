#!/usr/bin/env python3
"""
scripts/generate_sensitivity_analysis.py
=========================================
Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Generates Figure: Sensitivity of P(H) to prior specification parameters.

Produces a 3x3 panel figure showing P(H) at three representative sites
under variations in kappa (prior concentration), lambda (likelihood
sharpness), and w_1 (liquid hydrocarbon weight).

Usage
-----
    python scripts/generate_sensitivity_analysis.py

Output
------
    outputs/diagnostics/sensitivity_analysis.pdf
    outputs/diagnostics/sensitivity_analysis.png

Fully reproducible: no pipeline outputs required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.temporal_config import TemporalMode, get_prior_set
from configs.pipeline_config import BayesianPriorConfig

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_bpc = BayesianPriorConfig()
KAPPA_NOM  = _bpc.beta_concentration_default
LAMBDA_NOM = _bpc.likelihood_sharpness

_priors = get_prior_set(TemporalMode.PRESENT)
_feat_keys = ["liquid_hc", "organic", "acetylene", "methane", "sai", "topo", "geodiv", "ocean"]
BASELINE_WEIGHTS = dict(zip(_feat_keys, _priors.weights))
PRIOR_MEANS = dict(zip(_feat_keys, _priors.prior_means))

SITES = {
    "Towada Lacus": {
        "features": {
            "liquid_hc": 0.894, "organic": 0.915, "acetylene": 0.892,
            "methane":   0.689, "sai":     0.176, "topo":      0.834,
            "geodiv":    0.564, "ocean":   0.030,
        },
        "colour": "#2196F3",
    },
    "Belet Dunes": {
        "features": {
            "liquid_hc": 0.020, "organic": 0.820, "acetylene": 0.450,
            "methane":   0.090, "sai":     0.090, "topo":      0.550,
            "geodiv":    0.090, "ocean":   0.030,
        },
        "colour": "#8B5000",
    },
    # TODO(validation): this hardcoded Selk feature vector is inconsistent with the
    # canonical Selk values in tab:selk_features / RANKINGS.csv -- e.g. acetylene
    # 0.454 vs 0.229, liquid_hc 0.050 vs 0.010, organic 0.168 vs 0.232, ocean 0.070
    # vs 0.030 -- giving a sensitivity baseline P(H) ~= 0.239 rather than the
    # ranking's 0.210.  Reconcile with the canonical (45-km median) values, then
    # regenerate sensitivity_analysis.png and update the Selk deltas (-0.027 / +0.029)
    # quoted in discussion.tex (sec:sensitivity).
    "Selk Crater": {
        "features": {
            "liquid_hc": 0.050, "organic": 0.168, "acetylene": 0.454,
            "methane":   0.026, "sai":     0.010, "topo":      0.055,
            "geodiv":    0.659, "ocean":   0.070,
        },
        "colour": "#E91E63",
    },
}


def ph(features: dict, weights: dict, kappa: float, lam: float) -> float:
    mu0    = sum(weights[k] * PRIOR_MEANS[k] for k in weights)
    alpha0 = mu0 * kappa
    beta0  = (1.0 - mu0) * kappa
    w_sum  = sum(weights[k] * features[k] for k in weights)
    return (alpha0 + lam * w_sum) / (kappa + lam)


# -- Parameter variants --------------------------------------------------------
# NOTE: labels use "\n" (real newline) not r"\n" (literal backslash-n)
kappa_vals = [2.5,   5.0,   10.0]
kappa_lbls = [
    "$\\kappa=2.5$",
    "$\\kappa=5$ (base)",     # no \n - keep short to avoid overlap
    "$\\kappa=10$",
]

lambda_vals = [3.0,  6.0,   9.0]
lambda_lbls = [
    "$\\lambda=3$",
    "$\\lambda=6$ (base)",
    "$\\lambda=9$",
]

w1_vals = [0.15, 0.20, 0.25]
w1_lbls = [
    "$w_1=0.15$",
    "$w_1=0.20$",
    "$w_1=0.25$\n(base)",
]


def make_weights_w1(w1_new: float) -> dict:
    delta = BASELINE_WEIGHTS["liquid_hc"] - w1_new
    others = {k: v for k, v in BASELINE_WEIGHTS.items() if k != "liquid_hc"}
    total_others = sum(others.values())
    new_w = {"liquid_hc": w1_new}
    for k, v in others.items():
        new_w[k] = v + delta * (v / total_others)
    return new_w


def make_figure() -> plt.Figure:
    dark_bg  = "white"
    grid_col = "#cccccc"
    txt_col  = "#222222"

    n_sites = len(SITES)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(dark_bg)

    # Title - use plain % not \%
    fig.text(
        0.5, 0.995,
        "Sensitivity of P(H | features) to Prior Specification Parameters",
        ha="center", va="top", color=txt_col, fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, 0.965,
        ("Each panel shows P(H) under one parameter variation for one site.  "
         "Baseline: kappa=5, lambda=6, w_1=0.25.  "
         "Deviations from baseline shown in parentheses."),
        ha="center", va="top", color=txt_col, fontsize=11,
    )

    gs = gridspec.GridSpec(
        3, n_sites, figure=fig,
        top=0.875, bottom=0.1,
        left=0.14, right=0.97,
        hspace=0.65, wspace=0.35,
    )

    vary_configs = [
        ("kappa",  kappa_vals,  kappa_lbls,  "Vary kappa (prior strength)"),
        ("lambda", lambda_vals, lambda_lbls, "Vary lambda (likelihood weight)"),
        ("w1",     w1_vals,     w1_lbls,     "Vary w_1 (liquid HC weight)"),
    ]

    cmap = plt.cm.YlOrRd
    norm = Normalize(vmin=0.10, vmax=0.70)

    for row_i, (vary, vals, lbls, row_title) in enumerate(vary_configs):
        # Row label on left margin
        fig.text(
            0.01, 0.78 - row_i * 0.268,
            row_title,
            ha="left", va="center", color=txt_col, fontsize=11,
            rotation=90, fontweight="bold",
        )

        for col_i, (site_name, site) in enumerate(SITES.items()):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor(dark_bg)
            ax.tick_params(colors=txt_col, labelsize=11)
            ax.spines[:].set_color(grid_col)

            # Compute P(H) for each variant
            ph_vals = []
            for v in vals:
                if vary == "kappa":
                    ph_vals.append(ph(site["features"], BASELINE_WEIGHTS, v, LAMBDA_NOM))
                elif vary == "lambda":
                    ph_vals.append(ph(site["features"], BASELINE_WEIGHTS, KAPPA_NOM, v))
                else:
                    ph_vals.append(ph(site["features"], make_weights_w1(v), KAPPA_NOM, LAMBDA_NOM))

            baseline_val = ph_vals[1]  # middle value is always baseline

            bar_colours = [cmap(norm(v)) for v in ph_vals]
            bars = ax.bar(range(len(vals)), ph_vals, color=bar_colours,
                          edgecolor=grid_col, linewidth=0.8)

            # Annotate bars - value on top, delta below it
            # Fix: use delta:+.3f directly (no extra sign prefix)
            for j, (bar, v) in enumerate(zip(bars, ph_vals)):
                delta = v - baseline_val
                delta_str = f"{delta:+.3f}"    # e.g. "+0.031" or "-0.027"
                ax.text(
                    j, v + 0.006,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    color=txt_col,
                    fontsize=11, fontweight="bold",
                )
                ax.text(
                    j, v - 0.018,
                    f"({delta_str})",
                    ha="center", va="top",
                    color=("#4CAF50" if delta > 0.001 else
                           "#FF5252" if delta < -0.001 else "#888888"),
                    fontsize=11,
                )

            ax.axhline(baseline_val, color="#444444", lw=0.8, ls="--", alpha=0.4)

            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(lbls, color=txt_col, fontsize=11)
            ax.set_ylim(0.05, max(ph_vals) * 1.40)
            ax.set_ylabel("P(H)", color=txt_col, fontsize=11)
            ax.yaxis.label.set_color(txt_col)
            ax.grid(True, color=grid_col, alpha=0.3, axis="y", lw=0.5)

            if row_i == 0:
                ax.set_title(site_name, color=site["colour"],
                             fontsize=11, fontweight="bold", pad=5)

    # Colourbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.2, 0.00, 0.70, 0.015])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("P(H | features)", color=txt_col, fontsize=11)
    cb.ax.tick_params(colors=txt_col, labelsize=11)
    cb.outline.set_edgecolor(grid_col)

    return fig


if __name__ == "__main__":
    print("Titan Habitability Pipeline  Copyright (C) 2025/2026  Chris Meadows")
    print("Generating sensitivity analysis figure...")
    fig = make_figure()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"sensitivity_analysis.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved -> {out}")
    plt.close(fig)
    print("Done.")
