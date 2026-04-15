#!/usr/bin/env python3
"""
scripts/generate_hdi_comparison.py
====================================
Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk

Generates Figure: 95% HDI comparison across representative Titan sites.

Usage:   python scripts/generate_hdi_comparison.py
Output:  outputs/diagnostics/fig_hdi_comparison.pdf / .png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KAPPA  = 5.0
LAMBDA = 6.0

WEIGHTS = {
    "liquid_hc": 0.25, "organic":   0.20, "acetylene": 0.20,
    "methane":   0.15, "sai":       0.08, "topo":      0.06,
    "geodiv":    0.04, "ocean":     0.02,
}
PRIOR_MEANS = {
    "liquid_hc": 0.020, "organic":   0.700, "acetylene": 0.350,
    "methane":   0.400, "sai":       0.350, "topo":      0.250,
    "geodiv":    0.300, "ocean":     0.030,
}

SITES = [
    ("Kraken S Shore",  "lake",   {
        "liquid_hc":1.00,"organic":0.05,"acetylene":0.20,
        "methane":0.70,"sai":0.65,"topo":0.60,"geodiv":0.76,"ocean":0.04}),
    ("Ligeia E Shore",  "lake",   {
        "liquid_hc":1.00,"organic":0.05,"acetylene":0.20,
        "methane":0.70,"sai":0.62,"topo":0.55,"geodiv":0.76,"ocean":0.04}),
    ("Kraken N Shore",  "lake",   {
        "liquid_hc":1.00,"organic":0.05,"acetylene":0.18,
        "methane":0.68,"sai":0.60,"topo":0.52,"geodiv":0.72,"ocean":0.04}),
    ("Punga Shore",     "lake",   {
        "liquid_hc":0.90,"organic":0.06,"acetylene":0.18,
        "methane":0.65,"sai":0.55,"topo":0.45,"geodiv":0.65,"ocean":0.04}),
    ("Ontario Lacus",   "lake",   {
        "liquid_hc":0.85,"organic":0.08,"acetylene":0.22,
        "methane":0.45,"sai":0.48,"topo":0.42,"geodiv":0.60,"ocean":0.04}),
    ("Belet Dunes",     "land",   {
        "liquid_hc":0.02,"organic":0.82,"acetylene":0.45,
        "methane":0.09,"sai":0.09,"topo":0.55,"geodiv":0.09,"ocean":0.03}),
    ("Hotei Regio",     "land",   {
        "liquid_hc":0.03,"organic":0.65,"acetylene":0.42,
        "methane":0.18,"sai":0.22,"topo":0.35,"geodiv":0.58,"ocean":0.10}),
    ("Huygens Site",    "lander", {
        "liquid_hc":0.02,"organic":0.54,"acetylene":0.38,
        "methane":0.09,"sai":0.08,"topo":0.14,"geodiv":0.20,"ocean":0.03}),
    ("Selk Crater",     "lander", {
        "liquid_hc":0.05,"organic":0.215,"acetylene":0.379,
        "methane":0.025,"sai":0.010,"topo":0.054,"geodiv":0.629,"ocean":0.033}),
    ("Menrva Crater",   "land",   {
        "liquid_hc":0.02,"organic":0.30,"acetylene":0.38,
        "methane":0.08,"sai":0.08,"topo":0.18,"geodiv":0.55,"ocean":0.035}),
    ("Xanadu Centre",   "land",   {
        "liquid_hc":0.00,"organic":0.25,"acetylene":0.20,
        "methane":0.25,"sai":0.15,"topo":0.40,"geodiv":0.15,"ocean":0.030}),
    ("Mithrim Montes",  "land",   {
        "liquid_hc":0.00,"organic":0.18,"acetylene":0.20,
        "methane":0.08,"sai":0.12,"topo":0.70,"geodiv":0.22,"ocean":0.030}),
]

TYPE_COLOURS = {"lake": "#0075a3", "land": "#8B5000", "lander": "#EC407A"}
TYPE_LABELS  = {"lake": "Lake/sea shore", "land": "Land site",
                "lander": "Mission lander"}


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


def make_figure() -> plt.Figure:
    dark_bg  = "white"
    grid_col = "#cccccc"
    txt_col  = "#222222"

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)

    results = [(name, stype, *ph_hdi(feats)) for name, stype, feats in SITES]
    results.sort(key=lambda r: r[2], reverse=True)

    y_pos = np.arange(len(results))

    mu0    = sum(WEIGHTS[k] * PRIOR_MEANS[k] for k in WEIGHTS)
    alpha0 = mu0 * KAPPA
    p_min  = alpha0 / (KAPPA + LAMBDA)
    p_max  = (alpha0 + LAMBDA) / (KAPPA + LAMBDA)

    # P_min shaded band
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

    # Annotations for P_min and P_max at TOP of plot (not clashing with data)
    ax.text(p_min, len(results) - 0.2,
            f"P_min={p_min:.3f}", color="#FF5252",
            fontsize=7.5, va="bottom", ha="center")
    ax.text(p_max, len(results) - 0.2,
            f"P_max={p_max:.3f}", color="#666666",
            fontsize=7.5, va="bottom", ha="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in results], color=txt_col, fontsize=9.5)
    # Plain text - no LaTeX \% needed
    ax.set_xlabel("P(H | features) with 95% HDI", color=txt_col, fontsize=10)
    ax.set_xlim(0.04, 0.82)
    ax.set_ylim(-0.7, len(results) - 0.2)
    ax.tick_params(colors=txt_col, labelsize=9)
    ax.spines[:].set_color(grid_col)
    ax.xaxis.grid(True, color=grid_col, alpha=0.5, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Present-Epoch Bayesian Posterior Means with 95% HDI",
                 color=txt_col, fontsize=11, fontweight="bold", pad=10)

    # Legend placed in UPPER RIGHT, away from the data bars
    ax.legend(
        fontsize=8.5, framealpha=0.35, facecolor=dark_bg,
        edgecolor=grid_col, labelcolor=txt_col,
        loc="upper right",
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Titan Habitability Pipeline  Copyright (C) 2025/2026  Chris Meadows")
    print("Generating HDI comparison figure...")
    fig = make_figure()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_hdi_comparison.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved -> {out}")
    plt.close(fig)
    print("Done.")
