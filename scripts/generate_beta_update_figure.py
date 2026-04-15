#!/usr/bin/env python3
"""
scripts/generate_beta_update_figure.py
=======================================
Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk

Generates Figure: Beta prior-to-posterior update diagram.

Shows the Beta distribution updating as each of the eight Cassini
observational features is incorporated for three representative sites.

Usage:   python scripts/generate_beta_update_figure.py
Output:  outputs/diagnostics/beta_update_figure.pdf / .png
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import beta as beta_dist

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KAPPA  = 5.0
LAMBDA = 6.0

WEIGHTS = {
    "liquid_hc":  0.25, "organic":   0.20, "acetylene": 0.20,
    "methane":    0.15, "sai":       0.08, "topo":      0.06,
    "geodiv":     0.04, "ocean":     0.02,
}
PRIOR_MEANS = {
    "liquid_hc":  0.020, "organic":   0.700, "acetylene": 0.350,
    "methane":    0.400, "sai":       0.350, "topo":      0.250,
    "geodiv":     0.300, "ocean":     0.030,
}
FEATURE_LABELS = {
    "liquid_hc":  "$f_1$ liquid HC",
    "organic":    "$f_2$ organic",
    "acetylene":  "$f_3$ acetylene",
    "methane":    "$f_4$ methane",
    "sai":        "$f_5$ surf-atm",
    "topo":       "$f_6$ topo",
    "geodiv":     "$f_7$ geodiv",
    "ocean":      "$f_8$ ocean",
}

SITES = {
    "Kraken S Shore": {
        "colour": "#1565C0",
        "features": {
            "liquid_hc": 1.000, "organic":   0.050, "acetylene": 0.200,
            "methane":   0.700, "sai":       0.650, "topo":      0.600,
            "geodiv":    0.760, "ocean":     0.040,
        },
    },
    "Belet Dunes": {
        "colour": "#8B5000",
        "features": {
            "liquid_hc": 0.020, "organic":   0.820, "acetylene": 0.450,
            "methane":   0.090, "sai":       0.090, "topo":      0.550,
            "geodiv":    0.090, "ocean":     0.030,
        },
    },
    "Selk Crater": {
        "colour": "#E91E63",
        "features": {
            "liquid_hc": 0.050, "organic":   0.215, "acetylene": 0.379,
            "methane":   0.025, "sai":       0.010, "topo":      0.054,
            "geodiv":    0.629, "ocean":     0.033,
        },
    },
}


def posterior_params(features: dict, n_features: int) -> tuple[float, float]:
    mu0   = sum(WEIGHTS[k] * PRIOR_MEANS[k] for k in WEIGHTS)
    alpha = mu0 * KAPPA
    beta  = (1.0 - mu0) * KAPPA
    for key in list(WEIGHTS.keys())[:n_features]:
        f = features[key]; w = WEIGHTS[key]
        alpha += LAMBDA * w * f
        beta  += LAMBDA * w * (1.0 - f)
    return alpha, beta


def posterior_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def make_figure() -> plt.Figure:
    n_sites   = len(SITES)
    n_updates = 9
    x = np.linspace(0.001, 0.999, 800)

    dark_bg  = "white"
    grid_col = "#cccccc"
    txt_col  = "#222222"

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(dark_bg)

    gs_top = gridspec.GridSpec(
        1, n_sites, figure=fig,
        top=0.885, bottom=0.38,
        left=0.06, right=0.97, wspace=0.32,
    )
    gs_bot = gridspec.GridSpec(
        1, 1, figure=fig,
        top=0.30, bottom=0.07,
        left=0.06, right=0.97,
    )

    # Plain text - no LaTeX \% needed
    fig.text(
        0.5, 0.965,
        "Bayesian Beta Prior-to-Posterior Update by Feature",
        ha="center", va="top", color=txt_col, fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.930,
        ("Curves show Beta(alpha, beta) after adding each successive feature.  "
         "Shaded region = 95% HDI of final posterior.  "
         "Note: probability density can exceed 1.0 for concentrated distributions."),
        ha="center", va="top", color=txt_col, fontsize=9,
    )

    # Compute global y-limit for top panels
    max_pdf = 0.0
    for site in SITES.values():
        for step in range(n_updates):
            a, b = posterior_params(site["features"], step)
            max_pdf = max(max_pdf, beta_dist.pdf(x, a, b).max())
    ylim_top = min(max_pdf * 1.18, 12.0)   # cap at 12 for extreme cases

    for col, (site_name, site) in enumerate(SITES.items()):
        ax = fig.add_subplot(gs_top[0, col])
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors=txt_col, labelsize=8)
        ax.spines[:].set_color(grid_col)
        ax.set_title(site_name, color=site["colour"], fontsize=11,
                     fontweight="bold", pad=6)
        ax.set_xlabel("P(H | features)", color=txt_col, fontsize=9)
        ax.set_ylabel("Probability density", color=txt_col, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ylim_top)       # consistent y-axis across all panels

        n_features = list(WEIGHTS.keys())
        colours    = plt.cm.plasma(np.linspace(0.15, 0.85, n_updates))

        for step in range(n_updates):
            a_p, b_p = posterior_params(site["features"], step)
            pdf  = beta_dist.pdf(x, a_p, b_p)
            lw   = 2.5 if step in (0, n_updates - 1) else 1.0
            ls   = "--" if step == 0 else "-"
            lbl  = "Prior" if step == 0 else FEATURE_LABELS[n_features[step - 1]]
            alpha_line = 0.45 + 0.55 * (step / (n_updates - 1))
            ax.plot(x, pdf, color=colours[step], lw=lw, ls=ls,
                    alpha=alpha_line, label=lbl)

        # Shade final HDI
        a_f, b_f = posterior_params(site["features"], 8)
        lo = beta_dist.ppf(0.025, a_f, b_f)
        hi = beta_dist.ppf(0.975, a_f, b_f)
        ax.fill_between(
            x, beta_dist.pdf(x, a_f, b_f),
            where=(x >= lo) & (x <= hi),
            color=site["colour"], alpha=0.18,
        )
        p_mean = posterior_mean(a_f, b_f)
        ax.axvline(p_mean, color=site["colour"], lw=2.0, ls="-",
                   label=f"P(H) = {p_mean:.3f}")
        ax.axvline(0.331, color="#888888", lw=1.0, ls=":", alpha=0.7,
                   label="Prior mean")

        # Legend: prior + final P(H) + prior mean line + last 3 features
        handles, labels_l = ax.get_legend_handles_labels()
        # Indices: 0=Prior, 1-8=features, 9=P(H) line, 10=prior mean
        # Show: Prior, last 3 features, P(H), prior mean
        keep = [0, 6, 7, 8, 9, 10]
        keep = [i for i in keep if i < len(handles)]
        ax.legend(
            [handles[i] for i in keep], [labels_l[i] for i in keep],
            fontsize=6.8, framealpha=0.2, facecolor=dark_bg,
            edgecolor=grid_col, labelcolor=txt_col, loc="upper left",
        )
        ax.grid(True, color=grid_col, alpha=0.4, lw=0.5)

    # ── Bottom panel: HDI comparison ──────────────────────────────────────────
    ax_bot = fig.add_subplot(gs_bot[0, 0])
    ax_bot.set_facecolor(dark_bg)
    ax_bot.tick_params(colors=txt_col, labelsize=9)
    ax_bot.spines[:].set_color(grid_col)

    all_sites = {
        "Kraken S":   {"f": SITES["Kraken S Shore"]["features"], "c": "#1565C0"},
        "Ligeia E":   {"f": {"liquid_hc":1.0,"organic":0.05,"acetylene":0.20,
                             "methane":0.70,"sai":0.62,"topo":0.55,
                             "geodiv":0.76,"ocean":0.04}, "c": "#1976D2"},
        "Punga":      {"f": {"liquid_hc":0.90,"organic":0.06,"acetylene":0.18,
                             "methane":0.65,"sai":0.55,"topo":0.45,
                             "geodiv":0.65,"ocean":0.04}, "c": "#1E88E5"},
        "Ontario":    {"f": {"liquid_hc":0.85,"organic":0.08,"acetylene":0.22,
                             "methane":0.45,"sai":0.48,"topo":0.42,
                             "geodiv":0.60,"ocean":0.04}, "c": "#42A5F5"},
        "Belet":      {"f": SITES["Belet Dunes"]["features"], "c": "#8B5000"},
        "Huygens":    {"f": {"liquid_hc":0.02,"organic":0.54,"acetylene":0.38,
                             "methane":0.09,"sai":0.08,"topo":0.14,
                             "geodiv":0.20,"ocean":0.03}, "c": "#966000"},
        "Selk":       {"f": SITES["Selk Crater"]["features"], "c": "#E91E63"},
        "Xanadu":     {"f": {"liquid_hc":0.00,"organic":0.25,"acetylene":0.20,
                             "methane":0.25,"sai":0.15,"topo":0.40,
                             "geodiv":0.15,"ocean":0.03}, "c": "#9C27B0"},
    }

    names, means, los, his = [], [], [], []
    for name, sdata in all_sites.items():
        a_f, b_f = posterior_params(sdata["f"], 8)
        m  = posterior_mean(a_f, b_f)
        lo = beta_dist.ppf(0.025, a_f, b_f)
        hi = beta_dist.ppf(0.975, a_f, b_f)
        names.append(name); means.append(m)
        los.append(m - lo); his.append(hi - m)

    y_pos = np.arange(len(names))
    cols  = [all_sites[n]["c"] for n in names]
    for i, (y, m, lo_e, hi_e, col) in enumerate(zip(y_pos, means, los, his, cols)):
        ax_bot.barh(y, lo_e + hi_e, left=m - lo_e, height=0.55,
                    color=col, alpha=0.25, zorder=2)
        ax_bot.plot([m - lo_e, m + hi_e], [y, y], color=col, lw=1.5, zorder=3)
        ax_bot.scatter([m], [y], color=col, s=55, zorder=4)
        ax_bot.text(m + hi_e + 0.01, y, f"{m:.3f}", va="center",
                    color=col, fontsize=8.5, fontweight="bold")

    ax_bot.set_yticks(y_pos)
    ax_bot.set_yticklabels(names, color=txt_col, fontsize=9)
    # Plain % - no LaTeX
    ax_bot.set_xlabel("P(H | features) with 95% HDI", color=txt_col, fontsize=9)
    ax_bot.set_title("95% Highest Density Intervals at Key Present-Epoch Sites",
                     color=txt_col, fontsize=10, pad=4)
    ax_bot.axvline(0.331, color="#888888", lw=1.2, ls=":", alpha=0.7,
                   label="Prior mean (0.331)")
    ax_bot.set_xlim(0.05, 0.82)
    ax_bot.grid(True, color=grid_col, alpha=0.4, lw=0.5, axis="x")
    ax_bot.legend(fontsize=8, framealpha=0.2, facecolor=dark_bg,
                  edgecolor=grid_col, labelcolor=txt_col,
                  loc="upper right")          # moved to upper right
    ax_bot.set_facecolor(dark_bg)
    ax_bot.tick_params(colors=txt_col, labelsize=9)
    ax_bot.spines[:].set_color(grid_col)

    return fig


if __name__ == "__main__":
    print("Titan Habitability Pipeline  Copyright (C) 2025/2026  Chris Meadows")
    print("Generating Beta update figure...")
    fig = make_figure()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"beta_update_figure.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved -> {out}")
    plt.close(fig)
    print("Done.")
