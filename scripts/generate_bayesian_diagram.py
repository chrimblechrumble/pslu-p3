#!/usr/bin/env python3
"""
scripts/generate_bayesian_diagram.py
=====================================
Figure: Beta prior and posterior distributions for four representative
Titan surface environments, illustrating the Bayesian conjugate update.

Produces:
    outputs/diagnostics/fig_bayesian_update.pdf
    outputs/diagnostics/fig_bayesian_update.png

Run from the pipeline root:
    python scripts/generate_bayesian_diagram.py

No pipeline data needed — all values are computed analytically from the
hard-coded Bayesian parameters and site feature profiles.

Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist

# ── Bayesian parameters (must match temporal_config.py PRESENT epoch) ──────────
MU0    = 0.331   # prior mean = sum(w_i * mu_i)
KAPPA  = 5.0     # prior concentration
LAMBDA = 6.0     # likelihood sharpness

ALPHA0 = MU0   * KAPPA
BETA0  = (1 - MU0) * KAPPA

# ── Representative sites with their weighted feature sums ──────────────────────
# w_sum = sum(w_i * f_i)  from Table tab:selk_features / site feature profiles
SITES = [
    {
        "name":    "Kraken Mare\nshoreline",
        "w_sum":   0.5242,
        "colour":  "#007a73",   # cyan (lake)
        "marker":  "▲",
        "type":    "lake",
    },
    {
        "name":    "Ligeia Mare\nshoreline",
        "w_sum":   0.5188,
        "colour":  "#006b5a",
        "marker":  "▲",
        "type":    "lake",
    },
    {
        "name":    "Belet\ndune sea",
        "w_sum":   0.3177,
        "colour":  "#8B6200",   # amber (land)
        "marker":  "■",
        "type":    "land",
    },
    {
        "name":    "Selk\ncrater",
        "w_sum":   0.1649,
        "colour":  "#e17055",   # orange-red
        "marker":  "■",
        "type":    "land",
    },
]

def posterior_params(w_sum: float) -> tuple[float, float]:
    ap = ALPHA0 + LAMBDA * w_sum
    bp = BETA0  + LAMBDA * (1.0 - w_sum)
    return ap, bp


def main() -> None:
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#aaaaaa")

    x = np.linspace(0.0, 1.0, 1000)

    # ── Left panel: prior distribution ─────────────────────────────────────────
    ax = axes[0]
    prior_pdf = beta_dist.pdf(x, ALPHA0, BETA0)
    ax.plot(x, prior_pdf, color="#2255aa", lw=2.5, label=rf"Prior  $\mathrm{{Beta}}({ALPHA0:.3f},\,{BETA0:.3f})$")
    ax.fill_between(x, prior_pdf, alpha=0.20, color="#2255aa")
    ax.axvline(MU0, color="#2255aa", ls="--", lw=1.2, alpha=0.7)
    ax.text(MU0 + 0.01, max(prior_pdf) * 0.7,
            rf"$\mu_0 = {MU0}$", color="#2255aa", fontsize=9)
    # Mark P_min and P_max
    p_min = ALPHA0 / (KAPPA + LAMBDA)
    p_max = (ALPHA0 + LAMBDA) / (KAPPA + LAMBDA)
    ax.axvspan(0, p_min, alpha=0.08, color="#ff6b6b", label=rf"$P_\mathrm{{min}}={p_min:.3f}$")
    ax.axvspan(p_max, 1, alpha=0.08, color="#51cf66", label=rf"$P_\mathrm{{max}}={p_max:.3f}$")
    ax.set_xlabel("Posterior mean  $P(H \\mid \\mathbf{f})$", color="black", fontsize=11)
    ax.set_ylabel("Probability density", color="black", fontsize=11)
    ax.set_title("Prior distribution\n(before observing features)", color="black", fontsize=11, pad=8)
    ax.legend(fontsize=9, facecolor="white", labelcolor="black", framealpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, None)

    # ── Right panel: posteriors for all sites ────────────────────────────────────
    ax = axes[1]
    # Draw prior faintly for comparison
    ax.plot(x, prior_pdf, color="#2255aa", lw=1.0, alpha=0.35, ls="--",
            label="Prior (reference)")

    legend_handles = []
    for s in SITES:
        ap, bp = posterior_params(s["w_sum"])
        ph_mean = ap / (ap + bp)
        lo = beta_dist.ppf(0.025, ap, bp)
        hi = beta_dist.ppf(0.975, ap, bp)
        pdf = beta_dist.pdf(x, ap, bp)

        ax.plot(x, pdf, color=s["colour"], lw=2.2)
        ax.fill_between(x, pdf, alpha=0.18, color=s["colour"])
        # Mark posterior mean
        ax.axvline(ph_mean, color=s["colour"], ls=":", lw=1.2, alpha=0.8)
        # Label
        ax.text(ph_mean + 0.01, beta_dist.pdf(ph_mean, ap, bp) * 0.92,
                f"$P(H)={ph_mean:.3f}$",
                color=s["colour"], fontsize=8.5, va="top")
        handle = Line2D([0], [0], color=s["colour"], lw=2.2,
                        label=f"{s['marker']} {s['name'].replace(chr(10), ' ')}  "
                              f"$P(H)={ph_mean:.3f}$\n"
                              f"   95% HDI: [{lo:.2f}, {hi:.2f}]")
        legend_handles.append(handle)

    ax.set_xlabel("Posterior mean  $P(H \\mid \\mathbf{f})$", color="black", fontsize=11)
    ax.set_title("Posterior distributions\n(after observing Cassini features)", color="black", fontsize=11, pad=8)
    ax.legend(handles=legend_handles, fontsize=8.5, facecolor="white",
              labelcolor="black", framealpha=0.8, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, None)

    # ── Shared annotation ──────────────────────────────────────────────────────
    fig.suptitle(
        "Bayesian conjugate update:  "
        r"$H \sim \mathrm{Beta}(\alpha_0,\beta_0)$  "
        r"$\longrightarrow$  "
        r"$H \mid \mathbf{f} \sim \mathrm{Beta}(\alpha_\mathrm{post},\,\beta_\mathrm{post})$",
        color="black", fontsize=12, y=1.01,
    )

    fig.tight_layout(rect=[0, 0, 1, 1.0])

    for ext in ("pdf", "png"):
        p = out_dir / f"fig_bayesian_update.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {p}")

    plt.close(fig)


if __name__ == "__main__":
    main()
