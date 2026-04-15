#!/usr/bin/env python3
"""
scripts/generate_comparison_chart.py
======================================
Generate solar system habitability comparison chart (Figure D1).

Output: outputs/diagnostics/solar_system_comparison.pdf
    outputs/diagnostics/solar_system_comparison.png
        outputs/diagnostics/solar_system_comparison.png
Run:    python scripts/generate_comparison_chart.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Qualitative scores (1–5 scale) from published literature consensus.
# liquid: stable liquid availability at surface/subsurface
# organic: organic inventory (prebiotic substrate)
# energy: confirmed energy source for metabolism
# Scores annotated with key references.

BODIES = [
    # (name, liquid, organic, energy, P_H_model, note)
    ("Titan (surface, present)",  4.5, 5.0, 3.5, 0.53, "This work; polar lake margins"),
    ("Titan (future, +5.9 Gya)", 5.0, 5.0, 4.0, 0.69, "This work; global ocean phase"),
    ("Enceladus",                 4.0, 3.0, 4.5, None,  "Waite+2017; active plumes"),
    ("Europa",                    4.0, 2.0, 3.5, None,  "Pappalardo+1999; tidal heating"),
    ("Mars (ancient, >3 Gya)",    3.0, 2.5, 2.5, None,  "Grotzinger+2014; lake sediments"),
    ("Titan (past LHB, -3.5 Gya)",2.5, 2.0, 3.0, 0.41, "This work; impact-melt ponds"),
    ("Ceres (subsurface brines)", 2.0, 1.5, 1.5, None,  "De Sanctis+2016"),
]

CATEGORIES = ["Liquid\navailability", "Organic\ninventory", "Energy\nsource"]
COLORS = ["#1255aa", "#8B5000", "#2d7a00"]

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

n_bodies = len(BODIES)
n_cats   = len(CATEGORIES)
bar_h    = 0.22
gap      = 0.08
group_h  = n_cats * bar_h + gap

y_positions = {}
for bi, (name, *_) in enumerate(BODIES):
    y_base = bi * (group_h + 0.15)
    y_positions[name] = y_base

for bi, (name, liquid, organic, energy, p_h, note) in enumerate(BODIES):
    y_base = bi * (group_h + 0.15)
    scores = [liquid, organic, energy]
    for ci, (score, color) in enumerate(zip(scores, COLORS)):
        y = y_base + ci * (bar_h + 0.01)
        ax.barh(y, score, height=bar_h, color=color, alpha=0.85, edgecolor="none")
        ax.text(score + 0.05, y, f"{score:.1f}", va="center",
                fontsize=7.5, color="black")

    # Body label on left
    label = f"{'★ ' if p_h else ''}{name}"
    ax.text(-0.1, y_base + bar_h, label,
            ha="right", va="center", fontsize=9, color="black", fontweight="bold")
    # Model P(H) annotation
    if p_h is not None:
        ax.text(5.35, y_base + bar_h,
                f"Model $P(H)={p_h:.2f}$",
                ha="left", va="center", fontsize=7.5, color="#8B6200",
                style="italic")
    # Note
    ax.text(5.35, y_base, note, ha="left", va="center",
            fontsize=6.5, color="#555555", style="italic")

ax.set_xlim(-4.5, 7.5)
ax.set_ylim(-0.3, n_bodies * (group_h + 0.15) + 0.2)
ax.set_xlabel("Qualitative score (1 = low, 5 = very high)", color="black", fontsize=10)
ax.set_title("Candidate Habitable Environments in the Solar System",
             color="black", fontsize=11)
ax.axvline(5.0, color="#aaaaaa", linewidth=0.8, linestyle=":")
ax.set_xticks([1, 2, 3, 4, 5])
ax.tick_params(colors="black", left=False, labelleft=False)

# Legend for bar categories
from matplotlib.patches import Patch
legend_handles = [Patch(color=c, label=cat.replace("\n", " "))
                  for c, cat in zip(COLORS, CATEGORIES)]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
          facecolor="white", edgecolor="#aaaaaa", labelcolor="black",
          framealpha=0.7)

ax.text(-4.4, n_bodies * (group_h + 0.15) - 0.1,
        "★ = assessed in this work",
        fontsize=7, color="#8B6200", style="italic")

for spine in ax.spines.values():
    spine.set_edgecolor("#aaaaaa")
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.tight_layout()
for _ext in ("pdf", "png"):
    out = OUT_DIR / f"solar_system_comparison.{_ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved -> {out}")
plt.close(fig)
