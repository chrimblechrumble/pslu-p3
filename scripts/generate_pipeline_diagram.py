#!/usr/bin/env python3
"""
scripts/generate_pipeline_diagram.py
=====================================
Generate a five-stage pipeline architecture flowchart for the thesis (Figure M1).

Output: outputs/diagnostics/pipeline_flowchart.pdf
Run:    python scripts/generate_pipeline_diagram.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")
fig.patch.set_facecolor("#0d0d1a")
ax.set_facecolor("#0d0d1a")

STAGES = [
    {
        "x": 0.4, "w": 2.2,
        "title": "Stage 1\nAcquisition",
        "body": "SAR mosaic\nVIMS+ISS\nGTDE DEM\nLopes geomorphology\nMiller channels",
        "note": "PDS / USGS Astropedia\nCaltechDATA",
        "color": "#1a3a5c",
        "border": "#4488cc",
    },
    {
        "x": 3.1, "w": 2.4,
        "title": "Stage 2\nPreprocessing",
        "body": "Canonical grid\n4490 m/px\n1802×3603 px\nWest-positive\nequirectangular",
        "note": "titan/preprocessing.py\nCoverage: ~67-99%",
        "color": "#1a3a1a",
        "border": "#44cc44",
    },
    {
        "x": 5.9, "w": 2.5,
        "title": "Stage 3\nFeature Extraction",
        "body": "8 features\n[0,1] normalised\ngeo_only organic\nBackfill cascade\nper feature",
        "note": "titan/features.py\ntemporal_features.py",
        "color": "#3a1a3a",
        "border": "#cc44cc",
    },
    {
        "x": 8.8, "w": 2.5,
        "title": "Stage 4\nBayesian Inference",
        "body": "Beta conjugate\nprior κ=5, λ=6\nμ₀=0.331\nMedian-split labels\nsklearn RF backend",
        "color": "#3a1a1a",
        "border": "#cc4444",
        "note": "titan/bayesian/\ninference.py",
    },
    {
        "x": 11.7, "w": 2.1,
        "title": "Stage 5\nOutputs",
        "body": "GeoTIFF\nNetCDF stack\n72-epoch animation\n(modelled +\nfull_inference)",
        "note": "outputs/\ntemporal_maps/",
        "color": "#2a2a1a",
        "border": "#ccaa44",
    },
]

for s in STAGES:
    box = FancyBboxPatch(
        (s["x"], 0.5), s["w"], 3.8,
        boxstyle="round,pad=0.08",
        facecolor=s["color"], edgecolor=s["border"], linewidth=2.0,
    )
    ax.add_patch(box)
    ax.text(s["x"] + s["w"] / 2, 4.05, s["title"],
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", fontfamily="monospace")
    ax.text(s["x"] + s["w"] / 2, 2.55, s["body"],
            ha="center", va="center", fontsize=7.5, color="#ccddff",
            fontfamily="monospace", linespacing=1.5)
    ax.text(s["x"] + s["w"] / 2, 0.75, s["note"],
            ha="center", va="center", fontsize=6.5, color="#aaaaaa",
            fontfamily="monospace", style="italic", linespacing=1.4)

# Arrows
for i in range(len(STAGES) - 1):
    x_start = STAGES[i]["x"] + STAGES[i]["w"] + 0.02
    x_end   = STAGES[i + 1]["x"] - 0.02
    ax.annotate("", xy=(x_end, 2.5), xytext=(x_start, 2.5),
                arrowprops=dict(arrowstyle="->", color="#aaaacc", lw=2.0))

ax.set_title(
    "Titan Habitability Pipeline — Architecture (v5.0)",
    color="white", fontsize=12, pad=10,
)
ax.text(7, 0.1,
        "run_pipeline.py  ·  generate_temporal_maps.py  ·  analyse_location_habitability.py",
        ha="center", va="center", fontsize=7, color="#888899", style="italic")

plt.tight_layout()
out = OUT_DIR / "pipeline_flowchart.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out}")
