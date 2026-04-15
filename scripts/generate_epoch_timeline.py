#!/usr/bin/env python3
"""
scripts/generate_epoch_timeline.py
====================================
Generate a Gantt-style chart showing feature scale factors across geologic time
(Figure M4).  Uses FEATURE_SCALE_FUNCS from generate_temporal_maps.py.

Output: outputs/diagnostics/epoch_feature_timeline.pdf
    outputs/diagnostics/epoch_feature_timeline.png
        outputs/diagnostics/epoch_feature_timeline.png
Run:    python scripts/generate_epoch_timeline.py
"""
from __future__ import annotations
import sys, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import scale functions from the animation module
try:
    from generate_temporal_maps import (
        FEATURE_SCALE_FUNCS, titan_temp_K, EUTECTIC_K
    )
    print("Loaded scale functions from generate_temporal_maps.py")
except ImportError as e:
    print(f"[WARN] Could not import generate_temporal_maps: {e}")
    print("Using built-in approximations.")
    EUTECTIC_K = 176.0
    def titan_temp_K(t):
        age_now = 4.57; age = age_now + t
        if age <= 0: return 47.0
        if t <= 5.0:
            return 93.65 * max(0.5, (0.72 + 0.28*(age/age_now)**0.9))**0.25
        ta = t - 5.0
        if ta < 0.1: L = 1.0 + 17.0*ta
        elif ta < 0.5: L = max(2.0, 2700*math.exp(-0.5*((ta-0.4)/0.15)**2))
        elif ta < 1.0: L = max(1.0, 2700*math.exp(-3.0*(ta-0.4)))
        else: L = 0.8
        return 93.65 * L**0.25
    def lhc(t):
        T = titan_temp_K(t)
        if T >= EUTECTIC_K: return 1.0
        if t < -1.0: return 0.10
        elif t < -0.5: return 0.10+0.90*((t+1.0)/0.5)
        elif t < 4.0: return 1.0
        elif t < 5.0: return max(0.0, 1.0-(t-4.0))
        return 0.0
    FEATURE_SCALE_FUNCS = {
        "liquid_hydrocarbon": lhc,
        "organic_abundance": lambda t: min(min((4.0+t)/4.0, 2.5), 2.5)/2.5 if (4.0+t)>0 else 0,
        "acetylene_energy": lambda t: min(1.0, math.sqrt(4.57/max(0.01,4.57+t))) if t<5.0 else max(0,1-(t-5.0)),
        "methane_cycle": lambda t: 0.6 if t<-1 else (0.8 if t<-0.5 else (1.0 if t<4.5 else max(0,1-(t-4.5)/0.5))),
        "surface_atm_interaction": lambda t: 0.40+0.60*min(1.0, lhc(t)),
        "topographic_complexity": lambda t: (1.3 if t<-2 else (1.15 if t<-1 else 1.0))/1.3,
        "geomorphologic_diversity": lambda t: (0.7 if t<-3 else (0.85 if t<-2 else 1.0)),
        "subsurface_ocean": lambda t: min(1.0, (2.5 if t<-2 else (1.8 if t<-1 else (1.3 if t<-0.5 else 1.0)))/2.5),
        "impact_melt_proxy": lambda t: min(1.0, 0.40*math.exp(-0.5*((t+3.8)/0.5)**2) + 0.10*math.exp(-abs(t+3.8)/0.8)),
        "cryovolcanic_flux": lambda t: max(0.0, 1.0 - (t+1.0)/2.5) if t <= 1.5 else 0.0,
    }

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Time axis
t_arr = np.linspace(-4.0, 7.0, 500)

FEATURE_ORDER = [
    "liquid_hydrocarbon", "organic_abundance", "acetylene_energy",
    "methane_cycle", "surface_atm_interaction", "topographic_complexity",
    "geomorphologic_diversity", "subsurface_ocean",
    "impact_melt_proxy", "cryovolcanic_flux",
]

FEATURE_LABELS = {
    "liquid_hydrocarbon":       "liquid_hydrocarbon  (w=0.25)",
    "organic_abundance":        "organic_abundance   (w=0.20)",
    "acetylene_energy":         "acetylene_energy    (w=0.20)",
    "methane_cycle":            "methane_cycle       (w=0.15)",
    "surface_atm_interaction":  "surface_atm_interact (w=0.08)",
    "topographic_complexity":   "topographic_complex (w=0.06)",
    "geomorphologic_diversity": "geomorphol_diversity (w=0.04)",
    "subsurface_ocean":         "subsurface_ocean    (w=0.02)",
    "impact_melt_proxy":        "impact_melt_proxy   [PAST only]",
    "cryovolcanic_flux":        "cryovolcanic_flux   [PAST/LF]",
}

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

cmap = plt.get_cmap("YlOrRd")

for yi, fname in enumerate(FEATURE_ORDER):
    if fname not in FEATURE_SCALE_FUNCS:
        continue
    fn = FEATURE_SCALE_FUNCS[fname]
    vals = []
    for t in t_arr:
        try:
            v = fn(float(t))
        except Exception:
            v = 0.0
        vals.append(np.clip(float(v), 0.0, 1.0))
    vals = np.array(vals)

    # Draw as horizontal colour blocks
    dt = t_arr[1] - t_arr[0]
    for xi, (t, v) in enumerate(zip(t_arr, vals)):
        color = cmap(v)
        rect = mpatches.FancyBboxPatch(
            (t - dt/2, yi - 0.4), dt, 0.8,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="none", alpha=0.95,
        )
        ax.add_patch(rect)

    ax.text(-4.15, yi, FEATURE_LABELS[fname],
            ha="right", va="center", fontsize=8, color="black",
            fontfamily="monospace")

# Key event verticals
EVENTS = [
    (-3.8, "#cc4422", "LHB peak"),
    (-1.0, "#3355cc", "Lake formation"),
    ( 0.0, "#0088bb", "Present"),
    ( 0.25, "#226622", "+250 Myr"),
    ( 5.1, "#cc7700", "Eutectic"),
    ( 5.9, "#997700", "Ocean peak"),
    ( 6.0, "#cc2222", "End RGB"),
]
for xv, col, label in EVENTS:
    ax.axvline(xv, color=col, linewidth=1.2, linestyle="--", alpha=0.8)
    ax.text(xv, len(FEATURE_ORDER) - 0.1, label,
            ha="center", va="bottom", fontsize=6.5, color=col,
            rotation=90, style="italic")

# Ocean window
ax.axvspan(5.1, 6.0, alpha=0.10, color="#4488ff", label="Ocean window")

ax.set_xlim(-4.2, 7.2)
ax.set_ylim(-0.7, len(FEATURE_ORDER) + 0.3)
ax.set_yticks([])
ax.set_xlabel("Time (Gya from present; negative = past)", color="black", fontsize=10)
ax.set_title("Feature Activity and Scale Factor Across Geologic Time",
             color="black", fontsize=11)
ax.tick_params(colors="black")
ax.spines["bottom"].set_color("#aaaaaa")
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)

# Colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.01)
cbar.set_label("Scale factor $s_i(t)$  (0 = inactive, 1 = present-epoch amplitude)",
               color="black", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="black")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

plt.tight_layout()
for _ext in ("pdf", "png"):
    out = OUT_DIR / f"epoch_feature_timeline.{_ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved -> {out}")
plt.close(fig)
