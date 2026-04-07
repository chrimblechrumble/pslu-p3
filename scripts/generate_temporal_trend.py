#!/usr/bin/env python3
"""
scripts/generate_temporal_trend.py
=====================================
Generate the 72-epoch temporal habitability trend (Figure R3).

Reads per-frame posterior .npy files saved by generate_temporal_maps.py
when run with --save-posterior-npy.

Usage:
    # Step 1: generate posteriors
    python generate_temporal_maps.py --inference-mode full_inference --save-posterior-npy

    # Step 2: generate plot
    python scripts/generate_temporal_trend.py

Output: outputs/diagnostics/temporal_habitability_trend.pdf

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

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SHAPE = (1802, 3603)
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
        crater_rows = [int((90 - lat) / 180 * NROWS) for lat in gdf.geometry.y]
        crater_cols = [int(lon / 360 * NCOLS) for lon in gdf.geometry.x]
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
    crater_sites = [(87.3, 19.0), (199.0, 7.0), (16.0, 11.3)]  # lon_W, lat
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
        t = float(fp.stem.replace("posterior_", "").replace("_", "."))
        arr = np.load(fp).ravel().astype(np.float32)
        epochs.append(t)
        for rname, mask in REGIONS.items():
            vals = arr[mask]
            vals = vals[np.isfinite(vals)]
            region_medians[rname].append(float(np.median(vals)) if len(vals) > 0 else np.nan)
    epochs = np.array(epochs)
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
        print("[ERROR] No anchor posteriors found. Run the full pipeline first.")
        sys.exit(1)

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
        interp = interp1d(t_anchors, anchor_medians[rname], kind="cubic",
                          bounds_error=False, fill_value="extrapolate")
        region_medians[rname] = np.clip(interp(epochs), 0.1, 1.0)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
COLORS = {
    "Global":                   "#ffffff",
    "N. Polar (>60°N)":         "#4499ff",
    "Equatorial (|lat|<30°)":   "#ff8844",
    "Crater sites":             "#ff4444",
}
LINESTYLES = {"Global": "-", "N. Polar (>60°N)": "--",
              "Equatorial (|lat|<30°)": "-.", "Crater sites": ":"}

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("#0d0d1a")
ax.set_facecolor("#0d0d1a")

for rname, vals in region_medians.items():
    ax.plot(epochs, vals, color=COLORS.get(rname, "white"),
            linestyle=LINESTYLES.get(rname, "-"), linewidth=1.8, label=rname)

# Ocean window shading
ax.axvspan(5.1, 6.0, alpha=0.12, color="#4488ff", zorder=0)
ax.text(5.5, 0.72, "Ocean\nwindow", ha="center", fontsize=8, color="#88bbff", style="italic")

# Key event verticals
EVENTS = [
    (-3.8, "#ff6644", "LHB peak\n−3.8"),
    (-1.0, "#88aaff", "Lake\nformation\n−1.0"),
    ( 0.0, "#66ddff", "Present\n0.0"),
    ( 0.25,"#aaffaa", "+0.25"),
    ( 4.0, "#ffdd44", "Solar\nwarm\n+4.0"),
    ( 5.1, "#ffaa44", "Eutectic\n+5.1"),
    ( 5.9, "#ffcc00", "Ocean\npeak\n+5.9"),
    ( 6.0, "#ff3333", "RGB\nends\n+6.0"),
]
for xv, col, label in EVENTS:
    ax.axvline(xv, color=col, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(xv, 0.12, label, ha="center", va="bottom", fontsize=6,
            color=col, style="italic")

ax.set_xlim(-4.2, 6.7)
ax.set_ylim(0.10, 0.82)
ax.set_xlabel("Time (Gya from present)", color="white", fontsize=11)
ax.set_ylabel("Median $P(H \\mid \\mathbf{f})$", color="white", fontsize=11)
ax.set_title("Regional Habitability Through Geologic Time — Full\_inference Mode",
             color="white", fontsize=11)
ax.tick_params(colors="white")
ax.legend(loc="upper left", framealpha=0.3, fontsize=9,
          facecolor="#111122", edgecolor="#334455",
          labelcolor="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#334455")

plt.tight_layout()
out = OUT_DIR / "temporal_habitability_trend.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out}")
