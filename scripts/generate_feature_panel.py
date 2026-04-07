#!/usr/bin/env python3
"""
scripts/generate_feature_panel.py
===================================
Generate an 8-panel feature map mosaic at the present epoch (Figure M3).

Loads feature TIFs from outputs/present/features/tifs/ and renders
them as a 2×4 grid of equirectangular maps.

Output: outputs/diagnostics/feature_panel_present.pdf
Run:    python scripts/generate_feature_panel.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR = Path("outputs/present/features/tifs")

FEATURES = [
    ("liquid_hydrocarbon",       "plasma",   0.25, 0.020, "SAR lake class + dark anomaly"),
    ("organic_abundance",        "YlOrBr",   0.20, 0.700, "Lopes terrain scores (geo_only)"),
    ("acetylene_energy",         "PuBu",     0.20, 0.350, "SAR backscatter + DEM topo"),
    ("methane_cycle",            "GnBu",     0.15, 0.400, "VIMS density + CIRS gradient"),
    ("surface_atm_interaction",  "RdPu",     0.08, 0.350, "Slope + margin + channel density"),
    ("topographic_complexity",   "copper",   0.06, 0.250, "GTDE roughness (5×5 px window)"),
    ("geomorphologic_diversity", "summer",   0.04, 0.300, "Shannon entropy of terrain classes"),
    ("subsurface_ocean",         "Blues",    0.02, 0.030, "SAR annuli + k₂=0.589 floor"),
]

KEY_LOCS = {
    # name: (lon_W, lat) — annotations on each panel
    "Kraken":  (310, 68),
    "Selk":    (199,  7),
    "Xanadu":  (100,  0),
    "Ontario": (179,-72),
}

def lon_to_col(lon_w, ncols=3603):
    return int(lon_w / 360.0 * ncols)

def lat_to_row(lat, nrows=1802):
    return int((90.0 - lat) / 180.0 * nrows)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.patch.set_facecolor("#0d0d1a")

for ax_flat, (fname, cmap, wi, mu, src_desc) in zip(axes.flat, FEATURES):
    ax_flat.set_facecolor("#0d0d1a")
    tif_path = FEATURE_DIR / f"{fname}.tif"

    if HAS_RASTERIO and tif_path.exists():
        with rasterio.open(tif_path) as ds:
            arr = ds.read(1).astype(np.float32)
            nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    else:
        # Synthetic fallback for testing without real data
        arr = np.random.uniform(0, 1, (1802, 3603)).astype(np.float32)
        arr[arr < 0.1] = np.nan
        if not HAS_RASTERIO:
            print(f"  [WARN] rasterio not available — synthetic data for {fname}")
        elif not tif_path.exists():
            print(f"  [WARN] {tif_path} not found — synthetic data")

    im = ax_flat.imshow(
        arr, cmap=cmap, vmin=0, vmax=1,
        aspect="auto", interpolation="nearest",
        origin="upper",
    )
    # NaN → dark background
    cmap_obj = plt.get_cmap(cmap)
    cmap_obj.set_bad(color="#111122")

    # Annotate key locations
    nrows, ncols = arr.shape
    for loc_name, (lon_w, lat) in KEY_LOCS.items():
        col = lon_to_col(lon_w, ncols)
        row = lat_to_row(lat, nrows)
        ax_flat.plot(col, row, "o", color="#00ffcc", ms=3, markeredgewidth=0.5,
                     markeredgecolor="#003333")

    ax_flat.set_title(
        f"{fname}\n$w={wi:.2f}$   $\\mu={mu:.3f}$",
        color="white", fontsize=7.5, fontfamily="monospace", pad=3,
    )
    ax_flat.text(0.02, 0.02, src_desc, transform=ax_flat.transAxes,
                 fontsize=5.5, color="#aaaacc", va="bottom", style="italic")
    ax_flat.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax_flat.spines.values():
        spine.set_edgecolor("#334455")

    plt.colorbar(im, ax=ax_flat, fraction=0.03, pad=0.02).ax.tick_params(
        labelsize=6, colors="white"
    )

fig.suptitle(
    "All Eight Habitability-Proxy Feature Maps — Present (Cassini) Epoch",
    color="white", fontsize=12, y=1.01,
)
plt.tight_layout()
out = OUT_DIR / "feature_panel_present.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out}")
