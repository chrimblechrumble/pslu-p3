#!/usr/bin/env python3
"""
scripts/generate_feature_panel.py
===================================
Generate an 8-panel feature map mosaic at the present epoch (Figure M3).

Loads feature TIFs from outputs/present/features/tifs/ and renders
them as a 2×4 grid of equirectangular maps.

Output: outputs/diagnostics/feature_panel_present.pdf
    outputs/diagnostics/feature_panel_present.png
        outputs/diagnostics/feature_panel_present.png
Run:    python scripts/generate_feature_panel.py
"""
from __future__ import annotations
import sys
import zlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.temporal_config import TemporalMode, get_prior_set
from configs.pipeline_config import PipelineConfig

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR = Path("outputs/present/features/tifs")

_priors = get_prior_set(TemporalMode.PRESENT)
_w = dict(zip(_priors.feature_names, _priors.weights))
_m = dict(zip(_priors.feature_names, _priors.prior_means))

# Colourmap, description, and label are display-only; weights and priors from config
FEATURES = [
    ("liquid_hydrocarbon",       "YlOrBr",   _w["liquid_hydrocarbon"],      _m["liquid_hydrocarbon"],      "SAR lake class + dark anomaly",    "$f_1$"),
    ("organic_abundance",        "YlOrBr",   _w["organic_abundance"],       _m["organic_abundance"],       "Blended VIMS + terrain scores",    "$f_2$"),
    ("acetylene_energy",         "PuBu",     _w["acetylene_energy"],        _m["acetylene_energy"],        "SAR backscatter + DEM topo",       "$f_3$"),
    ("methane_cycle",            "GnBu",     _w["methane_cycle"],           _m["methane_cycle"],           "VIMS density + CIRS gradient",     "$f_4$"),
    ("surface_atm_interaction",  "RdPu",     _w["surface_atm_interaction"], _m["surface_atm_interaction"], "Slope + margin + channel density", "$f_5$"),
    ("topographic_complexity",   "copper",   _w["topographic_complexity"],  _m["topographic_complexity"],  "GTDE roughness (5\u00d75 px window)", "$f_6$"),
    ("geomorphologic_diversity", "summer",   _w["geomorphologic_diversity"],_m["geomorphologic_diversity"],"Shannon entropy of terrain classes","$f_7$"),
    ("subsurface_ocean",         "Blues",    _w["subsurface_ocean"],        _m["subsurface_ocean"],        "SAR annuli + k\u2082=0.589 floor", "$f_8$"),
]

KEY_LOCS = {
    # name: (lon_W, lat) — annotations on each panel
    "Towada": (244.2, 71.4),
    "Selk":    (199,  7),
    "Xanadu":  (100,  0),
    "Ontario": (179,-72),
}

_grid_shape = PipelineConfig().canonical_grid_shape

def lon_to_col(lon_w, ncols=_grid_shape[1]):
    return int(lon_w / 360.0 * ncols)

def lat_to_row(lat, nrows=_grid_shape[0]):
    return int((90.0 - lat) / 180.0 * nrows)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.patch.set_facecolor("white")

for ax_flat, (fname, cmap, wi, mu, src_desc, flabel) in zip(axes.flat, FEATURES):
    ax_flat.set_facecolor("white")
    tif_path = FEATURE_DIR / f"{fname}.tif"

    if HAS_RASTERIO and tif_path.exists():
        with rasterio.open(tif_path) as ds:
            arr = ds.read(1).astype(np.float32)
            nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    else:
        # Synthetic fallback for testing without real data.  Seed per-feature
        # (stable crc32 of the name) so the placeholder panels are deterministic
        # and distinct, without touching the global numpy RNG.
        _rng = np.random.default_rng(zlib.crc32(fname.encode()))
        arr = _rng.uniform(0, 1, _grid_shape).astype(np.float32)
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
    cmap_obj.set_bad(color="#eeeeee")

    # Annotate key locations
    nrows, ncols = arr.shape
    # for loc_name, (lon_w, lat) in KEY_LOCS.items():
    #     col = lon_to_col(lon_w, ncols)
    #     row = lat_to_row(lat, nrows)
    #     ax_flat.plot(col, row, "o", color="#007755", ms=3, markeredgewidth=0.5,
    #                  markeredgecolor="#003322")

    ax_flat.set_title(
        f"{flabel}: {fname}\n$w={wi:.2f}$   $\\mu={mu:.3f}$",
        color="black", fontsize=7.5, fontfamily="monospace", pad=3,
    )
    ax_flat.text(0.02, 0.02, src_desc, transform=ax_flat.transAxes,
                 fontsize=5.5, color="#555566", va="bottom", style="italic")
    # Add graticules
    nrows_arr, ncols_arr = arr.shape
    lon_ticks = [int(l / 360 * ncols_arr) for l in range(0, 361, 60)]
    lat_ticks = [int((90 - l) / 180 * nrows_arr) for l in range(-90, 91, 30)]
    ax_flat.set_xticks(lon_ticks)
    ax_flat.set_xticklabels([f"{l}°" for l in range(0, 361, 60)], fontsize=4.5, color="#666666")
    ax_flat.set_yticks(lat_ticks)
    ax_flat.set_yticklabels([f"{l}°" for l in range(-90, 91, 30)], fontsize=4.5, color="#666666")
    ax_flat.tick_params(length=2, width=0.5, colors="#999999")
    for spine in ax_flat.spines.values():
        spine.set_edgecolor("#aaaaaa")

    plt.colorbar(im, ax=ax_flat, fraction=0.03, pad=0.02).ax.tick_params(
        labelsize=6, colors="black"
    )

fig.suptitle(
    "All Eight Habitability-Proxy Feature Maps — Present (Cassini) Epoch",
    color="black", fontsize=12, y=1.01,
)
plt.tight_layout()
for _ext in ("pdf", "png"):
    out = OUT_DIR / f"feature_panel_present.{_ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved -> {out}")
plt.close(fig)
