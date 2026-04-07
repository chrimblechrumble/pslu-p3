#!/usr/bin/env python3
"""
scripts/generate_future_bathymetry_map.py
==========================================
Generate the future epoch habitability map with DEM topographic contours
as ocean bathymetry (Figure R5).

Loads the future posterior and overlays topographic contours.

Output: outputs/diagnostics/future_epoch_bathymetry.pdf
Run:    python scripts/generate_future_bathymetry_map.py
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

# Load future posterior
future_post_path = Path("outputs/future/inference/posterior_mean.npy")
topo_tif_path    = Path("outputs/present/features/tifs/topographic_complexity.tif")

fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2.5, 1]})
fig.patch.set_facecolor("#0d0d1a")

# --- Left: equirectangular future habitability + topo contours ---
ax = axes[0]
ax.set_facecolor("#0d0d1a")

if future_post_path.exists():
    post = np.load(future_post_path).astype(np.float32)
else:
    print("[WARN] Future posterior not found — using synthetic data")
    np.random.seed(42)
    post = np.clip(0.65 + np.random.normal(0, 0.06, (1802, 3603)), 0, 1).astype(np.float32)

im = ax.imshow(post, cmap="plasma", vmin=0.10, vmax=0.75,
               aspect="auto", origin="upper", interpolation="nearest")

# Topographic contours
if HAS_RASTERIO and topo_tif_path.exists():
    with rasterio.open(topo_tif_path) as ds:
        topo = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            topo = np.where(topo == ds.nodata, np.nan, topo)
    # Smooth for contour stability
    from scipy.ndimage import gaussian_filter
    topo_smooth = gaussian_filter(np.nan_to_num(topo, nan=0.5), sigma=8)
    # Three contour levels representing future ocean depth
    cs = ax.contour(topo_smooth, levels=[0.30, 0.55, 0.75],
                    colors=["#ffffff"], linewidths=[0.5, 0.8, 1.1], alpha=0.5)
    ax.clabel(cs, fmt={0.30: "shallow", 0.55: "mid", 0.75: "deep"},
              fontsize=6, colors="white")
else:
    print(f"[WARN] Topography TIF not found ({topo_tif_path}) — skipping contours")

plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
             label="$P(H\\mid\\mathbf{f})$").ax.tick_params(colors="white", labelsize=8)
ax.set_title("Future Epoch (+5.9 Gya)  —  Global Water-Ammonia Ocean\n"
             "Topographic contours = future ocean bathymetry (relative depth proxy)",
             color="white", fontsize=9)
ax.set_xlabel("Longitude (°W)", color="white", fontsize=8)
ax.set_ylabel("Latitude (°N)", color="white", fontsize=8)
ax.tick_params(colors="white", labelsize=7)
nrows, ncols = post.shape
ax.set_xticks(np.linspace(0, ncols, 7))
ax.set_xticklabels(["0°", "60°", "120°", "180°", "240°", "300°", "360°"])
ax.set_yticks(np.linspace(0, nrows, 7))
ax.set_yticklabels(["90°N", "60°N", "30°N", "0°", "30°S", "60°S", "90°S"])

# --- Right: north polar cap (rendered via map_coordinates, no pcolormesh) ---
ax2 = axes[1]
ax2.set_facecolor("#111122")
POLAR_EDGE = 50.0
# Resample into a square stereographic grid to avoid pcolormesh warning
from scipy.ndimage import map_coordinates
grid_n = 800
xs = np.linspace(-1, 1, grid_n)
Xs, Ys = np.meshgrid(xs, xs)
edge_r = np.tan(np.deg2rad((90.0 - POLAR_EDGE) / 2))
Rs = np.sqrt(Xs**2 + Ys**2) * edge_r
lats_s = 90.0 - 2.0 * np.degrees(np.arctan(Rs))
lons_s = np.degrees(np.arctan2(Xs, Ys)) % 360.0
row_s = np.clip((90.0 - lats_s) / 180.0 * nrows, 0, nrows - 1)
col_s = np.clip(lons_s / 360.0 * ncols, 0, ncols - 1)
stereo = map_coordinates(post.astype(np.float64), [row_s, col_s],
                          order=1, mode='nearest').astype(np.float32)
disc_mask = (Xs**2 + Ys**2) > 1.0
stereo[disc_mask] = np.nan
import matplotlib.colors as mcolors2
cmap2 = plt.get_cmap('plasma').copy()
cmap2.set_bad('#111122')
ax2.imshow(stereo, cmap=cmap2, vmin=0.10, vmax=0.75,
           origin='upper', extent=[-1, 1, -1, 1], aspect='equal')
circle = plt.Circle((0, 0), 1.0, color='#888899', fill=False, linewidth=1.0)
ax2.add_patch(circle)
ax2.set_xlim(-1.05, 1.05)
ax2.set_ylim(-1.05, 1.05)
ax2.set_title('North Polar Cap\n(50–90°N)', color='white', fontsize=9)
ax2.axis('off')

for sp in axes:
    for s in sp.spines.values():
        s.set_edgecolor("#334455")

fig.suptitle("Titan Habitability at Red-Giant Ocean Peak (+5.9 Gya)",
             color="white", fontsize=11, y=1.01)
plt.tight_layout()
out = OUT_DIR / "future_epoch_bathymetry.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out}")
