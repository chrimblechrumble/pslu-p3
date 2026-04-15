#!/usr/bin/env python3
"""
scripts/generate_future_bathymetry_map.py
==========================================
Generate the future epoch habitability map with DEM topographic contours
as ocean bathymetry (Figure R5).

Loadsthe future posterior and overlays topographic contours.
Panels: equirectangular main map + north polar cap + south polar cap.

Output: outputs/diagnostics/future_epoch_bathymetry.pdf
        outputs/diagnostics/future_epoch_bathymetry.png
Run:    python scripts/generate_future_bathymetry_map.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.ndimage import map_coordinates

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

if future_post_path.exists():
    post = np.load(future_post_path).astype(np.float32)
else:
    print("[WARN] Future posterior not found — using synthetic data")
    np.random.seed(42)
    post = np.clip(0.65 + np.random.normal(0, 0.06, (1802, 3603)), 0, 1).astype(np.float32)

nrows, ncols = post.shape
POLAR_EDGE   = 50.0   # degrees from pole
VMIN, VMAX   = 0.10, 0.75

cmap_main = plt.get_cmap("plasma").copy()
cmap_main.set_bad("#eeeeee")

# ── Layout: main map left (wide), two polar panels stacked on right ──────────
fig = plt.figure(figsize=(18, 6))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    width_ratios=[2.8, 1],
    height_ratios=[1, 1],
    hspace=0.08, wspace=0.06,
    left=0.05, right=0.97,
    top=0.88, bottom=0.10,
)

ax_main  = fig.add_subplot(gs[:, 0])   # spans both rows, left column
ax_north = fig.add_subplot(gs[0, 1])   # top-right
ax_south = fig.add_subplot(gs[1, 1])   # bottom-right

# ── Main equirectangular map ─────────────────────────────────────────────────
ax_main.set_facecolor("white")
im = ax_main.imshow(post, cmap=cmap_main, vmin=VMIN, vmax=VMAX,
                    aspect="auto", origin="upper", interpolation="nearest")

# Topographic contours
if HAS_RASTERIO and topo_tif_path.exists():
    with rasterio.open(topo_tif_path) as ds:
        topo = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            topo = np.where(topo == ds.nodata, np.nan, topo)
    from scipy.ndimage import gaussian_filter
    topo_smooth = gaussian_filter(np.nan_to_num(topo, nan=0.5), sigma=8)
    cs = ax_main.contour(topo_smooth, levels=[0.30, 0.55, 0.75],
                         colors=["#333333"], linewidths=[0.5, 0.8, 1.1], alpha=0.5)
    ax_main.clabel(cs, fmt={0.30: "shallow", 0.55: "mid", 0.75: "deep"},
                   fontsize=6, colors="black")
else:
    print(f"[WARN] Topography TIF not found ({topo_tif_path}) — skipping contours")

plt.colorbar(im, ax=ax_main, fraction=0.025, pad=0.02,
             label=r"$P(H\mid\mathbf{f})$").ax.tick_params(colors="black", labelsize=8)
ax_main.set_title(
    "Future Epoch (+5.9 Gya)  —  Global Water-Ammonia Ocean\n"
    "Topographic contours = future ocean bathymetry (relative depth proxy)",
    color="black", fontsize=9,
)
ax_main.set_xlabel("Longitude (°W)", color="black", fontsize=8)
ax_main.set_ylabel("Latitude (°N)", color="black", fontsize=8)
ax_main.tick_params(colors="black", labelsize=7)
ax_main.set_xticks(np.linspace(0, ncols, 7))
ax_main.set_xticklabels(["0°", "60°", "120°", "180°", "240°", "300°", "360°"])
ax_main.set_yticks(np.linspace(0, nrows, 7))
ax_main.set_yticklabels(["90°N", "60°N", "30°N", "0°", "30°S", "60°S", "90°S"])

# ── Shared stereographic resampler ──────────────────────────────────────────
def make_stereo(post_arr, pole_sign, edge_deg=POLAR_EDGE, grid_n=600):
    """pole_sign=+1 for north, -1 for south."""
    xs  = np.linspace(-1, 1, grid_n)
    Xs, Ys = np.meshgrid(xs, xs)
    edge_r  = np.tan(np.deg2rad((90.0 - edge_deg) / 2))
    Rs      = np.sqrt(Xs**2 + Ys**2) * edge_r
    lats_s  = pole_sign * (90.0 - 2.0 * np.degrees(np.arctan(Rs)))
    lons_s  = (np.degrees(np.arctan2(Xs, Ys)) + (0 if pole_sign == 1 else 180)) % 360.0
    row_s   = np.clip((90.0 - lats_s) / 180.0 * nrows, 0, nrows - 1)
    col_s   = np.clip(lons_s / 360.0 * ncols, 0, ncols - 1)
    stereo  = map_coordinates(post_arr.astype(np.float64), [row_s, col_s],
                              order=1, mode="nearest").astype(np.float32)
    stereo[Xs**2 + Ys**2 > 1.0] = np.nan
    return stereo

def draw_polar(ax, stereo, title):
    cmap_p = plt.get_cmap("plasma").copy()
    cmap_p.set_bad("#eeeeee")
    ax.imshow(stereo, cmap=cmap_p, vmin=VMIN, vmax=VMAX,
              origin="upper", extent=[-1, 1, -1, 1], aspect="equal")
    circle = plt.Circle((0, 0), 1.0, color="#555555", fill=False, linewidth=1.0)
    ax.add_patch(circle)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title, color="black", fontsize=8, pad=3)
    ax.set_facecolor("white")
    ax.axis("off")

# ── North polar cap ──────────────────────────────────────────────────────────
draw_polar(ax_north, make_stereo(post, pole_sign=+1),
           f"North Polar Cap\n({POLAR_EDGE:.0f}–90°N)")

# ── South polar cap ──────────────────────────────────────────────────────────
draw_polar(ax_south, make_stereo(post, pole_sign=-1),
           f"South Polar Cap\n({POLAR_EDGE:.0f}–90°S)")

fig.suptitle("Titan Habitability at Red-Giant Ocean Peak (+5.9 Gya)",
             color="black", fontsize=11)

for _ext in ("pdf", "png"):
    out = OUT_DIR / f"future_epoch_bathymetry.{_ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved -> {out}")
plt.close(fig)
