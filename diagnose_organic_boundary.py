"""
diagnose_organic_boundary.py
Probes the exact values at the 180°W seam in organic_abundance.
Run: python diagnose_organic_boundary.py
Saves: outputs/diagnostics/organic_boundary_*.png + report.txt
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path('outputs/diagnostics')
out_dir.mkdir(parents=True, exist_ok=True)

import rasterio
R = print

# Load canonical layers
def load(name: str) -> "np.ndarray | None":
    p: Path = Path(f'data/processed/{name}_canonical.tif')
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        arr: np.ndarray = src.read(1).astype(np.float64)
        nd = src.nodata
        if nd is not None:
            mask: np.ndarray = (np.isnan(arr) if np.isnan(float(nd))
                                else (arr == float(nd)))
            arr[mask] = np.nan
    return arr

vims = load('vims_mosaic')
geo  = load('geomorphology')

if vims is None or geo is None:
    R("ERROR: need vims_mosaic_canonical.tif and geomorphology_canonical.tif")
    sys.exit(1)

nrows, ncols = vims.shape
half = ncols // 2   # column closest to 180°W

# ── Call the actual _organic_abundance function from features.py ──────────────
# This ensures the diagnostic reflects the real pipeline code, not a
# reimplementation that would become stale when features.py changes.
import xarray as xr
from titan.features import FeatureExtractor, TERRAIN_ORGANIC_SCORES
from titan.preprocessing import CanonicalGrid, normalise_to_0_1

grid = CanonicalGrid()
lats = grid.lat_centres_deg()
lons = grid.lon_centres_deg()

stack = xr.Dataset({
    'vims_mosaic':    xr.DataArray(vims.astype(np.float32),  dims=['lat','lon'],
                                   coords={'lat': lats, 'lon': lons}),
    'geomorphology':  xr.DataArray(geo.astype(np.float32),   dims=['lat','lon'],
                                   coords={'lat': lats, 'lon': lons}),
})

extractor = FeatureExtractor(grid)
nan_arr = np.full((nrows, ncols), np.nan, dtype=np.float32)
org = extractor._organic_abundance(stack, nan_arr)

# For the boundary table, also compute vims_norm and cal_geo separately
vims_finite = vims[np.isfinite(vims)]
v2  = float(np.percentile(vims_finite, 2))
v98 = float(np.percentile(vims_finite, 98))
vims_norm = np.clip((vims - v2) / (v98 - v2 + 1e-12), 0.0, 1.0)
vims_norm[~np.isfinite(vims)] = np.nan
geo_int = np.where(np.isfinite(geo), geo, 0).astype(np.int32)
names = {1:'Craters',2:'Dunes',3:'Plains',4:'Basins',5:'Mountains',6:'Labyrinth',7:'Lakes'}
overlap = np.isfinite(vims_norm) & (geo_int > 0)
calibrated = {}
for cls in range(1, 8):
    px = (geo_int == cls) & overlap
    n = int(px.sum())
    if n >= 500:
        calibrated[cls] = float(np.mean(vims_norm[px]))
    else:
        calibrated[cls] = float(TERRAIN_ORGANIC_SCORES.get(cls, 0.5))
cal_geo = np.full((nrows, ncols), np.nan, dtype=np.float32)
for cls, score in calibrated.items():
    cal_geo[geo_int == cls] = float(score)

R("=" * 60)
R("VIMS-CALIBRATED GEO SCORES (from actual features.py)")
R("=" * 60)
for cls in range(1, 8):
    px = (geo_int == cls) & overlap
    n = int(px.sum())
    mu = calibrated[cls]
    published = TERRAIN_ORGANIC_SCORES.get(cls, float('nan'))
    R(f"  Class {cls} {names[cls]:12s}: n={n:>8,}  calibrated={mu:.4f}  published={published:.4f}  diff={mu-published:+.4f}")

R("\n" + "=" * 60)
R("BOUNDARY ANALYSIS (columns near 180°W)")
R("=" * 60)

# Equatorial strip ±20°
lat_centres = np.linspace(90 - 0.5*180/nrows, -90 + 0.5*180/nrows, nrows)
eq = np.where(np.abs(lat_centres) < 20)[0]

R(f"\nEquatorial strip (±20°), columns around boundary (half={half}):")
R(f"{'Col':>6}  {'Lon°W':>7}  {'VIMS_norm':>10}  {'CalGeo':>8}  {'Org_out':>8}  {'Source':>6}")
for dc in range(-8, 9):
    c = half + dc
    if c < 0 or c >= ncols: continue
    lon = 360.0 * c / ncols
    vv = float(np.nanmean(vims_norm[eq, c]))
    gv = float(np.nanmean(cal_geo[eq, c]))
    ov = float(np.nanmean(org[eq, c]))
    src = 'VIMS' if np.any(np.isfinite(vims_norm[eq, c])) else 'GEO'
    flag = " <-- BOUNDARY" if dc == 0 else ""
    R(f"{c:6d}  {lon:7.1f}°  {vv:10.4f}  {gv:8.4f}  {ov:8.4f}  {src:>6}{flag}")

# Compute seam magnitude
left_mean  = float(np.nanmean(org[eq, half-10:half]))
right_mean = float(np.nanmean(org[eq, half:half+10]))
R(f"\nLeft mean (cols {half-10}–{half}):  {left_mean:.4f}")
R(f"Right mean (cols {half}–{half+10}): {right_mean:.4f}")
R(f"Step size at boundary:        {right_mean - left_mean:+.4f}")
R(f"Typical within-VIMS variation (std over equatorial strip): {float(np.nanstd(vims_norm[eq, :half])):.4f}")

# ── Figures ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle("Organic Abundance Seam Diagnostic", fontsize=13)
lon_centres = np.linspace(0.5*360/ncols, 360-0.5*360/ncols, ncols)

# Panel 1: Organic abundance map
ax = axes[0,0]
ax.imshow(org, origin='upper', extent=[0,360,-90,90], aspect='auto',
          cmap='inferno', vmin=0, vmax=1)
ax.axvline(180, color='cyan', lw=1.5, ls='--', label='180°W seam')
ax.set_title('Organic abundance (combined)')
ax.set_xlabel('Lon (°W)'); ax.set_ylabel('Lat (°)')
ax.legend(fontsize=8)

# Panel 2: Column-mean profile
ax = axes[0,1]
col_mean = np.nanmean(org[eq, :], axis=0)
ax.plot(lon_centres, col_mean, 'k-', lw=0.8)
ax.axvline(180, color='r', lw=1.5, ls='--', label='180°W seam')
ax.fill_betweenx([0,1], 170, 180, alpha=0.15, color='orange', label='VIMS boundary ±10°')
ax.fill_betweenx([0,1], 180, 190, alpha=0.15, color='blue')
ax.set_title('Column mean (equatorial strip ±20°)')
ax.set_xlabel('Lon (°W)'); ax.set_ylabel('Organic abundance [0-1]')
ax.set_xlim(140, 220); ax.legend(fontsize=8)

# Panel 3: VIMS coverage mask
ax = axes[1,0]
coverage = np.isfinite(vims_norm).astype(float)
coverage[coverage == 0] = np.nan
ax.imshow(coverage, origin='upper', extent=[0,360,-90,90], aspect='auto',
          cmap='Greens', vmin=0, vmax=1)
ax.axvline(180, color='r', lw=1.5, ls='--')
ax.set_title('VIMS coverage (green=valid)')
ax.set_xlabel('Lon (°W)'); ax.set_ylabel('Lat (°)')

# Panel 4: Geo class scores in equatorial strip
ax = axes[1,1]
for cls, name in names.items():
    if cls not in calibrated: continue
    cls_cols = []
    cls_means = []
    for c in range(ncols):
        px_in_col = (geo_int[eq, c] == cls)
        if px_in_col.sum() > 2:
            cls_cols.append(lon_centres[c])
            cls_means.append(calibrated[cls])
    if cls_cols:
        ax.scatter(cls_cols, cls_means, s=0.5, label=f'{name}={calibrated[cls]:.3f}', alpha=0.3)
ax.axvline(180, color='r', lw=1.5, ls='--')
ax.set_title('Calibrated geo scores per column\n(should be flat lines per class)')
ax.set_xlabel('Lon (°W)'); ax.set_ylabel('Score')
ax.legend(fontsize=6, markerscale=5)

plt.tight_layout()
fig.savefig(out_dir/'organic_boundary_diag.png', dpi=120, bbox_inches='tight')
R(f"\nSaved: {out_dir/'organic_boundary_diag.png'}")

# Also save the organic abundance as a TIF
import rasterio
from rasterio.crs import CRS
from titan.preprocessing import CanonicalGrid
grid = CanonicalGrid()
out_tif = out_dir / 'organic_abundance_combined.tif'
nodata_val = -9999.0
arr_out = np.where(np.isfinite(org), org, nodata_val).astype(np.float32)
with rasterio.open(out_tif, 'w', driver='GTiff', dtype='float32', count=1,
                   width=grid.ncols, height=grid.nrows,
                   crs=grid.crs, transform=grid.transform,
                   nodata=nodata_val, compress='deflate',
                   tiled=True, blockxsize=256, blockysize=256) as dst:
    dst.write(arr_out, 1)
R(f"Saved TIF: {out_tif}")
