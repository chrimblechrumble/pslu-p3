# Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
# Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""
diagnose_organic_seam.py
========================
Diagnostic script for the organic_abundance seam at 180 degW.

Run from the project root:
    python diagnose_organic_seam.py

Saves:  outputs/diagnostics/organic_seam_diagnosis.png
        outputs/diagnostics/organic_seam_values.txt
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# -- Load the preprocessed canonical stack --------------------------------
processed = Path("data/processed")
raw_dir   = Path("data/raw")
out_dir   = Path("outputs/diagnostics")
out_dir.mkdir(parents=True, exist_ok=True)

import rasterio
print(
    "Titan Habitability Pipeline  Copyright (C) 2025/2026  Chris Meadows\n"
    "This program comes with ABSOLUTELY NO WARRANTY; for details, see the\n"
    "README.md at the project root.\n"
    "This is free software, and you are welcome to redistribute it\n"
    "under certain conditions; see the LICENSE.md file at the project\n"
    "root for details.\n"
)

def load_band1(path: Path) -> np.ndarray:
    """Load band 1 of a GeoTIFF as float32, nodata -> NaN."""
    with rasterio.open(path) as src:
        data: np.ndarray = src.read(1).astype(np.float64)
        nd = src.nodata
        if nd is not None:
            data[data == nd] = np.nan
    return data

# Look for the canonical files first, then raw
vims_paths = [
    processed / "vims_mosaic_canonical.tif",
    raw_dir   / "Titan_VIMS-ISS.tif",
]
iss_paths = [
    processed / "iss_mosaic_450m_canonical.tif",
    raw_dir   / "Titan_ISS_NearGlobal_450m.tif",
]

vims_path = next((p for p in vims_paths if p.exists()), None)
iss_path  = next((p for p in iss_paths  if p.exists()), None)

if not vims_path:
    print("ERROR: no VIMS mosaic found -- check data/raw/ or data/processed/")
    sys.exit(1)
if not iss_path:
    print("ERROR: no ISS mosaic found -- check data/raw/ or data/processed/")
    sys.exit(1)

print(f"VIMS file: {vims_path}")
print(f"ISS  file: {iss_path}")

vims = load_band1(vims_path)
iss  = load_band1(iss_path)

print(f"VIMS shape: {vims.shape}  ISS shape: {iss.shape}")
print(f"VIMS valid: {np.sum(np.isfinite(vims)):,}  ({100*np.sum(np.isfinite(vims))/vims.size:.1f}%)")
print(f"ISS  valid: {np.sum(np.isfinite(iss)):,}  ({100*np.sum(np.isfinite(iss))/iss.size:.1f}%)")

nrows, ncols = vims.shape
half_col = ncols // 2   # approximate 180 degW column

# -- Equatorial strip statistics (rows +/-10% of nrows) ---------------------
eq_lo = int(nrows * 0.45)
eq_hi = int(nrows * 0.55)

vims_eq = vims[eq_lo:eq_hi, :]
iss_eq  = iss[eq_lo:eq_hi, :]

# Columns near the boundary
boundary_range = max(5, ncols // 36)   # +/-5 deg

left_col  = half_col - boundary_range
right_col = half_col + boundary_range

# Mean profile along the equatorial strip
vims_profile = np.nanmean(vims_eq, axis=0)
iss_profile  = np.nanmean(iss_eq,  axis=0)

# -- Overlap statistics ----------------------------------------------------
overlap = np.isfinite(vims) & np.isfinite(iss)
print(f"\nOverlap pixels: {np.sum(overlap):,}")
if np.sum(overlap) > 0:
    print(f"VIMS in overlap: mean={np.nanmean(vims[overlap]):.4f}  "
          f"std={np.nanstd(vims[overlap]):.4f}  "
          f"median={np.nanmedian(vims[overlap]):.4f}")
    print(f"ISS  in overlap: mean={np.nanmean(iss[overlap]):.4f}  "
          f"std={np.nanstd(iss[overlap]):.4f}  "
          f"median={np.nanmedian(iss[overlap]):.4f}")

# Values just left and right of 180 degW
print(f"\nEquatorial strip mean values near 180 degW boundary:")
for c in range(max(0, half_col-5), min(ncols, half_col+5)):
    lon = 360.0 * c / ncols
    v = float(np.nanmean(vims_eq[:, c]))
    i = float(np.nanmean(iss_eq[:,  c]))
    flag = " <<< BOUNDARY" if abs(c - half_col) <= 1 else ""
    print(f"  col {c:4d}  lon={lon:6.1f} degW  VIMS={v:8.4f}  ISS={i:8.4f}{flag}")

# -- Plot ------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("Organic Abundance — Seam Diagnosis", fontsize=13)

lons = np.linspace(0, 360, ncols)

# Panel 1: raw VIMS and ISS profiles along equator
ax = axes[0]
ax.plot(lons, vims_profile, 'C1', lw=0.8, label='VIMS raw (equatorial mean)')
ax.plot(lons, iss_profile,  'C0', lw=0.8, label='ISS raw (equatorial mean)')
ax.axvline(180, color='r', ls='--', lw=1.5, label='180°W boundary')
ax.set_ylabel('Raw pixel value')
ax.set_title('Raw values before normalisation')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_xlim(0, 360)

# Panel 2: zoom near boundary +/-30 deg
ax = axes[1]
lo_lon, hi_lon = 150, 210
mask = (lons >= lo_lon) & (lons <= hi_lon)
ax.plot(lons[mask], vims_profile[mask], 'C1o-', ms=3, lw=1, label='VIMS raw')
ax.plot(lons[mask], iss_profile[mask],  'C0s-', ms=3, lw=1, label='ISS raw')
ax.axvline(180, color='r', ls='--', lw=1.5, label='180°W boundary')
ax.set_ylabel('Raw pixel value')
ax.set_title('Zoom: 150–210°W (boundary region)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3: VIMS coverage map
ax = axes[2]
vims_coverage = np.isfinite(vims).astype(float)
ax.imshow(vims_coverage, aspect='auto', extent=[0,360,-90,90],
          cmap='RdYlGn', origin='upper', vmin=0, vmax=1)
ax.axvline(180, color='r', ls='--', lw=1.5, label='180°W')
ax.set_xlabel('Longitude (°W)')
ax.set_ylabel('Latitude (°)')
ax.set_title('VIMS coverage (green=valid, red=NaN)')
ax.legend(fontsize=8)

plt.tight_layout()
out_fig = out_dir / "organic_seam_diagnosis.png"
fig.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_fig}")

# Save text summary
with open(out_dir / "organic_seam_values.txt", 'w') as f:
    f.write(f"VIMS file: {vims_path}\n")
    f.write(f"ISS  file: {iss_path}\n")
    f.write(f"VIMS shape: {vims.shape}  ISS shape: {iss.shape}\n\n")
    f.write(f"VIMS stats (all valid px): "
            f"min={np.nanmin(vims):.4f} max={np.nanmax(vims):.4f} "
            f"mean={np.nanmean(vims):.4f} std={np.nanstd(vims):.4f}\n")
    f.write(f"ISS  stats (all valid px): "
            f"min={np.nanmin(iss):.4f} max={np.nanmax(iss):.4f} "
            f"mean={np.nanmean(iss):.4f} std={np.nanstd(iss):.4f}\n\n")
    f.write(f"Overlap pixels: {np.sum(overlap):,}\n")
    if np.sum(overlap) > 0:
        f.write(f"VIMS in overlap: mean={np.nanmean(vims[overlap]):.4f} "
                f"std={np.nanstd(vims[overlap]):.4f}\n")
        f.write(f"ISS  in overlap: mean={np.nanmean(iss[overlap]):.4f} "
                f"std={np.nanstd(iss[overlap]):.4f}\n")
print(f"Saved: {out_dir / 'organic_seam_values.txt'}")
