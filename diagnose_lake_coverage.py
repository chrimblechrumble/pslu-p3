#!/usr/bin/env python3
"""
diagnose_lake_coverage.py
=========================
Confirm what fraction of pixels at each latitude are confirmed Birch lakes,
and whether the rings are the real polar sea boundaries or artefacts.

Run from project root:
    python diagnose_lake_coverage.py
"""
from pathlib import Path
import numpy as np

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed"); raise

NROWS = 1802
DEG_PER_ROW = 180.0 / NROWS

def row_to_lat(r):
    return 90.0 - (r + 0.5) * DEG_PER_ROW

# ── 1. Lake coverage fraction per latitude band ──────────────────────────────
print("=" * 65)
print("1. Birch confirmed-lake coverage per 2-degree latitude band")
print("=" * 65)

pl_tif = Path("data/processed/polar_lakes_canonical.tif")
if not pl_tif.exists():
    print(f"ERROR: {pl_tif} not found")
else:
    with rasterio.open(pl_tif) as src:
        pl = src.read(1)

    FILLED = 1
    print(f"  {'Lat range':>12}  {'n_pixels':>9}  {'n_lakes':>9}  {'%_lake':>7}  note")
    print("  " + "-"*60)
    for lat_hi in range(90, 50, -2):
        lat_lo = lat_hi - 2
        row_hi = int((90 - lat_hi) * NROWS / 180)
        row_lo = int((90 - lat_lo) * NROWS / 180)
        band = pl[row_hi:row_lo, :]
        n_tot = band.size
        n_lake = int((band == FILLED).sum())
        pct = 100.0 * n_lake / n_tot if n_tot > 0 else 0
        note = ""
        if lat_hi == 74:
            note = "  ← ring at ~73N"
        elif lat_hi == 84:
            note = "  ← ring at ~83N"
        if pct > 40:
            note += "  ** HIGH LAKE FRACTION"
        print(f"  {lat_lo:3d}-{lat_hi:3d}°N  {n_tot:9,}  {n_lake:9,}  {pct:7.1f}%{note}")

# ── 2. Surface atm interaction TIF profile ────────────────────────────────────
print()
print("=" * 65)
print("2. surface_atm_interaction profile near 73N and 83N")
print("=" * 65)

sat_tif = Path("outputs/present/features/tifs/surface_atm_interaction.tif")
lhc_tif = Path("outputs/present/features/tifs/liquid_hydrocarbon.tif")

for tif_path, name in [(sat_tif, "surface_atm_interaction"),
                        (lhc_tif, "liquid_hydrocarbon")]:
    if not tif_path.exists():
        print(f"  {name}: TIF not found")
        continue
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float64)
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan

    print(f"\n  {name} -- row values near 73N and 83N:")
    print(f"  {'Row':>4}  {'Lat°N':>6}  {'mean':>8}  {'non_zero_pct':>12}")
    for target_lat in [85, 84, 83, 82, 81, 80, 75, 74, 73, 72, 71, 70]:
        row = int((90 - target_lat) * NROWS / 180)
        if 0 <= row < NROWS:
            rowdata = arr[row, :]
            mean_val = float(np.nanmean(rowdata))
            nz_pct = 100.0 * float(np.nanmean(rowdata > 0.01))
            print(f"  {row:4d}  {target_lat:6.1f}  {mean_val:8.4f}  {nz_pct:11.1f}%")

# ── 3. Check if surface_atm ring matches lake margin dilation width ───────────
print()
print("=" * 65)
print("3. Diagnosis summary")
print("=" * 65)
print()
print("Expected values (v4.29+, correct Birch layout + antimeridian fix):")
print("  liquid_hydrocarbon row 170 (73N): ~0.09-0.15  (NOT 0.99)")
print("  surface_atm_interaction row 80 (82N): ~0.05   (NOT 0.32)")
print()
print("If values are anomalously high, re-run:")
print("  rm data/processed/polar_lakes_canonical.tif")
print("  python run_pipeline.py --temporal-mode present --overwrite")
