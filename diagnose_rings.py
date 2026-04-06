#!/usr/bin/env python3
"""
diagnose_rings.py
=================
Locate which feature TIF(s) contain the ring artefacts at ~73N and ~83N.

Strategy: for each feature TIF, extract a LATITUDE PROFILE (mean value per
row, averaged across all longitudes).  Then compute the row-to-row gradient.
Any sharp step in the gradient at row ~150 (75N) or ~126 (74N) etc. reveals
which feature is the source.

Run from the project root:
    python diagnose_rings.py

The script examines PRESENT mode TIFs (all other modes scale from these).
"""
import sys
from pathlib import Path
import numpy as np

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed.  Run: pip install rasterio")
    sys.exit(1)

NROWS = 1802
DEG_PER_ROW = 180.0 / NROWS

def row_to_lat(r):
    return 90.0 - r * DEG_PER_ROW

def lat_to_row(lat):
    return int((90.0 - lat) * NROWS / 180.0)

# Rows for the suspicious latitudes
ROW_73N = lat_to_row(73)
ROW_83N = lat_to_row(83)
WINDOW = 5   # rows either side to look for a step

tif_dir = Path("outputs/present/features/tifs")
if not tif_dir.exists():
    print(f"ERROR: {tif_dir} not found.  Run run_pipeline.py --temporal-mode present first.")
    sys.exit(1)

tifs = sorted(tif_dir.glob("*.tif"))
if not tifs:
    print(f"ERROR: No TIFs in {tif_dir}")
    sys.exit(1)

print("=" * 70)
print("RING DIAGNOSTICS -- latitude profile step-detection")
print(f"Checking for steps at ~73N (row ~{ROW_73N}) and ~83N (row ~{ROW_83N})")
print("=" * 70)
print()

results = []

for tif in tifs:
    name = tif.stem
    try:
        with rasterio.open(tif) as src:
            arr = src.read(1).astype(np.float64)
            nd = src.nodata
            if nd is not None:
                arr[arr == nd] = np.nan
    except Exception as e:
        print(f"  {name}: FAILED to read -- {e}")
        continue

    # Latitude profile: mean valid value per row
    profile = np.array([np.nanmean(arr[r, :]) for r in range(NROWS)])

    # Gradient (finite difference)
    grad = np.abs(np.diff(profile))

    # Look for peaks in gradient near the suspicious latitudes
    for label, centre_row in [("~83N", ROW_83N), ("~73N", ROW_73N)]:
        lo = max(0, centre_row - WINDOW)
        hi = min(len(grad), centre_row + WINDOW)
        local_max_grad = float(np.nanmax(grad[lo:hi]))
        global_p95_grad = float(np.nanpercentile(grad[np.isfinite(grad)], 95))
        ratio = local_max_grad / (global_p95_grad + 1e-12)

        flag = ""
        if ratio > 3.0:
            flag = " *** SHARP STEP -- RING SOURCE ***"
        elif ratio > 1.5:
            flag = " * notable step"

        results.append((name, label, local_max_grad, global_p95_grad, ratio, flag))

    # Print profile stats around both latitudes
    print(f"Feature: {name}")
    for label, centre_row in [("~83N", ROW_83N), ("~73N", ROW_73N)]:
        lo = max(0, centre_row - WINDOW)
        hi = min(NROWS, centre_row + WINDOW + 1)
        vals = profile[lo:hi]
        lat_lo = row_to_lat(hi - 1)
        lat_hi = row_to_lat(lo)
        local_max_grad = float(np.nanmax(np.abs(np.diff(vals)))) if len(vals) > 1 else 0
        global_p95_grad = float(np.nanpercentile(
            np.abs(np.diff(profile[np.isfinite(profile)])), 95
        ))
        ratio = local_max_grad / (global_p95_grad + 1e-12)
        flag = " *** SHARP STEP ***" if ratio > 3.0 else (" * notable" if ratio > 1.5 else "")
        print(f"  {label} (rows {lo}-{hi-1}, {lat_lo:.0f}-{lat_hi:.0f}°N): "
              f"max_grad={local_max_grad:.5f}  p95_global={global_p95_grad:.5f}  "
              f"ratio={ratio:.1f}x{flag}")

        # Show value immediately inside and outside the boundary
        if centre_row > 2 and centre_row < NROWS - 2:
            inside  = float(np.nanmean(profile[max(0, centre_row-3):centre_row]))
            outside = float(np.nanmean(profile[centre_row:min(NROWS, centre_row+3)]))
            print(f"    mean inside (poleward): {inside:.5f}   "
                  f"mean outside (equatorward): {outside:.5f}   "
                  f"delta: {outside - inside:+.5f}")
    print()

print("=" * 70)
print("SUMMARY -- features with sharp steps (ratio > 3x global p95 gradient):")
print("=" * 70)
found_any = False
for name, label, lg, pg, ratio, flag in sorted(results, key=lambda x: -x[4]):
    if ratio > 1.5:
        print(f"  {name:<35} {label}  ratio={ratio:5.1f}x  "
              f"local_grad={lg:.5f}  p95={pg:.5f}{flag}")
        found_any = True
if not found_any:
    print("  No sharp steps found at 73N or 83N in any feature TIF.")
    print("  The rings may come from the generate_temporal_maps.py animation code,")
    print("  not from the feature TIFs themselves.")

print()
print("NOTE: If no steps found in feature TIFs, the rings are likely from the")
print("animation's scale_features_to_epoch or polar_reproject functions.")
print("In that case, please also share the output of:")
print("  python run_pipeline.py --temporal-mode present 2>&1 | grep 'liquid_hydrocarbon'")
