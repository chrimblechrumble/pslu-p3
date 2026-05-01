#!/usr/bin/env python3
"""
Minimal test: verify VIMS+ISS east→west longitude fix.

Run AFTER replacing titan/preprocessing.py and re-preprocessing
the vims_mosaic (delete vims_mosaic_canonical.tif first to force
regeneration):

    rm processed/vims_mosaic_canonical.tif
    python run_pipeline.py --preprocess-only   # or however you trigger preprocessing

Then run this script:

    python test_vims_alignment.py <path_to_processed_dir>

Three tests must ALL pass:
  1. Kraken Mare (310°W, 72°N) has VIMS spectral texture (std > 0.02)
  2. Seignovert landmark (320°W, 70°N) has high f2 (> 0.85)
  3. No seam at 180°W (|step| < 0.01 at all latitude bands)
"""

import sys
import numpy as np

def main():
    if len(sys.argv) < 2:
        # Try default path
        import pathlib
        candidates = [
            pathlib.Path("processed/vims_mosaic_canonical.tif"),
            pathlib.Path("outputs/present/features/titan_features_present.nc"),
        ]
        print("Usage: python test_vims_alignment.py <vims_mosaic_canonical.tif>")
        print("   or: python test_vims_alignment.py <titan_features_present.nc>")
        print()
        # Try to find the file
        for c in candidates:
            if c.exists():
                print(f"Found: {c}")
                path = str(c)
                break
        else:
            print("No candidate files found. Please provide path as argument.")
            sys.exit(1)
    else:
        path = sys.argv[1]

    # Load the f2 organic abundance map
    if path.endswith(".nc"):
        import xarray as xr
        ds = xr.open_dataset(path)
        f2 = ds["organic_abundance"].values
    elif path.endswith(".tif"):
        import rasterio
        with rasterio.open(path) as src:
            f2 = src.read(1).astype(np.float32)
            f2[f2 == 0] = np.nan  # mask nodata if needed
        # Normalise to [0, 1] like the pipeline does
        p2, p98 = np.nanpercentile(f2, [2, 98])
        f2 = np.clip((f2 - p2) / (p98 - p2), 0, 1)
    else:
        print(f"Unknown file type: {path}")
        sys.exit(1)

    h, w = f2.shape
    print(f"Loaded f2 map: {h}x{w}")
    print()

    def rc(lat, lon_w):
        row = int(round((90.0 - lat) / 180.0 * h))
        col = int(round(lon_w / 360.0 * w)) % w
        return max(0, min(h-1, row)), max(0, min(w-1, col))

    def patch(lat, lon_w, r=5):
        row, col = rc(lat, lon_w)
        return f2[max(0,row-r):row+r+1, max(0,col-r):col+r+1]

    # ──────────────────────────────────────────────────────────────
    # TEST 1: Kraken Mare (310°W, 72°N) should have VIMS texture
    # ──────────────────────────────────────────────────────────────
    kraken = patch(72, 310, r=10)
    kraken_std = np.nanstd(kraken)
    test1 = kraken_std > 0.02
    print(f"TEST 1: Kraken (310°W, 72°N) local std = {kraken_std:.4f}")
    print(f"  Expected: > 0.02 (VIMS spectral texture present)")
    print(f"  Result:   {'PASS ✓' if test1 else 'FAIL ✗ — VIMS data not at correct location'}")
    print()

    # ──────────────────────────────────────────────────────────────
    # TEST 2: Seignovert (70°N, 40°E = 320°W) should be bright
    # ──────────────────────────────────────────────────────────────
    landmark = patch(70, 320, r=5)
    landmark_mean = np.nanmean(landmark)
    test2 = landmark_mean > 0.85
    print(f"TEST 2: Seignovert landmark (320°W, 70°N) mean = {landmark_mean:.4f}")
    print(f"  Expected: > 0.85 (saturated yellow in VIMS)")
    print(f"  Result:   {'PASS ✓' if test2 else 'FAIL ✗ — landmark not at expected location'}")
    print()

    # ──────────────────────────────────────────────────────────────
    # TEST 3: No seam at 180°W
    # ──────────────────────────────────────────────────────────────
    col_180 = w // 2
    max_step = 0
    worst_lat = None
    lat_bands = [
        ("80-90°N", 0, int(10/180*h)),
        ("60-70°N", int(20/180*h), int(30/180*h)),
        ("30-40°N", int(50/180*h), int(60/180*h)),
        ("0-10°N",  int(80/180*h), int(90/180*h)),
        ("0-10°S",  int(90/180*h), int(100/180*h)),
        ("30-40°S", int(120/180*h), int(130/180*h)),
        ("60-70°S", int(150/180*h), int(160/180*h)),
    ]
    print(f"TEST 3: Seam at 180°W")
    for label, r0, r1 in lat_bands:
        left  = np.nanmean(f2[r0:r1, col_180-10:col_180])
        right = np.nanmean(f2[r0:r1, col_180:col_180+10])
        step  = abs(right - left)
        if step > max_step:
            max_step = step
            worst_lat = label
        print(f"  {label:10s}: |step| = {step:.4f}")
    test3 = max_step < 0.01
    print(f"  Worst step: {max_step:.4f} at {worst_lat}")
    print(f"  Expected: < 0.01 (no visible seam)")
    print(f"  Result:   {'PASS ✓' if test3 else 'FAIL ✗ — seam still present'}")
    print()

    # ──────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────
    all_pass = test1 and test2 and test3
    print("=" * 50)
    if all_pass:
        print("ALL TESTS PASSED ✓")
        print("VIMS alignment is correct. Proceed with full pipeline run.")
    else:
        print("SOME TESTS FAILED ✗")
        if not test1:
            print("  → VIMS spectral data not at Kraken (310°W)")
        if not test2:
            print("  → Seignovert landmark not at 320°W")
        if not test3:
            print("  → Seam at 180°W still present")
        print("  Check the east_positive flip in _reproject_geotiff.")
    print("=" * 50)
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
