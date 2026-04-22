"""
check_all_canonical_tifs.py
===========================
Run from your titan_pipeline root:

    python check_all_canonical_tifs.py

Checks every canonical TIF in the preprocessed directory for the issues
discovered during the v6 audit:

  1.  Shape       -- must be (1802, 3603)
  2.  Nodata tag  -- should be set (missing tag lets -9999 through as valid data)
  3.  Seam        -- col=0 and col=3602 should agree (same physical meridian)
  4.  Elev range  -- DEM: peak should be >500 m after the 2ppd→8ppd fix
  5.  Coverage    -- how much valid data exists
  6.  Value range -- sanity-check min/max for each layer type
"""
import sys
import numpy as np
from pathlib import Path

try:
    import rasterio
except ImportError:
    sys.exit("rasterio not installed — run: pip install rasterio")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR   = Path("outputs/preprocessed")  # adjust if different
FEATURE_TIF_DIR = Path("outputs/present/features/tifs")

EXPECTED_SHAPE = (1802, 3603)

# Expected ranges for sanity checking [min_plausible, max_plausible]
VALUE_RANGES = {
    "topography":         (-2000,   3500, "m above ref sphere"),
    "sar_mosaic":         (-30,      10,  "dB backscatter"),
    "geomorphology":      (0,        8,   "terrain class int"),
    "cirs_temperature":   (85,      100,  "K surface temp"),
    "vims_5um_2um_ratio": (0,      500,   "raw I/F ratio DN"),
    "vims_mosaic":        (0,      500,   "raw I/F ratio DN"),
    "vims_5um":           (0,      0.5,   "I/F"),
    "vims_2um":           (0,      0.5,   "I/F"),
    "channel_density":    (0,        1,   "normalised [0,1]"),
    "vims_coverage":      (0,        1,   "normalised [0,1]"),
    "polar_lakes":        (0,        5,   "class label"),
    # feature TIFs are all normalised [0,1]
    "default_feature":    (0,        1,   "normalised [0,1]"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_tif(path: Path, label: str = "") -> dict:
    name = label or path.stem.replace("_canonical", "")
    result = {"name": name, "path": path, "issues": []}

    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nd  = src.nodata
            profile = src.profile
    except Exception as e:
        result["issues"].append(f"CANNOT OPEN: {e}")
        return result

    # --- 1. Shape ----------------------------------------------------------
    result["shape"] = arr.shape
    if arr.shape != EXPECTED_SHAPE:
        result["issues"].append(
            f"WRONG SHAPE: {arr.shape} (expected {EXPECTED_SHAPE})")

    # --- 2. Nodata handling ------------------------------------------------
    result["nodata_tag"] = nd
    if nd is None:
        # Check if -9999 sentinel is lurking without a tag
        sentinel_pct = 100 * float(np.mean(arr <= -9998))
        if sentinel_pct > 0.1:
            result["issues"].append(
                f"NO NODATA TAG but {sentinel_pct:.1f}% pixels = -9999 sentinel")
        result["nodata_applied"] = False
    elif np.isnan(float(nd)):
        arr[~np.isfinite(arr)] = np.nan
        result["nodata_applied"] = True
    else:
        # Sentinel value; also catch -9999 variants
        arr[arr == float(nd)] = np.nan
        arr[arr <= -9998] = np.nan
        result["nodata_applied"] = True

    # --- 3. Coverage -------------------------------------------------------
    finite_mask = np.isfinite(arr)
    result["valid_pct"]  = 100 * float(finite_mask.mean())
    result["nodata_pct"] = 100 - result["valid_pct"]

    if result["valid_pct"] == 0:
        result["issues"].append("ALL PIXELS ARE NODATA")
        return result

    # --- 4. Seam check (col 0 vs col 3602) ---------------------------------
    c0    = arr[:, 0]
    clast = arr[:, -1]
    c1    = arr[:, 1]
    cprev = arr[:, -2]
    wrap_diff    = float(np.nanmean(np.abs(c0     - clast)))
    interior_l   = float(np.nanmean(np.abs(c0     - c1)))
    interior_r   = float(np.nanmean(np.abs(clast  - cprev)))
    interior_avg = max(interior_l, interior_r, 1e-9)
    ratio = wrap_diff / interior_avg
    result["seam_diff"]     = wrap_diff
    result["seam_ratio"]    = ratio
    result["seam_interior"] = interior_avg
    if ratio > 10:
        result["issues"].append(
            f"CRITICAL seam at 0°W: wrap|diff|={wrap_diff:.4f}, ratio={ratio:.1f}x interior")
    elif ratio > 3:
        result["issues"].append(
            f"Significant seam at 0°W: wrap|diff|={wrap_diff:.4f}, ratio={ratio:.1f}x interior")

    # --- 5. Value range ----------------------------------------------------
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    vmean = float(np.nanmean(arr))
    vstd  = float(np.nanstd(arr))
    result["min"]  = vmin
    result["max"]  = vmax
    result["mean"] = vmean
    result["std"]  = vstd

    # Sanity check against known ranges
    spec_key = name if name in VALUE_RANGES else "default_feature"
    if spec_key in VALUE_RANGES:
        lo, hi, units = VALUE_RANGES[spec_key]
        if vmin < lo * 2:
            result["issues"].append(
                f"min={vmin:.1f} is below plausible range [{lo}, {hi}] {units}")
        if vmax > hi * 2:
            result["issues"].append(
                f"max={vmax:.1f} exceeds plausible range [{lo}, {hi}] {units}")
        result["units"] = units

    # Special: DEM peak check
    if "topograph" in name and "complex" not in name:
        if vmax < 400:
            result["issues"].append(
                f"DEM max={vmax:.1f} m is suspiciously low (expected >500 m). "
                f"Check DEM resolution fix (2ppd→8ppd) and delete+regenerate "
                f"topography_canonical.tif.")

    # --- 6. Latitude structure (north vs south) ----------------------------
    north_nan = float(np.mean(~np.isfinite(arr[:300, :])))
    south_nan = float(np.mean(~np.isfinite(arr[1500:, :])))
    result["north_nodata_pct"] = 100 * north_nan
    result["south_nodata_pct"] = 100 * south_nan

    return result


def severity_tag(r: dict) -> str:
    issues = r.get("issues", [])
    if not issues:
        return "✓"
    for i in issues:
        if "CRITICAL" in i or "WRONG SHAPE" in i or "ALL PIXELS" in i or \
           "DEM max" in i or "NO NODATA" in i:
            return "✗ CRITICAL"
    return "⚠ WARNING"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("=" * 78)
    print("CANONICAL TIF AUDIT")
    print("=" * 78)

    # Collect all TIFs to check
    tifs_to_check: list[tuple[Path, str]] = []

    # Preprocessed canonical TIFs
    if PROCESSED_DIR.exists():
        for p in sorted(PROCESSED_DIR.glob("*_canonical.tif")):
            name = p.stem.replace("_canonical", "")
            tifs_to_check.append((p, name))
    else:
        # Try common alternative locations
        for alt in [Path("outputs/present/preprocessed"),
                    Path("data/processed"), Path("outputs")]:
            if alt.exists():
                for p in sorted(alt.glob("*_canonical.tif")):
                    name = p.stem.replace("_canonical", "")
                    tifs_to_check.append((p, name))
                if tifs_to_check:
                    print(f"(Found canonical TIFs in {alt})")
                    break

    if not tifs_to_check:
        print(f"No *_canonical.tif files found in {PROCESSED_DIR}")
        print("Check PROCESSED_DIR path at the top of this script.")
        print()

    # Feature TIFs
    feature_tifs: list[tuple[Path, str]] = []
    if FEATURE_TIF_DIR.exists():
        for p in sorted(FEATURE_TIF_DIR.glob("*.tif")):
            feature_tifs.append((p, p.stem))

    # --------------- Print results ----------------------------------------
    def print_section(title, results):
        print(f"\n{'─' * 78}")
        print(f"  {title}")
        print(f"{'─' * 78}")
        print(f"  {'TIF name':<38} {'valid%':>7} {'min':>9} {'max':>9}  {'seam':>6}  status")
        print(f"  {'-'*38} {'------':>7} {'-----':>9} {'-----':>9}  {'------':>6}  ------")
        for r in results:
            name = r["name"][:38]
            vpct = f"{r.get('valid_pct', 0):.0f}%" if "valid_pct" in r else "ERR"
            vmin = f"{r.get('min', float('nan')):.1f}" if "min" in r else "—"
            vmax = f"{r.get('max', float('nan')):.1f}" if "max" in r else "—"
            sratio = f"{r.get('seam_ratio', 0):.1f}x" if "seam_ratio" in r else "—"
            status = severity_tag(r)
            print(f"  {name:<38} {vpct:>7} {vmin:>9} {vmax:>9}  {sratio:>6}  {status}")

            for issue in r.get("issues", []):
                print(f"    ↳ {issue}")

    # Run checks
    canonical_results = [check_tif(p, n) for p, n in tifs_to_check]
    feature_results   = [check_tif(p, n) for p, n in feature_tifs]

    if canonical_results:
        print_section("PREPROCESSED CANONICAL TIFs", canonical_results)
    if feature_results:
        print_section("FEATURE TIFs  (outputs/present/features/tifs/)", feature_results)

    # Summary
    all_results = canonical_results + feature_results
    n_ok   = sum(1 for r in all_results if not r["issues"])
    n_warn = sum(1 for r in all_results if any(
        "CRITICAL" not in i for i in r.get("issues", [])) and r.get("issues"))
    n_crit = sum(1 for r in all_results if any(
        "CRITICAL" in i or "WRONG SHAPE" in i or "DEM max" in i
        for i in r.get("issues", [])))

    print(f"\n{'=' * 78}")
    print(f"  SUMMARY: {len(all_results)} TIFs checked — "
          f"{n_ok} OK  |  {n_warn} warnings  |  {n_crit} critical")
    print(f"{'=' * 78}")

    print("""
  seam ratio = mean|col_0 - col_3602| / max interior pixel-to-pixel variation
  > 3x  = significant seam   > 10x = critical seam
  
  NEXT STEPS if topography_canonical.tif shows DEM max < 400 m:
    → Delete topography_canonical.tif (and the titan_canonical_stack.nc)
    → Run: python run_pipeline.py --all-temporal-modes --overwrite
    → The 2ppd→8ppd DEM fix in preprocessing.py will regenerate it correctly.
    → Re-run check_all_canonical_tifs.py to confirm DEM max > 500 m.
""")


if __name__ == "__main__":
    run()
