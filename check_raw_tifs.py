"""
check_raw_tifs.py
=================
Run from your titan_pipeline root directory:

    python check_raw_tifs.py

Identifies which raw input TIFs have genuine data seams at the 0°W/360°W
longitude boundary. This tells you whether the residual discontinuity
visible in the maps (after filter-mode fixes) is in the source data itself.
"""
import numpy as np
from pathlib import Path

try:
    import rasterio
except ImportError:
    raise SystemExit("rasterio not installed — run: pip install rasterio")


# ---------------------------------------------------------------------------
# Configuration: where to look for TIFs
# ---------------------------------------------------------------------------
FEATURE_TIF_DIR  = Path("outputs/present/features/tifs")
CANDIDATE_DIRS   = [
    FEATURE_TIF_DIR,
    Path("outputs/present/preprocessed"),
    Path("outputs/preprocessed"),
    Path("outputs"),
    Path("data"),
    Path("data/raw"),
]

NCOLS_EXPECTED = 3603   # canonical grid width


def find_tif(name_fragment: str) -> Path | None:
    """Find the first TIF whose filename contains name_fragment."""
    for d in CANDIDATE_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.tif")):
            if name_fragment.lower() in p.name.lower():
                return p
    return None


def check_tif(path: Path) -> dict:
    """Return seam statistics for a single TIF."""
    with rasterio.open(path) as ds:
        arr  = ds.read(1).astype(np.float64)
        ncols = arr.shape[1]
        nd   = ds.nodata
    if nd is not None:
        arr[arr == nd] = np.nan

    last_col = ncols - 1   # should be 3602 on canonical grid

    c0      = arr[:, 0]
    c_last  = arr[:, last_col]
    c1      = arr[:, 1]
    c_prev  = arr[:, last_col - 1]

    wrap_diff    = float(np.nanmean(np.abs(c0     - c_last)))
    interior_l   = float(np.nanmean(np.abs(c0     - c1)))
    interior_r   = float(np.nanmean(np.abs(c_last - c_prev)))
    ratio = wrap_diff / max(interior_l, interior_r, 1e-9)

    return {
        "path":       path,
        "ncols":      ncols,
        "c0_mean":    float(np.nanmean(c0)),
        "clast_mean": float(np.nanmean(c_last)),
        "wrap_diff":  wrap_diff,
        "interior_l": interior_l,
        "interior_r": interior_r,
        "ratio":      ratio,
        "seam":       ratio > 3.0,
    }


def severity(r: dict) -> str:
    if r["ratio"] > 10:  return "CRITICAL seam"
    if r["ratio"] >  3:  return "significant seam"
    if r["ratio"] >  1.5: return "minor seam"
    return "clean"


# ---------------------------------------------------------------------------
# 1. Check raw source TIFs
# ---------------------------------------------------------------------------
RAW_TIFS = [
    ("sar_mosaic",   "SAR backscatter (feeds acetylene_energy F3)"),
    ("topography",   "GTDE elevation DEM (feeds F3, F5, F6)"),
    ("geomorphology","Lopes terrain class raster (feeds F2, F7)"),
]

print("=" * 74)
print("RAW SOURCE TIF SEAM CHECK")
print("=" * 74)
print(f"{'TIF name':<32} {'col-0 mean':>10} {'col-last':>10} {'|diff|':>8} "
      f"{'ratio':>7}  verdict")
print("-" * 74)

raw_results = []
for name, desc in RAW_TIFS:
    p = find_tif(name)
    if p is None:
        print(f"  {name:<30}  NOT FOUND in standard locations")
        continue
    r = check_tif(p)
    flag = "  ← " + severity(r) if r["ratio"] > 1.5 else ""
    print(f"  {p.name:<30}  {r['c0_mean']:>10.4f}  {r['clast_mean']:>10.4f}  "
          f"{r['wrap_diff']:>8.4f}  {r['ratio']:>6.1f}x{flag}")
    print(f"    ({desc})")
    raw_results.append(r)

# ---------------------------------------------------------------------------
# 2. Check all feature TIFs
# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("FEATURE TIF SEAM CHECK")
print("=" * 74)
print(f"{'Feature TIF':<38} {'col-0':>8} {'col-last':>9} {'|diff|':>8} "
      f"{'ratio':>7}  verdict")
print("-" * 74)

feature_results = []
if FEATURE_TIF_DIR.exists():
    for p in sorted(FEATURE_TIF_DIR.glob("*.tif")):
        r = check_tif(p)
        flag = "  ← " + severity(r) if r["ratio"] > 1.5 else ""
        print(f"  {p.name:<36}  {r['c0_mean']:>8.4f}  {r['clast_mean']:>9.4f}  "
              f"{r['wrap_diff']:>8.4f}  {r['ratio']:>6.1f}x{flag}")
        feature_results.append(r)
else:
    print(f"  {FEATURE_TIF_DIR} not found.")

# ---------------------------------------------------------------------------
# 3. Interpretation guide
# ---------------------------------------------------------------------------
all_results  = raw_results + feature_results
seam_tifs    = [r for r in all_results if r["seam"]]

print()
print("=" * 74)
print("INTERPRETATION")
print("=" * 74)
print("""
  ratio = (mean |col_0 - col_last|) / max(interior pixel-to-pixel variation)

  ratio < 1.5  →  no seam; normal terrain variation across the meridian
  ratio 1.5-3  →  minor seam; may be borderline visible
  ratio > 3    →  significant seam in this file
  ratio > 10   →  critical seam — dominant source of the visible discontinuity
""")

if seam_tifs:
    print("FILES WITH SEAMS:")
    for r in seam_tifs:
        sev = severity(r)
        print(f"  {r['path'].name:<36}  {sev}  (ratio={r['ratio']:.1f}x)")
    print()
    print("ROOT CAUSE GUIDANCE:")
    raw_seam_names = [r["path"].name for r in raw_results if r["seam"]]
    if "sar_mosaic" in " ".join(raw_seam_names):
        print("  sar_mosaic: SAR backscatter has a calibration or coverage seam at 0°W.")
        print("  → This is likely a mosaicking artefact where two SAR swaths with")
        print("    slightly different radiometric calibrations were stitched together.")
        print("    Fix: check the SAR ingestion pipeline for radiometric normalisation")
        print("    across the seam, or mask the affected columns (±5 px = ±22 km).")
    if "topography" in " ".join(raw_seam_names):
        print("  topography: DEM has a data discontinuity at 0°W.")
        print("  → The GTDE interpolated DEM may not enforce periodicity at the map edge.")
        print("    Fix: force DEM periodicity by replacing the 5 edge columns on each")
        print("    side with a cosine-tapered blend of the two sides.")
    if not raw_seam_names:
        print("  No seams found in raw source TIFs.")
        print("  → The seam is entirely a filter-mode artifact, already fixed in")
        print("    the updated features.py / preprocessing.py.  Rerun run_pipeline.py.")
else:
    print("No significant seams found in any checked TIF.")
    print("→ All seams should be resolved by the filter-mode fixes in the updated")
    print("  features.py and preprocessing.py. Rerun run_pipeline.py to verify.")
