"""
diagnose_features.py
====================
Run from your pipeline root after the pipeline rerun:

    python diagnose_features.py

Checks whether suspicious feature changes are genuine physics or pipeline bugs.
Focuses on f4 (methane_cycle), f5 (surface_atm), f6 (topo_complexity),
and f8 (subsurface_ocean) at Selk crater.

Paste the full output here and I will update the thesis tex files.
"""

import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    sys.exit("rasterio not installed — run: pip install rasterio")

OUTPUTS  = Path("outputs")
NROWS, NCOLS = 1802, 3603
PIX_M    = 4490.0   # metres per pixel

# Selk crater  (199°W, 7°N)
SELK_LON_W, SELK_LAT = 199.0, 7.0
SELK_ROW = int((90.0 - SELK_LAT) / 180.0 * NROWS)   # ≈ 830
SELK_COL = int(SELK_LON_W / 360.0 * NCOLS)            # ≈ 1991

# Kraken Mare centre  (310°W, 74°N)
KRAKEN_ROW = int((90.0 - 74.0) / 180.0 * NROWS)      # ≈ 160
KRAKEN_COL = int(310.0 / 360.0 * NCOLS)               # ≈ 3101

# Comparison sites
SITES = {
    "Selk (7°N 199°W)":    (SELK_ROW,   SELK_COL),
    "Kraken (74°N 310°W)": (KRAKEN_ROW, KRAKEN_COL),
    "Belet (7°S 250°W)":   (int((90+7)/180*NROWS),  int(250/360*NCOLS)),
    "Ligeia (79°N 82°W)":  (int((90-79)/180*NROWS), int(82/360*NCOLS)),
}

R_PX = 20   # ±20 pixels (~90 km radius) for patch stats


def patch_stats(arr, row, col, r=R_PX):
    p = arr[max(0,row-r):row+r+1, max(0,col-r):col+r+1]
    f = p[np.isfinite(p)]
    if len(f) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(f)), float(np.min(f)), float(np.max(f))


def load_tif(path):
    path = Path(path)
    if not path.exists():
        return None, f"MISSING: {path}"
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nd = src.nodata
        if nd is not None:
            arr[arr == float(nd)] = np.nan
        arr[arr <= -9.0] = np.nan
    return arr, None


def check_tif(label, path, expected_old=None, sites=None):
    arr, err = load_tif(path)
    if arr is None:
        print(f"  {label}: {err}")
        return

    finite = arr[np.isfinite(arr)]
    gmin   = float(np.nanmin(finite))
    gmax   = float(np.nanmax(finite))
    gmed   = float(np.nanmedian(finite))
    gmean  = float(np.nanmean(finite))
    nan_pct= 100 * np.mean(~np.isfinite(arr))

    print(f"  {label}")
    print(f"    global:  min={gmin:.4f}  max={gmax:.4f}  "
          f"median={gmed:.4f}  mean={gmean:.4f}  nan={nan_pct:.1f}%")
    if expected_old is not None:
        print(f"    old thesis value at Selk: {expected_old}")

    if sites:
        for site_name, (r, c) in sites.items():
            med, lo, hi = patch_stats(arr, r, c)
            print(f"    {site_name}: median={med:.4f}  range=[{lo:.4f}, {hi:.4f}]")
    print()


# ═══════════════════════════════════════════════════════════════════════
print("═" * 65)
print("FEATURE DIAGNOSTIC  —  Titan habitability pipeline")
print("═" * 65)

# ── 1. Canonical TIF inputs (check they are sensible) ──────────────────
print("\n[1] CANONICAL INPUT TIFs (preprocessed)")
print("-" * 65)

check_tif("SAR mosaic", OUTPUTS / "preprocessed/sar_mosaic_canonical.tif",
          sites=SITES)

check_tif("Topography (DEM)", OUTPUTS / "preprocessed/topography_canonical.tif",
          sites=SITES)

check_tif("Geomorphology", OUTPUTS / "preprocessed/geomorphology_canonical.tif",
          sites=SITES)

check_tif("Channel density", OUTPUTS / "preprocessed/channel_density_canonical.tif",
          sites=SITES)

check_tif("CIRS temperature", OUTPUTS / "preprocessed/cirs_temperature_canonical.tif",
          sites=SITES)

check_tif("Polar lakes mask", OUTPUTS / "preprocessed/polar_lakes_canonical.tif",
          sites=SITES)

# ── 2. Present-epoch feature TIFs ──────────────────────────────────────
print("\n[2] PRESENT-EPOCH FEATURE TIFs (outputs/present/features/tifs/)")
print("-" * 65)

feat_dir = OUTPUTS / "present/features/tifs"
features = {
    "f1_liquid_hydrocarbon":     ("liquid_hydrocarbon.tif",     0.04,  "was 0.04 at Selk (ok if unchanged)"),
    "f2_organic_abundance":      ("organic_abundance.tif",       0.62,  "was 0.62 — dropped to 0.22 (why?)"),
    "f3_acetylene_energy":       ("acetylene_energy.tif",        0.41,  "was 0.41 — DEM change expected"),
    "f4_methane_cycle":          ("methane_cycle.tif",           0.38,  "was 0.38 — dropped to 0.025 — SUSPICIOUS"),
    "f5_surface_atm_interaction":("surface_atm_interaction.tif", 0.51,  "was 0.51 — dropped to 0.010 — SUSPICIOUS"),
    "f6_topographic_complexity": ("topographic_complexity.tif",  0.44,  "was 0.44 — DEM change expected"),
    "f7_geomorphologic_diversity":("geomorphologic_diversity.tif",0.72, "was 0.72 — minor change ok"),
    "f8_subsurface_ocean":       ("subsurface_ocean.tif",        0.22,  "was 0.22 — dropped to 0.033 — NOT DEM!"),
}

for label, (fname, old_val, note) in features.items():
    check_tif(f"{label}  [{note}]",
              feat_dir / fname,
              expected_old=old_val,
              sites=SITES)

# ── 3. Specific sub-components that feed suspect features ───────────────
print("\n[3] SUB-COMPONENT CHECK (intermediate files if saved)")
print("-" * 65)

# methane_cycle = f(CIRS_temperature, channel_density) essentially
# surface_atm   = f(DEM_slope, methane_cycle, channel_density)
# subsurface_ocean = f(SAR_mosaic annuli around craters)

# Check if intermediate slope TIF exists
for candidate in [
    OUTPUTS / "present/features/intermediate/dem_slope.tif",
    OUTPUTS / "preprocessed/dem_slope_canonical.tif",
    OUTPUTS / "present/intermediate/slope.tif",
]:
    if candidate.exists():
        check_tif("DEM slope (intermediate)", candidate, sites=SITES)
        break
else:
    print("  DEM slope intermediate TIF not found (not saved by pipeline)")
    print()

# Check SAR mosaic statistics near Selk specifically
sar_arr, err = load_tif(OUTPUTS / "preprocessed/sar_mosaic_canonical.tif")
if sar_arr is not None:
    print("  SAR mosaic at Selk (key input to f8):")
    med, lo, hi = patch_stats(sar_arr, SELK_ROW, SELK_COL, r_px=30)
    print(f"    ±30px patch: median={med:.4f}  min={lo:.4f}  max={hi:.4f}")
    # Look for any bright annular ring around Selk (should be ~90-130 km radius)
    for r_km in (45, 90, 120):
        r_px = int(r_km * 1000 / PIX_M)
        m, l, h = patch_stats(sar_arr, SELK_ROW, SELK_COL, r=r_px)
        print(f"    ±{r_km}km ({r_px}px): median={m:.4f}  min={l:.4f}  max={h:.4f}")
    print()

# ── 4. Check if subsurface_ocean feature is near-zero everywhere ────────
print("\n[4] SUBSURFACE OCEAN — is it broken everywhere or just at Selk?")
print("-" * 65)
f8_arr, err = load_tif(feat_dir / "subsurface_ocean.tif")
if f8_arr is not None:
    finite = f8_arr[np.isfinite(f8_arr)]
    # Distribution
    for pct in (50, 90, 95, 99, 99.9):
        print(f"  {pct:5.1f}th percentile = {np.percentile(finite, pct):.4f}")
    n_above_01 = int(np.sum(finite > 0.10))
    print(f"  Pixels > 0.10: {n_above_01:,}  ({100*n_above_01/len(finite):.2f}%)")
    print(f"  Pixels > 0.20: {int(np.sum(finite > 0.20)):,}  "
          f"({100*np.mean(finite > 0.20):.2f}%)")
    print()
    print("  Top-10 highest pixels (should be at crater sites):")
    flat = f8_arr.ravel()
    top_idx = np.argsort(flat)[::-1][:30]
    seen = 0
    for idx in top_idx:
        if not np.isfinite(flat[idx]):
            continue
        row_i = idx // NCOLS
        col_i = idx % NCOLS
        lat_i = 90.0 - row_i / NROWS * 180.0
        lon_i = col_i / NCOLS * 360.0
        print(f"    ({lat_i:+6.1f}°N, {lon_i:5.1f}°W): f8={flat[idx]:.4f}")
        seen += 1
        if seen >= 10:
            break
else:
    print(f"  {err}")

# ── 5. Summary ────────────────────────────────────────────────────────
print("\n" + "═" * 65)
print("WHAT TO LOOK FOR:")
print("═" * 65)
print("""
GOOD: f8 subsurface_ocean has peaks > 0.20 at known crater sites
      (Selk, Menrva, Sinlap, Ksa etc.) → feature is working correctly
BAD:  f8 is near-zero everywhere (max < 0.05) → pipeline bug in
      subsurface_ocean extraction (check if SAR mosaic loaded correctly)

GOOD: f4 methane_cycle shows polar concentration
      (Kraken > 0.6, Selk < 0.1) → feature is physically correct
BAD:  f4 is very low everywhere (global median < 0.1) → bug

GOOD: SAR mosaic shows backscatter variation at Selk (some bright annulus)
BAD:  SAR mosaic is uniform zero at Selk → SAR not loaded or all-zero

Paste this output in the chat and I will identify whether
the feature values in thesis_data.json are reliable.
""")

