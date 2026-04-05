#!/usr/bin/env python3
"""
diagnose_polar_lakes.py
========================
Run from the project root to diagnose the polar_lakes_canonical.tif
and the Birch shapefiles.

    python diagnose_polar_lakes.py

Exit code 0 = plausible data, 1 = problem detected.
"""
import sys
from pathlib import Path
import numpy as np

EXPECTED_LAKE_FRACTION_MAX = 0.05   # lakes should cover << 5% of Titan

problems: list[str] = []

# ── 1. polar_lakes_canonical.tif ─────────────────────────────────────────────
print("=" * 60)
print("1. polar_lakes_canonical.tif")
print("=" * 60)

tif = Path("data/processed/polar_lakes_canonical.tif")
if not tif.exists():
    print("  [MISSING] data/processed/polar_lakes_canonical.tif")
    print("  This means polar_lakes was never rasterised.")
else:
    try:
        import rasterio
        with rasterio.open(tif) as src:
            data = src.read(1)
            total = data.size
            u, c = np.unique(data, return_counts=True)
            print(f"  Shape:   {data.shape}")
            print(f"  Nodata:  {src.nodata}")
            print(f"  Transform: {src.transform}")
            for v, n in zip(u, c):
                names = {0: "NODATA", 1: "FILLED_LAKE", 2: "EMPTY_BASIN"}
                label = names.get(int(v), f"unknown({v})")
                frac = n / total
                flag = "  *** SUSPICIOUS (>5%)" if int(v) == 1 and frac > EXPECTED_LAKE_FRACTION_MAX else ""
                print(f"  class {int(v):2d} ({label}): {n:>10,} px  ({100*frac:.2f}%){flag}")
            filled_frac = int(c[list(u).index(1)]) / total if 1 in u else 0
            if filled_frac > EXPECTED_LAKE_FRACTION_MAX:
                problems.append(
                    f"polar_lakes_canonical.tif has {100*filled_frac:.1f}% FILLED_LAKE "
                    f"(expected <{100*EXPECTED_LAKE_FRACTION_MAX:.0f}%)"
                )
    except Exception as e:
        print(f"  [ERROR] Could not read TIF: {e}")

# ── 2. Birch shapefile inventory ─────────────────────────────────────────────
print()
print("=" * 60)
print("2. Birch shapefile inventory")
print("=" * 60)

birch_root = Path("data/raw/birch_polar_mapping")
for subdir in ["birch_filled", "birch_empty"]:
    subpath = birch_root / subdir
    print(f"\n  {subpath}:")
    if not subpath.exists():
        print("    [MISSING] Directory does not exist")
        continue
    shps = sorted(subpath.glob("*.shp"))
    if not shps:
        print("    [EMPTY] No .shp files found")
        continue
    for shp in shps:
        print(f"    {shp.name}")

# ── 3. Birch shapefile properties ────────────────────────────────────────────
print()
print("=" * 60)
print("3. Birch shapefile properties (CRS and bounds)")
print("=" * 60)

try:
    import geopandas as gpd
    for subdir in ["birch_filled", "birch_empty"]:
        subpath = birch_root / subdir
        if not subpath.exists():
            continue
        for shp in sorted(subpath.glob("*.shp")):
            try:
                gdf = gpd.read_file(shp)
                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                print(f"\n  {shp.name}")
                print(f"    Features:  {len(gdf)}")
                print(f"    CRS:       {gdf.crs}")
                print(f"    Bounds:    xmin={bounds[0]:.4f}  ymin={bounds[1]:.4f}")
                print(f"               xmax={bounds[2]:.4f}  ymax={bounds[3]:.4f}")
                # Detect coordinate system from bounds
                if abs(bounds[0]) <= 360 and abs(bounds[2]) <= 360 and abs(bounds[1]) <= 90:
                    if bounds[0] >= 0 and bounds[2] <= 360:
                        print("    Coord type: likely GEOGRAPHIC 0-360 E or 0-360 W degrees")
                    elif bounds[0] >= -180:
                        print("    Coord type: likely GEOGRAPHIC -180 to +180 E degrees")
                    else:
                        print("    Coord type: UNKNOWN degree-range")
                elif abs(bounds[0]) > 360 or abs(bounds[2]) > 360:
                    print("    Coord type: likely PROJECTED (metres or km) -- NOT geographic degrees!")
                    problems.append(
                        f"{shp.name} appears to be in a projected CRS "
                        f"(xmin={bounds[0]:.0f}, xmax={bounds[2]:.0f}) -- "
                        "the _to_canonical transform expects geographic degrees"
                    )
                # Geometry area stats
                areas = gdf.geometry.area
                print(f"    Area stats: min={areas.min():.4f}  max={areas.max():.4f}  "
                      f"sum={areas.sum():.4f}")
                # Very large features (area > 10 deg^2 = very suspicious for a lake)
                huge = gdf[areas > 1000]
                if not huge.empty:
                    print(f"    *** {len(huge)} features with area > 1000 sq-units -- "
                          "these may be continent-sized polygons")
                    problems.append(
                        f"{shp.name} has {len(huge)} huge polygon(s) "
                        f"(max area {areas.max():.0f}) -- possible inside-out polygon"
                    )
            except Exception as e:
                print(f"    [ERROR] {e}")
except ImportError:
    print("  geopandas not installed -- run: pip install geopandas")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
if problems:
    print(f"  {len(problems)} problem(s) found:")
    for p in problems:
        print(f"  [PROBLEM] {p}")
    sys.exit(1)
else:
    print("  No obvious problems detected.")
    sys.exit(0)
