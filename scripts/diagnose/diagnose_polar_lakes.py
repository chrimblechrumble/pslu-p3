#!/usr/bin/env python3
"""
diagnose_polar_lakes.py
========================
Run from the project root to diagnose the polar_lakes_canonical.tif
and the Birch shapefiles.

    python diagnose_polar_lakes.py

Exit code 0 = plausible data, 1 = problem detected.
"""
import math
import sys
from pathlib import Path
import numpy as np

EXPECTED_LAKE_FRACTION_MAX = 0.05   # lakes should cover << 5% of Titan

problems: list[str] = []

# -- 1. polar_lakes_canonical.tif ---------------------------------------------
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

# -- 2. Birch shapefile inventory ---------------------------------------------
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

# -- 3. Birch shapefile properties --------------------------------------------
print()
print("=" * 60)
print("3. Birch shapefile properties (CRS and bounds)")
print("=" * 60)
print("  NOTE: the Birch+2017 Cornell shapefiles have NO embedded CRS and use")
print("  a polar-stereographic projection (metres) centred on the pole. This")
print("  is the EXPECTED convention; titan/io/shapefile_rasteriser.py inverts")
print("  it in _stereo_to_canonical. Projected metres + CRS=None is correct,")
print("  not a defect -- so we validate the implied latitude coverage instead.")

TITAN_R_M = 2_575_000.0          # Titan sphere radius (matches rasteriser)
# The Birch polar mapping covers the polar terrains down to ~50 deg latitude.
# A radial extent reaching below this lowest latitude signals a corrupt or
# inside-out polygon spanning the whole projection plane.
MIN_POLAR_COVERAGE_LAT = 30.0    # lowest |lat| we still consider "polar"

def stereo_lowest_lat(bounds):
    """Lowest |latitude| covered, from the farthest bbox corner (metres)."""
    rho = max(math.hypot(x, y)
              for x in (bounds[0], bounds[2])
              for y in (bounds[1], bounds[3]))
    colat_deg = math.degrees(2.0 * math.atan2(rho, 2.0 * TITAN_R_M))
    return 90.0 - colat_deg, rho

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
                is_projected = (abs(bounds[0]) > 360 or abs(bounds[2]) > 360
                                or abs(bounds[1]) > 90 or abs(bounds[3]) > 90)
                if not is_projected:
                    # Unexpected for Birch files: the rasteriser assumes metres.
                    print("    Coord type: GEOGRAPHIC degrees -- UNEXPECTED for Birch "
                          "polar files (rasteriser expects stereographic metres)")
                    problems.append(
                        f"{shp.name} is in geographic degrees, but the polar-lake "
                        "rasteriser (_stereo_to_canonical) expects stereographic metres"
                    )
                else:
                    lowest_lat, rho = stereo_lowest_lat(bounds)
                    print("    Coord type: polar-stereographic metres, CRS=None "
                          "(EXPECTED Birch+2017 convention)")
                    print(f"    Implied coverage: down to ~{lowest_lat:.1f} deg latitude "
                          f"(max rho={rho:.0f} m)")
                    if lowest_lat < MIN_POLAR_COVERAGE_LAT:
                        print(f"    *** coverage extends below {MIN_POLAR_COVERAGE_LAT:.0f} "
                              "deg -- possible inside-out/corrupt polygon")
                        problems.append(
                            f"{shp.name} radial extent reaches ~{lowest_lat:.0f} deg "
                            f"latitude (below {MIN_POLAR_COVERAGE_LAT:.0f} deg) -- "
                            "possible inside-out polygon"
                        )
                    else:
                        print("    Coverage is polar [OK]")
                # Geometry area stats. In stereographic metres these are m^2 and
                # legitimately large (paleo-seas span tens of km); report in km^2
                # for readability without flagging size as a defect.
                areas = gdf.geometry.area
                scale = 1e6 if is_projected else 1.0
                unit  = "km^2" if is_projected else "deg^2"
                print(f"    Area stats ({unit}): min={areas.min()/scale:.4f}  "
                      f"max={areas.max()/scale:.4f}  sum={areas.sum()/scale:.4f}")
            except Exception as e:
                print(f"    [ERROR] {e}")
except ImportError:
    print("  geopandas not installed -- run: pip install geopandas")

# -- Summary -------------------------------------------------------------------
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
