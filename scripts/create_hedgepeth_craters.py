#!/usr/bin/env python3
"""Create hedgepeth_craters.gpkg from the complete Hedgepeth et al. (2020) Table 1.

Reference: Hedgepeth, J.E. et al. (2020). Titan's impact crater population
after Cassini. Icarus 344, 113664.

All 90 craters from Table 1. Coordinates are IAU west-positive longitude,
planetocentric latitude. Certainty: 1=certain, 2=nearly certain, 3=probable,
4=possible.

Run:  python scripts/create_hedgepeth_gpkg.py
Output: data/raw/hedgepeth_craters.gpkg
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# All 90 craters from Hedgepeth et al. (2020) Table 1
# (id, name, lon_W_deg, lat_deg, diameter_km, certainty)
CRATERS = [
    ( 1, "Menrva",              87.0,   20.0, 400, 1),
    ( 2, "Paxsi",              341.5,    5.5, 130, 2),
    ( 3, "Forseti",             10.7,   25.9, 125, 1),
    ( 4, "Afekan",             200.3,   26.0, 115, 1),
    ( 5, "Hano",               345.0,   40.5, 105, 1),
    ( 6, "Sinlap",              16.0,   11.5,  88, 1),
    ( 7, "Soi",                140.9,   24.3,  85, 1),
    ( 8, "Selk",               199.1,    6.9,  84, 1),
    ( 9, "Guabonito",          151.7,  -11.3,  72, 2),
    (10, "Crater #1 H",        240.3,   31.9,  70, 3),
    (11, "Crater #26 W",        86.0,   -8.1,  69, 2),
    (12, "Nath",                 7.8,  -30.6,  68, 2),
    (13, "Crater #3 H",        110.8,   -6.2,  67, 3),
    (14, "Crater #4 H",        347.8,    1.5,  67, 2),
    (15, "Crater #49 W",       188.9,  -10.9,  63, 3),
    (16, "Crater #5 H",        164.4,   31.3,  63, 3),
    (17, "Crater #6 H",        129.2,   14.5,  59, 3),
    (18, "Crater #7 H",        138.4,    1.8,  55, 2),
    (19, "Ksa",                 65.3,   13.7,  45, 1),
    (20, "Crater #8 H",        205.1,  -14.7,  43, 2),
    (21, "Crater #47 W",       184.5,   -7.6,  42, 3),
    (22, "Crater #25 W",        88.7,  -11.5,  42, 2),
    (23, "Crater #24 W",       165.1,   -7.8,  41, 1),
    (24, "Crater #9 H",        293.1,   52.1,  40, 3),
    (25, "Momoy",               44.6,   11.7,  40, 1),
    (26, "Crater #11 H",       340.8,   -8.8,  38, 3),
    (27, "Crater #6 NL",       147.6,    2.1,  34, 1),
    (28, "Crater #45 W",        18.6,    8.2,  34, 3),
    (29, "Crater #12 H",       154.0,   19.8,  34, 3),
    (30, "Crater #13 H",       188.4,   21.2,  34, 3),
    (31, "Crater #5 NL",       140.5,   12.1,  32, 2),
    (32, "Crater #43 W",        50.2,   10.6,  32, 3),
    (33, "Crater #23 W",        50.3,   48.0,  31, 2),
    (34, "Crater #46 W",        40.4,  -49.6,  29, 3),
    (35, "Crater #44 W",        62.1,   12.5,  29, 3),
    (36, "Crater #14 H",       280.4,    0.0,  29, 3),
    (37, "Crater #15 H",       353.9,   54.2,  27, 4),
    (38, "Crater #16 H",       349.0,   -6.3,  27, 3),
    (39, "Beag",               169.5,  -34.7,  26, 1),
    (40, "Crater #22 W",        30.0,   11.4,  25, 2),
    (41, "Crater #10 NL",      129.7,  -30.6,  25, 3),
    (42, "Crater #18 H",       207.8,  -13.1,  23, 4),
    (43, "Crater #4 NL",       141.8,   11.0,  23, 2),
    (44, "Crater #42 W",       166.9,  -10.6,  23, 3),
    (45, "Crater #19 H",       127.0,   -3.1,  23, 3),
    (46, "Crater #40 W",        67.3,  -12.4,  21, 3),
    (47, "Crater #21 W",        84.3,  -10.5,  20, 2),
    (48, "Crater #3 NL",       150.7,  -16.3,  20, 2),
    (49, "Crater #20 H",       121.5,  -15.6,  20, 3),
    (50, "Crater #8 NL",       194.8,    0.1,  20, 3),
    (51, "Crater #41 W",       260.2,   29.9,  18, 3),
    (52, "Crater #9 NL",       162.7,   19.2,  18, 3),
    (53, "Crater #21 H",       149.2,  -68.7,  17, 4),
    (54, "Crater #22 H",        62.5,   -9.8,  17, 2),
    (55, "Crater #20 W",        77.9,   -8.0,  17, 2),
    (56, "Crater #23 H",       170.7,   21.8,  16, 4),
    (57, "Crater #24 H",       353.8,   36.6,  16, 3),
    (58, "Crater #17 W",        75.9,  -13.2,  16, 2),
    (59, "Crater #39 W",       129.7,   33.4,  16, 3),
    (60, "Crater #25 H",        73.3,   12.3,  16, 4),
    (61, "Crater #13 W",       309.9,   83.3,  15, 2),
    (62, "Crater #18 W",       250.1,   39.4,  14, 2),
    (63, "Crater #19 W",        74.9,  -13.3,  14, 3),
    (64, "Crater #38 W",        24.5,  -57.1,  13, 3),
    (65, "Crater #26 H",       175.6,   -6.7,  12, 3),
    (66, "Crater #7 NL",       160.2,  -38.3,  11, 3),
    (67, "Crater #37 W",       349.5,   42.0,  11, 3),
    (68, "Crater #27 H",       235.3,  -60.1,  11, 3),
    (69, "Crater #15 W",       249.3,   38.2,  10, 2),
    (70, "Crater #48 W",       173.4,   22.7,  10, 3),
    (71, "Crater #28 H",       171.0,  -37.8,  10, 3),
    (72, "Crater #29 H",       104.6,   67.4,  10, 4),
    (73, "Crater #16 W",       238.9,   82.6,   9, 2),
    (74, "Crater #35 W",       266.9,   23.9,   9, 3),
    (75, "Crater #36 W",       105.1,   63.8,   9, 3),
    (76, "Crater #30 H",       171.4,  -40.0,   9, 3),
    (77, "Crater #34 W",       227.8,   34.8,   9, 3),
    (78, "Crater #7 W",         39.4,  -18.9,   9, 3),
    (79, "Crater #33 W",       324.1,   75.0,   8, 3),  # lat 105.1 in paper is a typo; ~75N from SAR T71
    (80, "Crater #11 W",       343.3,   25.3,   8, 3),
    (81, "Crater #32 W",       229.0,   83.0,   8, 3),
    (82, "Crater #9 W",        254.0,   39.0,   7, 3),
    (83, "Crater #14 W",       350.1,   44.2,   7, 2),
    (84, "Crater #10 W",        43.4,   18.6,   5, 2),
    (85, "Crater #6 W",         38.4,  -17.1,   5, 2),
    (86, "Crater #8 W",         17.1,   14.0,   4, 2),
    (87, "Crater #30 W",        43.2,  -18.9,   4, 3),
    (88, "Crater #29 W",       261.9,   30.9,   4, 3),
    (89, "Crater #12 W",        12.9,  -54.5,   3, 2),
    (90, "Crater #31 W",        45.4,  -28.3,   3, 3),
]


def main():
    out_path = Path("data/raw/hedgepeth_craters.gpkg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid, name, lon_w, lat, diam, cert in CRATERS:
        # GeoPackage convention: east-positive longitude [-180, 180]
        lon_e = (-lon_w) % 360
        if lon_e > 180:
            lon_e -= 360
        rows.append({
            "id": cid,
            "name": name,
            "lon_west": lon_w,
            "lat": lat,
            "diameter_km": diam,
            "certainty": cert,
            "geometry": Point(lon_e, lat),
        })

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(out_path, driver="GPKG")

    print(f"Wrote {len(gdf)} craters to {out_path}")
    print(f"\nCertainty breakdown:")
    for c in [1, 2, 3, 4]:
        n = sum(1 for _, r in gdf.iterrows() if r["certainty"] == c)
        label = {1: "certain", 2: "nearly certain", 3: "probable", 4: "possible"}[c]
        print(f"  {c} ({label}): {n}")
    named = sum(1 for _, r in gdf.iterrows() if not r["name"].startswith("Crater"))
    print(f"\nNamed craters: {named}")
    print(f"Total: {len(gdf)}")
    print(f"\nNote: Crater #33 W (ID 79) has lat=105.1 in Table 1 — likely a")
    print(f"typo. Corrected to 75.0N based on SAR swath T71 coverage.")


if __name__ == "__main__":
    main()
