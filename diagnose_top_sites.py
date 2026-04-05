#!/usr/bin/env python3
"""
diagnose_top_sites.py
=====================
Find the true top N habitable locations from the current PRESENT posterior,
cluster them spatially to avoid returning 100 pixels all from Kraken Mare,
and compare against the hardcoded TOP10 list in generate_temporal_maps.py.

Run from project root:
    python diagnose_top_sites.py
"""
import sys, math
from pathlib import Path
import numpy as np

NROWS, NCOLS = 1802, 3603
DEG_PER_ROW  = 180.0 / NROWS
DEG_PER_COL  = 360.0 / NCOLS
MIN_CLUSTER_SEP_DEG = 8.0   # minimum separation between reported sites

def row_col_to_lat_lon(r, c):
    lat =  90.0 - (r + 0.5) * DEG_PER_ROW
    lon =         (c + 0.5) * DEG_PER_COL
    return lat, lon

def angular_sep(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return math.degrees(2 * math.asin(math.sqrt(a)))

NAMED_FEATURES = [
    # Major polar seas and lakes (north)
    ("Kraken Mare",      310.0,  68.0),
    ("Ligeia Mare",       78.0,  79.0),
    ("Punga Mare",       339.0,  85.5),
    ("Jingpo Lacus",     336.3,  73.1),
    ("Hammar Lacus",      93.5,  70.7),
    ("Bolsena Lacus",     12.3,  75.4),
    ("Mackay Lacus",     262.8,  77.5),
    ("Woytchugga Lacus", 218.0,  73.0),
    ("Neagh Lacus",      180.5,  70.3),
    ("Cardiel Lacus",    251.5,  71.7),
    ("Kivu Lacus",       252.0,  68.5),
    ("Uvs Lacus",        199.0,  68.9),
    ("Kayangan Lacus",   173.0,  68.5),
    # South polar lake
    ("Ontario Lacus",    179.0, -72.0),
    # Major craters
    ("Selk Crater",      199.0,   7.0),
    ("Menrva Crater",     87.3,  19.0),
    ("Sinlap Crater",     16.0,  11.3),
    ("Ksa Crater",        65.0,  14.8),
    ("Afekan Crater",    200.0,  26.3),
    ("Guabonito Crater", 145.0,  -11.0),
    ("Nath Crater",      184.0,  -29.0),
    ("Paxsi Crater",     247.0,  -12.0),
    ("Forseti Crater",    13.0,  26.0),
    ("Momoy Crater",     113.0,   6.4),
    # Dune and organic-rich regions
    ("Shangri-La",       155.0,  -5.0),
    ("Belet",            250.0,   5.0),
    ("Senkyo",           330.0, -15.0),
    ("Aztlan",           220.0, -20.0),
    ("Aaru",             338.0,  -8.0),
    ("Adiri",            210.0,  -7.0),
    ("Fensal",            30.0,  15.0),
    ("Ching-Tu",         296.0,  -5.0),
    # Geologically complex / cryovolcanic candidate regions
    ("Xanadu",           100.0,  -5.0),
    ("Hotei Regio",       78.0, -20.0),
    ("Tui Regio",        126.0, -22.0),
    ("Sotra Patera",     164.0, -14.0),  # Lopes 2007 cryovolc. candidate
    ("Doom Mons",        163.0, -14.7),
    ("Erebor Mons",      172.0, -19.0),
    ("Tortola Facula",    30.0,   5.0),
    # Landing / flyby reference sites
    ("Huygens Landing",  192.3, -10.6),
    ("Dragonfly Target", 199.0,   7.0),  # Selk region
]

def nearest_feature(lat, lon):
    best_name, best_sep = "unknown", 999.0
    for name, flon, flat in NAMED_FEATURES:
        sep = angular_sep(lat, lon, flat, flon)
        if sep < best_sep:
            best_sep, best_name = sep, name
    return best_name, best_sep

try:
    from scipy.ndimage import uniform_filter
    _have_scipy = True
except ImportError:
    _have_scipy = False

N_CANDIDATES = 5000
N_REPORT     = 10

EPOCHS = ["past", "lake_formation", "present", "near_future", "future"]

all_results = {}   # epoch -> list of (lat, lon, prob, prob_sm, nearest, sep)

for epoch in EPOCHS:
    post_path = Path(f"outputs/{epoch}/inference/posterior_mean.npy")
    if not post_path.exists():
        print(f"SKIP {epoch}: {post_path} not found")
        continue

    posterior = np.load(post_path).reshape(NROWS, NCOLS)

    if _have_scipy:
        smoothed = uniform_filter(np.nan_to_num(posterior), size=30)
    else:
        smoothed = np.nan_to_num(posterior)

    flat_idx = np.argsort(smoothed.ravel())[-N_CANDIDATES:][::-1]
    rows_idx = flat_idx // NCOLS
    cols_idx = flat_idx  % NCOLS

    sites = []
    used_features: set = set()

    for r, c in zip(rows_idx, cols_idx):
        if len(sites) >= N_REPORT:
            break
        lat, lon = row_col_to_lat_lon(r, c)
        prob    = float(posterior[r, c])
        prob_sm = float(smoothed[r, c])
        # Must be at least MIN_CLUSTER_SEP_DEG from every accepted site
        too_close = any(
            angular_sep(lat, lon, slat, slon) < MIN_CLUSTER_SEP_DEG
            for slat, slon, *_ in sites
        )
        if too_close:
            continue
        # Nearest named feature must not already appear in the list
        fname, fsep = nearest_feature(lat, lon)
        if fname in used_features:
            continue
        used_features.add(fname)
        sites.append((lat, lon, prob, prob_sm, fname, fsep))

    all_results[epoch] = sites

# ── Print results ─────────────────────────────────────────────────────────────
EPOCH_LABELS = {
    "past":           "PAST  (~3.5 Gya, LHB)",
    "lake_formation": "LAKE FORMATION  (~1.0 Gya)",
    "present":        "PRESENT  (Cassini era)",
    "near_future":    "NEAR FUTURE  (+250 Myr)",
    "future":         "FUTURE  (~6 Gya, Red Giant)",
}

for epoch in EPOCHS:
    if epoch not in all_results:
        continue
    label = EPOCH_LABELS[epoch]
    sites = all_results[epoch]
    print()
    print("=" * 80)
    print(f"TOP 10  |  {label}")
    print(f"  {'#':>2}  {'lat':>6}  {'lon_W':>6}  {'P(hab)':>7}  {'Nearest feature':>22}  {'sep':>5}")
    print("-" * 80)
    for i, (lat, lon, prob, prob_sm, fname, fsep) in enumerate(sites, 1):
        print(f"  {i:2d}  {lat:6.1f}  {lon:6.1f}  {prob:7.4f}  {fname:>22}  {fsep:4.1f}°")

# ── Stability table across epochs ─────────────────────────────────────────────
print()
print("=" * 80)
print("STABILITY TABLE  (which features appear across multiple epochs)")
print("=" * 80)
feature_counts: dict = {}
for epoch, sites in all_results.items():
    for lat, lon, prob, prob_sm, fname, fsep in sites:
        feature_counts[fname] = feature_counts.get(fname, 0) + 1
print(f"  {'Feature':>24}  {'epochs in top 10':>16}")
print("-" * 45)
for fname, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
    bar = "█" * count
    print(f"  {fname:>24}  {bar:<5}  ({count}/{len(all_results)})")
