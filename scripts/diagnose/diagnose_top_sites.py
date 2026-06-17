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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.pipeline_config import PipelineConfig
from configs.site_catalogue import get_site, get_coords, sites_by_type, SITES

NROWS, NCOLS = PipelineConfig().canonical_grid_shape
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

# Named features for labelling discovered top-posterior pixels.
# The catalogue-backed entries are derived from configs.site_catalogue (the
# single source of truth) so they cannot drift out of sync with the pipeline's
# site coordinates -- e.g. after the north-polar lake registration fix.  The
# supplement adds large equatorial albedo/montes regions that are not in the
# site catalogue (their approximate centres suffice for nearest-feature labels).
_EXTRA_FEATURES = [
    ("Senkyo",          330.0, -15.0),
    ("Aaru",            338.0,  -8.0),
    ("Adiri",           210.0,  -7.0),
    ("Ching-Tu",        296.0,  -5.0),
    ("Tui Regio",       126.0, -22.0),
    ("Doom Mons",       163.0, -14.7),
    ("Erebor Mons",     172.0, -19.0),
    ("Tortola Facula",   30.0,   5.0),
]
NAMED_FEATURES = [(s.full_name, s.lon_W, s.lat) for s in SITES] + _EXTRA_FEATURES

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

# -- Print results -------------------------------------------------------------
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

# -- Stability table across epochs ---------------------------------------------
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
