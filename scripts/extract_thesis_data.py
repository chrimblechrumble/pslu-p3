"""
extract_thesis_data.py
======================
Run this from your pipeline root AFTER the pipeline rerun:

    python extract_thesis_data.py

Produces  thesis_data.json  (~2 KB) in the current directory.
Upload that file and I will rewrite the tex files with the correct numbers.
"""

import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.temporal_config import TemporalMode, get_prior_set, PRESENT_FEATURES
from configs.pipeline_config import BayesianPriorConfig, PipelineConfig
from configs.site_catalogue import get_coords as _get_coords

OUTPUTS   = Path("outputs")           # adjust if different

_cfg = PipelineConfig()
NROWS, NCOLS = _cfg.canonical_grid_shape
PIX_M     = _cfg.canonical_res_m

_priors = get_prior_set(TemporalMode.PRESENT)
FEATURE_NAMES = list(_priors.feature_names)
WEIGHTS = _priors.as_weight_dict()
PRIOR_MEANS = _priors.as_mean_dict()
KAPPA  = BayesianPriorConfig().beta_concentration_default
LAMBDA = BayesianPriorConfig().likelihood_sharpness

# Sites: (lon_W_deg, lat_deg, radius_km)
# Coordinates from configs.site_catalogue; radii are script-specific sampling areas.
def _site(name, radius):
    lon_W, lat = _get_coords(name)
    return (lon_W, lat, radius)

SITES = {
    "Selk":          _site("Selk",      45),
    "Kraken_S1":     (*_get_coords("Kraken S"),  25),
    "Kraken_S2":     (315.0,  71.0, 25),   # offset sub-sample, no catalogue entry
    "Kraken_S3":     (308.0,  73.5, 25),   # offset sub-sample, no catalogue entry
    "Ligeia_E1":     ( 80.0,  79.0, 25),   # offset sub-sample, no catalogue entry
    "Ligeia_E2":     ( 85.0,  78.0, 25),   # offset sub-sample, no catalogue entry
    "Huygens":       _site("Huygens",   45),
    "Belet":         _site("Belet",     45),
    "Xanadu_centre": (120.0,   0.0, 45),   # centre point, no catalogue entry
    "Menrva":        _site("Menrva",    45),
    "Sinlap":        _site("Sinlap",    45),
    "Ksa":           _site("Ksa",       25),
    "Hano":          _site("Hano",      25),
    # Dragonfly traverse samples (fine-grained offsets, not in catalogue)
    "Belet_dune":    (249.0,   7.5, 15),
    "Selk_ejecta":   (199.5,   7.8, 20),
    "Selk_floor":    (199.0,   6.8, 15),
}

def lonlat_to_rc(lon_W, lat):
    r = int(round((90.0 - lat) / 180.0 * NROWS))
    c = int(round(lon_W / 360.0 * NCOLS))
    return max(0, min(NROWS-1, r)), max(0, min(NCOLS-1, c))

def site_stats(arr, lon_W, lat, radius_km):
    r_px = max(1, int(round(radius_km * 1000 / PIX_M)))
    r, c  = lonlat_to_rc(lon_W, lat)
    patch = arr[max(0,r-r_px):r+r_px+1, max(0,c-r_px):c+r_px+1]
    finite = patch[np.isfinite(patch)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(np.median(finite)), float(np.mean(finite))

def epoch_stats(arr):
    f = arr[np.isfinite(arr)]
    return {
        "mean":   round(float(np.mean(f)),   3),
        "std":    round(float(np.std(f)),    3),
        "p90":    round(float(np.percentile(f, 90)), 3),
        "a_gt_05":round(float(100 * np.mean(f > 0.5)), 1),
        "median": round(float(np.median(f)), 3),
    }

def percentile_of(arr, value):
    f = arr[np.isfinite(arr)]
    return round(float(100 * np.mean(f < value)), 1)

def load_posterior(mode):
    p = OUTPUTS / mode / "inference" / "posterior_mean.npy"
    if not p.exists():
        print(f"  MISSING: {p}", file=sys.stderr)
        return None
    arr = np.load(p).astype(np.float32)
    # mask any sentinel values
    arr[arr <= -9.0] = np.nan
    return arr

def load_features(mode):
    try:
        import xarray as xr
        nc = OUTPUTS / mode / "features" / "features.nc"
        if nc.exists():
            ds = xr.open_dataset(nc)
            return {n: ds[n].values.astype(np.float32)
                    for n in FEATURE_NAMES if n in ds}
    except ImportError:
        pass
    # Fallback: load individual TIFs
    try:
        import rasterio
        tif_dir = OUTPUTS / mode / "features" / "tifs"
        result = {}
        for n in FEATURE_NAMES:
            p = tif_dir / f"{n}.tif"
            if p.exists():
                with rasterio.open(p) as src:
                    arr = src.read(1).astype(np.float32)
                    nd = src.nodata
                    if nd is not None:
                        arr[arr == float(nd)] = np.nan
                    arr[arr <= -9.0] = np.nan
                result[n] = arr
        return result if result else None
    except ImportError:
        return None

def bayesian_ph(w_sum):
    mu0    = sum(PRIOR_MEANS[k] * WEIGHTS[k] for k in FEATURE_NAMES)
    alpha0 = mu0 * KAPPA
    beta0  = (1 - mu0) * KAPPA
    a = alpha0 + LAMBDA * w_sum
    b = beta0  + LAMBDA * (1 - w_sum)
    ph = a / (a + b)
    # 95% equal-tailed credible interval
    try:
        from scipy import stats
        lo = float(stats.beta.ppf(0.025, a, b))
        hi = float(stats.beta.ppf(0.975, a, b))
    except ImportError:
        # Rough normal approximation
        sd = (a * b / ((a+b)**2 * (a+b+1)))**0.5
        lo, hi = ph - 1.96*sd, ph + 1.96*sd
    std = (a * b / ((a+b)**2 * (a+b+1)))**0.5
    return round(ph, 3), round(float(std), 3), round(lo, 3), round(hi, 3)

# --- Main -------------------------------------------------------------------

data = {"epoch_stats": {}, "site_posteriors": {}, "selk_features": {},
        "selk_features_global_median": {}, "selk_bayesian": {},
        "crater_comparison": {}}

print("Loading posteriors...")
modes = ["past", "lake_formation", "present", "near_future", "future"]
posts = {}
for m in modes:
    arr = load_posterior(m)
    if arr is not None:
        data["epoch_stats"][m] = epoch_stats(arr)
        posts[m] = arr
        print(f"  {m:15s}: mean={data['epoch_stats'][m]['mean']:.3f}")

print("\nComputing site posteriors (present epoch)...")
if "present" in posts:
    arr = posts["present"]
    for name, (lon_W, lat, r_km) in SITES.items():
        med, _ = site_stats(arr, lon_W, lat, r_km)
        data["site_posteriors"][name] = round(med, 3)
        pct = percentile_of(arr, med)
        if name == "Selk":
            data["site_posteriors"]["Selk_percentile_present"] = pct

if "past" in posts:
    arr = posts["past"]
    _selk_lon, _selk_lat = _get_coords("Selk")
    med, _ = site_stats(arr, _selk_lon, _selk_lat, 45)
    pct = percentile_of(arr, med)
    data["site_posteriors"]["Selk_past"] = round(med, 3)
    data["site_posteriors"]["Selk_percentile_past"] = pct

# Kraken and Ligeia ranges
if "present" in posts:
    arr = posts["present"]
    kraken_vals = [data["site_posteriors"].get(f"Kraken_S{i}") for i in range(1,4)]
    kraken_vals = [v for v in kraken_vals if v is not None]
    data["site_posteriors"]["kraken_shore_lo"] = round(min(kraken_vals), 3)
    data["site_posteriors"]["kraken_shore_hi"] = round(max(kraken_vals), 3)
    ligeia_vals = [data["site_posteriors"].get(f"Ligeia_E{i}") for i in range(1,3)]
    ligeia_vals = [v for v in ligeia_vals if v is not None]
    data["site_posteriors"]["ligeia_shore_lo"] = round(min(ligeia_vals), 3)
    data["site_posteriors"]["ligeia_shore_hi"] = round(max(ligeia_vals), 3)

print("\nLoading present-epoch features...")
feats = load_features("present")
if feats:
    _selk_lon, _selk_lat = _get_coords("Selk")
    for fi, fname in enumerate(FEATURE_NAMES, 1):
        if fname in feats:
            arr = feats[fname]
            med, _ = site_stats(arr, _selk_lon, _selk_lat, 45)
            data["selk_features"][f"f{fi}"] = round(med, 3)
            glob_med = float(np.nanmedian(arr[np.isfinite(arr)]))
            data["selk_features_global_median"][f"f{fi}"] = round(glob_med, 3)
            print(f"  f{fi} {fname:<30}: Selk={med:.3f}  glob_med={glob_med:.3f}")

    # Bayesian posterior from features
    n_missing = sum(1 for fi in range(1, len(FEATURE_NAMES) + 1)
                    if f"f{fi}" not in data["selk_features"])
    if n_missing > 0:
        print(f"  [WARNING] {n_missing} feature(s) missing from Selk — "
              "w_sum will be lower than expected", file=sys.stderr)
    w_sum = sum(data["selk_features"].get(f"f{fi}", 0) * WEIGHTS[fname]
                for fi, fname in enumerate(FEATURE_NAMES, 1)
                if f"f{fi}" in data["selk_features"])
    ph, std, lo, hi = bayesian_ph(w_sum)
    data["selk_bayesian"] = {"PH": ph, "std": std, "ci_lo": lo, "ci_hi": hi,
                             "w_sum": round(w_sum, 4)}
    print(f"\n  Selk Bayesian P(H) = {ph:.3f} ± {std:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")
else:
    print("  [WARNING] Could not load features — Bayesian CI will not be updated")

# --- Crater comparison table -------------------------------------------------
print("\nCrater comparison table (past + present)...")
crater_sites = {
    "Menrva": (*_get_coords("Menrva"), 45),
    "Sinlap": (*_get_coords("Sinlap"), 45),
    "Ksa":    (*_get_coords("Ksa"),    25),
    "Selk":   (*_get_coords("Selk"),   45),
    "Hano":   (*_get_coords("Hano"),   25),
}
for name, (lon_W, lat, r_km) in crater_sites.items():
    row = {}
    for mode in ("past", "present"):
        if mode in posts:
            med, _ = site_stats(posts[mode], lon_W, lat, r_km)
            row[mode] = round(med, 3)
    data["crater_comparison"][name] = row
    print(f"  {name}: past={row.get('past','?')}  present={row.get('present','?')}")

# --- Write JSON --------------------------------------------------------------
out = Path("thesis_data.json")
out.write_text(json.dumps(data, indent=2))
print(f"\nWritten: {out}  ({out.stat().st_size} bytes)")
print("Upload this file and I will regenerate the tex files.")

