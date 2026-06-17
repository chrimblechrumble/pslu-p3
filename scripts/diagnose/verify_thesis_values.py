#!/usr/bin/env python3
"""
Run this locally after the VIMS alignment fix to:
  1. Verify Selk feature table (Table 3.6) values
  2. Recompute feature importances (Table 3.5)
  3. Check Freeman feature values

Usage:
  python verify_thesis_values.py \
      outputs/present/features/titan_features_present.nc \
      outputs/present/inference/posterior_analytical.npy
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from numpy.linalg import lstsq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.temporal_config import TemporalMode, get_prior_set
from configs.site_catalogue import get_coords

nc_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/present/features/titan_features_present.nc"
npy_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/present/inference/posterior_analytical.npy"

ds = xr.open_dataset(nc_path)
post = np.load(npy_path)

_priors = get_prior_set(TemporalMode.PRESENT)
features = list(_priors.feature_names)
weights = list(_priors.weights)
h, w = post.shape

def sample(arr, lon_w, lat, r=1):
    col = int(round(lon_w / 360.0 * w)) % w
    row = int(round((90.0 - lat) / 180.0 * h))
    row = max(0, min(h-1, row))
    r0, r1 = max(0, row-r), min(h, row+r+1)
    c0, c1 = max(0, col-r), min(w, col+r+1)
    return np.nanmean(arr[r0:r1, c0:c1])

# ============================================================
# 1. SELK TABLE (Table 3.6) — paste these into thesis if changed
# ============================================================
print("=" * 70)
print("TABLE 3.6: SELK FEATURE VALUES AND GLOBAL MEDIANS")
print("=" * 70)
print(f"\n{'Feature':<30s}  {'Selk':>8s}  {'GlobalMed':>10s}")
print("-" * 52)
_selk_lon, _selk_lat = get_coords("Selk")
for feat in features:
    arr = ds[feat].values
    selk_val = sample(arr, _selk_lon, _selk_lat)
    glob_med = np.nanmedian(arr)
    print(f"  {feat:<28s}  {selk_val:8.3f}  {glob_med:10.3f}")

# ============================================================
# 2. FEATURE IMPORTANCES (Table 3.5)
# ============================================================
print(f"\n{'=' * 70}")
print("TABLE 3.5: FEATURE IMPORTANCES")
print("=" * 70)

step = 4
rows = np.arange(0, h, step)
cols = np.arange(0, w, step)
rr, cc = np.meshgrid(rows, cols, indexing='ij')
X = np.column_stack([ds[f].values[rr.ravel(), cc.ravel()] for f in features])
y = post[rr.ravel(), cc.ravel()]
y_bin = (y > np.median(y)).astype(int)

cal_gnb = CalibratedClassifierCV(GaussianNB(), cv=5, method='isotonic')
cal_gnb.fit(X, y_bin)
baseline = cal_gnb.score(X, y_bin)

imps = []
for i in range(8):
    X_rep = X.copy()
    X_rep[:, i] = np.mean(X[:, i])
    imps.append(max(0, baseline - cal_gnb.score(X_rep, y_bin)))

total = sum(imps)
print(f"\nBaseline accuracy: {baseline:.4f}")
print(f"\n{'Feature':<30s}  {'w_i':>5s}  {'Importance':>10s}  {'Ratio':>6s}")
print("-" * 56)
for i, feat in enumerate(features):
    n = imps[i]/total if total else 0
    print(f"  {feat:<28s}  {weights[i]:5.2f}  {n:10.3f}  {n/weights[i]:6.2f}")

# OLS R²
A_mat = np.column_stack([np.ones(len(y)), X])
coeffs, _, _, _ = lstsq(A_mat, y, rcond=None)
r2 = 1 - np.var(y - A_mat @ coeffs) / np.var(y)
print(f"\nOLS R² = {r2:.6f}")

# ============================================================
# 3. FREEMAN FEATURE VALUES
# ============================================================
print(f"\n{'=' * 70}")
print("FREEMAN FEATURE VALUES")
print("=" * 70)
_freeman_lon, _freeman_lat = get_coords("Freeman")
for feat in features:
    val = sample(ds[feat].values, _freeman_lon, _freeman_lat)
    print(f"  {feat:<28s}  {val:.4f}")

print(f"\n{'=' * 70}")
print("COPY THE OUTPUT ABOVE AND PASTE IT TO CLAUDE")
print("=" * 70)