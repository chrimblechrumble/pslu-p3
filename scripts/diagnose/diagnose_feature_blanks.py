#!/usr/bin/env python3
"""
diagnose_feature_blanks.py — Investigate f₁ and f₈ blank regions.

Checks whether blank (NaN/zero) pixels in the left hemisphere are from
genuine Cassini coverage gaps or from a processing error.

Usage:
  python diagnose_feature_blanks.py outputs/present/features/titan_features_present.nc
"""
import sys
import numpy as np
import xarray as xr

nc = sys.argv[1] if len(sys.argv) > 1 else "outputs/present/features/titan_features_present.nc"
ds = xr.open_dataset(nc)

features = ["liquid_hydrocarbon", "organic_abundance", "acetylene_energy",
            "methane_cycle", "surface_atm_interaction", "topographic_complexity",
            "geomorphologic_diversity", "subsurface_ocean"]
labels = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]

h, w = ds[features[0]].shape
col_180 = w // 2

print("=" * 70)
print("FEATURE BLANK DIAGNOSTICS")
print("=" * 70)
print(f"Grid: {h}×{w} pixels\n")

print(f"{'Feature':<28s} {'Left NaN%':>10s} {'Right NaN%':>10s} "
      f"{'Left Zero%':>10s} {'Right Zero%':>10s} {'Verdict':>10s}")
print("-" * 82)

for feat, label in zip(features, labels):
    arr = ds[feat].values
    left = arr[:, :col_180]
    right = arr[:, col_180:]

    left_nan = np.isnan(left).mean() * 100
    right_nan = np.isnan(right).mean() * 100
    left_zero = (left == 0).mean() * 100
    right_zero = (right == 0).mean() * 100

    # Verdict
    if abs(left_nan - right_nan) > 10:
        verdict = "ASYM NaN"
    elif abs(left_zero - right_zero) > 10:
        verdict = "ASYM Zero"
    else:
        verdict = "OK"

    print(f"  {label}: {feat:<24s} {left_nan:10.1f} {right_nan:10.1f} "
          f"{left_zero:10.1f} {right_zero:10.1f} {verdict:>10s}")

# Detailed check for f1 and f8
print(f"\n{'=' * 70}")
print("DETAILED: f1 (liquid_hydrocarbon) by longitude strip")
print("=" * 70)
f1 = ds["liquid_hydrocarbon"].values
for lon_start in range(0, 360, 30):
    c0 = int(lon_start / 360 * w)
    c1 = int((lon_start + 30) / 360 * w)
    strip = f1[:, c0:c1]
    nan_pct = np.isnan(strip).mean() * 100
    zero_pct = (strip == 0).mean() * 100
    valid_pct = (np.isfinite(strip) & (strip > 0)).mean() * 100
    print(f"  {lon_start:3d}-{lon_start + 30:3d}°W: NaN={nan_pct:5.1f}%  "
          f"Zero={zero_pct:5.1f}%  Valid>0={valid_pct:5.1f}%")

print(f"\n{'=' * 70}")
print("DETAILED: f8 (subsurface_ocean) by longitude strip")
print("=" * 70)
f8 = ds["subsurface_ocean"].values
for lon_start in range(0, 360, 30):
    c0 = int(lon_start / 360 * w)
    c1 = int((lon_start + 30) / 360 * w)
    strip = f8[:, c0:c1]
    nan_pct = np.isnan(strip).mean() * 100
    zero_pct = (strip == 0).mean() * 100
    valid_pct = (np.isfinite(strip) & (strip > 0)).mean() * 100
    print(f"  {lon_start:3d}-{lon_start + 30:3d}°W: NaN={nan_pct:5.1f}%  "
          f"Zero={zero_pct:5.1f}%  Valid>0={valid_pct:5.1f}%")

# Check if f8 has the k2 floor everywhere
print(f"\n{'=' * 70}")
print("f8: Does the k₂=0.589 floor (value ~0.030) cover the full globe?")
print("=" * 70)
f8_finite = f8[np.isfinite(f8)]
near_floor = ((f8_finite >= 0.025) & (f8_finite <= 0.035)).sum()
print(f"  Total finite pixels: {len(f8_finite)}")
print(f"  Pixels near floor (0.025-0.035): {near_floor} ({near_floor / len(f8_finite) * 100:.1f}%)")
print(f"  Total NaN pixels: {np.isnan(f8).sum()} ({np.isnan(f8).mean() * 100:.1f}%)")
if np.isnan(f8).mean() > 0.01:
    print(f"  ⚠ f8 has significant NaN coverage — the k₂ floor should fill ALL pixels")
    print(f"  → Check that _subsurface_ocean() applies the k₂ floor AFTER SAR annuli,")
    print(f"    not only WHERE SAR coverage exists.")

# f3 seam check
print(f"\n{'=' * 70}")
print("f3: Discontinuity at 180°W quantification")
print("=" * 70)
f3 = ds["acetylene_energy"].values
for lat_label, r0, r1 in [
    ("70-80°N", int(10 / 180 * h), int(20 / 180 * h)),
    ("0-10°N", int(80 / 180 * h), int(90 / 180 * h)),
    ("30-40°S", int(120 / 180 * h), int(130 / 180 * h)),
]:
    left = np.nanmean(f3[r0:r1, col_180 - 10:col_180])
    right = np.nanmean(f3[r0:r1, col_180:col_180 + 10])
    print(f"  {lat_label}: left={left:.3f}  right={right:.3f}  |step|={abs(right - left):.3f}")

print(f"\n{'=' * 70}")
print("PASTE THIS OUTPUT TO CLAUDE")
print("=" * 70)
