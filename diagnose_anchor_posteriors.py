#!/usr/bin/env python3
"""
diagnose_anchor_posteriors.py
=============================
Diagnose why full_inference frames 000-030 appear much brighter than expected.

Checks the value distribution of every anchor posterior and identifies
whether the bright appearance is caused by:
  (a) Colourbar saturation  — posterior values exceed VMAX=0.65
  (b) Distribution shift    — past/lake_formation posteriors peaked higher than present
  (c) Spatial pattern        — specific regions inflating the median

Run from project root:
    python diagnose_anchor_posteriors.py
"""
import sys
import numpy as np
from pathlib import Path

ANCHORS = [
    ("past",           "outputs/past/inference/posterior_mean.npy",           -3.5),
    ("lake_formation", "outputs/lake_formation/inference/posterior_mean.npy", -1.0),
    ("present",        "outputs/present/inference/posterior_mean.npy",         0.0),
    ("near_future",    "outputs/near_future/inference/posterior_mean.npy",    +0.25),
    ("future",         "outputs/future/inference/posterior_mean.npy",         +6.0),
]

VMIN, VMAX = 0.10, 0.65
NROWS, NCOLS = 1802, 3603

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 80)
print("ANCHOR POSTERIOR DISTRIBUTIONS")
print(f"Colourbar range: VMIN={VMIN}  VMAX={VMAX}")
print("=" * 80)

loaded = {}
for name, path, epoch in ANCHORS:
    p = Path(path)
    if not p.exists():
        print(f"  MISSING: {name}  ({path})")
        continue
    arr = np.load(p).astype(np.float32)
    loaded[name] = (arr, epoch)
    print(f"  Loaded: {name}  shape={arr.shape}")

if "present" not in loaded:
    sys.exit("present anchor required — run: python run_pipeline.py --temporal-mode present")

print()
print(f"{'Anchor':<16} {'p01':>6} {'p10':>6} {'p25':>6} {'median':>8}"
      f"  {'p75':>6} {'p90':>6} {'p99':>6} {'max':>6}"
      f"  {'%>VMAX':>8}  {'%<VMIN':>8}  {'mean':>7}")
print("-" * 100)

for name, path, epoch in ANCHORS:
    if name not in loaded:
        print(f"  {name:<14}  —  (not found)")
        continue
    arr, _ = loaded[name]
    flat = arr.ravel()
    valid = flat[np.isfinite(flat)]
    pcts = np.percentile(valid, [1, 10, 25, 50, 75, 90, 99])
    pct_above = 100.0 * (valid > VMAX).mean()
    pct_below = 100.0 * (valid < VMIN).mean()
    saturated_flag = " *** HIGH SATURATION" if pct_above > 20 else ""
    print(f"  {name:<14}  {pcts[0]:6.3f} {pcts[1]:6.3f} {pcts[2]:6.3f} {pcts[3]:8.3f}"
          f"  {pcts[4]:6.3f} {pcts[5]:6.3f} {pcts[6]:6.3f} {valid.max():6.3f}"
          f"  {pct_above:8.1f}%  {pct_below:8.1f}%  {valid.mean():7.3f}{saturated_flag}")

# ── Spatial breakdown for past and present ────────────────────────────────────
print()
print("=" * 80)
print("SPATIAL BREAKDOWN: past vs present")
print("=" * 80)

def lat_band_stats(arr, label, bands):
    """Print median posterior per latitude band."""
    a = arr.reshape(NROWS, NCOLS)
    lats = np.linspace(90, -90, NROWS, endpoint=False)
    print(f"\n  {label}:")
    print(f"    {'Band':>15}   {'median':>7}  {'mean':>7}  {'p95':>7}  {'% > VMAX':>9}")
    for name_b, lo, hi in bands:
        mask = (lats >= lo) & (lats < hi)
        vals = a[mask, :].ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        pct_sat = 100.0 * (vals > VMAX).mean()
        print(f"    {name_b:>15}   {np.median(vals):7.3f}  {vals.mean():7.3f}"
              f"  {np.percentile(vals, 95):7.3f}  {pct_sat:9.1f}%")

BANDS = [
    ("N. polar (>60N)",  60,  90),
    ("N. mid (30-60N)",  30,  60),
    ("Equator (-30-30)", -30,  30),
    ("S. mid (-60--30)", -60, -30),
    ("S. polar (<-60S)", -90, -60),
]

for name in ["past", "present"]:
    if name in loaded:
        lat_band_stats(loaded[name][0], name, BANDS)

# ── Correlation between past and present ─────────────────────────────────────
if "past" in loaded and "present" in loaded:
    print()
    print("=" * 80)
    print("PIXEL-WISE CORRELATION: past vs present")
    print("=" * 80)
    past_f   = loaded["past"][0].ravel().astype(np.float64)
    present_f = loaded["present"][0].ravel().astype(np.float64)
    valid = np.isfinite(past_f) & np.isfinite(present_f)
    r = np.corrcoef(past_f[valid], present_f[valid])[0, 1]
    diff = (past_f[valid] - present_f[valid])
    print(f"  Pearson r(past, present) = {r:.4f}")
    print(f"  Mean(past − present)     = {diff.mean():+.4f}  (positive = past brighter)")
    print(f"  Median(past − present)   = {np.median(diff):+.4f}")
    print(f"  % pixels where past > present: {100*(diff > 0).mean():.1f}%")
    print()
    if diff.mean() > 0.05:
        print("  DIAGNOSIS: past posterior is systematically higher than present.")
        print("  This inflates the colourbar and causes the bright yellow appearance.")
        print("  Likely cause: past-mode features produce less spatial contrast,")
        print("  pushing the classifier toward the prior mean (~0.5), which sits")
        print("  within the yellow-orange colourbar band.")
    elif r < 0.5:
        print("  DIAGNOSIS: past and present posteriors have LOW spatial correlation.")
        print("  The spatial patterns are fundamentally different (expected for")
        print("  different feature sets). Consider whether PCHIP interpolation")
        print("  between these is scientifically appropriate.")
    else:
        print("  DIAGNOSIS: past and present posteriors are correlated and similar.")
        print("  Brightness difference may be within expected range.")

# ── PCHIP interpolation preview ───────────────────────────────────────────────
if "past" in loaded and "lake_formation" in loaded and "present" in loaded:
    print()
    print("=" * 80)
    print("PCHIP INTERPOLATION PREVIEW: median posterior at key interpolated epochs")
    print("=" * 80)
    try:
        from scipy.interpolate import PchipInterpolator
        epochs = [-3.5, -1.0, 0.0, 0.25]
        arrs = [loaded[n][0].ravel().astype(np.float64) for n in
                ["past", "lake_formation", "present", "near_future"]
                if n in loaded]
        if len(arrs) >= 2:
            stacked = np.stack(arrs, axis=0)
            interp  = PchipInterpolator(epochs[:len(arrs)], stacked, axis=0)
            test_ts = [-3.8, -3.5, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.25]
            print(f"  {'t (Gya)':>10}  {'median':>8}  {'mean':>8}  {'% > VMAX':>10}")
            for t in test_ts:
                if epochs[0] <= t <= epochs[len(arrs)-1]:
                    v = interp(t)
                    valid = v[np.isfinite(v)]
                    pct_sat = 100.0 * (valid > VMAX).mean()
                    print(f"  {t:10.2f}  {np.median(valid):8.4f}  "
                          f"{valid.mean():8.4f}  {pct_sat:10.1f}%")
    except ImportError:
        print("  scipy not available — skipping PCHIP preview")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
if "past" in loaded and "present" in loaded:
    past_med   = np.nanmedian(loaded["past"][0])
    present_med = np.nanmedian(loaded["present"][0])
    shift = past_med - present_med
    if shift > 0.05:
        new_vmax = min(0.90, VMAX + shift + 0.05)
        print(f"  The past posterior median is {shift:+.3f} higher than present.")
        print(f"  Consider raising VMAX from {VMAX:.2f} to ~{new_vmax:.2f} for")
        print(f"  full_inference mode, or normalise anchor posteriors to a common")
        print(f"  quantile range before PCHIP interpolation.")
    else:
        print(f"  Posterior medians are similar (shift={shift:+.4f}).")
        print(f"  The bright appearance may reflect genuine LHB-era habitability.")
        print(f"  Consider whether the colourbar range is appropriate.")
