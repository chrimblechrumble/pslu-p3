import rasterio
import numpy as np
from pathlib import Path

# Check each feature TIF at the seam
tif_dir = Path("outputs/present/features/tifs")
print(f"{'Feature TIF':<40} {'col-0 mean':>12} {'col-3602 mean':>14} {'|diff|':>10}")
print("-" * 80)
for tif in sorted(tif_dir.glob("*.tif")):
    with rasterio.open(tif) as ds:
        arr = ds.read(1).astype(float)
        arr[arr == ds.nodata] = np.nan
    finite = np.isfinite(arr)
    c0    = float(np.nanmean(arr[:, 0]))
    c3602 = float(np.nanmean(arr[:, 3602]))
    diff  = abs(c0 - c3602)
    flag  = " ← SEAM" if diff > 0.005 else ""
    print(f"  {tif.name:<38}  {c0:>12.5f}  {c3602:>14.5f}  {diff:>10.5f}{flag}")
