"""
diagnose_organic.py
===================
Diagnostic script that probes exactly what data reaches _organic_abundance
and saves plot images you can upload to help debug the 180-360° blank.

Run from the project root:
    python diagnose_organic.py

Saves outputs/diagnostics/organic_diag_*.png
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

out = Path('outputs/diagnostics')
out.mkdir(parents=True, exist_ok=True)

processed = Path('data/processed')

print("=" * 60)
print("CANONICAL FILE INVENTORY")
print("=" * 60)

key_files = [
    'vims_mosaic_canonical.tif',
    'iss_mosaic_450m_canonical.tif',
    'geomorphology_canonical.tif',
    'topography_canonical.tif',
    'sar_mosaic_canonical.tif',
]

exists = {}
for f in key_files:
    p = processed / f
    exists[f] = p.exists()
    status = f"EXISTS  ({p.stat().st_size/1024/1024:.1f} MB)" if p.exists() else "MISSING"
    print(f"  {'✓' if p.exists() else '✗'}  {f:<45} {status}")

import rasterio

def load(name: str) -> "np.ndarray | None":
    p = processed / f'{name}_canonical.tif'
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        arr: np.ndarray = src.read(1).astype(np.float64)
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
    return arr

print("\n" + "=" * 60)
print("VIMS COVERAGE ANALYSIS")
print("=" * 60)
vims = load('vims_mosaic')
if vims is not None:
    nrows, ncols = vims.shape
    half = ncols // 2
    left_valid  = np.isfinite(vims[:, :half]).mean()
    right_valid = np.isfinite(vims[:, half:]).mean()
    print(f"  Shape: {vims.shape}")
    print(f"  Left  half (0-180°W)  valid: {left_valid:.1%}")
    print(f"  Right half (180-360°W) valid: {right_valid:.1%}")
    
    # Sample values around the boundary
    eq_rows = slice(int(nrows*0.45), int(nrows*0.55))
    print(f"\n  Values at equator around 180°W boundary:")
    for c in range(half-3, half+3):
        lon = 360.0 * c / ncols
        v = float(np.nanmean(vims[eq_rows, c])) if np.any(np.isfinite(vims[eq_rows, c])) else float('nan')
        flag = " ← BOUNDARY" if abs(c - half) <= 1 else ""
        print(f"    col {c:4d}  {lon:6.1f}°W  VIMS={v:.4f}{flag}")
else:
    print("  VIMS mosaic NOT FOUND in data/processed/")

print("\n" + "=" * 60)
print("GEOMORPHOLOGY ANALYSIS")
print("=" * 60)
geo = load('geomorphology')
if geo is not None:
    print(f"  Shape: {geo.shape}")
    classes, counts = np.unique(geo[np.isfinite(geo)], return_counts=True)
    total = np.isfinite(geo).sum()
    for cls, cnt in zip(classes, counts):
        print(f"    Class {int(cls)}: {cnt:>8,}  ({100*cnt/total:.1f}%)")
    right_valid = np.isfinite(geo[:, geo.shape[1]//2:]).mean()
    print(f"  Right half (180-360°W) valid: {right_valid:.1%}")
else:
    print("  ✗ geomorphology_canonical.tif NOT FOUND")
    print("    This is the geomorphology shapefile from JPL (Rosaly Lopes).")
    print("    Without this file, the geo-based gap-fill CANNOT WORK.")
    print("    → The fix needs a fallback that doesn't require this file.")

print("\n" + "=" * 60)
print("WHAT _organic_abundance WOULD RETURN")
print("=" * 60)
if vims is not None and geo is not None:
    print("  Both VIMS and geomorphology available → Option B should work")
elif vims is not None and geo is None:
    print("  ✗ Only VIMS available — right half will be NaN")
    print("    DIAGNOSIS CONFIRMED: geomorphology missing → 180-360° blank")
elif vims is None and geo is not None:
    print("  Only geomorphology available → geo-only path")
else:
    print("  ✗ Neither VIMS nor geomorphology → all NaN")

# ── Generate diagnostic images ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Organic Abundance Diagnostic — Raw Inputs", fontsize=13)

def show(ax: "matplotlib.axes.Axes", arr: "np.ndarray | None", title: str) -> None:
    if arr is None:
        ax.text(0.5, 0.5, 'NOT AVAILABLE', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title(title)
        return
    finite = arr[np.isfinite(arr)]
    vmin = float(np.percentile(finite, 2)) if len(finite) else 0
    vmax = float(np.percentile(finite, 98)) if len(finite) else 1
    im = ax.imshow(arr, origin='upper', extent=[0,360,-90,90],
                   aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    ax.axvline(180, color='cyan', lw=1.5, ls='--', label='180°W')
    ax.set_title(f"{title}\nvalid={np.isfinite(arr).mean():.1%}  "
                 f"range=[{vmin:.3f},{vmax:.3f}]", fontsize=9)
    ax.set_xlabel('Lon (°W)'); ax.set_ylabel('Lat (°)')
    plt.colorbar(im, ax=ax, shrink=0.8)

show(axes[0,0], vims, 'VIMS mosaic (raw)')
show(axes[0,1], geo,  'Geomorphology (raw)')
show(axes[1,0], load('iss_mosaic_450m'), 'ISS 450m mosaic (raw)')
show(axes[1,1], load('topography'), 'Topography (raw)')

plt.tight_layout()
p = out / 'organic_diag_inputs.png'
fig.savefig(p, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved: {p}")

# ── Simulate what _organic_abundance actually returns ─────────────────────────
print("\n" + "=" * 60)
print("SIMULATING _organic_abundance OUTPUT")
print("=" * 60)

if vims is not None:
    from titan.preprocessing import normalise_to_0_1
    vims_f = vims.astype(np.float32)
    vims_norm = normalise_to_0_1(vims_f, 2, 98)
    print(f"  vims_norm: valid={np.isfinite(vims_norm).mean():.1%}  "
          f"right-half valid={np.isfinite(vims_norm[:, vims_norm.shape[1]//2:]).mean():.1%}")

    if geo is not None:
        from titan.features import _geo_class_to_organic
        geo_int = np.where(np.isfinite(geo), geo, 0).astype(np.int32)
        geo_org = _geo_class_to_organic(geo_int)
        result = np.where(np.isfinite(vims_norm), vims_norm, geo_org)
        print(f"  geo_organic: valid={np.isfinite(geo_org).mean():.1%}")
        print(f"  combined:    valid={np.isfinite(result).mean():.1%}")
        
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        fig2.suptitle("Organic Abundance — Computed Values", fontsize=13)
        show(axes2[0], vims_norm, 'VIMS normalised [0-1]')
        show(axes2[1], geo_org,   'Geo organic scores [0-1]')
        show(axes2[2], result,    'Combined (Option B result)')
        plt.tight_layout()
        p2 = out / 'organic_diag_computed.png'
        fig2.savefig(p2, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {p2}")
    else:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Organic Abundance — NO GEOMORPHOLOGY AVAILABLE", fontsize=13)
        show(axes2[0], vims_norm, 'VIMS normalised [0-1]')
        axes2[1].text(0.5, 0.5, 
            'geomorphology_canonical.tif\nNOT FOUND\n\nRight half will be NaN\nwithout this file',
            ha='center', va='center', transform=axes2[1].transAxes,
            fontsize=12, color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes2[1].set_title('Gap-fill source')
        plt.tight_layout()
        p2 = out / 'organic_diag_computed.png'
        fig2.savefig(p2, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {p2}")

print("\nDone. Please upload organic_diag_inputs.png and organic_diag_computed.png")
