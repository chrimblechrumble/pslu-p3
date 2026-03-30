"""
diagnose_geomorphology.py
=========================
Probes the geomorphology_canonical.tif to determine whether terrain classes
land in the correct geographic positions, and checks the raw shapefile
coordinate conventions.

Run from the project root:
    python diagnose_geomorphology.py

Saves: outputs/diagnostics/geo_diag_*.png and geo_diag_report.txt
"""
import sys, math
sys.path.insert(0, '.')

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path('outputs/diagnostics')
out_dir.mkdir(parents=True, exist_ok=True)
report = []
R = lambda s: (report.append(s), print(s))

# ── 1. Read the geomorphology raster ─────────────────────────────────────────
import rasterio
geo_path = Path('data/processed/geomorphology_canonical.tif')
if not geo_path.exists():
    print("ERROR: geomorphology_canonical.tif not found")
    sys.exit(1)

with rasterio.open(geo_path) as src:
    geo = src.read(1)
    nrows, ncols = geo.shape
    transform = src.transform

R("=" * 60)
R("GEOMORPHOLOGY RASTER PROBE")
R("=" * 60)
R(f"Shape: {nrows} rows × {ncols} cols")
R(f"Transform: {transform}")

# Class distribution
classes, counts = np.unique(geo, return_counts=True)
R("\nClass distribution:")
names = {0:'NoData', 1:'Craters', 2:'Dunes', 3:'Plains',
         4:'Basins', 5:'Mountains', 6:'Labyrinth', 7:'Lakes'}
for cls, cnt in zip(classes, counts):
    R(f"  Class {cls} ({names.get(int(cls),'?'):12s}): {cnt:>10,}  ({100*cnt/geo.size:.1f}%)")

# ── 2. Check known feature locations ─────────────────────────────────────────
R("\n" + "=" * 60)
R("KNOWN FEATURE LOCATION CHECK")
R("=" * 60)
R("Checking where dunes appear vs where they should be.")
R("Equatorial dune belt known features:")
R("  Belet:      ~250°W, equator (class 2 = Dunes)")
R("  Shangri-La: ~155°W, equator (class 2 = Dunes)")
R("  Aztlan:     ~315°W, equator (class 2 = Dunes)")
R("")

# Find the equatorial band (lat ±15°)
lat_centres = np.linspace(90 - 0.5*180/nrows, -90 + 0.5*180/nrows, nrows)
lon_centres  = np.linspace(0.5*360/ncols, 360 - 0.5*360/ncols, ncols)
eq_rows = np.where(np.abs(lat_centres) < 15)[0]

# Where are dunes in the equatorial belt?
dune_cols_eq = np.where(geo[eq_rows[0]:eq_rows[-1]+1, :] == 2)
if len(dune_cols_eq[1]) > 0:
    dune_lons = lon_centres[dune_cols_eq[1]]
    R(f"Dune (class 2) pixels in ±15° equatorial belt:")
    R(f"  Count: {len(dune_lons)}")
    R(f"  Longitude range: {dune_lons.min():.1f}°W – {dune_lons.max():.1f}°W")
    
    # Check specific longitude bands
    for band_name, lo, hi in [
        ("Shangri-La zone (140-170°W)", 140, 170),
        ("Belet zone      (230-265°W)", 230, 265),
        ("Aztlan zone     (295-330°W)", 295, 330),
    ]:
        n_in_band = int(np.sum((dune_lons >= lo) & (dune_lons < hi)))
        R(f"  {band_name}: {n_in_band} dune pixels")
    
    # Check if 180-360 is a mirror of 0-180
    left_hist  = np.histogram(dune_lons[dune_lons < 180],  bins=18, range=(0, 180))[0]
    right_hist = np.histogram(dune_lons[dune_lons >= 180], bins=18, range=(180, 360))[0]
    right_mirrored = right_hist[::-1]
    corr = np.corrcoef(left_hist, right_mirrored)[0,1]
    R(f"\n  Mirror test: correlation(left, mirror(right)) = {corr:.4f}")
    R(f"  (>0.9 = strong mirror; ~0 = independent = correct)")
    if corr > 0.85:
        R(f"  *** MIRROR CONFIRMED: right half is ~reflection of left ***")
        R(f"  *** Shapefile coordinate convention is WRONG in rasteriser ***")
    else:
        R(f"  No strong mirror detected — coordinate conversion likely correct")
else:
    R("WARNING: No dune pixels found in equatorial belt!")

# ── 3. Check raw shapefile coordinates ────────────────────────────────────────
R("\n" + "=" * 60)
R("RAW SHAPEFILE COORDINATE CHECK")
R("=" * 60)

shp_dir = Path('data/raw/geomorphology_shapefiles')
dunes_shp = shp_dir / 'Dunes.shp'

if dunes_shp.exists():
    try:
        import geopandas as gpd
        gdf = gpd.read_file(dunes_shp)
        geom = gdf.geometry.iloc[0]
        coords = list(geom.exterior.coords)[:5]
        R(f"Dunes.shp first polygon, first 5 vertices:")
        for lon, lat, *_ in coords:
            R(f"  ({lon:.4f}, {lat:.4f})")
        all_lons = []
        for geom in gdf.geometry:
            if geom is not None and not geom.is_empty:
                try:
                    all_lons.extend([c[0] for c in geom.exterior.coords])
                except: pass
        if all_lons:
            R(f"\nDunes.shp longitude range: {min(all_lons):.2f} to {max(all_lons):.2f}")
            if min(all_lons) < -90:
                R("→ Coordinates are EAST-positive (-180 to +180) ✓")
            elif max(all_lons) > 180:
                R("→ Coordinates are in 0-360 range")
                if min(all_lons) >= 0:
                    R("→ This looks WEST-positive (0-360°W) — current formula negates wrongly!")
            else:
                R("→ Coordinates in 0-180 range (ambiguous)")
    except ImportError:
        R("geopandas not available for raw shapefile check")
    except Exception as e:
        R(f"Error reading shapefile: {e}")
else:
    R(f"Dunes.shp not found at {dunes_shp}")

# ── 4. Visual output ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle("Geomorphology Raster Diagnostic", fontsize=13)

# Raw class map
ax = axes[0, 0]
im = ax.imshow(geo, origin='upper', extent=[0,360,-90,90],
               aspect='auto', cmap='tab10', vmin=0, vmax=9,
               interpolation='nearest')
ax.set_title('Terrain classes (0-7)')
ax.axvline(180, color='white', lw=1, ls='--')
for lon, lat, label in [(250, 0,'Belet'), (155, 0,'Shangri-La'), (100,-5,'Xanadu')]:
    ax.plot(lon, lat, 'w+', ms=8)
    ax.text(lon+2, lat, label, color='white', fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel('Lon (°W)'); ax.set_ylabel('Lat (°)')

# Dunes only
ax = axes[0, 1]
dune_map = (geo == 2).astype(float)
dune_map[dune_map == 0] = np.nan
im2 = ax.imshow(dune_map, origin='upper', extent=[0,360,-90,90],
                aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_title('Dunes only (class 2)\n(should be equatorial belt, not symmetric)')
ax.axvline(180, color='blue', lw=1.5, ls='--', label='180°W')
for lon, lat, label in [(250, 0,'Belet'), (155, 0,'Shangri-La'), (315, 0,'Aztlan')]:
    ax.plot(lon, lat, 'b+', ms=8)
    ax.text(lon+2, lat, label, color='blue', fontsize=7)
ax.legend(fontsize=7)
ax.set_xlabel('Lon (°W)'); ax.set_ylabel('Lat (°)')

# Longitude histogram of dunes
ax = axes[1, 0]
dune_pixels = geo[np.abs(lat_centres[:, None]) < 30] == 2
dune_lons_all = []
for r, row in enumerate(geo):
    if abs(lat_centres[r]) < 30:
        cols_with_dunes = np.where(row == 2)[0]
        dune_lons_all.extend(lon_centres[cols_with_dunes].tolist())
if dune_lons_all:
    ax.hist(dune_lons_all, bins=72, range=(0, 360), color='C1', alpha=0.7)
    ax.axvline(180, color='r', lw=1.5, ls='--', label='180°W seam')
    ax.axvline(155, color='g', lw=1, ls=':', label='Shangri-La 155°W')
    ax.axvline(250, color='b', lw=1, ls=':', label='Belet 250°W')
    ax.axvline(315, color='m', lw=1, ls=':', label='Aztlan 315°W')
    ax.set_title('Dune pixel longitude histogram (±30° lat)\nShould peak at ~155, 250, 315°W')
    ax.set_xlabel('Longitude (°W)'); ax.set_ylabel('Count')
    ax.legend(fontsize=7)

# Left vs mirror(right) comparison for dunes
ax = axes[1, 1]
if dune_lons_all:
    lons = np.array(dune_lons_all)
    left_hist  = np.histogram(lons[lons < 180],  bins=36, range=(0, 180))[0]
    right_hist = np.histogram(lons[lons >= 180], bins=36, range=(180, 360))[0]
    bin_centres = np.linspace(2.5, 177.5, 36)
    ax.plot(bin_centres, left_hist,          'b-', lw=1.5, label='Left half (0-180°W)')
    ax.plot(bin_centres, right_hist[::-1],   'r--', lw=1.5, label='Right half MIRRORED')
    ax.set_title('Mirror test: if lines match → raster is mirrored')
    ax.set_xlabel('Longitude bucket (°W)'); ax.set_ylabel('Count')
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(out_dir / 'geo_diag_map.png', dpi=120, bbox_inches='tight')
plt.close()
R(f"\nSaved: {out_dir / 'geo_diag_map.png'}")

# Write text report
with open(out_dir / 'geo_diag_report.txt', 'w') as f:
    f.write('\n'.join(report))
R(f"Saved: {out_dir / 'geo_diag_report.txt'}")
