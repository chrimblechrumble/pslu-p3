# Data Installation Guide — Titan Habitability Pipeline

This file describes where to place every raw data file the pipeline expects.
The pipeline gracefully skips any dataset that is absent, so you do not need
all files to run; the minimum viable set is marked **[required]**.

---

## Quick reference: what you need

| Priority | File | Size | Feature(s) |
|---|---|---|---|
| **Required** | `GTIED00N090_T126_V01.IMG` | 8 MB | f₆ topographic complexity |
| **Required** | `GTIED00N270_T126_V01.IMG` | 8 MB | f₆ topographic complexity |
| Recommended | `Titan_VIMS-ISS.tif` + `.hdr` | 144 MB | f₂ organic abundance (VIMS spectral) |
| Recommended | `Titan_SAR_HiSAR_...128ppd.tif` | 1.0 GB | f₁ liquid HC, f₃ acetylene energy, f₇ geodiversity |
| Recommended | `geomorphology_shapefiles/` | 22 MB | f₂ organic (gap-fill), f₇ geodiversity |
| Recommended | `birch_polar_mapping/` | 6 MB | f₁ confirmed lake outlines |
| Optional | `Titan_ISS_NearGlobal_450m.tif` | 2.6 GB | f₂ fallback (ISS albedo proxy) |
| Optional | `hayes_topo/topo_4PPD_interp.cub` | 4 MB | f₆ south-polar gap-fill |
| Optional | `vims_cubes/` | ~1 GB | f₂ 5µm/2µm ratio blend (0.5% coverage) |
| Auto-created | `gravity_k2.json` | <1 KB | f₈ subsurface ocean floor value |

### Files you do NOT need (safe to delete)

| File | Why not needed |
|---|---|
| `GTDED00N090_T126_V01.IMG` / `.LBL` | **Not elevation** — this is the distance-to-nearest-measurement quality metric (km). The pipeline explicitly warns against using it. |
| `GTDED00N270_T126_V01.IMG` / `.LBL` | Same — quality metric, not elevation. |
| `GT2ED00N090_T126_V01.IMG` | Fallback sparse DEM (~25% coverage). Redundant when GTIE tiles are present. |
| `GT2ED00N270_T126_V01.IMG` | Same — fallback DEM, redundant. |
| `vims_footprints.parquet` | No longer consumed by any feature (removed in v5). Was used for VIMS coverage density. |
| `vims_footprints-backup.parquet` | Backup of above — not needed. |
| `vims_footprints_cubes.parquet` | Cube discovery index — only needed when rebuilding `vims_cubes/` from scratch. |
| `*.LBL` files | PDS3 label files. Informational only; not read by the pipeline. |
| `*.aux.xml` files | GDAL sidecar metadata. Auto-generated; safe to delete. |
| `Lakes.shp` | **Does not exist** in the Mendeley distribution. Lake geometry comes from Birch polar mapping (see §9). |

---

## Directory layout

```
titan_pipeline/
└── data/
    └── raw/                         ← set via --data-dir (default: data/raw)
        ├── GTIED00N090_T126_V01.IMG     [required] Elevation east tile
        ├── GTIED00N270_T126_V01.IMG     [required] Elevation west tile
        ├── Titan_SAR_HiSAR_MosaicThru_T104_Jan2015_clon180_128ppd.tif
        ├── Titan_ISS_NearGlobal_450m.tif
        ├── Titan_VIMS-ISS.tif
        ├── Titan_VIMS-ISS.hdr
        ├── gravity_k2.json              (auto-created if absent)
        ├── hayes_topo/
        │   └── topo_4PPD_interp.cub     Corlies 2017 gap-fill DEM
        ├── geomorphology_shapefiles/    Lopes+2019 (6 classes, no Lakes)
        │   ├── Craters.shp  (+ .dbf .prj .shx)
        │   ├── Dunes.shp
        │   ├── Plains_3.shp
        │   ├── Basins.shp
        │   ├── Mountains.shp
        │   ├── Labyrinth.shp
        │   └── global_channels.shp      Miller+2021 channels
        ├── birch_polar_mapping/         Birch+2017 polar lakes
        │   ├── birch_filled/            Confirmed present-day liquid
        │   │   ├── Fl_NORTH.shp (+ .dbf .shx .prj)
        │   │   └── Fl_SOUTH.shp
        │   └── birch_empty/             Empty basins / paleo-lakes
        │       ├── El_NORTH.shp
        │       ├── El_SOUTH.shp
        │       └── Em_SOUTH.shp
        └── vims_cubes/                  (optional) Raw VIMS spectral cubes
```

---

## Dataset-by-dataset download instructions

### 1. Topography — GTIE tiles [required]

Source: Cornell eCommons (Corlies et al. 2017)

> **CRITICAL — product code:** The correct topography product is **`GTIED`**
> (Interpolated Elevation in metres). Do **not** use `GTDED` — that product
> contains the *distance-to-nearest-measurement* quality map (units: km), not
> elevation. Both files have identical PDS3 structure and the mistake is easy
> to make; `GTIED` is what the pipeline expects.
>
> The `GT2ED` tiles (standard GTDR, ~25% coverage) are a fallback that the
> pipeline uses only if GTIE tiles are absent. If you have GTIE tiles, you
> do not need GT2E tiles.
>
> **Known south-truncation:** The Cornell-distributed GTIED T126 files are
> shorter than their labels state. This is a confirmed distribution
> characteristic (April 2026); re-downloading gives the same result.
> Coverage is approximately 90°N to 48–51°S. Ontario Lacus (72°S) falls
> in the missing region. The Corlies 2017 gap-filler (Section 6 below)
> compensates when available. All northern seas and equatorial sites are
> fully covered.

```bash
# East tile (0–180°W)
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTIED00N090_T126_V01.IMG.gz
gunzip GTIED00N090_T126_V01.IMG.gz
mv GTIED00N090_T126_V01.IMG data/raw/

# West tile (180–360°W)
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTIED00N270_T126_V01.IMG.gz
gunzip GTIED00N270_T126_V01.IMG.gz
mv GTIED00N270_T126_V01.IMG data/raw/
```

The pipeline also accepts the `.IMG.gz` files directly (auto-decompresses).

---

### 2. SAR mosaic

Source: USGS Astrogeology / PDS Imaging Node

URL: https://astrogeology.usgs.gov/search/map/saturn/titan/cassini/titan-sar-hism-ap-map2-simp-256px

Place as: `data/raw/Titan_SAR_HiSAR_MosaicThru_T104_Jan2015_clon180_128ppd.tif`

Used by: f₁ (liquid hydrocarbon SAR proxy), f₃ (acetylene energy backscatter component), f₇ (geomorphological diversity), f₈ (crater annuli detection).

---

### 3. ISS 938 nm mosaic

Source: USGS Astrogeology

URL: https://astrogeology.usgs.gov/search/map/saturn/titan/cassini/titan-iss-near-global-mosaic-450m

Place as: `data/raw/Titan_ISS_NearGlobal_450m.tif`

Used by: f₂ organic abundance (ISS albedo fallback when VIMS is absent — not normally triggered). This is a 2.6 GB file; it can be omitted if disk space is a concern.

---

### 4. VIMS+ISS spectral mosaic

Source: CaltechDATA (Seignovert et al. 2019), CC-BY-4.0

```bash
wget "https://data.caltech.edu/records/8q9an-yt176/files/Titan_VIMS-ISS.tif?download=1" \
     -O data/raw/Titan_VIMS-ISS.tif
```

Also download the ENVI header file:
```bash
wget "https://data.caltech.edu/records/8q9an-yt176/files/Titan_VIMS-ISS.hdr?download=1" \
     -O data/raw/Titan_VIMS-ISS.hdr
```

Used by: f₂ organic abundance (primary spectral source; VIMS 1.59/1.27 µm band ratio).

> **Longitude convention:** This mosaic uses IAU east-positive longitude.
> The pipeline converts to west-positive via a horizontal flip during
> preprocessing (see `_reproject_geotiff` with `east_positive=True`).

---

### 5. Lopes+2019 geomorphology shapefiles

Source: Mendeley Data — Schoenfeld (2024) — **CC-BY-4.0**

DOI: [10.17632/f6jrtyfp66.1](https://data.mendeley.com/datasets/f6jrtyfp66/1)

```bash
wget "https://data.mendeley.com/api/datasets/f6jrtyfp66/files/zip?version=1" \
     -O lopes_shapefiles.zip
mkdir -p data/raw/geomorphology_shapefiles
unzip lopes_shapefiles.zip -d data/raw/geomorphology_shapefiles/
```

> **Confirmed file listing** (6 shapefiles — no Lakes.shp):
>
> | File | Size |
> |------|------|
> | Basins.shp | 1.83 MB |
> | Craters.shp | 108 KB |
> | Dunes.shp | 2.32 MB |
> | Labyrinth.shp | 615 KB |
> | Mountains.shp | 7.14 MB |
> | Plains_3.shp | 9.50 MB |
>
> **`Lakes.shp` is NOT in this distribution** and is not needed.
> Lake polygon geometry comes from the Birch+2017 Cornell archive
> (Section 8 below), which provides higher-resolution polar lake outlines.

Override path with: `--shapefile-dir /path/to/shapefiles`

---

### 6. Corlies 2017 interpolated topography (gap-fill DEM)

Source: Cornell eCommons

URL: https://data.astro.cornell.edu/titan_topo_corlies/titan_topo_corlies.zip

```bash
wget https://data.astro.cornell.edu/titan_topo_corlies/titan_topo_corlies.zip
unzip titan_topo_corlies.zip
mkdir -p data/raw/hayes_topo
cp full_dataset/topo_4PPD_interp.cub data/raw/hayes_topo/
```

Used to fill the GTIE south-polar gap (south of ~48°S). Without this file,
Ontario Lacus and other south-polar sites have no elevation data.

---

### 7. Miller+2021 global channel map

Source: Cornell eCommons

URL: https://data.astro.cornell.edu/titan_channels_miller/titan_channels_miller.zip

```bash
wget https://data.astro.cornell.edu/titan_channels_miller/titan_channels_miller.zip
unzip titan_channels_miller.zip
cp full_dataset/global_channels.* data/raw/geomorphology_shapefiles/
```

Used by: f₄ methane cycle (channel density component, 35% weight) and f₅ surface–atmosphere interaction (channel density component, 30% weight).

---

### 8. Birch+2017 polar lake dataset

Source: Cornell eCommons

URL: https://data.astro.cornell.edu/titan_polar_mapping_birch/titan_polar_mapping_birch.zip

The full zip is 6.0 GB but only the lake shapefiles (~6 MB) are needed:

```bash
wget https://data.astro.cornell.edu/titan_polar_mapping_birch/titan_polar_mapping_birch.zip
unzip titan_polar_mapping_birch.zip -d birch_raw/

mkdir -p data/raw/birch_polar_mapping/birch_filled
mkdir -p data/raw/birch_polar_mapping/birch_empty

BIRCH="birch_raw/full_dataset/Various Mapping Shapefiles/Birch Polar Geomorphic (2017)"

# Confirmed-liquid lakes
for ext in shp dbf shx prj; do
  cp "$BIRCH/north/Fl_NORTH.$ext" data/raw/birch_polar_mapping/birch_filled/
  cp "$BIRCH/south/Fl_SOUTH.$ext" data/raw/birch_polar_mapping/birch_filled/
done

# Empty basins / paleo-lakes
for ext in shp dbf shx prj; do
  cp "$BIRCH/north/El_NORTH.$ext" data/raw/birch_polar_mapping/birch_empty/
  cp "$BIRCH/south/El_SOUTH.$ext" data/raw/birch_polar_mapping/birch_empty/
  cp "$BIRCH/south/Em_SOUTH.$ext" data/raw/birch_polar_mapping/birch_empty/
done
```

**What changes when Birch data is present:**

| Feature | Without Birch | With Birch |
|---------|---------------|------------|
| f₁ (liquid_hydrocarbon) | SAR low-backscatter proxy | Expert-mapped lake outlines (binary 1.0 for confirmed liquid) |
| f₅ (surface_atm_interaction) — lake margin | Zero (no shoreline data) | Exact Birch shoreline dilation (~13 km margin zone) |

Override path with: `--birch-dir /path/to/birch_polar_mapping`

---

### 9. VIMS spectral cubes (optional)

Source: Seignovert VIMS portal / Hayes Research Group

Place as: `data/raw/vims_cubes/` (directory of `.cub` or `.qub` files)

The pipeline builds 5.0 µm and 2.03 µm window mosaics from individual
VIMS cubes and computes a 5.0/2.03 µm ratio that is blended (50/50) with
the Seignovert mosaic in f₂ organic abundance. With the sample parquet
(1,000 rows), this provides only ~0.5% additional pixel coverage — a minor
refinement. The full 227 MB parquet catalogue would provide more coverage.

This directory can be omitted with negligible impact on results.

---

### 10. Gravity k₂ (auto-created)

The pipeline auto-creates `data/raw/gravity_k2.json` with the Iess et al.
(2012) tidal Love number ($k_2 = 0.589 \pm 0.150$). No manual download
is needed.

---

## Verify your installation

```bash
python run_pipeline.py --temporal-mode present 2>&1 | grep -E "INFO|WARNING|ERROR" | head -30
```

The log will report which datasets were found and which were skipped.

---

## Minimum viable run (topography only)

With only the two GTIE tiles installed, the pipeline will still complete
using synthetic CIRS temperature data and SAR/VIMS proxies, but habitability
scores will be based primarily on topographic complexity and a global
methane-cycle prior rather than lake or spectral data.

---

## Version history

| Version | Date | Key changes |
|---------|------|-------------|
| v1.0 | 2026-03 | Initial release — Lopes geomorphology, SAR, VIMS, GTDR |
| v2.0 | 2026-03 | Birch+2017 polar lake integration |
| v5.0 | 2026-04 | VIMS alignment fix (east-positive flip); VIMS parquet removed; blended organic mode; analytical posteriors |