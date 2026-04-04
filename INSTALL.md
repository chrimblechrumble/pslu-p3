# Data Installation Guide — Titan Habitability Pipeline

This file describes where to place every raw data file the pipeline expects.
The pipeline gracefully skips any dataset that is absent, so you do not need
all files to run; the minimum viable set is marked **[required]**.

---

## Directory layout

```
titan_pipeline/
└── data/
    └── raw/                         ← set via --data-dir (default: data/raw)
        ├── GTIED00N090_T126_V01.IMG     [required] Topography east tile
        ├── GTIED00N270_T126_V01.IMG     [required] Topography west tile
        ├── TitanSARHiSAR_MAP2_SIMP_256px.tif      SAR mosaic
        ├── Titan_ISS_NearGlobal_450m.tif           ISS mosaic
        ├── Titan_VIMS-ISS.tif                      VIMS+ISS mosaic
        ├── vims_footprints.parquet                 VIMS coverage index
        ├── hayes_topo/
        │   └── topo_4PPD_interp.cub                Corlies 2017 gap-fill DEM
        ├── geomorphology_shapefiles/               Lopes+2019 (global)
        │   ├── Craters.shp  (+ .dbf .prj .shx)
        │   ├── Dunes.shp
        │   ├── Plains_3.shp
        │   ├── Basins.shp
        │   ├── Mountains.shp
        │   ├── Labyrinth.shp
        │   ├── Lakes.shp                           (optional but recommended)
        │   └── global_channels.shp                 Miller+2021 channels
        └── birch_polar_mapping/                    Birch+2017 polar lake data
            ├── birch_filled/        ← confirmed present-day liquid surfaces
            │   ├── north_filled_lakes.shp   (any *.shp filename is fine)
            │   └── south_filled_lakes.shp
            └── birch_empty/         ← empty basins / paleo-lakes
            │   ├── north_empty_basins.shp
            │   └── ...
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
> **Known south-truncation:** The Cornell-distributed GTIED T126 files are
> shorter than their labels state. This is a confirmed distribution
> characteristic (April 2026); re-downloading gives the same result.
> Coverage is approximately 90°N to 48–51°S. Ontario Lacus (72°S) falls
> in the missing region. The Corlies 2017 gap-filler (Section 5 below)
> compensates when available. All northern seas and equatorial sites are
> fully covered.

```bash
# East tile (0–180°W)
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTIED00N090_T126_V01.IMG.gz
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTIED00N090_T126_V01.LBL
gunzip GTIED00N090_T126_V01.IMG.gz
mv GTIED00N090_T126_V01.IMG data/raw/

# West tile (180–360°W)
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTIED00N270_T126_V01.IMG.gz
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTIED00N270_T126_V01.LBL
gunzip GTIED00N270_T126_V01.IMG.gz
mv GTIED00N270_T126_V01.IMG data/raw/
```

The pipeline also accepts the `.IMG.gz` files directly (auto-decompresses).

---

### 2. SAR mosaic

Source: USGS Astrogeology / PDS Imaging Node

URL: https://astrogeology.usgs.gov/search/map/saturn/titan/cassini/titan-sar-hism-ap-map2-simp-256px

Place as: `data/raw/TitanSARHiSAR_MAP2_SIMP_256px.tif`

Or via the `--data-dir` override if stored elsewhere.

---

### 3. ISS 938 nm mosaic

Source: USGS Astrogeology

URL: https://astrogeology.usgs.gov/search/map/saturn/titan/cassini/titan-iss-near-global-mosaic-450m

Place as: `data/raw/Titan_ISS_NearGlobal_450m.tif`

---

### 4. VIMS+ISS mosaic

Source: CaltechDATA (Seignovert et al. 2019), CC-BY-4.0

```bash
wget "https://data.caltech.edu/records/8q9an-yt176/files/Titan_VIMS-ISS.tif?download=1" \
     -O data/raw/Titan_VIMS-ISS.tif
```

---

### 5. VIMS footprint parquet

Source: Nantes VIMS portal / Hayes Research Group

Place as: `data/raw/vims_footprints.parquet`

A 1,000-row development sample (`vims_sample_1000rows.parquet`) is bundled
with the pipeline code for testing without the full 227 MB catalogue.

---

### 6. Corlies 2017 interpolated topography (gap-fill DEM)

Source: Hayes Research Group

URL: https://hayesresearchgroup.com/data-products/ → titan_topo_corlies.zip

```bash
# After downloading and extracting the zip:
mkdir -p data/raw/hayes_topo
cp topo_4PPD_interp.cub data/raw/hayes_topo/
```

---

### 7. Lopes+2020 geomorphology shapefiles

Source: Mendeley Data — Schoenfeld (2024) — **CC-BY-4.0**

DOI: [10.17632/f6jrtyfp66.1](https://data.mendeley.com/datasets/f6jrtyfp66/1)

```bash
# Download the zip (11.6 MB) from Mendeley
wget "https://data.mendeley.com/api/datasets/f6jrtyfp66/files/zip?version=1" \
     -O lopes_shapefiles.zip

# SHA-256 of the zip (verify before use):
# 6b6848afa62344e50103cea37a95fccdd75609b5235be22cddedfd8f1e6b9535

mkdir -p data/raw/geomorphology_shapefiles
unzip lopes_shapefiles.zip -d data/raw/geomorphology_shapefiles/
```

> **CONFIRMED FILE LISTING** (from Mendeley API, April 2026 — 6 shapefiles):
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
> **`Lakes.shp` is NOT in this distribution.** The Lakes unit was not
> deposited in the public Mendeley archive. Lake polygon geometry comes
> from the separate **Birch+2017 Cornell archive** (Section 9 below),
> which provides higher-resolution polar lake outlines with filled vs
> empty basin distinction. The pipeline is designed to use Birch as the
> primary lake source; the Lopes lake class is a forward-compatibility
> stub that is currently inactive.

Override path with: `--shapefile-dir /path/to/shapefiles`

---

### 8. Miller+2021 global channel map

Source: Hayes Research Group

URL: https://hayesresearchgroup.com/data-products/ → titan_channels_miller.zip

```bash
# After extracting:
cp global_channels.shp data/raw/geomorphology_shapefiles/
cp global_channels.dbf data/raw/geomorphology_shapefiles/   # (and .shx etc.)
```

---

### 9. Birch+2017 polar lake dataset  ★ New in v2 ★

Source: Cornell eCommons archive (verified 2026-04)

```bash
wget https://data.astro.cornell.edu/titan_polar_mapping_birch/titan_polar_mapping_birch.zip
unzip titan_polar_mapping_birch.zip -d titan_polar_mapping_birch_raw/
```

The zip (6.0 GB) contains:

```
full_dataset/
  Various Mapping Shapefiles/
    Birch Polar Geomorphic (2017)/
      north/
        Fl_NORTH.shp   <- confirmed liquid lakes / seas (north)
        El_NORTH.shp   <- empty lake depressions / paleo-lakes (north)
        (+ Lfd, Lud, Hdb, Hdd, Hud, Vdb, Vmb, Vub, Af, Fm, Mtn,
           Fluvial_Valleys and *_LR_* variants — NOT used by this pipeline)
      south/
        Fl_SOUTH.shp   <- confirmed liquid lakes / seas (south)
        El_SOUTH.shp   <- empty lake depressions (south)
        Em_SOUTH.shp   <- empty seas / four large southern paleoseas
        (+ same geomorphic unit files as north)
    Miller Channels (2021)/
        (fluvial channel network — NOT used by this pipeline)
```


**Create the pipeline directory layout:**

```bash
mkdir -p data/raw/birch_polar_mapping/birch_filled
mkdir -p data/raw/birch_polar_mapping/birch_empty
```

Define a path variable for convenience:

```bash
BIRCH="titan_polar_mapping_birch_raw/full_dataset/Various Mapping Shapefiles/Birch Polar Geomorphic (2017)"
```

**Copy confirmed-liquid shapefiles into `birch_filled/`:**

```bash
for ext in shp dbf shx prj; do
  cp "$BIRCH/north/Fl_NORTH.$ext" data/raw/birch_polar_mapping/birch_filled/
  cp "$BIRCH/south/Fl_SOUTH.$ext" data/raw/birch_polar_mapping/birch_filled/
done
```

**Copy empty-basin and palaeosea shapefiles into `birch_empty/`:**

```bash
for ext in shp dbf shx prj; do
  cp "$BIRCH/north/El_NORTH.$ext" data/raw/birch_polar_mapping/birch_empty/
  cp "$BIRCH/south/El_SOUTH.$ext" data/raw/birch_polar_mapping/birch_empty/
  cp "$BIRCH/south/Em_SOUTH.$ext" data/raw/birch_polar_mapping/birch_empty/
done
```

The pipeline scans all `*.shp` files in each sub-directory; names do not matter.

Override path with: `--birch-dir /path/to/birch_polar_mapping`

**What changes when Birch data is present:**

| Feature | Without Birch | With Birch |
|---------|---------------|------------|
| Feature 1 (liquid_hydrocarbon) | SAR low-backscatter proxy in polar region | Expert-mapped lake outlines (binary 1.0 for confirmed liquid) |
| Feature 5 (surface_atm_interaction) — lake margin | Zero (no shoreline data) | Exact Birch shoreline dilation (~13 km margin zone) |
| Feature 5 — paleo_lake_indicator | Zero (absent) | Smoothed empty-basin proximity score (El_* + Em_SOUTH) |

---

## Verify your installation

```bash
python run_pipeline.py --temporal-mode present 2>&1 | grep -E "INFO|WARNING|ERROR" | head -30
```

The log will report which datasets were found and which were skipped.
For each missing dataset you will see a `WARNING` or `INFO` with download
instructions.

---

## Minimum viable run (topography only)

With only the two GTDR tiles installed, the pipeline will still complete
using synthetic CIRS temperature data and SAR/VIMS proxies, but habitability
scores will be based primarily on topographic complexity and a global
methane-cycle prior rather than lake or spectral data.

---

## Version history

| Version | Date       | Key changes |
|---------|------------|-------------|
| v1.0    | 2026-03    | Initial release — Lopes geomorphology, SAR, VIMS, GTDR |
| v2.0    | 2026-03    | Birch+2017 polar lake integration |
