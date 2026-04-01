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
        ├── GTDED00N090_T126_V01.IMG     [required] Topography east tile
        ├── GTDED00N270_T126_V01.IMG     [required] Topography west tile
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
        └── birch_polar_mapping/                    Birch+2017 / Palermo+2022
            ├── birch_filled/        ← confirmed present-day liquid surfaces
            │   ├── north_filled_lakes.shp   (any *.shp filename is fine)
            │   └── south_filled_lakes.shp
            ├── birch_empty/         ← empty basins / paleo-lakes
            │   ├── north_empty_basins.shp
            │   └── ...
            └── palermo/             ← Palermo+2022 alternative mapping
                └── palermo_seas.shp
```

---

## Dataset-by-dataset download instructions

### 1. Topography — GTDE tiles [required]

Source: Cornell eCommons (Corlies et al. 2017)

```bash
# East tile (0–180°W)
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N090_T126_V01.IMG.gz
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N090_T126_V01.LBL
gunzip GTDED00N090_T126_V01.IMG.gz
mv GTDED00N090_T126_V01.IMG data/raw/

# West tile (180–360°W)
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N270_T126_V01.IMG.gz
wget https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N270_T126_V01.LBL
gunzip GTDED00N270_T126_V01.IMG.gz
mv GTDED00N270_T126_V01.IMG data/raw/
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

### 7. Lopes+2019 geomorphology shapefiles

Source: JPL / Rosaly Lopes (personal communication or Mendeley Data)

DOI: 10.1038/s41550-019-0917-6

Place all `.shp`, `.dbf`, `.prj`, `.shx` files in:
`data/raw/geomorphology_shapefiles/`

Expected stems: `Craters`, `Dunes`, `Plains_3`, `Basins`, `Mountains`,
`Labyrinth`, `Lakes` (Lakes.shp is optional but improves Feature 1).

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

### 9. Birch+2017 / Palermo+2022 polar lake dataset  ★ New in v2 ★

Source: Cornell eCommons (Birch+2017 shapefile archive)

```bash
wget https://data.astro.cornell.edu/titan_polar_mapping_birch/titan_polar_mapping_birch.zip
unzip titan_polar_mapping_birch.zip -d titan_polar_mapping_birch_raw/
```

The zip contains a `Mapping Shapefiles/` folder with two sub-folders:
- `Birch+2017/` — filled lake, empty basin, and geomorphological unit shapefiles
- `Palermo+2022/` — alternative sea and lake mapping

**Create the pipeline directory layout:**

```bash
mkdir -p data/raw/birch_polar_mapping/birch_filled
mkdir -p data/raw/birch_polar_mapping/birch_empty
mkdir -p data/raw/birch_polar_mapping/palermo
```

**Copy Birch+2017 filled lake/sea shapefiles:**
```bash
# Copy all filled lake/sea .shp (+ .dbf .prj .shx) files:
cp titan_polar_mapping_birch_raw/Mapping\ Shapefiles/Birch+2017/*filled*.shp \
   data/raw/birch_polar_mapping/birch_filled/
cp titan_polar_mapping_birch_raw/Mapping\ Shapefiles/Birch+2017/*filled*.dbf \
   data/raw/birch_polar_mapping/birch_filled/
# ... repeat for .prj .shx
```

**Copy Birch+2017 empty basin shapefiles:**
```bash
cp titan_polar_mapping_birch_raw/Mapping\ Shapefiles/Birch+2017/*empty*.shp \
   data/raw/birch_polar_mapping/birch_empty/
# ... repeat for .dbf .prj .shx
```

**Copy Palermo+2022 shapefiles:**
```bash
cp titan_polar_mapping_birch_raw/Mapping\ Shapefiles/Palermo+2022/*.shp \
   data/raw/birch_polar_mapping/palermo/
# ... repeat for .dbf .prj .shx
```

The exact filenames inside each sub-directory do not matter — the pipeline
scans for all `*.shp` files in each sub-directory and merges them.

Override path with: `--birch-dir /path/to/birch_polar_mapping`

**What changes when Birch data is present:**

| Feature | Without Birch | With Birch |
|---------|---------------|------------|
| Feature 1 (liquid_hydrocarbon) | SAR low-backscatter proxy in polar region | Expert-mapped lake outlines (binary 1.0 for confirmed liquid) |
| Feature 5 (surface_atm_interaction) — lake margin | Zero (no Lakes.shp) | Exact Birch shoreline dilation (~13 km margin) |
| Feature 5 — paleo_lake_indicator | Zero (absent) | Smoothed empty-basin proximity score |

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
| v2.0    | 2026-03    | Birch+2017 / Palermo+2022 polar lake integration |
