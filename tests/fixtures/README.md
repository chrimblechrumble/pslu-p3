# tests/fixtures/

Real Cassini data files used by integration tests.
All tests auto-skip when a required file is absent.

## Directory layout

```
tests/fixtures/
├── README.md
│
├── shapefiles/                              ← Lopes et al. (2019) geomorphology
│   ├── Craters.shp   (+ .dbf .prj .shx)
│   ├── Dunes.shp, Plains_3.shp, Basins.shp
│   ├── Mountains.shp, Labyrinth.shp
│   └── Lakes.shp  (optional)
│
├── gtdr/                                    ← Cornell GTDR/GTDE topography
│   │
│   │   ── GTDE: Dense interpolated DEM (PREFERRED, ~90% global) ────────
│   ├── GTIED00N090_T126_V01.IMG  (or .IMG.gz)  ← east tile, 0–180°W
│   ├── GTIED00N090_T126_V01.LBL
│   ├── GTIED00N270_T126_V01.IMG  (or .IMG.gz)  ← west tile, 180–360°W
│   ├── GTIED00N270_T126_V01.LBL
│   │
│   │   ── GT0E: Standard sparse GTDR (~25% coverage) ──────────────────
│   ├── GT2ED00N090_T126_V01.IMG  (or .IMG.gz)  ← T126 east (final mission)
│   ├── GT2ED00N090_T126_V01.LBL
│   ├── GT0EB00N090_T077_V01.IMG                ← T077 east (legacy, also accepted)
│   └── GT0EB00N090_T077_V01.LBL
│
└── vims/                                    ← VIMS footprint index
    └── vims_footprints.parquet   (full ~227 MB, or sample 43 KB)
        (vims_sample_1000rows.parquet also accepted)
```

## GTDE vs GT0E — Cornell naming

| Prefix  | T-flyby | Coverage    | Use                               |
|---------|---------|-------------|-----------------------------------|
| `GTDE`  | T126    | ~90% global | **Preferred** — spline-interpolated |
| `GT0E`  | T126    | ~25%        | Sparse (final mission)            |
| `GT0E`  | T077    | ~15%        | Legacy (partial mission)          |

Cornell distributes all files as `.IMG.gz` (gzip-compressed).
The pipeline reader decompresses them transparently — place either
`.IMG` or `.IMG.gz` in the fixture directory.

## Pipeline DEM priority

The pipeline (`_preprocess_topography`) tries in this order:
1. `GTIED00N090_T126_V01` + `GTIED00N270_T126_V01` — **PREFERRED**
2. `GT2ED00N090_T126_V01` + `GT2ED00N270_T126_V01`
3. `GT0EB00N090_T077_V01` + matching west tile

## Where to get the data

### Cornell eCommons (direct download, no login, .IMG.gz)
  https://data.astro.cornell.edu/RADAR/DATA/GTDR/
  Download GTIED00N090_T126_V01.IMG.gz + GTIED00N270_T126_V01.IMG.gz
  and their .LBL companions.

### USGS gtdr-data.zip (same product set)
  http://astropedia.astrogeology.usgs.gov/download/Titan/Cassini/GTDR/gtdr-data.zip

### Shapefiles
  Contact JPL / Rosaly Lopes, or Mendeley Data:
  https://data.mendeley.com/research-data/?query=titan

### VIMS parquet
  Contact Stéphane Le Mouélic (LPG Nantes), or:
  https://github.com/seignovert/pyvims

## Which tests use which fixtures

| Fixture                           | Test                                           |
|-----------------------------------|------------------------------------------------|
| shapefiles/Craters.shp            | test_real_rasteriser_with_craters              |
| shapefiles/Craters.shp            | test_loads_real_sample_shapefile               |
| shapefiles/                       | test_real_all_layers_present                   |
| gtdr/GTIED00N090*.IMG[.gz]        | test_real_gtde_east_has_global_coverage        |
| gtdr/GTDE* (both)                 | test_real_gtde_mosaic_near_global              |
| gtdr/GTDE* (both)                 | test_preprocess_uses_gtde_when_available       |
| gtdr/GT0ED or GT0EB east          | test_real_gt0e_east_reads_correctly            |
| gtdr/GT0E east .LBL               | test_real_gt0e_label_metadata                  |
| gtdr/GT0E .IMG + .IMG.gz          | test_real_gt0e_gzip_equals_uncompressed        |
| vims/*.parquet                    | test_load_real_parquet + others                |
