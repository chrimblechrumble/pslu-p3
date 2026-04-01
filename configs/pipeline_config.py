# Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
# Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""
configs/pipeline_config.py
===========================
Central, fully-typed configuration for the Titan Habitability Pipeline.

All tuneable parameters live here. Researchers modifying the priors,
resolution, Bayesian backend, or dataset paths should edit this file
only -- nothing scientific is hard-coded elsewhere.

Coordinate system notes (from direct inspection of real data)
--------------------------------------------------------------
All raster products use the same projection family but differ in details:

  GeoTIFF mosaics (USGS Astropedia):
    - SimpleCylindrical, Titan sphere R=2,575,000 m
    - Coordinates stored in METRES (projected CRS)
    - Longitude: WEST-positive, 0 deg -> 360 deg  (clon180 convention)
    - Nodata: 0.0 (float32)
    - Example pixel scale: 351.11 m/px at 128 ppd

  GTDR .IMG tiles (PDS3, Stanford/Zebker):
    - Equirectangular, Titan sphere R=2,575,000 m
    - Coordinates in deg (geographic)
    - Longitude: WEST-positive, 0 deg -> 360 deg
    - Nodata: 0xFF7FFFFB = -3.4028x10^3^8  (PDS3 MISSING_CONSTANT, NOT NaN)
    - Resolution: 2.0 ppd (0.5 deg/pixel = 22.47 km/pixel)
    - Coverage: two half-globe tiles (0-180 deg and 180-360 deg west)

  Shapefiles (JPL/Lopes geomorphology):
    - GEOGCS GCS_Titan_2000, Titan sphere R=2,575,000 m
    - Coordinates in deg (geographic)
    - Longitude: EAST-positive, -180 deg -> +180 deg  <- DIFFERENT from rasters
    - PolygonM geometry type (has measure coordinate M)
    - One file per terrain class: Cr/Dn/Pl/Lb/Mt/Ba/Lk
    - Conversion: lon_west = (-lon_east) % 360

  VIMS parquet (spatial footprint index):
    - Tabular, no raster
    - lon column: WEST-positive (0->360 deg)
    - One row per VIMS pixel footprint
    - Links to PDS cube files via 'id' column (SCLK_counter format)

The canonical pipeline grid uses west-positive 0->360 deg in metres.

References
----------
Lorenz et al. (2013)      doi:10.1016/j.icarus.2013.04.002
Lopes et al. (2019)       doi:10.1038/s41550-019-0917-6
McKay & Smith (2005)      doi:10.1016/j.icarus.2005.05.018
Yanez et al. (2024)       doi:10.1016/j.icarus.2024.115969
Affholder et al. (2021)   doi:10.1038/s41550-021-01372-6
Affholder et al. (2025)   doi:10.3847/PSJ/addb09
Catling et al. (2018)     doi:10.1089/ast.2017.1737
Iess et al. (2012)        doi:10.1126/science.1219631
Neish et al. (2018)       doi:10.1089/ast.2017.1758
Schulze-Makuch & Grinspoon (2005) doi:10.1089/ast.2005.5.560
Malaska et al. (2025)     Titan After Cassini-Huygens Ch.9
Meyer-Dombard et al.(2025) Titan After Cassini-Huygens Ch.14
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Titan mean radius in metres (IAU 2015; confirmed in all data products)
TITAN_RADIUS_M: float = 2_575_000.0

#: Titan mean radius in km
TITAN_RADIUS_KM: float = TITAN_RADIUS_M / 1000.0

#: PROJ4 string for the canonical pipeline CRS.
#: SimpleCylindrical (equirectangular), Titan sphere, west-positive,
#: coordinates in metres, central meridian 0 deg (0 degW = prime meridian).
#: Longitude increases westward in this convention (matches all rasters).
CANONICAL_CRS_PROJ4: str = (
    "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 "
    f"+a={TITAN_RADIUS_M:.1f} +b={TITAN_RADIUS_M:.1f} "
    "+units=m +no_defs"
)

#: WKT for the Titan geographic CRS used by the Lopes shapefiles.
#: East-positive, deg.  Must be reprojected to canonical before rasterising.
SHAPEFILE_CRS_WKT: str = (
    'GEOGCS["GCS_Titan_2000",'
    'DATUM["D_Titan_2000",'
    'SPHEROID["Titan_2000_IAU_IAG",2575000.0,0.0]],'
    'PRIMEM["Reference_Meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)

#: PDS3 MISSING_CONSTANT for GTDR topography tiles (Howard Zebker, Stanford).
#: This is NOT a NaN -- it is 0xFF7FFFFB interpreted as little-endian float32.
#: Value = -3.4028226550889045e+38  (close to -FLT_MAX).
#: Must be masked by exact-value comparison, NOT np.isnan().
GTDR_MISSING_CONSTANT: float = -3.4028226550889045e+38

#: Nodata value in USGS GeoTIFF mosaic products (confirmed from file inspection).
GEOTIFF_NODATA: float = 0.0


# ---------------------------------------------------------------------------
# Terrain class definitions  (Lopes et al. 2019 six-unit global map)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerrainClass:
    """
    A single Lopes et al. (2019) terrain class.

    Parameters
    ----------
    code:
        Two-letter code used in the ``Meta_Terra`` DBF field.
    integer_label:
        Integer assigned in the canonical raster (1-based).
    name:
        Human-readable name.
    shapefile_stem:
        Filename stem (without extension) of the corresponding shapefile.
    surface_fraction:
        Approximate fraction of Titan's surface covered by this unit.
        Source: Malaska et al. (2025) Titan After Cassini-Huygens Ch.9.
    habitability_weight:
        Prior habitability weight for the Bayesian feature extractor.
        Rationale documented in BayesianPriorConfig below.
    """
    code: str
    integer_label: int
    name: str
    shapefile_stem: str
    surface_fraction: float
    habitability_weight: float


#: Ordered terrain class catalogue.
#: Surface fractions from Malaska et al. (2025) Ch.9:
#:   Plains ~65%, Dunes ~17%, remaining units share ~18%.
TERRAIN_CLASSES: Dict[str, TerrainClass] = {
    "Cr": TerrainClass("Cr", 1, "Craters",   "Craters",   0.01, 0.55),
    "Dn": TerrainClass("Dn", 2, "Dunes",     "Dunes",     0.17, 0.10),
    "Pl": TerrainClass("Pl", 3, "Plains",    "Plains_3",  0.65, 0.20),
    "Ba": TerrainClass("Ba", 4, "Basins",    "Basins",    0.03, 0.30),
    "Mt": TerrainClass("Mt", 5, "Mountains", "Mountains", 0.08, 0.25),
    "Lb": TerrainClass("Lb", 6, "Labyrinth", "Labyrinth", 0.04, 0.35),
    "Lk": TerrainClass("Lk", 7, "Lakes",     "Lakes",     0.02, 0.90),
}

#: Rasterisation priority (higher = drawn on top when polygons overlap).
#: Lakes are most important for habitability; dunes least.
TERRAIN_RASTER_PRIORITY: List[str] = [
    "Dn", "Pl", "Ba", "Mt", "Lb", "Cr", "Lk"
]


# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """
    Specification for one raw input dataset.

    Parameters
    ----------
    name:
        Short identifier (used as output filename stem).
    description:
        Human-readable description including data source.
    url:
        Primary download URL.  If manual steps are required, set to the
        product landing page and populate ``manual_instructions``.
    local_filename:
        Filename to save under ``PipelineConfig.data_dir``.
    file_format:
        One of: ``"geotiff"``, ``"pds3_img"``, ``"shapefile_dir"``,
        ``"parquet"``, ``"netcdf"``, ``"csv"``, ``"json"``,
        ``"synthesised"`` (generated at runtime; acquisition skips it).
    nodata_value:
        Value to treat as missing.  None = use file's embedded nodata.
        For GTDR use ``GTDR_MISSING_CONSTANT``; for GeoTIFFs use 0.0.
    band:
        For multi-band GeoTIFFs: 1-based band index to use.  None = all.
    units:
        Physical units of the data values.
    citation:
        BibTeX key (see references.bib).
    manual_instructions:
        Step-by-step instructions for datasets requiring web login or
        special download procedures.
    sha256:
        Expected SHA-256 hash of the downloaded file, for integrity
        checking.  Empty string = skip verification.
    """
    name: str
    description: str
    url: str
    local_filename: str
    file_format: Literal[
        "geotiff", "pds3_img", "shapefile_dir",
        "parquet", "netcdf", "csv", "json",
        "synthesised",   # generated at runtime; no download needed
    ]
    nodata_value: Optional[float] = None
    band: Optional[int] = None
    units: str = ""
    citation: str = ""
    manual_instructions: str = ""
    sha256: str = ""


def default_dataset_catalogue() -> Dict[str, DatasetSpec]:
    """
    Return the default dataset catalogue for the Titan pipeline.

    All URLs and format details are verified against the actual data
    products inspected during pipeline development.

    Download portals
    ----------------
    USGS Astropedia (GeoTIFF mosaics via Map-a-Planet):
        https://astrogeology.usgs.gov/search?target=titan
        https://astrocloud.wr.usgs.gov/index.php

    Cornell SAR/GTDR archive (direct download, no login):
        https://ecommons.cornell.edu/entities/publication/ed2bc6ea-e0e3-455c-8923-d9188e0dbdd8

    VIMS Nantes portal (CC-BY-4.0):
        https://vims.univ-nantes.fr/

    PDS Atmospheres Node (CIRS, INMS):
        https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/Cassini/

    PDS Plasma Interactions Node (MAG, CAPS):
        https://pds-ppi.igpp.ucla.edu/

    NASA Titan Trek (lake masks, GIS layers):
        https://trek.nasa.gov/titan/

    Mendeley Data (JPL cube files from Rosaly Lopes):
        https://data.mendeley.com/research-data/?query=titan
    """
    return {

        # -- Topography -- Cornell GTDR/GTDE product suite -----------------------
        #
        # All files downloadable from Cornell eCommons (gzip-compressed):
        #   https://data.astro.cornell.edu/RADAR/DATA/GTDR/
        # Also in USGS gtdr-data.zip:
        #   http://astropedia.astrogeology.usgs.gov/download/Titan/Cassini/GTDR/gtdr-data.zip
        #
        # Pipeline DEM priority in _preprocess_topography():
        #   1. GTDE T126 -- Dense interpolated (~90% global, Corlies 2017)  PREFERRED
        #   2. GT0E T126 -- Standard GTDR final mission (~25% coverage)
        #   3. GT0E T077 -- Standard GTDR legacy (~15% coverage, older release)
        #
        # The reader accepts both .IMG and .IMG.gz (Cornell distributes .gz).

        # -- GTDE: Dense interpolated DEM (PREFERRED, ~90% global) ------------
        "gtde_east": DatasetSpec(
            name="gtde_east",
            description=(
                "GTDE east tile -- Dense spline-interpolated global DEM. "
                "~90% of Titan's surface with valid elevation data. "
                "lon 0-180 degW (N090). 2 ppd (22.47 km/px). "
                "Corlies et al. (2017); through flyby T126 (final mission). "
                "From Cornell eCommons (distributed as GTDED00N090_T126_V01.IMG.gz)."
            ),
            url="https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N090_T126_V01.IMG.gz",
            local_filename="GTDED00N090_T126_V01.IMG",
            file_format="pds3_img",
            nodata_value=GTDR_MISSING_CONSTANT,
            units="metres (height above 2575 km sphere)",
            citation="Corlies2017Titan",
            manual_instructions=(
                "Download and gunzip from Cornell:\n"
                "  https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N090_T126_V01.IMG.gz\n"
                "  https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N090_T126_V01.LBL\n"
                "Or extract from USGS gtdr-data.zip:\n"
                "  http://astropedia.astrogeology.usgs.gov/download/Titan/Cassini/GTDR/gtdr-data.zip\n"
                "Note: the pipeline also accepts .IMG.gz directly (auto-decompresses)."
            ),
        ),

        "gtde_west": DatasetSpec(
            name="gtde_west",
            description=(
                "GTDE west tile -- companion to gtde_east. "
                "lon 180-360 degW (N270). Corlies et al. (2017); T126. "
                "From Cornell eCommons (GTDED00N270_T126_V01.IMG.gz)."
            ),
            url="https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N270_T126_V01.IMG.gz",
            local_filename="GTDED00N270_T126_V01.IMG",
            file_format="pds3_img",
            nodata_value=GTDR_MISSING_CONSTANT,
            units="metres (height above 2575 km sphere)",
            citation="Corlies2017Titan",
            manual_instructions=(
                "Download and gunzip from Cornell:\n"
                "  https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N270_T126_V01.IMG.gz\n"
                "  https://data.astro.cornell.edu/RADAR/DATA/GTDR/GTDED00N270_T126_V01.LBL"
            ),
        ),

        # -- GT0E T126: Standard GTDR final mission (~25% coverage) -----------
        "gtdr_east": DatasetSpec(
            name="gtdr_east",
            description=(
                "GT0E east tile -- standard GTDR altimetry + SARtopo tracks. "
                "~25% surface coverage (nadir corridors only). "
                "lon 0-180 degW (N090). 2 ppd. Through flyby T126 (final mission). "
                "Use GTDE tiles for global coverage. "
                "From Cornell eCommons (GT2ED00N090_T126_V01.IMG.gz). "
                "Legacy T077 variant (GT0EB00N090_T077_V01.IMG) also accepted."
            ),
            url="https://data.astro.cornell.edu/RADAR/DATA/GTDR/GT2ED00N090_T126_V01.IMG.gz",
            local_filename="GT2ED00N090_T126_V01.IMG",
            file_format="pds3_img",
            nodata_value=GTDR_MISSING_CONSTANT,
            units="metres (height above 2575 km sphere)",
            citation="Zebker2009",
            manual_instructions=(
                "Download from Cornell (T126 = final mission, preferred):\n"
                "  https://data.astro.cornell.edu/RADAR/DATA/GTDR/GT2ED00N090_T126_V01.IMG.gz\n"
                "  https://data.astro.cornell.edu/RADAR/DATA/GTDR/GT2ED00N090_T126_V01.LBL\n"
                "Legacy T077 variant also accepted if you have GT0EB00N090_T077_V01.IMG."
            ),
        ),

        "gtdr_west": DatasetSpec(
            name="gtdr_west",
            description=(
                "GT0E west tile -- companion to gtdr_east. "
                "lon 180-360 degW (N270). Through flyby T126. "
                "From Cornell eCommons (GT2ED00N270_T126_V01.IMG.gz)."
            ),
            url="https://data.astro.cornell.edu/RADAR/DATA/GTDR/GT2ED00N270_T126_V01.IMG.gz",
            local_filename="GT2ED00N270_T126_V01.IMG",
            file_format="pds3_img",
            nodata_value=GTDR_MISSING_CONSTANT,
            units="metres (height above 2575 km sphere)",
            citation="Zebker2009",
            manual_instructions=(
                "Download from Cornell:\n"
                "  https://data.astro.cornell.edu/RADAR/DATA/GTDR/GT2ED00N270_T126_V01.IMG.gz\n"
                "  https://data.astro.cornell.edu/RADAR/DATA/GTDR/GT2ED00N270_T126_V01.LBL"
            ),
        ),

        # -- SAR Backscatter (geomorphology proxy) -----------------------------
        "sar_mosaic": DatasetSpec(
            name="sar_mosaic",
            description=(
                "Cassini RADAR SAR + HiSAR global mosaic at 351 m/pixel "
                "(128 ppd), through flyby T104 (Jan 2015). SimpleCylindrical "
                "projection, west-positive clon180. Float32. Nodata=0.0. "
                "Produced by USGS Astrogeology / JPL."
            ),
            url=(
                "https://astrogeology.usgs.gov/search/map/"
                "titan_cassini_sar_hisar_global_mosaic_351m"
            ),
            local_filename="Titan_SAR_HiSAR_MosaicThru_T104_Jan2015_clon180_128ppd.tif",
            file_format="geotiff",
            nodata_value=GEOTIFF_NODATA,
            units="SAR radar backscatter (float32, log-stretched)",
            citation="Elachi2005",
            manual_instructions=(
                "Visit the Astropedia page above and click 'Map a Planet' "
                "to reproject and download as GeoTIFF. "
                "Alternatively the Cornell archive contains updated individual "
                "swaths: https://hdl.handle.net/1813/116147"
            ),
        ),

        # -- ISS Optical Mosaic ------------------------------------------------
        "iss_mosaic_450m": DatasetSpec(
            name="iss_mosaic_450m",
            description=(
                "Cassini ISS 938 nm near-global controlled mosaic at 450 m/pixel. "
                "Photogrammetrically registered, +45 deg to -65 deg latitude. "
                "Best positional accuracy for correlative studies."
            ),
            url=(
                "https://astrogeology.usgs.gov/search/map/"
                "titan_cassini_iss_near_global_mosaic_450m"
            ),
            local_filename="Titan_ISS_NearGlobal_450m.tif",
            file_format="geotiff",
            nodata_value=GEOTIFF_NODATA,
            units="surface reflectance (938 nm, normalised)",
            citation="Roatsch2016",
            manual_instructions=(
                "Download via Map-a-Planet at astrocloud.wr.usgs.gov. "
                "Also at Cornell: https://hdl.handle.net/1813/116147"
            ),
        ),

        # -- VIMS + ISS Global Mosaic (Caltech/JPL) ----------------------------
        #
        # The standalone VIMS mosaic is NOT available as a direct GeoTIFF
        # download from vims.univ-nantes.fr.  Instead, the pipeline uses the
        # published VIMS+ISS combined mosaic by Seignovert et al. (2019),
        # which uses the same VIMS band ratios as Le Mouelic et al. (2019) but
        # blends in the seamless ISS 938 nm map (PIA22770) to fill coverage gaps.
        #
        # Band layout (verified from Titan_VIMS-ISS.tif.aux.xml):
        #   Band 1: 1.59/1.27 umm  + 938 nm ISS  <- tholin proxy (organic_abundance)
        #   Band 2: 2.03/1.27 umm  + 938 nm ISS  <- surface 2 umm window
        #   Band 3: 1.27/1.08 umm  + 938 nm ISS  <- water-ice indicator
        #
        # This is BETTER than the pure VIMS mosaic for the pipeline because:
        #   1. Direct download URL -- no login, no contact required
        #   2. ISS blending fills high-latitude and low-incidence gaps in VIMS
        #   3. Same band-ratio definitions as the Nantes VIMS portal products
        #   4. BSD-3-Clause license (compatible with research use)
        #
        # Derived from:
        #   VIMS: Le Mouelic et al. (2019) doi:10.1016/j.icarus.2018.09.017
        #   ISS:  PIA22770, Cassini Imaging Team (Dec 2018)
        #
        # DOI: 10.22002/D1.1173
        # MD5: 42a94d36b9dbe6a63262c3a0cfeacc47
        "vims_mosaic": DatasetSpec(
            name="vims_mosaic",
            description=(
                "VIMS + ISS combined global mosaic (Seignovert et al. 2019, "
                "CaltechDATA doi:10.22002/D1.1173). 3-band GeoTIFF: "
                "Band 1 = 1.59/1.27 umm + 938 nm (tholin proxy); "
                "Band 2 = 2.03/1.27 umm + 938 nm (surface); "
                "Band 3 = 1.27/1.08 umm + 938 nm (water ice). "
                "ISS blending fills VIMS coverage gaps at high latitudes. "
                "Same band ratios as Le Mouelic et al. (2019). "
                "BSD-3-Clause licence. 144.2 MB GeoTIFF."
            ),
            url=(
                "https://data.caltech.edu/records/8q9an-yt176/files/"
                "Titan_VIMS-ISS.tif?download=1"
            ),
            local_filename="Titan_VIMS-ISS.tif",
            file_format="geotiff",
            nodata_value=None,   # no nodata embedded; NaN/zero handled in preprocessing
            band=1,              # Band 1: 1.59/1.27 umm = tholin proxy for organic_abundance
            units="VIMS band ratio + ISS 938 nm (dimensionless)",
            citation="Seignovert2019CaltechDATA",
            sha256="",           # MD5 known (see description); SHA-256 not published
            manual_instructions="",  # direct download -- no login required
        ),

        # -- VIMS+ISS ENVI header (companion to vims_mosaic) -------------------
        # The .hdr file contains the ENVI projection metadata for Titan_VIMS-ISS.tif.
        # 479 bytes; downloaded alongside the GeoTIFF for provenance.
        "vims_mosaic_hdr": DatasetSpec(
            name="vims_mosaic_hdr",
            description=(
                "ENVI header for Titan_VIMS-ISS.tif. "
                "Contains projection/CRS metadata. 479 bytes."
            ),
            url=(
                "https://data.caltech.edu/records/8q9an-yt176/files/"
                "Titan_VIMS-ISS.hdr?download=1"
            ),
            local_filename="Titan_VIMS-ISS.hdr",
            file_format="json",   # treated as plain text auxiliary file
            units="N/A",
            citation="Seignovert2019CaltechDATA",
            manual_instructions="",
        ),

        # -- VIMS Spatial Footprint Index (parquet) ----------------------------
        # -- VIMS Spatial Footprint Index (parquet) ----------------------------
        #
        # Accepted filenames (searched in data_dir in this order):
        #   1. vims_footprints.parquet       canonical full file (~5.4M rows, 227 MB)
        #   2. vims_sample_1000rows.parquet  the 1,000-row sample uploaded at project
        #                                    start -- fully usable for development and
        #                                    testing; covers sparse global sampling
        #   3. vims_*.parquet                any parquet with VIMS-matching schema
        #
        # Use  --vims-parquet PATH  to bypass discovery and point directly to any file.
        #
        # The sample (43 KB) is sufficient for spatial-coverage rasters at the
        # default 4490 m/px resolution but will undercount coverage in regions
        # only observed during later flybys.  The full file gives reliable global
        # coverage density and best-resolution maps.
        "vims_footprint": DatasetSpec(
            name="vims_footprint",
            description=(
                "VIMS spatial footprint index in parquet format. "
                "One row per VIMS pixel: id (SCLK format), flyby, "
                "obs_start/end, altitude (km), lon ( degW, 0-360), "
                "lat (deg), res (km/px). The 'id' column maps directly to "
                "cube download URLs on vims.univ-nantes.fr. "
                "Full file: ~5.4 million rows, 227 MB. "
                "Sample: 1,000 rows, 43 KB (vims_sample_1000rows.parquet). "
                "Both are accepted; use --vims-parquet PATH to specify directly."
            ),
            url="https://vims.univ-nantes.fr/",
            local_filename="vims_footprints.parquet",
            file_format="parquet",
            units="mixed (see column descriptions)",
            citation="LeMouelic2019",
            manual_instructions=(
                "Use --vims-parquet PATH to point to your file directly.\n"
                "Accepted filenames (auto-discovered in data_dir):\n"
                "  vims_footprints.parquet       (canonical full name)\n"
                "  vims_sample_1000rows.parquet  (1,000-row sample)\n"
                "  vims_*.parquet                (any VIMS-prefixed parquet)\n"
                "\n"
                "To obtain the full 227 MB catalogue:\n"
                "  Contact Stephane Le Mouelic (LPG Nantes), or build\n"
                "  from pyVIMS: https://github.com/seignovert/pyvims\n"
                "\n"
                "Individual cubes are downloadable without the parquet:\n"
                "  Calibrated: https://vims.univ-nantes.fr/cube/C{id}_ir.cub\n"
                "  Navigation: https://vims.univ-nantes.fr/cube/N{id}_ir.cub\n"
                "  Raw:        https://vims.univ-nantes.fr/cube/v{id}.qub\n"
                "\n"
                "Cube index: https://vims.univ-nantes.fr/target/titan/targeted"
            ),
        ),

        # -- Geomorphology Shapefiles (Lopes et al. 2019/2020) ----------------
        "geomorphology_shapefiles": DatasetSpec(
            name="geomorphology_shapefiles",
            description=(
                "Lopes et al. (2019/2020) global geomorphologic map. "
                "Six terrain classes: Craters(Cr), Dunes(Dn), Plains(Pl), "
                "Labyrinth(Lb), Mountains(Mt), Basins(Ba). "
                "One shapefile per class. PolygonM geometry. "
                "CRS: GCS_Titan_2000, east-positive deg. "
                "JPL cube files provided by Rosaly Lopes."
            ),
            url="https://data.mendeley.com/research-data/?query=titan",
            local_filename="geomorphology_shapefiles/",
            file_format="shapefile_dir",
            units="terrain class labels",
            citation="Lopes2019",
            manual_instructions=(
                "Shapefiles provided by JPL (Rosaly Lopes). "
                "Also check Mendeley Data: "
                "https://data.mendeley.com/research-data/?query=titan "
                "Place all .shp/.dbf/.prj/.shx files in a single directory. "
                "Expected files: Craters.shp, Dunes.shp, Plains_3.shp, "
                "Labyrinth.shp, Mountains.shp, Basins.shp, Lakes.shp"
            ),
        ),

        # -- Fluvial Channel Map (Miller et al. 2021) --------------------------
        "titan_channels": DatasetSpec(
            name="titan_channels",
            description=(
                "Global fluvial channel map of Titan from Cassini SAR. "
                "Miller et al. (2021). Each feature is a channel segment with "
                "attributes: certainty, stream order, width, length. "
                "Geometry in east-positive longitude. "
                "Source: Hayes Research Group, Cornell University. "
                "DOI: 10.7298/m4dv-gv95 (companion to titan_topo)"
            ),
            url="https://hayesresearchgroup.com/data-products/",
            local_filename="geomorphology_shapefiles/global_channels.shp",
            file_format="shapefile_dir",
            units="binary channel presence (rasterised to 0/1)",
            citation="Miller2021channels",
            manual_instructions=(
                "Download titan_channels_miller.zip from "
                "https://hayesresearchgroup.com/data-products/ and extract "
                "global_channels.shp and companions into "
                "data/raw/geomorphology_shapefiles/."
            ),
        ),

        # -- Corlies 2017 Globally Interpolated Topography (gap-filler) ----------
        # Source: Hayes Research Group (hayesresearchgroup.com/data-products/)
        # titan_topo_corlies.zip -> hayes_topo/topo_4PPD_interp.cub
        # This is the GLOBALLY INTERPOLATED version of the Corlies et al. (2017) GRL
        # topographic map (Geophys. Res. Lett. 44, 11754). It uses all Cassini RADAR
        # data (altimetry + SARtopo + stereophotogrammetry) with radial basis function
        # interpolation to fill the ~91% of Titan not directly measured.
        # Resolution: 4 ppd (~11 km/px). Coverage: 100% global.
        # Role: gap-filler for GTDE NaN pixels (~10% of globe); used by
        # preprocessing._preprocess_topography() automatically if present.
        # Not used as primary source (GTDE Cornell is preferred at 8ppd).
        "corlies_interp_topo": DatasetSpec(
            name="corlies_interp_topo",
            description=(
                "Globally interpolated Titan topography -- Corlies et al. (2017) "
                "Geophys. Res. Lett. 44, 11754. All Cassini RADAR data (altimetry, "
                "SARtopo, stereo). RBF interpolation gives 100%% global coverage "
                "at 4 ppd (~11 km/px). ISIS3 .cub format. "
                "Used as gap-filler for GTDE NaN pixels in preprocessing. "
                "NOT YET USED DIRECTLY -- optional enhancement for 100%% topographic coverage."
            ),
            url="https://hayesresearchgroup.com/data-products/",
            local_filename="hayes_topo/topo_4PPD_interp.cub",
            file_format="synthesised",   # optional; auto-detected by preprocessing
            units="metres above 2575 km sphere",
            citation="Corlies2017",
            manual_instructions=(
                "Download titan_topo_corlies.zip from "
                "https://hayesresearchgroup.com/data-products/ "
                "and extract topo_4PPD_interp.cub to data/raw/hayes_topo/. "
                "The preprocessing pipeline will then automatically use it "
                "to fill NaN pixels in the GTDE topography, pushing "
                "topographic coverage from ~90%% to ~100%%."
            ),
        ),

        # -- CIRS Atmospheric Temperature --------------------------------------
        #
        # NOTE: There is NO pre-packaged cirs_temperature.nc on PDS Atmospheres.
        # The PDS CIRS archive holds raw spectra only (COCIRS_0xxx volumes).
        # Published temperature retrievals exist in two forms:
        #
        #   1. Achterberg et al. (2008) Icarus 194, 263 -- latitude/pressure maps
        #      of the stratosphere from the nominal mission (75 degS-75 degN, 10-0.001 mbar).
        #      Supplementary data NOT in a public repository; contact first author.
        #
        #   2. Jennings et al. (2019) ApJL 877, L8 -- surface brightness temperatures
        #      in 10 deg latitude bins, full mission 2004-2017.
        #      DOI: 10.3847/2041-8213/ab1f91
        #      Supplementary table available from the journal; can be converted to NetCDF.
        #
        # The pipeline degrades gracefully when this file is absent -- features that
        # depend on it (acetylene_energy, methane_cycle) return NaN rather than failing.
        # Place the NetCDF at data/raw/cirs_temperature.nc to enable those features.
        "cirs_temperature": DatasetSpec(
            name="cirs_temperature",
            description=(
                "Titan surface brightness temperature map synthesised at runtime "
                "from the Jennings et al. (2019) ApJL 877, L8 analytical formula. "
                "No download required -- generated by _synthesise_cirs_temperature() "
                "in titan/preprocessing.py using the embedded formula from "
                "titan/atmospheric_profiles.py. "
                "Formula: T(L,Y) = (93.53-0.095Y)*cos[(L+0.85-3.2Y)*(0.0029-0.00006Y)] "
                "Valid for L=-90->+90 deg, Y=-4.9->+8.1 yr from equinox (2004-2017). "
                "sigma=0.4 K fit, sourced from Schinder et al. (2011) and HASI anchor."
            ),
            url="https://doi.org/10.3847/2041-8213/ab1f91",
            local_filename="cirs_temperature.nc",
            file_format="synthesised",
            units="Kelvin",
            citation="Jennings2019",
            manual_instructions=None,
        ),

        # -- INMS Atmospheric Composition --------------------------------------
        # NOTE: inms_composition is a forward stub -- not yet used by any feature.
        # Raw INMS data is per-flyby PDS3 binary (REDRs), not a simple CSV.
        # The correct PDS page is inst-inms.html (not cassini-inms.html).
        # Marked as "not_used" so acquisition never warns about it.
        "inms_composition": DatasetSpec(
            name="inms_composition",
            description=(
                "Cassini INMS upper-atmosphere neutral composition (~950-1200 km). "
                "N2, CH4, HCN, benzene, tholins. Waite et al. (2007). "
                "NOT YET USED BY ANY PIPELINE FEATURE -- forward stub only. "
                "Raw data: PDS Level 1A REDRs at "
                "https://pds-atmospheres.nmsu.edu/data_and_services/"
                "atmospheres_data/Cassini/inst-inms.html "
                "Titan Neutrals Guide (PDF) provides calibrated composition tables. "
                "Calibration: Teolis et al. (2015) sensitivity model required."
            ),
            url=(
                "https://pds-atmospheres.nmsu.edu/data_and_services/"
                "atmospheres_data/Cassini/inst-inms.html"
            ),
            local_filename="inms_composition.csv",
            file_format="synthesised",   # stub: not downloaded, not used
            units="number density (cm^-3) per species",
            citation="Waite2007",
            manual_instructions=None,
        ),

        # -- Magnetosphere / Plasma (MAG) -------------------------------------
        # NOTE: magnetosphere_mag is a forward stub -- not yet used by any feature.
        # The calibrated MAG archive (urn:nasa:pds:cassini-mag-cal) at the PDS
        # Plasma Interactions Node (pds-ppi.igpp.ucla.edu) contains ~5,000 daily
        # ASCII .TAB files for the full mission. There is no pre-made Titan-flyby
        # CSV; extraction requires cross-referencing the flyby timeline and
        # filtering by encounter window for each of the 127 Titan encounters.
        "magnetosphere_mag": DatasetSpec(
            name="magnetosphere_mag",
            description=(
                "Cassini MAG calibrated magnetometer data (Titan flybys). "
                "NOT YET USED BY ANY PIPELINE FEATURE -- forward stub only. "
                "Titan has no intrinsic field (Backes et al. 2005); data shows "
                "Saturn's external field modified by Titan's ionosphere. "
                "Archive: PDS PPI urn:nasa:pds:cassini-mag-cal (~5,000 daily TAB files). "
                "No pre-made Titan-flyby CSV exists; extraction requires filtering "
                "by flyby timeline across the full daily time-series archive."
            ),
            url="https://pds-ppi.igpp.ucla.edu/",
            local_filename="mag_titan_flybys.csv",
            file_format="synthesised",   # stub: not downloaded, not used
            units="nT (nanotesla)",
            citation="Backes2005",
            manual_instructions=None,
        ),

        # -- Gravity / Tidal Love Number ----------------------------------------
        "gravity_k2": DatasetSpec(
            name="gravity_k2",
            description=(
                "Titan's tidal Love number k2=0.589+/-0.150 (Iess et al. 2012), "
                "indicating a global subsurface water ocean. "
                "Stored as a JSON scalar -- applied globally as a constant "
                "habitability modifier in the Bayesian model."
            ),
            url="https://doi.org/10.1126/science.1219631",
            local_filename="gravity_k2.json",
            file_format="json",
            units="dimensionless (Love number)",
            citation="Iess2012",
            manual_instructions=(
                "Create this file manually with the published value: "
                '{"k2": 0.589, "k2_uncertainty": 0.150, '
                '"source": "Iess et al. 2012, Science 337, 457"}'
            ),
        ),

    }


# ---------------------------------------------------------------------------
# Bayesian prior configuration
# ---------------------------------------------------------------------------

@dataclass
class BayesianPriorConfig:
    """
    Prior probability parameters for the pixel-wise habitability model.

    Model overview
    --------------
    We model Titan habitability as a latent variable H in [0, 1] per pixel.
    Eight observable features f_i in [0, 1] are derived from the canonical
    data stack.  The posterior is:

        P(H | f_1,...,f_8) proportional to P(f_1,...,f_8 | H) . P(H)

    The sklearn backend approximates this using isotonically-calibrated
    Gaussian Naive Bayes, treating each feature as conditionally
    independent given H (a known simplification, but tractable and
    appropriate given the sparse, multi-source nature of the data).

    PyMC and NumPyro backends implement a full hierarchical logistic
    model with feature correlations.

    Feature definitions and prior justifications
    --------------------------------------------
    All prior means are informed by specific published measurements.

    Feature 1 -- liquid_hydrocarbon  (weight 0.25)
        Definition: Probability pixel has surface liquid hydrocarbon.
        Primary data: lake_mask from SAR; secondary: VIMS 5 umm emissivity.
        Prior mean 0.02: Lakes cover ~1.5-2% of Titan globally
        (Stofan et al. 2007; Hayes et al. 2008).  Polar pixels ~0.4.
        Rationale: Liquid hydrocarbon is the non-aqueous solvent for
        surface life (McKay & Smith 2005; McKay 2016).
        Highest weight because solvent availability is prerequisite.

    Feature 2 -- organic_abundance  (weight 0.20)
        Definition: Surface organic (tholin) compound abundance proxy.
        Primary data:   VIMS+ISS mosaic Band 1 (1.59/1.27 umm band ratio;
                        Seignovert et al. 2019).  Covers ~50% of globe.
        Gap-fill data:  Geomorphology-class scores from Lopes et al. (2019)
                        terrain map -- global 100% coverage via published VIMS
                        spectral studies of each terrain type.
                        Dunes=0.82, Plains=0.68, Basins=0.58,
                        Labyrinth=0.55, Craters=0.35, Mountains=0.25,
                        Lakes=0.05.
        Prior mean 0.70: ~82% of Titan's surface is organic terrain
        (plains 65% + dunes 17%; Lopes et al. 2019 Nature Astronomy).
        NOTE: ISS 938nm broadband is NOT used as gap-filler because it
        measures different physical quantities to the VIMS band ratio
        (absolute raw values differ by factor ~3000), producing an
        irremovable seam regardless of normalisation.  The geomorphology-
        based approach is scientifically superior and seam-free.
        Rationale: Tholins are primary chemical substrate for prebiotic
        and biotic chemistry (Neish et al. 2018; Meyer-Dombard 2025 Ch.14).
        High weight: without organics, no substrate for life.

    Feature 3 -- acetylene_energy  (weight 0.20)
        Definition: Chemical energy gradient proxy.
        Primary: SAR-dark surface (low C2H2 surface deposit -> consumed?)
        Secondary: surface-atmosphere H2 gradient from Strobel (2010).
        Prior mean 0.35: Strobel (2010) found a downward H2 flux consistent
        with surface consumption, though Teolis et al. (2015) recalibration
        of the INMS instrument may reduce its magnitude (Strobel 2022 notes
        the original INMS calibration may not need the 2.2x correction).
        The disequilibrium signal is real but its amplitude is uncertain.
        McKay & Smith (2005) compute DeltaG = 334 kJ/mol for C2H2 + 3H2 -> CH4.
        Yanez et al. (2024) compute acetylenotrophy energy 69-78 kJ/mol C.
        Rationale: Chemical disequilibrium is the energy source for life
        in absence of sunlight (Schulze-Makuch & Grinspoon 2005).
        High weight: energy is prerequisite for metabolism.

    Feature 4 -- methane_cycle  (weight 0.15)
        Definition: Active methane-cycle proxy.
        Derived from three components (blend weights):
          vims_coverage       (0.50) -- VIMS observation density (flybys target
                                       active methane-cycle regions)
          cirs_temperature    (0.25) -- Meridional |dT/dlat| gradient from Jennings
                                       et al. (2019); steep gradient -> strong
                                       evaporation differential -> active cycling
          lat_weight          (0.25) -- Gaussian prior at +/-45 deg (Mitchell & Lora 2016)
        Without vims_coverage: cirs_temperature 60%, lat_weight 40%.
        Without cirs_temperature: vims_coverage 60%, lat_weight 40%.
        Prior mean 0.40: Methane cycle active across mid-latitudes
        (Turtle et al. 2011; Mitchell & Lora 2016; Jennings et al. 2019).
        Rationale: Temperature gradient from Jennings (2019) provides
        physically-grounded spatial weighting without an external data file.

    Feature 5 -- surface_atm_interaction  (weight 0.08)
        Definition: Surface-atmosphere chemical exchange intensity.
        Derived from three components (weights):
          topographic_slope  (0.30) -- gradient-driven runoff and exposure
          lake_margin        (0.40) -- evaporation/condensation hotspots
          channel_density    (0.30) -- Miller et al. (2021) global channel map
        Prior mean 0.35: Exchange zones active at lake margins, channel
        networks, and topographic breaks (Hayes 2016; Miller et al. 2021).
        If channel_density layer is absent, weight is redistributed between
        slope and lake_margin proportionally.
        Rationale: Exchange zones provide chemical gradients analogous
        to Earth coastal/riparian habitats (most biologically productive).

    Feature 6 -- topographic_complexity  (weight 0.06)
        Definition: Local DEM roughness (std-dev in sliding window).
        Primary data: GTDR elevation.
        Prior mean 0.25: Terrain roughness correlated with dissected/
        hummocky units, which have highest water-ice content (Lopes 2019).
        Rationale: Complex terrain -> more micro-environments -> more
        chemical gradient opportunities.

    Feature 7 -- geomorphologic_diversity  (weight 0.04)
        Definition: Shannon diversity of terrain classes in local window.
        Primary data: rasterised Lopes et al. shapefiles.
        Prior mean 0.30: Multiple terrain units co-occur at boundaries
        (Malaska et al. 2025; Lopes 2019).
        Rationale: Ecotone effect -- boundaries between terrain types
        have highest biodiversity on Earth; analogous logic applied here.

    Feature 8 -- subsurface_ocean  (weight 0.02)
        Definition: Proxy for subsurface-surface exchange via cryovolcanism
        and past-liquid-water interaction.
        Primary: SAR bright annuli (radar-bright ring structures) interpreted
        as evidence of past liquid water contact -- impact melt ponds, lava flow
        fronts, or cryovolcanic ejecta blankets where liquid interacted with
        organics (D4 reasoning; Lopes et al. 2007; Wood et al. 2010).
        Secondary: global constant k2=0.589 (Iess et al. 2012) confirming a
        subsurface water-ammonia ocean at 55-80 km depth.
        Prior mean 0.03 (revised down from earlier 0.10): The subsurface ocean
        is confirmed but its surface expression is heavily time-gated.
        The ~3.5 Gya past-epoch constraint (D1) means organic-water contact
        products are ancient and largely degraded or buried; the future
        habitability window (100-400 Myr; D2) is not yet reached. The
        effective present-day surface habitability from this pathway is
        therefore low. (See HabitabilityWindowConfig for temporal parameters.)
        Rationale: Affholder et al. (2025) demonstrate glycine fermentation
        viable in Titan's subsurface ocean; cryovolcanic zones and bright
        annular SAR structures are the primary surface evidence for
        past/present ocean-surface exchange pathways.
        Low weight: surface expression poorly constrained.

    Weights sum to 1.0.

    References
    ----------
    Affholder et al. (2021) doi:10.1038/s41550-021-01372-6
    Affholder et al. (2025) doi:10.3847/PSJ/addb09
    Catling et al. (2018)   doi:10.1089/ast.2017.1737
    McKay & Smith (2005)    doi:10.1016/j.icarus.2005.05.018
    McKay (2016)            doi:10.3390/life6010008
    Yanez et al. (2024)     doi:10.1016/j.icarus.2024.115969
    Strobel (2010)          doi:10.1016/j.icarus.2010.02.009
    Iess et al. (2012)      doi:10.1126/science.1219631
    Neish et al. (2018)     doi:10.1089/ast.2017.1758
    Schulze-Makuch & Grinspoon (2005) doi:10.1089/ast.2005.5.560
    Stofan et al. (2007)    doi:10.1038/nature05608
    Hayes et al. (2008)     doi:10.1029/2007GL032324
    Lopes et al. (2019)     doi:10.1038/s41550-019-0917-6
    Malaska et al. (2025)   Titan After Cassini-Huygens Ch.9
    Meyer-Dombard et al.(2025) Titan After Cassini-Huygens Ch.14
    """

    # -- Feature weights ----------------------------------------------------
    # Must sum to 1.0 (enforced by validate())
    weight_liquid_hydrocarbon:      float = 0.25
    weight_organic_abundance:       float = 0.20
    weight_acetylene_energy:        float = 0.20
    weight_methane_cycle:           float = 0.15
    weight_surface_atm_interaction: float = 0.08
    weight_topographic_complexity:  float = 0.06
    weight_geomorphologic_diversity:float = 0.04
    weight_subsurface_ocean:        float = 0.02

    # -- Prior means (Beta distribution: alpha = mean x kappa) ---------------------
    prior_mean_liquid_hydrocarbon:      float = 0.02
    prior_mean_organic_abundance:       float = 0.60
    prior_mean_acetylene_energy:        float = 0.30
    prior_mean_methane_cycle:           float = 0.40
    prior_mean_surface_atm_interaction: float = 0.35
    prior_mean_topographic_complexity:  float = 0.25
    prior_mean_geomorphologic_diversity:float = 0.30
    prior_mean_subsurface_ocean:        float = 0.03

    # -- Beta prior concentration -------------------------------------------
    # kappa = alpha + beta in Beta(alpha, beta).  Higher kappa = more confident prior.
    # kappa=5: weakly informative (appropriate given Titan data uncertainty).
    # kappa=2: very diffuse (use for poorly constrained features).
    # kappa=20: moderately confident (use for well-measured quantities like lakes).
    beta_concentration_default: float = 5.0
    beta_concentration_liquid:  float = 20.0   # lakes well-measured by SAR
    beta_concentration_organics:float = 8.0    # VIMS coverage reasonable

    # -- Likelihood model parameters ----------------------------------------
    # For sklearn GNB: Gaussian variance per class
    gnb_var_smoothing: float = 1e-9

    # For PyMC / NumPyro logistic: sigmoid sharpness
    likelihood_sharpness: float = 6.0

    # -- Positive training label threshold ---------------------------------
    # Pixels with weighted feature sum > this threshold are used as
    # "habitable" training examples for the sklearn backend.
    positive_label_threshold: float = 0.35
    # Threshold for creating soft labels from the weighted feature score.
    # At 0.55 (old value) only best-case pixels score positive (degenerate).
    # Expected prior mean = 0.30; threshold = 0.35 keeps positives rare but
    # numerically meaningful (~top 5-15% of pixels score positive).

    def feature_weights(self) -> Dict[str, float]:
        """Return ordered dict of feature name -> weight."""
        return {
            "liquid_hydrocarbon":      self.weight_liquid_hydrocarbon,
            "organic_abundance":       self.weight_organic_abundance,
            "acetylene_energy":        self.weight_acetylene_energy,
            "methane_cycle":           self.weight_methane_cycle,
            "surface_atm_interaction": self.weight_surface_atm_interaction,
            "topographic_complexity":  self.weight_topographic_complexity,
            "geomorphologic_diversity":self.weight_geomorphologic_diversity,
            "subsurface_ocean":        self.weight_subsurface_ocean,
        }

    def prior_means(self) -> Dict[str, float]:
        """Return ordered dict of feature name -> prior mean."""
        return {
            "liquid_hydrocarbon":      self.prior_mean_liquid_hydrocarbon,
            "organic_abundance":       self.prior_mean_organic_abundance,
            "acetylene_energy":        self.prior_mean_acetylene_energy,
            "methane_cycle":           self.prior_mean_methane_cycle,
            "surface_atm_interaction": self.prior_mean_surface_atm_interaction,
            "topographic_complexity":  self.prior_mean_topographic_complexity,
            "geomorphologic_diversity":self.prior_mean_geomorphologic_diversity,
            "subsurface_ocean":        self.prior_mean_subsurface_ocean,
        }

    def validate(self) -> None:
        """Raise ValueError if weights do not sum to 1.0 +/- 0.01."""
        total = sum(self.feature_weights().values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"BayesianPriorConfig weights must sum to 1.0, got {total:.4f}. "
                "Adjust weight_* fields."
            )
        for name, w in self.feature_weights().items():
            if not (0.0 <= w <= 1.0):
                raise ValueError(f"Weight '{name}' = {w} is outside [0, 1].")

    # -- Convenience vector accessors (used by bayesian backends) --------------
    # These return ordered lists matching feature_weights() / prior_means() order.

    def weight_vector(self) -> List[float]:
        """Return feature weights as an ordered list (same order as prior_means)."""
        return list(self.feature_weights().values())

    def prior_mean_vector(self) -> List[float]:
        """Return prior means as an ordered list (same order as feature_weights)."""
        return list(self.prior_means().values())

    @property
    def beta_concentration(self) -> float:
        """Default Beta distribution concentration parameter (alias for beta_concentration_default)."""
        return self.beta_concentration_default


@dataclass
class HabitabilityWindowConfig:
    """
    Temporal parameters bounding Titan's habitability window.

    These parameters affect how the subsurface ocean and past-liquid-water
    features are interpreted and weighted in the Bayesian model.

    Design decisions
    ----------------
    D1 -- Past epoch (configurable, default 3.5 Gya)
        Titan's surface last had widespread access to liquid water ~3.5 Gya
        ago (analogous to the Late Heavy Bombardment epoch on Earth and Mars).
        Impact melt ponds, cryovolcanic flows, and large-basin flooding events
        during this epoch allowed liquid water to contact the rich organic
        inventory, potentially initiating prebiotic chemistry.
        The SAR bright annuli (ring structures) around impact basins are
        interpreted as preserved ejecta blanket / impact melt front signatures
        from this epoch (see D4 in features.py).
        Default 3.5 Gya; set lower (e.g. 0.5 Gya) for optimistic estimates
        or higher (e.g. 4.0 Gya) for conservative ones.

        References: Neish & Lorenz (2012) doi:10.1016/j.pss.2011.12.004
                    Artemieva & Lunine (2003) doi:10.1016/S0019-1035(02)00039-9
                    Wood et al. (2010) Icarus 206, 334-344

    D2 -- Future habitability window (configurable, 100-400 Myr default)
        As the Sun increases in luminosity (~10% per Gyr, standard main-
        sequence evolution), Titan's surface temperature will eventually warm
        sufficiently for liquid water to exist transiently or persistently.
        Recent estimates (Lorenz et al. 1997; McKay 2016) place this window
        at ~1.0-2.0 Gyr from now, but more recent Solar evolution models
        with improved Titan atmospheric radiative transfer constrain the
        ONSET to 100-400 Myr from now (Lunine & Lorenz 2009 review;
        Horst 2017; Tobie et al. 2020 updated).

        ASSUMPTION -- uniform global warming:
        This implementation assumes that solar-driven warming is spatially
        uniform across Titan's surface (i.e., the temperature increment per
        unit time is the same at all latitudes/longitudes). This is a
        deliberate simplification -- in reality polar regions will warm
        differently from equatorial ones due to Titan's obliquity and
        atmospheric dynamics. The uniform warming assumption is flagged here
        and should be revisited with GCM output when available.

        Under this assumption, the probability that any given pixel is
        within the future habitability window is modelled as uniform over
        [future_window_min_myr, future_window_max_myr], i.e., a uniform
        prior on the onset time:

            P(onset <= t) = (t - min) / (max - min)  for t in [min, max]

        This enters the `subsurface_ocean` feature as a multiplicative
        temporal prior on the probability of future surface liquid water.

        References: Lorenz et al. (1997) doi:10.1006/icar.1997.5647
                    Lunine & Lorenz (2009) Annual Rev. Earth Planet. Sci.
                    Horst (2017) doi:10.1002/2016JE005240
                    Tobie et al. (2020) Titan After Cassini-Huygens Ch.3

    Parameters
    ----------
    past_liquid_water_epoch_gya:
        Age (Gya before present) of the last major epoch of widespread
        liquid water on Titan's surface.  Default 3.5 Gya.
    future_window_min_myr:
        Earliest onset (Myr from now) of the future warm epoch during which
        liquid water will return to Titan's surface.  Default 100 Myr.
    future_window_max_myr:
        Latest onset (Myr from now) of the future warm epoch.  Default 400 Myr.
    assume_uniform_warming:
        If True (default), solar warming increment per unit time is applied
        uniformly over Titan's surface.  Set False to disable the future-
        window prior contribution (e.g., for present-day-only analysis).
    """
    past_liquid_water_epoch_gya: float = 3.5
    future_window_min_myr:       float = 100.0
    future_window_max_myr:       float = 400.0
    assume_uniform_warming:      bool  = True

    def future_window_centre_myr(self) -> float:
        """Midpoint of the future habitability window in Myr."""
        return (self.future_window_min_myr + self.future_window_max_myr) / 2.0

    def future_window_width_myr(self) -> float:
        """Width of the future habitability window in Myr."""
        return self.future_window_max_myr - self.future_window_min_myr

    def temporal_prior_weight(self) -> float:
        """
        Scalar weight representing the relative contribution of the
        future habitability window to the subsurface ocean feature.

        Under the uniform warming assumption, this is proportional to
        the inverse of the window width (narrower window = higher
        probability density = higher weight).

        A 300 Myr window (100-400 Myr) maps to weight =~ 0.14.
        """
        if not self.assume_uniform_warming:
            return 0.0
        # Normalise: 1 Gyr reference window -> weight 1.0
        return 1000.0 / self.future_window_width_myr()

    def validate(self) -> None:
        """Raise ValueError on inconsistent temporal parameters."""
        if self.past_liquid_water_epoch_gya <= 0:
            raise ValueError(
                f"past_liquid_water_epoch_gya must be > 0, "
                f"got {self.past_liquid_water_epoch_gya}"
            )
        if self.future_window_min_myr <= 0:
            raise ValueError(
                f"future_window_min_myr must be > 0, "
                f"got {self.future_window_min_myr}"
            )
        if self.future_window_max_myr <= self.future_window_min_myr:
            raise ValueError(
                f"future_window_max_myr ({self.future_window_max_myr}) must be "
                f"> future_window_min_myr ({self.future_window_min_myr})"
            )


# ---------------------------------------------------------------------------
# Master pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Master configuration object for the Titan Habitability Pipeline.

    Create one instance and pass it to every pipeline stage.

    Parameters
    ----------
    data_dir:
        Root directory for downloaded raw files.
    processed_dir:
        Directory for canonically reprojected / resampled outputs.
    output_dir:
        Directory for Bayesian results, figures, and reports.
    bayesian_backend:
        Which probabilistic programming library to use as default.
        ``"sklearn"`` (default), ``"pymc"``, or ``"numpyro"``.
    canonical_res_m:
        Output grid pixel size in metres.
        Default 4490 m (~0.1 deg/px, 10 ppd) -- a reasonable compromise
        between the 351 m SAR mosaic and 22 km GTDR topography.
        Set to 351 for full SAR resolution (very large arrays).
        Set to 22471 for GTDR-native resolution (fast, low-memory).
    mcmc_draws:
        MCMC posterior samples (PyMC / NumPyro backends).
    mcmc_tune:
        MCMC tuning / warmup steps.
    mcmc_chains:
        Number of parallel chains.
    random_seed:
        Global random seed for reproducibility.
    datasets:
        Dataset catalogue. Populated from default_dataset_catalogue()
        if not supplied.
    priors:
        Bayesian prior configuration. Uses BayesianPriorConfig defaults
        if not supplied.
    habitability_window:
        Temporal habitability window configuration (D1: past epoch,
        D2: future window).  Uses HabitabilityWindowConfig defaults
        (3.5 Gya past epoch, 100-400 Myr future window) if not supplied.
    shapefile_dir:
        Path to the directory containing Lopes geomorphology shapefiles.
        Overrides the path derived from data_dir if set.
    vims_parquet_path:
        Direct path to the VIMS spatial footprint parquet file.
        Accepts both the full catalogue (~5.4M rows, 227 MB) and the
        1,000-row development sample (vims_sample_1000rows.parquet).
        If None, the pipeline auto-discovers the file in data_dir by
        trying vims_footprints.parquet, vims_sample_1000rows.parquet,
        and any vims_*.parquet match.  Also set via --vims-parquet.
    """

    data_dir:       Path = field(default_factory=lambda: Path("data/raw"))
    processed_dir:  Path = field(default_factory=lambda: Path("data/processed"))
    output_dir:     Path = field(default_factory=lambda: Path("outputs"))

    bayesian_backend: Literal["sklearn", "pymc", "numpyro"] = "sklearn"

    # Grid resolution
    canonical_res_m: float = 4490.0   # ~0.1 deg/px, 10 ppd

    # MCMC parameters (PyMC / NumPyro only)
    mcmc_draws:  int = 2000
    mcmc_tune:   int = 1000
    mcmc_chains: int = 4
    random_seed: int = 42

    # Overrideable catalogue and priors
    datasets:            Dict[str, DatasetSpec]    = field(default_factory=dict)
    priors:              BayesianPriorConfig        = field(default_factory=BayesianPriorConfig)
    habitability_window: HabitabilityWindowConfig   = field(
        default_factory=HabitabilityWindowConfig
    )

    shapefile_dir:      Optional[Path] = None
    birch_dir:          Optional[Path] = None   # Birch+2017 / Palermo+2022 polar lake data
    vims_parquet_path:  Optional[Path] = None

    def __post_init__(self) -> None:
        self.data_dir      = Path(self.data_dir)
        self.processed_dir = Path(self.processed_dir)
        self.output_dir    = Path(self.output_dir)
        if self.shapefile_dir is not None:
            self.shapefile_dir = Path(self.shapefile_dir)
        if self.birch_dir is not None:
            self.birch_dir = Path(self.birch_dir)
        if self.vims_parquet_path is not None:
            self.vims_parquet_path = Path(self.vims_parquet_path)
        if not self.datasets:
            self.datasets = default_dataset_catalogue()
        self.priors.validate()
        self.habitability_window.validate()

    def make_dirs(self) -> None:
        """Create all required directories if they do not already exist."""
        for d in (self.data_dir, self.processed_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def canonical_crs(self) -> str:
        """PROJ4 string for the canonical pipeline CRS."""
        return CANONICAL_CRS_PROJ4

    @property
    def canonical_grid_shape(self) -> Tuple[int, int]:
        """
        Return (nrows, ncols) of the canonical global raster.

        Computed from the Titan sphere circumference and pixel size.
        """
        import math
        circumference_m = 2 * math.pi * TITAN_RADIUS_M
        # deg per metre at equator
        m_per_deg = circumference_m / 360.0
        ncols = round(360.0 * m_per_deg / self.canonical_res_m)
        nrows = round(180.0 * m_per_deg / self.canonical_res_m)
        return nrows, ncols

    def get_vims_parquet(self) -> Optional[Path]:
        """
        Return the VIMS parquet path, searching data_dir if not set.

        Search order:
          1. vims_parquet_path if explicitly set (via --vims-parquet)
          2. data_dir / vims_footprints.parquet       (canonical name)
          3. data_dir / vims_sample_1000rows.parquet  (dev sample)
          4. First match of data_dir / vims_*.parquet

        Returns None if no file is found.
        """
        if self.vims_parquet_path is not None:
            p = Path(self.vims_parquet_path)
            return p if p.exists() else None
        candidates = [
            self.data_dir / "vims_footprints.parquet",
            self.data_dir / "vims_sample_1000rows.parquet",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Glob fallback
        matches = sorted(self.data_dir.glob("vims_*.parquet"))
        return matches[0] if matches else None

    def get_shapefile_dir(self) -> Path:
        """Return the Lopes geomorphology shapefile directory."""
        if self.shapefile_dir is not None:
            return self.shapefile_dir
        return self.data_dir / "geomorphology_shapefiles"

    def get_birch_dir(self) -> Path:
        """
        Return the root directory for the Birch+2017 / Palermo+2022 polar
        lake shapefile dataset.

        Default location: ``data/raw/birch_polar_mapping/``

        Override via ``PipelineConfig(birch_dir=...)`` or the
        ``--birch-dir`` CLI flag.

        Expected sub-directory layout::

            birch_polar_mapping/
              birch_filled/      <- Birch+2017 confirmed liquid surfaces
              birch_empty/       <- Birch+2017 empty basins (paleo-lakes)
              palermo/           <- Palermo+2022 alternative lake mapping

        The pipeline gracefully handles missing sub-directories; each one
        that is absent is silently skipped.

        Download URL:
            https://data.astro.cornell.edu/titan_polar_mapping_birch/
            titan_polar_mapping_birch.zip

        Returns
        -------
        Path
            Resolved path to the Birch dataset root.
        """
        if self.birch_dir is not None:
            return self.birch_dir
        return self.data_dir / "birch_polar_mapping"
