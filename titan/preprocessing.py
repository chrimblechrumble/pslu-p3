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
titan/preprocessing.py
========================
Stage 2 -- Preprocessing to Canonical Format.

Converts all raw datasets to a single shared canonical raster grid:
  Projection : SimpleCylindrical (equirectangular), Titan sphere R=2,575,000 m
  Lon        : West-positive, 0 deg -> 360 deg  (matches all USGS raster products)
  Lat        : North-up, +90 deg -> -90 deg
  Units      : Metres (projected CRS)
  Dtype      : float32
  Nodata     : NaN
  Format     : Cloud-Optimised GeoTIFF (.tif)

All inputs are handled via format-specific sub-routines that correctly
address each product's CRS, nodata convention, and tile structure:

  GeoTIFF (USGS mosaics) : SimpleCylindrical metres, nodata=0.0
  GTDR .IMG (PDS3)       : Equirectangular deg, nodata=0xFF7FFFFB=-3.4e38
  Shapefiles             : GCS_Titan_2000 east-positive deg -> must flip lon
  Parquet (VIMS index)   : Tabular, converted to 2-D raster via spatial binning

References
----------
Lorenz et al. (2013)    doi:10.1016/j.icarus.2013.04.002
Lopes et al. (2019)     doi:10.1038/s41550-019-0917-6
Le Mouelic et al. (2019) doi:10.1016/j.icarus.2018.09.017
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from configs.pipeline_config import (
    CANONICAL_CRS_PROJ4,
    GEOTIFF_NODATA,
    GTDR_MISSING_CONSTANT,
    TITAN_RADIUS_M,
    PipelineConfig,
    TerrainClass,
    TERRAIN_CLASSES,
)
from titan.io.gtdr_reader import (
    mosaic_gtdr_tiles,
    read_gtdr_img,
    gtdr_affine_transform,
)
from titan.io.shapefile_rasteriser import GeomorphologyRasteriser
from titan.io.vims_reader import VIMSFootprintIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical grid
# ---------------------------------------------------------------------------

class CanonicalGrid:
    """
    Defines the shared spatial grid onto which all datasets are resampled.

    The grid is an equirectangular (SimpleCylindrical) raster in projected
    metres with the Titan sphere (R=2,575,000 m) as the ellipsoid.
    Longitude increases westward from 0 deg to 360 deg.

    Parameters
    ----------
    pixel_size_m:
        Pixel size in metres.  Default 4,490 m =~ 0.1 deg/px at equator
        (a compromise between the 351 m SAR mosaic and 22 km GTDR).
    """

    def __init__(self, pixel_size_m: float = 4490.0) -> None:
        self.pixel_size_m = pixel_size_m

        # Circumference-based grid dimensions
        circumference_m  = 2.0 * math.pi * TITAN_RADIUS_M
        m_per_deg        = circumference_m / 360.0

        self.ncols: int = round(360.0 * m_per_deg / pixel_size_m)
        self.nrows: int = round(180.0 * m_per_deg / pixel_size_m)

        # Metres per pixel (re-derived after rounding for exactness)
        self.dx_m: float =  360.0 * m_per_deg / self.ncols
        self.dy_m: float = -180.0 * m_per_deg / self.nrows

        # Extent in projected metres
        # West edge = 0 degW = 0 m;  North edge = 90 deg x (Rxpi/180)
        self.west_m:  float = 0.0
        self.north_m: float = 90.0 * m_per_deg
        self.east_m:  float = self.west_m  + self.ncols * self.dx_m
        self.south_m: float = self.north_m + self.nrows * self.dy_m

    @property
    def transform(self) -> "rasterio.transform.Affine":
        """Rasterio Affine transform for the canonical grid."""
        from rasterio.transform import from_origin
        return from_origin(self.west_m, self.north_m, self.dx_m, -self.dy_m)

    @property
    def crs(self) -> "rasterio.crs.CRS":
        """Rasterio CRS object for the canonical grid."""
        from rasterio.crs import CRS
        return CRS.from_proj4(CANONICAL_CRS_PROJ4)

    def lon_centres_deg(self) -> np.ndarray:
        """Centre longitudes of each column in west-positive deg."""
        return np.linspace(
            0.0 + 180.0 / (math.pi * TITAN_RADIUS_M) * self.dx_m * 0.5,
            360.0 - 180.0 / (math.pi * TITAN_RADIUS_M) * self.dx_m * 0.5,
            self.ncols,
        )

    def lat_centres_deg(self) -> np.ndarray:
        """Centre latitudes of each row in deg (north-first)."""
        m_per_deg = 2.0 * math.pi * TITAN_RADIUS_M / 360.0
        half_dy   = abs(self.dy_m) * 0.5 / m_per_deg
        return np.linspace(90.0 - half_dy, -90.0 + half_dy, self.nrows)

    def empty(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """Return an (nrows, ncols) NaN-filled float32 array."""
        arr = np.full((self.nrows, self.ncols), np.nan, dtype=dtype)
        return arr

    def __repr__(self) -> str:
        return (
            f"CanonicalGrid(res={self.pixel_size_m:.0f}m, "
            f"shape=({self.nrows},{self.ncols}), "
            f"lon=[0->360 degW], lat=[+90 deg->-90 deg])"
        )


# ---------------------------------------------------------------------------
# Format-specific preprocessors
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Suppress spurious PROJ "eqc: Invalid latitude" messages that rasterio
# emits during reprojection of equirectangular Titan data at +/-90 deg latitude.
# These are not errors -- PROJ flags lat=+/-90 in eqc as "invalid" but handles
# them correctly.  Silencing at the rasterio._err logger level is cleaner
# than patching PROJ env vars.
# ---------------------------------------------------------------------------
class _ProjEqcFilter(logging.Filter):
    """Drop CPLE_AppDefined PROJ eqc Invalid latitude log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("eqc" in msg and "Invalid latitude" in msg)

logging.getLogger("rasterio._err").addFilter(_ProjEqcFilter())


def _reproject_geotiff(
    src_path: Path,
    dst_path: Path,
    grid: CanonicalGrid,
    nodata_in: float = GEOTIFF_NODATA,
    band: int = 1,
) -> None:
    """
    Reproject a USGS SimpleCylindrical GeoTIFF to the canonical grid.

    GeoTIFF products from USGS Astropedia use:
      - Coordinates in metres (projected CRS)
      - West-positive longitude, 0->360 deg
      - Nodata = 0.0 (float32)

    Parameters
    ----------
    src_path:
        Input GeoTIFF path.
    dst_path:
        Output canonical GeoTIFF path.
    grid:
        Canonical grid definition.
    nodata_in:
        Input nodata value (0.0 for USGS GeoTIFFs).
    band:
        Band index to extract (1-based).
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        src_data = src.read(band).astype(np.float32)
        # Mask nodata -> NaN
        src_data[src_data == nodata_in] = np.nan
        src_crs       = src.crs or grid.crs  # fallback if no CRS embedded
        src_transform = src.transform

    dst_data = grid.empty()
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    _write_canonical_tif(dst_data, dst_path, grid)


def _reproject_gtdr(
    east_img: Path,
    west_img: Path,
    dst_path: Path,
    grid: CanonicalGrid,
    east_lbl: Optional[Path] = None,
    west_lbl: Optional[Path] = None,
) -> None:
    """
    Mosaic and reproject the two GTDR PDS3 half-globe tiles.

    GTDR tiles use:
      - Equirectangular projection, coordinates in deg
      - West-positive longitude, 0->360 deg
      - Nodata = 0xFF7FFFFB = -3.4028x10^3^8 (NOT NaN -- must exact-compare)

    Parameters
    ----------
    east_img:
        Eastern hemisphere IMG tile.
    west_img:
        Western hemisphere IMG tile.
    dst_path:
        Output canonical GeoTIFF.
    grid:
        Canonical grid.
    east_lbl, west_lbl:
        Optional companion label files.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    import math

    # Read and mosaic the two tiles
    mosaic, meta = mosaic_gtdr_tiles(east_img, west_img, east_lbl, west_lbl)

    # Build a source rasterio-compatible transform for the mosaic
    # Mosaic: 360 rows x 720 cols, 0.5 deg/pixel, west-positive, metres
    m_per_deg = 2.0 * math.pi * TITAN_RADIUS_M / 360.0
    src_transform = from_origin(
        west=0.0,
        north=90.0 * m_per_deg,
        xsize=0.5 * m_per_deg,
        ysize=0.5 * m_per_deg,
    )

    dst_data = grid.empty()
    reproject(
        source=mosaic,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=grid.crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    _write_canonical_tif(dst_data, dst_path, grid)


def _rasterise_geomorphology(
    shapefile_dir: Path,
    dst_path: Path,
    grid: CanonicalGrid,
) -> None:
    """
    Rasterise the Lopes et al. geomorphology shapefiles.

    Handles the east-positive -> west-positive longitude flip automatically.
    Output is an int16 terrain-class raster.

    Parameters
    ----------
    shapefile_dir:
        Directory containing the shapefile sets.
    dst_path:
        Output canonical GeoTIFF (int16).
    grid:
        Canonical grid.
    """
    rasteriser = GeomorphologyRasteriser(
        shapefile_dir   = shapefile_dir,
        output_shape    = (grid.nrows, grid.ncols),
        output_transform= grid.transform,
        output_crs      = CANONICAL_CRS_PROJ4,
    )
    canvas = rasteriser.rasterise(out_path=dst_path)
    logger.info("Geomorphology raster: %s", dst_path)


def _rasterise_channels(
    shp_path: Path,
    dst_path: Path,
    grid: CanonicalGrid,
) -> None:
    """
    Rasterise the Miller et al. (2021) fluvial channel shapefile onto
    the canonical grid, producing a continuous channel-density proxy.

    Steps
    -----
    1. Load the shapefile (east-positive) and flip to west-positive.
    2. Burn each channel segment as a binary 1 into a float32 canvas.
    3. Apply a Gaussian blur (sigma =~ 3 pixels =~ one grid cell half-width)
       so that the channel influence spreads to adjacent pixels.
    4. Normalise to [0, 1] and write to GeoTIFF.

    Parameters
    ----------
    shp_path:
        Path to global_channels.shp.
    dst_path:
        Output canonical GeoTIFF (float32, 0-1).
    grid:
        Canonical grid.
    """
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.features import rasterize as rio_rasterize
        from scipy.ndimage import gaussian_filter
        from titan.io.shapefile_rasteriser import flip_geodataframe_longitude
    except ImportError as e:
        logger.warning("Cannot rasterise channels (%s). Skipping.", e)
        return

    gdf = gpd.read_file(shp_path)
    gdf = flip_geodataframe_longitude(gdf)

    # Rasterise all channel geometries as binary (1 = channel present)
    canvas = np.zeros((grid.nrows, grid.ncols), dtype=np.float32)
    shapes = [(geom, 1.0) for geom in gdf.geometry if geom is not None]
    if shapes:
        canvas = rio_rasterize(
            shapes,
            out_shape=(grid.nrows, grid.ncols),
            transform=grid.transform,
            fill=0.0,
            dtype=np.float32,
        )

    # Smooth with Gaussian kernel to create a density proxy
    density = gaussian_filter(canvas, sigma=3.0)

    # Normalise to [0, 1]
    dmax = density.max()
    if dmax > 0:
        density = density / dmax

    _write_canonical_tif(density, dst_path, grid)
    logger.info("Channel density raster: %s  (%.1f%% non-zero pixels)",
                dst_path,
                100.0 * float(np.sum(density > 0)) / density.size)


def _bin_vims_to_raster(
    parquet_path: Path,
    dst_path: Path,
    grid: CanonicalGrid,
) -> None:
    """
    Convert the VIMS spatial footprint index to a coverage-density raster.

    The output represents normalised VIMS observation coverage (0->1),
    used as a proxy for surface composition data quality in the
    habitability feature extraction.

    Parameters
    ----------
    parquet_path:
        Path to the VIMS footprint parquet file.
    dst_path:
        Output canonical GeoTIFF.
    grid:
        Canonical grid.
    """
    index = VIMSFootprintIndex(parquet_path)
    coverage = index.coverage_map(
        nrows=grid.nrows,
        ncols=grid.ncols,
        lon_range=(0.0, 360.0),
        lat_range=(-90.0, 90.0),
    )
    _write_canonical_tif(coverage, dst_path, grid)
    logger.info("VIMS coverage raster: %s", dst_path)


# ---------------------------------------------------------------------------
# Main preprocessor
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """
    Orchestrates preprocessing of all datasets to the canonical grid.

    Parameters
    ----------
    config:
        Pipeline configuration object.
    grid:
        Canonical grid.  Created from config if not supplied.
    """

    def __init__(
        self,
        config: PipelineConfig,
        grid: Optional[CanonicalGrid] = None,
    ) -> None:
        self.config = config
        self.grid   = grid or CanonicalGrid(config.canonical_res_m)
        config.make_dirs()

    def preprocess_all(self, overwrite: bool = False) -> Dict[str, Path]:
        """
        Preprocess every available raw dataset.

        Skips datasets whose raw files are missing with a warning.

        Parameters
        ----------
        overwrite:
            Re-process even if canonical output already exists.

        Returns
        -------
        Dict[str, Path]
            Mapping of dataset name -> canonical GeoTIFF path.
        """
        results: Dict[str, Path] = {}

        # Topography (requires two GTDR tiles)
        results.update(self._preprocess_topography(overwrite))

        # SAR mosaic
        results.update(
            self._preprocess_geotiff("sar_mosaic", overwrite)
        )

        # ISS mosaic
        results.update(
            self._preprocess_geotiff("iss_mosaic_450m", overwrite)
        )

        # VIMS mosaic (GeoTIFF if downloaded)
        results.update(
            self._preprocess_geotiff("vims_mosaic", overwrite)
        )

        # VIMS footprint parquet -> coverage raster
        results.update(self._preprocess_vims_parquet(overwrite))

        # Geomorphology shapefiles
        results.update(self._preprocess_geomorphology(overwrite))
        results.update(self._preprocess_channels(overwrite))

        # Birch+2017 / Palermo+2022 polar lake shapefiles.
        # Produces a dedicated polar-lake raster with filled/empty/Palermo
        # classes that improves Feature 1 (liquid_hydrocarbon) and Feature 5
        # (surface_atm_interaction) in the polar regions.
        # Silently produces all-zeros if the Birch dataset is not installed.
        results.update(self._preprocess_polar_lakes(overwrite))

        # CIRS temperature -- synthesised from Jennings et al. (2019) formula.
        # No external file required; the analytical model is embedded in
        # titan/atmospheric_profiles.py and valid for the full Cassini mission.
        results.update(self._synthesise_cirs_temperature(overwrite))

        return results

    # -- Format-specific dispatch ----------------------------------------------

    def _preprocess_topography(
        self, overwrite: bool
    ) -> Dict[str, Path]:
        """
        Mosaic and reproject GTDR/GTDE topography tiles.

        Source priority (all from Cornell eCommons / USGS gtdr-data.zip)
        ------------------------------------------------------------------
        1. GTDE T126 -- Dense spline-interpolated global DEM  **PREFERRED**
           Files: GTDED00N090_T126_V01  +  GTDED00N270_T126_V01
           ~90% global coverage (Corlies et al. 2017).

        2. GT0E T126 -- Standard GTDR final mission (sparse)
           Files: GT2ED00N090_T126_V01  +  GT2ED00N270_T126_V01
           ~25% coverage (altimetry + SARtopo nadir corridors only).

        3. GT0E T077 -- Standard GTDR legacy (sparse, partial mission)
           Files: GT0EB00N090_T077_V01  +  matching west tile
           ~15% coverage. Fallback for users who have only the older release.

        The reader accepts both .IMG and .IMG.gz (Cornell distributes .gz;
        the reader decompresses transparently).
        """
        out = self.config.processed_dir / "topography_canonical.tif"
        if out.exists() and not overwrite:
            return {"topography": out}

        data_dir = self.config.data_dir

        def _find_img(stem: str) -> Optional[Path]:
            """Return path to IMG or IMG.gz if it exists, else None."""
            for suffix in (".IMG", ".IMG.gz"):
                p = data_dir / (stem + suffix)
                if p.exists():
                    return p
            return None

        def _find_corlies_interp() -> Optional[Path]:
            """
            Find the Corlies 2017 globally interpolated topography file.

            This is topo_4PPD_interp.cub from the Hayes Research Group
            (hayesresearchgroup.com/data-products/, titan_topo_corlies.zip).
            It has 100% global coverage at 4ppd resolution -- useful as a
            gap-filler where GTDE has NaN pixels (~10% of globe).
            """
            candidates = [
                data_dir / "hayes_topo" / "topo_4PPD_interp.cub",
                data_dir / "topo_4PPD_interp.cub",
            ]
            for p in candidates:
                if p.exists():
                    return p
            return None

        def _gap_fill_topo(primary_path: Path, fill_path: Path) -> None:
            """
            Fill NaN pixels in primary GeoTIFF using a secondary source.

            Reprojects fill_path to match primary_path's CRS/resolution,
            then writes NaN pixels in primary with values from fill.
            Modifies primary_path in place.
            """
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.warp import calculate_default_transform, reproject

            with rasterio.open(primary_path) as src:
                profile = src.profile.copy()
                primary_data = src.read(1)
                primary_nodata = src.nodata
                transform = src.transform
                crs = src.crs
                width, height = src.width, src.height

            nodata_val = primary_nodata if primary_nodata is not None else np.nan
            nan_mask = (
                np.isnan(primary_data) if np.issubdtype(primary_data.dtype, np.floating)
                else primary_data == nodata_val
            )
            n_nan = int(nan_mask.sum())
            if n_nan == 0:
                logger.info("Gap-fill: primary has no NaN pixels, skipping.")
                return

            logger.info(
                "Gap-fill: reprojecting %s to fill %d NaN pixels in topography ...",
                fill_path.name, n_nan,
            )
            fill_reprojected = np.full((height, width), np.nan, dtype=np.float32)
            with rasterio.open(fill_path) as fill_src:
                reproject(
                    source=rasterio.band(fill_src, 1),
                    destination=fill_reprojected,
                    src_transform=fill_src.transform,
                    src_crs=fill_src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                )

            filled = primary_data.copy()
            filled = filled.astype(np.float32)
            nan_in_primary = np.isnan(filled) if np.issubdtype(filled.dtype, np.floating) else filled == nodata_val
            filled[nan_in_primary] = fill_reprojected[nan_in_primary]

            n_filled = int(np.sum(nan_in_primary & np.isfinite(fill_reprojected)))
            logger.info(
                "Gap-fill: filled %d of %d NaN pixels (%.1f%%) from Corlies 2017.",
                n_filled, n_nan, 100.0 * n_filled / max(n_nan, 1),
            )

            profile.update(dtype="float32", nodata=np.nan)
            with rasterio.open(primary_path, "w", **profile) as dst:
                dst.write(filled, 1)

        def _find_lbl(img_path: Path) -> Optional[Path]:
            """Find companion .LBL, stripping .gz suffix if needed."""
            from pathlib import Path as _P
            base: Path = _P(img_path)
            if base.suffix.lower() == ".gz":
                base = base.with_suffix("")
            for suffix in (".LBL", ".lbl"):
                lbl: Path = base.with_suffix(suffix)
                if lbl.exists():
                    return lbl
            return None

        # -- Priority 1: GTDE interpolated global DEM ----------------------
        gtde_e = _find_img("GTDED00N090_T126_V01")
        gtde_w = _find_img("GTDED00N270_T126_V01")
        if gtde_e and gtde_w:
            logger.info(
                "Preprocessing GTDE T126 interpolated DEM "
                "(~90%% global coverage -- preferred source) ..."
            )
            _reproject_gtdr(
                gtde_e, gtde_w, out, self.grid,
                _find_lbl(gtde_e), _find_lbl(gtde_w),
            )
            corlies = _find_corlies_interp()
            if corlies:
                _gap_fill_topo(out, corlies)
            else:
                logger.info(
                    "Corlies 2017 interpolated topo not found "
                    "(topo_4PPD_interp.cub from hayesresearchgroup.com/data-products/). "
                    "Topography will have ~10%% NaN gaps. "
                    "Download titan_topo_corlies.zip for 100%% coverage."
                )
            return {"topography": out}
        logger.info(
            "GTDE tiles not found (GTDED00N090/270_T126_V01.IMG[.gz]). "
            "For ~90%% global DEM coverage, download from Cornell eCommons: "
            "https://data.astro.cornell.edu/RADAR/DATA/GTDR/"
        )

        # -- Priority 2: GT0E T126 standard GTDR (final mission) -----------
        gt0e_e = _find_img("GT2ED00N090_T126_V01")
        gt0e_w = _find_img("GT2ED00N270_T126_V01")
        if gt0e_e and gt0e_w:
            logger.info(
                "Preprocessing GT0E T126 GTDR (~25%% coverage). "
                "Download GTDE tiles for ~90%% global coverage."
            )
            _reproject_gtdr(
                gt0e_e, gt0e_w, out, self.grid,
                _find_lbl(gt0e_e), _find_lbl(gt0e_w),
            )
            return {"topography": out}

        # -- Priority 3: GT0E T077 legacy (partial mission) ----------------
        legacy_e = _find_img("GT0EB00N090_T077_V01")
        legacy_w = (
            _find_img("GT0WB00N270_T077_V01") or
            _find_img("GT0EB00N270_T077_V01") or
            _find_img("GT2ED00N270_T077_V01")
        )
        if legacy_e and legacy_w:
            logger.info(
                "Preprocessing GT0E T077 legacy GTDR tiles (~15%% coverage). "
                "For better coverage, use T126 GT0E or GTDE tiles."
            )
            _reproject_gtdr(
                legacy_e, legacy_w, out, self.grid,
                _find_lbl(legacy_e), _find_lbl(legacy_w),
            )
            return {"topography": out}

        logger.warning(
            "No GTDR/GTDE tile pairs found in %s. "
            "Topography features will be NaN. "
            "Download GTDE tiles (interpolated, ~90%% coverage) from: "
            "https://data.astro.cornell.edu/RADAR/DATA/GTDR/",
            data_dir,
        )
        return {}


    def _preprocess_geotiff(
        self, name: str, overwrite: bool
    ) -> Dict[str, Path]:
        """Reproject a single GeoTIFF dataset."""
        spec = self.config.datasets.get(name)
        if spec is None:
            return {}
        raw = self.config.data_dir / spec.local_filename
        if not raw.exists():
            logger.warning("Raw file missing for '%s': %s", name, raw)
            return {}
        out = self.config.processed_dir / f"{name}_canonical.tif"
        if out.exists() and not overwrite:
            return {name: out}
        logger.info("Preprocessing GeoTIFF: %s", name)
        nodata = spec.nodata_value if spec.nodata_value is not None \
            else GEOTIFF_NODATA
        _reproject_geotiff(raw, out, self.grid, nodata_in=nodata)
        return {name: out}

    def _preprocess_vims_parquet(
        self, overwrite: bool
    ) -> Dict[str, Path]:
        """
        Convert VIMS parquet footprint index to a coverage raster.

        Uses get_vims_parquet() to resolve the file, which searches for
        multiple alternative names (vims_footprints.parquet,
        vims_sample_1000rows.parquet, any vims_*.parquet) before giving up.
        The 1,000-row sample is sufficient for development-mode coverage maps.
        """
        raw = self.config.get_vims_parquet()
        if raw is None:
            logger.warning(
                "VIMS parquet not found in %s. "
                "Use --vims-parquet PATH or place vims_footprints.parquet / "
                "vims_sample_1000rows.parquet in the data directory.",
                self.config.data_dir,
            )
            return {}
        is_sample = raw.stat().st_size < 10_000_000  # <10 MB -> sample
        if is_sample:
            logger.info(
                "Using VIMS parquet SAMPLE (%s, %d KB). "
                "Coverage maps will be sparse. "
                "Full 227 MB catalogue gives reliable global coverage.",
                raw.name, raw.stat().st_size // 1024,
            )
        else:
            logger.info("Using VIMS parquet: %s (%.1f MB)", raw.name,
                        raw.stat().st_size / 1e6)
        out = self.config.processed_dir / "vims_coverage_canonical.tif"
        if out.exists() and not overwrite:
            return {"vims_coverage": out}
        logger.info("Binning VIMS footprint parquet -> coverage raster ...")
        _bin_vims_to_raster(raw, out, self.grid)
        return {"vims_coverage": out}

    def _preprocess_geomorphology(
        self, overwrite: bool
    ) -> Dict[str, Path]:
        """Rasterise the Lopes geomorphology shapefiles."""
        shp_dir = self.config.get_shapefile_dir()
        if not shp_dir.exists():
            logger.warning("Shapefile directory not found: %s", shp_dir)
            return {}
        out = self.config.processed_dir / "geomorphology_canonical.tif"
        if out.exists() and not overwrite:
            return {"geomorphology": out}
        logger.info("Rasterising geomorphology shapefiles ...")
        _rasterise_geomorphology(shp_dir, out, self.grid)
        return {"geomorphology": out}

    def _preprocess_polar_lakes(
        self, overwrite: bool
    ) -> Dict[str, Path]:
        """
        Rasterise the Birch+2017 / Palermo+2022 polar lake shapefiles.

        Produces ``data/processed/polar_lakes_canonical.tif`` -- a separate
        int16 raster with four classes:

        =====================  =====  ========================================
        Class                  Value  Meaning
        =====================  =====  ========================================
        NoData                 0      Outside polar-mapping coverage
        FilledLake_Birch       1      Confirmed liquid (Birch+2017 filled)
        EmptyBasin_Birch       2      Paleo-lake / empty basin (Birch+2017)
        FilledLake_Palermo     3      Confirmed liquid (Palermo+2022)
        =====================  =====  ========================================

        This raster is consumed by:
          - ``Feature 1`` (liquid_hydrocarbon): Birch/Palermo filled pixels
            replace the SAR proxy in the polar region, giving expert-mapped
            lake boundaries instead of a backscatter threshold.
          - ``Feature 5`` (surface_atm_interaction): Birch shorelines give
            the exact lake-margin zone; empty-basin pixels contribute a
            ``paleo_lake_indicator`` sub-component capturing wetting/drying
            cycles relevant to Mayer & Nixon (2025) vesicle formation.

        If the Birch directory does not exist, logs a download notice and
        returns an empty dict (feature extraction falls back gracefully to
        the existing SAR-proxy behaviour).

        Expected data layout::

            data/raw/birch_polar_mapping/
              birch_filled/      <- Birch+2017 filled lake/sea .shp files
              birch_empty/       <- Birch+2017 empty basin .shp files
              palermo/           <- Palermo+2022 .shp files

        Download:
            https://data.astro.cornell.edu/titan_polar_mapping_birch/
            titan_polar_mapping_birch.zip

        Parameters
        ----------
        overwrite:
            Re-process even if the canonical output already exists.

        Returns
        -------
        Dict[str, Path]
            ``{"polar_lakes": <path>}`` if rasterisation succeeded,
            else empty dict.

        References
        ----------
        Birch et al. (2017) Icarus doi:10.1016/j.icarus.2017.01.032
        Palermo et al. (2022) PSJ doi:10.3847/PSJ/ac4d9c
        Mayer & Nixon (2025) Int. J. Astrobiology
            doi:10.1017/S1473550425100037
        """
        from titan.io.shapefile_rasteriser import PolarLakeRasteriser

        birch_dir = self.config.get_birch_dir()
        out = self.config.processed_dir / "polar_lakes_canonical.tif"

        if out.exists() and not overwrite:
            return {"polar_lakes": out}

        rasteriser = PolarLakeRasteriser(
            birch_dir        = birch_dir if birch_dir.exists() else None,
            output_shape     = (self.grid.nrows, self.grid.ncols),
            output_transform = self.grid.transform,
            output_crs       = self.grid.crs,
        )

        if not rasteriser.is_available():
            logger.info(
                "Birch polar lake data not found at %s. "
                "Feature 1 will use SAR-proxy fallback; Feature 5 lake "
                "margins will use Lopes geomorphology dilation.\n"
                "  To install: download titan_polar_mapping_birch.zip from\n"
                "  https://data.astro.cornell.edu/titan_polar_mapping_birch/\n"
                "  Extract and place sub-folders at:\n"
                "  data/raw/birch_polar_mapping/birch_filled/\n"
                "  data/raw/birch_polar_mapping/birch_empty/\n"
                "  data/raw/birch_polar_mapping/palermo/",
                birch_dir,
            )
            return {}

        logger.info(
            "Rasterising Birch+2017 / Palermo+2022 polar lake shapefiles "
            "from %s ...", birch_dir,
        )
        rasteriser.rasterise(
            include_filled  = True,
            include_empty   = True,
            include_palermo = True,
            out_path        = out,
        )
        logger.info("Polar-lake raster: %s", out)
        return {"polar_lakes": out}

    def _preprocess_channels(
        self, overwrite: bool
    ) -> Dict[str, Path]:
        """
        Rasterise the Miller et al. (2021) fluvial channel shapefile into
        a binary channel-density raster on the canonical grid.

        The channel map (`global_channels.shp`) is looked for in the same
        directory as the Lopes geomorphology shapefiles.  Each pixel that
        intersects at least one channel segment is set to 1; others to 0.
        The result is then convolved with a Gaussian kernel (sigma=3 px)
        to produce a continuous channel-density proxy.
        """
        shp_dir = self.config.get_shapefile_dir()
        shp = shp_dir / "global_channels.shp"
        if not shp.exists():
            logger.info(
                "global_channels.shp not found in %s -- channel_density "
                "layer will be absent. Download titan_channels_miller.zip "
                "from https://hayesresearchgroup.com/data-products/",
                shp_dir,
            )
            return {}
        out = self.config.processed_dir / "channel_density_canonical.tif"
        if out.exists() and not overwrite:
            return {"channel_density": out}
        logger.info("Rasterising channel map -> channel_density ...")
        _rasterise_channels(shp, out, self.grid)
        return {"channel_density": out}

    def _synthesise_cirs_temperature(
        self, overwrite: bool
    ) -> Dict[str, Path]:
        """
        Generate a canonical surface-temperature raster from the Jennings et al.
        (2019) analytical model -- no external CIRS file required.

        Why the same epoch is used for all temporal modes
        -------------------------------------------------
        The Jennings formula is a fit to Cassini CIRS observations from 2004
        to 2017.  It captures the *spatial pattern* of surface temperature
        (the latitudinal gradient that drives differential evaporation and
        therefore methane cycling), not an absolute temperature value.

        For the pipeline's purposes the CIRS raster feeds only one feature:
        the meridional temperature gradient that contributes 25 % of the
        methane_cycle score (normalised to [0, 1]).  What matters is whether
        the gradient structure is physically plausible -- and the Cassini-era
        pattern (cooler poles, warmer equatorial band, slight hemispheric
        asymmetry driven by Titan's season) is the best available proxy for
        all three modes:

          PRESENT  -- exact fit: mid-mission near northern spring equinox.
          PAST     -- Titan's surface temperature distribution is driven by
                      its obliquity and distance from the Sun, neither of
                      which changes significantly on the million-year scales
                      of the pipeline's past mode.  The Cassini pattern is
                      the best available proxy.
          FUTURE   -- For the red-giant phase the absolute temperature rises
                      dramatically (handled by titan_temp_K() in
                      generate_temporal_maps.py), but by that point the
                      methane cycle feature is zeroed out by the temporal
                      scaling functions regardless of the gradient shape.
                      Using the Cassini pattern is harmless.

        Extrapolating the Jennings formula beyond its calibration window
        (approximately Y = +/-8 yr from the 2009.61 equinox) produces
        physically absurd results (temperatures of millions of K at 1 Gya),
        so the nominal mid-mission year is always used.

        Output
        ------
        2-D float32 raster (K) on the canonical grid; each pixel receives
        the zonal-mean surface brightness temperature for its latitude.

        Reference
        ---------
        Jennings, D. E. et al. (2019) ApJL 877, L8.
        DOI: 10.3847/2041-8213/ab1f91
        """
        from titan.atmospheric_profiles import jennings_temperature_grid

        out: Path = self.config.processed_dir / "cirs_temperature_canonical.tif"
        if out.exists() and not overwrite:
            return {"cirs_temperature": out}

        # Mid-Cassini mission, near Titan's northern spring equinox.
        # Used for all temporal modes -- see docstring for rationale.
        year_ce: float = 2011.0
        mode: str = getattr(self.config, "temporal_mode", "present")

        logger.info(
            "Synthesising CIRS surface temperature from Jennings 2019 "
            "formula (year_ce=%.1f, mode=%s) ...", year_ce, mode,
        )

        # Build a 2-D latitude grid on the canonical grid
        lats_1d = self.grid.lat_centres_deg()   # shape (nrows,)
        lat_grid = np.tile(lats_1d[:, np.newaxis], (1, self.grid.ncols))

        T_grid = jennings_temperature_grid(lat_grid.astype(np.float32), year_ce)
        _write_canonical_tif(T_grid, out, self.grid)
        logger.info("CIRS temperature raster written: %s", out)
        return {"cirs_temperature": out}

    def _preprocess_netcdf(
        self, name: str, overwrite: bool
    ) -> Dict[str, Path]:
        """Regrid a NetCDF dataset onto the canonical grid."""
        spec = self.config.datasets.get(name)
        if spec is None:
            return {}
        raw = self.config.data_dir / spec.local_filename
        if not raw.exists():
            logger.warning("NetCDF not found for '%s': %s", name, raw)
            return {}
        out = self.config.processed_dir / f"{name}_canonical.tif"
        if out.exists() and not overwrite:
            return {name: out}
        logger.info("Regridding NetCDF: %s", name)
        _regrid_netcdf(raw, out, self.grid)
        return {name: out}


# ---------------------------------------------------------------------------
# Canonical data stack loader
# ---------------------------------------------------------------------------

class CanonicalDataStack:
    """
    Loads all preprocessed canonical GeoTIFFs into a single xarray Dataset.

    Parameters
    ----------
    config:
        Pipeline configuration.
    grid:
        Canonical grid.
    """

    def __init__(
        self,
        config: PipelineConfig,
        grid: Optional[CanonicalGrid] = None,
    ) -> None:
        self.config = config
        self.grid   = grid or CanonicalGrid(config.canonical_res_m)

    def load(self, names: Optional[List[str]] = None) -> xr.Dataset:
        """
        Load processed GeoTIFFs into an xarray Dataset.

        Parameters
        ----------
        names:
            Layer names to load.  None = load all available.

        Returns
        -------
        xr.Dataset
            Dataset with dimensions (lat, lon) and one variable per layer.
            Coordinates are in west-positive deg (geographic,
            recomputed from the projected metre grid).
            Missing layers are absent without raising errors.
        """
        import rasterio

        lats = self.grid.lat_centres_deg()
        lons = self.grid.lon_centres_deg()

        all_names = names or [
            "topography", "sar_mosaic", "iss_mosaic_450m",
            "vims_mosaic", "vims_coverage", "geomorphology",
            "channel_density", "cirs_temperature",
        ]

        data_vars: Dict[str, xr.DataArray] = {}
        for name in all_names:
            tif = self.config.processed_dir / f"{name}_canonical.tif"
            if not tif.exists():
                logger.debug("Canonical file not found for '%s'", name)
                continue
            spec = self.config.datasets.get(name)
            units = spec.units if spec else ""
            with rasterio.open(tif) as src:
                arr = src.read(1).astype(np.float32)
                nd = src.nodata
                if nd is not None:
                    # Handle both NaN-as-nodata (legacy) and sentinel (-9999)
                    if np.isnan(float(nd)):
                        arr[~np.isfinite(arr)] = np.nan
                    else:
                        arr[arr == float(nd)] = np.nan
            data_vars[name] = xr.DataArray(
                arr,
                dims=["lat", "lon"],
                coords={"lat": lats, "lon": lons},
                attrs={"units": units, "source": str(tif)},
            )

        return xr.Dataset(
            data_vars,
            attrs={
                "title": "Titan Habitability Pipeline - Canonical Stack",
                "titan_radius_km": TITAN_RADIUS_M / 1000.0,
                "crs": CANONICAL_CRS_PROJ4,
                "pixel_size_m": self.grid.pixel_size_m,
                "conventions": "CF-1.8",
            },
        )

    def save_netcdf(self, ds: xr.Dataset, path: Optional[Path] = None) -> Path:
        """
        Save the canonical stack to NetCDF4.

        Parameters
        ----------
        ds:
            Dataset from ``load()``.
        path:
            Output path.  Defaults to
            ``processed_dir / "titan_canonical_stack.nc"``.

        Returns
        -------
        Path
            Path of the written file.
        """
        if path is None:
            path = self.config.processed_dir / "titan_canonical_stack.nc"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path, format="NETCDF4")
        logger.info("Canonical stack saved -> %s", path)
        return path


# ---------------------------------------------------------------------------
# Shared GeoTIFF writer
# ---------------------------------------------------------------------------

def _write_canonical_tif(
    data: np.ndarray,
    path: Path,
    grid: CanonicalGrid,
) -> None:
    """
    Write a 2-D array to a Cloud-Optimised GeoTIFF on the canonical grid.

    Parameters
    ----------
    data:
        2-D float32 array of shape (nrows, ncols).
    path:
        Output file path.
    grid:
        Canonical grid (provides transform, CRS, dimensions).
    """
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)

    dtype = data.dtype if data.dtype in (np.float32, np.int16) else np.float32

    # Use a concrete nodata sentinel rather than NaN.
    # NaN-as-nodata is valid per the TIFF/GDAL spec for float32, but many GIS
    # tools (older QGIS, ArcGIS, Python PIL/Pillow, macOS Preview) report the
    # file as "invalid" or silently ignore the nodata metadata.
    # A concrete fill value (-9999.0 for float, 0 for int) is universally
    # understood by all tools.
    if np.issubdtype(dtype, np.floating):
        nodata_val: float = -9999.0
        # Replace NaN in data with the sentinel before writing
        data_out = np.where(np.isfinite(data), data, nodata_val).astype(dtype)
    else:
        nodata_val = 0
        data_out = data.astype(dtype)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        dtype=dtype,
        count=1,
        width=grid.ncols,
        height=grid.nrows,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata_val,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data_out, 1)


def _regrid_netcdf(
    src_path: Path,
    dst_path: Path,
    grid: CanonicalGrid,
) -> None:
    """
    Regrid a NetCDF file to the canonical grid via xarray interp().

    Assumes the NetCDF has 'lat' and 'lon' dimensions (in deg).

    Parameters
    ----------
    src_path:
        Input NetCDF file.
    dst_path:
        Output canonical GeoTIFF.
    grid:
        Canonical grid.
    """
    ds = xr.open_dataset(src_path)
    var_name = [v for v in ds.data_vars][0]
    da = ds[var_name].astype(np.float32)

    # Normalise dimension names
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})

    target_lats = xr.DataArray(grid.lat_centres_deg(), dims=["lat"])
    target_lons = xr.DataArray(grid.lon_centres_deg(), dims=["lon"])
    regridded = da.interp(lat=target_lats, lon=target_lons, method="linear")
    _write_canonical_tif(regridded.values.astype(np.float32), dst_path, grid)
    ds.close()


# ---------------------------------------------------------------------------
# Normalisation utilities
# ---------------------------------------------------------------------------

def normalise_to_0_1(
    arr: np.ndarray,
    clip: bool = True,
    percentile_lo: float = 2.0,
    percentile_hi: float = 98.0,
) -> np.ndarray:
    """
    Linearly normalise an array to [0, 1] using robust percentile clipping.

    Parameters
    ----------
    arr:
        Input array (may contain NaN).
    clip:
        Clip output to [0, 1] after scaling.
    percentile_lo, percentile_hi:
        Percentiles used as min/max for stretch.  Robust to outliers.

    Returns
    -------
    np.ndarray
        Normalised float32 array.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, np.nan, dtype=np.float32)
    vmin = float(np.percentile(finite, percentile_lo))
    vmax = float(np.percentile(finite, percentile_hi))
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.float32)
    # Compute in float64 to avoid overflow when arr contains large values
    # (e.g. GTDE missing constant ~-3.4e38 or elevation extremes)
    out = (arr.astype(np.float64) - vmin) / (vmax - vmin)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def compute_topographic_roughness(
    dem: np.ndarray,
    window_radius: int = 5,
) -> np.ndarray:
    """
    Compute terrain roughness as local standard deviation of elevation.

    Parameters
    ----------
    dem:
        2-D elevation array (metres).  NaN for missing.
    window_radius:
        Half-width of sliding window in pixels.

    Returns
    -------
    np.ndarray
        Roughness map normalised to [0, 1].
    """
    from scipy.ndimage import generic_filter

    def _nanstd(values: np.ndarray) -> float:
        v = values[np.isfinite(values)]
        return float(np.std(v)) if len(v) > 1 else 0.0

    size = 2 * window_radius + 1
    roughness = generic_filter(dem, _nanstd, size=size)
    return normalise_to_0_1(roughness)


def compute_terrain_diversity(
    class_map: np.ndarray,
    n_classes: int = 7,
    window_radius: int = 7,
) -> np.ndarray:
    """
    Compute Shannon diversity of terrain classes in a local window.

    Parameters
    ----------
    class_map:
        2-D integer array of terrain class labels (0=nodata, 1-7=classes).
    n_classes:
        Total number of terrain classes (default 7 per Lopes 2019 + lakes).
    window_radius:
        Half-width of the sliding window.

    Returns
    -------
    np.ndarray
        Per-pixel Shannon diversity index, normalised to [0, 1].
    """
    from scipy.ndimage import generic_filter

    def _shannon(values: np.ndarray) -> float:
        valid = values[values > 0].astype(int)
        if len(valid) == 0:
            return 0.0
        counts = np.bincount(valid, minlength=n_classes + 1)[1:]
        total  = counts.sum()
        if total == 0:
            return 0.0
        probs = counts[counts > 0] / total
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    size    = 2 * window_radius + 1
    raw     = generic_filter(class_map.astype(float), _shannon, size=size)
    return normalise_to_0_1(raw)
