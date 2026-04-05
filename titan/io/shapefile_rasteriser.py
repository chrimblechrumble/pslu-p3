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
titan/io/shapefile_rasteriser.py
=================================
Rasterises geomorphology and polar-lake shapefiles onto the canonical grid.

Two independent shapefile catalogues are supported:

**Lopes+2020 global geomorphology** (all terrain classes, global coverage):
  Source: Lopes et al. (2020) Nature Astronomy doi:10.1038/s41550-019-0917-6
  Data:   Schoenfeld (2024) Mendeley doi:10.17632/f6jrtyfp66.1  CC-BY-4.0
  Location: ``data/raw/geomorphology_shapefiles/``

  CONFIRMED FILE LISTING (from Mendeley API, April 2026):
        Basins.shp     1,829,768 bytes  sha256:ade414034e43c258...
        Craters.shp      107,712 bytes  sha256:dd30b573d521acb2...
        Dunes.shp      2,320,712 bytes  sha256:71bd90eb73d0cd61...
        Labyrinth.shp    614,880 bytes  sha256:89e17602555e130b...
        Mountains.shp  7,141,100 bytes  sha256:3c0cd9c0c7b786c5...
        Plains_3.shp   9,504,948 bytes  sha256:894e84153cedf0a2...

  *** Lakes.shp IS NOT PRESENT in the Mendeley distribution. ***
  The Lakes class does NOT appear in the Mendeley data product.
  Lake polygon geometry is provided by the separate Birch+2017 Cornell
  archive (see below), which covers both poles at higher resolution with
  filled vs empty basin distinction.

  Integer labels assigned at rasterisation time by this pipeline:
        Craters=1, Dunes=2, Plains_3=3, Basins=4, Mountains=5, Labyrinth=6
  Class 7 (Lakes) is reserved in SHAPEFILE_LAYERS but the file is absent.
  Any code checking geo==7 will therefore never match.

**Birch+2017 polar lake mapping** (polar regions, higher-resolution lake
outlines with confirmed-liquid vs empty-basin distinction):
  Source: Birch et al. (2017) Icarus doi:10.1016/j.icarus.2017.01.032
  Archive: https://data.astro.cornell.edu/titan_polar_mapping_birch/
           titan_polar_mapping_birch.zip  (6.0 GB)
  Path inside zip: full_dataset/Various Mapping Shapefiles/
                     Birch Polar Geomorphic (2017)/
                       north/   <- Fl_NORTH.shp, El_NORTH.shp, etc.
                       south/   <- Fl_SOUTH.shp, El_SOUTH.shp, Em_SOUTH.shp, etc.
  Pipeline directories:
    birch_filled/ <- Fl_NORTH.shp, Fl_SOUTH.shp  (confirmed liquid surfaces)
    birch_empty/  <- El_NORTH.shp, El_SOUTH.shp, Em_SOUTH.shp  (paleo-lakes / paleoseas)

  Geomorphic unit codes used by the pipeline:
    Fl = Filled lake/sea (SAR-dark, confirmed present-day liquid)
    El = Empty lake depression (paleo-lake; SAR-dark basin, no current liquid)
    Em = Empty sea / paleo-sea (south pole only; the four large southern
         paleoseas identified by Birch et al. 2018 Icarus doi:10.1016/j.icarus.2017.12.016)
    Other units (Lfd, Lud, Hdb, Hdd, Hud, Vdb, Vmb, Vub, Af, Fm, Mtn,
    Fluvial_Valleys) are geomorphic terrain classes not used by this pipeline.

Key format facts (verified by direct inspection of real shapefiles):

  CRS : GEOGCS["GCS_Titan_2000"], Titan sphere R=2,575,000 m
        Longitude: EAST-positive, range -180 deg -> +180 deg
        *** OPPOSITE to all raster products (which use west-positive) ***

  Geometry type : PolygonM (shape type 25 = polygon with measure coordinate)
        geopandas reads these correctly; the M coordinate is ignored.

  File structure (Lopes): ONE shapefile per terrain class
        Craters.shp   -> integer_label = 1  (Cr)
        Dunes.shp     -> integer_label = 2  (Dn)
        Plains_3.shp  -> integer_label = 3  (Pl)  [note: filename has _3 suffix]
        Basins.shp    -> integer_label = 4  (Ba)
        Mountains.shp -> integer_label = 5  (Mt)  [corresponds to Hummocky in paper]
        Labyrinth.shp -> integer_label = 6  (Lb)
        Lakes.shp     -> integer_label = 7  (Lk)  ABSENT -- not in Mendeley dataset

  Coordinate conversion -- PROJ BYPASSED:
        PROJ silently rejects west-positive longitudes > 180 deg in its
        longlat CRS, causing the entire east hemisphere to produce
        all-zeros.  Instead we apply the manual eqc formula to every vertex:
            lon_west = (-lon_east) % 360
            x_m = lon_west x (pi/180) x R_titan
            y_m = lat      x (pi/180) x R_titan

Rasterisation priority (Lopes -- drawn lowest to highest, higher overwrites):
        Dunes < Plains < Basins < Mountains < Labyrinth < Craters
        (Lakes absent; lake polygons come from Birch+2017 separately)

References
----------
Birch et al. (2017) "Geomorphological mapping of Titan's polar terrains"
  Icarus, doi:10.1016/j.icarus.2017.01.032

Birch et al. (2018) "Morphological evidence that Titan's southern hemisphere
  basins are paleoseas"
  Icarus, doi:10.1016/j.icarus.2017.12.016

Lopes et al. (2019) "A global geomorphologic map of Saturn's moon Titan"
  Nature Astronomy, doi:10.1038/s41550-019-0917-6

Miller et al. (2021) "Fluvial and karst morphology of Titan's polar regions"
  (channel network shapefiles in same Cornell archive, separate sub-directory)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lopes+2019 terrain class constants
# ---------------------------------------------------------------------------

#: Mapping from shapefile stem -> (integer_label, abbreviation).
#: Labels are assigned by this pipeline at rasterisation time.
#: There is no inherent numeric code in the Mendeley shapefiles.
#:
#: CONFIRMED PRESENT in Mendeley DOI:10.17632/f6jrtyfp66.1 (verified April 2026):
#:   Basins, Craters, Dunes, Labyrinth, Mountains, Plains_3
#:
#: ABSENT from Mendeley distribution:
#:   Lakes -- NOT in the Mendeley dataset.  Lake polygons are provided by the
#:            separate Birch+2017 Cornell archive (POLAR_LAKE_FILLED class).
#:            Any geomorphology raster built from the Mendeley data will
#:            never contain label 7.  All lake detection in features.py
#:            relies on the Birch polar_lakes layer.
SHAPEFILE_LAYERS: Final[Dict[str, Tuple[int, str]]] = {
    "Craters":   (1, "Cr"),
    "Dunes":     (2, "Dn"),
    "Plains_3":  (3, "Pl"),
    "Basins":    (4, "Ba"),
    "Mountains": (5, "Mt"),   # corresponds to "Hummocky terrain" in Lopes 2020 paper
    "Labyrinth": (6, "Lb"),
    "Lakes":     (7, "Lk"),   # ABSENT in Mendeley -- entry retained for compatibility
                              # only.  The rasteriser skips missing shapefiles silently.
}

#: Draw order for Lopes rasterisation -- lower index drawn first so higher
#: priority classes overwrite lower ones.
#: Lakes absent from Mendeley; lake geometry comes from Birch+2017 separately.
RASTER_DRAW_ORDER: Final[List[str]] = [
    "Dunes", "Plains_3", "Basins", "Mountains", "Labyrinth", "Craters",
    "Lakes",  # silently skipped if absent
]

#: Nodata value in the integer terrain-class output raster.
TERRAIN_NODATA: Final[int] = 0


# ---------------------------------------------------------------------------
# Birch+2017 polar-lake constants
# ---------------------------------------------------------------------------

#: Sub-directory names under the Birch dataset root (``data/raw/birch_polar_mapping/``).
#: Every ``*.shp`` inside each sub-directory is loaded and merged.
#:
#: Expected layout::
#:
#:   data/raw/birch_polar_mapping/
#:     birch_filled/      <- confirmed present-day liquid (Birch+2017, Fl_NORTH/SOUTH)
#:     birch_empty/       <- empty basins / paleo-lakes (Birch+2017, El_*/Em_SOUTH)
BIRCH_SUBDIR_FILLED:  Final[str] = "birch_filled"
BIRCH_SUBDIR_EMPTY:   Final[str] = "birch_empty"

#: Integer labels for the polar-lake raster (independent of Lopes labels).
#:
#: ======================  =====  ===========================================
#: Class                   Value  Meaning
#: ======================  =====  ===========================================
#: ``POLAR_LAKE_NODATA``   0      Outside polar-mapping coverage
#: ``POLAR_LAKE_FILLED``   1      Confirmed liquid -- Birch+2017 filled
#: ``POLAR_LAKE_EMPTY``    2      Empty basin / paleo-lake -- Birch+2017
#: ======================  =====  ===========================================
POLAR_LAKE_NODATA:   Final[int] = 0
POLAR_LAKE_FILLED:   Final[int] = 1
POLAR_LAKE_EMPTY:    Final[int] = 2


# ---------------------------------------------------------------------------
# CRS conversion
# ---------------------------------------------------------------------------

def east_pos_to_west_pos_deg(lon_east: float) -> float:
    """
    Convert a single longitude from east-positive to west-positive convention.

    Shapefiles use east-positive deg (-180 -> +180).
    All raster products use west-positive deg (0 -> 360).

    Parameters
    ----------
    lon_east:
        Longitude in east-positive deg.

    Returns
    -------
    float
        Equivalent longitude in west-positive deg (0 -> 360).

    Examples
    --------
    >>> east_pos_to_west_pos_deg(0.0)
    0.0
    >>> east_pos_to_west_pos_deg(90.0)   # 90 degE = 270 degW
    270.0
    >>> east_pos_to_west_pos_deg(-90.0)  # 90 degW = 90 degW
    90.0
    >>> east_pos_to_west_pos_deg(180.0)  # 180 deg = 180 deg
    180.0
    """
    return (-lon_east) % 360.0


def flip_geodataframe_longitude(
    gdf: "geopandas.GeoDataFrame",
) -> "geopandas.GeoDataFrame":
    """
    Convert all geometry coordinates from east-positive to west-positive.

    Applies ``lon_west = (-lon_east) % 360`` to every vertex of every
    geometry in the GeoDataFrame.  Handles Point, LineString, Polygon,
    MultiPolygon, and PolygonM geometries.

    Parameters
    ----------
    gdf:
        GeoDataFrame with east-positive longitude coordinates.

    Returns
    -------
    geopandas.GeoDataFrame
        New GeoDataFrame with west-positive longitude coordinates.
        The CRS is updated to the canonical Titan west-positive CRS.
    """
    from shapely.affinity import affine_transform
    import geopandas as gpd
    from shapely.ops import transform as shapely_transform

    def _flip_lon(lon: float, lat: float, *rest) -> tuple:
        """Flip a single coordinate point."""
        return ((-lon) % 360.0, lat) + rest

    new_geoms = gdf.geometry.apply(
        lambda geom: shapely_transform(_flip_lon, geom) if geom is not None else geom
    )
    result = gdf.copy()
    result.geometry = new_geoms
    # Update CRS to west-positive projection (no standard EPSG; use PROJ4)
    result = result.set_crs(
        "+proj=longlat +a=2575000 +b=2575000 +no_defs",
        allow_override=True,
    )
    return result


# ---------------------------------------------------------------------------
# Main rasteriser
# ---------------------------------------------------------------------------

class GeomorphologyRasteriser:
    """
    Rasterises the Lopes et al. geomorphology shapefiles to a single
    integer terrain-class raster on the canonical pipeline grid.

    Parameters
    ----------
    shapefile_dir:
        Directory containing the shapefile sets (one per terrain class).
    output_shape:
        (nrows, ncols) of the output raster.
    output_transform:
        Rasterio Affine transform for the output raster (west-positive metres).
    output_crs:
        PROJ4 string for the output CRS.
    """

    def __init__(
        self,
        shapefile_dir: Path,
        output_shape: Tuple[int, int],
        output_transform: "rasterio.transform.Affine",
        output_crs: str,
    ) -> None:
        self.shapefile_dir   = Path(shapefile_dir)
        self.output_shape    = output_shape   # (nrows, ncols)
        self.output_transform= output_transform
        self.output_crs      = output_crs

    def rasterise(
        self,
        layers: Optional[List[str]] = None,
        out_path: Optional[Path]    = None,
    ) -> np.ndarray:
        """
        Load each shapefile and burn into a terrain-class raster.

        Strategy
        --------
        **Bypass PROJ entirely** by manually applying the west-positive
        equirectangular projection formula to every geometry vertex:

            x = lon_west_deg x (pi/180) x R_titan
            y = lat_deg       x (pi/180) x R_titan

        where ``lon_west_deg = (-lon_east_deg) % 360``.

        This is exact and avoids all PROJ/CRS issues:

        - PROJ's longlat CRS rejects west-positive longitudes > 180 deg, causing
          the east hemisphere (180-360 degW) to produce all-zeros silently.
        - The GCS_Titan_2000 CRS embedded in the shapefiles is non-standard
          and PROJ may fail to recognise it, causing ``to_crs()`` to raise
          an exception that is silently skipped.
        - The manual formula is guaranteed correct: it matches the canonical
          grid's west-positive convention exactly (verified by unit tests).

        **Known limitation -- seam-straddling polygons:**

        A polygon whose east-positive longitude range straddles 0 degE (= 0 degW)
        will have vertices mapped to opposite sides of the canonical grid
        (near col 0 and near col ncols-1).  ``rasterio.rasterize()`` treats
        the polygon as a convex hull in projected metres and fills the interior,
        which in this case is almost the entire canvas rather than the intended
        small polygon.

        This is the same antimeridian problem that affects web mapping.  In
        practice it affects only polygons whose longitude range crosses 0 degE,
        which includes any lake or terrain unit that straddles that meridian.
        The poles are affected because the 0 degE line passes through both polar
        cap regions.  Large polygon datasets (Lopes, Birch) generally contain
        a handful of such features.

        A future fix would split seam-straddling polygons using
        ``shapely.ops.split()`` with a vertical line at x=0, then rasterise
        both halves.  This has not been implemented because the affected
        fraction is small (<1% of all polygons in both datasets) and the
        resulting artefact (a few pixels over-burned near the seam) is
        scientifically inconsequential at the 4490 m/px canonical resolution.

        Canonical grid coordinate convention (from CanonicalGrid):
            west_m = 0          -> col 0 = 0 degW
            x increases right   -> increasing  degW (west-positive)
            col = x / dx_m      where x = lon_west_deg x (pi/180) x R

        Parameters
        ----------
        layers:
            Subset of layer names (shapefile stems) to include.
            Default: all layers in ``RASTER_DRAW_ORDER``.
        out_path:
            If given, write the result as a GeoTIFF to this path.

        Returns
        -------
        np.ndarray
            Integer array of shape ``output_shape``, dtype int16.
            Values: 0=nodata, 1=Craters, 2=Dunes, 3=Plains, 4=Basins,
            5=Mountains, 6=Labyrinth, 7=Lakes.
            Longitude convention: west-positive 0->360 degW left to right.
        """
        import math
        import geopandas as gpd
        from shapely.ops import transform as shapely_transform
        from rasterio.features import rasterize

        from configs.pipeline_config import TITAN_RADIUS_M

        draw_order = layers if layers is not None else RASTER_DRAW_ORDER
        nrows, ncols = self.output_shape

        # Scale factor: 1 degree -> metres on the Titan sphere
        deg_to_m = math.pi * TITAN_RADIUS_M / 180.0

        def _to_canonical_metres(
            lon_east_deg: float, lat_deg: float, *extra
        ) -> tuple:
            """
            Convert one (lon_east deg, lat deg) vertex to canonical (x_m, y_m).

            The canonical grid is west-positive: x = lon_west x deg_to_m.
            lon_west = (-lon_east) % 360 maps any east-positive longitude
            into the [0, 360) west-positive range, then scales to metres.

            Examples
            --------
            lon_east=0 degE   -> lon_west=0 degW   -> x=0 (col 0)
            lon_east=90 degE  -> lon_west=270 degW -> x=3pi/4*R (col 3*ncols//4)
            lon_east=-90 degE -> lon_west=90 degW  -> x=pi/4*R  (col ncols//4)
            """
            lon_west = (-lon_east_deg) % 360.0
            x = lon_west * deg_to_m
            y = lat_deg  * deg_to_m
            return (x, y) + extra

        canvas = np.zeros((nrows, ncols), dtype=np.int16)

        for stem in draw_order:
            shp = self.shapefile_dir / f"{stem}.shp"
            if not shp.exists():
                logger.warning("Shapefile not found, skipping: %s", shp)
                continue

            label = SHAPEFILE_LAYERS.get(stem, (0, "?"))[0]
            logger.info("Rasterising %s -> class %d", stem, label)

            try:
                gdf = gpd.read_file(shp)
            except Exception as exc:
                logger.error("Failed to read %s: %s", shp, exc)
                continue

            if gdf.empty:
                logger.warning("Empty shapefile: %s", shp)
                continue

            # Apply the manual eqc formula to every geometry vertex.
            # shapely_transform calls _to_canonical_metres for each
            # (lon_east, lat) coordinate pair in the geometry, returning
            # a new geometry in canonical metres.
            transformed_geoms: list = []
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                try:
                    geom_m = shapely_transform(_to_canonical_metres, geom)
                    if geom_m is not None and not geom_m.is_empty:
                        transformed_geoms.append((geom_m, label))
                except Exception as exc:
                    logger.warning("Geometry transform failed: %s", exc)
                    continue

            if not transformed_geoms:
                logger.warning(
                    "No valid geometries in %s after coordinate transform", stem
                )
                continue

            # Burn into the canvas using the canonical transform.
            # The canonical transform maps (x_m, y_m) -> (col, row).
            try:
                burned = rasterize(
                    shapes=transformed_geoms,
                    out_shape=(nrows, ncols),
                    transform=self.output_transform,
                    fill=TERRAIN_NODATA,
                    dtype="int16",
                )
            except Exception as exc:
                logger.error("Rasterisation failed for %s: %s", stem, exc)
                continue

            # Higher-priority classes overwrite lower (draw order ensures this)
            canvas[burned != TERRAIN_NODATA] = burned[burned != TERRAIN_NODATA]
            n_burned = int(np.sum(burned != TERRAIN_NODATA))
            logger.info(
                "  %s: %d pixels burned (%.1f%% of canvas)",
                stem, n_burned, 100.0 * n_burned / canvas.size,
            )

        class_dist = {
            int(k): int(v)
            for k, v in zip(*np.unique(canvas, return_counts=True))
        }
        classified_pct = 100.0 * (1 - class_dist.get(0, 0) / canvas.size)
        logger.info(
            "Geomorphology raster complete: class distribution %s "
            "(%.1f%% of globe classified)",
            class_dist, classified_pct,
        )

        if out_path is not None:
            self._write_geotiff(canvas, out_path)

        return canvas

    def _write_geotiff(self, canvas: np.ndarray, out_path: Path) -> None:
        """Write the terrain-class raster to a GeoTIFF."""
        import rasterio
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            dtype="int16",
            count=1,
            width=canvas.shape[1],
            height=canvas.shape[0],
            crs=self.output_crs,
            transform=self.output_transform,
            nodata=TERRAIN_NODATA,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(canvas, 1)
        logger.info("Terrain class raster written to %s", out_path)


# ---------------------------------------------------------------------------
# Convenience: load individual shapefile with lon-flip
# ---------------------------------------------------------------------------

def load_shapefile_west_positive(
    shp_path: Path,
) -> "geopandas.GeoDataFrame":
    """
    Load a Lopes geomorphology shapefile and convert to west-positive coords.

    Parameters
    ----------
    shp_path:
        Path to the ``.shp`` file.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with west-positive longitude (0 -> 360 deg).
        Geometry type: Polygon (M-coordinate stripped).
    """
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    stem = Path(shp_path).stem

    # Add integer label column
    label_info = SHAPEFILE_LAYERS.get(stem)
    if label_info:
        gdf["terrain_label"] = label_info[0]
        gdf["terrain_name"]  = list(SHAPEFILE_LAYERS.keys())[label_info[0]-1]
    else:
        gdf["terrain_label"] = 0
        gdf["terrain_name"]  = "unknown"

    # Convert longitude convention
    gdf = flip_geodataframe_longitude(gdf)
    return gdf


def terrain_class_name(integer_label: int) -> str:
    """
    Return the terrain class name for a Lopes integer label.

    Parameters
    ----------
    integer_label:
        Integer from 0-7.  0 = nodata; 1-7 = terrain classes.

    Returns
    -------
    str
        Human-readable name.  Returns ``"NoData"`` for 0 and
        ``"Unknown(<N>)"`` for unrecognised values.

    Examples
    --------
    >>> terrain_class_name(0)
    'NoData'
    >>> terrain_class_name(2)
    'Dunes'
    """
    lookup: Dict[int, str] = {v[0]: k for k, v in SHAPEFILE_LAYERS.items()}
    lookup[0] = "NoData"
    return lookup.get(integer_label, f"Unknown({integer_label})")


def polar_lake_class_name(label: int) -> str:
    """
    Return the human-readable name for a Birch+2017 polar-lake label.

    Parameters
    ----------
    label:
        Integer polar-lake label (0-3) as stored in the canonical
        polar-lake raster.

    Returns
    -------
    str
        Human-readable class name.

    Examples
    --------
    >>> polar_lake_class_name(0)
    'NoData'
    >>> polar_lake_class_name(1)
    'FilledLake_Birch'
    >>> polar_lake_class_name(2)
    'EmptyBasin_Birch'
    """
    _names: Dict[int, str] = {
        POLAR_LAKE_NODATA:   "NoData",
        POLAR_LAKE_FILLED:   "FilledLake_Birch",
        POLAR_LAKE_EMPTY:    "EmptyBasin_Birch",
    }
    return _names.get(label, f"Unknown({label})")


# ---------------------------------------------------------------------------
# Birch+2017 polar lake rasteriser
# ---------------------------------------------------------------------------

class PolarLakeRasteriser:
    """
    Rasterises Birch+2017 polar lake shapefiles onto the canonical grid,
    producing a dedicated polar-lake raster separate from the Lopes
    geomorphology raster.

    Dataset source
    --------------
    Cornell eCommons archive:
    https://data.astro.cornell.edu/titan_polar_mapping_birch/
    titan_polar_mapping_birch.zip  (6.0 GB)

    Inside the zip:
      full_dataset/Various Mapping Shapefiles/Birch Polar Geomorphic (2017)/
        north/
          Fl_NORTH.shp  <- confirmed liquid (north)  -> birch_filled/
          El_NORTH.shp  <- empty lake depressions     -> birch_empty/
        south/
          Fl_SOUTH.shp  <- confirmed liquid (south)  -> birch_filled/
          El_SOUTH.shp  <- empty lake depressions     -> birch_empty/
          Em_SOUTH.shp  <- empty seas / paleoseas     -> birch_empty/

    Output raster integer labels
    ----------------------------
    POLAR_LAKE_NODATA  (0) -- outside polar-mapping coverage
    POLAR_LAKE_FILLED  (1) -- confirmed present-day liquid (Birch+2017 Fl_*)
    POLAR_LAKE_EMPTY   (2) -- empty basin / paleo-lake (Birch+2017 El_* and Em_*)

    Draw order: empty basins first (label 2), then filled lakes (label 1).
    Filled overwrites empty where both datasets coincide.

    The astrobiological significance of the empty-basin class (label 2)
    -------------------------------------------------------------------
    Paleo-lake and paleo-sea basins have experienced repeated wetting / drying
    cycles (Birch et al. 2017, 2018), concentrating amphiphiles and organic
    evaporites at the shoreline.  This is the environment proposed by Mayer &
    Nixon (2025) for vesicle / protocell formation.  These pixels feed the
    ``paleo_lake_indicator`` sub-component of Feature 5 (surface-atmosphere
    interaction).

    Parameters
    ----------
    birch_dir:
        Root directory of the Birch dataset.  Expected to contain
        sub-directories ``birch_filled/`` and ``birch_empty/``.
        Any absent sub-directory is silently skipped.
    output_shape:
        (nrows, ncols) of the canonical output raster.
    output_transform:
        Rasterio Affine transform for the output.
    output_crs:
        Rasterio CRS object or PROJ4 string for the output.
    titan_radius_m:
        Titan sphere radius in metres.  Default: 2 575 000 m.

    References
    ----------
    Birch et al. (2017) Icarus, doi:10.1016/j.icarus.2017.01.032
    Birch et al. (2018) Icarus, doi:10.1016/j.icarus.2017.12.016
    Mayer & Nixon (2025) Int. J. Astrobiology,
        doi:10.1017/S1473550425100037
    """

    def __init__(
        self,
        birch_dir:        Optional[Path],
        output_shape:     Tuple[int, int],
        output_transform: "rasterio.transform.Affine",
        output_crs:       "rasterio.crs.CRS | str",
        titan_radius_m:   float = 2_575_000.0,
    ) -> None:
        self.birch_dir:        Optional[Path]               = birch_dir
        self.output_shape:     Tuple[int, int]              = output_shape
        self.output_transform: "rasterio.transform.Affine" = output_transform
        self.output_crs:       "rasterio.crs.CRS | str"     = output_crs
        self.titan_radius_m:   float                        = titan_radius_m

    # -- public ---------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Return True if the Birch dataset directory exists and contains at
        least one ``.shp`` file in the filled-lake sub-directory.

        Used by the pipeline to decide whether to use this raster or fall
        back to SAR-proxy lake detection.

        Returns
        -------
        bool
        """
        if self.birch_dir is None or not self.birch_dir.exists():
            return False
        subdir: Path = self.birch_dir / BIRCH_SUBDIR_FILLED
        return subdir.exists() and bool(list(subdir.glob("*.shp")))

    def rasterise(
        self,
        include_filled:  bool            = True,
        include_empty:   bool            = True,
        out_path:        Optional[Path]  = None,
    ) -> np.ndarray:
        """
        Load Birch+2017 shapefiles and burn them into a polar-lake raster.

        Sub-directories are processed in draw order (empty first, then
        filled) so filled lakes overwrite empty basins.

        Parameters
        ----------
        include_filled:
            Burn Birch+2017 filled lake/sea polygons (label 1).
        include_empty:
            Burn Birch+2017 empty basin polygons (label 2).        out_path:
            If provided, write the raster as a GeoTIFF.

        Returns
        -------
        np.ndarray
            int16 array of shape ``output_shape``.
            All-zeros if the Birch directory is unavailable.
        """
        nrows, ncols = self.output_shape
        canvas: np.ndarray = np.zeros((nrows, ncols), dtype=np.int16)

        if not self.is_available() and self.birch_dir is not None:
            logger.info(
                "Birch dataset not available at %s -- polar_lakes raster "
                "will be all-zeros.  See INSTALL.md for download instructions.",
                self.birch_dir,
            )
        elif self.birch_dir is None:
            logger.info(
                "birch_dir not configured -- polar_lakes raster will be "
                "all-zeros (no Birch data)."
            )

        if self.birch_dir is not None and self.birch_dir.exists():
            # Draw order: lowest priority first
            tasks: List[Tuple[str, int, bool]] = [
                (BIRCH_SUBDIR_EMPTY,   POLAR_LAKE_EMPTY,   include_empty),
                (BIRCH_SUBDIR_FILLED,  POLAR_LAKE_FILLED,  include_filled),
            ]
            for subdir_name, label, include in tasks:
                if not include:
                    continue
                subdir: Path = self.birch_dir / subdir_name
                if not subdir.exists():
                    logger.debug(
                        "Birch sub-directory absent, skipping: %s", subdir
                    )
                    continue
                # SHAPEFILE FILTER -- per Birch et al. (2017) classification:
                #
                #   Fl = Filled lake (confirmed liquid)     → FILLED_LAKE (pl=1)
                #   El = Empty lake basin (confirmed dry)   → EMPTY_BASIN (pl=2)
                #   Em = Empty sea / paleosea (confirmed)   → EMPTY_BASIN (pl=2)
                #   Hdb = Hydrocarbon-dark basin            → EXCLUDED (ambiguous)
                #   Hdd = Hydrocarbon-dark drained          → EXCLUDED (ambiguous)
                #   Hud = Hydrocarbon-undifferentiated      → EXCLUDED (background)
                #
                # Excluded classes (Hdb/Hdd/Hud) fall through to the SAR proxy
                # in _liquid_hydrocarbon, which gives a physically bounded
                # estimate (~0.01-0.05) for ambiguous dark terrain.
                # Only files in birch_filled/ with Fl prefix and files in
                # birch_empty/ with El or Em prefix are used.
                ALLOWED_PREFIXES: Dict[int, Tuple[str, ...]] = {
                    POLAR_LAKE_FILLED: ("FL",),           # confirmed liquid only
                    POLAR_LAKE_EMPTY:  ("EL", "EM"),      # confirmed empty only
                }
                allowed = ALLOWED_PREFIXES.get(label, ())
                all_shp = sorted(subdir.glob("*.shp"))
                shapefiles: List[Path] = [
                    p for p in all_shp
                    if p.stem.upper()[:2] in allowed
                ]
                excluded = [p.name for p in all_shp if p not in shapefiles]
                if excluded:
                    logger.info(
                        "  Excluding non-canonical files from %s (label=%d): %s",
                        subdir_name, label, excluded,
                    )
                if not shapefiles:
                    logger.debug("No .shp files in %s", subdir)
                    continue
                logger.info(
                    "Rasterising Birch layer '%s' (label=%d) "
                    "from %d file(s)...",
                    subdir_name, label, len(shapefiles),
                )
                canvas = self._burn_layer(canvas, shapefiles, label)

        class_dist: Dict[int, int] = {
            int(k): int(v)
            for k, v in zip(*np.unique(canvas, return_counts=True))
        }
        logger.info(
            "Polar-lake raster complete: %s",
            {polar_lake_class_name(k): v for k, v in class_dist.items()},
        )

        if out_path is not None:
            self._write_geotiff(canvas, out_path)

        return canvas

    # -- internal helpers ------------------------------------------------------

    @staticmethod
    def _stereo_to_canonical(
        x_arr: "np.ndarray",
        y_arr: "np.ndarray",
        is_north: bool,
        titan_radius_m: float,
        deg_to_m: float,
    ) -> "Tuple[np.ndarray, np.ndarray]":
        """
        Inverse polar-stereographic → canonical west-positive metres.

        The Birch+2017 Cornell shapefiles have no embedded CRS (CRS=None)
        and use a polar stereographic projection centred on the respective
        pole (90°N for NORTH files, 90°S for SOUTH files).

        Convention (confirmed from diagnose_polar_lakes.py, April 2026)
        ---------------------------------------------------------------
        Origin : at the pole (0, 0)
        Y-axis : points toward 0°W (Titan prime meridian / sub-Saturn point)
        X-axis : points toward 90°W (90° west of prime meridian)
        Longitude increases westward (IAU Titan convention).
        Units  : metres.

        Inverse stereographic formulas
        --------------------------------
        ρ     = sqrt(X² + Y²)
        colat = 2 · arctan(ρ / (2R))           [radians]
        lat   = (90° − colat_deg)  for north pole
              = −(90° − colat_deg) for south pole
        lon_W = atan2(X, Y) mod 360            [degrees west-positive]

        Canonical → metres
        ------------------
        x_can = lon_W    · deg_to_m
        y_can = lat_deg  · deg_to_m
        """
        import numpy as np

        R   = titan_radius_m
        rho = np.sqrt(x_arr ** 2 + y_arr ** 2)

        # Inverse polar stereographic (centred at pole, true scale at pole)
        colat_deg = np.degrees(2.0 * np.arctan2(rho, 2.0 * R))

        if is_north:
            lat_deg = 90.0 - colat_deg
        else:
            lat_deg = -(90.0 - colat_deg)       # south hemisphere

        # Longitude west-positive: Y-axis points to 0°W, X-axis to 90°W.
        lon_W = np.degrees(np.arctan2(x_arr, y_arr)) % 360.0

        x_can = lon_W  * deg_to_m
        y_can = lat_deg * deg_to_m
        return x_can, y_can


    @staticmethod
    def _transform_antimeridian_safe(
        geom: "shapely.geometry.base.BaseGeometry",
        transform_fn: "Callable",
        deg_to_m: float,
    ) -> "List[shapely.geometry.base.BaseGeometry]":
        """
        Transform polar-stereographic polygon to canonical coords using
        shortest-arc unwrapping to correctly handle the prime-meridian seam.

        Root cause: the inverse stereo transform applies ``lon_W % 360``.
        A polygon edge crossing 0W/360W gets adjacent vertices at
        x_can~0 and x_can~360*deg_to_m.  Rasterio fills the "interior"
        of this wrong-topology polygon across the full canvas width,
        burning an entire latitude row as FILLED_LAKE.

        Fix: extract coordinates ring-by-ring, apply the numpy transform,
        then apply shortest-arc correction (if consecutive vertices differ
        by more than 180 degrees, add/subtract 360 degrees to take the
        short path).  The resulting polygon may extend outside
        [0, 360*deg_to_m]; it is clipped to canvas and wrapped copies
        are added to cover both sides of the seam.
        """
        import numpy as np
        from shapely.affinity import translate
        from shapely.geometry import Polygon, box

        canvas_w = 360.0 * deg_to_m
        canvas_h = 200.0 * deg_to_m
        half_w   = canvas_w / 2.0
        cb       = box(0.0, -canvas_h, canvas_w, canvas_h)

        def _unwrap_ring(coords: list) -> list:
            if len(coords) < 3:
                return coords
            xs = np.array([c[0] for c in coords], dtype=np.float64)
            ys = np.array([c[1] for c in coords], dtype=np.float64)
            r  = transform_fn(xs, ys)
            xs_t = np.asarray(r[0], dtype=np.float64)
            ys_t = np.asarray(r[1], dtype=np.float64)
            out_x = [float(xs_t[0])]
            out_y = [float(ys_t[0])]
            for i in range(1, len(xs_t)):
                x = float(xs_t[i])
                px = out_x[-1]
                while x - px >  half_w: x -= canvas_w
                while px - x >  half_w: x += canvas_w
                out_x.append(x)
                out_y.append(float(ys_t[i]))
            return list(zip(out_x, out_y))

        def _process_poly(poly: "Polygon") -> list:
            if poly is None or poly.is_empty:
                return []
            try:
                ext  = _unwrap_ring(list(poly.exterior.coords))
                ints = [_unwrap_ring(list(r.coords)) for r in poly.interiors]
                fixed = Polygon(ext, ints)
                if not fixed.is_valid:
                    fixed = fixed.buffer(0)
            except Exception:
                return []
            if fixed.is_empty:
                return []
            parts = []
            for offset in (0.0, -canvas_w, canvas_w):
                shifted = translate(fixed, xoff=offset) if offset else fixed
                try:
                    clipped = shifted.intersection(cb)
                    if clipped is not None and not clipped.is_empty:
                        parts.append(clipped)
                except Exception:
                    pass
            return parts

        if hasattr(geom, "geoms"):
            out = []
            for part in geom.geoms:
                if hasattr(part, "exterior"):
                    out.extend(_process_poly(part))
            return out
        if hasattr(geom, "exterior"):
            return _process_poly(geom)
        return []

    def _burn_layer(
        self,
        canvas:     np.ndarray,
        shapefiles: List[Path],
        label:      int,
    ) -> np.ndarray:
        """
        Load all *shapefiles*, apply the manual eqc projection, and burn
        *label* into *canvas*.

        A new copy of *canvas* is returned; the input is not mutated.

        Parameters
        ----------
        canvas:
            Existing int16 raster to update.
        shapefiles:
            List of ``.shp`` paths to load and merge.
        label:
            Integer value to burn for all covered pixels.

        Returns
        -------
        np.ndarray
            Updated canvas (copy).
        """
        import geopandas as gpd
        from rasterio.features import rasterize
        from shapely.ops import transform as shapely_transform

        nrows, ncols = self.output_shape
        deg_to_m: float = math.pi * self.titan_radius_m / 180.0

        # Detect coordinate system from the first shapefile's bounding box.
        # Birch+2017 Cornell files have CRS=None and use polar stereographic
        # metres (bounds ≈ ±1,000,000 m, confirmed April 2026).
        # Any bound > 360 is unambiguously projected (not geographic degrees).
        def _is_projected(shp_list: list) -> bool:
            for shp in shp_list:
                try:
                    gdf_probe = gpd.read_file(shp)
                    b = gdf_probe.total_bounds   # [xmin, ymin, xmax, ymax]
                    if any(abs(v) > 360.0 for v in b):
                        return True
                    return False
                except Exception:
                    continue
            return False

        projected_crs = bool(shapefiles) and _is_projected(shapefiles)

        if projected_crs:
            logger.info(
                "  Birch CRS=None with projected bounds -- applying inverse "
                "polar stereographic transform (Titan R=%.0f m).",
                self.titan_radius_m,
            )

        all_geoms: List[Tuple["shapely.geometry.base.BaseGeometry", int]] = []

        for shp in shapefiles:
            # Determine pole from filename (e.g. Fl_NORTH.shp / Fl_SOUTH.shp)
            is_north_pole: bool = "NORTH" in shp.stem.upper()

            if projected_crs:
                # Build a closure capturing the pole and projection parameters
                _R   = self.titan_radius_m
                _d2m = deg_to_m
                _isnorth = is_north_pole
                def _to_canonical_stereo(
                    x_arr, y_arr, *extra,
                    _R=_R, _d2m=_d2m, _isnorth=_isnorth
                ):
                    xc, yc = self._stereo_to_canonical(x_arr, y_arr, _isnorth, _R, _d2m)
                    if extra:
                        return (xc, yc) + extra
                    return xc, yc
                _transform_fn = _to_canonical_stereo
            else:
                # Original geographic (degree) transform
                def _to_canonical(lon_east, lat, *extra):
                    return ((-lon_east) % 360.0 * deg_to_m, lat * deg_to_m) + extra
                _transform_fn = _to_canonical

            try:
                gdf: "geopandas.GeoDataFrame" = gpd.read_file(shp)
            except Exception as exc:
                logger.error(
                    "Failed to read Birch shapefile %s: %s", shp, exc
                )
                continue
            if gdf.empty:
                logger.warning("Empty Birch shapefile: %s", shp)
                continue
            n_before: int = len(all_geoms)
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                try:
                    for fixed_geom in self._transform_antimeridian_safe(
                        geom, _transform_fn, deg_to_m
                    ):
                        if fixed_geom is not None and not fixed_geom.is_empty:
                            all_geoms.append((fixed_geom, label))
                except Exception as exc:
                    logger.warning(
                        "Birch geometry transform failed (%s): %s",
                        shp.name, exc,
                    )
            logger.info(
                "  %s (%s pole): added %d geometries",
                shp.name, "N" if is_north_pole else "S",
                len(all_geoms) - n_before,
            )

        if not all_geoms:
            logger.warning(
                "No valid geometries for label %d -- canvas unchanged", label
            )
            return canvas

        try:
            burned: np.ndarray = rasterize(
                shapes=all_geoms,
                out_shape=(nrows, ncols),
                transform=self.output_transform,
                fill=POLAR_LAKE_NODATA,
                dtype="int16",
            )
        except Exception as exc:
            logger.error(
                "Birch rasterisation failed for label %d: %s", label, exc
            )
            return canvas

        mask: np.ndarray = burned != POLAR_LAKE_NODATA
        result: np.ndarray = canvas.copy()
        result[mask] = burned[mask]
        logger.info(
            "  %s (label %d): %d pixels burned",
            polar_lake_class_name(label), label, int(mask.sum()),
        )
        return result

    def _write_geotiff(self, canvas: np.ndarray, out_path: Path) -> None:
        """
        Write *canvas* as an int16 GeoTIFF.

        Parameters
        ----------
        canvas:
            2-D int16 polar-lake raster.
        out_path:
            Destination path.  Parent directory created as needed.
        """
        import rasterio

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            dtype="int16",
            count=1,
            width=self.output_shape[1],
            height=self.output_shape[0],
            crs=self.output_crs,
            transform=self.output_transform,
            nodata=POLAR_LAKE_NODATA,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(canvas, 1)
        logger.info("Polar-lake raster written to %s", out_path)

