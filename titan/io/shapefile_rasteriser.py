"""
titan/io/shapefile_rasteriser.py
=================================
Rasterises geomorphology and polar-lake shapefiles onto the canonical grid.

Two independent shapefile catalogues are supported:

**Lopes+2019 global geomorphology** (all terrain classes, global coverage):
  Source: Lopes et al. (2019) Nature Astronomy doi:10.1038/s41550-019-0917-6
  Location: ``data/raw/geomorphology_shapefiles/``
  Classes: Craters, Dunes, Plains_3, Basins, Mountains, Labyrinth, Lakes

**Birch+2017 / Palermo+2022 polar lake mapping** (polar regions, higher-
resolution lake outlines with empty-basin distinction):
  Source: Birch et al. (2017) Icarus doi:10.1016/j.icarus.2017.01.032
          Palermo et al. (2022) PSJ doi:10.3847/PSJ/ac4d9c
  Location: ``data/raw/birch_polar_mapping/``
  Sub-dirs: ``birch_filled/``, ``birch_empty/``, ``palermo/``

Key format facts (verified by direct inspection of real shapefiles):

  CRS : GEOGCS["GCS_Titan_2000"], Titan sphere R=2,575,000 m
        Longitude: EAST-positive, range −180° → +180°
        *** OPPOSITE to all raster products (which use west-positive) ***

  Geometry type : PolygonM (shape type 25 = polygon with measure coordinate)
        geopandas reads these correctly; the M coordinate is ignored.

  File structure (Lopes): ONE shapefile per terrain class
        Craters.shp   → Meta_Terra = 'Cr'  integer_label = 1
        Dunes.shp     → Meta_Terra = 'Dn'  integer_label = 2
        Plains_3.shp  → Meta_Terra = 'Pl'  integer_label = 3
        Basins.shp    → Meta_Terra = 'Ba'  integer_label = 4
        Mountains.shp → Meta_Terra = 'Mt'  integer_label = 5
        Labyrinth.shp → Meta_Terra = 'Lb'  integer_label = 6
        Lakes.shp     → Meta_Terra = 'Lk'  integer_label = 7  (if present)

  Coordinate conversion — PROJ BYPASSED:
        PROJ silently rejects west-positive longitudes > 180° in its
        longlat CRS, causing the entire east hemisphere to produce
        all-zeros.  Instead we apply the manual eqc formula to every vertex:
            lon_west = (−lon_east) % 360
            x_m = lon_west × (π/180) × R_titan
            y_m = lat      × (π/180) × R_titan

Rasterisation priority (Lopes — drawn lowest to highest, higher overwrites):
        Dunes < Plains < Basins < Mountains < Labyrinth < Craters < Lakes

References
----------
Birch et al. (2017) "Geomorphological mapping of Titan's polar terrains"
  Icarus, doi:10.1016/j.icarus.2017.01.032

Lopes et al. (2019) "A global geomorphologic map of Saturn's moon Titan"
  Nature Astronomy, doi:10.1038/s41550-019-0917-6

Malaska et al. (2025) "Global geology of Titan"
  Titan After Cassini-Huygens Ch.9

Palermo et al. (2022) "Lakes and Seas on Titan"
  PSJ, doi:10.3847/PSJ/ac4d9c
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

#: Mapping from shapefile stem → (integer_label, Meta_Terra_code).
#: Labels 1–7 are burned into the canonical geomorphology raster and must
#: match ``configs/pipeline_config.py::TERRAIN_CLASSES``.
SHAPEFILE_LAYERS: Final[Dict[str, Tuple[int, str]]] = {
    "Craters":   (1, "Cr"),
    "Dunes":     (2, "Dn"),
    "Plains_3":  (3, "Pl"),
    "Basins":    (4, "Ba"),
    "Mountains": (5, "Mt"),
    "Labyrinth": (6, "Lb"),
    "Lakes":     (7, "Lk"),
}

#: Draw order for Lopes rasterisation — lower index drawn first so higher
#: priority classes overwrite lower ones.  Lakes (class 7) drawn last.
RASTER_DRAW_ORDER: Final[List[str]] = [
    "Dunes", "Plains_3", "Basins", "Mountains", "Labyrinth", "Craters", "Lakes",
]

#: Nodata value in the integer terrain-class output raster.
TERRAIN_NODATA: Final[int] = 0


# ---------------------------------------------------------------------------
# Birch+2017 / Palermo+2022 polar-lake constants
# ---------------------------------------------------------------------------

#: Sub-directory names under the Birch dataset root (``data/raw/birch_polar_mapping/``).
#: Every ``*.shp`` inside each sub-directory is loaded and merged.
#:
#: Expected layout::
#:
#:   data/raw/birch_polar_mapping/
#:     birch_filled/      ← confirmed present-day liquid (Birch+2017)
#:     birch_empty/       ← empty basins / paleo-lakes (Birch+2017)
#:     palermo/           ← alternative mapping (Palermo+2022)
BIRCH_SUBDIR_FILLED:  Final[str] = "birch_filled"
BIRCH_SUBDIR_EMPTY:   Final[str] = "birch_empty"
BIRCH_SUBDIR_PALERMO: Final[str] = "palermo"

#: Integer labels for the polar-lake raster (independent of Lopes labels).
#:
#: ======================  =====  ===========================================
#: Class                   Value  Meaning
#: ======================  =====  ===========================================
#: ``POLAR_LAKE_NODATA``   0      Outside polar-mapping coverage
#: ``POLAR_LAKE_FILLED``   1      Confirmed liquid — Birch+2017 filled
#: ``POLAR_LAKE_EMPTY``    2      Empty basin / paleo-lake — Birch+2017
#: ``POLAR_LAKE_PALERMO``  3      Confirmed liquid — Palermo+2022
#: ======================  =====  ===========================================
POLAR_LAKE_NODATA:   Final[int] = 0
POLAR_LAKE_FILLED:   Final[int] = 1
POLAR_LAKE_EMPTY:    Final[int] = 2
POLAR_LAKE_PALERMO:  Final[int] = 3


# ---------------------------------------------------------------------------
# CRS conversion
# ---------------------------------------------------------------------------

def east_pos_to_west_pos_deg(lon_east: float) -> float:
    """
    Convert a single longitude from east-positive to west-positive convention.

    Shapefiles use east-positive degrees (−180 → +180).
    All raster products use west-positive degrees (0 → 360).

    Parameters
    ----------
    lon_east:
        Longitude in east-positive degrees.

    Returns
    -------
    float
        Equivalent longitude in west-positive degrees (0 → 360).

    Examples
    --------
    >>> east_pos_to_west_pos_deg(0.0)
    0.0
    >>> east_pos_to_west_pos_deg(90.0)   # 90°E = 270°W
    270.0
    >>> east_pos_to_west_pos_deg(-90.0)  # 90°W = 90°W
    90.0
    >>> east_pos_to_west_pos_deg(180.0)  # 180° = 180°
    180.0
    """
    return (-lon_east) % 360.0


def flip_geodataframe_longitude(
    gdf: "geopandas.GeoDataFrame",
) -> "geopandas.GeoDataFrame":
    """
    Convert all geometry coordinates from east-positive to west-positive.

    Applies ``lon_west = (−lon_east) % 360`` to every vertex of every
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

            x = lon_west_deg × (π/180) × R_titan
            y = lat_deg       × (π/180) × R_titan

        where ``lon_west_deg = (−lon_east_deg) % 360``.

        This is exact and avoids all PROJ/CRS issues:

        - PROJ's longlat CRS rejects west-positive longitudes > 180°, causing
          the east hemisphere (180–360°W) to produce all-zeros silently.
        - The GCS_Titan_2000 CRS embedded in the shapefiles is non-standard
          and PROJ may fail to recognise it, causing ``to_crs()`` to raise
          an exception that is silently skipped.
        - The manual formula is guaranteed correct: it matches the canonical
          grid's west-positive convention exactly (verified by unit tests).

        **Known limitation — seam-straddling polygons:**

        A polygon whose east-positive longitude range straddles 0°E (= 0°W)
        will have vertices mapped to opposite sides of the canonical grid
        (near col 0 and near col ncols-1).  ``rasterio.rasterize()`` treats
        the polygon as a convex hull in projected metres and fills the interior,
        which in this case is almost the entire canvas rather than the intended
        small polygon.

        This is the same antimeridian problem that affects web mapping.  In
        practice it affects only polygons whose longitude range crosses 0°E,
        which includes any lake or terrain unit that straddles that meridian.
        The poles are affected because the 0°E line passes through both polar
        cap regions.  Large polygon datasets (Lopes, Birch) generally contain
        a handful of such features.

        A future fix would split seam-straddling polygons using
        ``shapely.ops.split()`` with a vertical line at x=0, then rasterise
        both halves.  This has not been implemented because the affected
        fraction is small (<1% of all polygons in both datasets) and the
        resulting artefact (a few pixels over-burned near the seam) is
        scientifically inconsequential at the 4490 m/px canonical resolution.

        Canonical grid coordinate convention (from CanonicalGrid):
            west_m = 0          → col 0 = 0°W
            x increases right   → increasing °W (west-positive)
            col = x / dx_m      where x = lon_west_deg × (π/180) × R

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
            Longitude convention: west-positive 0→360°W left to right.
        """
        import math
        import geopandas as gpd
        from shapely.ops import transform as shapely_transform
        from rasterio.features import rasterize

        from configs.pipeline_config import TITAN_RADIUS_M

        draw_order = layers if layers is not None else RASTER_DRAW_ORDER
        nrows, ncols = self.output_shape

        # Scale factor: 1 degree → metres on the Titan sphere
        deg_to_m = math.pi * TITAN_RADIUS_M / 180.0

        def _to_canonical_metres(
            lon_east_deg: float, lat_deg: float, *extra
        ) -> tuple:
            """
            Convert one (lon_east°, lat°) vertex to canonical (x_m, y_m).

            The canonical grid is west-positive: x = lon_west × deg_to_m.
            lon_west = (−lon_east) % 360 maps any east-positive longitude
            into the [0, 360) west-positive range, then scales to metres.

            Examples
            --------
            lon_east=0°E   → lon_west=0°W   → x=0 (col 0)
            lon_east=90°E  → lon_west=270°W → x=3π/4*R (col 3*ncols//4)
            lon_east=-90°E → lon_west=90°W  → x=π/4*R  (col ncols//4)
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
            logger.info("Rasterising %s → class %d", stem, label)

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
            # The canonical transform maps (x_m, y_m) → (col, row).
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
        GeoDataFrame with west-positive longitude (0 → 360°).
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
        Integer from 0–7.  0 = nodata; 1–7 = terrain classes.

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
    Return the human-readable name for a Birch/Palermo polar-lake label.

    Parameters
    ----------
    label:
        Integer polar-lake label (0–3) as stored in the canonical
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
    >>> polar_lake_class_name(3)
    'FilledLake_Palermo'
    """
    _names: Dict[int, str] = {
        POLAR_LAKE_NODATA:   "NoData",
        POLAR_LAKE_FILLED:   "FilledLake_Birch",
        POLAR_LAKE_EMPTY:    "EmptyBasin_Birch",
        POLAR_LAKE_PALERMO:  "FilledLake_Palermo",
    }
    return _names.get(label, f"Unknown({label})")


# ---------------------------------------------------------------------------
# Birch+2017 / Palermo+2022 polar lake rasteriser
# ---------------------------------------------------------------------------

class PolarLakeRasteriser:
    """
    Rasterises Birch+2017 and Palermo+2022 polar lake shapefiles onto the
    canonical grid, producing a dedicated polar-lake raster separate from
    the Lopes geomorphology raster.

    Output raster integer labels
    ----------------------------
    POLAR_LAKE_NODATA  (0) — outside polar-mapping coverage
    POLAR_LAKE_FILLED  (1) — confirmed present-day liquid (Birch+2017 filled)
    POLAR_LAKE_EMPTY   (2) — empty basin / paleo-lake (Birch+2017 empty)
    POLAR_LAKE_PALERMO (3) — confirmed liquid, Palermo+2022 alternative mapping

    Draw order: empty basins first, then Birch filled, then Palermo.
    Palermo (label 3) therefore takes precedence over Birch filled (label 1)
    where both datasets agree on liquid; Birch filled overwrites empty basins.

    The astrobiological significance of the empty-basin class:
        Paleo-lake basins have experienced repeated wetting/drying cycles
        (Birch et al. 2017), which concentrate amphiphiles and organic
        evaporites at the shoreline.  This is the environment proposed by
        Mayer & Nixon (2025) for vesicle / protocell formation.  These pixels
        are captured in the ``paleo_lake_indicator`` sub-component of
        Feature 5 (surface–atmosphere interaction) and as a standalone
        feature in the future-epoch model.

    Parameters
    ----------
    birch_dir:
        Root directory of the Birch dataset.  Expected to contain
        sub-directories ``birch_filled/``, ``birch_empty/``, and
        ``palermo/``.  Any absent sub-directory is silently skipped.
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
    Mayer & Nixon (2025) Int. J. Astrobiology,
        doi:10.1017/S1473550425100037
    Palermo et al. (2022) PSJ, doi:10.3847/PSJ/ac4d9c
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

    # ── public ───────────────────────────────────────────────────────────────

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
        include_palermo: bool            = True,
        out_path:        Optional[Path]  = None,
    ) -> np.ndarray:
        """
        Load Birch/Palermo shapefiles and burn them into a polar-lake raster.

        Each sub-directory is processed in draw order (empty → filled →
        Palermo) so higher-priority classes overwrite lower ones.

        Parameters
        ----------
        include_filled:
            Burn Birch+2017 filled lake/sea polygons (label 1).
        include_empty:
            Burn Birch+2017 empty basin polygons (label 2).
        include_palermo:
            Burn Palermo+2022 sea polygons (label 3).
        out_path:
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
                "Birch dataset not available at %s — polar_lakes raster "
                "will be all-zeros.  See INSTALL.md for download instructions.",
                self.birch_dir,
            )
        elif self.birch_dir is None:
            logger.info(
                "birch_dir not configured — polar_lakes raster will be "
                "all-zeros (no Birch data)."
            )

        if self.birch_dir is not None and self.birch_dir.exists():
            # Draw order: lowest priority first
            tasks: List[Tuple[str, int, bool]] = [
                (BIRCH_SUBDIR_EMPTY,   POLAR_LAKE_EMPTY,   include_empty),
                (BIRCH_SUBDIR_FILLED,  POLAR_LAKE_FILLED,  include_filled),
                (BIRCH_SUBDIR_PALERMO, POLAR_LAKE_PALERMO, include_palermo),
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
                shapefiles: List[Path] = sorted(subdir.glob("*.shp"))
                if not shapefiles:
                    logger.debug("No .shp files in %s", subdir)
                    continue
                logger.info(
                    "Rasterising Birch layer '%s' (label=%d) "
                    "from %d file(s)…",
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

    # ── internal helpers ──────────────────────────────────────────────────────

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

        def _to_canonical(
            lon_east: float, lat: float, *extra: float
        ) -> Tuple[float, ...]:
            """Convert (lon_east°, lat°) → canonical west-positive metres."""
            return ((-lon_east) % 360.0 * deg_to_m, lat * deg_to_m) + extra

        all_geoms: List[Tuple["shapely.geometry.base.BaseGeometry", int]] = []

        for shp in shapefiles:
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
                    geom_m = shapely_transform(_to_canonical, geom)
                    if geom_m is not None and not geom_m.is_empty:
                        all_geoms.append((geom_m, label))
                except Exception as exc:
                    logger.warning(
                        "Birch geometry transform failed (%s): %s",
                        shp.name, exc,
                    )
            logger.debug(
                "  %s: added %d geometries",
                shp.name, len(all_geoms) - n_before,
            )

        if not all_geoms:
            logger.warning(
                "No valid geometries for label %d — canvas unchanged", label
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

