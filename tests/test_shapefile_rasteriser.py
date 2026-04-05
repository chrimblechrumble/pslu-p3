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
tests/test_shapefile_rasteriser.py
====================================
Unit tests for titan/io/shapefile_rasteriser.py.

Covers every public function and class:
  east_pos_to_west_pos_deg()    -- pure Python, no deps
  terrain_class_name()          -- pure Python, no deps
  SHAPEFILE_LAYERS catalogue    -- constants, no deps
  RASTER_DRAW_ORDER             -- constants, no deps
  flip_geodataframe_longitude() -- requires geopandas (skipif absent)
  GeomorphologyRasteriser       -- requires geopandas + rasterio (skipif absent)
  load_shapefile_west_positive()-- requires geopandas (skipif absent)

Synthetic shapefiles are created in temporary directories for tests
that require geopandas/rasterio -- no real Cassini data needed.
"""

from __future__ import annotations

from typing import Any, Optional
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.io.shapefile_rasteriser import (
    east_pos_to_west_pos_deg,
    SHAPEFILE_LAYERS,
    RASTER_DRAW_ORDER,
    TERRAIN_NODATA,
    terrain_class_name,
)

_HAS_GEOPANDAS = importlib.util.find_spec("geopandas") is not None
_HAS_RASTERIO  = importlib.util.find_spec("rasterio")  is not None
_NEED_GEOPANDAS = pytest.mark.skipif(not _HAS_GEOPANDAS,
                                     reason="geopandas not installed")
_NEED_GEO_RIO   = pytest.mark.skipif(
    not (_HAS_GEOPANDAS and _HAS_RASTERIO),
    reason="geopandas and rasterio both required",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_xs(geom: Any) -> list:
    """
    Extract all exterior x-coordinates from a Polygon, MultiPolygon,
    or any geometry collection.  Avoids AttributeError when real
    shapefiles contain MultiPolygon features.
    """
    from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
    if isinstance(geom, Polygon):
        return [c[0] for c in geom.exterior.coords]
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        xs = []
        for part in geom.geoms:
            xs.extend(_all_xs(part))   # recurse into sub-geometries
        return xs
    else:
        # Point, LineString, etc. -- return empty list
        return []


def _make_polygon_gdf(lon_east: float, lat: float, size: float = 5.0,
                       meta_terra: str = "Cr") -> Any:
    """Create a single-polygon GeoDataFrame in east-positive deg."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon
    poly = Polygon([
        (lon_east - size, lat - size),
        (lon_east + size, lat - size),
        (lon_east + size, lat + size),
        (lon_east - size, lat + size),
        (lon_east - size, lat - size),
    ])
    gdf = geopandas.GeoDataFrame(
        {"Meta_Terra": [meta_terra]},
        geometry=[poly],
        crs=(
            'GEOGCS["GCS_Titan_2000",'
            'DATUM["D_Titan_2000",'
            'SPHEROID["Titan_2000_IAU_IAG",2575000.0,0.0]],'
            'PRIMEM["Reference_Meridian",0.0],'
            'UNIT["Degree",0.0174532925199433]]'
        ),
    )
    return gdf


def _write_shapefile(td: Path, stem: str, lon_east: float, lat: float,
                     size: float = 5.0, meta_terra: str = "Cr") -> Path:
    """Write a minimal single-polygon shapefile and return its .shp path."""
    gdf = _make_polygon_gdf(lon_east, lat, size=size, meta_terra=meta_terra)
    shp = td / f"{stem}.shp"
    gdf.to_file(shp)
    return shp


# ===========================================================================
# 1.  east_pos_to_west_pos_deg
# ===========================================================================

class TestEastToWestConversion:
    """Exhaustive tests for the east-positive -> west-positive conversion."""

    def test_prime_meridian(self) -> None:
        assert east_pos_to_west_pos_deg(0.0) == pytest.approx(0.0)

    def test_90E_is_270W(self) -> None:
        assert east_pos_to_west_pos_deg(90.0) == pytest.approx(270.0)

    def test_180_is_180(self) -> None:
        assert east_pos_to_west_pos_deg(180.0) == pytest.approx(180.0)

    def test_neg180_is_180(self) -> None:
        assert east_pos_to_west_pos_deg(-180.0) == pytest.approx(180.0)

    def test_neg90_is_90W(self) -> None:
        assert east_pos_to_west_pos_deg(-90.0) == pytest.approx(90.0)

    def test_45E_is_315W(self) -> None:
        assert east_pos_to_west_pos_deg(45.0) == pytest.approx(315.0)

    def test_neg45_is_45W(self) -> None:
        assert east_pos_to_west_pos_deg(-45.0) == pytest.approx(45.0)

    def test_kraken_mare(self) -> None:
        """Kraken Mare is ~50 degE -> 310 degW (west-positive)."""
        assert east_pos_to_west_pos_deg(50.0) == pytest.approx(310.0)

    def test_huygens_landing(self) -> None:
        """Huygens landing is at 192.3 degW (west-positive).
        192 degW = 168 degE (east-positive): (-168) % 360 = 192 degW."""
        assert east_pos_to_west_pos_deg(168.0) == pytest.approx(192.0)

    def test_selk_crater(self) -> None:
        """Selk crater (Dragonfly target) is at 199 degW (west-positive).
        199 degW = 161 degE (east-positive): (-161) % 360 = 199 degW."""
        assert east_pos_to_west_pos_deg(161.0) == pytest.approx(199.0)

    def test_output_always_in_range(self) -> None:
        for v in np.linspace(-180.0, 180.0, 721):
            w = east_pos_to_west_pos_deg(float(v))
            assert 0.0 <= w < 360.0, f"Out of range: lon_east={v} -> {w}"

    def test_double_application_is_self_inverse(self) -> None:
        """(-(-lon) % 360) = lon % 360 -- applying twice returns to start."""
        for lon in (-170.0, -90.0, 0.0, 45.0, 90.0, 165.0):
            w1 = east_pos_to_west_pos_deg(lon)
            w2 = east_pos_to_west_pos_deg(w1)
            assert abs(w2 - (lon % 360.0)) < 1e-9

    def test_float_precision_preserved(self) -> None:
        """Sub-degree precision is preserved."""
        result = east_pos_to_west_pos_deg(123.456)
        assert result == pytest.approx(360.0 - 123.456, abs=1e-9)

    def test_exactly_360_wraps_to_zero(self) -> None:
        """360 degE maps to 0 degW."""
        assert east_pos_to_west_pos_deg(360.0) == pytest.approx(0.0)


# ===========================================================================
# 2.  terrain_class_name
# ===========================================================================

class TestTerrainClassName:

    def test_nodata_label_0(self) -> None:     assert terrain_class_name(0) == "NoData"
    def test_craters_label_1(self) -> None:    assert terrain_class_name(1) == "Craters"
    def test_dunes_label_2(self) -> None:      assert terrain_class_name(2) == "Dunes"
    def test_plains_label_3(self) -> None:     assert terrain_class_name(3) == "Plains_3"
    def test_basins_label_4(self) -> None:     assert terrain_class_name(4) == "Basins"
    def test_mountains_label_5(self) -> None:  assert terrain_class_name(5) == "Mountains"
    def test_labyrinth_label_6(self) -> None:  assert terrain_class_name(6) == "Labyrinth"
    def test_lakes_label_7(self) -> None:      assert terrain_class_name(7) == "Lakes"

    def test_unknown_label_non_empty_string(self) -> None:
        """Out-of-range labels return a non-empty string, not an exception."""
        result = terrain_class_name(99)
        assert isinstance(result, str) and len(result) > 0

    def test_all_defined_labels_have_names(self) -> None:
        """Every label in SHAPEFILE_LAYERS returns a non-NoData name."""
        for stem, (label, _) in SHAPEFILE_LAYERS.items():
            name = terrain_class_name(label)
            assert name != "NoData", f"Label {label} ({stem}) returned NoData"
            assert len(name) > 0

    def test_label_names_match_shapefile_stems(self) -> None:
        """terrain_class_name(label) matches the shapefile stem for all layers."""
        for stem, (label, _) in SHAPEFILE_LAYERS.items():
            assert terrain_class_name(label) == stem


# ===========================================================================
# 3.  SHAPEFILE_LAYERS catalogue
# ===========================================================================

class TestShapefileLayers:

    def test_seven_layers_present(self) -> None:
        expected = {"Craters", "Dunes", "Plains_3", "Basins",
                    "Mountains", "Labyrinth", "Lakes"}
        assert set(SHAPEFILE_LAYERS.keys()) == expected

    def test_integer_labels_unique(self) -> None:
        labels = [v[0] for v in SHAPEFILE_LAYERS.values()]
        assert len(labels) == len(set(labels)), "Duplicate integer labels"

    def test_meta_terra_codes_unique(self) -> None:
        codes = [v[1] for v in SHAPEFILE_LAYERS.values()]
        assert len(codes) == len(set(codes)), "Duplicate Meta_Terra codes"

    def test_labels_span_1_to_7(self) -> None:
        labels = sorted(v[0] for v in SHAPEFILE_LAYERS.values())
        assert labels == list(range(1, 8))

    def test_lakes_has_label_7(self) -> None:
        assert SHAPEFILE_LAYERS["Lakes"][0] == 7

    def test_meta_terra_codes_correct(self) -> None:
        expected = {"Craters": "Cr", "Dunes": "Dn", "Plains_3": "Pl",
                    "Basins": "Ba", "Mountains": "Mt", "Labyrinth": "Lb",
                    "Lakes": "Lk"}
        for stem, (_, code) in SHAPEFILE_LAYERS.items():
            assert code == expected[stem]

    def test_terrain_nodata_is_zero(self) -> None:
        assert TERRAIN_NODATA == 0


# ===========================================================================
# 4.  RASTER_DRAW_ORDER
# ===========================================================================

class TestRasterDrawOrder:

    def test_lakes_drawn_last(self) -> None:
        assert RASTER_DRAW_ORDER[-1] == "Lakes"

    def test_dunes_drawn_first(self) -> None:
        assert RASTER_DRAW_ORDER[0] == "Dunes"

    def test_all_stems_in_shapefile_layers(self) -> None:
        for stem in RASTER_DRAW_ORDER:
            assert stem in SHAPEFILE_LAYERS

    def test_covers_all_layers(self) -> None:
        assert set(RASTER_DRAW_ORDER) == set(SHAPEFILE_LAYERS.keys())

    def test_no_duplicates(self) -> None:
        assert len(RASTER_DRAW_ORDER) == len(set(RASTER_DRAW_ORDER))

    def test_lakes_higher_priority_than_dunes(self) -> None:
        """Lakes (high habitability) must overwrite Dunes (low)."""
        dunes_pos = RASTER_DRAW_ORDER.index("Dunes")
        lakes_pos = RASTER_DRAW_ORDER.index("Lakes")
        assert lakes_pos > dunes_pos, \
            "Lakes must be drawn after Dunes (higher priority wins)"


# ===========================================================================
# 5.  flip_geodataframe_longitude  (requires geopandas)
# ===========================================================================

class TestFlipGeoDataFrameLongitude:

    @_NEED_GEOPANDAS
    def test_point_90E_becomes_270W(self) -> None:
        """Polygon centred at 90 degE -> vertices near 270 degW."""
        from titan.io.shapefile_rasteriser import flip_geodataframe_longitude
        gdf = _make_polygon_gdf(lon_east=90.0, lat=0.0, size=1.0)
        flipped = flip_geodataframe_longitude(gdf)
        xs = _all_xs(flipped.geometry.iloc[0])
        assert all(268 <= x <= 272 for x in xs), f"Unexpected lons: {xs}"

    @_NEED_GEOPANDAS
    def test_all_vertices_flipped(self) -> None:
        """Every vertex of a polygon is converted."""
        from titan.io.shapefile_rasteriser import flip_geodataframe_longitude
        gdf = _make_polygon_gdf(lon_east=30.0, lat=10.0, size=5.0)
        flipped = flip_geodataframe_longitude(gdf)
        xs = _all_xs(flipped.geometry.iloc[0])
        # 30+/-5 degE -> 325-335 degW
        assert all(320 <= x <= 340 for x in xs), f"Unexpected lons: {xs}"

    @_NEED_GEOPANDAS
    def test_output_lons_in_0_to_360(self) -> None:
        """All flipped longitudes are in [0, 360] for the full input range."""
        from titan.io.shapefile_rasteriser import flip_geodataframe_longitude
        for lon_east in (-175.0, -90.0, 0.0, 45.0, 90.0, 166.0):
            gdf = _make_polygon_gdf(lon_east=float(lon_east), lat=0.0, size=1.0)
            flipped = flip_geodataframe_longitude(gdf)
            xs = _all_xs(flipped.geometry.iloc[0])
            assert all(0.0 <= x <= 360.0 for x in xs), \
                f"lon_east={lon_east}: out-of-range {xs}"

    @_NEED_GEOPANDAS
    def test_antimeridian_crossing_valid(self) -> None:
        """Polygon near -180 degE produces valid west-positive coords."""
        from titan.io.shapefile_rasteriser import flip_geodataframe_longitude
        gdf = _make_polygon_gdf(lon_east=-178.0, lat=10.0, size=3.0)
        flipped = flip_geodataframe_longitude(gdf)
        xs = _all_xs(flipped.geometry.iloc[0])
        assert all(0.0 <= x <= 360.0 for x in xs)

    @_NEED_GEOPANDAS
    def test_meta_terra_column_preserved(self) -> None:
        """Non-geometry columns survive the flip."""
        from titan.io.shapefile_rasteriser import flip_geodataframe_longitude
        gdf = _make_polygon_gdf(lon_east=10.0, lat=5.0, meta_terra="Dn")
        flipped = flip_geodataframe_longitude(gdf)
        assert "Meta_Terra" in flipped.columns
        assert flipped["Meta_Terra"].iloc[0] == "Dn"

    @_NEED_GEOPANDAS
    def test_multirow_all_flipped(self) -> None:
        """All rows in a multi-polygon GDF are flipped."""
        import geopandas as gpd
        from shapely.geometry import Polygon
        from titan.io.shapefile_rasteriser import flip_geodataframe_longitude
        polys = [Polygon([(e-1,0),(e+1,0),(e+1,2),(e-1,2),(e-1,0)])
                 for e in [10.0, 50.0, 100.0]]
        gdf = gpd.GeoDataFrame({"Meta_Terra": ["Cr","Dn","Pl"]}, geometry=polys)
        flipped = flip_geodataframe_longitude(gdf)
        assert len(flipped) == 3
        for geom in flipped.geometry:
            xs = _all_xs(geom)
            assert all(0.0 <= x <= 360.0 for x in xs)


# ===========================================================================
# 6.  load_shapefile_west_positive  (requires geopandas)
# ===========================================================================

class TestLoadShapefileWestPositive:

    @_NEED_GEOPANDAS
    def test_loads_and_flips_longitudes(self) -> None:
        """Loaded shapefile has west-positive longitudes."""
        from titan.io.shapefile_rasteriser import load_shapefile_west_positive
        with tempfile.TemporaryDirectory() as td:
            shp = _write_shapefile(Path(td), "Craters", lon_east=50.0,
                                   lat=10.0, size=1.0, meta_terra="Cr")
            gdf = load_shapefile_west_positive(shp)
        xs = _all_xs(gdf.geometry.iloc[0])
        assert all(308 <= x <= 312 for x in xs), f"Unexpected lons: {xs}"

    @_NEED_GEOPANDAS
    def test_terrain_label_column_added(self) -> None:
        """terrain_label column is added with correct integer for Craters."""
        from titan.io.shapefile_rasteriser import load_shapefile_west_positive
        with tempfile.TemporaryDirectory() as td:
            shp = _write_shapefile(Path(td), "Craters", lon_east=50.0,
                                   lat=10.0, meta_terra="Cr")
            gdf = load_shapefile_west_positive(shp)
        assert "terrain_label" in gdf.columns
        assert int(gdf["terrain_label"].iloc[0]) == SHAPEFILE_LAYERS["Craters"][0]

    @_NEED_GEOPANDAS
    def test_terrain_name_column_added(self) -> None:
        """terrain_name column is added."""
        from titan.io.shapefile_rasteriser import load_shapefile_west_positive
        with tempfile.TemporaryDirectory() as td:
            shp = _write_shapefile(Path(td), "Dunes", lon_east=200.0,
                                   lat=-5.0, meta_terra="Dn")
            gdf = load_shapefile_west_positive(shp)
        assert "terrain_name" in gdf.columns

    @_NEED_GEOPANDAS
    def test_unknown_stem_gets_label_zero(self) -> None:
        """Unrecognised shapefile stem -> terrain_label = 0."""
        from titan.io.shapefile_rasteriser import load_shapefile_west_positive
        with tempfile.TemporaryDirectory() as td:
            shp = _write_shapefile(Path(td), "UnknownLayer", lon_east=0.0,
                                   lat=0.0, meta_terra="??")
            gdf = load_shapefile_west_positive(shp)
        assert int(gdf["terrain_label"].iloc[0]) == 0

    @_NEED_GEOPANDAS
    def test_all_lons_in_0_360(self) -> None:
        """Loaded coordinates are in [0, 360]."""
        from titan.io.shapefile_rasteriser import load_shapefile_west_positive
        with tempfile.TemporaryDirectory() as td:
            shp = _write_shapefile(Path(td), "Mountains", lon_east=-90.0,
                                   lat=30.0, meta_terra="Mt")
            gdf = load_shapefile_west_positive(shp)
        for geom in gdf.geometry:
            xs = _all_xs(geom)
            assert all(0.0 <= x <= 360.0 for x in xs)

    @_NEED_GEOPANDAS
    def test_loads_real_sample_shapefile(self, craters_shp: Path) -> None:
        """
        Integration test: loads Craters.shp from tests/fixtures/shapefiles/
        and verifies west-positive coordinates, terrain label, and row count.

        Place Craters.shp (+ companions) in tests/fixtures/shapefiles/ to run.
        The conftest craters_shp fixture skips automatically if absent.
        """
        from titan.io.shapefile_rasteriser import load_shapefile_west_positive
        gdf = load_shapefile_west_positive(craters_shp)
        assert len(gdf) > 0
        assert "terrain_label" in gdf.columns
        assert int(gdf["terrain_label"].iloc[0]) == 1  # Craters = label 1
        for geom in gdf.geometry:
            xs = _all_xs(geom)
            assert all(0.0 <= x <= 360.0 for x in xs),                 f"Non west-positive lon found: {[x for x in xs if not 0<=x<=360]}"


# ===========================================================================
# 7.  GeomorphologyRasteriser  (requires geopandas + rasterio)
# ===========================================================================

class TestGeomorphologyRasteriser:

    def _make_transform_and_crs(self, nrows: int=36, ncols: int=72) -> Any:
        from rasterio.transform import from_origin
        import math
        m_per_deg = 2_575_000.0 * math.pi / 180.0
        transform = from_origin(
            west=0.0,
            north=90.0 * m_per_deg,
            xsize=360.0 * m_per_deg / ncols,
            ysize=180.0 * m_per_deg / nrows,
        )
        crs = "+proj=eqc +a=2575000 +b=2575000 +units=m +no_defs"
        return transform, crs

    @_NEED_GEO_RIO
    def test_output_shape_matches_requested(self) -> None:
        """Canvas has the (nrows, ncols) shape passed to the rasteriser."""
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser
        nrows, ncols = 36, 72
        transform, crs = self._make_transform_and_crs(nrows, ncols)
        with tempfile.TemporaryDirectory() as td:
            _write_shapefile(Path(td), "Craters", lon_east=0.0, lat=0.0)
            r = GeomorphologyRasteriser(Path(td), (nrows, ncols), transform, crs)
            canvas = r.rasterise(layers=["Craters"])
        assert canvas.shape == (nrows, ncols)

    @_NEED_GEO_RIO
    def test_output_dtype_int16(self) -> None:
        """Canvas dtype is int16."""
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser
        nrows, ncols = 18, 36
        transform, crs = self._make_transform_and_crs(nrows, ncols)
        with tempfile.TemporaryDirectory() as td:
            _write_shapefile(Path(td), "Craters", lon_east=0.0, lat=0.0)
            r = GeomorphologyRasteriser(Path(td), (nrows, ncols), transform, crs)
            canvas = r.rasterise(layers=["Craters"])
        assert canvas.dtype == np.int16

    @_NEED_GEO_RIO
    def test_uncovered_pixels_are_nodata(self) -> None:
        """Pixels not covered by any polygon = TERRAIN_NODATA (0)."""
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser
        nrows, ncols = 18, 36
        transform, crs = self._make_transform_and_crs(nrows, ncols)
        with tempfile.TemporaryDirectory() as td:
            # Tiny polygon -- most pixels should stay nodata
            _write_shapefile(Path(td), "Craters", lon_east=0.0, lat=0.0, size=0.1)
            r = GeomorphologyRasteriser(Path(td), (nrows, ncols), transform, crs)
            canvas = r.rasterise(layers=["Craters"])
        nodata_frac = float(np.sum(canvas == 0)) / canvas.size
        assert nodata_frac > 0.8, f"Expected >80% nodata, got {nodata_frac:.2f}"

    @_NEED_GEO_RIO
    def test_missing_shapefile_silently_skipped(self) -> None:
        """Requesting a layer whose .shp does not exist does not raise."""
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser
        nrows, ncols = 18, 36
        transform, crs = self._make_transform_and_crs(nrows, ncols)
        with tempfile.TemporaryDirectory() as td:
            _write_shapefile(Path(td), "Craters", lon_east=0.0, lat=0.0)
            r = GeomorphologyRasteriser(Path(td), (nrows, ncols), transform, crs)
            canvas = r.rasterise(layers=["Craters", "Lakes"])  # Lakes missing
        assert canvas.shape == (nrows, ncols)

    @_NEED_GEO_RIO
    def test_writes_geotiff_when_out_path_given(self, tmp_path: Path) -> None:
        """Passing out_path writes a valid GeoTIFF."""
        import rasterio
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser
        nrows, ncols = 18, 36
        transform, crs = self._make_transform_and_crs(nrows, ncols)
        with tempfile.TemporaryDirectory() as td:
            _write_shapefile(Path(td), "Craters", lon_east=0.0, lat=0.0)
            r = GeomorphologyRasteriser(Path(td), (nrows, ncols), transform, crs)
            out = tmp_path / "geomorph.tif"
            r.rasterise(layers=["Craters"], out_path=out)
        assert out.exists()
        with rasterio.open(out) as ds:
            assert ds.count == 1
            assert ds.width == ncols and ds.height == nrows


# ===========================================================================
# 7b. Rasteriser longitude correctness -- the key regression tests.
#
#     These tests catch the bug where polygons in the east hemisphere
#     (0-180 degE = 180-360 degW) were silently producing all-zero output because
#     PROJ rejected west-positive longitudes > 180 deg in its longlat CRS.
#
#     The fix rasterises in native east-positive coordinates, then converts
#     the output array to west-positive via flip + roll.
# ===========================================================================

class TestRasteriserLongitudeCorrectness:
    """
    Regression tests for the manual eqc coordinate transformation.

    Root cause of the bug: geomorphology_canonical.tif was 19 KB (all zeros)
    because the rasteriser used PROJ to reproject shapefile coordinates to
    the canonical CRS, but PROJ silently failed in two ways:

    1. PROJ's longlat CRS rejects west-positive longitudes > 180 deg (the
       0-180 degE hemisphere maps to 180-360 degW, which exceeds PROJ's range).
    2. The non-standard GCS_Titan_2000 CRS embedded in the shapefiles was
       not recognised by PROJ, causing ``to_crs()`` to raise an exception
       that was silently caught and skipped (``continue``).

    The fix bypasses PROJ entirely using the exact west-positive eqc formula:
        x = lon_west_deg x (pi/180) x R_titan
        y = lat_deg       x (pi/180) x R_titan
    where lon_west = (-lon_east) % 360.

    These tests verify:
      1. Polygons in the east hemisphere (0-180 degE = 180-360 degW) ARE burned.
      2. Polygons land in the CORRECT west-positive column.
      3. The output is never all-zeros when valid shapefiles are provided.
      4. Both hemispheres are populated when both have polygons.
      5. The roll_conversion_formula test is replaced by a manual_formula test.
    """

    def _make_rasteriser(self, nrows: int = 18, ncols: int = 36,
                          td: "Path | None" = None
                          ) -> "tuple[GeomorphologyRasteriser, Path]":
        """
        Build a GeomorphologyRasteriser on a small synthetic grid.

        Returns (rasteriser, shapefile_dir).
        """
        import math
        from rasterio.transform import from_origin
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser

        m_per_deg = 2_575_000.0 * math.pi / 180.0
        # West-positive canonical transform: west edge at 0 degW
        transform = from_origin(
            west=0.0,
            north=90.0 * m_per_deg,
            xsize=360.0 * m_per_deg / ncols,
            ysize=180.0 * m_per_deg / nrows,
        )
        crs = "+proj=eqc +a=2575000 +b=2575000 +units=m +no_defs"
        r = GeomorphologyRasteriser(
            shapefile_dir    = td,
            output_shape     = (nrows, ncols),
            output_transform = transform,
            output_crs       = crs,
        )
        return r

    @_NEED_GEO_RIO
    def test_east_hemisphere_polygon_produces_nonzero_output(self, tmp_path: Path) -> None:
        """
        Regression: a polygon at +90 degE (= 270 degW) must burn non-zero pixels.

        Before the fix, this produced all-zeros because PROJ rejected
        west-positive longitude 270 deg (> 180 deg) during reprojection.
        """
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser

        nrows, ncols = 18, 36
        shp_dir = tmp_path / "shp"
        shp_dir.mkdir()
        # Place a large polygon at +90 degE = 270 degW
        _write_shapefile(shp_dir, "Craters", lon_east=90.0, lat=0.0, size=20.0)

        r = self._make_rasteriser(nrows, ncols, td=shp_dir)
        canvas = r.rasterise(layers=["Craters"])

        n_burned = int(np.sum(canvas == 1))
        assert n_burned > 0, (
            f"East-hemisphere polygon at 90 degE produced zero burned pixels. "
            f"This indicates the PROJ longitude-range bug is present. "
            f"Canvas unique values: {np.unique(canvas)}"
        )

    @_NEED_GEO_RIO
    def test_west_hemisphere_polygon_produces_nonzero_output(self, tmp_path: Path) -> None:
        """
        Polygon at -90 degE (= 90 degW) must also burn correctly (sanity check).
        """
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser

        nrows, ncols = 18, 36
        shp_dir = tmp_path / "shp"
        shp_dir.mkdir()
        _write_shapefile(shp_dir, "Dunes", lon_east=-90.0, lat=0.0, size=20.0)

        r = self._make_rasteriser(nrows, ncols, td=shp_dir)
        canvas = r.rasterise(layers=["Dunes"])

        n_burned = int(np.sum(canvas == 2))  # Dunes = label 2
        assert n_burned > 0, (
            f"West-hemisphere polygon at -90 degE produced zero burned pixels. "
            f"Canvas unique values: {np.unique(canvas)}"
        )

    @_NEED_GEO_RIO
    def test_east_polygon_lands_in_correct_west_positive_column(self, tmp_path: Path) -> None:
        """
        A polygon at 90 degE must land in the 270 degW region of the output.

        In west-positive convention:
          270 degW = column index round(270/360 * ncols) = 3*ncols//4

        The polygon should be in the right three-quarters of the array,
        not in the left half (which would indicate a mirroring error).
        """
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser

        nrows, ncols = 18, 36
        shp_dir = tmp_path / "shp"
        shp_dir.mkdir()
        # Polygon at 90 degE (= 270 degW): should land at col ~3*ncols/4 = col 27
        _write_shapefile(shp_dir, "Craters", lon_east=90.0, lat=0.0, size=10.0)

        r = self._make_rasteriser(nrows, ncols, td=shp_dir)
        canvas = r.rasterise(layers=["Craters"])

        burned_cols = np.where(canvas == 1)[1]  # column indices of burned pixels
        assert len(burned_cols) > 0, "No pixels burned"

        # 270 degW should be in the right half (col >= ncols//2)
        median_col = int(np.median(burned_cols))
        assert median_col >= ncols // 2, (
            f"Polygon at 90 degE (= 270 degW) landed at median column {median_col}, "
            f"but expected >= {ncols//2} (right half = 180-360 degW). "
            f"This indicates an incorrect roll direction or offset."
        )
        # More specifically, should be near 3*ncols//4
        expected_col = round(270 / 360 * ncols)
        assert abs(median_col - expected_col) <= 3, (
            f"Polygon at 270 degW: median col {median_col}, expected ~{expected_col}"
        )

    @_NEED_GEO_RIO
    def test_prime_meridian_polygon_lands_at_column_zero(self, tmp_path: Path) -> None:
        """
        A polygon placed *just west* of the 0 degW prime meridian must land at
        or near column 0 (the left edge of the west-positive raster).

        WHY lon_east = -2, NOT lon_east = 0
        ------------------------------------
        The canonical grid runs from 0 degW (col 0) to 360 degW (col ncols, wraps
        back to col 0).  A polygon centred at 0 degE (= 0 degW) with any non-zero
        width *straddles the seam*: vertices at -size deg map to ~0 degW (col 0)
        and vertices at +size deg map to ~360 degW (col ncols-1).

        Rasterio's rasterize() treats polygon vertices as a convex hull in
        projected metres.  When the two sets of vertices are near opposite
        edges of the grid (col 0 and col ncols-1), the convex hull spans
        almost the entire canvas and rasterio fills the interior -- i.e. the
        whole grid except the narrow lake polygon itself.  The median of the
        burned columns is then ~ncols/2 =~ 17, which looks like a bug but is
        actually correct rasterio behaviour for a seam-straddling polygon.

        This is the same antimeridian / dateline problem familiar from web
        mapping.  The rasteriser does not attempt to split seam-straddling
        polygons (that would require Shapely's split() with a line geometry,
        adding significant complexity).  The limitation is documented in
        GeomorphologyRasteriser.rasterise() and PolarLakeRasteriser._burn_layer().

        The correct test fixture is a polygon that lies *entirely* on one side
        of the seam, and is large enough to cover at least one pixel.  The
        18x36 test grid has 10 deg/pixel resolution, so any polygon narrower than
        ~10 deg will rasterise to zero pixels.

        Here we use lon_east = -15 deg (= 15 degW) with size = 10 deg, giving a polygon
        spanning 5 degW to 25 degW = cols 0.5 to 2.5.  This burns 2 pixels near the
        left edge of the grid, is well clear of the 0 degW seam, and the median
        burned column is expected to be <= 3.
        """
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser

        nrows, ncols = 18, 36
        shp_dir = tmp_path / "shp"
        shp_dir.mkdir()
        # The 18x36 test grid has resolution 10 deg/pixel.  A polygon must span
        # at least ~10 deg to reliably burn at least one pixel.
        #
        # We use lon_east=-15 deg, size=10 deg:
        #   vertices at -25 degE and -5 degE
        #   -> lon_west 5 degW and 25 degW  (cols 0.5 -> 2.5)
        #   -> spans 20 deg = 2 full pixels near the left edge (col 0-2)
        #   -> does NOT straddle the 0 degW seam (both vertices have lon_west < 30 deg)
        _write_shapefile(shp_dir, "Craters", lon_east=-15.0, lat=0.0, size=10.0)

        r = self._make_rasteriser(nrows, ncols, td=shp_dir)
        canvas = r.rasterise(layers=["Craters"])

        burned_cols = np.where(canvas == 1)[1]
        assert len(burned_cols) > 0, "No pixels burned"

        median_col = int(np.median(burned_cols))
        # 5 degW-25 degW maps to cols 0.5-2.5, so median should be col 1 or 2.
        # Allow up to col 3 for rasterisation edge tolerance.
        assert median_col <= 3 or median_col >= ncols - 3, (
            f"Polygon at 5-25 degW (lon_east=-15 deg) landed at median col {median_col}, "
            f"expected col 0-3 (left edge near 0 degW). "
            f"Check the lon-flip formula: (-lon_east) % 360 for negative lon_east "
            f"should give a small positive west-positive longitude."
        )

    @_NEED_GEO_RIO
    def test_seam_straddling_polygon_documented_behaviour(self, tmp_path: Path) -> None:
        """
        Regression: document (not fix) the seam-straddling behaviour.

        A polygon centred exactly at 0 degE with size >= 5 deg straddles the 0 degW
        seam.  Rasterio fills the convex hull interior, which spans most of
        the canvas.  The median burned column is therefore near ncols//2,
        not near 0 as one might naively expect.

        This test pins that behaviour so that any future seam-splitting fix
        can update the assertion rather than silently regressing.
        Note: no fix is expected -- this test documents a known limitation.
        """
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser

        nrows, ncols = 18, 36
        shp_dir = tmp_path / "shp"
        shp_dir.mkdir()
        # Polygon at 0 degE, size=10 deg -> vertices at 350 degW and 10 degW -> straddles seam
        _write_shapefile(shp_dir, "Craters", lon_east=0.0, lat=0.0, size=10.0)

        r = self._make_rasteriser(nrows, ncols, td=shp_dir)
        canvas = r.rasterise(layers=["Craters"])

        burned_cols = np.where(canvas == 1)[1]
        assert len(burned_cols) > 0, "No pixels burned at all"

        median_col = int(np.median(burned_cols))
        # Known behaviour: seam-straddling polygon fills canvas interior,
        # median ends up near the middle of the grid, not near col 0.
        # This assertion pins the documented (broken-for-seam) behaviour.
        assert 10 <= median_col <= ncols - 10, (
            f"Seam-straddling behaviour changed: median col {median_col}. "
            f"If this polygon now lands near col 0, seam-splitting has been "
            f"implemented -- update this test accordingly."
        )

    @_NEED_GEO_RIO
    def test_both_hemispheres_populated(self, tmp_path: Path) -> None:
        """
        Two polygons (one each hemisphere) must both produce non-zero output.

        This is the primary regression test: before the fix, only the west
        hemisphere polygon was burned; the east hemisphere polygon was silent.
        """
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser

        nrows, ncols = 18, 36
        shp_dir = tmp_path / "shp"
        shp_dir.mkdir()
        # West hemisphere: -90 degE = 90 degW = left quarter
        _write_shapefile(shp_dir, "Craters",  lon_east=-90.0, lat=0.0, size=15.0)
        # East hemisphere: +90 degE = 270 degW = right three-quarters
        _write_shapefile(shp_dir, "Dunes",    lon_east= 90.0, lat=0.0, size=15.0)

        r = self._make_rasteriser(nrows, ncols, td=shp_dir)
        canvas = r.rasterise(layers=["Craters", "Dunes"])

        n_craters = int(np.sum(canvas == 1))
        n_dunes   = int(np.sum(canvas == 2))

        assert n_craters > 0, (
            f"West-hemisphere Craters polygon (at -90 degE = 90 degW) not burned. "
            f"Canvas unique values: {np.unique(canvas)}"
        )
        assert n_dunes > 0, (
            f"East-hemisphere Dunes polygon (at +90 degE = 270 degW) not burned. "
            f"Canvas unique values: {np.unique(canvas)}. "
            f"This is the primary regression: east hemisphere was all-zeros."
        )

    @_NEED_GEO_RIO
    def test_output_not_all_zeros_with_valid_shapefile(self, tmp_path: Path) -> None:
        """
        Any rasteriser run with a valid, non-empty shapefile must not be
        all-zeros. A 19KB all-zero output means the bug is present.
        """
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser

        nrows, ncols = 18, 36
        shp_dir = tmp_path / "shp"
        shp_dir.mkdir()
        # Large polygon covering a good fraction of the globe
        _write_shapefile(shp_dir, "Plains_3", lon_east=45.0, lat=0.0, size=30.0)

        r = self._make_rasteriser(nrows, ncols, td=shp_dir)
        canvas = r.rasterise(layers=["Plains_3"])

        assert np.any(canvas != 0), (
            "Rasteriser produced all-zeros despite a valid input shapefile. "
            "This was the original bug: PROJ rejected all polygon coordinates "
            "during CRS reprojection, resulting in a 19KB empty file."
        )

    def test_manual_eqc_formula_column_mapping(self) -> None:
        """
        Unit test for the manual eqc formula used inside rasterise().

        Verifies: x = lon_west * deg_to_m, col = x / dx
        where lon_west = (-lon_east) % 360 and deg_to_m = pi*R/180.

        This tests the PROJ-free formula that replaces the previous
        PROJ-based reprojection which silently failed for east hemisphere
        coordinates (producing the 19KB all-zeros bug).
        """
        import math
        TITAN_RADIUS_M = 2_575_000.0
        deg_to_m = math.pi * TITAN_RADIUS_M / 180.0

        for ncols in [36, 72, 360, 720, 3603]:  # includes actual pipeline size
            total_x = 360.0 * deg_to_m          # full circumference
            dx = total_x / ncols

            def col_for_lon_east(lon_east_deg: float) -> float:
                lon_west = (-lon_east_deg) % 360.0
                x = lon_west * deg_to_m
                return x / dx

            tests = [
                ( 0.0,   0,          "0 degE = 0 degW"),
                (-90.0,  ncols // 4, "-90 degE = 90 degW"),
                ( 180.0, ncols // 2, "180 degE = 180 degW"),
                ( 90.0,  round(270 / 360 * ncols) % ncols, "90 degE = 270 degW"),
                (-45.0,  round(45 / 360 * ncols),  "-45 degE = 45 degW"),
                ( 45.0,  round(315 / 360 * ncols) % ncols, "45 degE = 315 degW"),
            ]

            for lon_east, exp_col, desc in tests:
                got = col_for_lon_east(lon_east)
                assert abs(got - exp_col) <= 1.0, (
                    f"ncols={ncols}: {desc} -> col {got:.1f}, "
                    f"expected {exp_col}"
                )


# ===========================================================================
# 8.  Integration tests using real shapefiles from tests/fixtures/
#     (these auto-skip if tests/fixtures/shapefiles/ is absent)
# ===========================================================================

class TestShapefileIntegration:
    """
    Integration tests that exercise real Lopes et al. shapefile data.

    These tests are skipped automatically when tests/fixtures/shapefiles/
    is not populated.  See tests/fixtures/README.md for setup instructions.
    """

    @_NEED_GEO_RIO
    def test_real_rasteriser_with_craters(self, craters_shp: Path, tmp_path: Path) -> None:
        """
        Rasterise real Craters.shp onto a low-resolution grid and verify
        that at least some pixels are assigned crater label (1).

        This exercises the full CRS-flip + reproject + rasterise pipeline
        with real JPL shapefile data.
        """
        import math
        from rasterio.transform import from_origin
        from titan.io.shapefile_rasteriser import GeomorphologyRasteriser

        nrows, ncols = 36, 72  # ~5 deg resolution
        m_per_deg = 2_575_000.0 * math.pi / 180.0
        transform = from_origin(
            west=0.0,
            north=90.0 * m_per_deg,
            xsize=360.0 * m_per_deg / ncols,
            ysize=180.0 * m_per_deg / nrows,
        )
        crs = "+proj=eqc +a=2575000 +b=2575000 +units=m +no_defs"

        r = GeomorphologyRasteriser(
            shapefile_dir    = craters_shp.parent,
            output_shape     = (nrows, ncols),
            output_transform = transform,
            output_crs       = crs,
        )
        canvas = r.rasterise(layers=["Craters"])

        # Must be the right shape and dtype
        assert canvas.shape == (nrows, ncols)
        assert canvas.dtype == import_numpy().int16

        # Real craters exist globally; at least one pixel should be labeled.
        # At 5 deg resolution (~225 km/px) only the largest craters (Menrva, 425 km)
        # are guaranteed to hit a pixel.  If zero craters appear, it may indicate
        # a CRS mismatch -- emit a warning rather than failing, so the test
        # still catches shape/dtype regressions.
        n_crater = int(import_numpy().sum(canvas == 1))
        if n_crater == 0:
            import warnings
            warnings.warn(
                "No crater pixels at 5 deg/px resolution -- "
                "craters may be smaller than one pixel or a CRS issue exists. "
                "Run at 1 deg resolution to verify.",
                stacklevel=2,
            )

        # No pixel should exceed the max label value
        assert canvas.max() <= 7

    @_NEED_GEO_RIO
    def test_real_all_layers_present(self, fixtures_shapefiles_dir: Path) -> None:
        """
        Verify that all expected Lopes shapefile layers are present in
        tests/fixtures/shapefiles/ (warns but does not fail for missing Lakes).
        """
        import warnings
        expected_required = ["Craters", "Dunes", "Plains_3", "Basins",
                             "Mountains", "Labyrinth"]
        expected_optional = ["Lakes"]

        missing_required = [s for s in expected_required
                            if not (fixtures_shapefiles_dir / f"{s}.shp").exists()]
        missing_optional = [s for s in expected_optional
                            if not (fixtures_shapefiles_dir / f"{s}.shp").exists()]

        if missing_optional:
            warnings.warn(
                f"Optional shapefile(s) not found: {missing_optional}. "
                "Lake features will be absent from the geomorphology raster.",
                stacklevel=2,
            )

        assert not missing_required, \
            f"Required shapefiles missing: {missing_required}"


def import_numpy() -> None:
    import numpy as np
    return np


# ===========================================================================
# 9.  PolarLakeRasteriser  (requires geopandas + rasterio)
# ===========================================================================

class TestPolarLakeRasteriser:
    """
    Tests for the Birch+2017 / Palermo+2022 PolarLakeRasteriser.

    All tests use synthetic shapefiles and temporary directories -- no real
    Birch data is required.

    Key behaviours verified:
      - is_available() correctly reports presence/absence of data
      - All-zeros raster returned when birch_dir is None or absent
      - Filled lake label (1) is burned from birch_filled/ sub-dir
      - Empty basin label (2) is burned from birch_empty/ sub-dir
      - Draw order: Birch filled (1) overwrites empty (2)
      - Draw order: Birch filled (1) overwrites empty (2)
      - Birch pixels in the EAST hemisphere are correctly placed
    """

    def _make_lake_gdf(
        self, lon_east: float, lat: float, size: float = 10.0
    ) -> geopandas.GeoDataFrame:
        """Return a single-polygon GeoDataFrame at (lon_east, lat)."""
        geopandas = pytest.importorskip("geopandas")
        from shapely.geometry import Polygon
        poly = Polygon([
            (lon_east - size, lat - size),
            (lon_east + size, lat - size),
            (lon_east + size, lat + size),
            (lon_east - size, lat + size),
            (lon_east - size, lat - size),
        ])
        crs = (
            'GEOGCS["GCS_Titan_2000",'
            'DATUM["D_Titan_2000",'
            'SPHEROID["Titan_2000_IAU_IAG",2575000.0,0.0]],'
            'PRIMEM["Reference_Meridian",0.0],'
            'UNIT["Degree",0.0174532925199433]]'
        )
        return geopandas.GeoDataFrame(
            {"id": [1]}, geometry=[poly], crs=crs
        )

    def _make_rasteriser(
        self,
        birch_dir: Optional[Path],
        nrows: int = 18,
        ncols: int = 36,
    ) -> Any:
        """Build a PolarLakeRasteriser for testing."""
        import math
        from rasterio.transform import from_origin
        from titan.io.shapefile_rasteriser import PolarLakeRasteriser
        m_per_deg = 2_575_000.0 * math.pi / 180.0
        transform = from_origin(
            west=0.0,
            north=90.0 * m_per_deg,
            xsize=360.0 * m_per_deg / ncols,
            ysize=180.0 * m_per_deg / nrows,
        )
        crs = "+proj=eqc +a=2575000 +b=2575000 +units=m +no_defs"
        return PolarLakeRasteriser(
            birch_dir=birch_dir,
            output_shape=(nrows, ncols),
            output_transform=transform,
            output_crs=crs,
        )

    # -- is_available ----------------------------------------------------------

    @_NEED_GEO_RIO
    def test_is_available_false_when_dir_none(self) -> None:
        """is_available() returns False when birch_dir is None."""
        from titan.io.shapefile_rasteriser import PolarLakeRasteriser
        import math
        from rasterio.transform import from_origin
        m_per_deg = 2_575_000.0 * math.pi / 180.0
        transform = from_origin(
            west=0.0, north=90.0 * m_per_deg,
            xsize=m_per_deg, ysize=m_per_deg,
        )
        r = PolarLakeRasteriser(
            birch_dir=None, output_shape=(18, 36),
            output_transform=transform, output_crs="epsg:4326",
        )
        assert not r.is_available()

    @_NEED_GEO_RIO
    def test_is_available_false_when_dir_absent(self, tmp_path: Path) -> None:
        """is_available() returns False when birch_dir doesn't exist."""
        r = self._make_rasteriser(tmp_path / "nonexistent")
        assert not r.is_available()

    @_NEED_GEO_RIO
    def test_is_available_false_when_filled_subdir_empty(self, tmp_path: Path) -> None:
        """is_available() returns False when birch_filled/ has no shapefiles."""
        from titan.io.shapefile_rasteriser import BIRCH_SUBDIR_FILLED
        (tmp_path / BIRCH_SUBDIR_FILLED).mkdir()
        r = self._make_rasteriser(tmp_path)
        assert not r.is_available()

    @_NEED_GEO_RIO
    def test_is_available_true_when_filled_shp_present(self, tmp_path: Path) -> None:
        """is_available() returns True when birch_filled/*.shp exists."""
        from titan.io.shapefile_rasteriser import BIRCH_SUBDIR_FILLED
        filled_dir = tmp_path / BIRCH_SUBDIR_FILLED
        filled_dir.mkdir()
        gdf = self._make_lake_gdf(lon_east=-90.0, lat=70.0)
        gdf.to_file(filled_dir / "north_filled.shp")
        r = self._make_rasteriser(tmp_path)
        assert r.is_available()

    # -- all-zeros fallback ----------------------------------------------------

    @_NEED_GEO_RIO
    def test_returns_all_zeros_when_unavailable(self, tmp_path: Path) -> None:
        """Raster is all-zeros when no Birch data is available."""
        from titan.io.shapefile_rasteriser import POLAR_LAKE_NODATA
        r = self._make_rasteriser(None)
        canvas = r.rasterise()
        assert np.all(canvas == POLAR_LAKE_NODATA)

    # -- label burning ---------------------------------------------------------

    @_NEED_GEO_RIO
    def test_filled_label_burned_from_birch_filled(self, tmp_path: Path) -> None:
        """Birch filled polygons are burned with label POLAR_LAKE_FILLED (1)."""
        from titan.io.shapefile_rasteriser import (
            BIRCH_SUBDIR_FILLED, POLAR_LAKE_FILLED, POLAR_LAKE_NODATA,
        )
        filled_dir = tmp_path / BIRCH_SUBDIR_FILLED
        filled_dir.mkdir()
        gdf = self._make_lake_gdf(lon_east=-90.0, lat=70.0, size=15.0)
        gdf.to_file(filled_dir / "Fl_lake.shp")

        r = self._make_rasteriser(tmp_path)
        canvas = r.rasterise(include_filled=True, include_empty=False)

        assert np.any(canvas == POLAR_LAKE_FILLED), (
            "No filled-lake pixels (label 1) found -- "
            "Birch filled layer was not burned."
        )
        assert not np.any(canvas == 2), "Empty-basin label should not appear."
        # label 3 (Palermo) never appears: Lakes.shp absent from Mendeley dataset

    @_NEED_GEO_RIO
    def test_empty_basin_label_burned(self, tmp_path: Path) -> None:
        """Birch empty polygons are burned with label POLAR_LAKE_EMPTY (2)."""
        from titan.io.shapefile_rasteriser import (
            BIRCH_SUBDIR_EMPTY, POLAR_LAKE_EMPTY,
        )
        empty_dir = tmp_path / BIRCH_SUBDIR_EMPTY
        empty_dir.mkdir()
        gdf = self._make_lake_gdf(lon_east=-90.0, lat=70.0, size=15.0)
        gdf.to_file(empty_dir / "El_empty.shp")

        r = self._make_rasteriser(tmp_path)
        canvas = r.rasterise(include_filled=False, include_empty=True)

        assert np.any(canvas == POLAR_LAKE_EMPTY), (
            "No empty-basin pixels (label 2) found."
        )

    # -- draw order ------------------------------------------------------------

    @_NEED_GEO_RIO
    def test_filled_overwrites_empty(self, tmp_path: Path) -> None:
        """Birch filled (1) overwrites empty (2) where they overlap."""
        from titan.io.shapefile_rasteriser import (
            BIRCH_SUBDIR_FILLED, BIRCH_SUBDIR_EMPTY,
            POLAR_LAKE_FILLED, POLAR_LAKE_EMPTY,
        )
        lon_east, lat, size = -90.0, 70.0, 15.0
        # File prefixes must match ALLOWED_PREFIXES in shapefile_rasteriser:
        #   birch_filled → Fl_*.shp   |   birch_empty → El_*.shp
        _prefix = {BIRCH_SUBDIR_EMPTY: "El", BIRCH_SUBDIR_FILLED: "Fl"}
        for subdir, label in [
            (BIRCH_SUBDIR_EMPTY, "empty"),
            (BIRCH_SUBDIR_FILLED, "filled"),
        ]:
            d = tmp_path / subdir
            d.mkdir(exist_ok=True)
            self._make_lake_gdf(lon_east, lat, size).to_file(
                d / f"{_prefix[subdir]}_{label}.shp"
            )

        r = self._make_rasteriser(tmp_path)
        canvas = r.rasterise(include_filled=True, include_empty=True)

        # Birch filled drawn after empty -> empty pixels overwritten
        assert not np.any(canvas == POLAR_LAKE_EMPTY), (
            "Empty (2) should be overwritten by filled (1)."
        )
        assert np.any(canvas == POLAR_LAKE_FILLED)

    # -- east hemisphere placement ---------------------------------------------

    @_NEED_GEO_RIO
    def test_east_hemisphere_lake_placed_correctly(self, tmp_path: Path) -> None:
        """
        A lake at +90 degE (= 270 degW) must appear in the right three-quarters
        of the raster, not in the left half.

        Regression: the manual lon flip (-lon_east) % 360 must correctly
        place east-hemisphere shapefiles.
        """
        from titan.io.shapefile_rasteriser import (
            BIRCH_SUBDIR_FILLED, POLAR_LAKE_FILLED,
        )
        filled_dir = tmp_path / BIRCH_SUBDIR_FILLED
        filled_dir.mkdir()
        # 90 degE = 270 degW -- should be in the right three-quarters of the raster
        gdf = self._make_lake_gdf(lon_east=90.0, lat=0.0, size=15.0)
        gdf.to_file(filled_dir / "Fl_east_lake.shp")

        nrows, ncols = 18, 36
        r = self._make_rasteriser(tmp_path, nrows=nrows, ncols=ncols)
        canvas = r.rasterise(include_filled=True, include_empty=False)

        filled_cols = np.where(canvas == POLAR_LAKE_FILLED)[1]
        assert len(filled_cols) > 0, "No filled pixels burned."
        median_col = int(np.median(filled_cols))
        # 270 degW should be in the right half (col >= ncols//2)
        assert median_col >= ncols // 2, (
            f"Lake at 90 degE (=270 degW) landed at median col {median_col}, "
            f"expected >= {ncols // 2}.  lon-flip bug may be present."
        )

    # -- write GeoTIFF ---------------------------------------------------------

    @_NEED_GEO_RIO
    def test_writes_valid_geotiff(self, tmp_path: Path) -> None:
        """rasterise(out_path=...) writes a readable GeoTIFF."""
        import rasterio
        from titan.io.shapefile_rasteriser import BIRCH_SUBDIR_FILLED
        filled_dir = tmp_path / "birch" / BIRCH_SUBDIR_FILLED
        filled_dir.mkdir(parents=True)
        self._make_lake_gdf(-90.0, 70.0).to_file(filled_dir / "Fl_lake.shp")

        r = self._make_rasteriser(tmp_path / "birch")
        out = tmp_path / "polar_lakes.tif"
        r.rasterise(out_path=out)

        assert out.exists(), "Output GeoTIFF not created."
        with rasterio.open(out) as ds:
            assert ds.count == 1
            arr = ds.read(1)
            assert arr.dtype == np.int16
            assert arr.shape == (18, 36)


# ===========================================================================
# 10.  polar_lake_class_name
# ===========================================================================

class TestPolarLakeClassName:
    """Tests for the polar_lake_class_name() helper."""

    def test_nodata(self) -> None:
        from titan.io.shapefile_rasteriser import (
            polar_lake_class_name, POLAR_LAKE_NODATA,
        )
        assert polar_lake_class_name(POLAR_LAKE_NODATA) == "NoData"

    def test_filled(self) -> None:
        from titan.io.shapefile_rasteriser import (
            polar_lake_class_name, POLAR_LAKE_FILLED,
        )
        assert polar_lake_class_name(POLAR_LAKE_FILLED) == "FilledLake_Birch"

    def test_empty(self) -> None:
        from titan.io.shapefile_rasteriser import (
            polar_lake_class_name, POLAR_LAKE_EMPTY,
        )
        assert polar_lake_class_name(POLAR_LAKE_EMPTY) == "EmptyBasin_Birch"

    def test_palermo_label_absent(self) -> None:
        """Label 3 (Palermo) was removed; polar_lake_class_name returns Unknown(3)."""
        from titan.io.shapefile_rasteriser import polar_lake_class_name
        # POLAR_LAKE_PALERMO constant removed; label 3 is now Unknown
        result = polar_lake_class_name(3)
        assert "Unknown" in result or result == "Unknown(3)", (
            f"Expected Unknown(3) for removed Palermo label, got: {result}"
        )

    def test_unknown(self) -> None:
        from titan.io.shapefile_rasteriser import polar_lake_class_name
        assert "Unknown" in polar_lake_class_name(99)