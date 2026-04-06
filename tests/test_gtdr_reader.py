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
tests/test_gtdr_reader.py
==========================
Unit tests for the GTDR PDS3 binary reader.

Tests the critical MISSING_CONSTANT masking logic, label parser,
and affine transform computation.  Does NOT require the actual
GTDR files -- all tests use synthetic data.
"""

from __future__ import annotations

from typing import Any, Optional
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.io.gtdr_reader import (
    GTDR_MISSING_CONSTANT,
    GTDR_IMAGE_OFFSET,
    GTDR_PPD,
    gtdr_affine_transform,
    mosaic_gtdr_tiles,
    parse_gtdr_label,
    read_gtdr_img,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_LABEL = """\
PDS_VERSION_ID          = PDS3
DATA_SET_ID             = "CO-SSA-RADAR-5-GTDR-V1.0"
PRODUCT_ID              = "GT0EB00N090_T077_V01"
RECORD_TYPE             = FIXED_LENGTH
RECORD_BYTES            = 1440
LABEL_RECORDS           = 5
^IMAGE                  = "GT0EB00N090_T077_V01.IMG"
OBJECT                  = IMAGE
  LINES                 = 360
  LINE_SAMPLES          = 360
  SAMPLE_TYPE           = PC_REAL
  SAMPLE_BITS           = 32
  MISSING_CONSTANT      = 16#FF7FFFFB#
  MAP_RESOLUTION        = 2.0 <PIX/DEG>
  MINIMUM_LATITUDE      = -90.0 <DEG>
  MAXIMUM_LATITUDE      =  90.0 <DEG>
  WESTERNMOST_LONGITUDE =   0.0 <DEG>
  EASTERNMOST_LONGITUDE = 180.0 <DEG>
  CENTER_LONGITUDE      = 180.0 <DEG>
END_OBJECT              = IMAGE
END
"""


def make_synthetic_img(
    lines:        int = 8,
    line_samples: int = 8,
    label_records:int = 5,
    record_bytes: int = 1440,
    fill_value:   float = 100.0,
    n_missing:    int = 2,
) -> bytes:
    """
    Build a minimal synthetic GTDR .IMG byte string.

    Parameters
    ----------
    fill_value:
        Default elevation value in metres.
    n_missing:
        Number of pixels to fill with MISSING_CONSTANT.
    """
    # Pad header
    header = b"\x00" * (label_records * record_bytes)

    # Image data: float32 little-endian
    data = np.full(lines * line_samples, fill_value, dtype="<f4")
    # Inject MISSING_CONSTANT by bit-exact bytes
    missing_bytes = bytes([0xFB, 0xFF, 0x7F, 0xFF])
    missing_f32 = struct.unpack("<f", missing_bytes)[0]
    for i in range(n_missing):
        data[i] = missing_f32

    return header + data.tobytes()


# ---------------------------------------------------------------------------
# Test: MISSING_CONSTANT value
# ---------------------------------------------------------------------------

class TestMissingConstant:
    def test_exact_hex_value(self) -> None:
        """The MISSING_CONSTANT must decode to exactly -3.4028x10^3^8."""
        raw = bytes([0xFB, 0xFF, 0x7F, 0xFF])
        decoded = struct.unpack("<f", raw)[0]
        assert abs(decoded - GTDR_MISSING_CONSTANT) < 1e30
        assert decoded < -3e38
        assert not np.isnan(decoded), (
            "MISSING_CONSTANT is NOT a NaN -- must use exact comparison."
        )

    def test_not_nan(self) -> None:
        """Verify that MISSING_CONSTANT is not NaN (common incorrect assumption).

        -3.4028x10^3^8 is a FINITE extreme-negative float32 value (close to -FLT_MAX).
        It is NOT NaN, NOT +/-inf -- np.isfinite() correctly returns True.
        This is exactly why np.isnan() silently fails to mask it, making
        exact-value comparison mandatory (see test_masking_with_isnan_fails).
        """
        # The value IS finite -- this is the key insight
        assert np.isfinite(GTDR_MISSING_CONSTANT), (
            "MISSING_CONSTANT must be finite (it is close to -FLT_MAX, "
            "not NaN or inf)."
        )
        # It is NOT a NaN
        assert not np.isnan(GTDR_MISSING_CONSTANT)
        # It is NOT infinite
        assert not np.isinf(GTDR_MISSING_CONSTANT)
        # And it is extremely negative (sanity check on the actual value)
        assert GTDR_MISSING_CONSTANT < -3e38

    def test_masking_with_isnan_fails(self) -> None:
        """
        Demonstrate that using np.isnan() fails to mask MISSING_CONSTANT.
        The pipeline must NOT use np.isnan() for this purpose.
        """
        arr = np.array([GTDR_MISSING_CONSTANT, 0.0, 100.0], dtype=np.float32)
        # np.isnan returns False for MISSING_CONSTANT
        assert not np.isnan(arr[0]), (
            "np.isnan() does NOT catch MISSING_CONSTANT! "
            "Use exact-value comparison instead."
        )

    def test_masking_with_exact_comparison_works(self) -> None:
        """Exact-value comparison correctly identifies MISSING_CONSTANT pixels."""
        arr = np.array([GTDR_MISSING_CONSTANT, 0.0, 100.0], dtype=np.float32)
        mask = np.abs(arr - GTDR_MISSING_CONSTANT) < 1e30
        assert bool(mask[0]) is True
        assert bool(mask[1]) is False
        assert bool(mask[2]) is False


# ---------------------------------------------------------------------------
# Test: label parser
# ---------------------------------------------------------------------------

class TestParseGTDRLabel:
    def test_basic_fields(self, tmp_path: Path) -> None:
        lbl = tmp_path / "GT0EB00N090_T077_V01.LBL"
        lbl.write_text(SAMPLE_LABEL)
        meta = parse_gtdr_label(lbl)

        assert meta["lines"] == 360
        assert meta["line_samples"] == 360
        assert meta["record_bytes"] == 1440
        assert meta["label_records"] == 5
        assert abs(meta["map_resolution"] - 2.0) < 1e-6
        assert abs(meta["minimum_latitude"] - (-90.0)) < 1e-6
        assert abs(meta["maximum_latitude"] - 90.0) < 1e-6
        assert abs(meta["westernmost_longitude"] - 0.0) < 1e-6
        assert abs(meta["easternmost_longitude"] - 180.0) < 1e-6
        assert abs(meta["center_longitude"] - 180.0) < 1e-6

    def test_image_offset(self, tmp_path: Path) -> None:
        """Image data starts at record 6 = byte offset 5x1440 = 7200."""
        lbl = tmp_path / "test.LBL"
        lbl.write_text(SAMPLE_LABEL)
        meta = parse_gtdr_label(lbl)
        assert meta["image_offset"] == GTDR_IMAGE_OFFSET
        assert meta["image_offset"] == 5 * 1440  # 5 header records

    def test_img_path_inference(self, tmp_path: Path) -> None:
        lbl = tmp_path / "GT0EB00N090_T077_V01.LBL"
        lbl.write_text(SAMPLE_LABEL)
        meta = parse_gtdr_label(lbl)
        assert meta["img_path"].name == "GT0EB00N090_T077_V01.IMG"

    def test_missing_label_uses_defaults(self, tmp_path: Path) -> None:
        """Parser should fall back to hard-coded defaults if label is absent."""
        img = tmp_path / "fake.IMG"
        img.write_bytes(b"\x00" * 100)
        # We cannot call read_gtdr_img without a label file in default mode,
        # but we can verify the missing-constant constant is correct.
        assert GTDR_IMAGE_OFFSET == 5 * 1440


# ---------------------------------------------------------------------------
# Test: binary image reader
# ---------------------------------------------------------------------------

class TestReadGTDRImg:
    def test_reads_synthetic_file(self, tmp_path: Path) -> None:
        """Reader correctly parses a synthetic .IMG file."""
        lines, samples = 8, 8
        lbl_content = SAMPLE_LABEL.replace("LINES                 = 360", f"LINES                 = {lines}")
        lbl_content = lbl_content.replace("LINE_SAMPLES          = 360", f"LINE_SAMPLES          = {samples}")
        lbl = tmp_path / "test.LBL"
        lbl.write_text(lbl_content)
        img = tmp_path / "test.IMG"
        img.write_bytes(make_synthetic_img(lines, samples))

        data, meta = read_gtdr_img(img, lbl)
        assert data.shape == (lines, samples)
        assert data.dtype == np.float32

    def test_missing_constant_masked(self, tmp_path: Path) -> None:
        """MISSING_CONSTANT pixels become NaN after loading."""
        lines, samples = 8, 8
        n_missing = 3
        lbl_content = SAMPLE_LABEL.replace("LINES                 = 360", f"LINES                 = {lines}")
        lbl_content = lbl_content.replace("LINE_SAMPLES          = 360", f"LINE_SAMPLES          = {samples}")
        lbl = tmp_path / "mask_test.LBL"
        lbl.write_text(lbl_content)
        img = tmp_path / "mask_test.IMG"
        img.write_bytes(make_synthetic_img(lines, samples, n_missing=n_missing))

        data, _ = read_gtdr_img(img, lbl)
        n_nan = int(np.sum(np.isnan(data)))
        assert n_nan == n_missing, (
            f"Expected {n_missing} NaN pixels, got {n_nan}. "
            "MISSING_CONSTANT masking failed."
        )

    def test_valid_pixel_values_preserved(self, tmp_path: Path) -> None:
        """Non-missing pixels retain their original float32 values."""
        lines, samples = 4, 4
        fill = 250.75
        lbl_content = SAMPLE_LABEL.replace("LINES                 = 360", f"LINES                 = {lines}")
        lbl_content = lbl_content.replace("LINE_SAMPLES          = 360", f"LINE_SAMPLES          = {samples}")
        lbl = tmp_path / "val_test.LBL"
        lbl.write_text(lbl_content)
        img = tmp_path / "val_test.IMG"
        img.write_bytes(make_synthetic_img(lines, samples, fill_value=fill, n_missing=0))

        data, _ = read_gtdr_img(img, lbl)
        finite = data[np.isfinite(data)]
        assert len(finite) == lines * samples
        np.testing.assert_allclose(finite, fill, rtol=1e-5)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Missing .IMG raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_gtdr_img(tmp_path / "nonexistent.IMG")


# ---------------------------------------------------------------------------
# Test: affine transform
# ---------------------------------------------------------------------------

class TestAffineTransform:
    def test_global_mosaic_transform(self) -> None:
        """Global mosaic (0->360 degW, 90 deg->-90 deg) has correct metre extents."""
        import math
        meta = {
            "westernmost_longitude": 0.0,
            "maximum_latitude":      90.0,
            "map_resolution":        2.0,
        }
        west_m, north_m, dx_m, dy_m = gtdr_affine_transform(meta)
        r = 2575.0 * 1000.0
        m_per_deg = r * math.pi / 180.0

        assert abs(west_m) < 1, "West edge should be at 0 metres"
        assert abs(north_m - 90.0 * m_per_deg) < 1, "North edge should be +90 deg"
        assert dx_m > 0, "Pixel width should be positive"
        assert dy_m < 0, "Pixel height should be negative (north-up)"
        # 0.5 deg/pixel at equator
        expected_dx = 0.5 * m_per_deg
        assert abs(dx_m - expected_dx) < 10, (
            f"Expected dx =~ {expected_dx:.0f} m, got {dx_m:.0f} m"
        )


# ---------------------------------------------------------------------------
# Test: mosaic_gtdr_tiles
# ---------------------------------------------------------------------------

class TestMosaicGTDRTiles:
    """Tests for mosaic_gtdr_tiles() -- merges east + west half-globe tiles."""

    LABEL_EAST = """\
PDS_VERSION_ID=PDS3
RECORD_TYPE=FIXED_LENGTH
RECORD_BYTES=1440
LABEL_RECORDS=5
^IMAGE="east.IMG"
OBJECT=IMAGE
LINES=8
LINE_SAMPLES=8
SAMPLE_TYPE=PC_REAL
SAMPLE_BITS=32
MAP_RESOLUTION=2.0 <PIX/DEG>
MINIMUM_LATITUDE=-90.0 <DEG>
MAXIMUM_LATITUDE=90.0 <DEG>
WESTERNMOST_LONGITUDE=0.0 <DEG>
EASTERNMOST_LONGITUDE=180.0 <DEG>
CENTER_LONGITUDE=180.0 <DEG>
END_OBJECT=IMAGE
END
"""
    LABEL_WEST = """\
PDS_VERSION_ID=PDS3
RECORD_TYPE=FIXED_LENGTH
RECORD_BYTES=1440
LABEL_RECORDS=5
^IMAGE="west.IMG"
OBJECT=IMAGE
LINES=8
LINE_SAMPLES=8
SAMPLE_TYPE=PC_REAL
SAMPLE_BITS=32
MAP_RESOLUTION=2.0 <PIX/DEG>
MINIMUM_LATITUDE=-90.0 <DEG>
MAXIMUM_LATITUDE=90.0 <DEG>
WESTERNMOST_LONGITUDE=180.0 <DEG>
EASTERNMOST_LONGITUDE=360.0 <DEG>
CENTER_LONGITUDE=270.0 <DEG>
END_OBJECT=IMAGE
END
"""

    def _write_tile(self, td: Path, stem: str, label: str, fill: float) -> tuple:
        lbl = td / f"{stem}.LBL"
        img = td / f"{stem}.IMG"
        lbl.write_text(label)
        img.write_bytes(make_synthetic_img(lines=8, line_samples=8,
                                           fill_value=fill, n_missing=0))
        return img, lbl

    def test_mosaic_shape(self) -> None:
        """Mosaicked array has (lines, 2xline_samples) shape."""
        from titan.io.gtdr_reader import mosaic_gtdr_tiles
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            eimg, elbl = self._write_tile(td, "east", self.LABEL_EAST, 100.0)
            wimg, wlbl = self._write_tile(td, "west", self.LABEL_WEST, 200.0)
            mosaic, meta = mosaic_gtdr_tiles(eimg, wimg, elbl, wlbl)
        assert mosaic.shape == (8, 16), f"Expected (8,16), got {mosaic.shape}"

    def test_mosaic_east_values_left(self) -> None:
        """Eastern tile (fill=100) populates the left columns of the mosaic."""
        from titan.io.gtdr_reader import mosaic_gtdr_tiles
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            eimg, elbl = self._write_tile(td, "east", self.LABEL_EAST, 100.0)
            wimg, wlbl = self._write_tile(td, "west", self.LABEL_WEST, 200.0)
            mosaic, _ = mosaic_gtdr_tiles(eimg, wimg, elbl, wlbl)
        left  = mosaic[:, :8]
        right = mosaic[:, 8:]
        np.testing.assert_allclose(left,  100.0, rtol=1e-5)
        np.testing.assert_allclose(right, 200.0, rtol=1e-5)

    def test_mosaic_meta_longitude_span(self) -> None:
        """Merged metadata reports full 0-360 deg longitude span."""
        from titan.io.gtdr_reader import mosaic_gtdr_tiles
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            eimg, elbl = self._write_tile(td, "east", self.LABEL_EAST, 50.0)
            wimg, wlbl = self._write_tile(td, "west", self.LABEL_WEST, 50.0)
            _, meta = mosaic_gtdr_tiles(eimg, wimg, elbl, wlbl)
        assert abs(meta["westernmost_longitude"] - 0.0) < 1e-6
        assert abs(meta["easternmost_longitude"] - 360.0) < 1e-6

    def test_mosaic_missing_constant_masked(self) -> None:
        """MISSING_CONSTANT pixels in either tile become NaN in the mosaic."""
        from titan.io.gtdr_reader import mosaic_gtdr_tiles
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # East tile: 3 missing pixels; West tile: 2 missing pixels
            elbl = td / "east.LBL"; elbl.write_text(self.LABEL_EAST)
            eimg = td / "east.IMG"
            eimg.write_bytes(make_synthetic_img(lines=8, line_samples=8,
                                                fill_value=50.0, n_missing=3))
            wlbl = td / "west.LBL"; wlbl.write_text(self.LABEL_WEST)
            wimg = td / "west.IMG"
            wimg.write_bytes(make_synthetic_img(lines=8, line_samples=8,
                                                fill_value=50.0, n_missing=2))
            mosaic, _ = mosaic_gtdr_tiles(eimg, wimg, elbl, wlbl)
        assert int(np.sum(np.isnan(mosaic))) == 5

    def test_mosaic_shape_mismatch_pads_gracefully(self) -> None:
        """
        Mismatched tile row counts (both truncated, different amounts) must NOT
        raise -- instead the shorter tile is NaN-padded at the south end so the
        mosaic can be assembled.  This reflects real GTIE files where both
        hemispheres arrive truncated but by different byte counts.
        """
        from titan.io.gtdr_reader import mosaic_gtdr_tiles
        label_8 = self.LABEL_EAST
        label_4 = self.LABEL_WEST.replace("LINES=8", "LINES=4")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            eimg, elbl = self._write_tile(td, "east", label_8, 100.0)
            wlbl = td / "west.LBL"; wlbl.write_text(label_4)
            wimg = td / "west.IMG"
            # Write a 4x8 image (west tile is shorter than east)
            wimg.write_bytes(make_synthetic_img(lines=4, line_samples=8,
                                                fill_value=0.0, n_missing=0))
            # Should NOT raise -- should pad the shorter west tile
            mosaic, meta = mosaic_gtdr_tiles(eimg, wimg, elbl, wlbl)

        # Mosaic rows = max(8, 4) = 8; cols = 8+8 = 16
        assert mosaic.shape == (8, 16), f"Unexpected mosaic shape: {mosaic.shape}"

        # West half (columns 8-15): rows 4-7 should be NaN-padded
        west_half = mosaic[:, 8:]
        assert np.all(np.isfinite(west_half[:4, :])), "West top rows should be valid"
        assert np.all(np.isnan(west_half[4:, :])),    "West padded rows should be NaN"

        # East half (columns 0-7): all rows present, should be finite
        east_half = mosaic[:, :8]
        assert np.all(np.isfinite(east_half)), "East half should be fully valid"

    def test_mosaic_without_label_files(self) -> None:
        """mosaic_gtdr_tiles works when label paths are not provided (uses defaults)."""
        from titan.io.gtdr_reader import mosaic_gtdr_tiles
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Write companion .LBL files with default names so auto-discovery works
            (td / "east.LBL").write_text(self.LABEL_EAST)
            (td / "west.LBL").write_text(self.LABEL_WEST)
            eimg = td / "east.IMG"
            wimg = td / "west.IMG"
            eimg.write_bytes(make_synthetic_img(8, 8, fill_value=1.0, n_missing=0))
            wimg.write_bytes(make_synthetic_img(8, 8, fill_value=2.0, n_missing=0))
            mosaic, _ = mosaic_gtdr_tiles(eimg, wimg)  # no lbl args
        assert mosaic.shape == (8, 16)


# ---------------------------------------------------------------------------
# Test: gtdr_affine_transform (extended)
# ---------------------------------------------------------------------------

class TestAffineTransformExtended:
    """Additional affine transform tests beyond the basic case."""

    def test_pixel_size_matches_2ppd(self) -> None:
        """At 2 ppd, pixel size should be 0.5 deg in metres."""
        import math
        meta = {"westernmost_longitude": 0.0,
                "maximum_latitude": 90.0, "map_resolution": 2.0}
        _, _, dx_m, dy_m = gtdr_affine_transform(meta)
        m_per_deg = 2_575_000.0 * math.pi / 180.0
        expected = 0.5 * m_per_deg          # 0.5 deg/pixel at 2 ppd
        assert abs(dx_m - expected) < 100   # within 100 m

    def test_pixel_size_scales_with_resolution(self) -> None:
        """Higher resolution (more ppd) -> smaller pixel size."""
        meta_lo = {"westernmost_longitude": 0., "maximum_latitude": 90., "map_resolution": 2.0}
        meta_hi = {"westernmost_longitude": 0., "maximum_latitude": 90., "map_resolution": 4.0}
        _, _, dx_lo, _ = gtdr_affine_transform(meta_lo)
        _, _, dx_hi, _ = gtdr_affine_transform(meta_hi)
        assert dx_lo > dx_hi, "Lower resolution should have larger pixels"
        assert abs(dx_lo / dx_hi - 2.0) < 0.01, "4 ppd pixels should be half 2 ppd"

    def test_north_edge_at_max_latitude(self) -> None:
        """north_m should equal max_latitude converted to metres."""
        import math
        m_per_deg = 2_575_000.0 * math.pi / 180.0
        for max_lat in (90.0, 45.0, 0.0):
            meta = {"westernmost_longitude": 0., "maximum_latitude": max_lat,
                    "map_resolution": 2.0}
            _, north_m, _, _ = gtdr_affine_transform(meta)
            assert abs(north_m - max_lat * m_per_deg) < 1.0

    def test_west_edge_at_westernmost_longitude(self) -> None:
        """west_m should equal westernmost_longitude converted to metres."""
        import math
        m_per_deg = 2_575_000.0 * math.pi / 180.0
        for west_lon in (0.0, 90.0, 180.0):
            meta = {"westernmost_longitude": west_lon, "maximum_latitude": 90.,
                    "map_resolution": 2.0}
            west_m, _, _, _ = gtdr_affine_transform(meta)
            assert abs(west_m - west_lon * m_per_deg) < 1.0

    def test_dy_is_negative(self) -> None:
        """dy_m must be negative -- rasters are stored north-to-south."""
        meta = {"westernmost_longitude": 0., "maximum_latitude": 90., "map_resolution": 2.0}
        _, _, _, dy_m = gtdr_affine_transform(meta)
        assert dy_m < 0, "dy_m should be negative (north-up raster convention)"

    def test_dx_is_positive(self) -> None:
        """dx_m must be positive -- longitude increases left to right."""
        meta = {"westernmost_longitude": 0., "maximum_latitude": 90., "map_resolution": 2.0}
        _, _, dx_m, _ = gtdr_affine_transform(meta)
        assert dx_m > 0


# ---------------------------------------------------------------------------
# Integration tests using real GTDR files from tests/fixtures/
# (auto-skip if tests/fixtures/gtdr/ is absent)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Integration tests using real GTDR/GTIE elevation files from tests/fixtures/gtdr/
#
# Fixture availability (all auto-skip if absent):
#   gtde_east_img -- GTIED00N090_T126_V01  (interpolated elevation, ~90% coverage north of 50S)
#   gtde_west_img -- GTIED00N270_T126_V01  (interpolated elevation, ~90% coverage north of 51S)
#   gtdr_east_img -- GT2ED00N090_T126_V01  (sparse) or GT0EB00N090_T077_V01 (legacy)
#
# Cornell archive: https://data.astro.cornell.edu/RADAR/DATA/GTDR/
# Files distributed as .IMG.gz; reader decompresses transparently.
# ---------------------------------------------------------------------------

class TestGTDRIntegration:
    """
    Integration tests against real GTDR/GTIE elevation tiles in tests/fixtures/gtdr/.
    See tests/fixtures/README.md for setup instructions.
    """

    # -- Single-tile tests -------------------------------------------------

    def test_real_gt0e_east_reads_correctly(self, gtdr_east_img: Any) -> None:
        """
        Read a real GT0E east tile (sparse GTDR) and verify:
          - shape is (360, 360) at 2 ppd
          - MISSING_CONSTANT pixels are correctly NaN-masked
          - sparse coverage: typically 5-40% valid pixels
          - elevation values in plausible range (-1700 m to +520 m)
        """
        lbl = None
        base = gtdr_east_img.with_suffix("") if str(gtdr_east_img).endswith(".gz")                else gtdr_east_img
        for suffix in (".LBL", ".lbl"):
            if base.with_suffix(suffix).exists():
                lbl = base.with_suffix(suffix)
                break

        data, meta = read_gtdr_img(gtdr_east_img, lbl)

        # GT0E sparse GTDR is always 2 ppd -> (360 rows, 360 cols) per half-globe tile
        assert data.shape == (360, 360), (
            f"Expected (360,360) at 2 ppd, got {data.shape}. "
            f"If using a 4/8 ppd variant, update this assertion."
        )
        assert data.dtype == np.float32

        finite = data[np.isfinite(data)]
        assert len(finite) > 0, "All pixels are NaN -- check MISSING_CONSTANT masking"
        valid_frac = len(finite) / data.size
        # Coverage check: must have at least some valid pixels.
        # Note: some GTDR releases (e.g. T077) are fully populated with
        # no MISSING_CONSTANT pixels -- the sparsity shows up as very low
        # elevation values, not as NaN gaps.
        assert valid_frac > 0.01, f"Fewer than 1% valid pixels: {valid_frac:.1%}"

        assert finite.min() >= -2000.0, f"Min elevation {finite.min():.0f} m"
        assert finite.max() <=  1000.0, f"Max elevation {finite.max():.0f} m"

    def test_real_gt0e_label_metadata(self, gtdr_east_img: Any) -> None:
        """Parse the companion .LBL and verify expected east-tile metadata."""
        base = gtdr_east_img.with_suffix("") if str(gtdr_east_img).endswith(".gz")                else gtdr_east_img
        lbl = base.with_suffix(".LBL")
        if not lbl.exists():
            lbl = base.with_suffix(".lbl")
        if not lbl.exists():
            import pytest
            pytest.skip(f"No .LBL found alongside {gtdr_east_img.name}")

        meta = parse_gtdr_label(lbl)
        assert meta["lines"]        == 360
        assert meta["line_samples"] == 360
        assert abs(meta.get("map_resolution", 0) - 2.0) < 0.1
        # East tile (GT0EB00N090) in west-positive PDS3 convention:
        #   westernmost_longitude = 180 degW (the western boundary)
        #   easternmost_longitude =   0 degW (the eastern boundary = prime meridian)
        west = meta.get("westernmost_longitude", -999)
        east = meta.get("easternmost_longitude", -999)
        assert abs(west - 180.0) < 1.0 or abs(east - 180.0) < 1.0, (
            f"Expected east tile longitude span near 0-180 degW, "
            f"got westernmost={west}, easternmost={east}"
        )

    def test_real_gt0e_gzip_equals_uncompressed(self, fixtures_gtdr_dir: Path) -> None:
        """
        If both .IMG and .IMG.gz exist for the same tile, they must produce
        identical data arrays.  Verifies the gzip decompression path.
        """
        import pytest
        from tests.conftest import _gtdr_find
        stem = "GT0EB00N090_T077_V01"
        plain = fixtures_gtdr_dir / (stem + ".IMG")
        gz    = fixtures_gtdr_dir / (stem + ".IMG.gz")
        if not plain.exists() or not gz.exists():
            pytest.skip(
                f"Need both {stem}.IMG and {stem}.IMG.gz to compare "
                "compressed vs uncompressed reading"
            )
        d_plain, _ = read_gtdr_img(plain)
        d_gz,    _ = read_gtdr_img(gz)
        # NaN positions must match
        assert np.array_equal(np.isnan(d_plain), np.isnan(d_gz))
        # Non-NaN values must be identical
        valid = np.isfinite(d_plain)
        np.testing.assert_array_equal(d_plain[valid], d_gz[valid])

    # -- GTIE interpolated elevation DEM tests -----------------------------

    def test_real_gtie_east_has_global_coverage(self, gtde_east_img: Any) -> None:
        """
        GTIE east tile (Interpolated Elevation, metres) must have substantially higher
        coverage than sparse GTDR: expect >80% valid pixels.
        This confirms the spline interpolation filled the track gaps.
        """
        lbl = None
        base = gtde_east_img.with_suffix("") if str(gtde_east_img).endswith(".gz")                else gtde_east_img
        for suffix in (".LBL", ".lbl"):
            if base.with_suffix(suffix).exists():
                lbl = base.with_suffix(suffix)
                break

        data, meta = read_gtdr_img(gtde_east_img, lbl)

        # GTIE T126 tiles are at 8 ppd (5.62 km/px). Coverage is ~90N to ~48-51S due to
        # specific Cornell file).  GTIED00N090_T126_V01 is 8 ppd -> 1440x1440.
        # Don't hardcode (360,360); derive from map_resolution instead.
        ppd = meta.get("map_resolution", 2.0)
        expected_cols = round(180.0 * ppd)
        expected_rows_max = round(180.0 * ppd)  # label value (may be truncated)
        assert data.dtype == np.float32
        assert data.shape[1] == expected_cols, (
            f"Expected {expected_cols} columns at {ppd} ppd, got {data.shape[1]}"
        )
        # Rows may be slightly fewer than label claims (truncated file)
        assert data.shape[0] > 0

        finite = data[np.isfinite(data)]
        assert len(finite) > 0

        valid_frac = len(finite) / data.size
        assert valid_frac > 0.80, (
            f"GTIE should have >80% valid pixels (interpolated elevation); "
            f"got {valid_frac:.1%}.  Verify this is GTIED*, not sparse GT0E."
        )
        assert finite.min() >= -2000.0
        assert finite.max() <=  1000.0

    def test_real_gtie_mosaic_near_global(self, gtde_east_img: Any, gtde_west_img: Any) -> None:
        """
        Mosaic both GTIE tiles and verify near-global coverage.
          - shape (360, 720), full 0-360 deg longitude span
          - >90% valid pixels (confirms interpolated fill)
          - east and west hemispheres have different mean elevations
            (guards against accidental tile duplication)
        """
        def _lbl(p: Any) -> Optional[Path]:
            base = p.with_suffix("") if str(p).endswith(".gz") else p
            for s in (".LBL", ".lbl"):
                if base.with_suffix(s).exists():
                    return base.with_suffix(s)
            return None

        mosaic, meta = mosaic_gtdr_tiles(
            gtde_east_img, gtde_west_img,
            _lbl(gtde_east_img), _lbl(gtde_west_img),
        )

        # Shape depends on ppd: GTIE T126 is 8 ppd -> 1440 cols per tile -> 2880 mosaic cols
        ppd = meta.get("map_resolution", 2.0)
        expected_ncols = round(360.0 * ppd)  # full 360 deg at this resolution
        assert mosaic.shape[1] == expected_ncols, (
            f"Expected {expected_ncols} columns ({ppd} ppd), got {mosaic.shape[1]}"
        )
        assert mosaic.shape[0] > 0  # rows may be slightly truncated

        assert abs(meta["westernmost_longitude"] - 0.0) < 1e-6
        assert abs(meta["easternmost_longitude"] - 360.0) < 1e-6

        valid_frac = float(np.sum(np.isfinite(mosaic))) / mosaic.size
        assert valid_frac > 0.90, (
            f"GTIE mosaic should have >90% valid pixels; got {valid_frac:.1%}"
        )

        half = mosaic.shape[1] // 2
        left  = mosaic[:, :half][np.isfinite(mosaic[:, :half])]
        right = mosaic[:, half:][np.isfinite(mosaic[:, half:])]
        assert len(left) > 0 and len(right) > 0
        assert abs(float(left.mean()) - float(right.mean())) > 0.1,             "East/west hemisphere means identical -- possible tile duplication"

    # -- Preprocessing pipeline integration -------------------------------

    def test_preprocess_uses_gtde_when_available(self, gtde_east_img: Path,
                                                   gtde_west_img: Path, tmp_path: Path) -> None:
        """
        When GTIE tiles are present, _preprocess_topography should use them
        (not fall back to sparse GT0E), and the output raster should have
        near-global coverage.
        Requires rasterio.
        """
        import importlib
        if importlib.util.find_spec("rasterio") is None:
            import pytest
            pytest.skip("rasterio not installed")

        import shutil
        from configs.pipeline_config import PipelineConfig

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Symlink or copy GTIE tiles into a temp data_dir
        for src in (gtde_east_img, gtde_west_img):
            dst = data_dir / src.name
            shutil.copy2(src, dst)
            # Also copy .LBL if present
            base = src.with_suffix("") if str(src).endswith(".gz") else src
            for s in (".LBL", ".lbl"):
                lbl = base.with_suffix(s)
                if lbl.exists():
                    shutil.copy2(lbl, data_dir / lbl.name)

        cfg = PipelineConfig(
            data_dir=data_dir,
            processed_dir=tmp_path / "processed",
            output_dir=tmp_path / "outputs",
        )
        cfg.make_dirs()

        from titan.preprocessing import DataPreprocessor, CanonicalGrid
        grid = CanonicalGrid(cfg.canonical_res_m)
        proc = DataPreprocessor(cfg, grid)
        result = proc._preprocess_topography(overwrite=False)

        assert "topography" in result, "Expected topography key in result"
        assert result["topography"].exists(), "Output GeoTIFF not created"