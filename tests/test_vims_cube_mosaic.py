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
tests/test_vims_cube_mosaic.py
==============================
Unit tests for the VIMS individual-cube mosaic builder.

Tests cover:
  1. Wavelength <-> band index conversion
  2. ISIS3 PVL label parsing (with synthetic labels)
  3. ISIS3 binary cube reading (with synthetic cubes written in-memory)
  4. VIMSWindowMosaicker.select_cube_ids() with synthetic parquet
  5. VIMSWindowMosaicker.build_mosaic() with mocked downloader
  6. organic_abundance feature uses vims_5um_2um_ratio when present
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Wavelength / band conversion
# ---------------------------------------------------------------------------

class TestWavelengthConversion:

    def test_bands_for_5um_returns_nonempty(self) -> None:
        from titan.io.vims_cube_mosaic import bands_for_wavelength
        bands = bands_for_wavelength(5.0)
        assert len(bands) > 0, "5.0 umm should have bands in VIMS-IR range"

    def test_bands_for_5um_are_near_end_of_range(self) -> None:
        from titan.io.vims_cube_mosaic import bands_for_wavelength, VIMS_IR_N_BANDS
        bands = bands_for_wavelength(5.0)
        # All bands should be in the top 10% of the 256-band array
        assert min(bands) > int(VIMS_IR_N_BANDS * 0.85), (
            "5.0 umm should be near the end of the 256-band VIMS-IR array"
        )

    def test_bands_for_2um_are_in_middle(self) -> None:
        from titan.io.vims_cube_mosaic import bands_for_wavelength, VIMS_IR_N_BANDS
        bands = bands_for_wavelength(2.03)
        mid = VIMS_IR_N_BANDS // 2
        # 2.03 umm is about 27% through the VIMS-IR range (0.88->5.11)
        assert min(bands) < mid, "2.03 umm should be in first half of band array"

    def test_out_of_range_returns_empty(self) -> None:
        from titan.io.vims_cube_mosaic import bands_for_wavelength
        assert bands_for_wavelength(0.1) == [], "Below VIMS-IR range"
        assert bands_for_wavelength(6.0) == [], "Above VIMS-IR range"

    def test_band_count_scales_with_half_width(self) -> None:
        from titan.io.vims_cube_mosaic import bands_for_wavelength
        narrow = bands_for_wavelength(2.03, half_width_um=0.02)
        wide   = bands_for_wavelength(2.03, half_width_um=0.15)
        assert len(wide) > len(narrow), (
            "Wider window should include more bands"
        )

    def test_wavelength_array_monotone(self) -> None:
        from titan.io.vims_cube_mosaic import VIMS_IR_WAVELENGTHS
        assert np.all(np.diff(VIMS_IR_WAVELENGTHS) > 0), (
            "Wavelengths must be strictly increasing"
        )
        assert VIMS_IR_WAVELENGTHS[0]   < 0.95, "Band 0 should be near 0.88 umm"
        assert VIMS_IR_WAVELENGTHS[-1]  > 5.0,  "Band 255 should be near 5.11 umm"


# ---------------------------------------------------------------------------
# 2. ISIS3 PVL label parsing
# ---------------------------------------------------------------------------

def _make_isis3_label(
    start_byte: int = 1025,
    samples:    int = 16,
    lines:      int = 8,
    bands:      int = 256,
    pix_type:   str = "Real",
    byte_order: str = "Lsb",
    base:       float = 0.0,
    multiplier: float = 1.0,
    fmt:        str = "Bsq",
) -> str:
    """Generate a minimal ISIS3 PVL label for testing."""
    return (
        f"Object = IsisCube\n"
        f"  Object = Core\n"
        f"    StartByte = {start_byte}\n"
        f"    Format = {fmt}\n"
        f"    Group = Dimensions\n"
        f"      Samples = {samples}\n"
        f"      Lines   = {lines}\n"
        f"      Bands   = {bands}\n"
        f"    End_Group = Dimensions\n"
        f"    Group = Pixels\n"
        f"      Type      = {pix_type}\n"
        f"      ByteOrder = {byte_order}\n"
        f"      Base      = {base}\n"
        f"      Multiplier = {multiplier}\n"
        f"    End_Group = Pixels\n"
        f"  End_Object = Core\n"
        f"End_Object = IsisCube\n"
        f"End\n"
    )


def _write_synthetic_isis3_cube(
    path: Path,
    data: np.ndarray,     # shape (bands, lines, samples)
    pix_type:   str = "Real",
    byte_order: str = "Lsb",
) -> None:
    """Write a synthetic ISIS3 BSQ cube file for testing."""
    # Pad label to a fixed size
    pad_size = 1024
    label = _make_isis3_label(
        start_byte=pad_size + 1,
        samples=data.shape[2],
        lines=data.shape[1],
        bands=data.shape[0],
        pix_type=pix_type,
        byte_order=byte_order,
    )
    label_bytes = label.encode("ascii")
    assert len(label_bytes) < pad_size, "Synthetic label exceeds pad_size"
    # Pad to pad_size bytes
    padding = b"\x00" * (pad_size - len(label_bytes))

    if pix_type == "Real":
        dtype = np.dtype("<f4") if byte_order == "Lsb" else np.dtype(">f4")
    else:  # SignedWord
        dtype = np.dtype("<i2") if byte_order == "Lsb" else np.dtype(">i2")

    with open(path, "wb") as fh:
        fh.write(label_bytes)
        fh.write(padding)
        fh.write(data.astype(dtype).tobytes())


class TestISIS3LabelParsing:

    def test_parse_basic_real_label(self) -> None:
        from titan.io.vims_cube_mosaic import _parse_isis3_label
        label = _make_isis3_label(start_byte=1025, samples=64, lines=32, bands=256)
        meta = _parse_isis3_label(label)
        assert meta["start_byte"] == 1024    # 1-indexed -> 0-indexed
        assert meta["samples"]    == 64
        assert meta["lines"]      == 32
        assert meta["bands"]      == 256
        assert meta["dtype"]      == "<f4"   # Real, Lsb
        assert meta["format"]     == "bsq"
        assert meta["base"]       == 0.0
        assert meta["multiplier"] == 1.0

    def test_parse_signed_word(self) -> None:
        from titan.io.vims_cube_mosaic import _parse_isis3_label
        label = _make_isis3_label(
            pix_type="SignedWord", byte_order="Msb",
            multiplier=0.0001, base=0.0
        )
        meta = _parse_isis3_label(label)
        assert meta["dtype"]      == ">i2"
        assert meta["multiplier"] == pytest.approx(0.0001)

    def test_parse_raises_on_missing_fields(self) -> None:
        from titan.io.vims_cube_mosaic import _parse_isis3_label, ISIS3LabelError
        bad_label = "Object = IsisCube\nEnd_Object\nEnd\n"
        with pytest.raises(ISIS3LabelError):
            _parse_isis3_label(bad_label)

    def test_read_label_text_finds_end(self, tmp_path: Path) -> None:
        from titan.io.vims_cube_mosaic import _read_isis3_label_text
        label = _make_isis3_label()
        cub = tmp_path / "test.cub"
        cub.write_bytes(label.encode("ascii") + b"\x00" * 4096)
        result = _read_isis3_label_text(cub)
        assert result.strip().endswith("End")

    def test_read_label_text_raises_without_end(self, tmp_path: Path) -> None:
        from titan.io.vims_cube_mosaic import (
            _read_isis3_label_text, ISIS3LabelError,
        )
        cub = tmp_path / "no_end.cub"
        cub.write_bytes(b"Object = IsisCube\n" + b"\x00" * 100)
        with pytest.raises(ISIS3LabelError):
            _read_isis3_label_text(cub)


# ---------------------------------------------------------------------------
# 3. ISIS3 binary cube reading
# ---------------------------------------------------------------------------

class TestReadISIS3Cube:

    def test_read_float32_bsq(self, tmp_path: Path) -> None:
        from titan.io.vims_cube_mosaic import read_isis3_cube
        B, L, S = 4, 6, 8
        data = np.arange(B * L * S, dtype=np.float32).reshape(B, L, S) * 0.001
        cub = tmp_path / "cal.cub"
        _write_synthetic_isis3_cube(cub, data)
        result = read_isis3_cube(cub)
        assert result.shape == (B, L, S)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_read_selected_bands(self, tmp_path: Path) -> None:
        from titan.io.vims_cube_mosaic import read_isis3_cube
        B, L, S = 10, 4, 4
        data = np.random.default_rng(42).uniform(0, 1, (B, L, S)).astype(np.float32)
        cub = tmp_path / "bands.cub"
        _write_synthetic_isis3_cube(cub, data)
        result = read_isis3_cube(cub, band_indices=[0, 2, 4])
        assert result.shape == (3, L, S)
        np.testing.assert_allclose(result[0], data[0], atol=1e-5)
        np.testing.assert_allclose(result[1], data[2], atol=1e-5)
        np.testing.assert_allclose(result[2], data[4], atol=1e-5)

    def test_read_applies_scaling(self, tmp_path: Path) -> None:
        from titan.io.vims_cube_mosaic import read_isis3_cube, _parse_isis3_label
        B, L, S = 2, 3, 4
        # Store as SignedWord with multiplier=0.0001
        data_real  = np.full((B, L, S), 0.0300, dtype=np.float32)
        data_int16 = (data_real / 0.0001).astype(np.int16)
        pad_size = 1024
        label = _make_isis3_label(
            start_byte=pad_size + 1, samples=S, lines=L, bands=B,
            pix_type="SignedWord", multiplier=0.0001,
        )
        label_bytes = label.encode("ascii")
        cub = tmp_path / "scaled.cub"
        with open(cub, "wb") as fh:
            fh.write(label_bytes)
            fh.write(b"\x00" * (pad_size - len(label_bytes)))
            fh.write(data_int16.tobytes())
        result = read_isis3_cube(cub)
        assert result.shape == (B, L, S)
        np.testing.assert_allclose(result, data_real, atol=1e-4)

    def test_raises_if_file_missing(self) -> None:
        from titan.io.vims_cube_mosaic import read_isis3_cube
        with pytest.raises(FileNotFoundError):
            read_isis3_cube(Path("/no/such/file.cub"))


# ---------------------------------------------------------------------------
# 4. VIMSWindowMosaicker.select_cube_ids()
# ---------------------------------------------------------------------------

def _make_vims_parquet(tmp_path: Path, n_cubes: int = 5) -> Path:
    """Create a minimal synthetic VIMS parquet for testing."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n_cubes):
        cube_id = f"100000000{i}_1"
        for _ in range(20):
            rows.append({
                "id":    cube_id,
                "lat":   rng.uniform(-80, 80),
                "lon":   rng.uniform(0, 360),
                "res":   rng.uniform(1.0, 30.0),
                "flyby": f"T{i:02d}",
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "vims_test.parquet"
    df.to_parquet(path)
    return path


class TestSelectCubeIds:

    def test_returns_cube_ids(self, tmp_path: Path) -> None:
        from titan.io.vims_cube_mosaic import VIMSWindowMosaicker
        parquet = _make_vims_parquet(tmp_path, n_cubes=5)
        mosaic = VIMSWindowMosaicker(tmp_path / "cubes", max_resolution_km=25.0)
        ids = mosaic.select_cube_ids(parquet)
        assert len(ids) > 0

    def test_max_cubes_limits_results(self, tmp_path: Path) -> None:
        from titan.io.vims_cube_mosaic import VIMSWindowMosaicker
        parquet = _make_vims_parquet(tmp_path, n_cubes=10)
        mosaic = VIMSWindowMosaicker(
            tmp_path / "cubes", max_resolution_km=50.0, max_cubes=3
        )
        ids = mosaic.select_cube_ids(parquet)
        assert len(ids) <= 3

    def test_resolution_filter_rejects_coarse_cubes(self, tmp_path: Path) -> None:
        """A very tight resolution filter should reject all or most cubes."""
        from titan.io.vims_cube_mosaic import VIMSWindowMosaicker
        parquet = _make_vims_parquet(tmp_path, n_cubes=5)
        mosaic = VIMSWindowMosaicker(
            tmp_path / "cubes", max_resolution_km=0.001
        )
        ids = mosaic.select_cube_ids(parquet)
        # With 0.001 km/px threshold, synthetic cubes (res 1-30) all fail
        assert len(ids) == 0

    def test_sorted_best_resolution_first(self, tmp_path: Path) -> None:
        """Cube IDs should be sorted by ascending median resolution."""
        from titan.io.vims_cube_mosaic import VIMSWindowMosaicker
        import pandas as pd
        # Create parquet with two cubes of known resolutions
        df = pd.DataFrame([
            {"id": "cube_good", "lat": 0.0, "lon": 10.0, "res": 5.0,  "flyby": "T01"},
            {"id": "cube_good", "lat": 5.0, "lon": 20.0, "res": 6.0,  "flyby": "T01"},
            {"id": "cube_bad",  "lat": 0.0, "lon": 10.0, "res": 25.0, "flyby": "T02"},
            {"id": "cube_bad",  "lat": 5.0, "lon": 20.0, "res": 28.0, "flyby": "T02"},
        ])
        parquet = tmp_path / "sorted.parquet"
        df.to_parquet(parquet)
        mosaic = VIMSWindowMosaicker(tmp_path / "cubes", max_resolution_km=50.0)
        ids = mosaic.select_cube_ids(parquet)
        assert ids[0] == "cube_good", "Best-resolution cube should be first"


# ---------------------------------------------------------------------------
# 5. VIMSWindowMosaicker.build_mosaic() with mocked downloads
# ---------------------------------------------------------------------------

def _write_nav_cube(path: Path, nrows: int, ncols: int, lats, lons_wp) -> None:
    """Write a synthetic N*_ir.cub with 6 geometry bands."""
    B = 6
    data = np.zeros((B, nrows, ncols), dtype=np.float32)
    data[0, :, :] = lats.reshape(nrows, ncols).astype(np.float32)
    # Convert west-positive to east-positive for the nav cube
    data[1, :, :] = ((-lons_wp) % 360).reshape(nrows, ncols).astype(np.float32)
    data[3, :, :] = 20.0  # emission angle 20 deg (good geometry)
    data[5, :, :] = 5.0   # resolution 5 km/px
    _write_synthetic_isis3_cube(path, data)


def _write_cal_cube(path: Path, nrows: int, ncols: int, fill_value: float = 0.05) -> None:
    """Write a synthetic C*_ir.cub with 256 spectral bands."""
    B, L, S = 256, nrows, ncols
    data = np.full((B, L, S), fill_value, dtype=np.float32)
    # Make 5 umm band region slightly brighter to test extraction
    from titan.io.vims_cube_mosaic import bands_for_wavelength
    for b in bands_for_wavelength(5.0):
        data[b, :, :] = fill_value * 2.0
    _write_synthetic_isis3_cube(path, data)


class TestBuildMosaic:

    def test_all_nan_when_no_cubes_pass_filter(self, tmp_path: Path) -> None:
        from titan.io.vims_cube_mosaic import VIMSWindowMosaicker
        parquet = _make_vims_parquet(tmp_path, n_cubes=3)
        mosaic = VIMSWindowMosaicker(
            tmp_path / "cubes",
            max_resolution_km=0.0001,  # nothing passes
            max_cubes=10,
        )
        result = mosaic.build_mosaic(parquet, target_um=5.0, nrows=8, ncols=16)
        assert result.shape == (8, 16)
        assert np.all(np.isnan(result)), "No cubes selected -> all NaN"

    def test_mosaic_shape_matches_grid(self, tmp_path: Path) -> None:
        """build_mosaic returns array matching requested nrows/ncols."""
        from titan.io.vims_cube_mosaic import VIMSWindowMosaicker
        parquet = _make_vims_parquet(tmp_path, n_cubes=5)
        mosaic = VIMSWindowMosaicker(
            tmp_path / "cubes", max_resolution_km=50.0, max_cubes=0,
        )
        result = mosaic.build_mosaic(parquet, target_um=5.0, nrows=18, ncols=36)
        assert result.shape == (18, 36)

    def test_mosaic_with_synthetic_cubes(self, tmp_path: Path) -> None:
        """
        Build a real mosaic from synthetic cube files.
        Verifies that pixels are projected correctly onto the grid.
        """
        from titan.io.vims_cube_mosaic import VIMSWindowMosaicker

        cube_dir = tmp_path / "cubes"
        cube_dir.mkdir()

        # One synthetic cube covering a 10x10 deg patch centred at (lat=60, lon_wp=90)
        NR, NC = 4, 4
        lats_flat    = np.full(NR * NC, 60.0, dtype=np.float32)
        lons_wp_flat = np.full(NR * NC, 90.0, dtype=np.float32)
        # Scatter them a little
        rng = np.random.default_rng(1)
        lats_flat    += rng.uniform(-3, 3, NR * NC).astype(np.float32)
        lons_wp_flat += rng.uniform(-3, 3, NR * NC).astype(np.float32)

        cube_id = "1234567890_1"
        cal_path = cube_dir / f"C{cube_id}_ir.cub"
        nav_path = cube_dir / f"N{cube_id}_ir.cub"
        _write_cal_cube(cal_path, NR, NC, fill_value=0.04)
        _write_nav_cube(nav_path, NR, NC, lats_flat, lons_wp_flat)

        # Parquet with a single cube
        df = pd.DataFrame([{
            "id":    cube_id,
            "lat":   float(lats_flat.mean()),
            "lon":   float(lons_wp_flat.mean()),
            "res":   5.0,
            "flyby": "T01",
        }])
        parquet = tmp_path / "test.parquet"
        df.to_parquet(parquet)

        mosaic = VIMSWindowMosaicker(
            cube_dir, max_resolution_km=50.0, max_cubes=5
        )
        result = mosaic.build_mosaic(parquet, target_um=5.0, nrows=36, ncols=72)

        assert result.shape == (36, 72)
        # The region around (lat=60, lon_wp=90) should be non-NaN
        # Row for lat=60: (90-60)/180 * 36 = 6
        # Col for lon=90: 90/360 * 72 = 18
        row_c = int((90.0 - 60.0) / 180.0 * 36)
        col_c = int(90.0 / 360.0 * 72)
        # Check a small neighbourhood
        patch = result[max(0,row_c-2):row_c+3, max(0,col_c-2):col_c+3]
        assert np.any(np.isfinite(patch)), (
            f"Pixels near (lat=60, lon=90) should be non-NaN. "
            f"row_c={row_c}, col_c={col_c}"
        )
        # Check the filled values are in reasonable I/F range
        valid = patch[np.isfinite(patch)]
        assert np.all(valid > 0), "I/F values should be positive"
        assert np.all(valid < 10.0), "I/F values should be < 10.0"


# ---------------------------------------------------------------------------
# 6. organic_abundance uses vims_5um_2um_ratio
# ---------------------------------------------------------------------------

class TestOrganic5um:
    """Verify features.py _organic_abundance blends vims_5um_2um_ratio."""

    def _make_stack(
        self,
        nrows: int = 8,
        ncols: int = 16,
        with_5um: bool = False,
        with_vims: bool = False,
    ):
        """Build a minimal xarray Dataset for organic_abundance testing."""
        import xarray as xr
        lats = np.linspace(80, -80, nrows)
        lons = np.linspace(0, 360, ncols)
        rng  = np.random.default_rng(42)
        dvs  = {}
        if with_vims:
            dvs["vims_mosaic"] = xr.DataArray(
                rng.uniform(.2, .9, (nrows, ncols)).astype(np.float32),
                dims=["lat", "lon"], coords={"lat": lats, "lon": lons},
            )
        if with_5um:
            # Ratio 1.5 = organic-rich mid-value
            dvs["vims_5um_2um_ratio"] = xr.DataArray(
                np.full((nrows, ncols), 1.5, dtype=np.float32),
                dims=["lat", "lon"], coords={"lat": lats, "lon": lons},
            )
        return xr.Dataset(dvs)

    def test_5um_ratio_used_when_no_seignovert(self) -> None:
        """When Seignovert mosaic absent, 5um ratio becomes the spectral proxy."""
        from titan.features import FeatureExtractor
        from titan.preprocessing import CanonicalGrid
        grid  = CanonicalGrid(pixel_size_m=500_000)
        calc  = FeatureExtractor(grid)
        stack = self._make_stack(with_vims=False, with_5um=True)
        nan   = np.full((8, 16), np.nan, dtype=np.float32)
        result = calc._organic_abundance(stack, nan)
        assert result.shape == (8, 16)
        assert np.any(np.isfinite(result)), (
            "5um ratio alone should produce some finite output"
        )

    def test_5um_blended_with_seignovert(self) -> None:
        """When both are available, result should differ from Seignovert-only."""
        from titan.features import FeatureExtractor
        from titan.preprocessing import CanonicalGrid
        grid  = CanonicalGrid(pixel_size_m=500_000)
        calc  = FeatureExtractor(grid)

        stack_no5  = self._make_stack(with_vims=True,  with_5um=False)
        stack_with5= self._make_stack(with_vims=True,  with_5um=True)
        nan = np.full((8, 16), np.nan, dtype=np.float32)

        r_no5   = calc._organic_abundance(stack_no5,   nan)
        r_with5 = calc._organic_abundance(stack_with5, nan)

        # Results should differ because the 5um ratio alters the blend
        assert not np.allclose(r_no5, r_with5, equal_nan=True), (
            "Adding 5um ratio should change the organic_abundance output"
        )

    def test_no_5um_no_vims_falls_back_gracefully(self) -> None:
        """Without VIMS or 5um, feature falls back to geo-only or NaN."""
        from titan.features import FeatureExtractor
        from titan.preprocessing import CanonicalGrid
        grid  = CanonicalGrid(pixel_size_m=500_000)
        calc  = FeatureExtractor(grid)
        stack = self._make_stack(with_vims=False, with_5um=False)
        nan   = np.full((8, 16), np.nan, dtype=np.float32)
        result = calc._organic_abundance(stack, nan)
        # All NaN is acceptable (no geomorphology either)
        assert result.shape == (8, 16)
