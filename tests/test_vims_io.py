"""
tests/test_vims_io.py
======================
Unit tests for all classes and functions in titan/io/vims_reader.py
that are NOT URL helpers (those are in test_vims_urls.py).

Coverage:
  VIMSFootprintIndex
    load()                    — requires pyarrow (skipif absent)
    df property               — in-memory
    cubes_covering_region()   — in-memory
    get_download_urls()       — in-memory
    coverage_map()            — in-memory  (north-up, normalisation, shape)
    best_resolution_map()     — in-memory  (min-res, NaN coverage, north-up)
    flyby_count_map()         — in-memory  (distinct flybys, north-up)
    summary()                 — in-memory

  VIMSCubeDownloader
    download_cube()           — mocked HTTP (unittest.mock)
    download_batch()          — mocked HTTP
    download_preview()        — mocked HTTP
    _fetch()                  — mocked HTTP

  read_navigation_cube()      — requires rasterio (skipif absent)

Synthetic parquet files and mock HTTP responses are used throughout.
No real Cassini data or network access is required.
"""

from __future__ import annotations

from typing import Any
import importlib
import io
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.io.vims_reader import (
    VIMSFootprintIndex,
    VIMSCubeDownloader,
    read_navigation_cube,
    cube_url_calibrated,
    cube_url_navigation,
    cube_url_raw,
    VIMS_COLUMNS,
)

_HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None
_HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None
_NEED_PYARROW  = pytest.mark.skipif(not _HAS_PYARROW,  reason="pyarrow not installed")
_NEED_RASTERIO = pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = list(VIMS_COLUMNS.keys())


def make_footprint_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """
    Return a minimal in-memory DataFrame matching the VIMS parquet schema.

    Parameters
    ----------
    n:
        Number of rows.
    seed:
        NumPy random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    flybys = ["TA", "TB", "T3", "T4", "T5"]
    return pd.DataFrame({
        "id":        [f"{1477222875 + i}_1" for i in range(n)],
        "flyby":     [flybys[i % len(flybys)] for i in range(n)],
        "obs_start": pd.to_datetime(["2004-10-26"] * n),
        "obs_end":   pd.to_datetime(["2004-10-26"] * n),
        "altitude":  rng.uniform(500.0, 2000.0, n).astype("float32"),
        "lon":       rng.uniform(0.0, 360.0, n).astype("float32"),
        "lat":       rng.uniform(-90.0, 90.0, n).astype("float32"),
        "res":       rng.uniform(1.0, 100.0, n).astype("float32"),
    })


def make_index(df: pd.DataFrame | None = None) -> VIMSFootprintIndex:
    """Return a VIMSFootprintIndex pre-loaded with the given DataFrame."""
    idx = VIMSFootprintIndex.__new__(VIMSFootprintIndex)
    idx.parquet_path = Path("synthetic.parquet")
    idx.max_resolution_km = None
    idx._df = (df if df is not None else make_footprint_df()).reset_index(drop=True)
    return idx


# ===========================================================================
# 1.  VIMSFootprintIndex.load()  (requires pyarrow)
# ===========================================================================

class TestVIMSFootprintIndexLoad:

    @_NEED_PYARROW
    def test_load_from_parquet_file(self, tmp_path: Path) -> None:
        """load() reads a real parquet file and populates _df."""
        df = make_footprint_df(n=50)
        pq = tmp_path / "test.parquet"
        df.to_parquet(pq)

        idx = VIMSFootprintIndex(pq)
        idx.load()
        assert idx._df is not None
        assert len(idx._df) == 50

    @_NEED_PYARROW
    def test_load_all_required_columns_present(self, tmp_path: Path) -> None:
        """After load(), all VIMS_COLUMNS keys are present."""
        df = make_footprint_df(n=10)
        pq = tmp_path / "test.parquet"
        df.to_parquet(pq)

        idx = VIMSFootprintIndex(pq)
        idx.load()
        for col in REQUIRED_COLUMNS:
            assert col in idx._df.columns, f"Missing column: {col}"

    @_NEED_PYARROW
    def test_load_missing_columns_raises(self, tmp_path: Path) -> None:
        """load() raises ValueError when required columns are absent."""
        pq = tmp_path / "bad.parquet"
        pd.DataFrame({"x": [1, 2]}).to_parquet(pq)
        idx = VIMSFootprintIndex(pq)
        with pytest.raises(ValueError, match="missing expected columns"):
            idx.load()

    @_NEED_PYARROW
    def test_load_max_resolution_filter(self, tmp_path: Path) -> None:
        """max_resolution_km filters out high-res footprints on load."""
        df = make_footprint_df(n=100, seed=42)
        # Force half the rows above the threshold
        df.loc[:49, "res"] = 200.0
        df.loc[50:, "res"] = 5.0
        pq = tmp_path / "test.parquet"
        df.to_parquet(pq)

        idx = VIMSFootprintIndex(pq, max_resolution_km=10.0)
        idx.load()
        assert len(idx._df) == 50
        assert idx._df["res"].max() <= 10.0

    @_NEED_PYARROW
    def test_df_property_triggers_load(self, tmp_path: Path) -> None:
        """Accessing .df on an unloaded index triggers load()."""
        df = make_footprint_df(n=5)
        pq = tmp_path / "test.parquet"
        df.to_parquet(pq)
        idx = VIMSFootprintIndex(pq)
        assert idx._df is None
        _ = idx.df
        assert idx._df is not None


# ===========================================================================
# 2.  VIMSFootprintIndex.df property  (in-memory)
# ===========================================================================

class TestVIMSFootprintIndexDf:

    def test_df_returns_dataframe(self) -> None:
        idx = make_index()
        assert isinstance(idx.df, pd.DataFrame)

    def test_df_has_required_columns(self) -> None:
        idx = make_index()
        for col in REQUIRED_COLUMNS:
            assert col in idx.df.columns

    def test_df_length_matches(self) -> None:
        df = make_footprint_df(n=7)
        idx = make_index(df)
        assert len(idx.df) == 7


# ===========================================================================
# 3.  VIMSFootprintIndex.cubes_covering_region()
# ===========================================================================

class TestCubesCoveringRegion:

    def test_returns_dataframe(self) -> None:
        idx = make_index()
        result = idx.cubes_covering_region(0, 360, -90, 90)
        assert isinstance(result, pd.DataFrame)

    def test_full_globe_returns_all(self) -> None:
        df = make_footprint_df(n=30)
        idx = make_index(df)
        result = idx.cubes_covering_region(0, 360, -90, 90)
        assert len(result) == len(df)

    def test_empty_region_returns_empty(self) -> None:
        """A region that contains no footprints returns an empty DataFrame."""
        df = make_footprint_df(n=20, seed=1)
        df["lon"] = 200.0   # all footprints at lon=200
        df["lat"] = 0.0
        idx = make_index(df)
        result = idx.cubes_covering_region(0, 100, -90, 90)
        assert len(result) == 0

    def test_spatial_filter_works(self) -> None:
        """Only footprints within the box are returned."""
        df = make_footprint_df(n=10, seed=2)
        df["lon"] = [50.0] * 5 + [250.0] * 5   # two groups
        df["lat"] = 0.0
        idx = make_index(df)
        result = idx.cubes_covering_region(0, 100, -10, 10)
        assert len(result) == 5

    def test_sorted_by_resolution(self) -> None:
        """Results are sorted ascending by resolution (best first)."""
        df = make_footprint_df(n=5)
        df["lon"] = 10.0
        df["lat"] = 5.0
        df["res"] = [50.0, 10.0, 5.0, 20.0, 1.0]
        idx = make_index(df)
        result = idx.cubes_covering_region(0, 50, 0, 10)
        assert list(result["res"]) == sorted(result["res"])

    def test_deduplicates_by_id(self) -> None:
        """Duplicate cube IDs are removed."""
        df = make_footprint_df(n=4)
        df["id"] = ["cube_1", "cube_1", "cube_2", "cube_2"]
        df["lon"] = 10.0; df["lat"] = 5.0
        idx = make_index(df)
        result = idx.cubes_covering_region(0, 50, 0, 10)
        assert len(result) == 2

    def test_max_resolution_filter(self) -> None:
        """max_resolution_km limits results."""
        df = make_footprint_df(n=10)
        df["lon"] = 10.0; df["lat"] = 5.0
        df["res"] = [float(i * 10) for i in range(1, 11)]
        idx = make_index(df)
        result = idx.cubes_covering_region(0, 50, 0, 10, max_resolution_km=50.0)
        assert all(result["res"] <= 50.0)


# ===========================================================================
# 4.  VIMSFootprintIndex.get_download_urls()
# ===========================================================================

class TestGetDownloadUrls:

    def test_returns_dict_with_four_keys(self) -> None:
        idx = make_index()
        urls = idx.get_download_urls("1477222875_1")
        assert set(urls.keys()) == {"raw", "label", "calibrated", "navigation"}

    def test_raw_url_matches_helper(self) -> None:
        idx = make_index()
        urls = idx.get_download_urls("1477222875_1")
        assert urls["raw"] == cube_url_raw("1477222875_1")

    def test_calibrated_url_matches_helper(self) -> None:
        idx = make_index()
        urls = idx.get_download_urls("1477222875_1")
        assert urls["calibrated"] == cube_url_calibrated("1477222875_1")

    def test_navigation_url_matches_helper(self) -> None:
        idx = make_index()
        urls = idx.get_download_urls("1477222875_1")
        assert urls["navigation"] == cube_url_navigation("1477222875_1")

    def test_all_urls_are_https(self) -> None:
        idx = make_index()
        urls = idx.get_download_urls("9999999999_1")
        for key, url in urls.items():
            assert url.startswith("https://"), f"{key} URL not HTTPS: {url}"

    def test_cube_id_appears_in_all_urls(self) -> None:
        cube_id = "1234567890_2"
        idx = make_index()
        urls = idx.get_download_urls(cube_id)
        for key, url in urls.items():
            assert cube_id in url, f"{key} URL missing cube_id: {url}"


# ===========================================================================
# 5.  VIMSFootprintIndex.coverage_map()
# ===========================================================================

class TestCoverageMap:

    def test_shape_matches_requested(self) -> None:
        idx = make_index(make_footprint_df(n=50))
        out = idx.coverage_map(nrows=18, ncols=36)
        assert out.shape == (18, 36)

    def test_dtype_float32(self) -> None:
        idx = make_index()
        out = idx.coverage_map(18, 36)
        assert out.dtype == np.float32

    def test_values_in_0_to_1(self) -> None:
        idx = make_index(make_footprint_df(n=100))
        out = idx.coverage_map(18, 36)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    def test_max_is_1_when_coverage_present(self) -> None:
        """Normalisation: the most-observed pixel is exactly 1.0."""
        df = make_footprint_df(n=50)
        # Stack 20 footprints in one cell
        df.loc[:19, "lon"] = 10.0
        df.loc[:19, "lat"] = 70.0
        idx = make_index(df)
        out = idx.coverage_map(18, 36)
        assert abs(out.max() - 1.0) < 1e-5

    def test_north_up_row0_is_north(self) -> None:
        """Row 0 corresponds to the north (+90°). Footprint at lat=+80 → top rows."""
        df = make_footprint_df(n=1)
        df["lon"] = 10.0
        df["lat"] = 80.0
        idx = make_index(df)
        out = idx.coverage_map(18, 36)
        northern_rows = out[:3, :]    # rows 0-2: lat 90°→60°
        southern_rows = out[12:, :]   # rows 12-17: lat ~20°→-90°
        assert northern_rows.sum() > 0, "Footprint at +80° should land in top rows"
        assert southern_rows.sum() == 0

    def test_south_footprint_in_bottom_rows(self) -> None:
        """Footprint at lat=−80° ends up in the bottom rows."""
        df = make_footprint_df(n=1)
        df["lon"] = 180.0
        df["lat"] = -80.0
        idx = make_index(df)
        out = idx.coverage_map(18, 36)
        assert out[15:, :].sum() > 0, "lat=−80 should be in bottom rows"
        assert out[:3, :].sum() == 0

    def test_empty_df_all_zeros(self) -> None:
        idx = make_index(make_footprint_df(n=0))
        out = idx.coverage_map(18, 36)
        assert out.max() == 0.0

    def test_custom_lon_lat_range(self) -> None:
        """Custom range restricts which footprints are counted."""
        df = make_footprint_df(n=10)
        df["lon"] = [10.0] * 5 + [200.0] * 5
        df["lat"] = 0.0
        idx = make_index(df)
        out = idx.coverage_map(18, 36, lon_range=(0.0, 100.0), lat_range=(-90.0, 90.0))
        # Only the 5 footprints at lon=10 should appear; max should be 1.0
        assert abs(out.max() - 1.0) < 1e-5


# ===========================================================================
# 6.  VIMSFootprintIndex.best_resolution_map()
# ===========================================================================

class TestBestResolutionMap:

    def test_shape_matches_requested(self) -> None:
        idx = make_index(make_footprint_df(n=20))
        out = idx.best_resolution_map(18, 36)
        assert out.shape == (18, 36)

    def test_dtype_float32(self) -> None:
        idx = make_index()
        out = idx.best_resolution_map(18, 36)
        assert out.dtype == np.float32

    def test_uncovered_pixels_are_nan(self) -> None:
        """Pixels with no observations → NaN."""
        df = make_footprint_df(n=1)
        df["lon"] = 10.0; df["lat"] = 10.0; df["res"] = 5.0
        idx = make_index(df)
        out = idx.best_resolution_map(18, 36)
        nan_fraction = float(np.sum(np.isnan(out))) / out.size
        assert nan_fraction > 0.9, "Most pixels should be NaN (single footprint)"

    def test_picks_minimum_resolution(self) -> None:
        """When multiple footprints overlap, the smallest resolution wins."""
        df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "flyby": ["TA", "TB", "TA"],
            "obs_start": pd.to_datetime(["2004-10-26"] * 3),
            "obs_end":   pd.to_datetime(["2004-10-26"] * 3),
            "altitude":  [1000.0] * 3,
            "lon":       [50.0, 50.0, 50.0],
            "lat":       [30.0, 30.0, 30.0],
            "res":       [20.0, 5.0, 10.0],
        })
        idx = make_index(df)
        out = idx.best_resolution_map(18, 36)
        finite = out[np.isfinite(out)]
        assert len(finite) > 0
        assert abs(finite.min() - 5.0) < 1.0, \
            f"Expected min resolution 5.0, got {finite.min()}"

    def test_north_up_orientation(self) -> None:
        """Footprint at lat=+75° lands in the top rows (row 0 = +90°)."""
        df = pd.DataFrame({
            "id": ["a"], "flyby": ["TA"],
            "obs_start": pd.to_datetime(["2004-10-26"]),
            "obs_end": pd.to_datetime(["2004-10-26"]),
            "altitude": [1000.0], "lon": [100.0], "lat": [75.0], "res": [3.0],
        })
        idx = make_index(df)
        out = idx.best_resolution_map(18, 36)
        assert np.any(np.isfinite(out[:4, :])), \
            "lat=+75° footprint should appear in rows 0-3 (north)"
        assert not np.any(np.isfinite(out[14:, :])), \
            "No footprints near south"

    def test_no_inf_in_output(self) -> None:
        """Output should contain no +inf values (only finite values or NaN)."""
        idx = make_index(make_footprint_df(n=50))
        out = idx.best_resolution_map(18, 36)
        assert not np.any(np.isposinf(out))
        assert not np.any(np.isneginf(out))

    def test_empty_df_all_nan(self) -> None:
        idx = make_index(make_footprint_df(n=0))
        out = idx.best_resolution_map(18, 36)
        assert np.all(np.isnan(out))


# ===========================================================================
# 7.  VIMSFootprintIndex.flyby_count_map()
# ===========================================================================

class TestFlybyCountMap:

    def test_shape_matches_requested(self) -> None:
        idx = make_index(make_footprint_df(n=20))
        out = idx.flyby_count_map(18, 36)
        assert out.shape == (18, 36)

    def test_dtype_int16(self) -> None:
        idx = make_index()
        out = idx.flyby_count_map(18, 36)
        assert out.dtype == np.int16

    def test_uncovered_pixels_are_zero(self) -> None:
        df = make_footprint_df(n=1)
        df["lon"] = 10.0; df["lat"] = 10.0
        idx = make_index(df)
        out = idx.flyby_count_map(18, 36)
        zero_fraction = float(np.sum(out == 0)) / out.size
        assert zero_fraction > 0.9

    def test_counts_distinct_flybys(self) -> None:
        """Two rows from same flyby in one cell → count = 1, not 2."""
        df = pd.DataFrame({
            "id": ["a", "b"], "flyby": ["TA", "TA"],
            "obs_start": pd.to_datetime(["2004-10-26"] * 2),
            "obs_end":   pd.to_datetime(["2004-10-26"] * 2),
            "altitude":  [1000.0, 1000.0],
            "lon":       [50.0, 50.0],
            "lat":       [30.0, 30.0],
            "res":       [5.0, 5.0],
        })
        idx = make_index(df)
        out = idx.flyby_count_map(18, 36)
        assert out.max() == 1, "Two same-flyby rows should count as 1 flyby"

    def test_counts_two_distinct_flybys(self) -> None:
        df = pd.DataFrame({
            "id": ["a", "b"], "flyby": ["TA", "TB"],
            "obs_start": pd.to_datetime(["2004-10-26"] * 2),
            "obs_end":   pd.to_datetime(["2004-10-26"] * 2),
            "altitude":  [1000.0, 900.0],
            "lon":       [50.0, 50.0],
            "lat":       [30.0, 30.0],
            "res":       [5.0, 3.0],
        })
        idx = make_index(df)
        out = idx.flyby_count_map(18, 36)
        assert out.max() >= 2, "Two different flybys should give count ≥ 2"

    def test_north_up_orientation(self) -> None:
        """Footprint at lat=+80° lands in northern rows."""
        df = pd.DataFrame({
            "id": ["a"], "flyby": ["TA"],
            "obs_start": pd.to_datetime(["2004-10-26"]),
            "obs_end": pd.to_datetime(["2004-10-26"]),
            "altitude": [1000.0], "lon": [60.0], "lat": [80.0], "res": [3.0],
        })
        idx = make_index(df)
        out = idx.flyby_count_map(18, 36)
        assert out[:3, :].max() >= 1, "lat=+80 should appear in rows 0-2"
        assert out[10:, :].max() == 0, "No footprints near south"

    def test_empty_df_all_zeros(self) -> None:
        idx = make_index(make_footprint_df(n=0))
        out = idx.flyby_count_map(18, 36)
        assert out.max() == 0


# ===========================================================================
# 8.  VIMSFootprintIndex.summary()
# ===========================================================================

class TestSummary:

    def test_returns_string(self) -> None:
        idx = make_index(make_footprint_df(n=20))
        assert isinstance(idx.summary(), str)

    def test_contains_footprint_count(self) -> None:
        df = make_footprint_df(n=20)
        idx = make_index(df)
        summary = idx.summary()
        assert "20" in summary

    def test_contains_flyby_count(self) -> None:
        df = make_footprint_df(n=20)   # 5 distinct flybys cycling
        idx = make_index(df)
        assert "5" in idx.summary()

    def test_non_empty(self) -> None:
        idx = make_index()
        assert len(idx.summary()) > 10


# ===========================================================================
# 9.  VIMSCubeDownloader (mocked HTTP)
# ===========================================================================

def _make_mock_response(content: bytes = b"FAKE_DATA", status: int = 200,
                        content_length: int | None = None) -> Any:
    """Return a mock requests.Response that streams the given content."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {}
    if content_length is not None:
        resp.headers["Content-Length"] = str(content_length)
    resp.raise_for_status = MagicMock()
    resp.iter_content = MagicMock(
        return_value=iter([content[i:i+64] for i in range(0, len(content), 64)])
    )
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestVIMSCubeDownloaderFetch:
    """Tests for VIMSCubeDownloader._fetch() — the raw HTTP layer."""

    def test_fetch_writes_file(self, tmp_path: Path) -> None:
        """_fetch writes content to dest."""
        downloader = VIMSCubeDownloader(tmp_path)
        dest = tmp_path / "test.cub"
        mock_resp = _make_mock_response(b"CUBE_DATA_12345")
        with patch("requests.get", return_value=mock_resp):
            downloader._fetch("https://vims.univ-nantes.fr/cube/test.cub", dest)
        assert dest.exists()
        assert dest.read_bytes() == b"CUBE_DATA_12345"

    def test_fetch_sends_user_agent_header(self, tmp_path: Path) -> None:
        """_fetch includes a browser User-Agent header."""
        downloader = VIMSCubeDownloader(tmp_path)
        dest = tmp_path / "out.cub"
        mock_resp = _make_mock_response(b"x")
        with patch("requests.get", return_value=mock_resp) as mock_get:
            downloader._fetch("https://example.com/file", dest)
        call_kwargs = mock_get.call_args[1]
        assert "headers" in call_kwargs
        ua = call_kwargs["headers"].get("User-Agent", "")
        assert "Mozilla" in ua, f"User-Agent header missing Mozilla: {ua}"

    def test_fetch_retries_on_failure(self, tmp_path: Path) -> None:
        """_fetch retries up to max_retries times before raising."""
        downloader = VIMSCubeDownloader(tmp_path, max_retries=3)
        dest = tmp_path / "out.cub"
        with patch("requests.get", side_effect=ConnectionError("refused")) as mock_get:
            with pytest.raises(ConnectionError):
                downloader._fetch("https://example.com/file", dest)
        assert mock_get.call_count == 3

    def test_fetch_cleans_up_partial_file_on_failure(self, tmp_path: Path) -> None:
        """If download fails, the .partial temp file is removed."""
        downloader = VIMSCubeDownloader(tmp_path, max_retries=1)
        dest = tmp_path / "out.cub"
        with patch("requests.get", side_effect=IOError("io error")):
            with pytest.raises(IOError):
                downloader._fetch("https://example.com/file", dest)
        partial = dest.with_suffix(".cub.partial")
        assert not partial.exists(), ".partial file should be cleaned up on failure"


class TestVIMSCubeDownloaderDownloadCube:

    def test_download_calibrated_creates_file(self, tmp_path: Path) -> None:
        """download_cube('calibrated') writes the calibrated .cub file."""
        downloader = VIMSCubeDownloader(tmp_path)
        cube_id = "1477222875_1"
        mock_resp = _make_mock_response(b"CALIBRATED_DATA")
        with patch("requests.get", return_value=mock_resp):
            result = downloader.download_cube(cube_id, download=("calibrated",))
        assert "calibrated" in result
        assert result["calibrated"].name == f"C{cube_id}_ir.cub"
        assert result["calibrated"].exists()

    def test_download_navigation_creates_file(self, tmp_path: Path) -> None:
        """download_cube('navigation') writes the navigation .cub file."""
        downloader = VIMSCubeDownloader(tmp_path)
        cube_id = "1477222875_1"
        mock_resp = _make_mock_response(b"NAV_DATA")
        with patch("requests.get", return_value=mock_resp):
            result = downloader.download_cube(cube_id, download=("navigation",))
        assert "navigation" in result
        assert result["navigation"].name == f"N{cube_id}_ir.cub"

    def test_download_raw_creates_qub_file(self, tmp_path: Path) -> None:
        """download_cube('raw') writes the .qub file."""
        downloader = VIMSCubeDownloader(tmp_path)
        cube_id = "1477222875_1"
        mock_resp = _make_mock_response(b"RAW_QUB_DATA")
        with patch("requests.get", return_value=mock_resp):
            result = downloader.download_cube(cube_id, download=("raw",))
        assert "raw" in result
        assert result["raw"].name == f"v{cube_id}.qub"

    def test_skip_existing_without_overwrite(self, tmp_path: Path) -> None:
        """Existing file is not re-downloaded when overwrite=False."""
        downloader = VIMSCubeDownloader(tmp_path)
        cube_id = "1477222875_1"
        dest = tmp_path / f"C{cube_id}_ir.cub"
        dest.write_bytes(b"EXISTING")
        with patch("requests.get") as mock_get:
            result = downloader.download_cube(cube_id, download=("calibrated",),
                                              overwrite=False)
        mock_get.assert_not_called()
        assert result["calibrated"] == dest

    def test_overwrite_re_downloads(self, tmp_path: Path) -> None:
        """overwrite=True re-downloads even if file exists."""
        downloader = VIMSCubeDownloader(tmp_path)
        cube_id = "1477222875_1"
        dest = tmp_path / f"C{cube_id}_ir.cub"
        dest.write_bytes(b"OLD_DATA")
        mock_resp = _make_mock_response(b"NEW_DATA")
        with patch("requests.get", return_value=mock_resp):
            result = downloader.download_cube(cube_id, download=("calibrated",),
                                              overwrite=True)
        assert result["calibrated"].read_bytes() == b"NEW_DATA"

    def test_unknown_type_skipped(self, tmp_path: Path) -> None:
        """An unrecognised file type is skipped without error."""
        downloader = VIMSCubeDownloader(tmp_path)
        with patch("requests.get"):
            result = downloader.download_cube("1477222875_1",
                                              download=("nonexistent_type",))
        assert "nonexistent_type" not in result

    def test_returns_dict(self, tmp_path: Path) -> None:
        """Return value is a dict mapping file type to Path."""
        downloader = VIMSCubeDownloader(tmp_path)
        mock_resp = _make_mock_response(b"data")
        with patch("requests.get", return_value=mock_resp):
            result = downloader.download_cube("1477222875_1",
                                              download=("calibrated", "navigation"))
        assert isinstance(result, dict)
        assert set(result.keys()) == {"calibrated", "navigation"}


class TestVIMSCubeDownloaderBatch:

    def test_batch_downloads_all_cubes(self, tmp_path: Path) -> None:
        """download_batch returns results for every requested cube_id."""
        downloader = VIMSCubeDownloader(tmp_path)
        cube_ids = ["111_1", "222_1", "333_1"]
        mock_resp = _make_mock_response(b"data")
        with patch("requests.get", return_value=mock_resp):
            results = downloader.download_batch(cube_ids, download=("calibrated",))
        assert set(results.keys()) == set(cube_ids)

    def test_batch_max_cubes_limits_downloads(self, tmp_path: Path) -> None:
        """max_cubes truncates the list."""
        downloader = VIMSCubeDownloader(tmp_path)
        cube_ids = [f"{i}_1" for i in range(10)]
        mock_resp = _make_mock_response(b"data")
        with patch("requests.get", return_value=mock_resp):
            results = downloader.download_batch(cube_ids, download=("calibrated",),
                                                max_cubes=3)
        assert len(results) == 3


class TestVIMSCubeDownloaderPreview:

    def test_download_preview_creates_jpg(self, tmp_path: Path) -> None:
        """download_preview writes a .jpg file."""
        downloader = VIMSCubeDownloader(tmp_path)
        mock_resp = _make_mock_response(b"\xff\xd8\xff" + b"FAKE_JPEG")
        with patch("requests.get", return_value=mock_resp):
            path = downloader.download_preview(
                "1477222875_1", "00ATI", band_combo="surface_rgb"
            )
        assert path is not None
        assert path.suffix == ".jpg"
        assert path.exists()

    def test_download_preview_skip_existing(self, tmp_path: Path) -> None:
        """Existing preview is not re-downloaded."""
        downloader = VIMSCubeDownloader(tmp_path)
        existing = tmp_path / "previews" / "1477222875_1_surface_rgb.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"EXISTING_JPEG")
        with patch("requests.get") as mock_get:
            path = downloader.download_preview(
                "1477222875_1", "00ATI", overwrite=False
            )
        mock_get.assert_not_called()
        assert path == existing

    def test_download_preview_returns_none_on_error(self, tmp_path: Path) -> None:
        """Returns None if download fails."""
        downloader = VIMSCubeDownloader(tmp_path, max_retries=1)
        with patch("requests.get", side_effect=IOError("network error")):
            result = downloader.download_preview("bad_id", "FLYBY")
        assert result is None


# ===========================================================================
# 10.  read_navigation_cube  (requires rasterio)
# ===========================================================================

def _make_synthetic_nav_cube(path: Path, nrows: int = 4, ncols: int = 8) -> Any:
    """
    Write a synthetic ISIS3-compatible GeoTIFF with 6 bands (navigation cube layout).

    Band 1: Latitude       (-90 to +90, varies by row)
    Band 2: Longitude E    (-180 to +180)
    Band 3: Incidence      (constant 30°)
    Band 4: Emission       (constant 10°)
    Band 5: Phase          (constant 40°)
    Band 6: Resolution     (constant 5.0 km/px)
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    transform = from_bounds(0, -90, 360, 90, ncols, nrows)
    crs = CRS.from_proj4("+proj=longlat +a=2575000 +b=2575000 +no_defs")

    # Build 6 bands (float32)
    lats  = np.linspace(90, -90, nrows)[:, np.newaxis] * np.ones((nrows, ncols))
    lons_e = np.linspace(-180, 180, ncols)[np.newaxis, :] * np.ones((nrows, ncols))

    bands = np.stack([
        lats.astype(np.float32),                          # band 1: lat
        lons_e.astype(np.float32),                        # band 2: lon east-positive
        np.full((nrows, ncols), 30.0, dtype=np.float32),  # band 3: incidence
        np.full((nrows, ncols), 10.0, dtype=np.float32),  # band 4: emission
        np.full((nrows, ncols), 40.0, dtype=np.float32),  # band 5: phase
        np.full((nrows, ncols),  5.0, dtype=np.float32),  # band 6: resolution
    ])  # shape (6, nrows, ncols)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        dtype="float32",
        count=6,
        width=ncols,
        height=nrows,
        crs=crs,
        transform=transform,
    ) as dst:
        for b in range(6):
            dst.write(bands[b], b + 1)


class TestReadNavigationCube:

    @_NEED_RASTERIO
    def test_returns_dict_with_required_keys(self, tmp_path: Path) -> None:
        """Return value has the expected keys."""
        nav = tmp_path / "N1234_ir.cub"
        _make_synthetic_nav_cube(nav)
        result = read_navigation_cube(nav)
        for key in ("lat", "lon_east", "lon_west", "incidence",
                    "emission", "phase", "resolution"):
            assert key in result, f"Missing key: {key}"

    @_NEED_RASTERIO
    def test_all_arrays_float32(self, tmp_path: Path) -> None:
        """All returned arrays are float32."""
        nav = tmp_path / "N1234_ir.cub"
        _make_synthetic_nav_cube(nav)
        result = read_navigation_cube(nav)
        for key, arr in result.items():
            assert arr.dtype == np.float32, f"{key}: expected float32, got {arr.dtype}"

    @_NEED_RASTERIO
    def test_all_arrays_same_shape(self, tmp_path: Path) -> None:
        """All arrays have the same (nrows, ncols) shape."""
        nav = tmp_path / "N1234_ir.cub"
        _make_synthetic_nav_cube(nav, nrows=4, ncols=8)
        result = read_navigation_cube(nav)
        shapes = {k: v.shape for k, v in result.items()}
        assert len(set(shapes.values())) == 1, f"Inconsistent shapes: {shapes}"

    @_NEED_RASTERIO
    def test_lat_in_range(self, tmp_path: Path) -> None:
        """Latitude values are in [−90, +90]."""
        nav = tmp_path / "N1234_ir.cub"
        _make_synthetic_nav_cube(nav)
        result = read_navigation_cube(nav)
        lat = result["lat"]
        finite = lat[np.isfinite(lat)]
        if len(finite) > 0:
            assert finite.min() >= -90.0
            assert finite.max() <= 90.0

    @_NEED_RASTERIO
    def test_lon_west_in_0_to_360(self, tmp_path: Path) -> None:
        """West-positive longitudes are in [0, 360]."""
        nav = tmp_path / "N1234_ir.cub"
        _make_synthetic_nav_cube(nav)
        result = read_navigation_cube(nav)
        lw = result["lon_west"]
        finite = lw[np.isfinite(lw)]
        if len(finite) > 0:
            assert finite.min() >= 0.0
            assert finite.max() <= 360.0

    @_NEED_RASTERIO
    def test_lon_west_is_neg_lon_east_mod_360(self, tmp_path: Path) -> None:
        """lon_west = (−lon_east) % 360 for all finite pixels."""
        nav = tmp_path / "N1234_ir.cub"
        _make_synthetic_nav_cube(nav)
        result = read_navigation_cube(nav)
        le, lw = result["lon_east"], result["lon_west"]
        valid = np.isfinite(le) & np.isfinite(lw)
        expected = (-le[valid]) % 360.0
        np.testing.assert_allclose(lw[valid], expected.astype(np.float32),
                                   atol=1e-4)

    @_NEED_RASTERIO
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError or similar when file does not exist."""
        with pytest.raises(Exception):
            read_navigation_cube(tmp_path / "nonexistent.cub")


# ===========================================================================
# 11.  Integration tests using real VIMS parquet from tests/fixtures/
#      (auto-skip when tests/fixtures/vims/ is absent or empty)
# ===========================================================================

class TestVIMSParquetIntegration:
    """
    Integration tests against a real VIMS footprint parquet file.

    These tests run against either the full catalogue (~227 MB, ~5.4M rows)
    or the 1,000-row development sample (43 KB) — both are accepted.
    Skipped automatically when no parquet is found in tests/fixtures/vims/.
    """

    @_NEED_PYARROW
    def test_load_real_parquet(self, vims_parquet_path: Path) -> None:
        """
        Load the real parquet and verify schema, types, and value ranges.
        """
        idx = VIMSFootprintIndex(vims_parquet_path)
        idx.load()
        df = idx.df

        # Schema
        for col in VIMS_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

        # Lon in [0, 360]
        assert df["lon"].min() >= 0.0
        assert df["lon"].max() <= 360.0

        # Lat in [-90, 90]
        assert df["lat"].min() >= -90.0
        assert df["lat"].max() <= 90.0

        # Resolution positive
        assert df["res"].min() > 0.0

        # At least one row
        assert len(df) >= 1

    @_NEED_PYARROW
    def test_real_parquet_has_titan_flybys(self, vims_parquet_path: Path) -> None:
        """
        The parquet should contain at least flyby TA (the first Titan flyby).
        """
        idx = VIMSFootprintIndex(vims_parquet_path)
        flybys = set(idx.df["flyby"].unique())
        assert "TA" in flybys, \
            f"Expected flyby TA in data; found: {sorted(flybys)[:10]}"

    @_NEED_PYARROW
    def test_real_coverage_map_global(self, vims_parquet_path: Path) -> None:
        """
        Build a low-resolution coverage map from the real parquet and verify:
          - correct shape
          - values in [0, 1]
          - at least some non-zero coverage
          - north-up orientation: lat=+70° coverage in top rows
        """
        idx = VIMSFootprintIndex(vims_parquet_path)
        out = idx.coverage_map(nrows=18, ncols=36)

        assert out.shape == (18, 36)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6
        assert out.max() > 0.0, "Coverage map is all zeros — check parquet content"

        # North polar region (rows 0-3, lat >60°) should have high coverage
        # because VIMS observed Titan's north polar lakes extensively
        north_coverage = out[:3, :].mean()
        south_coverage = out[15:, :].mean()
        # North pole has more observations than south for most of the mission
        # (this may not hold for a 1000-row sample, so only assert if full file)
        if vims_parquet_path.stat().st_size > 1_000_000:  # >1 MB → not sample
            assert north_coverage > south_coverage, \
                "Expected higher north-polar VIMS coverage (lakes region)"

    @_NEED_PYARROW
    def test_real_best_resolution_no_inf(self, vims_parquet_path: Path) -> None:
        """
        Best-resolution map from real data must have no +inf values.
        """
        idx = VIMSFootprintIndex(vims_parquet_path)
        out = idx.best_resolution_map(nrows=18, ncols=36)
        assert not np.any(np.isposinf(out)), "Found +inf in best_resolution_map"
        assert out.dtype == np.float32

    @_NEED_PYARROW
    def test_real_summary_contains_row_count(self, vims_parquet_path: Path) -> None:
        """
        Summary string should be non-empty and contain flyby information.
        """
        idx = VIMSFootprintIndex(vims_parquet_path)
        s = idx.summary()
        assert len(s) > 10
        assert "flyby" in s.lower() or "footprint" in s.lower()