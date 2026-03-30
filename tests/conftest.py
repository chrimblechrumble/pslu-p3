"""
tests/conftest.py
==================
Shared pytest fixtures for the Titan Habitability Pipeline test suite.

Fixtures that provide paths to real Cassini data in tests/fixtures/ all use
``pytest.skip`` when the file is absent, so the suite runs cleanly with or
without local data.  Populate tests/fixtures/ to run the integration tests.

See tests/fixtures/README.md for the expected directory layout and data
sources.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Root of the tests/ directory
TESTS_DIR    = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


# ---------------------------------------------------------------------------
# Shapefile fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_shapefiles_dir() -> Path:
    """
    Path to the shapefiles fixture directory (tests/fixtures/shapefiles/).
    Skips if the directory does not exist.
    """
    d = FIXTURES_DIR / "shapefiles"
    if not d.exists():
        pytest.skip(
            f"Shapefile fixtures not found at {d}. "
            "See tests/fixtures/README.md."
        )
    return d


@pytest.fixture(scope="session")
def craters_shp(fixtures_shapefiles_dir: Path) -> Path:
    """Path to Craters.shp in the fixture directory. Skips if absent."""
    shp = fixtures_shapefiles_dir / "Craters.shp"
    if not shp.exists():
        pytest.skip(f"Craters.shp not found at {shp}")
    return shp


# ---------------------------------------------------------------------------
# GTDR topography fixtures
# ---------------------------------------------------------------------------
#
# Cornell GTDR product types (data.astro.cornell.edu/RADAR/DATA/GTDR/):
#   GTDE = Dense interpolated DEM (~90% global, PREFERRED)
#   GT0E = Standard sparse GTDR (nadir tracks only)
#
# Files distributed as .IMG.gz (the reader decompresses transparently).
# Place .IMG, .IMG.gz, or .LBL files in tests/fixtures/gtdr/.

@pytest.fixture(scope="session")
def fixtures_gtdr_dir() -> Path:
    """
    Path to the GTDR/GTDE fixture directory (tests/fixtures/gtdr/).

    Required for integration tests:
      GTDED00N090_T126_V01.IMG[.gz] + .LBL  (GTDE east — preferred)
      GTDED00N270_T126_V01.IMG[.gz] + .LBL  (GTDE west — preferred)
      GT2ED00N090_T126_V01.IMG[.gz] + .LBL  (GT0E east — T126 sparse)
      GT0EB00N090_T077_V01.IMG     + .LBL   (GT0E east — T077 legacy)

    See tests/fixtures/README.md for download instructions.
    """
    d = FIXTURES_DIR / "gtdr"
    if not d.exists():
        pytest.skip(
            f"GTDR fixtures not found at {d}. "
            "Download from https://data.astro.cornell.edu/RADAR/DATA/GTDR/ "
            "or from USGS gtdr-data.zip — see tests/fixtures/README.md."
        )
    return d


def _gtdr_find(gtdr_dir: Path, stem: str) -> Path:
    """Return .IMG or .IMG.gz path for given stem, or None."""
    for suffix in (".IMG", ".IMG.gz"):
        p = gtdr_dir / (stem + suffix)
        if p.exists():
            return p
    return None


@pytest.fixture(scope="session")
def gtde_east_img(fixtures_gtdr_dir: Path) -> Path:
    """
    GTDE east tile — Dense interpolated DEM, lon 0–180°W.
    File: GTDED00N090_T126_V01.IMG or .IMG.gz
    """
    p = _gtdr_find(fixtures_gtdr_dir, "GTDED00N090_T126_V01")
    if p is None:
        pytest.skip(
            "GTDE east tile not found (GTDED00N090_T126_V01.IMG[.gz]). "
            "Download from https://data.astro.cornell.edu/RADAR/DATA/GTDR/"
        )
    return p


@pytest.fixture(scope="session")
def gtde_west_img(fixtures_gtdr_dir: Path) -> Path:
    """
    GTDE west tile — Dense interpolated DEM, lon 180–360°W.
    File: GTDED00N270_T126_V01.IMG or .IMG.gz
    """
    p = _gtdr_find(fixtures_gtdr_dir, "GTDED00N270_T126_V01")
    if p is None:
        pytest.skip(
            "GTDE west tile not found (GTDED00N270_T126_V01.IMG[.gz]). "
            "Download from https://data.astro.cornell.edu/RADAR/DATA/GTDR/"
        )
    return p


@pytest.fixture(scope="session")
def gtdr_east_img(fixtures_gtdr_dir: Path) -> Path:
    """
    GT0E east tile — standard sparse GTDR.
    Tries T126 (GT2ED00N090) first, then legacy T077 (GT0EB00N090).
    """
    p = (_gtdr_find(fixtures_gtdr_dir, "GT2ED00N090_T126_V01") or
         _gtdr_find(fixtures_gtdr_dir, "GT0EB00N090_T077_V01"))
    if p is None:
        pytest.skip(
            "GT0E east tile not found. "
            "Expected: GT2ED00N090_T126_V01.IMG[.gz] or GT0EB00N090_T077_V01.IMG"
        )
    return p


# ---------------------------------------------------------------------------
# VIMS parquet fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vims_parquet_path() -> Path:
    """
    Path to a VIMS spatial footprint parquet file.

    Searches tests/fixtures/vims/ for:
      1. vims_footprints.parquet      (full catalogue ~227 MB)
      2. vims_sample_1000rows.parquet (dev sample, 43 KB)
      3. vims_*.parquet               (any VIMS-prefixed parquet)
    """
    vims_dir = FIXTURES_DIR / "vims"
    if not vims_dir.exists():
        pytest.skip(
            f"VIMS fixture directory not found at {vims_dir}. "
            "See tests/fixtures/README.md."
        )
    candidates = [
        vims_dir / "vims_footprints.parquet",
        vims_dir / "vims_sample_1000rows.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = sorted(vims_dir.glob("vims_*.parquet"))
    if matches:
        return matches[0]
    pytest.skip(
        f"No VIMS parquet file found in {vims_dir}. "
        "Place vims_footprints.parquet or vims_sample_1000rows.parquet there."
    )
