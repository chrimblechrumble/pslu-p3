"""
titan/io/gtdr_reader.py
========================
Reader for PDS3 GTDR (Global Topography Data Record) binary image tiles.

Format details verified against actual label file GT0EB00N090_T077_V01.LBL:

  DATA_SET_ID    : CO-SSA-RADAR-5-GTDR-V1.0
  RECORD_TYPE    : FIXED_LENGTH
  RECORD_BYTES   : 1440
  LABEL_RECORDS  : 5
  IMAGE start    : record 6  → byte offset = (6-1) × 1440 = 7200
  LINES          : 360  (latitude rows)
  LINE_SAMPLES   : 360  (longitude columns per tile = 180° at 2 ppd)
  SAMPLE_TYPE    : PC_REAL  → IEEE-754 float32, little-endian
  SAMPLE_BITS    : 32
  SCALING_FACTOR : 1.0  (values are raw metres, no scaling needed)
  MISSING_CONSTANT: 16#FF7FFFFB# = -3.4028226550889045e+38
    NOTE: This is NOT a NaN. Must mask by exact value comparison.
  MAP_PROJECTION_TYPE : EQUIRECTANGULAR
  A_AXIS_RADIUS  : 2575.0 km
  POSITIVE_LONGITUDE_DIRECTION : WEST (west-positive, 0→360°)
  CENTER_LONGITUDE : 180.0°  (clon180 convention)
  MAP_RESOLUTION : 2.0 ppd  →  0.5°/pixel  →  22.47 km/pixel

Cornell archive product types (data.astro.cornell.edu/RADAR/DATA/GTDR/)
------------------------------------------------------------------------
Files are stored gzip-compressed (.IMG.gz); this reader decompresses them
transparently.  Both .IMG and .IMG.gz paths are accepted.

  GT0ED00N{lon}_{flyby}_V01  — standard GTDR (sparse, altimetry + SARtopo only)
  GTDED00N{lon}_{flyby}_V01  — Dense interpolated DEM (~90% global coverage)
  GTBED00N{lon}_{flyby}_V01  — spherical harmonic / ellipsoid model

Each product has two half-globe tiles:
  N090 tile: lon  0°W → 180°W
  N270 tile: lon 180°W → 360°W

Pipeline DEM priority (see preprocessing.py):
  1. GTDE T126 (interpolated, ~90% global) — PREFERRED
  2. GT0E T126 (sparse, final mission coverage)
  3. GT0E T077 (sparse, partial mission — legacy fallback)

USGS gtdr-data.zip contains the same product set.

References
----------
Zebker et al. (2009) "Size and shape of Saturn's moon Titan"
  doi:10.1126/science.1168905
Lorenz et al. (2013) "A global topographic map of Titan"
  doi:10.1016/j.icarus.2013.04.002
Corlies et al. (2017) "Titan's Topography and Shape at the End of the Cassini Mission"
  doi:10.1002/2017GL075518
PDS label: CO-SSA-RADAR-5-GTDR-V1.0 / DSMAP.CAT
"""

from __future__ import annotations

import gzip
import logging
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from label file
# ---------------------------------------------------------------------------

#: Byte offset to the start of image data (5 label records × 1440 bytes).
GTDR_IMAGE_OFFSET: int = 7200

#: PDS3 MISSING_CONSTANT as a float32 value.
#: 16#FF7FFFFB# decoded as little-endian IEEE-754:
#:   bytes = [0xFB, 0xFF, 0x7F, 0xFF]
#:   value = -3.4028226550889045e+38
#: This is NOT a NaN; must be compared as exact float32.
GTDR_MISSING_CONSTANT: float = struct.unpack(
    "<f", bytes([0xFB, 0xFF, 0x7F, 0xFF])
)[0]

#: Spatial resolution: 2.0 pixels per degree.
GTDR_PPD: float = 2.0

#: Titan sphere radius in km (from label A_AXIS_RADIUS).
GTDR_TITAN_RADIUS_KM: float = 2575.0


# ---------------------------------------------------------------------------
# Label parser
# ---------------------------------------------------------------------------

def parse_gtdr_label(lbl_path: Path) -> dict:
    """
    Parse a PDS3 GTDR label file and return a metadata dictionary.

    Parameters
    ----------
    lbl_path:
        Path to the ``.LBL`` file (e.g. ``GT0EB00N090_T077_V01.LBL``).

    Returns
    -------
    dict
        Keys include: ``lines``, ``line_samples``, ``record_bytes``,
        ``label_records``, ``image_offset``, ``map_resolution``,
        ``minimum_latitude``, ``maximum_latitude``,
        ``westernmost_longitude``, ``easternmost_longitude``,
        ``missing_constant_float32``, ``img_path``.
    """
    lbl_path = Path(lbl_path)
    meta: dict = {}

    with open(lbl_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if "=" not in line or line.startswith("/*"):
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().rstrip(";").strip('"')
            # Strip unit annotations like <KM>, <DEG>, <PIX/DEG>
            if "<" in val:
                val = val[:val.index("<")].strip()
            meta[key] = val

    # Derived fields
    record_bytes   = int(meta.get("RECORD_BYTES", 1440))
    label_records  = int(meta.get("LABEL_RECORDS", 5))
    image_offset   = (label_records) * record_bytes  # records are 1-indexed in PDS

    # IMG file path (same stem as LBL)
    img_file = meta.get("^IMAGE", "").strip('"').split(",")[0].strip('"')
    if img_file:
        img_path = lbl_path.parent / img_file
    else:
        img_path = lbl_path.with_suffix(".IMG")

    return {
        "lines":                int(meta.get("LINES", 360)),
        "line_samples":         int(meta.get("LINE_SAMPLES", 360)),
        "record_bytes":         record_bytes,
        "label_records":        label_records,
        "image_offset":         image_offset,
        "map_resolution":       float(meta.get("MAP_RESOLUTION", 2.0)),
        "minimum_latitude":     float(meta.get("MINIMUM_LATITUDE", -90.0)),
        "maximum_latitude":     float(meta.get("MAXIMUM_LATITUDE",  90.0)),
        "westernmost_longitude":float(meta.get("WESTERNMOST_LONGITUDE", 0.0)),
        "easternmost_longitude":float(meta.get("EASTERNMOST_LONGITUDE", 180.0)),
        "center_longitude":     float(meta.get("CENTER_LONGITUDE", 180.0)),
        "missing_constant_float32": GTDR_MISSING_CONSTANT,
        "img_path":             img_path,
        "data_set_id":          meta.get("DATA_SET_ID", ""),
        "product_id":           meta.get("PRODUCT_ID", ""),
    }


# ---------------------------------------------------------------------------
# Binary image reader
# ---------------------------------------------------------------------------

def read_gtdr_img(
    img_path: Path,
    lbl_path: Optional[Path] = None,
    replace_missing_with_nan: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Read a GTDR binary image file and return elevation data.

    Parameters
    ----------
    img_path:
        Path to the ``.IMG`` binary file.
    lbl_path:
        Path to the companion ``.LBL`` label.  If None, looks for a
        ``.LBL`` file with the same stem as ``img_path``.
    replace_missing_with_nan:
        If True (default), replace the PDS3 MISSING_CONSTANT
        (−3.4028×10³⁸) with ``np.nan``.

    Returns
    -------
    data : np.ndarray
        2-D float32 array of shape (lines, line_samples).
        Rows run north → south (row 0 = MAXIMUM_LATITUDE).
        Columns run west → east in the west-positive convention
        (col 0 = WESTERNMOST_LONGITUDE, increasing westward).
    meta : dict
        Label metadata from ``parse_gtdr_label()``.

    Raises
    ------
    FileNotFoundError
        If the IMG or LBL file cannot be found.
    ValueError
        If the file size does not match the expected dimensions.
    """
    img_path = Path(img_path)

    # Accept both .IMG and .IMG.gz; Cornell files are distributed gzip-compressed.
    # Try the given path first, then the .gz companion if not found.
    is_gzip = False
    if not img_path.exists():
        gz_path = img_path.with_suffix(img_path.suffix + ".gz")
        if gz_path.exists():
            img_path = gz_path
            is_gzip = True
        else:
            raise FileNotFoundError(
                f"GTDR IMG not found: {img_path} (also tried {gz_path})"
            )
    elif img_path.suffix.lower() == ".gz":
        is_gzip = True

    # Locate label — strip .gz suffix when looking for the companion .LBL
    lbl_base = img_path.with_suffix("") if is_gzip else img_path
    if lbl_path is None:
        lbl_path = lbl_base.with_suffix(".LBL")
        if not lbl_path.exists():
            lbl_path = lbl_base.with_suffix(".lbl")
    if not Path(lbl_path).exists():
        logger.warning(
            "Label file not found for %s; using hard-coded defaults.", img_path
        )
        meta = {
            "lines": 360, "line_samples": 360,
            "image_offset": GTDR_IMAGE_OFFSET,
            "map_resolution": GTDR_PPD,
            "minimum_latitude": -90.0, "maximum_latitude": 90.0,
            "westernmost_longitude": 0.0, "easternmost_longitude": 180.0,
            "center_longitude": 180.0,
            "missing_constant_float32": GTDR_MISSING_CONSTANT,
            "img_path": img_path,
        }
    else:
        meta = parse_gtdr_label(Path(lbl_path))

    lines        = meta["lines"]
    line_samples = meta["line_samples"]
    offset       = meta["image_offset"]
    expected_bytes = lines * line_samples * 4  # float32 = 4 bytes
    expected_bytes = lines * line_samples * 4  # float32 = 4 bytes

    if is_gzip:
        # Decompress fully in memory — GTDR tiles are ≤15 MB uncompressed
        with gzip.open(img_path, "rb") as fh:
            file_bytes = fh.read()
        available = len(file_bytes) - offset
    else:
        file_size = img_path.stat().st_size
        available = file_size - offset
        file_bytes = None

    if available < expected_bytes:
        # Some GTDE files (e.g. T126 at 4 ppd) are slightly shorter than
        # the label claims — the last few rows may be absent.  Clamp to
        # what is actually available rather than raising.
        actual_lines = available // (line_samples * 4)
        if actual_lines < 1:
            raise ValueError(
                f"IMG contains no usable data: offset={offset}, "
                f"available={available}, expected={expected_bytes}."
            )
        if actual_lines < lines:
            logger.warning(
                "%s: label says %d lines but only %d rows available "
                "(%d bytes short). Reading %d lines.",
                img_path.name, lines, actual_lines,
                expected_bytes - available, actual_lines,
            )
            lines = actual_lines
            expected_bytes = lines * line_samples * 4

    if file_bytes is not None:
        raw = file_bytes[offset: offset + expected_bytes]
    else:
        with open(img_path, "rb") as fh:
            fh.seek(offset)
            raw = fh.read(expected_bytes)

    # PC_REAL = little-endian IEEE-754 float32
    data = np.frombuffer(raw, dtype="<f4").reshape(lines, line_samples).copy()

    if replace_missing_with_nan:
        # Must compare by value, NOT np.isnan(), because MISSING_CONSTANT
        # is -3.4028e38 which is a valid (very negative) float, not NaN.
        missing = meta["missing_constant_float32"]
        # Cast to float64 before the subtraction to avoid float32 overflow
        # (missing ≈ -3.4e38, which is near the float32 min).
        # Suppress the "invalid value in cast" warning: the missing constant
        # itself is an extreme float32 value; the cast is intentional.
        with np.errstate(invalid='ignore'):
            data64 = data.astype(np.float64)
        data[np.abs(data64 - float(missing)) < 1e30] = np.nan

    logger.info(
        "Read GTDR tile %s: shape=%s, valid px=%d, nan px=%d",
        img_path.name,
        data.shape,
        int(np.sum(np.isfinite(data))),
        int(np.sum(~np.isfinite(data))),
    )
    return data, meta


# ---------------------------------------------------------------------------
# Global mosaic assembler
# ---------------------------------------------------------------------------

def mosaic_gtdr_tiles(
    east_img: Path,
    west_img: Path,
    east_lbl: Optional[Path] = None,
    west_lbl: Optional[Path] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Merge eastern and western GTDR half-globe tiles into a single global array.

    The eastern tile covers lon 0°–180°W (columns 0→359 in the mosaic).
    The western tile covers lon 180°–360°W (columns 360→719 in the mosaic).

    Parameters
    ----------
    east_img:
        Path to eastern hemisphere IMG (e.g. GT0EB00N090_T077_V01.IMG).
    west_img:
        Path to western hemisphere IMG (e.g. GT0WB00N270_T077_V01.IMG).
    east_lbl, west_lbl:
        Optional companion label files.

    Returns
    -------
    mosaic : np.ndarray
        Global array of shape (360, 720) at 2 ppd.
        Longitude runs 0°W → 360°W left to right.
        Latitude runs +90° → −90° top to bottom.
    meta : dict
        Merged metadata dict.
    """
    east_data, east_meta = read_gtdr_img(east_img, east_lbl)
    west_data, west_meta = read_gtdr_img(west_img, west_lbl)

    if east_data.shape != west_data.shape:
        # Both tiles are truncated, but by different amounts. The label's
        # expected row count is the authority — pad the shorter tile with
        # NaN rows at the bottom so the two halves can be concatenated.
        # This preserves geographic alignment (both tiles are north-up and
        # the truncation is always at the south end).
        target_rows = max(east_data.shape[0], west_data.shape[0])
        def _pad_rows(arr: np.ndarray, n: int) -> np.ndarray:
            if arr.shape[0] >= n:
                return arr
            pad = np.full((n - arr.shape[0], arr.shape[1]), np.nan, dtype=arr.dtype)
            logger.warning(
                "GTDR mosaic: padding %d missing south rows with NaN "
                "(tile was %d rows, target %d)",
                n - arr.shape[0], arr.shape[0], n,
            )
            return np.concatenate([arr, pad], axis=0)
        east_data = _pad_rows(east_data, target_rows)
        west_data = _pad_rows(west_data, target_rows)

    mosaic = np.concatenate([east_data, west_data], axis=1)
    meta = east_meta.copy()
    meta["westernmost_longitude"] = 0.0
    meta["easternmost_longitude"] = 360.0
    meta["line_samples"] = mosaic.shape[1]

    logger.info(
        "GTDR global mosaic assembled: shape=%s, valid=%.1f%%",
        mosaic.shape,
        100.0 * np.sum(np.isfinite(mosaic)) / mosaic.size,
    )
    return mosaic, meta


# ---------------------------------------------------------------------------
# Affine transform helper
# ---------------------------------------------------------------------------

def gtdr_affine_transform(meta: dict) -> Tuple[float, float, float, float]:
    """
    Compute the affine transform for a GTDR tile or mosaic.

    Returns the (west_edge_m, north_edge_m, pixel_width_m, pixel_height_m)
    tuple needed to build a rasterio Affine or xarray coordinates.

    The transform is in METRES (projected equirectangular, Titan sphere).

    Parameters
    ----------
    meta:
        Metadata dict from ``parse_gtdr_label()`` or ``mosaic_gtdr_tiles()``.

    Returns
    -------
    Tuple (west_m, north_m, dx_m, dy_m)
        west_m  : X coordinate of left edge (metres)
        north_m : Y coordinate of top edge (metres)
        dx_m    : pixel width (metres, positive)
        dy_m    : pixel height (metres, negative = north-up)
    """
    import math
    r = GTDR_TITAN_RADIUS_KM * 1000.0  # metres
    deg_to_m = r * math.pi / 180.0

    west_deg  = meta["westernmost_longitude"]
    north_deg = meta["maximum_latitude"]
    ppd       = meta["map_resolution"]

    west_m  = west_deg  * deg_to_m
    north_m = north_deg * deg_to_m
    dx_m    = (1.0 / ppd) * deg_to_m   # positive: moving east in pixel space
    dy_m    = -(1.0 / ppd) * deg_to_m  # negative: moving south in pixel space

    return west_m, north_m, dx_m, dy_m
