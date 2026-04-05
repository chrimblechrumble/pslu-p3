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
titan/io/vims_cube_mosaic.py
============================
Builds global mosaics of individual VIMS spectral windows from calibrated
cubes downloaded from the Nantes portal (vims.univ-nantes.fr).

Scientific motivation
---------------------
The pre-built Seignovert+2019 VIMS+ISS GeoTIFF mosaic (used by the pipeline
as its primary organic_abundance input) provides three fixed band ratios:

  Band 1: 1.59/1.27 umm + ISS 938 nm  (tholin proxy)
  Band 2: 2.03/1.27 umm + ISS 938 nm  (surface window)
  Band 3: 1.27/1.08 umm + ISS 938 nm  (water-ice proxy)

These were chosen to represent surface spectral heterogeneity in Titan's
three most commonly-used VIMS atmospheric windows.  What the mosaic does NOT
provide is the 5.0 umm window -- Titan's deepest and least haze-affected
infrared surface window.

The 5.0 / 2.03 umm ratio provides the strongest organic-vs-water-ice
discrimination of any VIMS band combination (Solomonidou et al. 2018):

  5.0/2.03 > 2.0  : organic-rich terrain (dunes, tholins, organic plains)
  5.0/2.03 < 0.8  : water-ice-rich terrain (mountains, Xanadu basement)
  5.0/2.03 ~ 1.0  : mixed or plains material

This ratio is physically different from the 1.59/1.27 ratio in the mosaic
and provides genuinely new compositional information, especially in the
equatorial mountain chains where tholins and exposed ice are interleaved.

Technical approach
------------------
This module downloads individual ISIS3-calibrated C*_ir.cub files from the
Nantes portal (CC-BY-4.0), reads their binary data without requiring ISIS3
to be installed, and reprojects each per-pixel spectrum onto the canonical
pipeline grid using the per-pixel lat/lon from the companion N*_ir.cub
navigation cube.

Mosaicking strategy: for each grid cell, keep the pixel from the cube that
observed it at the smallest emission angle (most nadir-looking geometry).
This maximises surface signal and minimises path-length through the
atmospheric haze column.

ISIS3 cube format (implemented here without ISIS3 dependency)
-------------------------------------------------------------
ISIS3 cubes have a text PVL label starting at byte 0 and ending with
the literal string ``End\\n``.  The label contains a ``Core`` object that
specifies the data start byte, spatial dimensions, number of bands, data
type (Real = float32, SignedWord = int16), byte order, and scaling.

Calibrated C*_ir.cub from Nantes:
  Type      : Real (float32)
  ByteOrder : Lsb (little-endian)
  Format    : Bsq (band-sequential)
  Bands     : 256 (VIMS-IR, 0.88-5.12 umm)
  Pixels    : variable (typically 64x64 to 640x64)

Navigation N*_ir.cub from Nantes:
  Same format; 6 bands
  Band 1 : Latitude  (deg, planetocentric, east-positive)
  Band 2 : Longitude (deg, east-positive, -180 to +180)
  Band 3 : Incidence angle
  Band 4 : Emission angle
  Band 5 : Phase angle
  Band 6 : Resolution (km/pixel)

VIMS-IR wavelength calibration
-------------------------------
Linear approximation (accurate to ~0.01 umm; good enough for window selection):
  Band 0  (1-indexed band 1) : 0.8842 umm
  Band 255 (1-indexed band 256) : 5.1088 umm
  Spacing : ~0.01657 umm/band

References
----------
Le Mouelic et al. (2019)   doi:10.1016/j.icarus.2018.09.017
Solomonidou et al. (2018)  doi:10.1029/2017JE005462
Brown et al. (2004)         doi:10.1007/s11214-004-1453-x
"""

from __future__ import annotations

import logging
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VIMS-IR wavelength calibration
# ---------------------------------------------------------------------------

#: VIMS-IR wavelength at the start of band 0 (umm).
VIMS_IR_WAVE_START: float = 0.8842
#: VIMS-IR wavelength at the end of band 255 (umm).
VIMS_IR_WAVE_STOP:  float = 5.1088
#: Number of VIMS-IR bands in calibrated cubes.
VIMS_IR_N_BANDS:    int   = 256

#: Nominal VIMS-IR band-centre wavelengths (umm), 0-indexed.
#: Linear approximation: accurate to ~0.01 umm.  Sufficient for window
#: selection (which averages over a ~0.1 umm window around the target).
VIMS_IR_WAVELENGTHS: np.ndarray = np.linspace(
    VIMS_IR_WAVE_START, VIMS_IR_WAVE_STOP, VIMS_IR_N_BANDS,
    dtype=np.float32,
)

#: VIMS atmospheric windows on Titan (centre wavelength in umm).
#: Only these bands carry surface photons; all others are opaque in haze.
TITAN_SURFACE_WINDOWS: Dict[str, float] = {
    "1.08um":  1.08,
    "1.27um":  1.27,
    "1.59um":  1.59,
    "2.03um":  2.03,
    "2.69um":  2.69,
    "2.79um":  2.79,
    "5.0um":   5.00,
}

#: Window half-width used when averaging bands around a target wavelength.
WINDOW_HALF_WIDTH_UM: float = 0.06


def bands_for_wavelength(
    target_um: float,
    half_width_um: float = WINDOW_HALF_WIDTH_UM,
) -> List[int]:
    """
    Return the 0-based band indices within +/- half_width_um of target_um.

    Parameters
    ----------
    target_um:
        Target wavelength in umm.
    half_width_um:
        Half-width of the integration window in umm.

    Returns
    -------
    List[int]
        0-based band indices (may be empty if target is out of range).

    Examples
    --------
    >>> len(bands_for_wavelength(5.0))
    8
    >>> 248 in bands_for_wavelength(5.0)
    True
    """
    lo = target_um - half_width_um
    hi = target_um + half_width_um
    return [i for i, w in enumerate(VIMS_IR_WAVELENGTHS) if lo <= w <= hi]


# ---------------------------------------------------------------------------
# Minimal ISIS3 PVL label reader  (no ISIS3 or pvl dependency)
# ---------------------------------------------------------------------------

class ISIS3LabelError(ValueError):
    """Raised when an ISIS3 cube label cannot be parsed."""


def _read_isis3_label_text(path: Path, max_bytes: int = 131_072) -> str:
    """
    Read the PVL text label from an ISIS3 .cub file.

    ISIS3 labels start at byte 0 and end with the literal token ``End\\n``
    (case-sensitive) on a line by itself.  The label is pure ASCII text.

    Parameters
    ----------
    path:
        Path to the .cub file.
    max_bytes:
        Maximum bytes to read looking for the label end (default 128 KB).

    Returns
    -------
    str
        The complete label text up to and including ``End\\n``.
    """
    with open(path, "rb") as fh:
        raw = fh.read(max_bytes)
    text = raw.decode("ascii", errors="replace")
    # ISIS3 label ends with a line containing exactly "End"
    match = re.search(r"^End\s*$", text, re.MULTILINE)
    if match is None:
        raise ISIS3LabelError(
            f"Could not find ISIS3 label terminator 'End' in first "
            f"{max_bytes} bytes of {path}"
        )
    return text[: match.end()]


def _parse_isis3_label(label_text: str) -> dict:
    """
    Extract the fields needed to read the binary data from an ISIS3 label.

    Parses (case-insensitive) from the ``IsisCube/Core`` object:
      StartByte, Samples, Lines, Bands, Type, ByteOrder, Base, Multiplier, Format

    Returns
    -------
    dict with keys:
        start_byte   : int   -- byte offset of first data byte (1-indexed in PVL)
        samples      : int
        lines        : int
        bands        : int
        dtype        : str   -- 'f4' (Real) or 'i2' (SignedWord) or 'i4' (Integer)
        byte_order   : str   -- '<' (Lsb) or '>' (Msb)
        base         : float -- additive offset
        multiplier   : float -- scale factor  (DN * multiplier + base = I/F)
        format       : str   -- 'bsq', 'bil', or 'bip'
    """

    def _get(key: str, text: str) -> Optional[str]:
        """Return first match of 'KEY = value' (case-insensitive)."""
        m = re.search(
            rf"^\s*{re.escape(key)}\s*=\s*(.+?)(?:\s*<[^>]+>)?\s*$",
            text, re.MULTILINE | re.IGNORECASE,
        )
        return m.group(1).strip() if m else None

    # Extract fields (labels may use different capitalisation between cubes)
    start_byte  = _get("StartByte",   label_text)
    samples     = _get("Samples",     label_text)
    lines       = _get("Lines",       label_text)
    bands       = _get("Bands",       label_text)
    pix_type    = _get("Type",        label_text)
    byte_order  = _get("ByteOrder",   label_text)
    base        = _get("Base",        label_text)
    multiplier  = _get("Multiplier",  label_text)
    fmt         = _get("Format",      label_text)
    tile_s      = _get("TileSamples", label_text)   # present when Format = Tile
    tile_l      = _get("TileLines",   label_text)   # present when Format = Tile

    if any(v is None for v in [start_byte, samples, lines, bands]):
        raise ISIS3LabelError(
            f"Missing required field in ISIS3 label. "
            f"start_byte={start_byte}, samples={samples}, lines={lines}, "
            f"bands={bands}"
        )

    # Map ISIS3 type strings to numpy dtype characters
    type_map = {
        "real":        "f4",
        "double":      "f8",
        "unsignedbyte":"u1",
        "signedword":  "i2",
        "integer":     "i4",
        "unsignedinteger": "u4",
    }
    dtype_key = (pix_type or "real").lower().replace(" ", "")
    dtype_char = type_map.get(dtype_key, "f4")

    order_char = "<"  # default: little-endian
    if byte_order and byte_order.lower() in ("msb", "ieee_msb", "big_endian"):
        order_char = ">"

    fmt_str = (fmt or "bsq").lower().strip()

    return {
        "start_byte":   int(start_byte) - 1,  # PVL is 1-indexed; convert to 0-indexed
        "samples":      int(samples),
        "lines":        int(lines),
        "bands":        int(bands),
        "dtype":        order_char + dtype_char,
        "base":         float(base)       if base       else 0.0,
        "multiplier":   float(multiplier) if multiplier else 1.0,
        "format":       fmt_str,
        "tile_samples": int(tile_s) if tile_s else None,
        "tile_lines":   int(tile_l) if tile_l else None,
    }


def read_isis3_cube(
    path: Path,
    band_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Read one or more bands from an ISIS3 .cub file.

    No ISIS3 installation required.  Reads the PVL label then mmaps the
    binary data section using numpy.

    Parameters
    ----------
    path:
        Path to the ISIS3 .cub file.
    band_indices:
        0-based indices of bands to return.  If None, all bands are read.
        Specifying a subset saves memory for large cubes.

    Returns
    -------
    np.ndarray
        float32 array, shape (n_bands_selected, lines, samples).
        Values are in I/F units for calibrated C*_ir.cub files.
        Values are in deg / km for navigation N*_ir.cub files.

    Raises
    ------
    ISIS3LabelError
        If the label cannot be parsed.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ISIS3 cube not found: {path}")

    label_text  = _read_isis3_label_text(path)
    meta        = _parse_isis3_label(label_text)

    S = meta["samples"]
    L = meta["lines"]
    B = meta["bands"]
    dtype       = np.dtype(meta["dtype"])
    start_byte  = meta["start_byte"]
    base        = meta["base"]
    multiplier  = meta["multiplier"]
    fmt         = meta["format"]
    tile_s      = meta.get("tile_samples")
    tile_l      = meta.get("tile_lines")

    if fmt in ("bsq", "bil", "bip"):
        # Contiguous linear formats -- single memmap read
        n_values = B * L * S
        raw: np.ndarray = np.memmap(
            path, dtype=dtype, mode="r",
            offset=start_byte,
            shape=(n_values,),
        ).copy().astype(np.float32)

        if fmt == "bsq":
            cube = raw.reshape((B, L, S))
        elif fmt == "bil":
            cube = raw.reshape((L, B, S)).transpose(1, 0, 2)   # -> (B, L, S)
        else:  # bip
            cube = raw.reshape((L, S, B)).transpose(2, 0, 1)   # -> (B, L, S)

    elif fmt == "tile":
        # ----------------------------------------------------------------
        # ISIS3 Tile format
        # ----------------------------------------------------------------
        # Data is stored as TileSamples × TileLines spatial tiles,
        # band-sequential.  Each band is divided into a grid of tiles;
        # tiles at the right and bottom edges are padded to full tile
        # dimensions with the ISIS3 NULL special value.
        #
        # Storage order (all formats verified against ISIS3 source):
        #   for each band b = 0 .. B-1:
        #     for each tile_row tr = 0 .. n_tile_rows-1:
        #       for each tile_col tc = 0 .. n_tile_cols-1:
        #         TileLines × TileSamples pixels (row-major)
        #
        # NULL special value (float32): hex 0xFF7FFFFA = -3.4028e+38
        # We replace it with NaN after reading.
        # ----------------------------------------------------------------
        if tile_s is None or tile_l is None:
            raise ISIS3LabelError(
                f"Format = Tile but TileSamples/TileLines missing in {path}"
            )

        import math
        TS = tile_s   # tile width  in samples
        TL = tile_l   # tile height in lines

        n_tile_cols = math.ceil(S / TS)
        n_tile_rows = math.ceil(L / TL)
        n_tiles_per_band = n_tile_cols * n_tile_rows
        tile_pixels  = TS * TL        # pixels per tile (padded)

        # ISIS3 NULL for float32 is a specific bit pattern; values < -3e38
        # are treated as NULL.
        ISIS3_NULL_THRESHOLD: float = -3.0e38

        # Output array (selected bands or all)
        sel = list(band_indices) if band_indices is not None else list(range(B))
        cube = np.full((len(sel), L, S), np.nan, dtype=np.float32)

        # Memory-map the full data section for random tile access
        total_pixels = B * n_tiles_per_band * tile_pixels
        raw_all = np.memmap(
            path, dtype=dtype, mode="r",
            offset=start_byte,
            shape=(total_pixels,),
        )

        for out_idx, b in enumerate(sel):
            band_offset = b * n_tiles_per_band * tile_pixels
            for tr in range(n_tile_rows):
                for tc in range(n_tile_cols):
                    tile_idx    = tr * n_tile_cols + tc
                    tile_offset = band_offset + tile_idx * tile_pixels
                    tile_raw = raw_all[tile_offset: tile_offset + tile_pixels]
                    tile_arr = tile_raw.reshape(TL, TS).astype(np.float32)
                    # Apply scaling
                    if multiplier != 1.0 or base != 0.0:
                        tile_arr = tile_arr * multiplier + base
                    # Replace ISIS3 NULL with NaN
                    tile_arr[tile_arr < ISIS3_NULL_THRESHOLD] = np.nan
                    # Paste into output, clipping to actual cube dimensions
                    l_start = tr * TL;  l_end = min(l_start + TL, L)
                    s_start = tc * TS;  s_end = min(s_start + TS, S)
                    cube[out_idx, l_start:l_end, s_start:s_end] = \
                        tile_arr[:l_end - l_start, :s_end - s_start]

        # Scaling already applied per-tile above; skip global scale below
        if band_indices is not None:
            return cube
        return cube

    else:
        raise ISIS3LabelError(f"Unknown cube format '{fmt}' in {path}")

    # Apply scaling: value = raw * multiplier + base  (for linear formats)
    if multiplier != 1.0 or base != 0.0:
        cube = cube * multiplier + base

    # Select requested bands
    if band_indices is not None:
        cube = cube[band_indices, :, :]

    return cube


# ---------------------------------------------------------------------------
# Window mosaic builder
# ---------------------------------------------------------------------------

#: Minimum emission angle filter (deg).  Pixels observed above this angle
#: have a longer atmospheric path length and weaker surface signal.
MAX_EMISSION_DEG: float = 40.0

#: Maximum pixel resolution to include in the global mosaic (km/px).
#: Smaller = higher resolution = more selective.  50 km/px catches most
#: Titan-targeting observations while excluding distant, unfocused cubes.
MAX_RESOLUTION_KM: float = 50.0


class VIMSWindowMosaicker:
    """
    Builds a global raster mosaic of a VIMS spectral window from individual
    calibrated cubes downloaded from the Nantes portal.

    The output is a float32 GeoTIFF on the canonical pipeline grid, where
    each pixel holds the mean I/F value in the target spectral window from
    the observation with the smallest emission angle at that location.

    DECLARED ASSUMPTIONS
    --------------------
    1. **Wavelength calibration**: band-to-wavelength mapping uses a linear
       approximation.  The actual VIMS-IR wavelengths deviate from linear by
       up to ~0.01 umm at the band edges (Brown et al. 2004 Table II).  The
       window integration (averaging multiple bands around the target) reduces
       the sensitivity to this error.

    2. **No atmospheric correction**: values are raw I/F from the vimscal
       pipeline (RC19 calibration).  No haze removal, phase-angle correction,
       or bidirectional reflectance modelling is applied here.  The
       Seignovert+2019 mosaic uses empirical atmospheric corrections not
       reproduced here.  For window ratios (5.0/2.03 umm), common-mode
       atmospheric effects partially cancel.

    3. **Best-emission-angle mosaicking**: per-pixel winner is the cube with
       the smallest emission angle.  This is a valid but not optimal strategy.
       It does not account for seasonal lighting variations, surface roughness,
       or photometric angle effects.

    4. **Navigation-cube geometry**: lat/lon come from the N*_ir.cub ISIS3
       geometry cube.  The per-pixel geometry is derived from SPICE kernels
       by the Nantes calibration pipeline; we take these as ground truth
       without independent verification.

    5. **Parquet lon convention**: the parquet footprint index uses west-positive
       longitude (0->360 deg), consistent with the canonical pipeline grid.
       The N*_ir.cub geometry cube uses east-positive longitude (-180->+180 deg).
       Conversion is applied per-pixel before projecting onto the canonical grid.

    Parameters
    ----------
    cube_cache_dir:
        Directory to cache downloaded C*_ir.cub and N*_ir.cub files.
        Defaults to ``data/raw/vims_cubes/``.
    max_emission_deg:
        Reject pixels with emission angle > this value.
    max_resolution_km:
        Only consider cubes whose median pixel resolution is <= this value.
    max_cubes:
        Maximum number of cubes to download.  Set low for development/testing.
    """

    def __init__(
        self,
        cube_cache_dir: Path,
        max_emission_deg:  float = MAX_EMISSION_DEG,
        max_resolution_km: float = MAX_RESOLUTION_KM,
        max_cubes:         Optional[int] = None,
    ) -> None:
        self.cube_cache_dir   = Path(cube_cache_dir)
        self.max_emission_deg = max_emission_deg
        self.max_resolution_km= max_resolution_km
        self.max_cubes        = max_cubes
        self.cube_cache_dir.mkdir(parents=True, exist_ok=True)

    # -- Public API -----------------------------------------------------------

    def select_cube_ids(self, parquet_path: Path) -> List[str]:
        """
        Filter the VIMS parquet index to good-quality cube IDs.

        Criteria:
        - Minimum resolution (median per cube) <= max_resolution_km
        - Cubes are sorted ascending by median resolution (best first)
        - At most max_cubes IDs returned

        Parameters
        ----------
        parquet_path:
            Path to the VIMS footprint parquet.

        Returns
        -------
        List[str]
            Cube IDs, sorted best-resolution first.
        """
        logger.info("Loading VIMS parquet for cube selection: %s", parquet_path)
        df = pd.read_parquet(parquet_path)

        if "id" not in df.columns or "res" not in df.columns:
            raise ValueError(
                f"VIMS parquet must have 'id' and 'res' columns. "
                f"Found: {list(df.columns)}"
            )

        # Per-cube median resolution
        cube_stats = (
            df.groupby("id")["res"]
            .median()
            .reset_index(name="median_res")
        )
        good = cube_stats[cube_stats["median_res"] <= self.max_resolution_km]
        good = good.sort_values("median_res")

        ids = list(good["id"].astype(str))
        if self.max_cubes is not None:
            ids = ids[: self.max_cubes]

        logger.info(
            "Selected %d / %d unique cubes (median_res <= %.0f km/px)",
            len(ids), len(cube_stats), self.max_resolution_km,
        )
        return ids

    def build_mosaic(
        self,
        parquet_path: Path,
        target_um:    float,
        nrows:        int,
        ncols:        int,
    ) -> np.ndarray:
        """
        Build a global mosaic of the target spectral window.

        Steps:
        1. Select cube IDs from the parquet index.
        2. For each cube: download C*_ir.cub + N*_ir.cub (cached).
        3. Extract per-pixel: window mean I/F, lat, lon, emission angle.
        4. Project onto canonical grid; best-emission-angle wins per pixel.

        Parameters
        ----------
        parquet_path:
            Path to VIMS footprint parquet.
        target_um:
            Target spectral window centre wavelength (umm).
        nrows, ncols:
            Canonical grid dimensions.

        Returns
        -------
        np.ndarray
            float32 array of shape (nrows, ncols).  NaN where no valid
            cube observed that pixel within the emission-angle threshold.
        """
        band_idxs = bands_for_wavelength(target_um)
        if not band_idxs:
            raise ValueError(
                f"No VIMS-IR bands within {WINDOW_HALF_WIDTH_UM} umm of "
                f"{target_um} umm. Valid range: {VIMS_IR_WAVE_START}-"
                f"{VIMS_IR_WAVE_STOP} umm."
            )
        logger.info(
            "Building %.2f umm mosaic (%d bands averaged: %s ... %s)",
            target_um, len(band_idxs),
            f"{VIMS_IR_WAVELENGTHS[band_idxs[0]]:.3f}",
            f"{VIMS_IR_WAVELENGTHS[band_idxs[-1]]:.3f}",
        )

        cube_ids = self.select_cube_ids(parquet_path)
        if not cube_ids:
            logger.warning(
                "No cubes passed the resolution filter. "
                "Returning all-NaN mosaic."
            )
            return np.full((nrows, ncols), np.nan, dtype=np.float32)

        from titan.io.vims_reader import VIMSCubeDownloader
        downloader = VIMSCubeDownloader(self.cube_cache_dir)

        # Accumulators: best I/F value and corresponding emission angle
        mosaic     = np.full((nrows, ncols), np.nan, dtype=np.float32)
        best_emiss = np.full((nrows, ncols), np.inf, dtype=np.float32)
        n_good     = 0
        n_fail     = 0

        for cube_id in cube_ids:
            try:
                vals, lats, lons_wp, emiss = self._extract_cube(
                    cube_id, band_idxs, downloader
                )
            except Exception as exc:
                logger.warning("Skipping cube %s: %s", cube_id, exc)
                n_fail += 1
                continue

            if vals is None:
                continue

            # Project onto canonical grid
            # Canonical grid: row 0 = north (+90 deg), row nrows-1 = south (-90 deg)
            # lon west-positive: col 0 = 0 degW, col ncols-1 = 360 degW
            row_f = (90.0 - lats)  / 180.0 * nrows
            col_f = lons_wp         / 360.0 * ncols

            row_i = np.clip(row_f.astype(np.int32), 0, nrows - 1)
            col_i = np.clip(col_f.astype(np.int32), 0, ncols - 1)

            # Keep pixel if its emission angle is lower than the current best
            better = emiss < best_emiss[row_i, col_i]
            row_sel = row_i[better]
            col_sel = col_i[better]
            mosaic[row_sel, col_sel]     = vals[better]
            best_emiss[row_sel, col_sel] = emiss[better]
            n_good += 1

            if n_good % 20 == 0:
                valid_pct = 100.0 * np.isfinite(mosaic).sum() / mosaic.size
                logger.info(
                    "  %.2f umm mosaic: %d cubes processed, "
                    "%.1f%% coverage, %d failed",
                    target_um, n_good, valid_pct, n_fail,
                )

        valid_pct = 100.0 * np.isfinite(mosaic).sum() / mosaic.size
        logger.info(
            "%.2f umm mosaic complete: %d cubes, %.1f%% coverage, "
            "%d cubes failed/skipped",
            target_um, n_good, valid_pct, n_fail,
        )
        return mosaic

    def build_ratio_mosaic(
        self,
        parquet_path:   Path,
        numerator_um:   float,
        denominator_um: float,
        nrows:          int,
        ncols:          int,
    ) -> np.ndarray:
        """
        Build a spectral ratio mosaic (numerator_um / denominator_um).

        For each pixel, the ratio is computed from the SAME cube (so both
        windows share identical atmospheric path and geometry), then the
        best-emission-angle cube wins.

        This is more physically meaningful than ratioing two separate mosaics.

        Parameters
        ----------
        parquet_path:
            VIMS footprint parquet path.
        numerator_um:
            Numerator window wavelength (umm). e.g. 5.0
        denominator_um:
            Denominator window wavelength (umm). e.g. 2.03
        nrows, ncols:
            Canonical grid dimensions.

        Returns
        -------
        np.ndarray
            float32 ratio array, shape (nrows, ncols). NaN = no data.
        """
        num_bands   = bands_for_wavelength(numerator_um)
        denom_bands = bands_for_wavelength(denominator_um)

        if not num_bands or not denom_bands:
            raise ValueError(
                f"No bands found for {numerator_um} or {denominator_um} umm."
            )

        logger.info(
            "Building %.2f/%.2f umm ratio mosaic",
            numerator_um, denominator_um,
        )

        cube_ids = self.select_cube_ids(parquet_path)
        if not cube_ids:
            logger.warning("No cubes selected; returning all-NaN.")
            return np.full((nrows, ncols), np.nan, dtype=np.float32)

        from titan.io.vims_reader import VIMSCubeDownloader
        downloader = VIMSCubeDownloader(self.cube_cache_dir)

        mosaic     = np.full((nrows, ncols), np.nan, dtype=np.float32)
        best_emiss = np.full((nrows, ncols), np.inf, dtype=np.float32)
        n_good = n_fail = 0

        all_band_idxs = sorted(set(num_bands + denom_bands))

        for cube_id in cube_ids:
            try:
                cube_data, lats, lons_wp, emiss = self._extract_cube(
                    cube_id, all_band_idxs, downloader
                )
            except Exception as exc:
                logger.warning("Skipping cube %s: %s", cube_id, exc)
                n_fail += 1
                continue

            if cube_data is None:
                continue

            # cube_data has shape (len(all_band_idxs),); it's the per-pixel
            # mean over the selected band indices.  But we need separate means
            # for numerator and denominator windows.
            # Re-extract per window directly from the downloaded files.
            cal_path = self.cube_cache_dir / f"C{cube_id}_ir.cub"
            nav_path = self.cube_cache_dir / f"N{cube_id}_ir.cub"

            if not cal_path.exists() or not nav_path.exists():
                continue

            try:
                cal = read_isis3_cube(cal_path, band_indices=all_band_idxs)
                nav = read_isis3_cube(nav_path)  # all 6 nav bands
            except Exception as exc:
                logger.warning("Read failed for %s: %s", cube_id, exc)
                n_fail += 1
                continue

            # Build index mapping from all_band_idxs position
            idx_map = {b: i for i, b in enumerate(all_band_idxs)}

            num_pos   = [idx_map[b] for b in num_bands   if b in idx_map]
            denom_pos = [idx_map[b] for b in denom_bands if b in idx_map]

            num_mean   = cal[num_pos,   :, :].mean(axis=0)   # (L, S)
            denom_mean = cal[denom_pos, :, :].mean(axis=0)   # (L, S)

            # Navigation geometry
            nav_lat   = nav[0, :, :]   # deg planetocentric, east-positive
            nav_lon_e = nav[1, :, :]   # deg east-positive, -180 to +180
            nav_emiss = nav[3, :, :]   # emission angle, deg

            # Convert lon east-positive -> west-positive (0->360)
            nav_lon_wp = ((-nav_lon_e) % 360.0).astype(np.float32)

            # Flat pixel arrays
            flat_lat  = nav_lat.ravel()
            flat_lon  = nav_lon_wp.ravel()
            flat_emis = nav_emiss.ravel()
            flat_num  = num_mean.ravel()
            flat_den  = denom_mean.ravel()

            # Quality mask
            mask = (
                (flat_emis <= self.max_emission_deg)
                & np.isfinite(flat_lat)
                & np.isfinite(flat_lon)
                & (flat_den > 1e-6)          # avoid div by zero
                & np.isfinite(flat_num)
                & np.isfinite(flat_den)
                & (flat_lat >= -90) & (flat_lat <= 90)
                & (flat_lon >= 0)   & (flat_lon <= 360)
            )

            if not mask.any():
                continue

            ratio_vals = (flat_num / flat_den).astype(np.float32)

            row_i = np.clip(
                ((90.0 - flat_lat[mask]) / 180.0 * nrows).astype(np.int32),
                0, nrows - 1,
            )
            col_i = np.clip(
                (flat_lon[mask] / 360.0 * ncols).astype(np.int32),
                0, ncols - 1,
            )
            emiss_m = flat_emis[mask]
            ratio_m = ratio_vals[mask]

            better = emiss_m < best_emiss[row_i, col_i]
            mosaic[row_i[better], col_i[better]]     = ratio_m[better]
            best_emiss[row_i[better], col_i[better]] = emiss_m[better]
            n_good += 1

            if n_good % 20 == 0:
                pct = 100.0 * np.isfinite(mosaic).sum() / mosaic.size
                logger.info(
                    "  %.2f/%.2f ratio: %d cubes, %.1f%% coverage",
                    numerator_um, denominator_um, n_good, pct,
                )

        pct = 100.0 * np.isfinite(mosaic).sum() / mosaic.size
        logger.info(
            "%.2f/%.2f ratio mosaic: %d cubes processed, %.1f%% coverage",
            numerator_um, denominator_um, n_good, pct,
        )
        return mosaic

    # -- Private helpers -------------------------------------------------------

    def _extract_cube(
        self,
        cube_id:    str,
        band_idxs:  List[int],
        downloader: "VIMSCubeDownloader",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Download (if needed) and extract per-pixel data for one cube.

        Parameters
        ----------
        cube_id:
            Cube identifier.
        band_idxs:
            0-based band indices to average in the calibrated cube.
        downloader:
            VIMSCubeDownloader for fetching files.

        Returns
        -------
        (vals, lats, lons_wp, emiss) each of shape (N_pixels,)
        or (None, None, None, None) if no pixels pass the emission filter.
        """
        cal_path = self.cube_cache_dir / f"C{cube_id}_ir.cub"
        nav_path = self.cube_cache_dir / f"N{cube_id}_ir.cub"

        # Download if missing
        if not cal_path.exists() or not nav_path.exists():
            files = downloader.download_cube(
                cube_id, download=("calibrated", "navigation"),
            )
            if "calibrated" not in files or "navigation" not in files:
                raise IOError(
                    f"Download incomplete for cube {cube_id}: "
                    f"got {list(files.keys())}"
                )

        # Read calibrated cube -- requested bands only
        cal = read_isis3_cube(cal_path, band_indices=band_idxs)  # (n_bands, L, S)
        # Window mean across bands
        window_mean = cal.mean(axis=0)   # (L, S)

        # Read navigation cube
        nav = read_isis3_cube(nav_path)  # (6, L, S)
        nav_lat   = nav[0, :, :]         # latitude, deg east-positive
        nav_lon_e = nav[1, :, :]         # longitude, deg east-positive -180->+180
        nav_emiss = nav[3, :, :]         # emission angle, deg

        # Convert lon east-positive -> west-positive (0 -> 360 deg)
        # Canonical pipeline uses west-positive longitude
        nav_lon_wp = ((-nav_lon_e) % 360.0).astype(np.float32)

        # Flatten
        flat_val  = window_mean.ravel()
        flat_lat  = nav_lat.ravel()
        flat_lon  = nav_lon_wp.ravel()
        flat_emis = nav_emiss.ravel()

        # Quality mask: emission angle, valid geometry, finite I/F
        mask = (
            (flat_emis  <= self.max_emission_deg)
            & np.isfinite(flat_val)
            & np.isfinite(flat_lat)
            & np.isfinite(flat_lon)
            & (flat_lat >= -90) & (flat_lat <= 90)
            & (flat_lon >= 0)   & (flat_lon <= 360)
            & (flat_val >= 0)   # I/F must be non-negative
        )

        if not mask.any():
            return None, None, None, None

        return (
            flat_val[mask].astype(np.float32),
            flat_lat[mask].astype(np.float32),
            flat_lon[mask].astype(np.float32),
            flat_emis[mask].astype(np.float32),
        )
