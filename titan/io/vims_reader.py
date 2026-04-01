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
titan/io/vims_reader.py
========================
Reader and downloader for Cassini VIMS data from the Nantes portal.

URL scheme (verified by direct inspection of https://vims.univ-nantes.fr):
---------------------------------------------------------------------------
All cubes are downloadable with NO login and NO contact required.

Given a cube ID (format: ``{sclk}_{counter}``, e.g. ``1477222875_1``):

  Raw PDS (.qub):
    https://vims.univ-nantes.fr/cube/v{id}.qub
    -> HTTP 302 redirects to the exact file on PDS-Imaging JPL
    -> same shortcut for .lbl:
    https://vims.univ-nantes.fr/cube/v{id}.lbl

  ISIS3-calibrated IR cube (I/F units):
    https://vims.univ-nantes.fr/cube/C{id}_ir.cub
    (hosted directly on Nantes server; no redirect)

  ISIS3 navigation/geometry cube (per-pixel lat/lon/angles):
    https://vims.univ-nantes.fr/cube/N{id}_ir.cub
    (hosted directly on Nantes server; no redirect)

  Preview JPEGs (all available without authentication):
    https://vims.univ-nantes.fr/data/previews/{band_combo}/{flyby_code}/{id}.jpg

    Where band_combo can be:
      RGB_203_158_279      standard surface composite  (2.03; 1.58; 2.79 umm)
      RGBR_158_128_204_128_128_107  band-ratio RGB (1.58/1.28; 2.04/1.28; 1.28/1.07)
      R_159_126            1.59/1.26 umm ratio  <- tholin proxy used in pipeline
      G_203                2.03 umm surface window
      R_203_210            2.03/2.10 umm ratio
      G_212                2.12 umm  (atmospheric)
      G_101                1.01 umm  (atmospheric)
      G_501                5.0 umm   (deep surface window)
      RGB_501_158_129      5.0; 1.58; 1.29 umm
      RGB_501_275_203      5.0; 2.75; 2.03 umm
      RGB_501_332_322      5.0; 3.32; 3.22 umm
      RGB_231_269_195      strat/tropo/surface  (2.31; 2.69; 1.95 umm)

  Flyby code for each Titan flyby:
    TA -> 00ATI, TB -> 01BTI, T3 -> 03TI, ...
    For simplicity use the flyby column from the parquet index.

Cube index (parquet):
---------------------
The ``id`` column in the VIMS parquet footprint index maps DIRECTLY to
these URLs.  No transformation is needed beyond prepending ``v``, ``C``,
or ``N`` as described above.

ISIS3 calibration pipeline (for reference):
-------------------------------------------
The Nantes-hosted C*.cub files were produced using:
  vims2isis  -> convert .qub to ISIS3 .cub (splits VIS and IR)
  spiceinit  -> inject SPICE geometry (NAIF kernels)
  vimscal    -> calibrate to I/F units (RC19 calibration, solar model)
  campt      -> extract per-pixel lat/lon/phase geometry

The navigation N*.cub files contain the per-pixel geometry bands:
  Band 1: Latitude  (deg, planetocentric, east-positive)
  Band 2: Longitude (deg, east-positive -180->+180)
  Band 3: Incidence angle
  Band 4: Emission angle
  Band 5: Phase angle
  Band 6: Pixel resolution (km/pixel)

The calibrated C*.cub files contain 256 spectral bands in I/F units
covering 0.35-5.12 umm (VIMS-VIS: 0.35-1.04 umm; VIMS-IR: 0.88-5.12 umm).

References
----------
Le Mouelic et al. (2019)  doi:10.1016/j.icarus.2018.09.017
Brown et al. (2004)       doi:10.1007/s11214-004-1453-x
ISIS3 VIMS tutorials:     https://isis.astrogeology.usgs.gov/
VIMS portal:              https://vims.univ-nantes.fr/
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

PORTAL_BASE = "https://vims.univ-nantes.fr"

#: Band combos available as preview JPEGs on the portal.
#: Keys are short descriptive names; values are the URL path component.
PREVIEW_BAND_COMBOS: Dict[str, str] = {
    "surface_rgb":     "RGB_203_158_279",       # 2.03; 1.58; 2.79 umm  (standard)
    "band_ratio_rgb":  "RGBR_158_128_204_128_128_107",  # ratio RGB
    "tholin_ratio":    "R_159_126",             # 1.59/1.26 umm  <- tholin proxy
    "surface_2um":     "G_203",                 # 2.03 umm window
    "surf_atm_ratio":  "R_203_210",             # 2.03/2.10 ratio
    "atm_212":         "G_212",                 # 2.12 umm atmosphere
    "atm_101":         "G_101",                 # 1.01 umm atmosphere
    "deep_surface_5um":"G_501",                 # 5.0 umm deep surface
    "rgb_5_158_129":   "RGB_501_158_129",       # 5.0; 1.58; 1.29 umm
    "rgb_5_275_203":   "RGB_501_275_203",       # 5.0; 2.75; 2.03 umm
    "rgb_5_332_322":   "RGB_501_332_322",       # 5.0; 3.32; 3.22 umm
    "strat_tropo_surf":"RGB_231_269_195",        # 2.31; 2.69; 1.95 umm
}


def cube_url_raw(cube_id: str) -> str:
    """
    Return the portal redirect URL for a raw PDS .qub file.

    The portal returns HTTP 302 -> PDS-Imaging JPL URL.

    Parameters
    ----------
    cube_id:
        Cube identifier in ``{sclk}_{counter}`` format, e.g. ``1477222875_1``.

    Returns
    -------
    str
        URL that resolves (via redirect) to the raw .qub on PDS.

    Examples
    --------
    >>> cube_url_raw("1477222875_1")
    'https://vims.univ-nantes.fr/cube/v1477222875_1.qub'
    """
    return f"{PORTAL_BASE}/cube/v{cube_id}.qub"


def cube_url_label(cube_id: str) -> str:
    """Return portal redirect URL for the PDS .lbl label file."""
    return f"{PORTAL_BASE}/cube/v{cube_id}.lbl"


def cube_url_calibrated(cube_id: str) -> str:
    """
    Return direct URL for the ISIS3-calibrated IR cube (I/F units).

    Hosted on the Nantes server; no redirect.

    Parameters
    ----------
    cube_id:
        Cube ID, e.g. ``1477222875_1``.

    Returns
    -------
    str
        ``https://vims.univ-nantes.fr/cube/C{cube_id}_ir.cub``
    """
    return f"{PORTAL_BASE}/cube/C{cube_id}_ir.cub"


def cube_url_navigation(cube_id: str) -> str:
    """
    Return direct URL for the ISIS3 navigation cube (per-pixel geometry).

    Navigation cube bands:
      1: Latitude ( deg, east-positive)
      2: Longitude ( deg, east-positive, -180->+180)
      3: Incidence angle (deg)
      4: Emission angle (deg)
      5: Phase angle (deg)
      6: Pixel resolution (km/pixel)

    Parameters
    ----------
    cube_id:
        Cube ID, e.g. ``1477222875_1``.

    Returns
    -------
    str
        ``https://vims.univ-nantes.fr/cube/N{cube_id}_ir.cub``
    """
    return f"{PORTAL_BASE}/cube/N{cube_id}_ir.cub"


def cube_url_preview(
    cube_id: str,
    flyby_code: str,
    band_combo: str = "surface_rgb",
) -> str:
    """
    Return direct URL for a preview JPEG image.

    All preview images are served as static files; no authentication needed.

    Parameters
    ----------
    cube_id:
        Cube ID, e.g. ``1477222875_1``.
    flyby_code:
        Internal flyby code as stored in the portal (e.g. ``00ATI`` for TA).
        For Titan flybys TA-T126 the code is visible in the URL on the portal.
    band_combo:
        One of the keys in ``PREVIEW_BAND_COMBOS`` (default ``"surface_rgb"``).

    Returns
    -------
    str
        Full JPEG URL.

    Examples
    --------
    >>> cube_url_preview("1477222875_1", "00ATI", "tholin_ratio")
    'https://vims.univ-nantes.fr/data/previews/R_159_126/00ATI/1477222875_1.jpg'
    """
    band_path = PREVIEW_BAND_COMBOS.get(band_combo, band_combo)
    return f"{PORTAL_BASE}/data/previews/{band_path}/{flyby_code}/{cube_id}.jpg"


# ---------------------------------------------------------------------------
# Cube downloader
# ---------------------------------------------------------------------------

class VIMSCubeDownloader:
    """
    Downloads VIMS cubes from the Nantes portal by cube ID.

    All downloads are direct HTTP GETs; no login or form submission required.
    The portal redirects `.qub` requests to the appropriate PDS-Imaging
    archive location automatically.

    Parameters
    ----------
    dest_dir:
        Local directory to save downloaded files.
    max_retries:
        Number of download retries on network errors.
    """

    def __init__(self, dest_dir: Path, max_retries: int = 3) -> None:
        self.dest_dir    = Path(dest_dir)
        self.max_retries = max_retries
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    def download_cube(
        self,
        cube_id:    str,
        download:   Tuple[str, ...] = ("calibrated", "navigation"),
        overwrite:  bool = False,
    ) -> Dict[str, Path]:
        """
        Download one or more files associated with a VIMS cube ID.

        Parameters
        ----------
        cube_id:
            Cube identifier, e.g. ``"1477222875_1"``.
        download:
            Tuple of file types to download. Options:
              ``"raw"``         -- raw PDS .qub file (large, ~3-50 MB per cube)
              ``"label"``       -- PDS .lbl metadata
              ``"calibrated"``  -- ISIS3 calibrated .cub in I/F units
              ``"navigation"``  -- ISIS3 navigation .cub with lat/lon per pixel
        overwrite:
            Re-download even if local file already exists.

        Returns
        -------
        Dict[str, Path]
            Mapping of file type -> local path for successfully downloaded files.
        """
        url_map = {
            "raw":        (cube_url_raw(cube_id),        f"v{cube_id}.qub"),
            "label":      (cube_url_label(cube_id),      f"v{cube_id}.lbl"),
            "calibrated": (cube_url_calibrated(cube_id), f"C{cube_id}_ir.cub"),
            "navigation": (cube_url_navigation(cube_id), f"N{cube_id}_ir.cub"),
        }

        results: Dict[str, Path] = {}
        for ftype in download:
            if ftype not in url_map:
                logger.warning("Unknown file type '%s' -- skip.", ftype)
                continue
            url, filename = url_map[ftype]
            dest = self.dest_dir / filename
            if dest.exists() and not overwrite:
                results[ftype] = dest
                continue
            try:
                self._fetch(url, dest)
                results[ftype] = dest
            except Exception as exc:
                logger.error("Failed to download %s (%s): %s", cube_id, ftype, exc)

        return results

    def download_preview(
        self,
        cube_id:    str,
        flyby_code: str,
        band_combo: str = "surface_rgb",
        overwrite:  bool = False,
    ) -> Optional[Path]:
        """
        Download a JPEG preview image for a cube.

        Parameters
        ----------
        cube_id:
            Cube identifier.
        flyby_code:
            Internal flyby code (e.g. ``"00ATI"`` for flyby TA).
        band_combo:
            One of the keys in ``PREVIEW_BAND_COMBOS``.
        overwrite:
            Re-download even if file exists.

        Returns
        -------
        Path or None
            Local path of downloaded file, or None on failure.
        """
        url      = cube_url_preview(cube_id, flyby_code, band_combo)
        filename = f"{cube_id}_{band_combo}.jpg"
        dest     = self.dest_dir / "previews" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not overwrite:
            return dest
        try:
            self._fetch(url, dest)
            return dest
        except Exception as exc:
            logger.error("Preview download failed for %s: %s", cube_id, exc)
            return None

    def download_batch(
        self,
        cube_ids:  List[str],
        download:  Tuple[str, ...] = ("calibrated", "navigation"),
        overwrite: bool = False,
        max_cubes: Optional[int] = None,
    ) -> Dict[str, Dict[str, Path]]:
        """
        Download multiple cubes in sequence.

        Parameters
        ----------
        cube_ids:
            List of cube IDs to download.
        download:
            File types to fetch per cube.
        overwrite:
            Overwrite existing files.
        max_cubes:
            Limit to this many cubes (useful for testing).

        Returns
        -------
        Dict[str, Dict[str, Path]]
            Mapping of cube_id -> {file_type -> local path}.
        """
        try:
            from tqdm import tqdm
            ids = list(cube_ids)[:max_cubes] if max_cubes else list(cube_ids)
            results = {}
            for cube_id in tqdm(ids, desc="Downloading VIMS cubes"):
                results[cube_id] = self.download_cube(
                    cube_id, download=download, overwrite=overwrite
                )
            return results
        except ImportError:
            ids = list(cube_ids)[:max_cubes] if max_cubes else list(cube_ids)
            results = {}
            for i, cube_id in enumerate(ids):
                logger.info("Downloading cube %d/%d: %s", i+1, len(ids), cube_id)
                results[cube_id] = self.download_cube(
                    cube_id, download=download, overwrite=overwrite
                )
            return results

    def _fetch(self, url: str, dest: Path) -> None:
        """Download a URL to dest, following redirects (handles PDS 302s)."""
        import requests

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")

        for attempt in range(self.max_retries):
            try:
                # Browser-like User-Agent required by some CDN/servers
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                }
                with requests.get(url, headers=headers, stream=True,
                                  timeout=120, allow_redirects=True) as resp:
                    resp.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 16):
                            fh.write(chunk)
                shutil.move(str(tmp), str(dest))
                logger.debug("Downloaded %s -> %s", url, dest)
                return
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt + 1, self.max_retries, url, exc,
                    )
                else:
                    if tmp.exists():
                        tmp.unlink()
                    raise


# ---------------------------------------------------------------------------
# Navigation cube reader (lat/lon per pixel)
# ---------------------------------------------------------------------------

def read_navigation_cube(nav_path: Path) -> Dict[str, np.ndarray]:
    """
    Read an ISIS3 navigation cube and return per-pixel geometry arrays.

    The navigation cube (N{id}_ir.cub) contains per-pixel geometry
    computed from SPICE kernels during the ISIS3 calibration pipeline:

    Band layout (1-indexed, as produced by vims.univ-nantes.fr pipeline):
      Band 1: Latitude      ( deg, east-positive, planetocentric)
      Band 2: Longitude     ( deg, east-positive, -180->+180)
      Band 3: Incidence     (deg)
      Band 4: Emission      (deg)
      Band 5: Phase         (deg)
      Band 6: Resolution    (km/pixel)

    Parameters
    ----------
    nav_path:
        Path to the ISIS3 navigation .cub file.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: ``"lat"``, ``"lon_east"``, ``"lon_west"``,
        ``"incidence"``, ``"emission"``, ``"phase"``, ``"resolution"``

    Notes
    -----
    Longitude in the output is provided in BOTH conventions:
      - ``lon_east``: east-positive (as in the ISIS3 cube, -180->+180)
      - ``lon_west``: west-positive (0->360, matching pipeline convention)
    """
    try:
        import pvl
    except ImportError:
        pvl = None

    # Try reading as ISIS3 cube via GDAL
    try:
        import rasterio
        with rasterio.open(nav_path) as src:
            n_bands = src.count
            bands   = [src.read(i + 1).astype(np.float32) for i in range(n_bands)]
    except Exception:
        # Fallback: read raw binary assuming ISIS3 tile layout is not available
        logger.error(
            "Could not read ISIS3 cube %s. "
            "Install GDAL with ISIS3 support, or use SpiceyPy directly.",
            nav_path,
        )
        raise

    def _get_band(idx: int) -> np.ndarray:
        if idx < len(bands):
            arr = bands[idx]
            # ISIS3 uses -1e32 as nodata
            arr[arr < -1e30] = np.nan
            return arr
        return np.full_like(bands[0], np.nan)

    lat     = _get_band(0)
    lon_e   = _get_band(1)  # east-positive
    lon_w   = (-lon_e) % 360.0   # convert to west-positive 0->360

    return {
        "lat":        lat,
        "lon_east":   lon_e,
        "lon_west":   lon_w.astype(np.float32),
        "incidence":  _get_band(2),
        "emission":   _get_band(3),
        "phase":      _get_band(4),
        "resolution": _get_band(5),
    }


# ---------------------------------------------------------------------------
# Parquet spatial index (unchanged from original, documented with real schema)
# ---------------------------------------------------------------------------

VIMS_COLUMNS = {
    "id":        "Cube ID  (SCLK format '{sclk}_{counter}', e.g. '1477222875_1')",
    "flyby":     "Cassini flyby name (TA, T001-T126)",
    "obs_start": "Observation start date",
    "obs_end":   "Observation end date",
    "altitude":  "Cassini altitude at observation (km)",
    "lon":       "Titan surface longitude, WEST-positive, 0->360 deg",
    "lat":       "Titan surface latitude, deg (-90 to +90)",
    "res":       "Spatial resolution (km/pixel)",
}


class VIMSFootprintIndex:
    """
    In-memory index of VIMS pixel footprints for spatial queries.

    Loads the Nantes parquet footprint catalogue once.  The ``id`` column
    maps directly to cube download URLs via the helper functions above.

    Parameters
    ----------
    parquet_path:
        Path to the VIMS footprint parquet file.
    max_resolution_km:
        Only load footprints with ``res <= max_resolution_km``.
    """

    def __init__(
        self,
        parquet_path: Path,
        max_resolution_km: Optional[float] = None,
    ) -> None:
        self.parquet_path      = Path(parquet_path)
        self.max_resolution_km = max_resolution_km
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Load the parquet index into memory."""
        logger.info("Loading VIMS footprint index from %s ...", self.parquet_path)
        df = pd.read_parquet(self.parquet_path)

        expected = set(VIMS_COLUMNS.keys())
        missing  = expected - set(df.columns)
        if missing:
            raise ValueError(
                f"VIMS parquet missing expected columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        if self.max_resolution_km is not None:
            before = len(df)
            df = df[df["res"] <= self.max_resolution_km].copy()
            logger.info(
                "Filtered to res <= %.1f km/px: %d -> %d rows",
                self.max_resolution_km, before, len(df),
            )

        self._df = df.reset_index(drop=True)
        logger.info(
            "VIMS index loaded: %d footprints, %d unique flybys",
            len(self._df), self._df["flyby"].nunique(),
        )

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df

    def cubes_covering_region(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        max_resolution_km: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Return cubes whose footprints intersect a lon/lat bounding box.

        Useful for targeted cube downloads: identify which cube IDs cover
        a region of interest (e.g. Selk crater, Ontario Lacus) before
        downloading the corresponding calibrated .cub files.

        Parameters
        ----------
        lon_min, lon_max:
            Longitude bounds, west-positive (0->360 deg).
        lat_min, lat_max:
            Latitude bounds (-90->+90 deg).
        max_resolution_km:
            Optional resolution filter in addition to the spatial filter.

        Returns
        -------
        pd.DataFrame
            Subset of the footprint index, sorted by ascending resolution.
        """
        df = self.df
        mask = (
            (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
            & (df["lat"] >= lat_min) & (df["lat"] <= lat_max)
        )
        if max_resolution_km is not None:
            mask &= df["res"] <= max_resolution_km

        result = df[mask].drop_duplicates(subset=["id"]).sort_values("res")
        logger.info(
            "Found %d unique cubes covering lon=[%.1f,%.1f], lat=[%.1f,%.1f]",
            len(result), lon_min, lon_max, lat_min, lat_max,
        )
        return result

    def get_download_urls(self, cube_id: str) -> Dict[str, str]:
        """
        Return all download URLs for a given cube ID.

        Parameters
        ----------
        cube_id:
            Cube identifier from the ``id`` column.

        Returns
        -------
        Dict[str, str]
            ``{"raw": url, "label": url, "calibrated": url, "navigation": url}``
        """
        return {
            "raw":        cube_url_raw(cube_id),
            "label":      cube_url_label(cube_id),
            "calibrated": cube_url_calibrated(cube_id),
            "navigation": cube_url_navigation(cube_id),
        }

    def coverage_map(
        self,
        nrows: int,
        ncols: int,
        lon_range: Tuple[float, float] = (0.0, 360.0),
        lat_range: Tuple[float, float] = (-90.0, 90.0),
    ) -> np.ndarray:
        """
        Build a 2-D coverage density map (footprint count per pixel).

        Returns float32 array normalised to [0, 1], north-up (row 0 = +90 deg).
        """
        df       = self.df
        lon_bins = np.linspace(lon_range[0], lon_range[1], ncols + 1)
        # np.histogram2d requires ASCENDING bins -- use south-to-north,
        # then flip the result so row 0 is north (+90 deg).
        lat_bins = np.linspace(lat_range[0], lat_range[1], nrows + 1)

        counts, _, _ = np.histogram2d(
            df["lat"].values, df["lon"].values,
            bins=[lat_bins, lon_bins],
        )
        counts = counts[::-1]   # flip south->north to north->south (row 0 = +90 deg)
        max_count = counts.max()
        if max_count > 0:
            counts /= max_count
        return counts.astype(np.float32)

    def best_resolution_map(
        self,
        nrows: int,
        ncols: int,
        lon_range: Tuple[float, float] = (0.0, 360.0),
        lat_range: Tuple[float, float] = (-90.0, 90.0),
    ) -> np.ndarray:
        """
        Build a map of the best (minimum) VIMS resolution per pixel (km/px).

        NaN where no coverage. North-up (row 0 = +90 deg).
        """
        df        = self.df
        lon_edges = np.linspace(lon_range[0], lon_range[1], ncols + 1)
        # Use ascending lat edges for searchsorted; convert to north-up row index.
        lat_edges = np.linspace(lat_range[0], lat_range[1], nrows + 1)

        col_idx = np.clip(
            np.searchsorted(lon_edges, df["lon"].values, side="right") - 1,
            0, ncols - 1,
        )
        # searchsorted on ascending edges gives south-first index (0 = southernmost).
        # Flip to north-up: row 0 = highest latitude.
        south_idx = np.clip(
            np.searchsorted(lat_edges, df["lat"].values, side="right") - 1,
            0, nrows - 1,
        )
        row_idx = (nrows - 1) - south_idx

        # Initialise with +inf (not NaN): np.minimum(NaN, x) = NaN, which
        # would silently leave every pixel as NaN.  Replace inf->NaN after.
        result   = np.full((nrows, ncols), np.inf, dtype=np.float32)
        res_vals = df["res"].values
        np.minimum.at(result, (row_idx, col_idx), res_vals)
        result[np.isinf(result)] = np.nan   # pixels with no coverage -> NaN
        return result

    def flyby_count_map(
        self,
        nrows: int,
        ncols: int,
        lon_range: Tuple[float, float] = (0.0, 360.0),
        lat_range: Tuple[float, float] = (-90.0, 90.0),
    ) -> np.ndarray:
        """Count distinct flybys per pixel. Returns int16 array, north-up."""
        df        = self.df
        lon_edges = np.linspace(lon_range[0], lon_range[1], ncols + 1)
        # Ascending lat edges; convert to north-up row index after searchsorted.
        lat_edges = np.linspace(lat_range[0], lat_range[1], nrows + 1)

        col_idx = np.clip(
            np.searchsorted(lon_edges, df["lon"].values, side="right") - 1,
            0, ncols - 1,
        )
        south_idx = np.clip(
            np.searchsorted(lat_edges, df["lat"].values, side="right") - 1,
            0, nrows - 1,
        )
        row_idx = (nrows - 1) - south_idx

        df_tmp = pd.DataFrame({"row": row_idx, "col": col_idx,
                               "flyby": df["flyby"].values})
        counts = (
            df_tmp.drop_duplicates()
            .groupby(["row", "col"]).size()
            .reset_index(name="n_flybys")
        )
        result = np.zeros((nrows, ncols), dtype=np.int16)
        result[counts["row"].values, counts["col"].values] = \
            counts["n_flybys"].values.astype(np.int16)
        return result

    def summary(self) -> str:
        df = self.df
        return (
            f"VIMSFootprintIndex: {len(df):,} footprints, "
            f"{df['flyby'].nunique()} flybys, "
            f"res [{df['res'].min():.1f}-{df['res'].max():.1f}] km/px"
        )
