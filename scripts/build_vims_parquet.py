#!/usr/bin/env python3
"""
scripts/build_vims_parquet.py
==============================
Build ``data/raw/vims_footprints.parquet`` from scratch using the public
Cassini VIMS portal at https://vims.univ-nantes.fr.

No login, no contact, no pre-existing data required.

What this script does
---------------------
Stage 1 — FAST  (~30 min,  ~15 000 HTTP requests, fully resumable)
    Scrapes the Nantes portal for every Titan flyby cube.
    For each cube fetches:
      • cube ID, flyby, observation mid-time  (JSON API /api/cube/{id})
      • mean resolution km/px, sub-spacecraft lat/lon, distance km
        (HTML page /cube/{id})
    Writes a per-CUBE parquet: data/raw/vims_footprints_cubes.parquet
    This alone is sufficient for the current pipeline (select_cube_ids
    filters cubes by median resolution; geo_only mode is unaffected).

Stage 2 — SLOW  (~4–8 h,  ~3 000 nav-cube downloads, resumable)  [OPTIONAL]
    Downloads the ISIS3 navigation cube (N*_ir.cub) for cubes that pass
    the resolution filter (median_res <= max_resolution_km).
    Extracts per-pixel lat, lon, emission angle and resolution.
    Writes the full per-PIXEL parquet: data/raw/vims_footprints.parquet
    (~16 M rows).  Only needed if you want the detailed VIMS coverage map
    or plan to switch ORGANIC_SOURCE_MODE back to "blended".

Usage
-----
    # Stage 1 only (recommended for most uses):
    python scripts/build_vims_parquet.py

    # Stage 1 + Stage 2 (full pixel-level rebuild):
    python scripts/build_vims_parquet.py --full

    # Limit to specific flybys (useful for testing):
    python scripts/build_vims_parquet.py --flybys TA T3 T4

    # Adjust resolution threshold (default 25 km/px):
    python scripts/build_vims_parquet.py --full --max-res 50

    # Resume an interrupted run — just rerun the same command:
    python scripts/build_vims_parquet.py

Scientific notes
----------------
* Resolution filtering: the pipeline selects up to 200 cubes with
  median_res <= 25 km/px to build the VIMS 5.0/2.03 µm ratio mosaic
  (Seignovert et al. 2019 tholin proxy).  Coarser cubes are kept in the
  parquet but not used for mosaicking.

* Longitude convention: the parquet uses WEST-POSITIVE longitude (0–360°W)
  consistent with the pipeline canonical grid.  The Nantes portal uses
  east-positive longitude (-180–+180°E).
  Conversion: lon_W = (-lon_E) % 360

* The per-cube "Mean resolution" from the portal is the mean pixel scale
  over the full cube footprint.  This is used as the single row's ``res``
  value in Stage 1; Stage 2 computes per-pixel values from the nav cube.

* Date source: the portal gives only the observation MID-TIME per cube.
  ``obs_start`` and ``obs_end`` are both set to mid-time in Stage 1.

License: CC-BY-4.0 — NASA/Caltech-JPL/University of Arizona/
         Osuna-CNRS-Nantes Université
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import html as _html
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PORTAL_BASE     = "https://vims.univ-nantes.fr"

# All Titan targeted flybys in chronological order (TA = Oct 2004, T126 = Apr 2017)
TITAN_FLYBYS: List[str] = (
    ["TA", "TB", "TC"] + [f"T{n}" for n in range(3, 127)]
)

# Regex patterns applied to STRIPPED (tag-free) page text.
# The portal renders values in separate <td> tags from their <th> labels,
# so raw-HTML regex matching fails.  _strip_html() removes all tags first.
_RE_CUBE_IDS   = re.compile(r"\b(\d{10}_\d+)\b")
_RE_RESOLUTION = re.compile(r"Mean resolution\s+([\d,]+)\s+km/pixel")
_RE_SUB_SC     = re.compile(
    r"Sub-Spacecraft point\s+([-\d]+)\s*\u00b0\s*N\s*\|\s*([-\d]+)\s*\u00b0\s*E"
)
_RE_DISTANCE   = re.compile(r"Distance\s+([\d,]+)\s+km")
_RE_TAGS       = re.compile(r"<[^>]+>")


def _strip_html(raw_html: str) -> str:
    """Strip HTML tags, decode entities (e.g. &deg; -> °), collapse whitespace.
    The portal serves degree symbols as &deg; entities which requests fetches
    verbatim; html.unescape() converts them before regex matching."""
    return _html.unescape(_RE_TAGS.sub(" ", raw_html))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vims_parquet")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get_text(url: str, session, retries: int = 4, backoff: float = 2.0) -> str:
    """HTTP GET with exponential backoff; returns response text."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = 'utf-8'  # force UTF-8; server omits charset in Content-Type header
            return r.text
        except Exception as exc:
            if attempt == retries - 1:
                raise
            delay = backoff ** attempt
            log.warning("  retry %d/%d %s  (%s, wait %.1fs)",
                        attempt + 1, retries, url, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def _get_json(url: str, session) -> object:
    return json.loads(_get_text(url, session))


# ---------------------------------------------------------------------------
# Stage 1 — per-cube scraper
# ---------------------------------------------------------------------------
def _cube_ids_for_flyby(flyby: str, session) -> List[str]:
    """Scrape all cube IDs from the flyby HTML page."""
    html = _get_text(f"{PORTAL_BASE}/flyby/{flyby}", session)
    ids  = _RE_CUBE_IDS.findall(html)
    return list(dict.fromkeys(ids))   # deduplicate, preserve order


def _fetch_cube_meta(cube_id: str, flyby: str, session) -> Optional[Dict]:
    """
    Fetch one cube's metadata from the API + HTML page.

    Returns a dict with keys matching VIMS_COLUMNS:
        id, flyby, obs_start, obs_end, altitude, lon, lat, res
    Returns None if the cube cannot be fetched.
    """
    # --- JSON API: mid-time ---
    mid_iso: Optional[str] = None
    try:
        records  = _get_json(f"{PORTAL_BASE}/api/cube/{cube_id}", session)
        mid_iso  = records[0].get("time") if records else None
    except Exception:
        pass   # mid-time will be None; not fatal

    # --- HTML page: resolution, sub-spacecraft point, distance ---
    try:
        raw_html = _get_text(f"{PORTAL_BASE}/cube/{cube_id}", session)
    except Exception as exc:
        log.debug("  HTML fetch failed %s: %s", cube_id, exc)
        return None

    # Strip HTML tags so regexes can match label+value across <th>/<td> boundaries
    html = _strip_html(raw_html)

    def _parse_int(m: Optional[re.Match], g: int = 1) -> Optional[int]:
        return int(m.group(g).replace(",", "")) if m else None

    res_m  = _RE_RESOLUTION.search(html)
    sub_m  = _RE_SUB_SC.search(html)
    dist_m = _RE_DISTANCE.search(html)

    res_km   = _parse_int(res_m)
    lat_deg  = _parse_int(sub_m, 1)
    lon_e    = _parse_int(sub_m, 2)    # east-positive
    dist_km  = _parse_int(dist_m)

    # Convert east-positive to west-positive 0–360°W
    lon_w = float((-lon_e) % 360) if lon_e is not None else float("nan")

    return {
        "id":        cube_id,
        "flyby":     flyby,
        "obs_start": mid_iso or "",
        "obs_end":   mid_iso or "",
        "altitude":  float(dist_km)  if dist_km  is not None else float("nan"),
        "lon":       lon_w,
        "lat":       float(lat_deg)  if lat_deg  is not None else float("nan"),
        "res":       float(res_km)   if res_km   is not None else float("nan"),
    }


def build_stage1(
    flybys: List[str],
    out_path: Path,
    session,
    rate_limit: float = 0.25,
) -> None:
    """
    Scrape per-cube metadata for all specified flybys and write a parquet.
    Resumes if out_path already exists.
    """
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume — skip any cube already present in the parquet.
    # Use repair_vims_parquet.py to fix rows with NaN lat/lon/res.
    done: set  = set()
    rows: list = []
    if out_path.exists():
        df0   = pd.read_parquet(out_path)
        done  = set(df0["id"].tolist())
        rows  = df0.to_dict("records")
        log.info("Resuming Stage 1: %d cubes already saved", len(done))

    for fi, flyby in enumerate(flybys, 1):
        log.info("[flyby %d/%d] %s", fi, len(flybys), flyby)
        try:
            ids = _cube_ids_for_flyby(flyby, session)
        except Exception as exc:
            log.warning("  Could not scrape %s: %s", flyby, exc)
            continue

        new_ids = [c for c in ids if c not in done]
        log.info("  %d cubes, %d new", len(ids), len(new_ids))

        for ci, cid in enumerate(new_ids, 1):
            time.sleep(rate_limit)
            try:
                meta = _fetch_cube_meta(cid, flyby, session)
            except Exception as exc:
                log.warning("  [%d/%d] %s FAILED: %s", ci, len(new_ids), cid, exc)
                continue
            if meta is None:
                continue
            rows.append(meta)
            done.add(cid)

            if ci % 50 == 0 or ci == len(new_ids):
                log.info("  [%d/%d] %s  res=%.0f km/px",
                         ci, len(new_ids), cid, meta["res"])
                pd.DataFrame(rows).to_parquet(out_path, index=False)

        pd.DataFrame(rows).to_parquet(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    log.info("Stage 1 done: %d cubes → %s", len(df), out_path)
    log.info("Resolution summary (km/px):\n%s", df["res"].describe().to_string())


# ---------------------------------------------------------------------------
# Stage 2 — per-pixel nav-cube downloader
# ---------------------------------------------------------------------------
def _read_nav_pixels(
    nav_path: Path,
    cube_res_km: float,
    max_emission: float = 70.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Read per-pixel lat and lon (west-positive) from an ISIS3 nav cube.

    Nav cube band layout (Nantes portal documentation):
        Band 1: Latitude   (deg, planetocentric, east-positive)
        Band 2: Longitude  (deg, east-positive, -180 to +180)
        Band 3: Incidence angle
        Band 4: Emission angle

    Returns (lat_array, lon_W_array) for valid pixels, or None.
    """
    try:
        pipeline_root = Path(__file__).parents[1]
        if str(pipeline_root) not in sys.path:
            sys.path.insert(0, str(pipeline_root))
        from titan.io.vims_cube_mosaic import read_isis3_cube

        data = read_isis3_cube(nav_path, band_indices=[0, 1, 3])
        if data is None or data.shape[0] < 3:
            return None

        lat   = data[0].ravel().astype(np.float32)
        lon_e = data[1].ravel().astype(np.float32)
        emis  = data[2].ravel().astype(np.float32)
        lon_w = (-lon_e) % 360.0

        valid = (
            np.isfinite(lat) & np.isfinite(lon_w) & np.isfinite(emis)
            & (np.abs(lat) <= 90.0) & (emis <= max_emission)
        )
        return (lat[valid], lon_w[valid]) if np.any(valid) else None

    except Exception as exc:
        log.debug("  nav cube read error %s: %s", nav_path, exc)
        return None


def validate_parquet(
    parquet_path: Path,
    session,
    n_sample_cubes: int = 10,
    n_sample_flybys: int = 5,
    titan_flybys: Optional[List[str]] = None,
) -> bool:
    """
    Validate an existing vims_footprints.parquet against the live Nantes portal.

    Checks performed
    ----------------
    1. Schema: all 8 required columns present with correct dtypes.
    2. Physical: lat in [-90, 90], lon in [0, 360], res > 0, altitude >= 0.
    3. Portal cross-check: sample ``n_sample_cubes`` cube IDs from the parquet
       and verify each exists on the portal via /api/cube/{id}.
    4. Coverage: for ``n_sample_flybys`` Titan flybys, compare the number of
       cubes in the parquet vs the number listed on the portal.
    5. Summary statistics: row count, unique cube count, flyby list.

    Returns
    -------
    True if all checks pass, False if any check fails.
    """
    import pandas as pd
    import random

    if not parquet_path.exists():
        log.error("Parquet not found: %s", parquet_path)
        return False

    log.info("=" * 60)
    log.info("VALIDATE: %s", parquet_path)
    log.info("=" * 60)

    # -------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        log.error("Cannot read parquet: %s", exc)
        return False

    log.info("Rows: %d  |  Columns: %s", len(df), list(df.columns))

    passes: List[str] = []
    fails:  List[str] = []

    def _pass(msg: str) -> None:
        passes.append(msg)
        log.info("  ✓  %s", msg)

    def _fail(msg: str) -> None:
        fails.append(msg)
        log.warning("  ✗  %s", msg)

    # -------------------------------------------------------------------
    # Check 1: Schema
    # -------------------------------------------------------------------
    REQUIRED = {
        "id":        ("object", "string"),
        "flyby":     ("object", "string"),
        "obs_start": ("object", "string", "datetime64[ns]"),
        "obs_end":   ("object", "string", "datetime64[ns]"),
        "altitude":  ("float32", "float64"),
        "lon":       ("float32", "float64"),
        "lat":       ("float32", "float64"),
        "res":       ("float32", "float64"),
    }
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        _fail(f"Missing columns: {missing_cols}")
    else:
        _pass("All 8 required columns present")

    wrong_types = []
    for col, allowed in REQUIRED.items():
        if col in df.columns:
            dtype_str = str(df[col].dtype)
            if not any(a in dtype_str for a in allowed):
                wrong_types.append(f"{col}:{dtype_str}")
    if wrong_types:
        _fail(f"Unexpected column dtypes: {wrong_types}")
    else:
        _pass("All column dtypes acceptable")

    # -------------------------------------------------------------------
    # Check 2: Physical validity
    # -------------------------------------------------------------------
    if "lat" in df.columns:
        bad_lat = (~df["lat"].between(-90.0, 90.0)).sum()
        if bad_lat == 0:
            _pass("lat ∈ [−90, +90]")
        else:
            _fail(f"lat out of range: {bad_lat} rows")

    if "lon" in df.columns:
        bad_lon = (~df["lon"].between(0.0, 360.0)).sum()
        if bad_lon == 0:
            _pass("lon ∈ [0, 360] (west-positive)")
        else:
            _fail(f"lon out of range: {bad_lon} rows")

    if "res" in df.columns:
        bad_res = (df["res"] <= 0).sum()
        if bad_res == 0:
            _pass("res > 0 for all rows")
        else:
            _fail(f"res ≤ 0: {bad_res} rows")

    if "altitude" in df.columns:
        bad_alt = (df["altitude"] < 0).sum()
        if bad_alt == 0:
            _pass("altitude ≥ 0 for all rows")
        else:
            _fail(f"altitude < 0: {bad_alt} rows")

    # -------------------------------------------------------------------
    # Check 3: Portal cross-check — sample cube IDs
    # -------------------------------------------------------------------
    unique_ids = list(df["id"].unique()) if "id" in df.columns else []
    if unique_ids:
        sample_ids = random.sample(
            unique_ids, min(n_sample_cubes, len(unique_ids))
        )
        log.info("Portal cross-check: %d sampled cube IDs ...", len(sample_ids))
        ok_count = 0
        for cid in sample_ids:
            try:
                time.sleep(0.3)
                records = _get_json(
                    f"{PORTAL_BASE}/api/cube/{cid}", session
                )
                if records and records[0].get("name") == cid:
                    ok_count += 1
                    log.debug("  ✓ %s  target=%s  flyby=%s",
                              cid, records[0].get("target"),
                              records[0].get("flyby"))
                else:
                    log.warning("  ✗ %s not found on portal", cid)
            except Exception as exc:
                log.warning("  ✗ %s portal error: %s", cid, exc)

        if ok_count == len(sample_ids):
            _pass(f"Portal cross-check: {ok_count}/{len(sample_ids)} cube IDs verified")
        elif ok_count >= len(sample_ids) * 0.8:
            _pass(
                f"Portal cross-check: {ok_count}/{len(sample_ids)} cube IDs found "
                f"(≥80% pass — acceptable; some cubes may have been removed)"
            )
        else:
            _fail(
                f"Portal cross-check: only {ok_count}/{len(sample_ids)} cube IDs "
                f"found on portal — parquet may be from a different source"
            )

    # -------------------------------------------------------------------
    # Check 4: Coverage — compare per-flyby cube counts for Titan flybys
    # -------------------------------------------------------------------
    all_flybys = titan_flybys or TITAN_FLYBYS
    if "flyby" in df.columns:
        parquet_flybys = set(df["flyby"].unique())
        titan_in_parquet = [f for f in all_flybys if f in parquet_flybys]
        log.info(
            "Titan flybys in parquet: %d / %d  (parquet may also contain "
            "non-Titan observations)",
            len(titan_in_parquet), len(all_flybys),
        )

        # Sample a few flybys and compare cube counts
        sample_flybys = random.sample(
            titan_in_parquet, min(n_sample_flybys, len(titan_in_parquet))
        )
        count_diffs: List[str] = []
        for flyby in sample_flybys:
            time.sleep(0.3)
            try:
                portal_ids = set(_cube_ids_for_flyby(flyby, session))
                parquet_ids = set(
                    df.loc[df["flyby"] == flyby, "id"].unique()
                )
                overlap = len(portal_ids & parquet_ids)
                portal_only = len(portal_ids - parquet_ids)
                parquet_only = len(parquet_ids - portal_ids)
                log.info(
                    "  %-6s  portal=%d  parquet=%d  overlap=%d  "
                    "portal-only=%d  parquet-only=%d",
                    flyby, len(portal_ids), len(parquet_ids),
                    overlap, portal_only, parquet_only,
                )
                if overlap < len(portal_ids) * 0.5:
                    count_diffs.append(
                        f"{flyby}: only {overlap}/{len(portal_ids)} "
                        f"portal cubes in parquet"
                    )
            except Exception as exc:
                log.warning("  Coverage check failed for %s: %s", flyby, exc)

        if not count_diffs:
            _pass(
                f"Coverage check: {len(sample_flybys)} sampled flybys "
                f"have ≥50% overlap with portal"
            )
        else:
            _fail(f"Coverage gaps: {count_diffs}")

    # -------------------------------------------------------------------
    # Check 5: Summary stats
    # -------------------------------------------------------------------
    n_cubes  = df["id"].nunique()   if "id"    in df.columns else "?"
    n_flybys = df["flyby"].nunique() if "flyby" in df.columns else "?"
    if "res" in df.columns:
        is_per_pixel = len(df) > n_cubes * 2 if isinstance(n_cubes, int) else False
        data_type = "per-pixel" if is_per_pixel else "per-cube"
    else:
        data_type = "unknown"

    log.info("-" * 60)
    log.info("Summary:")
    log.info("  Rows          : %d", len(df))
    log.info("  Unique cubes  : %s", n_cubes)
    log.info("  Unique flybys : %s", n_flybys)
    log.info("  Data type     : %s", data_type)
    if "res" in df.columns:
        log.info(
            "  res (km/px)   : min=%.0f  median=%.0f  max=%.0f",
            df["res"].min(), df["res"].median(), df["res"].max(),
        )
    log.info("-" * 60)

    # -------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------
    log.info("Passed: %d  |  Failed: %d", len(passes), len(fails))
    if fails:
        log.warning("VALIDATION FAILED:")
        for f in fails:
            log.warning("  ✗ %s", f)
        return False
    else:
        log.info("VALIDATION PASSED — parquet is consistent with the portal")
        return True


def build_stage2(
    stage1_path: Path,
    out_path: Path,
    cube_cache: Path,
    session,
    max_res_km: float = 25.0,
    rate_limit:  float = 1.0,
) -> None:
    """
    Download navigation cubes for resolution-qualifying cubes and build
    the full per-pixel parquet.  Resumes from out_path if present.
    """
    import pandas as pd

    if not stage1_path.exists():
        raise FileNotFoundError(
            f"Stage 1 parquet not found: {stage1_path}\n"
            "Run Stage 1 first (omit --full flag)."
        )

    df1     = pd.read_parquet(stage1_path)
    targets = df1[df1["res"] <= max_res_km].copy()
    log.info("Stage 2: %d / %d cubes pass res <= %.0f km/px",
             len(targets), len(df1), max_res_km)

    # Resume
    done: set  = set()
    rows: list = []
    if out_path.exists():
        df0  = pd.read_parquet(out_path)
        done = set(df0["id"].unique().tolist())
        rows = df0.to_dict("records")
        log.info("Resuming Stage 2: %d cubes already processed", len(done))

    cube_cache.mkdir(parents=True, exist_ok=True)

    for ri, (_, cube) in enumerate(targets.iterrows(), 1):
        cid   = str(cube["id"])
        flyby = str(cube["flyby"])
        res   = float(cube["res"])
        ts    = str(cube.get("obs_start", ""))

        if cid in done:
            continue

        log.info("[%d/%d] %s  flyby=%s  res=%.0f km/px",
                 ri, len(targets), cid, flyby, res)

        # Download navigation cube
        nav_url   = f"{PORTAL_BASE}/cube/N{cid}_ir.cub"
        nav_cache = cube_cache / f"N{cid}_ir.cub"

        if not nav_cache.exists():
            time.sleep(rate_limit)
            try:
                r = session.get(nav_url, timeout=120, stream=True)
                r.raise_for_status()
                with open(nav_cache, "wb") as fh:
                    for chunk in r.iter_content(65536):
                        fh.write(chunk)
                log.debug("  saved %s (%.2f MB)",
                          nav_cache.name, nav_cache.stat().st_size / 1e6)
            except Exception as exc:
                log.warning("  download failed for %s: %s", cid, exc)
                continue

        result = _read_nav_pixels(nav_cache, cube_res_km=res)
        if result is None:
            log.debug("  no valid pixels in nav cube for %s", cid)
            done.add(cid)
            continue

        lat_px, lon_px = result
        alt = float(cube.get("altitude", float("nan")))
        for j in range(len(lat_px)):
            rows.append({
                "id":        cid,
                "flyby":     flyby,
                "obs_start": ts,
                "obs_end":   ts,
                "altitude":  alt,
                "lon":       float(lon_px[j]),
                "lat":       float(lat_px[j]),
                "res":       res,
            })
        done.add(cid)

        if ri % 20 == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            log.info("  checkpoint: %d pixel rows", len(rows))

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    log.info("Stage 2 done: %d pixel rows from %d cubes → %s",
             len(df), df["id"].nunique(), out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Build vims_footprints.parquet from the Nantes VIMS portal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/build_vims_parquet.py             # Stage 1 only\n"
            "  python scripts/build_vims_parquet.py --full      # Stages 1 + 2\n"
            "  python scripts/build_vims_parquet.py --flybys TA T3 T4  # test run\n"
        ),
    )
    p.add_argument("--full", action="store_true",
                   help="Also run Stage 2 (download nav cubes, build per-pixel parquet).")
    p.add_argument("--validate", action="store_true",
                   help="Validate an existing parquet against the live portal and exit.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the automatic post-build validation step.")
    p.add_argument("--flybys", nargs="+", metavar="FLYBY",
                   help="Only process these flyby names (default: all 127).")
    p.add_argument("--max-res", type=float, default=25.0,
                   help="Resolution threshold km/px for Stage 2 (default: 25).")
    p.add_argument("--rate-limit", type=float, default=0.25,
                   help="Seconds between Stage 1 requests (default: 0.25).")
    p.add_argument("--output-dir", type=Path, default=Path("data/raw"),
                   help="Output directory for parquet files (default: data/raw).")
    args = p.parse_args()

    # Check dependencies
    try:
        import requests   # noqa: F401
        import pandas     # noqa: F401
        import pyarrow    # noqa: F401
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}\n"
                 "Install with: pip install requests pandas pyarrow")

    import requests as req

    flybys      = args.flybys or TITAN_FLYBYS
    stage1_path = args.output_dir / "vims_footprints_cubes.parquet"
    stage2_path = args.output_dir / "vims_footprints.parquet"
    cube_cache  = args.output_dir / "vims_cubes"

    log.info("=" * 60)
    log.info("VIMS parquet builder  —  Nantes portal scraper")
    log.info("Portal : %s", PORTAL_BASE)
    log.info("Flybys : %d  (%s … %s)", len(flybys), flybys[0], flybys[-1])
    log.info("Full   : %s  (Stage 2 nav-cube download)", args.full)
    log.info("=" * 60)

    session = req.Session()
    session.headers["User-Agent"] = (
        "TitanHabitabilityPipeline/4 "
        "(academic; see github for contact)"
    )

    # ------------------------------------------------------------------
    # Validate-only mode
    # ------------------------------------------------------------------
    if args.validate:
        target = args.output_dir / "vims_footprints.parquet"
        ok = validate_parquet(target, session)
        sys.exit(0 if ok else 1)

    # Stage 1 — always run
    build_stage1(
        flybys=flybys,
        out_path=stage1_path,
        session=session,
        rate_limit=args.rate_limit,
    )

    if args.full:
        build_stage2(
            stage1_path=stage1_path,
            out_path=stage2_path,
            cube_cache=cube_cache,
            session=session,
            max_res_km=args.max_res,
        )
    else:
        # Use per-cube parquet as the pipeline parquet — it works for
        # select_cube_ids (resolution filtering) and geo_only mode.
        shutil.copy2(stage1_path, stage2_path)
        log.info(
            "Per-cube parquet copied to %s\n"
            "  This is sufficient for the current pipeline.\n"
            "  For the full per-pixel parquet (~16 M rows) needed for\n"
            "  detailed VIMS coverage maps or blended organic mode,\n"
            "  rerun with --full.",
            stage2_path,
        )

    # ------------------------------------------------------------------
    # Auto-validate the freshly built parquet (unless suppressed)
    # ------------------------------------------------------------------
    if not args.no_validate:
        log.info("\nAuto-validating freshly built parquet ...")
        ok = validate_parquet(
            stage2_path, session,
            n_sample_cubes=10,
            n_sample_flybys=3,
        )
        if not ok:
            log.warning(
                "Validation found issues — review warnings above.\n"
                "The parquet may still be usable; check the details."
            )

    log.info("Done → %s", stage2_path)


if __name__ == "__main__":
    main()
