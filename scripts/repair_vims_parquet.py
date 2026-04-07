#!/usr/bin/env python3
"""
scripts/repair_vims_parquet.py
================================
Repair an existing vims_footprints.parquet where lat/lon/res are all NaN
because the original build_vims_parquet.py was scraping raw HTML without
stripping tags (causing regexes to fail across <th>/<td> boundaries).

This script re-fetches only the cubes with NaN values — much faster than
a full rebuild (~15-30 min vs ~30 min, and skips any cubes that have
genuine n/a values on the portal).

Usage:
    python scripts/repair_vims_parquet.py

    # Dry-run: show how many rows need repair without fetching
    python scripts/repair_vims_parquet.py --dry-run

    # Custom parquet path:
    python scripts/repair_vims_parquet.py --parquet data/raw/vims_footprints.parquet

    # Limit requests per second (default: 4/s):
    python scripts/repair_vims_parquet.py --rate 2

The script is resumable: re-running will only re-fetch rows still NaN.
"""

from __future__ import annotations

import argparse
import logging
import re
import html as _html
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("repair_vims")

PORTAL_BASE = "https://vims.univ-nantes.fr"

# Regex patterns — applied to HTML with tags stripped.
# Values sit in <td> adjacent to <th> labels; stripping tags first
# makes label + value appear as consecutive tokens.
_RE_TAGS       = re.compile(r"<[^>]+>")
_RE_RESOLUTION = re.compile(r"Mean resolution\s+([\d,]+)\s+km/pixel")
_RE_SUB_SC     = re.compile(
    r"Sub-Spacecraft point\s+([-\d]+)\s*\u00b0\s*N\s*\|\s*([-\d]+)\s*\u00b0\s*E"
)
_RE_DISTANCE   = re.compile(r"Distance\s+([\d,]+)\s+km")


def _strip_html(raw_html: str) -> str:
    """Strip HTML tags, decode entities (e.g. &deg; -> °), collapse whitespace.
    The portal serves degree symbols as &deg; entities which requests fetches
    verbatim; html.unescape() converts them before regex matching."""
    return _html.unescape(_RE_TAGS.sub(" ", raw_html))


def _get(url: str, session, retries: int = 4, backoff: float = 2.0) -> str:
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
            log.warning("  retry %d/%d (%s, wait %.1fs)", attempt + 1, retries, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"Failed: {url}")


def fetch_meta(cube_id: str, session) -> tuple[float, float, float, float]:
    """
    Re-fetch lat, lon_W, res_km, altitude_km for one cube.
    Returns (lat, lon_W, res_km, altitude_km) — NaN for any field not found.
    """
    import math
    NAN = float("nan")

    try:
        raw = _get(f"{PORTAL_BASE}/cube/{cube_id}", session)
    except Exception as exc:
        log.debug("  fetch failed %s: %s", cube_id, exc)
        return NAN, NAN, NAN, NAN

    html = _strip_html(raw)

    def parse_int(m: Optional[re.Match], g: int = 1) -> Optional[int]:
        return int(m.group(g).replace(",", "")) if m else None

    res_m  = _RE_RESOLUTION.search(html)
    sub_m  = _RE_SUB_SC.search(html)
    dist_m = _RE_DISTANCE.search(html)

    res_km  = parse_int(res_m)
    lat_deg = parse_int(sub_m, 1)
    lon_e   = parse_int(sub_m, 2)
    dist_km = parse_int(dist_m)

    lon_w = float((-lon_e) % 360) if lon_e is not None else NAN

    return (
        float(lat_deg) if lat_deg is not None else NAN,
        lon_w,
        float(res_km)  if res_km  is not None else NAN,
        float(dist_km) if dist_km is not None else NAN,
    )


def repair(parquet_path: Path, rate: float, dry_run: bool) -> None:
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas not installed. Run: pip install pandas pyarrow")
        raise SystemExit(1)
    try:
        import requests
    except ImportError:
        log.error("requests not installed. Run: pip install requests")
        raise SystemExit(1)

    if not parquet_path.exists():
        log.error("Parquet not found: %s", parquet_path)
        raise SystemExit(1)

    df = pd.read_parquet(parquet_path)
    log.info("Loaded: %s  (%d rows)", parquet_path, len(df))

    # Identify rows needing repair: any of lat/lon/res is NaN
    need_repair = df["lat"].isna() | df["lon"].isna() | df["res"].isna()
    n_repair = need_repair.sum()
    n_total  = len(df)
    log.info("Rows needing repair: %d / %d (%.1f%%)",
             n_repair, n_total, 100 * n_repair / max(n_total, 1))

    if n_repair == 0:
        log.info("Nothing to repair — parquet is already complete.")
        return

    if dry_run:
        log.info("[DRY RUN] Would re-fetch %d cube HTML pages.", n_repair)
        sample = df.loc[need_repair, "id"].head(5).tolist()
        log.info("  First 5 cube IDs to repair: %s", sample)
        return

    # Estimate time
    min_time = n_repair / rate / 60
    log.info("Estimated time at %.1f req/s: ~%.0f minutes", rate, min_time)
    log.info("Starting repair (resumable — re-run anytime) ...")

    session = requests.Session()
    session.headers.update({"User-Agent": "titan-habitability-pipeline/5.0 (research)"})

    delay = 1.0 / rate
    repair_indices = df.index[need_repair].tolist()

    fixed = 0
    still_nan = 0
    save_every = 500  # checkpoint every N repairs

    for i, idx in enumerate(repair_indices):
        cube_id = df.at[idx, "id"]
        lat, lon_w, res_km, alt_km = fetch_meta(cube_id, session)

        df.at[idx, "lat"] = lat
        df.at[idx, "lon"] = lon_w
        df.at[idx, "res"] = res_km
        # Only update altitude if we got a value and current is NaN
        if not pd.isna(alt_km) and pd.isna(df.at[idx, "altitude"]):
            df.at[idx, "altitude"] = alt_km

        if not pd.isna(lat):
            fixed += 1
        else:
            still_nan += 1  # legitimately n/a on portal (non-Titan obs etc.)

        # Progress log
        if (i + 1) % 100 == 0 or (i + 1) == n_repair:
            log.info("  %d / %d  fixed=%d  still-nan=%d",
                     i + 1, n_repair, fixed, still_nan)

        # Checkpoint save
        if (i + 1) % save_every == 0:
            df.to_parquet(parquet_path, index=False)
            log.info("  Checkpoint saved → %s", parquet_path)

        time.sleep(delay)

    # Final save
    df.to_parquet(parquet_path, index=False)
    log.info("Repair complete → %s", parquet_path)
    log.info("  Fixed: %d rows  |  Still NaN (genuine n/a): %d rows",
             fixed, still_nan)

    # Quick summary
    valid = df["lat"].notna()
    log.info("Final parquet: %d total rows, %d with valid lat/lon/res",
             len(df), valid.sum())
    if valid.sum() > 0:
        log.info("  lat  : min=%.1f  max=%.1f", df.loc[valid,"lat"].min(), df.loc[valid,"lat"].max())
        log.info("  lon  : min=%.1f  max=%.1f", df.loc[valid,"lon"].min(), df.loc[valid,"lon"].max())
        log.info("  res  : median=%.1f km/px", df.loc[df["res"].notna(),"res"].median())


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair NaN lat/lon/res in vims_footprints.parquet")
    ap.add_argument("--parquet", default="data/raw/vims_footprints.parquet",
                    help="Path to parquet file (default: data/raw/vims_footprints.parquet)")
    ap.add_argument("--rate", type=float, default=4.0,
                    help="HTTP requests per second (default: 4.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be repaired without making any requests")
    args = ap.parse_args()
    repair(Path(args.parquet), args.rate, args.dry_run)


if __name__ == "__main__":
    main()
