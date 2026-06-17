#!/usr/bin/env python3
"""
generate_rankings.py
====================
Samples all five run_pipeline anchor posteriors at every named Titan site
and produces:

  outputs/diagnostics/RANKINGS.md          -- authoritative rankings for thesis
  outputs/diagnostics/RANKINGS.csv         -- machine-readable version

This is the SINGLE SOURCE OF TRUTH for all P(H) values in the thesis tables.
Run this after ``python run_pipeline.py --all-temporal-modes``.

Usage (from project root):
    python scripts/generate_rankings.py

The output RANKINGS.md replaces the hand-written NEW_RANKINGS.md.
"""
from __future__ import annotations

import sys
import csv
from pathlib import Path

import numpy as np

from configs.pipeline_config import PipelineConfig
from configs.site_catalogue import RANKING_SITES, SITE_DICT

# ---------------------------------------------------------------------------
# Anchor posterior paths
# ---------------------------------------------------------------------------

ANCHORS: dict[str, tuple[float, str]] = {
    # name: (epoch_Gya, path)
    "past":           (-3.500, "outputs/past/inference/posterior_analytical.npy"),
    "lake_formation": (-1.000, "outputs/lake_formation/inference/posterior_analytical.npy"),
    "present":        ( 0.000, "outputs/present/inference/posterior_analytical.npy"),
    "near_future":    (+0.250, "outputs/near_future/inference/posterior_analytical.npy"),
    "future":         (+5.900, "outputs/future/inference/posterior_analytical.npy"),
}

ANCHOR_ORDER = ["past", "lake_formation", "present", "near_future", "future"]

EPOCH_LABELS = {
    "past":           "PAST (-3.5 Gya)",
    "lake_formation": "LAKE FORMATION (-1.0 Gya)",
    "present":        "PRESENT (Cassini era, 0.0 Gya)",
    "near_future":    "NEAR FUTURE (+0.25 Gya)",
    "future":         "FUTURE (+5.9 Gya)",
}

GRID_SHAPE = PipelineConfig().canonical_grid_shape  # canonical posterior grid (nrows, ncols)

# ---------------------------------------------------------------------------
# All sites to sample
# ---------------------------------------------------------------------------
# Columns: (short_name, full_name, lon_W_deg, lat_deg, site_type)
# site_type: "lake"   -- appears in ranked table
#            "lander" -- appears below midrule (Dragonfly targets)
#            "other"  -- informational only
#
# The catalogue uses broader type labels; map them to the three display
# categories used by this script:
#   catalogue "crater"  -> "lander"  (Selk is the Dragonfly target crater)
#   catalogue "land"    -> "other"   (equatorial terrain regions)
#   catalogue "lander"  -> "lander"  (Huygens)
#   catalogue "lake"    -> "lake"

_SITE_TYPE_MAP: dict[str, str] = {
    "lake":   "lake",
    "sea":    "lake",
    "crater": "lander",
    "lander": "lander",
    "land":   "other",
}

SITES: list[tuple[str, str, float, float, str]] = [
    (s.short_name, s.full_name, s.lon_W, s.lat,
     _SITE_TYPE_MAP.get(s.site_type, "other"))
    for name in RANKING_SITES
    for s in [SITE_DICT[name]]
]


def sample_posterior(arr: np.ndarray, lon_W: float, lat: float,
                     radius: int = 1) -> float:
    """
    Sample a 2-D posterior array at (lon_W, lat) using a (2r+1)×(2r+1)
    neighbourhood nanmean.

    Grid convention (equirectangular, full globe):
      col = round(lon_W / 360 * ncols)  mod ncols
      row = round((90 - lat) / 180 * nrows), clipped to [0, nrows-1]
    """
    nrows, ncols = arr.shape
    col = int(round(lon_W / 360.0 * ncols)) % ncols
    row = int(round((90.0 - lat) / 180.0 * nrows))
    row = max(0, min(nrows - 1, row))
    r0, r1 = max(0, row - radius), min(nrows, row + radius + 1)
    c0, c1 = max(0, col - radius), min(ncols, col + radius + 1)
    patch = arr[r0:r1, c0:c1].astype(np.float64)
    patch[~np.isfinite(patch)] = np.nan
    return float(np.nanmean(patch))


def load_anchors() -> dict[str, np.ndarray]:
    """Load all five posterior_analytical.npy arrays."""
    arrays: dict[str, np.ndarray] = {}
    for name in ANCHOR_ORDER:
        epoch_t, npy_path = ANCHORS[name]
        p = Path(npy_path)
        if not p.exists():
            print(f"ERROR: {p} not found.")
            print(f"  Run: python run_pipeline.py --temporal-mode "
                  f"{name.replace('_', '-')}")
            sys.exit(1)
        arr = np.load(p)
        if arr.shape != GRID_SHAPE:
            print(f"ERROR: {p}: expected {GRID_SHAPE}, got {arr.shape}")
            sys.exit(1)
        arrays[name] = arr.astype(np.float32)
        print(f"  Loaded {name:17s} (t={epoch_t:+.3f} Gya)  {p}")
    return arrays


def sample_all_sites(
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """
    Returns values[anchor_name][site_short_name] = P(H).
    """
    values: dict[str, dict[str, float]] = {name: {} for name in ANCHOR_ORDER}
    for short, full, lon_W, lat, stype in SITES:
        for name in ANCHOR_ORDER:
            values[name][short] = sample_posterior(arrays[name], lon_W, lat)
    return values


def write_rankings_md(
    values: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """
    Write RANKINGS.md in the same format as the hand-written NEW_RANKINGS.md,
    suitable for direct use as a thesis table reference.
    """
    lander_sites = {s[0] for s in SITES if s[4] == "lander"}
    other_sites  = {s[0] for s in SITES if s[4] == "other"}
    # full name lookup
    full_name  = {s[0]: s[1] for s in SITES}
    lon_lookup = {s[0]: s[2] for s in SITES}
    lat_lookup = {s[0]: s[3] for s in SITES}

    lines = [
        "# DEFINITIVE RANKINGS — BLENDED VIMS (auto-generated)",
        "#",
        "# Source : run_pipeline.py --all-temporal-modes",
        "#          -> posterior_analytical.npy sampled at site coordinates",
        "# Script : scripts/generate_rankings.py",
        f"# Grid   : {GRID_SHAPE[0]} x {GRID_SHAPE[1]}  (equirectangular, 5x5 pixel neighbourhood mean)",
        "# DO NOT EDIT BY HAND.  Re-run the script to update.",
        "",
    ]

    for anchor_name in ANCHOR_ORDER:
        epoch_t, _ = ANCHORS[anchor_name]
        label = EPOCH_LABELS[anchor_name]
        v = values[anchor_name]

        # Rank lake sites only
        lake_sites  = {s[0]: v[s[0]] for s in SITES if s[4] == "lake"}
        ranked = sorted(lake_sites.items(), key=lambda x: x[1], reverse=True)

        all_vals = list(v.values())
        lo, hi = min(all_vals), max(all_vals)

        lines.append(f"## {label} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"{'Rank':<4} {'Name':<18} {'Type':<7} "
                     f"{'lon_W':>6} {'lat':>7} {'P(H)':>6}")
        lines.append("-" * 54)

        for rank, (short, ph) in enumerate(ranked, 1):
            lon_W = lon_lookup[short]
            lat   = lat_lookup[short]
            fname = full_name[short]
            lines.append(
                f"{rank:>2}.  {fname:<18} {'lake':<7} "
                f"{lon_W:6.1f} {lat:+7.1f} {ph:.3f}"
            )

        # Lander sites below separator
        lines.append("      " + "-" * 46)
        lander_vals = sorted(
            [(s[0], v[s[0]]) for s in SITES if s[4] == "lander"],
            key=lambda x: x[1], reverse=True
        )
        for short, ph in lander_vals:
            lon_W = lon_lookup[short]
            lat   = lat_lookup[short]
            fname = full_name[short]
            lines.append(
                f"      {fname:<18} {'lander':<7} "
                f"{lon_W:6.1f} {lat:+7.1f} {ph:.3f}"
            )

        lines.append("")

    # Summary table
    lines += [
        "## SUMMARY TABLE — all sites, all epochs",
        "",
        f"{'Site':<20} {'Past':>6} {'LkFm':>6} {'Pres':>6} {'NF':>6} {'Fut':>6}",
        "-" * 52,
    ]
    for short, full, lon_W, lat, stype in SITES:
        row = f"{full:<20}"
        for anchor_name in ANCHOR_ORDER:
            row += f" {values[anchor_name][short]:6.3f}"
        lines.append(row)

    lines.append("")
    lines.append("## NOTES")
    lines.append("# Rankings are of lake/lacus sites only (type=lake).")
    lines.append("# Lander sites (Selk, Huygens) appear below midrule in thesis tables.")
    lines.append("# 'other' sites (dunes, plains) are excluded from tables.")
    lines.append("# P(H) values are posterior means sampled over a 5×5 pixel patch.")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Written: {out_path}")


def write_rankings_csv(
    values: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Write machine-readable CSV of all sites and epochs."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (["short_name", "full_name", "lon_W", "lat", "type"]
                  + [f"P(H)_{n}" for n in ANCHOR_ORDER])
        writer.writerow(header)
        for short, full, lon_W, lat, stype in SITES:
            row = [short, full, lon_W, lat, stype]
            row += [f"{values[n][short]:.4f}" for n in ANCHOR_ORDER]
            writer.writerow(row)
    print(f"  Written: {out_path}")


def print_summary(values: dict[str, dict[str, float]]) -> None:
    """Print console summary matching the old NEW_RANKINGS.md format."""
    print()
    print(f"{'Site':<20} {'Past':>6} {'LkFm':>6} {'Pres':>6} "
          f"{'NF':>6} {'Fut':>6}")
    print("-" * 55)
    for short, full, lon_W, lat, stype in SITES:
        vals = [f"{values[n][short]:6.3f}" for n in ANCHOR_ORDER]
        print(f"  {full:<20} {' '.join(vals)}")

    print()
    print("Top 3 per epoch:")
    for anchor_name in ANCHOR_ORDER:
        v = values[anchor_name]
        lake_only = {s[0]: v[s[0]] for s in SITES if s[4] == "lake"}
        top3 = sorted(lake_only.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = "  ".join(f"{s}={p:.3f}" for s, p in top3)
        label = EPOCH_LABELS[anchor_name]
        print(f"  {label}: {top3_str}")


def main() -> None:
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading anchor posteriors...")
    arrays = load_anchors()

    print("\nSampling all sites...")
    values = sample_all_sites(arrays)

    print("\nWriting outputs...")
    write_rankings_md(values,  out_dir / "RANKINGS.md")
    write_rankings_csv(values, out_dir / "RANKINGS.csv")

    print_summary(values)
    print("\nDone.")
    print(f"  RANKINGS.md  -> {out_dir / 'RANKINGS.md'}")
    print(f"  RANKINGS.csv -> {out_dir / 'RANKINGS.csv'}")
    print("\nUse RANKINGS.md as the authoritative source for all thesis table values.")


if __name__ == "__main__":
    main()