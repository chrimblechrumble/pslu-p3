#!/usr/bin/env python3
# Titan Habitability Pipeline - Standalone Validation Script
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
validate_pipeline.py
====================
Standalone post-run diagnostic.  Run after each temporal mode
(or after --all-temporal-modes) to catch problems before generating
the animation.

Usage
-----
    python validate_pipeline.py                        # check all modes that exist
    python validate_pipeline.py --mode present         # check one mode only
    python validate_pipeline.py --strict               # fail on any WARNING

Exit codes
----------
    0   all checks passed (or only warnings in non-strict mode)
    1   one or more ERROR-level checks failed
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES = ["present", "past", "lake_formation", "near_future", "future"]

GRID_SHAPE = (1802, 3603)
GRID_PIXELS = GRID_SHAPE[0] * GRID_SHAPE[1]   # 6,492,606

OUTPUTS_DIR = Path("outputs")

# Per-mode expected feature TIF names (must match temporal_features.py exactly)
# present / near_future: full 8-feature standard set
# past / lake_formation:  7 standard (no subsurface_ocean) + 2 temporal extras
# future:                 4 transformed + 4 adapted (completely different names)
_STANDARD_8 = [
    "liquid_hydrocarbon",
    "organic_abundance",
    "acetylene_energy",
    "methane_cycle",
    "surface_atm_interaction",
    "topographic_complexity",
    "geomorphologic_diversity",
    "subsurface_ocean",
]
_STANDARD_7_NO_SS = [f for f in _STANDARD_8 if f != "subsurface_ocean"]

FEATURES_BY_MODE: Dict[str, List[str]] = {
    "present":        _STANDARD_8,
    "near_future":    _STANDARD_8,
    "past":           _STANDARD_7_NO_SS + ["impact_melt_proxy", "cryovolcanic_flux"],
    "lake_formation": _STANDARD_7_NO_SS + ["impact_melt_proxy", "cryovolcanic_flux"],
    "future": [
        "water_ammonia_solvent", "organic_stockpile",
        "dissolved_energy",      "water_ammonia_cycle",
        "surface_atm_interaction", "topographic_complexity",
        "geomorphologic_diversity", "global_ocean_habitability",
    ],
}
# Fallback for the validator's generic checks
FEATURES = _STANDARD_8

# Expected value ranges for PRESENT epoch.
# (mean_lo, mean_hi, pct1_lo, pct99_hi)
# Other temporal modes should be within 2x of these given the scale functions.
FEATURE_CHECKS: Dict[str, dict] = {
    "liquid_hydrocarbon": dict(
        mean=(0.01, 0.18),    # lakes=1.0 (4.3% of globe), SAR=0-0.05, baseline=0.01
        pct1=(0.0,  0.02),
        pct99=(0.03, 1.01),   # pct99 can be 1.0 (polar lake pixels)
        nan_frac_max=0.001,   # NaN fill applied for SAR gaps; should be near 0%
        desc="Polar lakes ~1.0, SAR-proxy capped 0.05, SAR-gap baseline 0.01",
    ),
    "organic_abundance": dict(
        mean=(0.30, 0.75),
        pct1=(0.00, 0.15),    # a few pixels genuinely reach 0 (water-ice craters after -0.13 shift)
        pct99=(0.70, 1.01),   # a few pixels reach 1.0 (bright VIMS patches; clipped)
        nan_frac_max=0.001,
        desc="Dunes 0.685, plains 0.545, mountains 0.115 (post offset-shift); extremes expected",
    ),
    "acetylene_energy": dict(
        mean=(0.15, 0.70),
        pct1=(0.00, 0.30),
        pct99=(0.40, 1.01),
        nan_frac_max=0.001,
        desc="Latitude proxy: equatorial high, polar low",
    ),
    "methane_cycle": dict(
        mean=(0.15, 0.75),
        pct1=(0.00, 0.30),
        pct99=(0.40, 1.01),
        nan_frac_max=0.001,
        desc="CIRS temperature + latitude blend",
    ),
    "surface_atm_interaction": dict(
        mean=(0.01, 0.20),    # sparse channels; margin band is ~13 km; low mean is real
        pct1=(0.00, 0.10),
        pct99=(0.15, 1.01),
        nan_frac_max=0.001,
        desc="Channel density + Birch lake margins; low mean expected (sparse coverage)",
    ),
    "topographic_complexity": dict(
        mean=(0.05, 0.70),
        pct1=(0.00, 0.20),
        pct99=(0.40, 1.01),
        nan_frac_max=0.001,
        desc="DEM roughness: mountains high, plains low",
    ),
    "geomorphologic_diversity": dict(
        mean=(0.05, 0.70),
        pct1=(0.00, 0.20),
        pct99=(0.30, 1.01),
        nan_frac_max=0.001,
        desc="Shannon terrain class diversity",
    ),
    "subsurface_ocean": dict(
        mean=(0.02, 0.30),
        pct1=(0.01, 0.10),
        pct99=(0.20, 1.01),
        nan_frac_max=0.001,
        desc="Prior 0.03, boosted near SAR annuli",
    ),
}

# Colours
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ---------------------------------------------------------------------------
# Minimal GeoTIFF reader (no rasterio / GDAL required)
# ---------------------------------------------------------------------------

def _read_geotiff_stats(path: Path) -> Optional[dict]:
    """
    Read a GeoTIFF and return basic statistics without rasterio.
    Returns None if the file cannot be parsed.

    Works for single-band float32 GeoTIFFs written by rasterio with LZW or
    no compression.  Falls back to numpy.fromfile for raw binary data.
    """
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")
            import numpy as np
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = float("nan")
            valid = data[~np.isnan(data)]
            if valid.size == 0:
                return {"n_valid": 0, "n_nan": data.size,
                        "mean": float("nan"), "std": float("nan"),
                        "pct1": float("nan"), "pct99": float("nan"),
                        "shape": data.shape}
            pct1, pct99 = float(np.percentile(valid, 1)), float(np.percentile(valid, 99))
            return {
                "n_valid": int(valid.size),
                "n_nan":   int(data.size - valid.size),
                "mean":    float(np.mean(valid)),
                "std":     float(np.std(valid)),
                "pct1":    pct1,
                "pct99":   pct99,
                "shape":   data.shape,
            }
    except Exception as e:
        return {"error": str(e)}


def _read_npy_stats(path: Path) -> Optional[dict]:
    """Read a .npy posterior map and return stats."""
    try:
        import numpy as np
        data = np.load(path).astype("float32")
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return {"n_valid": 0, "n_nan": data.size,
                    "mean": float("nan"), "pct1": float("nan"),
                    "pct99": float("nan"), "shape": data.shape}
        return {
            "n_valid": int(valid.size),
            "n_nan":   int(data.size - valid.size),
            "mean":    float(np.mean(valid)),
            "std":     float(np.std(valid)),
            "pct1":    float(np.percentile(valid, 1)),
            "pct99":   float(np.percentile(valid, 99)),
            "shape":   data.shape,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

_errors: List[str] = []
_warnings: List[str] = []


def _err(msg: str) -> None:
    print(f"  {RED}[ERROR]{RESET} {msg}")
    _errors.append(msg)


def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN ]{RESET} {msg}")
    _warnings.append(msg)


def _ok(msg: str) -> None:
    print(f"  {GREEN}[OK   ]{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {CYAN}[INFO ]{RESET} {msg}")


# ---------------------------------------------------------------------------
# Stage checks
# ---------------------------------------------------------------------------

def check_stage3_features(mode: str) -> None:
    """Check all feature GeoTIFFs for a given mode."""
    tif_dir = OUTPUTS_DIR / mode / "features" / "tifs"
    print(f"\n{BOLD}[Stage 3 Features] mode={mode}{RESET}")

    if not tif_dir.exists():
        _err(f"Feature TIF directory missing: {tif_dir}")
        return

    expected_features = FEATURES_BY_MODE.get(mode, FEATURES)

    for feat in expected_features:
        tif_path = tif_dir / f"{feat}.tif"
        if not tif_path.exists():
            _err(f"  Feature TIF missing: {tif_path.name}")
            continue

        stats = _read_geotiff_stats(tif_path)
        if stats is None or "error" in stats:
            _warn(f"  Could not read {tif_path.name}: {stats}")
            continue

        size_mb = tif_path.stat().st_size / 1_048_576
        nan_frac = stats["n_nan"] / max(1, stats["n_valid"] + stats["n_nan"])

        # Shape check
        shape = stats.get("shape", (0, 0))
        if shape != GRID_SHAPE:
            _err(f"  {feat}: shape {shape} != expected {GRID_SHAPE}")

        # NaN fraction check
        checks = FEATURE_CHECKS.get(feat)
        if checks:
            if nan_frac > checks["nan_frac_max"]:
                _warn(f"  {feat}: {100*nan_frac:.2f}% NaN "
                      f"(expected <{100*checks['nan_frac_max']:.2f}%)")

            m = stats["mean"]
            if not (checks["mean"][0] <= m <= checks["mean"][1]):
                _warn(f"  {feat}: mean={m:.4f} outside expected "
                      f"[{checks['mean'][0]:.3f}, {checks['mean'][1]:.3f}]  "
                      f"-- {checks['desc']}")

            p1 = stats["pct1"]
            if p1 < checks["pct1"][0] or p1 > checks["pct1"][1]:
                _warn(f"  {feat}: pct1={p1:.4f} outside expected "
                      f"[{checks['pct1'][0]:.3f}, {checks['pct1'][1]:.3f}]")

            p99 = stats["pct99"]
            if p99 < checks["pct99"][0] or p99 > checks["pct99"][1]:
                _warn(f"  {feat}: pct99={p99:.4f} outside expected "
                      f"[{checks['pct99'][0]:.3f}, {checks['pct99'][1]:.3f}]")

        _ok(f"  {feat:<30} mean={stats['mean']:.4f}  "
            f"std={stats['std']:.4f}  pct[1,99]=[{stats['pct1']:.3f},{stats['pct99']:.3f}]  "
            f"NaN={100*nan_frac:.2f}%  ({size_mb:.1f} MB)")

    # No unexpected TIFs (catches phantom features)
    actual_tifs = {p.stem for p in tif_dir.glob("*.tif")}
    expected_set = set(expected_features)
    phantom = actual_tifs - expected_set
    if phantom:
        _warn(f"  Unexpected TIFs (may be stale): {sorted(phantom)}")


def check_stage4_inference(mode: str) -> None:
    """Check the Stage 4 posterior .npy and log for class balance."""
    inf_dir = OUTPUTS_DIR / mode / "inference"
    print(f"\n{BOLD}[Stage 4 Inference] mode={mode}{RESET}")

    # Posterior .npy
    npy_path = inf_dir / "posterior_mean.npy"
    if not npy_path.exists():
        _err(f"posterior_mean.npy missing: {npy_path}")
    else:
        stats = _read_npy_stats(npy_path)
        if stats and "error" not in stats:
            nan_frac = stats["n_nan"] / max(1, stats["n_valid"] + stats["n_nan"])

            if stats["shape"] != GRID_SHAPE:
                _err(f"posterior_mean.npy shape {stats['shape']} != {GRID_SHAPE}")

            if nan_frac > 0.01:
                _warn(f"posterior_mean.npy: {100*nan_frac:.1f}% NaN "
                      f"({stats['n_nan']:,} pixels)")

            if not (0.10 <= stats["mean"] <= 0.65):
                _warn(f"posterior mean={stats['mean']:.4f} outside display range "
                      f"[0.10, 0.65] -- check VMIN/VMAX")

            _ok(f"posterior_mean.npy  mean={stats['mean']:.4f}  "
                f"pct[1,99]=[{stats['pct1']:.3f},{stats['pct99']:.3f}]  "
                f"NaN={100*nan_frac:.2f}%")
        else:
            _warn(f"Could not read posterior_mean.npy: {stats}")

    # Parse pipeline.log for Stage 4 label balance and importances
    log_path = OUTPUTS_DIR / "logs" / "pipeline.log"
    if log_path.exists():
        _check_log_stage4(log_path, mode)
    else:
        _info("pipeline.log not found -- skipping log analysis")


def _check_log_stage4(log_path: Path, mode: str) -> None:
    """Parse the pipeline log for Stage 4 label balance and feature importances."""
    import re

    text = log_path.read_text()

    # Find the most recent Stage 4 block for this mode
    pattern_labels = re.compile(
        r"Labels: (\d+) positive.*?(\d+) negative.*?(\d+\.\d+)% positive"
    )
    pattern_importance = re.compile(
        r"(\w+)\s+(0\.\d+)\s*$", re.MULTILINE
    )

    all_label_matches = list(pattern_labels.finditer(text))
    if all_label_matches:
        m = all_label_matches[-1]   # last run
        n_pos, n_neg, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if pct > 70:
            _warn(f"Label imbalance: {pct:.1f}% positive "
                  f"(expected ~50% with median-split fix).  "
                  f"Was the new code picked up?  "
                  f"n_pos={n_pos:,}  n_neg={n_neg:,}")
        elif pct < 30:
            _warn(f"Label imbalance: {pct:.1f}% positive (too few positives)")
        else:
            _ok(f"Label balance: {pct:.1f}% positive  "
                f"(n_pos={n_pos:,}  n_neg={n_neg:,})")

    # Top importance
    # Look for lines after "Top importances:"
    # Split on "Top importances:" and take the last segment
    # This ensures we always parse the most recent pipeline run
    _imp_parts = text.split("Top importances:")
    if len(_imp_parts) > 1:
        _last_imp_text = _imp_parts[-1]  # text after last "Top importances:"
        # Truncate at the next "---" separator (end of that log block)
        _last_imp_text = _last_imp_text.split("---")[0]
        imp_section = type("_NS", (), {"group": staticmethod(lambda n: _last_imp_text)})()
    else:
        imp_section = None
    if imp_section:
        imps = {}
        for feat, val in re.findall(
            r"(liquid_hydrocarbon|organic_abundance|acetylene_energy|"
            r"methane_cycle|surface_atm_interaction|topographic_complexity|"
            r"geomorphologic_diversity|subsurface_ocean)\s+([\d.]+)",
            imp_section.group(1)
        ):
            imps[feat] = float(val)

        if imps:
            top_feat = max(imps, key=imps.get)
            top_val  = imps[top_feat]

            if top_val > 0.50:
                _warn(f"Feature importance dominated by {top_feat}={top_val:.3f} "
                      f"(>50% -- model may be using only this feature).  "
                      f"Check label balance above.")
            else:
                _ok(f"Feature importances well-distributed: "
                    f"top={top_feat} ({top_val:.3f})")

            for feat, val in sorted(imps.items(), key=lambda x: -x[1]):
                _info(f"    {feat:<35} {val:.3f}")


def check_stage5_figures(mode: str) -> None:
    """Check that all expected Stage 5 figures were produced."""
    fig_dir = OUTPUTS_DIR / mode / "figures"
    print(f"\n{BOLD}[Stage 5 Figures] mode={mode}{RESET}")

    expected_figures = [
        "fig1_posterior.pdf",
        "fig1_posterior.png",
        "fig3_features.pdf",
        "fig4_top_sites.pdf",
        "fig5_interactive.html",
    ]
    for fig in expected_figures:
        p = fig_dir / fig
        if not p.exists():
            _warn(f"Figure missing: {fig}")
        else:
            size_kb = p.stat().st_size / 1024
            if size_kb < 5:
                _warn(f"{fig}: suspiciously small ({size_kb:.1f} KB)")
            else:
                _ok(f"{fig}  ({size_kb:.0f} KB)")


def check_animation_inputs() -> None:
    """Verify all inputs needed by generate_temporal_maps.py exist."""
    print(f"\n{BOLD}[Animation inputs]{RESET}")

    # Present-epoch TIFs are required (the animation scales from these)
    tif_dir = OUTPUTS_DIR / "present" / "features" / "tifs"
    if not tif_dir.exists():
        _err(f"Present feature TIF directory missing: {tif_dir}  "
             f"-- run: python run_pipeline.py --temporal-mode present")
        return

    missing_present = [f for f in FEATURES
                       if not (tif_dir / f"{f}.tif").exists()]
    if missing_present:
        _err(f"Present TIFs missing (required for animation): {missing_present}")
    else:
        _ok("All present-epoch feature TIFs exist")

    # full_inference mode also needs all 5 anchor posteriors
    anchors_ok = True
    for mode in MODES:
        p = OUTPUTS_DIR / mode / "inference" / "posterior_mean.npy"
        if p.exists():
            _ok(f"Anchor posterior found: {mode}")
        else:
            _info(f"Anchor posterior absent: {mode}  "
                  f"(only needed for --inference-mode full_inference)")
            anchors_ok = False

    if anchors_ok:
        _ok("All 5 anchor posteriors present -- full_inference mode available")
    else:
        _info("Run all 5 modes then use --inference-mode modelled (default) for animation")


def check_code_version() -> None:
    """Verify the installed code is the expected version."""
    print(f"\n{BOLD}[Code version]{RESET}")
    EXPECTED_VERSION = "5.0"
    try:
        import sys
        sys.path.insert(0, str(Path(".").resolve()))
        from titan.features import PIPELINE_CODE_VERSION
        if PIPELINE_CODE_VERSION == EXPECTED_VERSION:
            _ok(f"titan/features.py version {PIPELINE_CODE_VERSION} (correct)")
        else:
            _err(
                f"titan/features.py version {PIPELINE_CODE_VERSION} -- "
                f"expected {EXPECTED_VERSION}.  "
                f"You are running OLD code.  "
                f"Extract the new zip and overwrite titan/features.py, "
                f"then re-run the pipeline before validating."
            )
    except ImportError:
        _err(
            "Cannot import PIPELINE_CODE_VERSION from titan/features.py. "
            "The installed code pre-dates v5.0.  Install the new zip."
        )

    # Also check log for the version sentinel
    import re
    log_path = OUTPUTS_DIR / "logs" / "pipeline.log"
    if log_path.exists():
        text = log_path.read_text()
        m = re.search(r"TITAN HABITABILITY PIPELINE\s+v(\S+)", text)
        if m:
            logged_ver = m.group(1)
            if logged_ver == EXPECTED_VERSION:
                _ok(f"pipeline.log confirms run was v{logged_ver}")
            else:
                _warn(
                    f"pipeline.log shows v{logged_ver} -- "
                    f"outputs were generated by OLD code even if new code is installed. "
                    f"Re-run the pipeline."
                )
        else:
            _warn(
                "No version line found in pipeline.log -- "
                "outputs were generated before v5.0. Re-run the pipeline."
            )


def check_organic_global_offset() -> None:
    """
    Parse the pipeline log to verify organic_abundance global_offset is
    in a reasonable range and the seignovert_valid_mask fix is active.
    """
    print(f"\n{BOLD}[organic_abundance calibration]{RESET}")
    import re

    log_path = OUTPUTS_DIR / "logs" / "pipeline.log"
    if not log_path.exists():
        _info("pipeline.log not found -- skipping")
        return

    text = log_path.read_text()
    matches = re.findall(r"global level shift = ([-\d.]+).*?overlap=(\d+)", text)
    if not matches:
        _warn("No 'global level shift' line found in log")
        return

    offset, overlap = float(matches[-1][0]), int(matches[-1][1])
    _info(f"global_offset={offset:.4f}  overlap={overlap:,} px")

    if not (-0.30 <= offset <= 0.10):
        _warn(f"global_offset={offset:.4f} is outside expected range [-0.30, 0.10]. "
              f"Large negative values suggest Seignovert mosaic is systematically "
              f"lower than published geo scores -- check VIMS data.")
    else:
        _ok(f"global_offset={offset:.4f} is in range [-0.30, 0.10]")

    if overlap < 500_000:
        _warn(f"Overlap only {overlap:,} px -- seignovert_valid_mask may have "
              f"excluded too many pixels or Seignovert mosaic has poor coverage")
    else:
        _ok(f"Overlap {overlap:,} px (sufficient for calibration)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Titan pipeline outputs")
    parser.add_argument("--mode", default=None,
                        help="Check one mode only (default: all that exist)")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip Stage 5 figure checks")
    args = parser.parse_args()

    modes_to_check = (
        [args.mode] if args.mode
        else [m for m in MODES if (OUTPUTS_DIR / m).exists()]
    )

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  TITAN PIPELINE VALIDATOR{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  Modes to validate: {modes_to_check}")
    print(f"  Outputs directory: {OUTPUTS_DIR.resolve()}")

    for mode in modes_to_check:
        print(f"\n{BOLD}{'─'*65}{RESET}")
        print(f"{BOLD}  MODE: {mode.upper()}{RESET}")
        print(f"{BOLD}{'─'*65}{RESET}")
        check_stage3_features(mode)
        check_stage4_inference(mode)
        if not args.skip_figures:
            check_stage5_figures(mode)

    check_code_version()
    check_organic_global_offset()
    check_animation_inputs()

    # Summary
    n_err  = len(_errors)
    n_warn = len(_warnings)
    print(f"\n{BOLD}{'='*65}{RESET}")
    if n_err == 0 and n_warn == 0:
        print(f"{GREEN}{BOLD}  ALL CHECKS PASSED{RESET}")
    elif n_err == 0:
        print(f"{YELLOW}{BOLD}  {n_warn} WARNING(S) -- review above{RESET}")
    else:
        print(f"{RED}{BOLD}  {n_err} ERROR(S)  {n_warn} WARNING(S){RESET}")
        print(f"{RED}  Do NOT proceed to animation until errors are resolved.{RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")

    if args.strict and n_warn > 0:
        return 1
    return 1 if n_err > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
