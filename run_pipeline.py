#!/usr/bin/env python3
"""
run_pipeline.py
================
Main entry-point for the Titan Habitability Pipeline.

Temporal modes
--------------
  present  Cassini epoch (~2004–2017).  Original 8 features.
           Two priors corrected vs initial implementation:
             organic_abundance 0.60 → 0.70  (Cable 2012; Malaska 2025)
             subsurface_ocean  0.10 → 0.03  (D3; Neish et al. 2024)

  past     Early Titan ~3.5 Gya (Late Heavy Bombardment era). [D1 default]
           9 features: 7 adapted PRESENT features + impact_melt_proxy
           + cryovolcanic_flux.  All features derived from current Cassini
           data used as proxies for past conditions.

  future   Red giant epoch ~6 Gya from now (Lorenz, Lunine & McKay 1997).
           8 transformed features: methane-cycle features replaced by
           water-ammonia analogs.  All derived from current Cassini data.
           NOTE: Distinct from the D2 near-future solar warming window (see
           below), which affects only the subsurface_ocean feature.

Accepted design decisions (D1–D4)
-----------------------------------
D1 — Past epoch: 3.5 Gya default, configurable via --past-epoch-gya.
     SAR-bright crater annuli (D4) are interpreted as relics of impact-melt
     liquid water from this epoch.
     Ref: Neish & Lorenz (2012); Artemieva & Lunine (2003)

D2 — Near-future solar warming window: 100–400 Myr (onset from now).
     More recent radiative-transfer estimates narrow the window to this range,
     much sooner than the classical red-giant estimate (~6 Gya). Used ONLY in
     the subsurface_ocean feature's temporal prior; does not affect the FUTURE
     temporal mode (which models the full red-giant-era scenario at ~6 Gya).
     ASSUMPTION — uniform global warming (explicit):
       Solar brightening is applied uniformly across Titan's surface. A real
       GCM would show differential polar/equatorial warming, but no spatially
       resolved model matches the SAR resolution. This assumption is
       conservative: low latitudes likely warm sooner. Configurable via
       --future-window-min, --future-window-max, --no-uniform-warming.
     Ref: Lorenz et al. (1997); Lunine & Lorenz (2009)

D3 — Subsurface ocean prior: 0.03 (default, configurable via
     --subsurface-ocean-prior). Revised down from 0.10 based on Neish et al.
     (2024): organic flux to the subsurface ocean is ~7,500 kg/yr glycine
     (~one elephant/year), severely limiting its habitability in the present
     epoch. The k2=0.589 measurement (Iess 2012) confirms the ocean EXISTS
     but does not constrain habitability.

D4 — SAR bright annuli as past-liquid-water proxy: ACCEPTED.
     Radar-bright ring structures in Cassini SAR are interpreted as impact
     melt rims and cryovolcanic flow fronts — locations where liquid water
     briefly contacted Titan's organic surface. Used in Feature 8
     (subsurface_ocean) via a ring-shaped morphological filter on the SAR.
     Ref: Neish et al. (2018); Wood et al. (2010); Lopes et al. (2007, 2013)

Quick start
-----------
  python run_pipeline.py --status                    # data status
  python run_pipeline.py                             # present mode (defaults)
  python run_pipeline.py --temporal-mode past        # early Titan
  python run_pipeline.py --temporal-mode future      # red giant era (~6 Gya)
  python run_pipeline.py --all-temporal-modes        # compare all three
  python run_pipeline.py --backend pymc              # full MCMC

  # Override D1 past epoch (e.g. more conservative ~4.0 Gya):
  python run_pipeline.py --past-epoch-gya 4.0

  # Override D2 future window (e.g. wider uncertainty range):
  python run_pipeline.py --future-window-min 50 --future-window-max 600

  # Disable D2 uniform warming assumption (future prior = 0):
  python run_pipeline.py --no-uniform-warming

  # Override D3 subsurface ocean prior (sensitivity test):
  python run_pipeline.py --subsurface-ocean-prior 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Titan Habitability Pipeline — end-to-end temporal analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir",       default="data/raw",       type=Path)
    p.add_argument("--processed-dir",  default="data/processed", type=Path)
    p.add_argument("--output-dir",     default="outputs",        type=Path)
    p.add_argument("--shapefile-dir",  default=None,             type=Path,
                   help="Override path to Lopes geomorphology shapefiles dir.")
    p.add_argument(
        "--birch-dir", default=None, type=Path,
        metavar="DIR",
        help=(
            "Root directory of the Birch+2017 / Palermo+2022 polar lake "
            "dataset.  Expected sub-dirs: birch_filled/, birch_empty/, "
            "palermo/.  Defaults to data/raw/birch_polar_mapping/. "
            "Download: "
            "https://data.astro.cornell.edu/titan_polar_mapping_birch/"
        ),
    )
    p.add_argument(
        "--vims-parquet",
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Path to the VIMS spatial footprint parquet file. "
            "Accepts both the full file (~5.4M rows, 227 MB) and the "
            "1000-row development sample (vims_sample_1000rows.parquet). "
            "If not set, the pipeline searches data_dir for "
            "vims_footprints.parquet, vims_sample_1000rows.parquet, "
            "or any vims_*.parquet file."
        ),
    )
    p.add_argument("--backend",        default="sklearn",
                   choices=["sklearn", "pymc", "numpyro"])
    p.add_argument("--res",          default=4490,  type=float,  metavar="METRES")
    p.add_argument("--mcmc-draws",   default=2000,  type=int)
    p.add_argument("--mcmc-chains",  default=4,     type=int)
    p.add_argument("--seed",         default=42,    type=int)
    p.add_argument("--paper-dpi",    default=300,   type=int)
    p.add_argument(
        "--no-labels",
        action="store_true",
        dest="no_labels",
        default=False,
        help=(
            "Suppress all geographic feature labels on output maps. "
            "Default: labels are shown."
        ),
    )
    p.add_argument(
        "--label-categories",
        nargs="+",
        metavar="CATEGORY",
        default=None,
        choices=["sea", "lake", "terrain", "mission"],
        help=(
            "Show only labels from these categories. "
            "Choices: sea lake terrain mission. "
            "Default: all categories. "
            "Example: --label-categories sea lake"
        ),
    )
    p.add_argument(
        "--label-names",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Show only these specific named features. "
            'Overrides --label-categories if both are given. '
            'Example: --label-names "Kraken Mare" "Selk (DFly)"'
        ),
    )
    p.add_argument(
        "--temporal-mode",
        default="present",
        choices=["past", "present", "future"],
        help="Temporal habitability mode (default: present)",
    )
    p.add_argument(
        "--all-temporal-modes",
        action="store_true",
        help="Run all three temporal modes and produce comparison output",
    )

    # ── D1: Past epoch (configurable; default 3.5 Gya) ───────────────────────
    # The age of the last major epoch of widespread liquid water on Titan's
    # surface (Late Heavy Bombardment / early cryovolcanic era).
    # SAR-bright annuli around craters are interpreted as relics of this epoch.
    # References: Neish & Lorenz (2012); Artemieva & Lunine (2003)
    p.add_argument(
        "--past-epoch-gya",
        default=3.5,
        type=float,
        metavar="GYA",
        help=(
            "[D1] Age (Gya before present) of the last major surface liquid-water "
            "epoch. Default 3.5 Gya (Late Heavy Bombardment era). Lower = more "
            "recent cryovolcanism; raise to 4.0 for accretion-era estimate."
        ),
    )

    # ── D2: Near-future solar-warming habitability window (100–400 Myr) ───────
    # As the Sun brightens (~10%/Gyr on the main sequence), Titan's surface
    # temperature will eventually rise enough for episodic or sustained liquid
    # water. More recent radiative-transfer estimates place the ONSET of this
    # window at 100–400 Myr from now — much sooner than the classical red-giant
    # estimate (~6 Gya). This is DISTINCT from the FUTURE temporal mode (which
    # models the full red-giant-era scenario).
    #
    # ASSUMPTION — uniform global warming (explicit, per D2):
    # Solar brightening is applied uniformly across all latitudes and
    # longitudes. Real GCMs would predict differential polar vs equatorial
    # warming due to Titan's obliquity and atmospheric dynamics, but no
    # spatially resolved warming model is currently available at the resolution
    # of the SAR mosaic. The uniform assumption is conservative: if anything
    # it underestimates habitability at low latitudes (which warm first) and
    # overestimates it at the poles. Set --no-uniform-warming to disable the
    # future-window prior contribution entirely.
    #
    # References: Lorenz et al. (1997) GRL 24:2905; Lunine & Lorenz (2009)
    p.add_argument(
        "--future-window-min",
        default=100.0,
        type=float,
        metavar="MYR",
        help=(
            "[D2] Earliest onset (Myr from now) of the near-future solar-warming "
            "habitability window. Default 100 Myr. Used in the subsurface_ocean "
            "feature temporal prior. Distinct from the --temporal-mode future "
            "(red-giant epoch, ~6 Gya)."
        ),
    )
    p.add_argument(
        "--future-window-max",
        default=400.0,
        type=float,
        metavar="MYR",
        help=(
            "[D2] Latest onset (Myr from now) of the near-future solar-warming "
            "habitability window. Default 400 Myr."
        ),
    )
    p.add_argument(
        "--no-uniform-warming",
        action="store_true",
        help=(
            "[D2] Disable the uniform global warming assumption. When set, the "
            "future-window temporal prior contributes zero to the subsurface_ocean "
            "feature (present-day-only analysis). By default, uniform warming is "
            "assumed (spatially constant solar heating increment)."
        ),
    )

    # ── D3: Subsurface ocean prior (configurable; default 0.03) ──────────────
    # Revised down from an initial 0.10 based on Neish et al. (2024), which
    # shows organic flux to the subsurface ocean is ~7,500 kg/yr glycine
    # (equivalent to ~one elephant). This severely limits the ocean's access
    # to surface organic substrate and thus its habitability in the present
    # epoch. The k2=0.589 measurement (Iess 2012) confirms the ocean EXISTS
    # but does not constrain its habitability.
    # References: Neish et al. (2024) doi:10.1089/ast.2023.0055
    #             Iess et al. (2012)  doi:10.1126/science.1219631
    p.add_argument(
        "--subsurface-ocean-prior",
        default=0.03,
        type=float,
        metavar="PROB",
        help=(
            "[D3] Base prior probability for the subsurface_ocean feature "
            "(global base; SAR bright annuli boost this locally). "
            "Default 0.03 (revised down from 0.10; Neish et al. 2024). "
            "Range [0, 1]."
        ),
    )

    p.add_argument("--overwrite",            action="store_true")
    p.add_argument("--skip-acquisition",     action="store_true")
    p.add_argument("--skip-preprocessing",   action="store_true")
    p.add_argument("--skip-features",        action="store_true")
    p.add_argument("--skip-inference",       action="store_true")
    p.add_argument("--only-visualise",       action="store_true")
    p.add_argument("--status",               action="store_true")
    p.add_argument("-v", "--verbose",        action="store_true")
    return p


def setup_logging(verbose: bool, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"
    level    = logging.DEBUG if verbose else logging.INFO
    fmt      = "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s"
    logging.basicConfig(
        level=level, format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )
    for noisy in ("rasterio", "matplotlib", "pymc", "pytensor",
                  "fiona", "shapely", "numexpr", "jax"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.info("Log file → %s", log_file)


class Timer:
    def __init__(self, name: str) -> None:
        self.name = name
        self._t0  = time.perf_counter()

    def elapsed(self) -> str:
        s = time.perf_counter() - self._t0
        return f"{s/60:.1f} min" if s >= 60 else f"{s:.1f} s"

    def done(self, log: logging.Logger) -> None:
        log.info("─── %s done (%s) ───", self.name, self.elapsed())


# ---------------------------------------------------------------------------
# Single-mode pipeline run
# ---------------------------------------------------------------------------

def run_single_mode(
    mode_str:   str,
    args:       argparse.Namespace,
    cfg:        "PipelineConfig",
    stack:      "xr.Dataset",
    grid:       "CanonicalGrid",
    log:        logging.Logger,
) -> dict:
    """
    Run stages 3–5 for one temporal mode.

    Returns a dict with result arrays for comparison.
    """
    import numpy as np
    import xarray as xr
    from configs.temporal_config import TemporalMode, describe_prior_changes

    mode = TemporalMode(mode_str)
    mode_out = cfg.output_dir / mode_str

    log.info("")
    log.info("══════════════════════════════════════════════════")
    log.info("  TEMPORAL MODE: %s", mode_str.upper())
    log.info("══════════════════════════════════════════════════")
    log.info("%s", describe_prior_changes(mode))

    feat_dir = mode_out / "features"
    feat_nc  = feat_dir / f"titan_features_{mode_str}.nc"
    inf_dir  = mode_out / "inference"

    # ── Stage 3: Features ─────────────────────────────────────────────────
    if not (args.skip_features or args.only_visualise):
        log.info("")
        log.info("[%s] STAGE 3 — Feature Extraction", mode_str.upper())
        t = Timer(f"Stage 3 [{mode_str}]")
        from titan.temporal_features import TemporalFeatureExtractor
        extractor = TemporalFeatureExtractor(
            grid,
            mode,
            window_config               = cfg.habitability_window,
            subsurface_ocean_base_prior = cfg.priors.prior_mean_subsurface_ocean,
        )
        features  = extractor.extract(stack)

        log.info("  Features:")
        for name, frac in features.coverage_fraction().items():
            log.info("    %-35s %5.1f%%", name, frac * 100)

        feat_dir.mkdir(parents=True, exist_ok=True)
        features.to_xarray().to_netcdf(feat_nc)
        log.info("  Saved → %s", feat_nc)

        # Save individual feature TIFs for external inspection.
        # These are float32 GeoTIFFs with nodata=-9999.
        # Note: float32 TIFFs cannot be opened in macOS Preview (only
        # uint8/uint16 are supported). Use QGIS or open with Python/rasterio.
        tif_dir = feat_dir / "tifs"
        tif_dir.mkdir(exist_ok=True)
        import rasterio
        from rasterio.transform import from_origin
        _NODATA_SENTINEL = -9999.0
        for feat_name in features.feature_names():
            arr = features.get_feature(feat_name)
            if arr is None:
                continue
            arr_out = np.where(np.isfinite(arr), arr, _NODATA_SENTINEL).astype(np.float32)
            tif_path = tif_dir / f"{feat_name}.tif"
            with rasterio.open(
                tif_path, "w",
                driver="GTiff", dtype="float32", count=1,
                width=grid.ncols, height=grid.nrows,
                crs=grid.crs, transform=grid.transform,
                nodata=_NODATA_SENTINEL,
                compress="deflate", tiled=True,
                blockxsize=256, blockysize=256,
            ) as dst:
                dst.write(arr_out, 1)
        log.info("  Feature TIFs → %s", tif_dir)
        t.done(log)
    else:
        log.info("[%s] Stage 3 — loading features from %s", mode_str.upper(), feat_nc)
        if not feat_nc.exists():
            log.error("Feature file not found: %s", feat_nc)
            return {}
        ds = xr.open_dataset(feat_nc)
        from configs.temporal_config import TEMPORAL_FEATURE_NAMES
        feat_names = TEMPORAL_FEATURE_NAMES[mode]
        from titan.temporal_features import TemporalFeatureStack
        features = TemporalFeatureStack(
            mode     = mode,
            features = {n: ds[n].values.astype(np.float32) for n in feat_names if n in ds},
            grid     = grid,
        )
        ds.close()

    # ── Stage 4: Inference ─────────────────────────────────────────────────
    if not (args.skip_inference or args.only_visualise):
        log.info("")
        log.info("[%s] STAGE 4 — Bayesian Inference [%s]",
                 mode_str.upper(), cfg.bayesian_backend)
        t = Timer(f"Stage 4 [{mode_str}]")
        from titan.bayesian.temporal_inference import run_temporal_inference
        result = run_temporal_inference(features, cfg)
        log.info(
            "  [%s] %d valid pixels, backend=%s",
            mode_str, result.n_valid_pixels, result.backend,
        )
        log.info("  Top importances:")
        for name, imp in sorted(
            result.feature_importances.items(), key=lambda x: -x[1]
        )[:3]:
            log.info("    %-35s %.3f", name, imp)
        result.save(inf_dir)
        t.done(log)
    else:
        log.info("[%s] Stage 4 — loading inference from %s", mode_str.upper(), inf_dir)
        from titan.bayesian.inference import HabitabilityResult
        if not (inf_dir / "posterior_mean.npy").exists():
            log.error("Posterior not found in %s", inf_dir)
            return {}

        class _R:
            posterior_mean = np.load(inf_dir / "posterior_mean.npy")
            posterior_std  = np.load(inf_dir / "posterior_std.npy")
            hdi_low        = np.load(inf_dir / "hdi_low.npy")
            hdi_high       = np.load(inf_dir / "hdi_high.npy")
            feature_importances = json.loads(
                (inf_dir / "feature_importances.json").read_text()
            )
            backend = "loaded"
            n_valid_pixels = int(np.sum(np.isfinite(posterior_mean)))
        result = _R()

    # ── Stage 5: Visualisation ─────────────────────────────────────────────
    log.info("")
    log.info("[%s] STAGE 5 — Visualisation", mode_str.upper())
    t = Timer(f"Stage 5 [{mode_str}]")
    from titan.visualisation import generate_paper_figures, plot_interactive

    hdi_width = None
    if np.any(np.isfinite(result.hdi_high)) and np.any(np.isfinite(result.hdi_low)):
        hdi_width = np.where(
            np.isfinite(result.hdi_high) & np.isfinite(result.hdi_low),
            result.hdi_high - result.hdi_low, np.nan,
        ).astype(np.float32)

    # Create a FeatureStack-compatible object for visualisation
    fig_dir = mode_out / "figures"
    from titan.features import FeatureStack, FEATURE_NAMES

    # For visualisation, use only the PRESENT-style features where available
    vis_features = _make_vis_feature_stack(features, grid)
    if vis_features is not None:
        generate_paper_figures(
            posterior           = result.posterior_mean,
            posterior_std       = result.posterior_std,
            hdi_low             = result.hdi_low,
            hdi_high            = result.hdi_high,
            features            = vis_features,
            importances         = result.feature_importances,
            out_dir             = fig_dir,
            dpi                 = args.paper_dpi,
            annotate            = not args.no_labels,
            feature_names       = args.label_names,
            feature_categories  = args.label_categories,
        )
        plot_interactive(result.posterior_mean, fig_dir / "fig5_interactive.html",
                         title=f"Titan Habitability — {mode_str.upper()} mode")
        log.info("  Figures → %s", fig_dir)

    t.done(log)

    return {
        "mode":             mode_str,
        "posterior_mean":   result.posterior_mean,
        "posterior_std":    result.posterior_std,
        "importances":      result.feature_importances,
        "n_valid_pixels":   result.n_valid_pixels,
    }


def _make_vis_feature_stack(
    temporal_features: "titan.temporal_features.TemporalFeatureResult",
    grid: "titan.preprocessing.CanonicalGrid",
) -> "Optional[titan.features.FeatureStack]":
    """
    Build a visualisation-compatible FeatureStack from temporal features.

    Maps temporal feature names to PRESENT-mode names where possible,
    filling with NaN for features that don't have a PRESENT equivalent.
    """
    from titan.features import FeatureStack, FEATURE_NAMES
    import numpy as np

    nan = np.full((grid.nrows, grid.ncols), np.nan, dtype=np.float32)

    # Mapping: temporal name → PRESENT name
    ALIAS = {
        "water_ammonia_solvent":    "liquid_hydrocarbon",
        "organic_stockpile":        "organic_abundance",
        "dissolved_energy":         "acetylene_energy",
        "water_ammonia_cycle":      "methane_cycle",
        "global_ocean_habitability":"subsurface_ocean",
        "impact_melt_proxy":        "subsurface_ocean",   # closest analog
        "cryovolcanic_flux":        "subsurface_ocean",
    }
    tf = temporal_features.features
    kwargs = {}
    for present_name in FEATURE_NAMES:
        # Direct match
        if present_name in tf:
            kwargs[present_name] = tf[present_name]
        else:
            # Try alias
            found = None
            for temporal_name, alias in ALIAS.items():
                if alias == present_name and temporal_name in tf:
                    found = tf[temporal_name]
                    break
            kwargs[present_name] = found if found is not None else nan.copy()

    try:
        return FeatureStack(**kwargs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Comparison figure across all three modes
# ---------------------------------------------------------------------------

def make_comparison_figure(
    results: list,
    out_dir: Path,
    dpi:     int = 300,
) -> None:
    """
    Generate a 3-panel comparison figure: past / present / future.

    Parameters
    ----------
    results:
        List of result dicts from run_single_mode().
    out_dir:
        Output directory.
    dpi:
        Figure resolution.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    from titan.visualisation import _hab_cmap, _add_feature_labels

    valid = [r for r in results if r and "posterior_mean" in r]
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(8 * len(valid), 6), dpi=dpi)
    if len(valid) == 1:
        axes = [axes]

    cmap = _hab_cmap()
    for ax, res in zip(axes, valid):
        im = ax.imshow(
            res["posterior_mean"],
            origin="upper",
            extent=[0, 360, -90, 90],
            cmap=cmap, vmin=0, vmax=1, aspect="auto",
            interpolation="bilinear",
        )
        ax.set_title(
            f"{res['mode'].upper()} habitability\n"
            f"(n_valid={res['n_valid_pixels']:,})",
            fontsize=12,
        )
        ax.set_xlabel("Lon (°W)", fontsize=9)
        ax.set_ylabel("Lat (°)", fontsize=9)
        ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
        ax.grid(True, ls="--", alpha=0.2, color="white")
        _add_feature_labels(ax, fontsize=6, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.8).set_label(
            "P(habitable|data)", fontsize=8
        )

    fig.suptitle("Titan Habitability: Past vs Present vs Future", fontsize=14, y=1.02)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"temporal_comparison.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger("pipeline").info("Temporal comparison → %s", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args   = build_parser().parse_args(argv)
    log_dir = Path(args.output_dir) / "logs"
    setup_logging(args.verbose, log_dir)
    log = logging.getLogger("pipeline")

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from configs.pipeline_config import (
        PipelineConfig, HabitabilityWindowConfig, BayesianPriorConfig
    )

    # ── Build temporal window config from CLI args (D1, D2) ──────────────────
    window_cfg = HabitabilityWindowConfig(
        past_liquid_water_epoch_gya = args.past_epoch_gya,
        future_window_min_myr       = args.future_window_min,
        future_window_max_myr       = args.future_window_max,
        assume_uniform_warming      = not args.no_uniform_warming,
    )
    window_cfg.validate()

    # ── Build Bayesian prior config with D3 override ──────────────────────────
    prior_cfg = BayesianPriorConfig(
        prior_mean_subsurface_ocean = args.subsurface_ocean_prior,
    )
    prior_cfg.validate()

    cfg = PipelineConfig(
        data_dir             = args.data_dir,
        processed_dir        = args.processed_dir,
        output_dir           = args.output_dir,
        bayesian_backend     = args.backend,
        canonical_res_m      = args.res,
        mcmc_draws           = args.mcmc_draws,
        mcmc_chains          = args.mcmc_chains,
        random_seed          = args.seed,
        shapefile_dir        = args.shapefile_dir,
        birch_dir            = args.birch_dir,
        vims_parquet_path    = args.vims_parquet,
        priors               = prior_cfg,
        habitability_window  = window_cfg,
    )
    cfg.make_dirs()

    modes_to_run = (
        ["past", "present", "future"] if args.all_temporal_modes
        else [args.temporal_mode]
    )

    grid_rows, grid_cols = cfg.canonical_grid_shape
    log.info("=" * 65)
    log.info("  TITAN HABITABILITY PIPELINE")
    log.info("=" * 65)
    log.info("  Temporal mode(s)  : %s", ", ".join(modes_to_run))
    log.info("  Backend           : %s", cfg.bayesian_backend)
    log.info("  Resolution        : %.0f m/px", cfg.canonical_res_m)
    log.info("  Grid              : %d × %d px", grid_rows, grid_cols)
    log.info("  ── Temporal parameters (D1, D2) ──────────────────────")
    log.info(
        "  [D1] Past liquid-water epoch : %.2f Gya",
        cfg.habitability_window.past_liquid_water_epoch_gya,
    )
    log.info(
        "  [D2] Near-future warm window : %d–%d Myr from now",
        int(cfg.habitability_window.future_window_min_myr),
        int(cfg.habitability_window.future_window_max_myr),
    )
    log.info(
        "  [D2] Uniform warming assumed : %s",
        "YES (explicit assumption: spatially constant ΔT)"
        if cfg.habitability_window.assume_uniform_warming
        else "NO (future-window prior disabled)",
    )
    log.info(
        "  [D3] Subsurface ocean prior  : %.3f  "
        "(Neish et al. 2024: ~1 elephant/yr organic flux to ocean)",
        cfg.priors.prior_mean_subsurface_ocean,
    )
    log.info("=" * 65)

    # Status mode
    if args.status:
        from titan.acquisition import DataAcquisitionManager
        DataAcquisitionManager(cfg).status().print_summary()
        return 0

    t_total = time.perf_counter()

    # ── Stage 1: Acquisition ─────────────────────────────────────────────
    if not (args.skip_acquisition or args.only_visualise):
        log.info("\nSTAGE 1 — Data Acquisition")
        t = Timer("Stage 1")
        from titan.acquisition import DataAcquisitionManager
        mgr = DataAcquisitionManager(cfg)
        mgr.create_gravity_k2_json()
        report = mgr.acquire_all()
        report.print_summary()
        report.save(cfg.output_dir / "acquisition_report.json")
        t.done(log)

    # ── Stage 2: Preprocessing ────────────────────────────────────────────
    if not (args.skip_preprocessing or args.only_visualise):
        log.info("\nSTAGE 2 — Preprocessing → canonical %.0f m/px grid",
                 cfg.canonical_res_m)
        t = Timer("Stage 2")
        from titan.preprocessing import DataPreprocessor, CanonicalGrid
        grid = CanonicalGrid(cfg.canonical_res_m)
        proc = DataPreprocessor(cfg, grid)
        processed = proc.preprocess_all(overwrite=args.overwrite)
        log.info("  Layers produced: %s", list(processed.keys()))
        t.done(log)

    # ── Load canonical stack ──────────────────────────────────────────────
    import numpy as np
    import xarray as xr
    from titan.preprocessing import CanonicalDataStack, CanonicalGrid

    grid   = CanonicalGrid(cfg.canonical_res_m)
    loader = CanonicalDataStack(cfg, grid)
    log.info("\nLoading canonical data stack …")
    stack  = loader.load()
    if stack.data_vars:
        log.info("  Layers: %s", list(stack.data_vars.keys()))
    else:
        log.warning("  No canonical layers found — stages 3–5 will use priors only.")
    nc_path = cfg.processed_dir / "titan_canonical_stack.nc"
    if (not nc_path.exists() or args.overwrite) and stack.data_vars:
        loader.save_netcdf(stack, nc_path)

    # ── Stages 3–5 per temporal mode ─────────────────────────────────────
    all_results = []
    for mode_str in modes_to_run:
        result = run_single_mode(mode_str, args, cfg, stack, grid, log)
        all_results.append(result)

    # ── Comparison figure (only when running multiple modes) ──────────────
    if len(modes_to_run) > 1:
        log.info("\nGenerating temporal comparison figure …")
        make_comparison_figure(
            all_results,
            out_dir=cfg.output_dir / "temporal_comparison",
            dpi=args.paper_dpi,
        )

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    log.info("")
    log.info("=" * 65)
    log.info(
        "  PIPELINE COMPLETE  (%.1f min)  modes=%s  backend=%s",
        elapsed / 60, modes_to_run, cfg.bayesian_backend,
    )
    log.info("  Outputs → %s", cfg.output_dir.resolve())
    log.info("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
