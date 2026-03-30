"""
titan/visualisation.py
=======================
Stage 5 — Visualisation.

Produces publication-quality figures and interactive maps from the
canonical data stack, feature maps, and Bayesian inference results.

All maps use Titan's SimpleCylindrical projection and display longitude
in west-positive convention (0→360°W) to match the data products.

Output formats
--------------
  Static (matplotlib, 300 dpi):
    - Fig 1: Global habitability posterior map (main result)
    - Fig 2: Uncertainty map (HDI width or posterior std)
    - Fig 3: Feature importance bar chart
    - Fig 4: Eight-panel feature map grid
    - Fig 5: Top-20 highest-probability sites

  Interactive (plotly HTML):
    - Zoomable posterior map with named feature overlays

Named features annotated on all maps
--------------------------------------
Lakes/seas, terrain regions, Huygens landing site, and Dragonfly target
are annotated using IAU nomenclature and published coordinates.

References
----------
Hayes et al. (2008)   doi:10.1029/2007GL032324  (lake coordinates)
Neish et al. (2018)   doi:10.1089/ast.2017.1758  (Dragonfly/Selk target)
Lorenz et al. (2013)  doi:10.1016/j.icarus.2013.04.002  (topography)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named geographic features  (lon_W°, lat°)
# ---------------------------------------------------------------------------

#: Full catalogue of named geographic features on Titan.
#:
#: Each entry is ``name → (lon_W°, lat°, category)`` where category is one of:
#:
#: ``"sea"``
#:     Major hydrocarbon seas (maria).
#:     Source: Stofan et al. (2007) doi:10.1038/nature05608;
#:             Hayes et al. (2008) doi:10.1029/2007GL032324
#:
#: ``"lake"``
#:     Smaller hydrocarbon lakes (lacus).
#:     Source: Hayes et al. (2008); Turtle et al. (2009)
#:
#: ``"terrain"``
#:     Named albedo/terrain regions (regiones, planitiae).
#:     Source: IAU nomenclature; Lopes et al. (2019) doi:10.1038/s41550-019-0917-6
#:
#: ``"mission"``
#:     Spacecraft landing sites and mission targets.
#:     Huygens: Lebreton et al. (2005) doi:10.1038/nature04347
#:     Selk (Dragonfly): Barnes et al. (2021) doi:10.3847/PSJ/abf7c9
#:
#: To customise which labels appear, pass a subset to
#: :class:`TitanMapPlotter` via ``feature_categories`` or ``feature_names``.
TITAN_FEATURES: Dict[str, Tuple[float, float, str]] = {
    # ── Hydrocarbon seas (maria) ─────────────────────────────────────────────
    "Kraken Mare":   (310.0,  68.0, "sea"),    # largest, ~400,000 km²
    "Ligeia Mare":   ( 78.0,  79.0, "sea"),    # second largest, liquid methane-rich
    "Punga Mare":    ( 17.0,  85.0, "sea"),    # northernmost major sea
    # ── Hydrocarbon lakes (lacus) ────────────────────────────────────────────
    "Ontario Lacus": (180.0, -72.0, "lake"),   # largest southern hemisphere lake
    # ── Terrain regions ──────────────────────────────────────────────────────
    "Xanadu":        (100.0,  -5.0, "terrain"),  # bright radar/IR region, water-ice
    "Shangri-La":    (160.0,  -5.0, "terrain"),  # large equatorial dune sea
    "Belet":         (250.0,   5.0, "terrain"),  # equatorial dune sea
    # ── Mission sites ────────────────────────────────────────────────────────
    "Huygens":       (192.0, -10.2, "mission"),  # ESA probe landing site, Jan 2005
    "Selk (DFly)":   (199.0,   5.0, "mission"),  # NASA Dragonfly target, ~80 km crater
}

#: Default colours and markers per feature category.
#: Override via ``TitanMapPlotter.category_styles``.
CATEGORY_STYLES: Dict[str, Dict[str, object]] = {
    "sea":     {"color": "cyan",    "marker": "^", "fontsize": 7},
    "lake":    {"color": "cyan",    "marker": "v", "fontsize": 7},
    "terrain": {"color": "white",   "marker": "+", "fontsize": 7},
    "mission": {"color": "#FFD700", "marker": "*", "fontsize": 7},  # gold stars
}


def _hab_cmap() -> "matplotlib.colors.LinearSegmentedColormap":
    """Custom perceptually-uniform habitability colourmap."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "titan_hab",
        [(0.10, 0.05, 0.20), (0.10, 0.40, 0.55),
         (0.25, 0.75, 0.60), (0.95, 0.90, 0.10)],
    )


# ---------------------------------------------------------------------------
# Global map helper
# ---------------------------------------------------------------------------

def _base_map(
    ax: "Axes",
    data: np.ndarray,
    cmap: Optional[object] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: str = "",
    cbar_label: str = "",
    annotate: bool = True,
    interp: str = "bilinear",
    feature_names: Optional[List[str]] = None,
    feature_categories: Optional[List[str]] = None,
    category_styles: Optional[Dict[str, Dict[str, object]]] = None,
) -> "AxesImage":
    """
    Render a single Titan global map panel.

    Parameters
    ----------
    ax:
        Matplotlib Axes to draw into.
    data:
        2-D float32 array (nrows, ncols), values in [0, 1].
    cmap:
        Matplotlib colourmap.  Defaults to the custom habitability colourmap.
    vmin, vmax:
        Colourscale limits.
    title:
        Panel title text.
    cbar_label:
        Colourbar axis label.
    annotate:
        If ``True``, overlay named geographic feature labels.
    interp:
        Matplotlib interpolation mode (``"nearest"`` for feature maps,
        ``"bilinear"`` for smooth posterior maps).
    feature_names:
        Passed to :func:`_add_feature_labels` — explicit list of names to show.
    feature_categories:
        Passed to :func:`_add_feature_labels` — filter labels by category.
    category_styles:
        Passed to :func:`_add_feature_labels` — override per-category style.

    Returns
    -------
    AxesImage
        The imshow image object (useful for shared colourbar creation).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if cmap is None:
        cmap = _hab_cmap()

    im = ax.imshow(
        data, origin="upper",
        extent=[0, 360, -90, 90],
        cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="auto", interpolation=interp,
    )
    ax.set_xlabel("Longitude (°W)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.grid(True, ls="--", alpha=0.25, color="white")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.85)
    cb.set_label(cbar_label, fontsize=9)
    if annotate:
        _add_feature_labels(
            ax,
            feature_names=feature_names,
            feature_categories=feature_categories,
            category_styles=category_styles,
        )
    return im


def _add_feature_labels(
    ax: "Axes",
    fontsize: int = 7,
    alpha: float = 0.85,
    feature_names: Optional[List[str]] = None,
    feature_categories: Optional[List[str]] = None,
    category_styles: Optional[Dict[str, Dict[str, object]]] = None,
) -> None:
    """
    Annotate named geographic features on a Titan map axis.

    Parameters
    ----------
    ax:
        Matplotlib Axes to annotate.
    fontsize:
        Label font size in points (default 7).
    alpha:
        Marker and text opacity (default 0.85).
    feature_names:
        Explicit list of feature names to show.  If ``None``, all features
        whose category passes the ``feature_categories`` filter are shown.
        Names not in :data:`TITAN_FEATURES` are silently skipped.
        Example: ``["Kraken Mare", "Selk (DFly)"]``
    feature_categories:
        List of category strings to include: ``"sea"``, ``"lake"``,
        ``"terrain"``, ``"mission"``.  If ``None``, all categories are shown.
        Ignored when ``feature_names`` is explicitly provided.
        Example: ``["sea", "lake"]``  — show only liquid bodies.
    category_styles:
        Override the default per-category appearance.  Dict maps category
        name → dict with keys ``"color"``, ``"marker"``, ``"fontsize"``.
        Merged with :data:`CATEGORY_STYLES`; only specified keys are overridden.
        Example: ``{"mission": {"color": "red"}}``
    """
    styles = {cat: dict(props) for cat, props in CATEGORY_STYLES.items()}
    if category_styles:
        for cat, overrides in category_styles.items():
            if cat in styles:
                styles[cat].update(overrides)
            else:
                styles[cat] = dict(overrides)

    for name, entry in TITAN_FEATURES.items():
        lon, lat, category = entry

        # Filter by explicit name list
        if feature_names is not None and name not in feature_names:
            continue

        # Filter by category (only when name list not given)
        if feature_names is None and feature_categories is not None:
            if category not in feature_categories:
                continue

        style = styles.get(category, {"color": "white", "marker": "+", "fontsize": 7})
        colour  = str(style.get("color",    "white"))
        marker  = str(style.get("marker",   "+"))
        fs      = int(style.get("fontsize", fontsize))
        ms      = 5 if marker == "*" else 4

        ax.plot(lon, lat, marker, color=colour, ms=ms, alpha=alpha)
        ax.text(lon + 1.5, lat, name, fontsize=fs,
                color=colour, alpha=alpha, va="center")


# ---------------------------------------------------------------------------
# Publication figure suite
# ---------------------------------------------------------------------------

class TitanMapPlotter:
    """
    Creates publication-quality Titan map figures.

    Parameters
    ----------
    dpi:
        Output resolution (300 for publication, 150 for screen).
    annotate:
        If ``True``, overlay named geographic feature labels on all maps.
        Set ``False`` to suppress all labels globally.
    feature_names:
        Explicit list of feature names to label (e.g. ``["Kraken Mare",
        "Selk (DFly)"]``).  When provided, only these labels appear.
        When ``None``, ``feature_categories`` controls the filter.
    feature_categories:
        List of category strings to include: ``"sea"``, ``"lake"``,
        ``"terrain"``, ``"mission"``.  ``None`` (default) shows all categories.
        Example: ``["sea", "lake"]`` shows only liquid bodies.
        Ignored when ``feature_names`` is explicitly provided.
    category_styles:
        Override default per-category appearance.  Dict maps category →
        ``{"color": ..., "marker": ..., "fontsize": ...}``.
        Merged with :data:`CATEGORY_STYLES`.

    Examples
    --------
    Show all labels (default)::

        plotter = TitanMapPlotter()

    Suppress all labels::

        plotter = TitanMapPlotter(annotate=False)

    Show only seas and lakes::

        plotter = TitanMapPlotter(feature_categories=["sea", "lake"])

    Show only mission sites, in red::

        plotter = TitanMapPlotter(
            feature_categories=["mission"],
            category_styles={"mission": {"color": "red", "marker": "*"}},
        )

    Show exactly two named features::

        plotter = TitanMapPlotter(feature_names=["Kraken Mare", "Selk (DFly)"])
    """

    def __init__(
        self,
        dpi:                int = 150,
        annotate:           bool = True,
        feature_names:      Optional[List[str]] = None,
        feature_categories: Optional[List[str]] = None,
        category_styles:    Optional[Dict[str, Dict[str, object]]] = None,
    ) -> None:
        self.dpi                = dpi
        self.annotate           = annotate
        self.feature_names      = feature_names
        self.feature_categories = feature_categories
        self.category_styles    = category_styles

    # ── Fig 1: Posterior mean (± uncertainty panel) ────────────────────────

    def plot_posterior(
        self,
        posterior:  np.ndarray,
        hdi_width:  Optional[np.ndarray] = None,
        out_path:   Optional[Path] = None,
        title:      str = "Titan Habitability Proxy — Posterior P(habitable | data)",
    ) -> "Figure":
        import matplotlib.pyplot as plt
        ncols = 2 if (hdi_width is not None and np.any(np.isfinite(hdi_width))) else 1
        fig, axes = plt.subplots(1, ncols, figsize=(14 * ncols / 1.4, 6), dpi=self.dpi)
        if ncols == 1:
            axes = [axes]

        _base_map(axes[0], posterior, cbar_label="P(habitable | data)",
                  title=title, annotate=self.annotate,
                  feature_names=self.feature_names,
                  feature_categories=self.feature_categories,
                  category_styles=self.category_styles)

        if ncols == 2:
            _base_map(axes[1], hdi_width, cmap="plasma_r",
                      cbar_label="94% HDI width",
                      title="Uncertainty", annotate=self.annotate,
                      feature_names=self.feature_names,
                      feature_categories=self.feature_categories,
                      category_styles=self.category_styles)

        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Posterior map → %s", out_path)
        return fig

    # ── Fig 2: 8-panel feature maps ───────────────────────────────────────

    def plot_features(
        self,
        features: "FeatureStack",
        out_path: Optional[Path] = None,
    ) -> "Figure":
        import matplotlib.pyplot as plt
        from titan.features import FEATURE_NAMES

        TITLES = {
            "liquid_hydrocarbon":       "Liquid Hydrocarbon\n(lake/sea proxy)",
            "organic_abundance":        "Organic Abundance\n(VIMS/tholin)",
            "acetylene_energy":         "Chemical Energy\n(C₂H₂+H₂ gradient)",
            "methane_cycle":            "Methane Cycle\n(solvent transport)",
            "surface_atm_interaction":  "Surface–Atm\nInteraction",
            "topographic_complexity":   "Topographic\nComplexity",
            "geomorphologic_diversity": "Terrain\nDiversity",
            "subsurface_ocean":         "Subsurface Ocean\n(k₂ proxy)",
        }

        fig, axes = plt.subplots(2, 4, figsize=(20, 8), dpi=self.dpi)
        # Features with typically sparse signals need auto-scaled colorbars
        # so that non-zero values are visible (fixed vmax=1.0 renders them black)
        SPARSE_FEATURES = {"liquid_hydrocarbon", "geomorphologic_diversity"}
        for i, name in enumerate(FEATURE_NAMES):
            ax  = axes.flat[i]
            arr = getattr(features, name)
            # For sparse features use 99.5th-percentile as vmax so small signals
            # are visible; for all others use fixed [0,1]
            if name in SPARSE_FEATURES:
                finite = arr[np.isfinite(arr)]
                vmax = float(np.percentile(finite, 99.5)) if len(finite) > 0 else 1.0
                vmax = max(vmax, 1e-3)  # guard against all-zero arrays
            else:
                vmax = 1.0
            _base_map(ax, arr, cmap="inferno",
                      title=TITLES[name],
                      cbar_label="[0–1]",
                      vmax=vmax,
                      annotate=self.annotate and i == 0,
                      interp="nearest",
                      feature_names=self.feature_names,
                      feature_categories=self.feature_categories,
                      category_styles=self.category_styles)
        fig.suptitle("Titan Habitability Feature Maps", fontsize=14, y=1.01)
        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Feature maps → %s", out_path)
        return fig

    # ── Fig 3: Feature importances ────────────────────────────────────────

    def plot_importances(
        self,
        importances: Dict[str, float],
        out_path: Optional[Path] = None,
    ) -> "Figure":
        import matplotlib.pyplot as plt

        LABELS = {
            "liquid_hydrocarbon":       "Liquid HC surface",
            "organic_abundance":        "Organic abundance",
            "acetylene_energy":         "Chem. energy (C₂H₂)",
            "methane_cycle":            "Methane cycle",
            "surface_atm_interaction":  "Surface–atm exchange",
            "topographic_complexity":   "Topographic complexity",
            "geomorphologic_diversity": "Terrain diversity",
            "subsurface_ocean":         "Subsurface ocean",
        }

        names  = list(importances.keys())
        values = [importances[n] for n in names]
        order  = np.argsort(values)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)
        bars = ax.barh(
            [LABELS.get(names[i], names[i]) for i in order],
            [values[i] for i in order],
            color="#2d7eb5", edgecolor="white",
        )
        ax.set_xlabel("Relative importance")
        ax.set_title("Feature Importances — Titan Habitability Model")
        for bar, val in zip(bars, sorted(values)):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ── Fig 4: Top sites ──────────────────────────────────────────────────

    def plot_top_sites(
        self,
        posterior: np.ndarray,
        top_n:     int = 20,
        out_path:  Optional[Path] = None,
    ) -> "Figure":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)
        _base_map(ax, posterior, cbar_label="P(habitable|data)",
                  title=f"Top {top_n} Highest-Probability Habitability Sites",
                  annotate=self.annotate,
                  feature_names=self.feature_names,
                  feature_categories=self.feature_categories,
                  category_styles=self.category_styles)

        nrows, ncols = posterior.shape
        flat  = posterior.flatten()
        valid = np.where(np.isfinite(flat))[0]
        top   = valid[np.argsort(flat[valid])[-top_n:]]

        for idx in top:
            row = idx // ncols
            col = idx % ncols
            lon = col * 360.0 / ncols
            lat = 90.0 - row * 180.0 / nrows
            ax.plot(lon, lat, "w*", ms=9, mec="black", lw=0.5, mew=0.5)

        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Top sites → %s", out_path)
        return fig


# ---------------------------------------------------------------------------
# Interactive Plotly map
# ---------------------------------------------------------------------------

def plot_interactive(
    posterior: np.ndarray,
    out_path:  Optional[Path] = None,
    title:     str = "Titan Habitability (Interactive)",
) -> None:
    """
    Write an interactive HTML habitability map using Plotly.

    Parameters
    ----------
    posterior:
        Posterior mean map.
    out_path:
        Output path.  Defaults to ``outputs/posterior_interactive.html``.
    title:
        Plot title.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed; skipping interactive map.")
        return

    nrows, ncols = posterior.shape
    step = max(1, min(nrows, ncols) // 500)
    z = posterior[::step, ::step]
    x = np.linspace(0, 360, z.shape[1])
    y = np.linspace(90, -90, z.shape[0])

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[
            [0.0, "rgb(25,13,51)"], [0.3, "rgb(26,102,140)"],
            [0.6, "rgb(64,191,153)"], [1.0, "rgb(242,229,26)"],
        ],
        zmin=0, zmax=1,
        colorbar=dict(title="P(habitable|data)"),
    ))
    lons = [v[0] for v in TITAN_FEATURES.values()]
    lats = [v[1] for v in TITAN_FEATURES.values()]
    fig.add_trace(go.Scatter(
        x=lons, y=lats, mode="markers+text",
        text=list(TITAN_FEATURES.keys()), textposition="top right",
        marker=dict(symbol="star", size=8, color="white",
                    line=dict(color="black", width=1)),
        name="Features",
    ))
    fig.update_layout(
        title=title, template="plotly_dark", width=1200, height=600,
        xaxis=dict(title="Longitude (°W)", range=[0, 360]),
        yaxis=dict(title="Latitude (°)", range=[-90, 90]),
    )
    if out_path is None:
        out_path = Path("outputs/posterior_interactive.html")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    logger.info("Interactive map → %s", out_path)


# ---------------------------------------------------------------------------
# Full paper figure suite
# ---------------------------------------------------------------------------

def generate_paper_figures(
    posterior:          np.ndarray,
    posterior_std:      np.ndarray,
    hdi_low:            np.ndarray,
    hdi_high:           np.ndarray,
    features:           "FeatureStack",
    importances:        Dict[str, float],
    out_dir:            Path,
    dpi:                int = 300,
    annotate:           bool = True,
    feature_names:      Optional[List[str]] = None,
    feature_categories: Optional[List[str]] = None,
    category_styles:    Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Path]:
    """
    Generate the complete suite of publication-ready figures at 300 dpi.

    Parameters
    ----------
    posterior, posterior_std, hdi_low, hdi_high:
        From HabitabilityResult.
    features:
        From FeatureExtractor.extract().
    importances:
        From HabitabilityResult.feature_importances.
    out_dir:
        Output directory.
    dpi:
        Resolution (300 for publication, 150 for screen).
    annotate:
        If ``True`` (default), overlay geographic feature labels on all maps.
        Pass ``False`` to suppress all labels (e.g. for clean figure export).
    feature_names:
        Explicit list of feature names to label.  ``None`` = use
        ``feature_categories`` filter.  See :class:`TitanMapPlotter`.
    feature_categories:
        List of categories to label: ``"sea"``, ``"lake"``, ``"terrain"``,
        ``"mission"``.  ``None`` = all categories.
    category_styles:
        Override per-category label appearance.  See :class:`TitanMapPlotter`.

    Returns
    -------
    List[Path]
        Paths of all generated files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = TitanMapPlotter(
        dpi=dpi,
        annotate=annotate,
        feature_names=feature_names,
        feature_categories=feature_categories,
        category_styles=category_styles,
    )
    paths: List[Path] = []

    hdi_width = None
    if np.any(np.isfinite(hdi_high)) and np.any(np.isfinite(hdi_low)):
        hdi_width = np.where(
            np.isfinite(hdi_high) & np.isfinite(hdi_low),
            hdi_high - hdi_low, np.nan,
        ).astype(np.float32)

    for ext in ("pdf", "png"):
        p = out_dir / f"fig1_posterior.{ext}"
        plotter.plot_posterior(posterior, hdi_width=hdi_width, out_path=p)
        paths.append(p)

    p = out_dir / "fig2_importances.pdf"
    plotter.plot_importances(importances, out_path=p)
    paths.append(p)

    p = out_dir / "fig3_features.pdf"
    plotter.plot_features(features, out_path=p)
    paths.append(p)

    p = out_dir / "fig4_top_sites.pdf"
    plotter.plot_top_sites(posterior, top_n=20, out_path=p)
    paths.append(p)

    plot_interactive(posterior, out_dir / "fig5_interactive.html")
    paths.append(out_dir / "fig5_interactive.html")

    logger.info("Generated %d paper figures in %s", len(paths), out_dir)
    return paths
