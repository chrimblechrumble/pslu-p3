"""
titan/bayesian/temporal_inference.py
=====================================
Adapts the Bayesian inference engine to work with any temporal mode.

The core inference logic (SklearnHabitabilityModel etc.) is unchanged.
This module provides a thin adapter that:
  1. Converts a TemporalFeatureStack to the format expected by the backends
  2. Injects the correct temporal priors
  3. Returns a HabitabilityResult as before
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from configs.temporal_config import TemporalMode, get_prior_set
from configs.pipeline_config import PipelineConfig, BayesianPriorConfig
from titan.temporal_features import TemporalFeatureStack
from titan.bayesian.inference import (
    HabitabilityResult,
    SklearnHabitabilityModel,
    PyMCHabitabilityModel,
    NumPyroHabitabilityModel,
)

logger = logging.getLogger(__name__)


def _temporal_prior_config(mode: TemporalMode) -> BayesianPriorConfig:
    """
    Build a BayesianPriorConfig from the temporal prior set.

    Since BayesianPriorConfig has fixed field names for the 8 PRESENT
    features, we create a minimal compatible version.  For PAST/FUTURE
    modes with different feature sets, we store the priors in the
    temporal prior set and handle them externally in the inference loop.

    For the sklearn backend, the weighted_sum computation is done from
    the TemporalPriorSet directly, bypassing BayesianPriorConfig.
    This function is only needed for the PyMC/NumPyro backends.
    """
    prior_set = get_prior_set(mode)
    # Create a minimal BayesianPriorConfig with defaults
    cfg = BayesianPriorConfig()
    # Override the threshold only — weights and means are from temporal prior set
    return cfg


def run_temporal_inference(
    temporal_features: TemporalFeatureStack,
    config:            PipelineConfig,
) -> HabitabilityResult:
    """
    Run Bayesian inference for any temporal mode.

    Uses the temporal prior set appropriate to the mode, rather than
    the default PRESENT priors in PipelineConfig.

    Parameters
    ----------
    temporal_features:
        TemporalFeatureStack from TemporalFeatureExtractor.extract().
    config:
        Pipeline configuration (used for backend selection, MCMC params).

    Returns
    -------
    HabitabilityResult
    """
    mode = temporal_features.mode
    prior_set = get_prior_set(mode)
    prior_set.validate()

    logger.info(
        "Running temporal inference: mode=%s, backend=%s, features=%s",
        mode.value, config.bayesian_backend, temporal_features.feature_names(),
    )

    backend = config.bayesian_backend.lower()
    if backend == "sklearn":
        return _sklearn_temporal(temporal_features, prior_set, config)
    elif backend == "pymc":
        return _pymc_temporal(temporal_features, prior_set, config)
    elif backend == "numpyro":
        return _numpyro_temporal(temporal_features, prior_set, config)
    else:
        raise ValueError(f"Unknown backend: {config.bayesian_backend}")


def _sklearn_temporal(
    temporal_features: TemporalFeatureStack,
    prior_set:         "TemporalPriorSet",
    config:            PipelineConfig,
) -> HabitabilityResult:
    """sklearn GNB inference with temporal priors."""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV

    weights = prior_set.as_weight_dict()
    means   = prior_set.as_mean_dict()
    names   = temporal_features.feature_names()
    cfg     = config.priors  # for gnb_var_smoothing, threshold, alpha

    # Stack features into (n_pixels, n_features)
    arr_3d  = temporal_features.as_array()      # (F, H, W)
    nfeats, nrows, ncols = arr_3d.shape
    arr_2d  = arr_3d.reshape(nfeats, -1).T       # (N, F)

    valid   = np.all(np.isfinite(arr_2d), axis=1)
    X_valid = arr_2d[valid]

    if X_valid.shape[0] < 100:
        logger.warning(
            "Only %d fully-valid pixels for mode=%s. Results unreliable.",
            X_valid.shape[0], prior_set.mode.value,
        )

    # Soft labels from weighted feature sum.
    # Use float64 for w_vec and matmul; wrap in errstate to suppress spurious
    # BLAS floating-point exception flags on large matrices with finite inputs.
    w_vec = np.array([weights[n] for n in names], dtype=np.float64)
    X_f64 = X_valid.astype(np.float64)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        scores = X_f64 @ w_vec
    threshold = cfg.positive_label_threshold
    y_label = (scores > threshold).astype(int)

    n_pos = int(y_label.sum())
    n_neg = int((~y_label.astype(bool)).sum())
    logger.info(
        "[%s] Labels: %d positive, %d negative, %.1f%% positive",
        prior_set.mode.value, n_pos, n_neg,
        100.0 * n_pos / len(y_label) if len(y_label) > 0 else 0.0,
    )

    # Handle degenerate case
    if n_pos == 0 or n_neg == 0:
        logger.warning("[%s] Degenerate labels; returning flat posterior.", prior_set.mode.value)
        posterior = np.full(nrows * ncols, np.nan, dtype=np.float32)
        posterior[valid] = float(n_pos > 0)
        std = np.full(nrows * ncols, np.nan, dtype=np.float32)
        return HabitabilityResult(
            posterior_mean      = posterior.reshape(nrows, ncols),
            posterior_std       = std.reshape(nrows, ncols),
            hdi_low             = std.reshape(nrows, ncols),
            hdi_high            = std.reshape(nrows, ncols),
            feature_importances = {n: w for n, w in zip(names, w_vec)},
            backend             = f"sklearn[{prior_set.mode.value}]",
            n_valid_pixels      = int(valid.sum()),
        )

    # Fit GNB with isotonic calibration
    gnb = GaussianNB(var_smoothing=cfg.gnb_var_smoothing)
    cal = CalibratedClassifierCV(gnb, cv=min(3, min(n_pos, n_neg)), method="isotonic")
    cal.fit(X_valid, y_label)

    proba_raw = cal.predict_proba(X_valid)[:, 1]

    # Prior correction: blend with temporal prior means
    prior_mean_global = float(
        np.array([means[n] for n in names], dtype=np.float64) @ w_vec
    )
    alpha = 0.8
    proba = np.clip(alpha * proba_raw + (1.0 - alpha) * prior_mean_global, 0, 1)

    # Permutation importances
    importances = _permutation_importances(cal, X_valid, y_label, names, w_vec)

    posterior = np.full(nrows * ncols, np.nan, dtype=np.float32)
    posterior[valid] = proba.astype(np.float32)
    std = np.full(nrows * ncols, np.nan, dtype=np.float32)

    return HabitabilityResult(
        posterior_mean      = posterior.reshape(nrows, ncols),
        posterior_std       = std.reshape(nrows, ncols),
        hdi_low             = std.reshape(nrows, ncols),
        hdi_high            = std.reshape(nrows, ncols),
        feature_importances = importances,
        backend             = f"sklearn[{prior_set.mode.value}]",
        n_valid_pixels      = int(valid.sum()),
    )


def _permutation_importances(
    cal:   "sklearn.calibration.CalibratedClassifierCV",
    X:     np.ndarray,
    y:     np.ndarray,
    names: list[str],
    w_vec: np.ndarray,
) -> dict[str, float]:
    """
    Estimate feature importances via permutation of each feature column.

    Parameters
    ----------
    cal:
        Fitted calibrated classifier exposing ``predict_proba``.
    X:
        Feature matrix, shape (n_samples, n_features).
    y:
        Binary target labels, shape (n_samples,).  Not used in computation
        but kept for API consistency with sklearn convention.
    names:
        Feature names in the same column order as *X*.
    w_vec:
        Prior weight vector used as fallback if all deltas are zero.

    Returns
    -------
    dict[str, float]
        Mapping of feature name → normalised importance (sums to 1).
    """
    base: float = cal.predict_proba(X)[:, 1].mean()
    deltas: dict[str, float] = {}
    for i, name in enumerate(names):
        Xp: np.ndarray = X.copy()
        Xp[:, i]       = X[:, i].mean()
        perturbed: float = cal.predict_proba(Xp)[:, 1].mean()
        deltas[name]   = abs(base - perturbed)
    total: float = sum(deltas.values())
    if total > 0:
        return {k: v / total for k, v in deltas.items()}
    return {n: float(w) for n, w in zip(names, w_vec)}


def _pymc_temporal(
    temporal_features: TemporalFeatureStack,
    prior_set:         "TemporalPriorSet",
    config:            PipelineConfig,
) -> HabitabilityResult:
    """PyMC inference with temporal priors.  Delegates to sklearn on ImportError."""
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        logger.warning("PyMC not available; falling back to sklearn for temporal inference.")
        return _sklearn_temporal(temporal_features, prior_set, config)

    # Reuse PyMCHabitabilityModel but inject temporal priors
    weights = prior_set.as_weight_dict()
    means   = prior_set.as_mean_dict()
    names   = temporal_features.feature_names()

    # Build a minimal prior config that passes to PyMC model
    temp_cfg = BayesianPriorConfig()
    temp_cfg = _inject_temporal_priors(temp_cfg, names, weights, means)

    model = PyMCHabitabilityModel(
        prior_config=temp_cfg,
        draws=config.mcmc_draws,
        tune=config.mcmc_tune,
        chains=config.mcmc_chains,
        seed=config.random_seed,
    )
    # TemporalFeatureStack is duck-type compatible with FeatureStack for fit_predict:
    # both expose .as_array() returning (n_features, nrows, ncols) float32.
    result = model.fit_predict(temporal_features)  # type: ignore[arg-type]
    result.backend = f"pymc[{prior_set.mode.value}]"
    return result


def _numpyro_temporal(
    temporal_features: TemporalFeatureStack,
    prior_set:         "TemporalPriorSet",
    config:            PipelineConfig,
) -> HabitabilityResult:
    """NumPyro inference — falls back to sklearn on import error."""
    try:
        import numpyro  # noqa
    except ImportError:
        logger.warning("NumPyro not available; falling back to sklearn for temporal inference.")
        return _sklearn_temporal(temporal_features, prior_set, config)

    from titan.bayesian.inference import NumPyroHabitabilityModel
    weights = prior_set.as_weight_dict()
    means   = prior_set.as_mean_dict()
    names   = temporal_features.feature_names()
    temp_cfg = BayesianPriorConfig()
    temp_cfg = _inject_temporal_priors(temp_cfg, names, weights, means)

    model  = NumPyroHabitabilityModel(
        prior_config=temp_cfg,
        draws=config.mcmc_draws,
        warmup=config.mcmc_tune,
        chains=config.mcmc_chains,
        seed=config.random_seed,
    )
    result = model.fit_predict(temporal_features)  # type: ignore[arg-type]
    result.backend = f"numpyro[{prior_set.mode.value}]"
    return result


def _inject_temporal_priors(
    cfg:     BayesianPriorConfig,
    names:   list[str],
    weights: dict[str, float],
    means:   dict[str, float],
) -> "BayesianPriorConfig":
    """
    Return a duck-typed object that satisfies the BayesianPriorConfig interface
    with temporal feature weights and prior means injected.

    BayesianPriorConfig uses fixed field names derived from the present-epoch
    feature set.  For temporal modes the features differ, so we wrap the config
    in a lightweight inner class that overrides ``feature_weights()`` and
    ``prior_means()`` while preserving all other config fields.

    Parameters
    ----------
    cfg:
        Base config whose scalar fields (gnb_var_smoothing, etc.) are copied.
    names:
        Ordered list of temporal feature names.
    weights:
        Dict mapping each feature name to its prior weight.
    means:
        Dict mapping each feature name to its prior mean.

    Returns
    -------
    BayesianPriorConfig
        Duck-typed object compatible with the sklearn/PyMC/NumPyro backends.
    """
    class _TemporalCfg:
        gnb_var_smoothing:        float = cfg.gnb_var_smoothing
        positive_label_threshold: float = cfg.positive_label_threshold
        likelihood_sharpness:     float = cfg.likelihood_sharpness
        _weights: dict[str, float] = weights
        _means:   dict[str, float] = means
        _names:   list[str]        = names

        def feature_weights(self) -> dict[str, float]:
            return dict(zip(self._names,
                            [self._weights[n] for n in self._names]))

        def prior_means(self) -> dict[str, float]:
            return dict(zip(self._names,
                            [self._means[n] for n in self._names]))

        def validate(self) -> None:
            pass

    return _TemporalCfg()  # type: ignore[return-value]
