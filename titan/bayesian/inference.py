"""
titan/bayesian/inference.py
============================
Stage 4 — Bayesian Habitability Inference.

Produces a pixel-wise posterior probability map P(habitable | data)
from the eight feature maps extracted in Stage 3.

Model
-----
Following Catling et al. (2018) and Affholder et al. (2021) we frame
the problem as:

    P(H | D) ∝ P(D | H) · P(H)

where:
  H  = latent habitability score ∈ [0,1] per pixel
  D  = vector of 8 observed feature values at each pixel

Three backend implementations are available:

  sklearn  (default)
    Gaussian Naive Bayes with isotonic calibration.
    Self-supervised: uses weighted feature sum to generate soft labels,
    then calibrates.  Fast, low-memory, no MCMC required.
    Appropriate for exploratory analysis and large rasters.

  pymc
    Full hierarchical Bayesian logistic regression.
    Pixel-wise posterior with MCMC (NUTS sampler).
    Provides uncertainty quantification (HDI intervals).
    Computationally expensive; best used on region-of-interest subsets.

  numpyro
    JAX-accelerated equivalent of the PyMC model.
    ~10× faster than PyMC on GPU/TPU hardware.

All backends produce the same output: a posterior probability map plus
optional uncertainty bounds.

Design follows:
  Catling et al. (2018)    doi:10.1089/ast.2017.1737
  Affholder et al. (2021)  doi:10.1038/s41550-021-01372-6
  Affholder et al. (2025)  doi:10.3847/PSJ/addb09

References
----------
Catling et al. (2018)    "Exoplanet Biosignatures: A Framework for Assessment"
  Astrobiology 18, 709–738. doi:10.1089/ast.2017.1737
Affholder et al. (2021)  "Bayesian analysis of Enceladus's plume data"
  Nature Astronomy 5, 805–814. doi:10.1038/s41550-021-01372-6
Affholder et al. (2025)  "Viability of Glycine Fermentation in Titan's Ocean"
  Planet. Sci. J. 6, 86. doi:10.3847/PSJ/addb09
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from configs.pipeline_config import BayesianPriorConfig, PipelineConfig
from titan.features import FeatureStack, FEATURE_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class HabitabilityResult:
    """
    Output of the Bayesian inference stage.

    Attributes
    ----------
    posterior_mean:
        Per-pixel posterior mean of the habitability proxy, float32 [0,1].
        Shape (nrows, ncols).
    posterior_std:
        Per-pixel posterior standard deviation.  NaN if unavailable.
    hdi_low, hdi_high:
        94% highest-density interval bounds.  NaN if unavailable.
    feature_importances:
        Relative importance of each feature (sum=1.0).
    backend:
        Which inference backend was used.
    n_valid_pixels:
        Number of pixels with valid (non-NaN) posterior values.
    """
    posterior_mean:      np.ndarray
    posterior_std:       np.ndarray
    hdi_low:             np.ndarray
    hdi_high:            np.ndarray
    feature_importances: Dict[str, float]
    backend:             str
    n_valid_pixels:      int

    def save(self, out_dir: Path) -> None:
        """
        Save all result arrays to a directory as .npy files.

        Parameters
        ----------
        out_dir:
            Directory to save into (created if absent).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "posterior_mean.npy", self.posterior_mean)
        np.save(out_dir / "posterior_std.npy",  self.posterior_std)
        np.save(out_dir / "hdi_low.npy",        self.hdi_low)
        np.save(out_dir / "hdi_high.npy",       self.hdi_high)
        import json
        with open(out_dir / "feature_importances.json", "w") as fh:
            json.dump(self.feature_importances, fh, indent=2)
        logger.info("HabitabilityResult saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Backend: scikit-learn (default)
# ---------------------------------------------------------------------------

class SklearnHabitabilityModel:
    """
    Self-supervised Gaussian Naive Bayes habitability estimator.

    Algorithm
    ---------
    1. Compute a soft label for each pixel using the weighted feature sum
       (weights from BayesianPriorConfig).
    2. Threshold soft labels at `prior_config.positive_label_threshold`
       to create binary training labels.
    3. Fit a Gaussian Naive Bayes classifier on all valid pixels.
    4. Apply isotonic calibration to correct for GNB's probability
       overconfidence.
    5. Predict calibrated posterior probability for every pixel.

    This is an approximation to the full Bayesian update, but is
    appropriate for exploratory analysis given:
      - The high degree of missing data (not all datasets cover all pixels)
      - The computational cost of full MCMC on global rasters
      - The unknown prior P(life|Titan) which makes absolute posteriors
        uninterpretable without a reference model

    The sklearn output is best interpreted as a *relative* habitability
    score (which pixels are more vs less favourable) rather than an
    absolute probability.

    References
    ----------
    Catling et al. (2018): Bayesian framework for biosignature assessment.
    Affholder et al. (2021): Prior probability ranges for astrobiology.
    """

    def __init__(self, prior_config: BayesianPriorConfig) -> None:
        self.prior_config = prior_config
        self._model       = None
        self._calibrator  = None

    def fit_predict(self, features: FeatureStack) -> HabitabilityResult:
        """
        Fit the model and produce a habitability probability map.

        Parameters
        ----------
        features:
            Extracted feature stack (all 8 features).

        Returns
        -------
        HabitabilityResult
        """
        from sklearn.naive_bayes import GaussianNB
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import StandardScaler

        cfg     = self.prior_config
        weights = cfg.feature_weights()
        means   = cfg.prior_means()

        # ── 1. Stack features into (n_pixels, n_features) matrix ─────────────
        arr_3d = features.as_array()          # (8, nrows, ncols)
        nfeats, nrows, ncols = arr_3d.shape
        arr_2d = arr_3d.reshape(nfeats, -1).T  # (n_pixels, 8)

        # ── 2. Valid pixel mask (all 8 features finite) ───────────────────────
        valid_mask  = np.all(np.isfinite(arr_2d), axis=1)   # (n_pixels,)
        X_valid     = arr_2d[valid_mask]                     # (n_valid, 8)

        if X_valid.shape[0] < 100:
            logger.warning(
                "Only %d fully-valid pixels. Results will be unreliable. "
                "Consider reducing canonical_res_m or checking data coverage.",
                X_valid.shape[0],
            )

        # ── 3. Generate soft labels from weighted feature sum ─────────────────
        w_vec  = np.array([weights[n] for n in FEATURE_NAMES], dtype=np.float64)
        X_f64  = X_valid.astype(np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = X_f64 @ w_vec                             # (n_valid,)
        y_label = (scores > cfg.positive_label_threshold).astype(int)

        # Require at least some positives and negatives
        n_pos = int(y_label.sum())
        n_neg = int((~y_label.astype(bool)).sum())
        logger.info(
            "Training labels: %d positive (habitable), %d negative, "
            "%.1f%% positive",
            n_pos, n_neg, 100.0 * n_pos / len(y_label) if len(y_label) > 0 else 0,
        )
        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Degenerate labels: all pixels classified the same. "
                "Check feature coverage and threshold."
            )
            posterior = np.full(nrows * ncols, np.nan, dtype=np.float32)
            posterior[valid_mask] = float(n_pos > 0)
            return self._make_result(
                posterior.reshape(nrows, ncols),
                valid_mask, features, "sklearn",
            )

        # ── 4. Fit GNB with isotonic calibration ──────────────────────────────
        gnb = GaussianNB(var_smoothing=cfg.gnb_var_smoothing)
        # Use cross-validated calibration for better probability estimates
        cal_clf = CalibratedClassifierCV(gnb, cv=3, method="isotonic")
        cal_clf.fit(X_valid, y_label)
        self._model = cal_clf

        # ── 5. Apply prior correction ─────────────────────────────────────────
        # Adjust raw posterior by Bayesian prior means
        proba_raw   = cal_clf.predict_proba(X_valid)[:, 1]   # P(habitable)
        prior_score = np.array(
            [means[n] for n in FEATURE_NAMES], dtype=np.float32
        )
        # Weight-adjusted prior: dot(prior_means, weights)
        prior_mean_global = float(prior_score @ w_vec)
        # Bayesian update: posterior ∝ likelihood × prior
        # Simplified: blend calibrated posterior with prior
        alpha = 0.8   # weight on data-driven posterior vs prior
        proba_posterior = alpha * proba_raw + (1.0 - alpha) * prior_mean_global
        proba_posterior = np.clip(proba_posterior, 0.0, 1.0)

        # ── 6. Feature importances (permutation-based approximation) ──────────
        feature_importances = self._compute_importances(X_valid, y_label, w_vec)

        # ── 7. Assemble result ────────────────────────────────────────────────
        posterior = np.full(nrows * ncols, np.nan, dtype=np.float32)
        posterior[valid_mask] = proba_posterior
        return self._make_result(
            posterior.reshape(nrows, ncols),
            valid_mask, features, "sklearn",
            importances=feature_importances,
        )

    def _compute_importances(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> Dict[str, float]:
        """
        Estimate feature importances via mean-shift sensitivity.

        For each feature, measure how much the mean posterior changes
        when the feature is replaced by its prior mean.

        Parameters
        ----------
        X:
            Feature matrix (n_valid, 8).
        y:
            Binary labels.
        weights:
            Feature weight vector.

        Returns
        -------
        Dict[str, float]
            Normalised importances summing to 1.0.
        """
        model   = self._model
        base    = model.predict_proba(X)[:, 1].mean()
        deltas  = {}
        for i, name in enumerate(FEATURE_NAMES):
            X_perturbed        = X.copy()
            X_perturbed[:, i]  = X[:, i].mean()  # replace with feature mean
            perturbed_score    = model.predict_proba(X_perturbed)[:, 1].mean()
            deltas[name]       = abs(base - perturbed_score)

        total = sum(deltas.values())
        if total > 0:
            return {k: v / total for k, v in deltas.items()}
        # Fallback to weights
        return {n: float(w) for n, w in zip(FEATURE_NAMES, weights)}

    @staticmethod
    def _make_result(
        posterior: np.ndarray,
        valid_mask: np.ndarray,
        features: FeatureStack,
        backend: str,
        importances: Optional[Dict[str, float]] = None,
    ) -> HabitabilityResult:
        std_arr = np.full_like(posterior, np.nan)
        return HabitabilityResult(
            posterior_mean=posterior,
            posterior_std=std_arr,
            hdi_low=np.full_like(posterior, np.nan),
            hdi_high=np.full_like(posterior, np.nan),
            feature_importances=importances or {n: 1.0/8 for n in FEATURE_NAMES},
            backend=backend,
            n_valid_pixels=int(valid_mask.sum()),
        )


# ---------------------------------------------------------------------------
# Backend: PyMC (optional)
# ---------------------------------------------------------------------------

class PyMCHabitabilityModel:
    """
    Full hierarchical Bayesian logistic regression via PyMC.

    Model specification
    -------------------
    For pixel i with feature vector x_i:

        α       ~ Normal(0, 1)                        # intercept
        β_j     ~ Normal(μ_j, σ_j)  for j=1..8       # feature coefficients
                  where μ_j = logit(prior_mean_j)     # prior-informed mean
                  and   σ_j = 1.0                     # weakly informative

        logit(p_i) = α + Σ_j β_j · x_ij
        y_i     ~ Bernoulli(p_i)

    The prior means (μ_j) are set from BayesianPriorConfig and grounded
    in published measurements (McKay 2005, Iess 2012, etc.).

    NOTE: PyMC MCMC on full global rasters is computationally prohibitive.
    This backend is best used on small regions of interest (ROIs) or
    on a sparse stratified sample of pixels.

    References
    ----------
    Salvatier et al. (2016) "PyMC3: Probabilistic Programming in Python"
      PeerJ Computer Science 2, e55. doi:10.7717/peerj-cs.55
    Catling et al. (2018) doi:10.1089/ast.2017.1737
    """

    def __init__(
        self,
        prior_config: BayesianPriorConfig,
        draws:  int = 2000,
        tune:   int = 1000,
        chains: int = 4,
        seed:   int = 42,
    ) -> None:
        self.prior_config = prior_config
        self.draws  = draws
        self.tune   = tune
        self.chains = chains
        self.seed   = seed

    def fit_predict(
        self,
        features: FeatureStack,
        max_pixels: int = 50_000,
    ) -> HabitabilityResult:
        """
        Run MCMC and return posterior probability map.

        Parameters
        ----------
        features:
            Extracted feature stack.
        max_pixels:
            Maximum number of pixels to use for MCMC.  Random stratified
            sample if the valid set is larger.

        Returns
        -------
        HabitabilityResult
        """
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            raise ImportError(
                "PyMC backend requires: pip install pymc pytensor arviz"
            )

        cfg     = self.prior_config
        weights = cfg.feature_weights()
        means   = cfg.prior_means()

        arr_3d = features.as_array()
        nfeats, nrows, ncols = arr_3d.shape
        arr_2d  = arr_3d.reshape(nfeats, -1).T
        valid   = np.all(np.isfinite(arr_2d), axis=1)
        X_valid = arr_2d[valid]

        # Generate soft labels (same as sklearn)
        w_vec  = np.array([weights[n] for n in FEATURE_NAMES])
        scores = X_valid @ w_vec
        y_soft = (scores > cfg.positive_label_threshold).astype(int)

        # Subsample for MCMC tractability
        rng = np.random.default_rng(self.seed)
        if X_valid.shape[0] > max_pixels:
            idx = rng.choice(X_valid.shape[0], max_pixels, replace=False)
            X_fit, y_fit = X_valid[idx], y_soft[idx]
            logger.info(
                "PyMC: subsampled %d / %d valid pixels for MCMC.",
                max_pixels, X_valid.shape[0],
            )
        else:
            X_fit, y_fit = X_valid, y_soft

        # Prior means in logit space
        def safe_logit(p: float) -> float:
            p = np.clip(p, 1e-4, 1 - 1e-4)
            return float(np.log(p / (1 - p)))

        mu_beta = np.array([safe_logit(means[n]) for n in FEATURE_NAMES])

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
            beta  = pm.Normal(
                "beta",
                mu=mu_beta,
                sigma=1.0,
                shape=nfeats,
            )
            logit_p = alpha + pm.math.dot(X_fit, beta)
            p       = pm.Deterministic("p", pm.math.sigmoid(logit_p))
            pm.Bernoulli("obs", p=p, observed=y_fit)

            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.seed,
                progressbar=True,
                return_inferencedata=True,
            )

        # Extract posterior predictive for ALL valid pixels
        beta_samples  = trace.posterior["beta"].values       # (chains, draws, 8)
        alpha_samples = trace.posterior["alpha"].values      # (chains, draws)

        beta_flat  = beta_samples.reshape(-1, nfeats)         # (S, 8)
        alpha_flat = alpha_samples.reshape(-1)                # (S,)

        logit_all = alpha_flat[:, np.newaxis] + X_valid @ beta_flat.T  # (n_valid, S)
        p_all     = 1.0 / (1.0 + np.exp(-logit_all))                   # sigmoid

        p_mean = p_all.mean(axis=1).astype(np.float32)
        p_std  = p_all.std(axis=1).astype(np.float32)
        hdi    = az.hdi(p_all.T[np.newaxis], hdi_prob=0.94)  # shape workaround
        hdi_lo = hdi[..., 0].flatten()[:len(p_mean)].astype(np.float32)
        hdi_hi = hdi[..., 1].flatten()[:len(p_mean)].astype(np.float32)

        # Reconstruct full grid
        def _fill(vals: np.ndarray) -> np.ndarray:
            out = np.full(nrows * ncols, np.nan, dtype=np.float32)
            out[valid] = vals
            return out.reshape(nrows, ncols)

        importances = {
            n: float(abs(beta_flat[:, i].mean()))
            for i, n in enumerate(FEATURE_NAMES)
        }
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return HabitabilityResult(
            posterior_mean=_fill(p_mean),
            posterior_std=_fill(p_std),
            hdi_low=_fill(hdi_lo),
            hdi_high=_fill(hdi_hi),
            feature_importances=importances,
            backend="pymc",
            n_valid_pixels=int(valid.sum()),
        )


# ---------------------------------------------------------------------------
# Backend: NumPyro (optional)
# ---------------------------------------------------------------------------

class NumPyroHabitabilityModel:
    """
    JAX-accelerated Bayesian logistic regression via NumPyro.

    Identical model specification to PyMCHabitabilityModel but uses
    NumPyro's NUTS sampler, which is ~5–10× faster on modern hardware
    and supports GPU/TPU acceleration via JAX.

    References
    ----------
    Phan et al. (2019) "Composable Effects for Flexible and Accelerated
      Probabilistic Programming in NumPyro." arXiv:1912.11554
    """

    def __init__(
        self,
        prior_config: BayesianPriorConfig,
        draws:  int = 2000,
        warmup: int = 1000,
        chains: int = 4,
        seed:   int = 42,
    ) -> None:
        self.prior_config = prior_config
        self.draws  = draws
        self.warmup = warmup
        self.chains = chains
        self.seed   = seed

    def fit_predict(
        self,
        features: FeatureStack,
        max_pixels: int = 100_000,
    ) -> HabitabilityResult:
        """
        Run NumPyro NUTS and return posterior probability map.

        Parameters
        ----------
        features:
            Extracted feature stack.
        max_pixels:
            Maximum pixels for MCMC fitting.

        Returns
        -------
        HabitabilityResult
        """
        try:
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
            import jax
            import jax.numpy as jnp
        except ImportError:
            raise ImportError(
                "NumPyro backend requires: pip install numpyro jax jaxlib"
            )

        cfg     = self.prior_config
        weights = cfg.feature_weights()
        means   = cfg.prior_means()

        arr_3d = features.as_array()
        nfeats, nrows, ncols = arr_3d.shape
        arr_2d  = arr_3d.reshape(nfeats, -1).T
        valid   = np.all(np.isfinite(arr_2d), axis=1)
        X_valid = arr_2d[valid]

        w_vec  = np.array([weights[n] for n in FEATURE_NAMES])
        scores = X_valid @ w_vec
        y_soft = (scores > cfg.positive_label_threshold).astype(int)

        rng_np = np.random.default_rng(self.seed)
        if X_valid.shape[0] > max_pixels:
            idx    = rng_np.choice(X_valid.shape[0], max_pixels, replace=False)
            X_fit  = X_valid[idx]
            y_fit  = y_soft[idx]
        else:
            X_fit, y_fit = X_valid, y_soft

        def safe_logit(p: float) -> float:
            p = np.clip(p, 1e-4, 1 - 1e-4)
            return float(np.log(p / (1 - p)))

        mu_beta = np.array([safe_logit(means[n]) for n in FEATURE_NAMES])

        X_jax = jnp.array(X_fit, dtype=jnp.float32)
        y_jax = jnp.array(y_fit, dtype=jnp.int32)
        mu_jax = jnp.array(mu_beta, dtype=jnp.float32)

        def numpyro_model(X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> None:
            alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
            beta  = numpyro.sample(
                "beta", dist.Normal(mu_jax, jnp.ones(nfeats))
            )
            logit_p = alpha + jnp.dot(X, beta)
            numpyro.sample("obs", dist.Bernoulli(logits=logit_p), obs=y)

        kernel = NUTS(numpyro_model)
        mcmc   = MCMC(
            kernel,
            num_warmup=self.warmup,
            num_samples=self.draws,
            num_chains=self.chains,
        )
        mcmc.run(jax.random.PRNGKey(self.seed), X_jax, y_jax)
        samples = mcmc.get_samples()

        beta_np  = np.array(samples["beta"])   # (S, 8)
        alpha_np = np.array(samples["alpha"])  # (S,)

        X_full   = jnp.array(X_valid, dtype=jnp.float32)
        logit_all = alpha_np[:, np.newaxis] + X_valid @ beta_np.T
        p_all     = 1.0 / (1.0 + np.exp(-logit_all))

        p_mean = p_all.mean(axis=1).astype(np.float32)
        p_std  = p_all.std(axis=1).astype(np.float32)
        hdi_lo = np.percentile(p_all, 3.0, axis=1).astype(np.float32)
        hdi_hi = np.percentile(p_all, 97.0, axis=1).astype(np.float32)

        def _fill(vals: np.ndarray) -> np.ndarray:
            out = np.full(nrows * ncols, np.nan, dtype=np.float32)
            out[valid] = vals
            return out.reshape(nrows, ncols)

        importances = {n: float(abs(beta_np[:, i].mean()))
                       for i, n in enumerate(FEATURE_NAMES)}
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return HabitabilityResult(
            posterior_mean=_fill(p_mean),
            posterior_std=_fill(p_std),
            hdi_low=_fill(hdi_lo),
            hdi_high=_fill(hdi_hi),
            feature_importances=importances,
            backend="numpyro",
            n_valid_pixels=int(valid.sum()),
        )


# ---------------------------------------------------------------------------
# Factory: select backend from config
# ---------------------------------------------------------------------------

def build_model(
    config: PipelineConfig,
) -> "SklearnHabitabilityModel | PyMCHabitabilityModel | NumPyroHabitabilityModel":
    """
    Instantiate the appropriate Bayesian backend from pipeline config.

    Parameters
    ----------
    config:
        Pipeline configuration (uses ``config.bayesian_backend``).

    Returns
    -------
    Model instance (one of the three backend classes).

    Raises
    ------
    ValueError
        If ``config.bayesian_backend`` is not a recognised option.
    """
    backend = config.bayesian_backend.lower()
    if backend == "sklearn":
        return SklearnHabitabilityModel(config.priors)
    elif backend == "pymc":
        return PyMCHabitabilityModel(
            config.priors,
            draws=config.mcmc_draws,
            tune=config.mcmc_tune,
            chains=config.mcmc_chains,
            seed=config.random_seed,
        )
    elif backend == "numpyro":
        return NumPyroHabitabilityModel(
            config.priors,
            draws=config.mcmc_draws,
            warmup=config.mcmc_tune,
            chains=config.mcmc_chains,
            seed=config.random_seed,
        )
    else:
        raise ValueError(
            f"Unknown Bayesian backend: '{config.bayesian_backend}'. "
            "Choose from: 'sklearn', 'pymc', 'numpyro'."
        )


def run_inference(
    features: FeatureStack,
    config:   PipelineConfig,
) -> HabitabilityResult:
    """
    Run the full Bayesian inference pipeline.

    Convenience wrapper that builds the model and calls fit_predict().

    Parameters
    ----------
    features:
        Extracted feature stack from Stage 3.
    config:
        Pipeline configuration.

    Returns
    -------
    HabitabilityResult
    """
    model = build_model(config)
    logger.info("Running Bayesian inference with backend: %s", config.bayesian_backend)
    return model.fit_predict(features)
