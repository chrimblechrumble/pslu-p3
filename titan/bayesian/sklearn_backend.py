"""
titan/bayesian/sklearn_backend.py
===================================
Default Bayesian backend using scikit-learn.

This backend implements a conjugate Beta–Bernoulli update that runs in
seconds on the full global grid without MCMC.  It is the default backend
and is suitable for exploratory analysis and publication-quality maps.

Mathematical model
------------------
For each pixel i with feature vector D[i] ∈ [0,1]^F:

    Prior:   H ~ Beta(alpha_0, beta_0)
             where alpha_0 = mu_0 * kappa,  beta_0 = (1-mu_0) * kappa,
             and mu_0 = weighted mean of prior_mean_* values.

    Likelihood update:
    Each feature f is treated as a noisy observation of H:
        alpha_i += w_f * D[i,f] * sharpness
        beta_i  += w_f * (1 - D[i,f]) * sharpness

    Posterior:  H | D[i] ~ Beta(alpha_i, beta_i)

    Posterior mean:   E[H|D] = alpha_i / (alpha_i + beta_i)
    Posterior std:    Std[H|D] = sqrt(alpha_i*beta_i / ((alpha_i+beta_i)^2
                                      * (alpha_i+beta_i+1)))
    95% CI:           Beta(alpha_i, beta_i).ppf([0.025, 0.975])

This is the analytical conjugate update, not MCMC.  It is exact under
the Beta–Bernoulli model and scales linearly with the number of pixels.

References
----------
Affholder et al. (2021)  DOI:10.1038/s41550-021-01372-6
Gelman et al. (2013) Bayesian Data Analysis, 3rd ed. CRC Press.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import beta as beta_dist

from configs.pipeline_config import BayesianPriorConfig
from titan.bayesian.base import BayesianBackend, BayesianResult
from titan.features import FeatureStack
from titan.preprocessing import CanonicalGrid

logger = logging.getLogger(__name__)


class SklearnBayesianBackend(BayesianBackend):
    """
    Conjugate Beta–Bernoulli Bayesian update (analytic, no MCMC).

    This is the default backend.  It produces the exact posterior under
    the Beta–Bernoulli likelihood model and runs in < 10 s on the full
    global grid.

    Parameters
    ----------
    priors:
        Prior configuration.
    grid:
        Canonical spatial grid.
    random_seed:
        Random seed (used for any stochastic tie-breaking; not MCMC).
    """

    def __init__(
        self,
        priors: BayesianPriorConfig,
        grid: CanonicalGrid,
        random_seed: int = 42,
    ) -> None:
        super().__init__(priors, grid, random_seed)

    @property
    def name(self) -> str:
        return "sklearn_conjugate_beta"

    def infer(self, features: FeatureStack) -> BayesianResult:
        """
        Run the analytical conjugate Beta update on all valid pixels.

        Parameters
        ----------
        features:
            Extracted feature stack.

        Returns
        -------
        BayesianResult
            Posterior mean, std, and 95% credible interval maps.
        """
        logger.info("Running conjugate Beta–Bernoulli update (sklearn backend)…")

        # ── Flatten features ──────────────────────────────────────────────
        X, valid_idx = self._feature_matrix(features)
        # X shape: (N_valid, n_features), values in [0, 1]
        N, F = X.shape
        logger.info("  Valid pixels: %d / %d", N,
                    self.grid.nrows * self.grid.ncols)

        # ── Prior parameters ──────────────────────────────────────────────
        # Global prior: single Beta(alpha_0, beta_0) for H per pixel.
        # alpha_0, beta_0 are scalar (same prior for all pixels).
        mu_global = np.dot(
            np.array(self.priors.weight_vector()),
            np.array(self.priors.prior_mean_vector()),
        )
        kappa = self.priors.beta_concentration
        alpha_0 = float(mu_global * kappa)
        beta_0  = float((1.0 - mu_global) * kappa)

        logger.info(
            "  Prior: Beta(alpha=%.2f, beta=%.2f)  →  prior mean = %.3f",
            alpha_0, beta_0, alpha_0 / (alpha_0 + beta_0),
        )

        # ── Bayesian update ───────────────────────────────────────────────
        # For each feature f with weight w_f and observation D[i,f]:
        #   alpha[i] += w_f * D[i,f] * sharpness
        #   beta[i]  += w_f * (1-D[i,f]) * sharpness
        weights    = np.array(self.priors.weight_vector(), dtype=np.float64)
        sharpness  = self.priors.likelihood_sharpness

        # Initialise posterior params at prior
        alpha_post = np.full(N, alpha_0, dtype=np.float64)
        beta_post  = np.full(N, beta_0,  dtype=np.float64)

        for f in range(F):
            wf = weights[f]
            d  = X[:, f]
            alpha_post += wf * d       * sharpness
            beta_post  += wf * (1 - d) * sharpness

        # ── Posterior statistics ──────────────────────────────────────────
        ab_sum = alpha_post + beta_post

        post_mean = alpha_post / ab_sum

        post_var  = (alpha_post * beta_post) / (ab_sum ** 2 * (ab_sum + 1))
        post_std  = np.sqrt(np.maximum(post_var, 0.0))

        # 95% credible interval via Beta PPF
        # Vectorised Beta CDF inversion
        post_lower = beta_dist.ppf(0.025, alpha_post, beta_post)
        post_upper = beta_dist.ppf(0.975, alpha_post, beta_post)

        logger.info(
            "  Posterior mean range: [%.3f, %.3f]  (global median %.3f)",
            post_mean.min(), post_mean.max(), float(np.median(post_mean)),
        )

        # ── Reconstruct full spatial maps ─────────────────────────────────
        fill = float(alpha_0 / (alpha_0 + beta_0))  # prior mean for invalid px

        result = BayesianResult(
            posterior_mean  = self._reconstruct_map(post_mean,  valid_idx, fill),
            posterior_std   = self._reconstruct_map(post_std,   valid_idx, 0.0),
            posterior_lower = self._reconstruct_map(post_lower, valid_idx, fill),
            posterior_upper = self._reconstruct_map(post_upper, valid_idx, fill),
            prior_mean      = self._prior_mean_map(),
            n_pixels_mcmc   = 0,
            backend         = self.name,
        )

        logger.info("sklearn backend inference complete.")
        return result
