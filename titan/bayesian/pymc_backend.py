"""
titan/bayesian/pymc_backend.py
================================
Optional PyMC MCMC Bayesian backend.

Uses NUTS (No-U-Turn Sampler) for full posterior sampling.  Because
running MCMC on millions of pixels independently is intractable, this
backend:

  1. Spatially subsamples ``n_mcmc_pixels`` representative pixels
  2. Runs NUTS on the subsampled pixel set
  3. Interpolates posterior statistics back to the full grid using
     nearest-neighbour assignment

This approach is justified because the likelihood model is separable
across pixels (each pixel's posterior depends only on its own feature
vector and the global hyperpriors).

For the full global grid at publication quality, use the sklearn backend
(conjugate update) and reserve PyMC for validation, sensitivity analysis,
and hierarchical extensions.

Installation
------------
    pip install pymc>=5.10 pytensor>=2.18

References
----------
Affholder et al. (2021)  DOI:10.1038/s41550-021-01372-6
Salvatier et al. (2016) PeerJ Comp. Sci. 2:e55  (PyMC3 paper)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from configs.pipeline_config import BayesianPriorConfig
from titan.bayesian.base import BayesianBackend, BayesianResult
from titan.features import FeatureStack
from titan.preprocessing import CanonicalGrid

logger = logging.getLogger(__name__)


class PyMCBayesianBackend(BayesianBackend):
    """
    Full MCMC posterior via PyMC NUTS on a representative pixel sample.

    Parameters
    ----------
    priors:
        Prior configuration.
    grid:
        Canonical spatial grid.
    random_seed:
        MCMC random seed.
    n_mcmc_pixels:
        Number of pixels to sample for MCMC (default 5000).
        Higher = more accurate interpolation but slower.
    draws:
        MCMC posterior draws per chain.
    tune:
        MCMC tuning steps.
    chains:
        Number of parallel chains.
    target_accept:
        NUTS target acceptance rate (0.8–0.95 recommended).
    """

    def __init__(
        self,
        priors: BayesianPriorConfig,
        grid: CanonicalGrid,
        random_seed: int = 42,
        n_mcmc_pixels: int = 5_000,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
    ) -> None:
        super().__init__(priors, grid, random_seed)
        self.n_mcmc_pixels = n_mcmc_pixels
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept

    @property
    def name(self) -> str:
        return "pymc_nuts"

    def infer(self, features: FeatureStack) -> BayesianResult:
        """
        Run PyMC NUTS on a pixel sample and interpolate to full grid.

        Parameters
        ----------
        features:
            Extracted feature stack.

        Returns
        -------
        BayesianResult
        """
        try:
            import pymc as pm
            import arviz as az
        except ImportError as exc:
            raise ImportError(
                "PyMC backend requires: pip install pymc>=5.10 arviz>=0.18"
            ) from exc

        logger.info("Running PyMC NUTS backend (n_mcmc_pixels=%d)…",
                    self.n_mcmc_pixels)

        # ── Sample valid pixels ───────────────────────────────────────────
        X_all, valid_idx = self._feature_matrix(features)
        N_valid = len(X_all)
        rng = np.random.default_rng(self.random_seed)

        sample_idx = rng.choice(
            N_valid,
            size=min(self.n_mcmc_pixels, N_valid),
            replace=False,
        )
        X_sample = X_all[sample_idx]       # (M, F)
        M, F = X_sample.shape
        logger.info("  Sampling %d / %d valid pixels for MCMC", M, N_valid)

        # ── Prior parameters ──────────────────────────────────────────────
        mu_vec    = np.array(self.priors.prior_mean_vector(), dtype=np.float64)
        kappa     = float(self.priors.beta_concentration)
        weights   = np.array(self.priors.weight_vector(), dtype=np.float64)
        sharpness = self.priors.likelihood_sharpness

        # Global prior for H
        mu_global = float(np.dot(weights, mu_vec))
        alpha_0   = mu_global * kappa
        beta_0    = (1.0 - mu_global) * kappa

        # ── PyMC model ────────────────────────────────────────────────────
        with pm.Model() as model:
            # Latent habitability per sampled pixel: H ~ Beta(alpha_0, beta_0)
            H = pm.Beta(
                "H",
                alpha=alpha_0,
                beta=beta_0,
                shape=M,
            )

            # Likelihood: each feature is a Beta observation of H,
            # weighted by w_f and sharpened by likelihood_sharpness.
            for f in range(F):
                wf = weights[f]
                obs_f = X_sample[:, f].astype(np.float64)
                # Beta likelihood parameters
                alpha_lk = H * wf * sharpness + 0.5
                beta_lk  = (1.0 - H) * wf * sharpness + 0.5
                _ = pm.Beta(
                    f"D_{f}",
                    alpha=alpha_lk,
                    beta=beta_lk,
                    observed=np.clip(obs_f, 1e-6, 1.0 - 1e-6),
                )

            # ── Sample ───────────────────────────────────────────────────
            logger.info("  Starting NUTS sampling (draws=%d, tune=%d, chains=%d)…",
                        self.draws, self.tune, self.chains)
            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=True,
                return_inferencedata=True,
            )

        # ── Posterior statistics for sampled pixels ───────────────────────
        H_samples = trace.posterior["H"].values  # (chains, draws, M)
        H_flat    = H_samples.reshape(-1, M)     # (chains*draws, M)

        post_mean_sample  = H_flat.mean(axis=0)
        post_std_sample   = H_flat.std(axis=0)
        post_lower_sample = np.percentile(H_flat, 2.5,  axis=0)
        post_upper_sample = np.percentile(H_flat, 97.5, axis=0)

        # ── Interpolate back to full valid-pixel set ──────────────────────
        # Use nearest-neighbour in feature space (sklearn KNN)
        from sklearn.neighbors import KNeighborsRegressor

        knn_mean  = KNeighborsRegressor(n_neighbors=5, algorithm="ball_tree")
        knn_std   = KNeighborsRegressor(n_neighbors=5, algorithm="ball_tree")
        knn_lower = KNeighborsRegressor(n_neighbors=5, algorithm="ball_tree")
        knn_upper = KNeighborsRegressor(n_neighbors=5, algorithm="ball_tree")

        knn_mean .fit(X_sample, post_mean_sample)
        knn_std  .fit(X_sample, post_std_sample)
        knn_lower.fit(X_sample, post_lower_sample)
        knn_upper.fit(X_sample, post_upper_sample)

        post_mean_all  = knn_mean .predict(X_all).astype(np.float32)
        post_std_all   = knn_std  .predict(X_all).astype(np.float32)
        post_lower_all = knn_lower.predict(X_all).astype(np.float32)
        post_upper_all = knn_upper.predict(X_all).astype(np.float32)

        fill = float(mu_global)

        result = BayesianResult(
            posterior_mean  = self._reconstruct_map(post_mean_all,  valid_idx, fill),
            posterior_std   = self._reconstruct_map(post_std_all,   valid_idx, 0.0),
            posterior_lower = self._reconstruct_map(post_lower_all, valid_idx, fill),
            posterior_upper = self._reconstruct_map(post_upper_all, valid_idx, fill),
            prior_mean      = self._prior_mean_map(),
            n_pixels_mcmc   = M,
            backend         = self.name,
        )

        # Log ArviZ diagnostics summary
        logger.info("ArviZ diagnostics:")
        logger.info(az.summary(trace, var_names=["H"]).to_string())

        return result
