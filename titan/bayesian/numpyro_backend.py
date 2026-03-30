"""
titan/bayesian/numpyro_backend.py
===================================
Optional NumPyro (JAX-based) Bayesian backend.

NumPyro uses JAX for automatic differentiation and JIT compilation,
making it significantly faster than PyMC for large datasets while
supporting the same NUTS sampler.

Like the PyMC backend, this runs MCMC on a pixel subsample and
interpolates back to the full grid.

Installation
------------
    pip install numpyro>=0.13 jax>=0.4.25 jaxlib>=0.4.25

References
----------
Phan et al. (2019) arXiv:1912.11554  (NumPyro paper)
Affholder et al. (2021)  DOI:10.1038/s41550-021-01372-6
"""

from __future__ import annotations

import logging

import numpy as np

from configs.pipeline_config import BayesianPriorConfig
from titan.bayesian.base import BayesianBackend, BayesianResult
from titan.features import FeatureStack
from titan.preprocessing import CanonicalGrid

logger = logging.getLogger(__name__)


class NumPyroBayesianBackend(BayesianBackend):
    """
    NUTS posterior via NumPyro/JAX on a representative pixel sample.

    Parameters
    ----------
    priors:
        Prior configuration.
    grid:
        Canonical spatial grid.
    random_seed:
        JAX PRNG key seed.
    n_mcmc_pixels:
        Number of pixels to sample for MCMC.
    num_warmup:
        NUTS warmup steps.
    num_samples:
        NUTS posterior samples per chain.
    num_chains:
        Number of chains (recommend 1 with JAX unless using pmap).
    """

    def __init__(
        self,
        priors: BayesianPriorConfig,
        grid: CanonicalGrid,
        random_seed: int = 42,
        n_mcmc_pixels: int = 5_000,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 1,
    ) -> None:
        super().__init__(priors, grid, random_seed)
        self.n_mcmc_pixels = n_mcmc_pixels
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains

    @property
    def name(self) -> str:
        return "numpyro_nuts"

    def infer(self, features: FeatureStack) -> BayesianResult:
        """
        Run NumPyro NUTS on a pixel sample and interpolate to full grid.

        Parameters
        ----------
        features:
            Extracted feature stack.

        Returns
        -------
        BayesianResult
        """
        try:
            import jax
            import jax.numpy as jnp
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
        except ImportError as exc:
            raise ImportError(
                "NumPyro backend requires: "
                "pip install numpyro>=0.13 jax>=0.4.25 jaxlib>=0.4.25"
            ) from exc

        logger.info("Running NumPyro NUTS backend (n_mcmc_pixels=%d)…",
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
        X_sample = X_all[sample_idx].astype(np.float32)
        M, F = X_sample.shape

        # ── Priors ───────────────────────────────────────────────────────
        weights   = np.array(self.priors.weight_vector(), dtype=np.float32)
        mu_global = float(np.dot(weights, self.priors.prior_mean_vector()))
        kappa     = float(self.priors.beta_concentration)
        sharpness = float(self.priors.likelihood_sharpness)
        alpha_0   = float(mu_global * kappa)
        beta_0    = float((1.0 - mu_global) * kappa)

        # Convert to JAX arrays
        jax_X    = jnp.array(X_sample)          # (M, F)
        jax_w    = jnp.array(weights)           # (F,)

        # ── NumPyro model ─────────────────────────────────────────────────
        def model(obs: jnp.ndarray) -> None:
            """
            H[m] ~ Beta(alpha_0, beta_0)
            obs[m,f] ~ Beta(H[m]*w[f]*sharp + 0.5, (1-H[m])*w[f]*sharp + 0.5)
            """
            H = numpyro.sample(
                "H",
                dist.Beta(alpha_0, beta_0).expand([M]),
            )
            for f in range(F):
                wf = jax_w[f]
                alpha_lk = H * wf * sharpness + 0.5
                beta_lk  = (1.0 - H) * wf * sharpness + 0.5
                numpyro.sample(
                    f"D_{f}",
                    dist.Beta(alpha_lk, beta_lk),
                    obs=jnp.clip(obs[:, f], 1e-6, 1.0 - 1e-6),
                )

        # ── Run NUTS ─────────────────────────────────────────────────────
        nuts_kernel = NUTS(model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=True,
        )
        jax_key = jax.random.PRNGKey(self.random_seed)
        mcmc.run(jax_key, jax_X)
        mcmc.print_summary()

        H_samples = np.array(mcmc.get_samples()["H"])  # (draws, M)

        post_mean_sample  = H_samples.mean(axis=0)
        post_std_sample   = H_samples.std(axis=0)
        post_lower_sample = np.percentile(H_samples, 2.5,  axis=0)
        post_upper_sample = np.percentile(H_samples, 97.5, axis=0)

        # ── Interpolate to full valid set ─────────────────────────────────
        from sklearn.neighbors import KNeighborsRegressor

        def _knn(y: np.ndarray) -> np.ndarray:
            m = KNeighborsRegressor(n_neighbors=5)
            m.fit(X_sample, y)
            return m.predict(X_all).astype(np.float32)

        fill = float(mu_global)
        return BayesianResult(
            posterior_mean  = self._reconstruct_map(_knn(post_mean_sample),  valid_idx, fill),
            posterior_std   = self._reconstruct_map(_knn(post_std_sample),   valid_idx, 0.0),
            posterior_lower = self._reconstruct_map(_knn(post_lower_sample), valid_idx, fill),
            posterior_upper = self._reconstruct_map(_knn(post_upper_sample), valid_idx, fill),
            prior_mean      = self._prior_mean_map(),
            n_pixels_mcmc   = M,
            backend         = self.name,
        )
