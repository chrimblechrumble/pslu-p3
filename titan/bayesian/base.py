"""
titan/bayesian/base.py
======================
Abstract base class for all Bayesian inference backends.

All three backends (sklearn, PyMC, NumPyro) implement this interface,
so the rest of the pipeline is backend-agnostic.

The inference model
-------------------
We follow the framework of Affholder et al. (2021) adapted for Titan's
surface habitability proxy rather than Enceladus subsurface methanogenesis.

The model is:

    H[i] ~ Beta(alpha, beta)          # latent habitability at pixel i
    alpha = mu_0 * kappa              # from BayesianPriorConfig
    beta  = (1 - mu_0) * kappa

    D[i,f] | H[i] ~ Beta(            # observation of feature f at pixel i
        H[i] * sharpness + 0.5,
        (1 - H[i]) * sharpness + 0.5
    )

    Posterior: P(H[i] | D[i,:]) ∝ P(D[i,:] | H[i]) * P(H[i])

Because the full MCMC is prohibitively expensive at 46080×23040 global
pixels, all backends operate on a **vectorised approximate model** where:

  1. Features are pre-extracted to an (N, F) matrix (N pixels, F=8 features)
  2. The latent habitability H is inferred per pixel independently
  3. The posterior mean E[H|D] is stored as the output habitability map

For the sklearn backend, this reduces to a weighted combination of the
features against the prior, efficiently computed as a matrix operation.

For the PyMC / NumPyro backends, MCMC is run on a spatially subsampled
set of representative pixels (configurable via ``n_mcmc_pixels``), with
the posterior then interpolated back to the full grid.

References
----------
Affholder et al. (2021)   DOI:10.1038/s41550-021-01372-6
Catling et al. (2018)     DOI:10.1089/ast.2017.1737
Bayes (1763)              Phil. Trans. Royal Soc. 53:370-418
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from configs.pipeline_config import BayesianPriorConfig
from titan.features import FeatureStack
from titan.preprocessing import CanonicalGrid


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BayesianResult:
    """
    Posterior habitability probability maps.

    All arrays are float32 of shape (nrows, ncols).

    Attributes
    ----------
    posterior_mean:
        E[H | D] — pixel-wise posterior mean habitability probability.
        This is the primary output for visualisation and publication.
    posterior_std:
        Std[H | D] — posterior uncertainty (standard deviation).
    posterior_lower:
        2.5th percentile of posterior (lower 95% credible interval bound).
    posterior_upper:
        97.5th percentile of posterior (upper 95% credible interval bound).
    prior_mean:
        E[H] — prior mean map for comparison with posterior.
    n_pixels_mcmc:
        Number of pixels used for MCMC inference (backends that subsample).
    backend:
        Name of the backend that produced this result.
    """
    posterior_mean:  np.ndarray
    posterior_std:   np.ndarray
    posterior_lower: np.ndarray
    posterior_upper: np.ndarray
    prior_mean:      np.ndarray
    n_pixels_mcmc:   int = 0
    backend:         str = "unknown"

    def to_xarray(
        self,
        grid: CanonicalGrid,
    ) -> "xr.Dataset":  # type: ignore[name-defined]
        """Convert to an xarray Dataset with lat/lon coordinates."""
        import xarray as xr
        lats = grid.lat_centres_deg()
        lons = grid.lon_centres_deg()
        coords = {"lat": lats, "lon": lons}

        return xr.Dataset(
            {
                "posterior_mean": xr.DataArray(
                    self.posterior_mean, dims=["lat", "lon"], coords=coords,
                    attrs={"long_name": "Posterior mean habitability P(H|D)",
                           "units": "probability [0,1]"},
                ),
                "posterior_std": xr.DataArray(
                    self.posterior_std, dims=["lat", "lon"], coords=coords,
                    attrs={"long_name": "Posterior std dev",
                           "units": "probability"},
                ),
                "posterior_lower": xr.DataArray(
                    self.posterior_lower, dims=["lat", "lon"], coords=coords,
                    attrs={"long_name": "2.5th percentile (lower CI)",
                           "units": "probability [0,1]"},
                ),
                "posterior_upper": xr.DataArray(
                    self.posterior_upper, dims=["lat", "lon"], coords=coords,
                    attrs={"long_name": "97.5th percentile (upper CI)",
                           "units": "probability [0,1]"},
                ),
                "prior_mean": xr.DataArray(
                    self.prior_mean, dims=["lat", "lon"], coords=coords,
                    attrs={"long_name": "Prior mean habitability E[H]",
                           "units": "probability [0,1]"},
                ),
            },
            attrs={
                "title": "Titan Pixel-wise Habitability Posterior",
                "backend": self.backend,
                "n_pixels_mcmc": self.n_pixels_mcmc,
                "description": (
                    "Posterior probability of habitability proxy P(H|D) "
                    "per canonical pixel, inferred via Bayesian update of "
                    "prior using eight geophysical/geochemical features."
                ),
            },
        )


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class BayesianBackend(ABC):
    """
    Abstract base class for Bayesian habitability inference backends.

    All backends accept a FeatureStack and return a BayesianResult.

    Parameters
    ----------
    priors:
        Prior configuration from BayesianPriorConfig.
    grid:
        Canonical spatial grid.
    random_seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        priors: BayesianPriorConfig,
        grid: CanonicalGrid,
        random_seed: int = 42,
    ) -> None:
        self.priors = priors
        self.grid = grid
        self.random_seed = random_seed

    @abstractmethod
    def infer(self, features: FeatureStack) -> BayesianResult:
        """
        Run the Bayesian inference and return posterior maps.

        Parameters
        ----------
        features:
            Extracted feature stack (all values in [0, 1]).

        Returns
        -------
        BayesianResult
            Posterior mean, std, and credible interval maps.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (e.g. 'sklearn')."""
        ...

    # ------------------------------------------------------------------
    # Shared utilities available to all backends
    # ------------------------------------------------------------------

    def _prior_alpha_beta(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return Beta distribution alpha and beta vectors from prior config.

        Uses: alpha = mu * kappa,  beta = (1-mu) * kappa

        Returns
        -------
        Tuple of (alpha_vec, beta_vec), each shape (n_features,).
        """
        mu = np.array(self.priors.prior_mean_vector(), dtype=np.float64)
        kappa = self.priors.beta_concentration
        alpha = mu * kappa
        beta_  = (1.0 - mu) * kappa
        return alpha, beta_

    def _feature_matrix(
        self,
        features: "FeatureStack",
    ) -> "Tuple[np.ndarray, np.ndarray]":
        """
        Flatten the feature stack to an (N_valid_pixels, n_features) matrix,
        imputing NaN feature values with their per-feature prior mean.

        **Why impute rather than exclude?**

        The original approach (``valid = np.all(np.isfinite, axis=1)``) treats
        a pixel as invalid if *any* feature is NaN.  This caused the 180–360°W
        hemisphere — where the VIMS mosaic has no coverage — to be filled with
        the flat global prior mean rather than a posterior informed by the 7
        other features that *are* available there (geomorphology, topography,
        SAR, methane cycle, etc.).

        The correct Bayesian approach for partial feature coverage is to
        impute missing feature values with their prior mean.  This makes the
        missing feature contribute *zero net evidence* to the update (the
        prior mean observation neither confirms nor denies habitability), while
        all non-NaN features still update the posterior normally.

        A pixel is only excluded entirely (treated as truly "no data") if
        *every* feature is NaN, which in practice happens only outside Titan's
        disc.

        Parameters
        ----------
        features:
            FeatureStack with shape (n_features, nrows, ncols).

        Returns
        -------
        X : np.ndarray
            Shape (N, F) float64 feature matrix, N = number of pixels
            where at least one feature is finite.  NaN entries are imputed
            with the corresponding feature's prior mean.
        valid_flat_idx : np.ndarray
            1-D array of flat pixel indices for the N returned rows.
        """
        arr = features.as_array()          # (F, H, W)
        F, H, W = arr.shape
        X_flat = arr.reshape(F, H * W).T   # (H*W, F)

        # A pixel is valid if at least one feature is finite.
        # All-NaN pixels (outside Titan's disc) are excluded entirely.
        valid = np.any(np.isfinite(X_flat), axis=1)
        X_valid = X_flat[valid].astype(np.float64)  # (N, F)

        # Impute NaN features with their prior mean.
        # This makes the missing feature contribute no net evidence to the
        # Bayesian update: the prior mean observation is neutral.
        prior_means: np.ndarray = np.array(
            self.priors.prior_mean_vector(), dtype=np.float64
        )                                            # shape (F,)
        for f in range(F):
            nan_mask = ~np.isfinite(X_valid[:, f])
            if nan_mask.any():
                X_valid[nan_mask, f] = prior_means[f]

        # Clip to valid feature range [0, 1]
        X_valid = np.clip(X_valid, 0.0, 1.0)

        n_imputed = int(np.sum(~np.isfinite(X_flat[valid])))
        if n_imputed > 0:
            logger.debug(
                "_feature_matrix: imputed %d NaN feature values with "
                "prior means across %d valid pixels",
                n_imputed, int(valid.sum()),
            )

        return X_valid, np.where(valid)[0]

    def _reconstruct_map(
        self,
        values: np.ndarray,
        valid_idx: np.ndarray,
        fill_value: float,
    ) -> np.ndarray:
        """
        Reconstruct a full (nrows, ncols) map from sparse valid-pixel values.

        Parameters
        ----------
        values:
            1-D array of values for valid pixels, shape (N,).
        valid_idx:
            1-D array of flat indices of valid pixels.
        fill_value:
            Value to assign to invalid/missing pixels.

        Returns
        -------
        np.ndarray
            Shape (nrows, ncols), dtype float32.
        """
        out = np.full(
            self.grid.nrows * self.grid.ncols, fill_value, dtype=np.float32
        )
        out[valid_idx] = values.astype(np.float32)
        return out.reshape(self.grid.nrows, self.grid.ncols)

    def _prior_mean_map(self) -> np.ndarray:
        """
        Compute the weighted-average prior mean as a uniform spatial map.

        Returns
        -------
        np.ndarray
            Shape (nrows, ncols), float32, constant everywhere.
        """
        w = np.array(self.priors.weight_vector(), dtype=np.float64)
        mu = np.array(self.priors.prior_mean_vector(), dtype=np.float64)
        global_mean = float(np.dot(w, mu))
        return np.full(
            (self.grid.nrows, self.grid.ncols), global_mean, dtype=np.float32
        )
