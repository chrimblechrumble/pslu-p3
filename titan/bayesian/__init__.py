# Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
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
titan/bayesian/__init__.py
===========================
Bayesian backend factory.

Usage
-----
    from titan.bayesian import get_backend
    from configs.pipeline_config import PipelineConfig

    cfg = PipelineConfig(bayesian_backend="sklearn")
    backend = get_backend(cfg)
    result = backend.infer(features)
"""

from __future__ import annotations

from configs.pipeline_config import BayesianPriorConfig, PipelineConfig
from titan.bayesian.base import BayesianBackend, BayesianResult
from titan.preprocessing import CanonicalGrid


def get_backend(
    config: PipelineConfig,
    grid: CanonicalGrid | None = None,
) -> BayesianBackend:
    """
    Return the configured Bayesian inference backend.

    Parameters
    ----------
    config:
        Pipeline configuration.  ``config.bayesian_backend`` selects the
        backend: ``"sklearn"`` | ``"pymc"`` | ``"numpyro"``.
    grid:
        Canonical grid.  Created from config if not supplied.

    Returns
    -------
    BayesianBackend
        Instantiated backend ready to call ``.infer(features)``.

    Raises
    ------
    ValueError
        If ``config.bayesian_backend`` is not one of the known options.
    """
    grid = grid or CanonicalGrid(config.canonical_res_m)
    backend_name = config.bayesian_backend.lower()

    if backend_name == "sklearn":
        from titan.bayesian.sklearn_backend import SklearnBayesianBackend
        return SklearnBayesianBackend(
            priors=config.priors,
            grid=grid,
            random_seed=config.random_seed,
        )

    elif backend_name == "pymc":
        from titan.bayesian.pymc_backend import PyMCBayesianBackend
        return PyMCBayesianBackend(
            priors=config.priors,
            grid=grid,
            random_seed=config.random_seed,
            draws=config.mcmc_draws,
            tune=config.mcmc_tune,
            chains=config.mcmc_chains,
        )

    elif backend_name == "numpyro":
        from titan.bayesian.numpyro_backend import NumPyroBayesianBackend
        return NumPyroBayesianBackend(
            priors=config.priors,
            grid=grid,
            random_seed=config.random_seed,
            num_warmup=config.mcmc_tune,
            num_samples=config.mcmc_draws,
            num_chains=config.mcmc_chains,
        )

    else:
        raise ValueError(
            f"Unknown bayesian_backend: '{backend_name}'. "
            "Choose from: 'sklearn', 'pymc', 'numpyro'."
        )


__all__ = ["get_backend", "BayesianBackend", "BayesianResult"]
