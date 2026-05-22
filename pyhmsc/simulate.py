"""Simulation helpers for pure-Python validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_fixed_effect_data(
    n_sites: int = 40,
    beta: np.ndarray | None = None,
    distr: str = "normal",
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_sites)
    X = pd.DataFrame({"x": x})
    beta = np.asarray(beta if beta is not None else [[0.2, -0.2], [0.8, -0.8]], dtype=float)
    design = np.column_stack([np.ones(n_sites), x])
    eta = design @ beta
    key = distr.lower()
    if key in {"normal", "gaussian"}:
        Y = eta + rng.normal(scale=0.25, size=eta.shape)
    elif key == "poisson":
        Y = rng.poisson(np.exp(np.clip(eta, -6, 6)))
    elif key in {"probit", "bernoulli"}:
        from scipy.special import expit

        Y = rng.binomial(1, expit(eta))
    else:
        raise ValueError(f"Unsupported distribution {distr!r}")
    species = [f"sp{i + 1}" for i in range(beta.shape[1])]
    truth = pd.DataFrame(beta, index=["Intercept", "x"], columns=species)
    return pd.DataFrame(Y, columns=species), X, truth
