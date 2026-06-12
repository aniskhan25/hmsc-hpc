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


def simulate_spatial_effect_data(
    n_sites: int = 60,
    n_species: int = 6,
    beta: np.ndarray | None = None,
    spatial_range: float = 0.25,
    spatial_sd: float = 1.0,
    noise_sd: float = 0.05,
    distr: str = "probit",
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Simulate a small spatial HMSC-like validation dataset.

    The simulation has one environmental covariate, one shared spatial latent
    site effect, and species-specific loadings. It is intentionally simple and
    deterministic under ``seed`` so tests and validation projects can compare
    fixed, iid, and full-spatial random-intercept fits against known truth.
    """
    if n_sites <= 1:
        raise ValueError("n_sites must be greater than 1")
    if n_species <= 0:
        raise ValueError("n_species must be positive")
    if spatial_range <= 0 or spatial_sd <= 0:
        raise ValueError("spatial_range and spatial_sd must be positive")
    rng = np.random.default_rng(seed)
    coords = _unit_square_grid(n_sites, rng)
    env = rng.normal(size=n_sites)
    beta = np.asarray(
        beta if beta is not None else _default_spatial_beta(n_species),
        dtype=float,
    )
    if beta.shape != (2, n_species):
        raise ValueError(f"beta must have shape {(2, n_species)}")
    dist = _pairwise_distances(coords)
    weights = np.exp(-dist / spatial_range)
    raw_latent = rng.normal(size=n_sites)
    latent = weights @ raw_latent / np.maximum(weights.sum(axis=1), np.finfo(float).eps)
    latent = latent * spatial_sd
    latent = (latent - latent.mean()) / max(latent.std(ddof=1), np.finfo(float).eps)
    loadings = _default_spatial_loadings(n_species)
    design = np.column_stack([np.ones(n_sites), env])
    linear = design @ beta + latent[:, None] * loadings[None, :]
    key = distr.lower()
    if key in {"normal", "gaussian"}:
        Y = linear + rng.normal(scale=noise_sd, size=linear.shape)
    elif key == "poisson":
        Y = rng.poisson(np.exp(np.clip(linear, -6, 6)))
    elif key in {"probit", "bernoulli", "binomial"}:
        probability = _normal_cdf(linear)
        Y = rng.binomial(1, probability)
    else:
        raise ValueError(f"Unsupported distribution {distr!r}")
    site_names = [f"site_{idx + 1:03d}" for idx in range(n_sites)]
    species = [f"sp{idx + 1}" for idx in range(n_species)]
    X = pd.DataFrame({"env": env}, index=site_names)
    study_design = pd.DataFrame(
        {
            "plot": site_names,
            "xcoord": coords[:, 0],
            "ycoord": coords[:, 1],
        },
        index=site_names,
    )
    truth = {
        "beta": pd.DataFrame(beta, index=["Intercept", "env"], columns=species),
        "site_effect": pd.DataFrame({"eta": latent}, index=site_names),
        "lambda": pd.DataFrame([loadings], index=["factor_0"], columns=species),
        "linear_predictor": pd.DataFrame(linear, index=site_names, columns=species),
    }
    return pd.DataFrame(Y, index=site_names, columns=species), X, study_design, truth


def simulate_random_slope_effect_data(
    n_groups: int = 12,
    sites_per_group: int = 4,
    n_species: int = 5,
    beta: np.ndarray | None = None,
    random_sd: float = 0.9,
    distr: str = "probit",
    seed: int = 31,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Simulate a small iid random-slope validation dataset."""
    if n_groups <= 1:
        raise ValueError("n_groups must be greater than 1")
    if sites_per_group <= 0:
        raise ValueError("sites_per_group must be positive")
    if n_species <= 0:
        raise ValueError("n_species must be positive")
    rng = np.random.default_rng(seed)
    n_sites = n_groups * sites_per_group
    groups = np.repeat(np.arange(n_groups), sites_per_group)
    env = rng.normal(size=n_sites)
    group_slope_env = np.linspace(-1.0, 1.0, n_groups)
    group_slope_env = group_slope_env + rng.normal(scale=0.05, size=n_groups)
    slope_env = group_slope_env[groups]
    beta = np.asarray(beta if beta is not None else _default_spatial_beta(n_species), dtype=float)
    if beta.shape != (2, n_species):
        raise ValueError(f"beta must have shape {(2, n_species)}")
    eta = rng.normal(scale=random_sd, size=n_groups)
    lambda_intercept = np.linspace(0.7, -0.7, n_species)
    lambda_slope = np.linspace(-0.9, 0.9, n_species)
    design = np.column_stack([np.ones(n_sites), env])
    random_effect = eta[groups, None] * (
        lambda_intercept[None, :] + slope_env[:, None] * lambda_slope[None, :]
    )
    linear = design @ beta + random_effect
    key = distr.lower()
    if key in {"normal", "gaussian"}:
        Y = linear + rng.normal(scale=0.15, size=linear.shape)
    elif key == "poisson":
        Y = rng.poisson(np.exp(np.clip(linear, -6, 6)))
    elif key in {"probit", "bernoulli", "binomial"}:
        Y = rng.binomial(1, _normal_cdf(linear))
    else:
        raise ValueError(f"Unsupported distribution {distr!r}")
    site_names = [f"site_{idx + 1:03d}" for idx in range(n_sites)]
    plot_names = [f"plot_{idx + 1:02d}" for idx in groups]
    species = [f"sp{idx + 1}" for idx in range(n_species)]
    X = pd.DataFrame({"env": env}, index=site_names)
    study_design = pd.DataFrame(
        {
            "plot": plot_names,
            "slope_env": slope_env,
        },
        index=site_names,
    )
    truth = {
        "beta": pd.DataFrame(beta, index=["Intercept", "env"], columns=species),
        "site_effect": pd.DataFrame({"eta": eta}, index=[f"plot_{idx + 1:02d}" for idx in range(n_groups)]),
        "lambda": pd.DataFrame(
            [lambda_intercept, lambda_slope],
            index=["Intercept", "slope_env"],
            columns=species,
        ),
        "linear_predictor": pd.DataFrame(linear, index=site_names, columns=species),
    }
    return pd.DataFrame(Y, index=site_names, columns=species), X, study_design, truth


def _unit_square_grid(n_sites: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.ceil(np.sqrt(n_sites)))
    grid = np.array([(x, y) for y in np.linspace(0, 1, side) for x in np.linspace(0, 1, side)], dtype=float)
    coords = grid[:n_sites].copy()
    if side > 1:
        coords += rng.normal(scale=0.02 / side, size=coords.shape)
    return np.clip(coords, 0.0, 1.0)


def _default_spatial_beta(n_species: int) -> np.ndarray:
    intercept = np.linspace(-0.8, 0.4, n_species)
    slope = np.linspace(0.9, -0.9, n_species)
    return np.vstack([intercept, slope])


def _default_spatial_loadings(n_species: int) -> np.ndarray:
    values = np.linspace(1.0, -1.0, n_species)
    midpoint = n_species // 2
    if n_species > 1:
        values[midpoint:] -= 0.25
    return values


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr
    except ImportError:
        return 1.0 / (1.0 + np.exp(-value))
    return ndtr(value)
