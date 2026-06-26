"""Simulation corpus generation for amortized Neural-HMSC benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.neural.datasets import write_json


@dataclass(frozen=True)
class FixedEffectDataset:
    """One simulated fixed-effect HMSC-like dataset."""

    Y: pd.DataFrame
    X: pd.DataFrame
    truth_beta: pd.DataFrame
    linear_predictor: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TraitEffectDataset(FixedEffectDataset):
    """One simulated trait-mediated fixed-effect dataset."""

    traits: pd.DataFrame
    trait_design: pd.DataFrame
    truth_gamma: pd.DataFrame


@dataclass(frozen=True)
class IidLatentEffectDataset(FixedEffectDataset):
    """One simulated iid latent random-intercept dataset."""

    study_design: pd.DataFrame
    group_codes: np.ndarray
    truth_eta: pd.DataFrame
    truth_lambda: pd.DataFrame
    truth_random_effect: pd.DataFrame


@dataclass(frozen=True)
class SpatialLatentEffectDataset(IidLatentEffectDataset):
    """One simulated full-spatial latent random-effect dataset."""

    coords: pd.DataFrame
    train_mask: np.ndarray
    test_mask: np.ndarray


def simulate_fixed_effect_dataset(
    *,
    n_sites: int,
    n_species: int,
    distribution: str,
    seed: int,
    beta_scale: float = 0.75,
    beta_zero_probability: float = 0.0,
    gaussian_sigma: float = 0.35,
    poisson_eta_clip: tuple[float, float] = (-6.0, 6.0),
) -> FixedEffectDataset:
    """Simulate a variable-shape fixed-effect benchmark dataset.

    The raw covariate table contains ``x1`` and ``x2``. The intercept appears in
    ``truth_beta`` and in compiled artifacts through the Patsy formula
    ``~ x1 + x2``.
    """
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if n_species <= 0:
        raise ValueError("n_species must be positive")
    if beta_scale <= 0:
        raise ValueError("beta_scale must be positive")
    if not 0.0 <= beta_zero_probability < 1.0:
        raise ValueError("beta_zero_probability must be in [0, 1)")
    if gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")
    if poisson_eta_clip[0] >= poisson_eta_clip[1]:
        raise ValueError("poisson_eta_clip must be ordered as (low, high)")

    rng = np.random.default_rng(seed)
    site_names = [f"site_{idx + 1:04d}" for idx in range(n_sites)]
    species_names = [f"sp{idx + 1}" for idx in range(n_species)]
    covariate_names = ["Intercept", "x1", "x2"]

    x1 = rng.normal(size=n_sites)
    x2 = rng.normal(size=n_sites)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=site_names)
    design = np.column_stack([np.ones(n_sites), x1, x2])

    beta = rng.normal(loc=0.0, scale=beta_scale, size=(len(covariate_names), n_species))
    if beta_zero_probability > 0:
        keep = rng.uniform(size=beta.shape) >= beta_zero_probability
        beta = beta * keep
    linear = design @ beta

    key = _normalize_distribution(distribution)
    if key == "normal":
        Y_values = linear + rng.normal(scale=gaussian_sigma, size=linear.shape)
    elif key == "probit":
        Y_values = rng.binomial(1, ndtr(linear))
    elif key == "poisson":
        low, high = poisson_eta_clip
        Y_values = rng.poisson(np.exp(np.clip(linear, low, high)))
    else:
        raise ValueError(f"Unsupported fixed-effect benchmark distribution {distribution!r}")

    Y = pd.DataFrame(Y_values, index=site_names, columns=species_names)
    truth_beta = pd.DataFrame(beta, index=covariate_names, columns=species_names)
    linear_predictor = pd.DataFrame(linear, index=site_names, columns=species_names)
    metadata = {
        "distribution": key,
        "seed": int(seed),
        "n_sites": int(n_sites),
        "n_species": int(n_species),
        "n_covariates": len(covariate_names),
        "formula": "~ x1 + x2",
        "beta_scale": float(beta_scale),
        "beta_zero_probability": float(beta_zero_probability),
    }
    if key == "normal":
        metadata["gaussian_sigma"] = float(gaussian_sigma)
    if key == "poisson":
        metadata["poisson_eta_clip"] = [float(poisson_eta_clip[0]), float(poisson_eta_clip[1])]
    return FixedEffectDataset(
        Y=Y,
        X=X,
        truth_beta=truth_beta,
        linear_predictor=linear_predictor,
        metadata=metadata,
    )


def simulate_spatial_latent_effect_dataset(
    *,
    n_sites: int,
    n_species: int,
    n_factors: int = 1,
    distribution: str = "normal",
    seed: int,
    beta_scale: float = 0.25,
    spatial_range: float = 0.25,
    spatial_sd: float = 1.0,
    lambda_scale: float = 1.0,
    gaussian_sigma: float = 0.08,
    holdout_stride: int = 5,
    poisson_eta_clip: tuple[float, float] = (-6.0, 6.0),
) -> SpatialLatentEffectDataset:
    """Simulate fixed effects plus full-spatial latent random intercepts."""
    if n_sites <= 3:
        raise ValueError("n_sites must be greater than three")
    if n_species <= 1:
        raise ValueError("n_species must be greater than one")
    if n_factors <= 0:
        raise ValueError("n_factors must be positive")
    if spatial_range <= 0 or spatial_sd <= 0 or lambda_scale <= 0:
        raise ValueError("spatial_range, spatial_sd, and lambda_scale must be positive")
    if gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")
    if holdout_stride < 2:
        raise ValueError("holdout_stride must be at least two")
    if poisson_eta_clip[0] >= poisson_eta_clip[1]:
        raise ValueError("poisson_eta_clip must be ordered as (low, high)")

    rng = np.random.default_rng(seed)
    site_names = [f"site_{idx + 1:04d}" for idx in range(n_sites)]
    species_names = [f"sp{idx + 1}" for idx in range(n_species)]
    factor_names = [f"factor_{idx}" for idx in range(n_factors)]
    covariate_names = ["Intercept", "x1", "x2"]
    coords_array = _unit_square_grid(n_sites, rng)

    x1 = rng.normal(size=n_sites)
    x2 = rng.normal(size=n_sites)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=site_names)
    design = np.column_stack([np.ones(n_sites), x1, x2])
    coords = pd.DataFrame({"xcoord": coords_array[:, 0], "ycoord": coords_array[:, 1]}, index=site_names)
    group_codes = np.arange(n_sites, dtype=int)
    study_design = pd.DataFrame(
        {"plot": site_names, "xcoord": coords_array[:, 0], "ycoord": coords_array[:, 1]},
        index=site_names,
    )

    distances = _pairwise_distances(coords_array)
    covariance = np.exp(-distances / spatial_range) + np.eye(n_sites) * 1e-6
    eta_columns = []
    for _ in range(n_factors):
        latent = rng.multivariate_normal(np.zeros(n_sites), covariance)
        latent = latent * spatial_sd
        latent = (latent - latent.mean()) / max(latent.std(ddof=1), np.finfo(float).eps)
        eta_columns.append(latent)
    eta = np.column_stack(eta_columns)
    loadings = rng.normal(scale=lambda_scale, size=(n_factors, n_species))
    beta = rng.normal(scale=beta_scale, size=(len(covariate_names), n_species))
    random_effect = eta @ loadings
    linear = design @ beta + random_effect

    key = _normalize_distribution(distribution)
    if key == "normal":
        Y_values = linear + rng.normal(scale=gaussian_sigma, size=linear.shape)
    elif key == "probit":
        Y_values = rng.binomial(1, ndtr(linear))
    elif key == "poisson":
        low, high = poisson_eta_clip
        Y_values = rng.poisson(np.exp(np.clip(linear, low, high)))
    else:
        raise ValueError(f"Unsupported spatial latent benchmark distribution {distribution!r}")

    test_mask = np.arange(n_sites) % holdout_stride == 0
    train_mask = ~test_mask
    if not test_mask.any() or not train_mask.any():
        raise ValueError("holdout split must include both train and test sites")
    Y = pd.DataFrame(Y_values, index=site_names, columns=species_names)
    truth_beta = pd.DataFrame(beta, index=covariate_names, columns=species_names)
    truth_eta = pd.DataFrame(eta, index=site_names, columns=factor_names)
    truth_lambda = pd.DataFrame(loadings, index=factor_names, columns=species_names)
    truth_random_effect = pd.DataFrame(random_effect, index=site_names, columns=species_names)
    linear_predictor = pd.DataFrame(linear, index=site_names, columns=species_names)
    metadata = {
        "distribution": key,
        "seed": int(seed),
        "n_sites": int(n_sites),
        "n_species": int(n_species),
        "n_covariates": len(covariate_names),
        "n_groups": int(n_sites),
        "n_factors": int(n_factors),
        "formula": "~ x1 + x2",
        "random_level": {
            "name": "plot",
            "column": "plot",
            "type": "spatial_full",
            "coords": ["xcoord", "ycoord"],
        },
        "beta_scale": float(beta_scale),
        "spatial_range": float(spatial_range),
        "spatial_sd": float(spatial_sd),
        "lambda_scale": float(lambda_scale),
        "holdout_stride": int(holdout_stride),
    }
    return SpatialLatentEffectDataset(
        Y=Y,
        X=X,
        truth_beta=truth_beta,
        linear_predictor=linear_predictor,
        metadata=metadata,
        study_design=study_design,
        group_codes=group_codes,
        truth_eta=truth_eta,
        truth_lambda=truth_lambda,
        truth_random_effect=truth_random_effect,
        coords=coords,
        train_mask=train_mask,
        test_mask=test_mask,
    )


def simulate_iid_latent_effect_dataset(
    *,
    n_sites: int,
    n_species: int,
    n_groups: int | None = None,
    n_factors: int = 1,
    distribution: str = "normal",
    seed: int,
    beta_scale: float = 0.35,
    eta_scale: float = 1.0,
    lambda_scale: float = 1.0,
    gaussian_sigma: float = 0.15,
    poisson_eta_clip: tuple[float, float] = (-6.0, 6.0),
) -> IidLatentEffectDataset:
    """Simulate fixed effects plus iid latent random intercepts."""
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if n_species <= 1:
        raise ValueError("n_species must be greater than one")
    if n_factors <= 0:
        raise ValueError("n_factors must be positive")
    n_groups = int(n_groups if n_groups is not None else n_sites)
    if n_groups <= 0 or n_groups > n_sites:
        raise ValueError("n_groups must be in [1, n_sites]")
    if n_groups < n_factors:
        raise ValueError("n_groups must be at least n_factors")
    if beta_scale <= 0 or eta_scale <= 0 or lambda_scale <= 0:
        raise ValueError("beta_scale, eta_scale, and lambda_scale must be positive")
    if gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")
    if poisson_eta_clip[0] >= poisson_eta_clip[1]:
        raise ValueError("poisson_eta_clip must be ordered as (low, high)")

    rng = np.random.default_rng(seed)
    site_names = [f"site_{idx + 1:04d}" for idx in range(n_sites)]
    species_names = [f"sp{idx + 1}" for idx in range(n_species)]
    factor_names = [f"factor_{idx}" for idx in range(n_factors)]
    group_names = [f"plot_{idx + 1:04d}" for idx in range(n_groups)]
    covariate_names = ["Intercept", "x1", "x2"]

    x1 = rng.normal(size=n_sites)
    x2 = rng.normal(size=n_sites)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=site_names)
    design = np.column_stack([np.ones(n_sites), x1, x2])
    group_codes = np.arange(n_sites, dtype=int) % n_groups
    if n_groups < n_sites:
        rng.shuffle(group_codes)
    study_design = pd.DataFrame(
        {"plot": [group_names[code] for code in group_codes]},
        index=site_names,
    )

    beta = rng.normal(scale=beta_scale, size=(len(covariate_names), n_species))
    eta_raw = rng.normal(size=(n_groups, n_factors))
    eta, _ = np.linalg.qr(eta_raw)
    eta = eta[:, :n_factors] * eta_scale
    loadings = rng.normal(scale=lambda_scale, size=(n_factors, n_species))
    random_effect = eta[group_codes] @ loadings
    linear = design @ beta + random_effect

    key = _normalize_distribution(distribution)
    if key == "normal":
        Y_values = linear + rng.normal(scale=gaussian_sigma, size=linear.shape)
    elif key == "probit":
        Y_values = rng.binomial(1, ndtr(linear))
    elif key == "poisson":
        low, high = poisson_eta_clip
        Y_values = rng.poisson(np.exp(np.clip(linear, low, high)))
    else:
        raise ValueError(f"Unsupported iid latent benchmark distribution {distribution!r}")

    Y = pd.DataFrame(Y_values, index=site_names, columns=species_names)
    truth_beta = pd.DataFrame(beta, index=covariate_names, columns=species_names)
    truth_eta = pd.DataFrame(eta, index=group_names, columns=factor_names)
    truth_lambda = pd.DataFrame(loadings, index=factor_names, columns=species_names)
    truth_random_effect = pd.DataFrame(random_effect, index=site_names, columns=species_names)
    linear_predictor = pd.DataFrame(linear, index=site_names, columns=species_names)
    metadata = {
        "distribution": key,
        "seed": int(seed),
        "n_sites": int(n_sites),
        "n_species": int(n_species),
        "n_covariates": len(covariate_names),
        "n_groups": int(n_groups),
        "n_factors": int(n_factors),
        "formula": "~ x1 + x2",
        "random_level": {"name": "plot", "column": "plot", "type": "iid"},
        "beta_scale": float(beta_scale),
        "eta_scale": float(eta_scale),
        "lambda_scale": float(lambda_scale),
    }
    if key == "normal":
        metadata["gaussian_sigma"] = float(gaussian_sigma)
    if key == "poisson":
        metadata["poisson_eta_clip"] = [float(poisson_eta_clip[0]), float(poisson_eta_clip[1])]
    return IidLatentEffectDataset(
        Y=Y,
        X=X,
        truth_beta=truth_beta,
        linear_predictor=linear_predictor,
        metadata=metadata,
        study_design=study_design,
        group_codes=group_codes,
        truth_eta=truth_eta,
        truth_lambda=truth_lambda,
        truth_random_effect=truth_random_effect,
    )


def simulate_trait_effect_dataset(
    *,
    n_sites: int,
    n_species: int,
    distribution: str,
    seed: int,
    gamma_scale: float = 0.75,
    beta_residual_scale: float = 0.05,
    gaussian_sigma: float = 0.35,
    poisson_eta_clip: tuple[float, float] = (-6.0, 6.0),
) -> TraitEffectDataset:
    """Simulate a fixed-effect dataset where species traits mediate Beta.

    The trait design is ``Intercept, body`` and the environmental design is
    ``Intercept, x1, x2``. Truth follows ``Beta = Gamma @ T.T + residual``.
    """
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if n_species <= 1:
        raise ValueError("n_species must be greater than one for trait effects")
    if gamma_scale <= 0:
        raise ValueError("gamma_scale must be positive")
    if beta_residual_scale < 0:
        raise ValueError("beta_residual_scale must be non-negative")
    if gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")
    if poisson_eta_clip[0] >= poisson_eta_clip[1]:
        raise ValueError("poisson_eta_clip must be ordered as (low, high)")

    rng = np.random.default_rng(seed)
    site_names = [f"site_{idx + 1:04d}" for idx in range(n_sites)]
    species_names = [f"sp{idx + 1}" for idx in range(n_species)]
    covariate_names = ["Intercept", "x1", "x2"]
    trait_names = ["Intercept", "body"]

    x1 = rng.normal(size=n_sites)
    x2 = rng.normal(size=n_sites)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=site_names)
    design = np.column_stack([np.ones(n_sites), x1, x2])

    body = np.linspace(-1.0, 1.0, n_species) + rng.normal(scale=0.15, size=n_species)
    traits = pd.DataFrame({"body": body}, index=species_names)
    trait_design = pd.DataFrame(
        {"Intercept": np.ones(n_species), "body": body},
        index=species_names,
    )
    gamma = rng.normal(loc=0.0, scale=gamma_scale, size=(len(covariate_names), len(trait_names)))
    beta = gamma @ trait_design.to_numpy(dtype=float).T
    if beta_residual_scale > 0:
        beta = beta + rng.normal(scale=beta_residual_scale, size=beta.shape)
    linear = design @ beta

    key = _normalize_distribution(distribution)
    if key == "normal":
        Y_values = linear + rng.normal(scale=gaussian_sigma, size=linear.shape)
    elif key == "probit":
        Y_values = rng.binomial(1, ndtr(linear))
    elif key == "poisson":
        low, high = poisson_eta_clip
        Y_values = rng.poisson(np.exp(np.clip(linear, low, high)))
    else:
        raise ValueError(f"Unsupported trait-effect benchmark distribution {distribution!r}")

    Y = pd.DataFrame(Y_values, index=site_names, columns=species_names)
    truth_beta = pd.DataFrame(beta, index=covariate_names, columns=species_names)
    truth_gamma = pd.DataFrame(gamma, index=covariate_names, columns=trait_names)
    linear_predictor = pd.DataFrame(linear, index=site_names, columns=species_names)
    metadata = {
        "distribution": key,
        "seed": int(seed),
        "n_sites": int(n_sites),
        "n_species": int(n_species),
        "n_covariates": len(covariate_names),
        "n_traits": len(trait_names),
        "formula": "~ x1 + x2",
        "trait_formula": "~ body",
        "gamma_scale": float(gamma_scale),
        "beta_residual_scale": float(beta_residual_scale),
    }
    if key == "normal":
        metadata["gaussian_sigma"] = float(gaussian_sigma)
    if key == "poisson":
        metadata["poisson_eta_clip"] = [float(poisson_eta_clip[0]), float(poisson_eta_clip[1])]
    return TraitEffectDataset(
        Y=Y,
        X=X,
        truth_beta=truth_beta,
        linear_predictor=linear_predictor,
        metadata=metadata,
        traits=traits,
        trait_design=trait_design,
        truth_gamma=truth_gamma,
    )


def generate_fixed_effect_corpus(
    config: dict[str, Any],
    output: str | Path,
    *,
    profile: str = "smoke",
    chains: int | None = None,
) -> dict[str, Any]:
    """Generate a fixed-effect Neural-HMSC benchmark corpus from a config."""
    _validate_fixed_effect_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    benchmark = config["benchmark"]
    model = config["model"]
    simulation = config["simulation"]
    corpus_sizes = simulation["corpus_sizes"][profile]
    dimensions = simulation["dimensions"]
    response = simulation.get("response", {})
    beta_config = simulation["beta"]
    rng = np.random.default_rng(int(simulation.get("seed", 1)))
    formula = model["formula"]["X"] if isinstance(model.get("formula"), dict) else model["formula"]
    distribution = _normalize_distribution(model["distribution"])
    chain_count = int(chains if chains is not None else _mcmc_local_chains(config))

    manifest: dict[str, Any] = {
        "benchmark": benchmark.get("name"),
        "profile": profile,
        "model_type": model.get("model_type"),
        "distribution": distribution,
        "formula": formula,
        "chains": chain_count,
        "splits": {},
        "config": config,
    }

    for split, count in corpus_sizes.items():
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for dataset_idx in range(int(count)):
            dataset_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            n_sites = _sample_dimension(rng, dimensions["n_sites"])
            n_species = _sample_dimension(rng, dimensions["n_species"])
            dataset = simulate_fixed_effect_dataset(
                n_sites=n_sites,
                n_species=n_species,
                distribution=distribution,
                seed=dataset_seed,
                beta_scale=float(beta_config.get("sd", 0.75)),
                beta_zero_probability=float(beta_config.get("zero_probability", 0.0)),
                gaussian_sigma=float(response.get("gaussian_sigma", 0.35)),
                poisson_eta_clip=tuple(response.get("eta_clip", [-6.0, 6.0])),
            )
            dataset_name = f"dataset_{dataset_idx:06d}"
            dataset_dir = split_dir / dataset_name
            _write_fixed_effect_dataset(
                dataset,
                dataset_dir,
                formula=formula,
                distribution=distribution,
                chains=chain_count,
            )
            records.append(
                {
                    "name": dataset_name,
                    "path": str(dataset_dir.relative_to(output)),
                    "seed": dataset_seed,
                    "n_sites": n_sites,
                    "n_species": n_species,
                    "compiled": str((dataset_dir / "compiled" / "init.json").relative_to(output)),
                    "truth_beta": str((dataset_dir / "data" / "truth_beta.csv").relative_to(output)),
                }
            )
        manifest["splits"][split] = {"count": int(count), "datasets": records}

    write_json(output / "corpus_metadata.json", manifest)
    return manifest


def _write_fixed_effect_dataset(
    dataset: FixedEffectDataset,
    output: Path,
    *,
    formula: str,
    distribution: str,
    chains: int,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Dataset output directory is not empty: {output}")
    data_dir = output / "data"
    compiled_dir = output / "compiled"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.Y.to_csv(data_dir / "Y.csv")
    dataset.X.to_csv(data_dir / "X.csv")
    dataset.truth_beta.to_csv(data_dir / "truth_beta.csv")
    dataset.linear_predictor.to_csv(data_dir / "truth_linear_predictor.csv")
    write_json(output / "dataset_metadata.json", dataset.metadata)
    _write_model_yaml(
        output / "model.yaml",
        formula=formula,
        distribution=distribution,
        chains=chains,
    )
    compile_hmsc_model(
        Y=dataset.Y,
        X=dataset.X,
        formula=formula,
        distr=distribution,
        chains=chains,
        output=compiled_dir,
    )


def _write_model_yaml(path: Path, *, formula: str, distribution: str, chains: int) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to write benchmark model configs") from exc
    payload = {
        "response": "data/Y.csv",
        "covariates": "data/X.csv",
        "formula": {"X": formula},
        "distribution": distribution,
        "chains": int(chains),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _validate_fixed_effect_config(config: dict[str, Any]) -> None:
    required = ["benchmark", "model", "simulation"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Benchmark config missing required fields: {missing}")
    model = config["model"]
    if model.get("model_type") != "fixed_effect":
        raise ValueError("Milestone 1 generator only supports model_type='fixed_effect'")
    unsupported = model.get("unsupported", {})
    if not all(bool(unsupported.get(key, False)) for key in ["traits", "phylogeny", "random_levels", "spatial"]):
        raise ValueError("Fixed-effect benchmark config must explicitly mark advanced features unsupported")
    simulation = config["simulation"]
    for key in ["dimensions", "beta", "corpus_sizes"]:
        if key not in simulation:
            raise ValueError(f"simulation.{key} is required")


def _sample_dimension(rng: np.random.Generator, spec: dict[str, Any]) -> int:
    if "value" in spec:
        return int(spec["value"])
    values = spec.get("values")
    if not values:
        raise ValueError("Dimension spec must provide 'value' or non-empty 'values'")
    return int(rng.choice(np.asarray(values, dtype=int)))


def _mcmc_local_chains(config: dict[str, Any]) -> int:
    return int(config.get("mcmc_reference", {}).get("local", {}).get("chains", 2))


def _unit_square_grid(n_sites: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.ceil(np.sqrt(n_sites)))
    grid = np.array([(x, y) for y in np.linspace(0, 1, side) for x in np.linspace(0, 1, side)], dtype=float)
    coords = grid[:n_sites].copy()
    coords += rng.normal(scale=0.01, size=coords.shape)
    return np.clip(coords, 0.0, 1.0)


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _normalize_distribution(distribution: str) -> str:
    key = str(distribution).lower()
    if key in {"gaussian", "normal"}:
        return "normal"
    if key in {"probit", "bernoulli", "binomial"}:
        return "probit"
    if key == "poisson":
        return "poisson"
    raise ValueError(f"Unsupported distribution {distribution!r}")
