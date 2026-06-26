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


def _normalize_distribution(distribution: str) -> str:
    key = str(distribution).lower()
    if key in {"gaussian", "normal"}:
        return "normal"
    if key in {"probit", "bernoulli", "binomial"}:
        return "probit"
    if key == "poisson":
        return "poisson"
    raise ValueError(f"Unsupported distribution {distribution!r}")

