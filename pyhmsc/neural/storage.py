"""Storage adapters for experimental Neural-HMSC posterior outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyhmsc.neural.calibration import BetaScaleCalibration, calibration_metadata
from pyhmsc.neural.posterior_heads import BetaPosterior, GammaPosterior, IidLatentPosterior


def write_beta_posterior_hdf5(
    posterior: BetaPosterior,
    output: str | Path,
    *,
    covariate_names: Sequence[str],
    species_names: Sequence[str],
    distribution: str = "normal",
    formula: str = "~ x1 + x2",
    chains: int = 1,
    draws: int = 100,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    calibration: BetaScaleCalibration | dict[str, Any] | None = None,
) -> Path:
    """Write a neural Beta posterior to HDF5 using pyhmsc posterior shapes.

    The written ``Beta`` dataset has shape
    ``chains x draws x n_covariates x n_species`` and is readable by
    :meth:`pyhmsc.posterior.HmscFit.from_file`.
    """
    if chains <= 0:
        raise ValueError("chains must be positive")
    if draws <= 0:
        raise ValueError("draws must be positive")
    mean = _as_numpy(posterior.mean)
    scale = _as_numpy(posterior.scale)
    if mean.ndim != 3 or mean.shape[0] != 1:
        raise ValueError("write_beta_posterior_hdf5 currently supports one posterior dataset at a time")
    if scale.shape != mean.shape:
        raise ValueError("posterior mean and scale must have the same shape")
    beta_mean = mean[0]
    beta_scale = scale[0]
    if beta_mean.shape != (len(covariate_names), len(species_names)):
        raise ValueError("covariate/species names do not match posterior Beta shape")

    rng = np.random.default_rng(seed)
    beta = rng.normal(
        loc=beta_mean[None, None, :, :],
        scale=beta_scale[None, None, :, :],
        size=(chains, draws) + beta_mean.shape,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    posterior_metadata = _neural_metadata(
        covariate_names=covariate_names,
        species_names=species_names,
        distribution=distribution,
        formula=formula,
        chains=chains,
        draws=draws,
        seed=seed,
        metadata=metadata,
        calibration=calibration,
    )

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to write Neural-HMSC posterior files") from exc
    with h5py.File(output, "w") as handle:
        handle.create_dataset("Beta", data=beta)
        handle.attrs["nChains"] = int(chains)
        handle.attrs["nDraws"] = int(draws)
        handle.attrs["pyhmsc_metadata"] = json.dumps(posterior_metadata)
    return output


def write_gamma_posterior_hdf5(
    posterior: GammaPosterior,
    output: str | Path,
    *,
    covariate_names: Sequence[str],
    trait_names: Sequence[str],
    distribution: str = "normal",
    formula: str = "~ x1 + x2",
    trait_formula: str = "~ body",
    chains: int = 1,
    draws: int = 100,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a neural Gamma posterior to HDF5 using pyhmsc posterior shapes."""
    if chains <= 0:
        raise ValueError("chains must be positive")
    if draws <= 0:
        raise ValueError("draws must be positive")
    mean = _as_numpy(posterior.mean)
    scale = _as_numpy(posterior.scale)
    if mean.ndim != 3 or mean.shape[0] != 1:
        raise ValueError("write_gamma_posterior_hdf5 currently supports one posterior dataset at a time")
    if scale.shape != mean.shape:
        raise ValueError("posterior mean and scale must have the same shape")
    gamma_mean = mean[0]
    gamma_scale = scale[0]
    if gamma_mean.shape != (len(covariate_names), len(trait_names)):
        raise ValueError("covariate/trait names do not match posterior Gamma shape")

    rng = np.random.default_rng(seed)
    gamma = rng.normal(
        loc=gamma_mean[None, None, :, :],
        scale=gamma_scale[None, None, :, :],
        size=(chains, draws) + gamma_mean.shape,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    gamma_metadata = dict(metadata or {})
    gamma_formula = dict(gamma_metadata.get("formula", {}))
    gamma_formula["T"] = trait_formula
    gamma_metadata["formula"] = gamma_formula
    gamma_names = dict(gamma_metadata.get("names", {}))
    gamma_names["traits"] = [str(name) for name in trait_names]
    gamma_metadata["names"] = gamma_names
    posterior_metadata = _neural_metadata(
        covariate_names=covariate_names,
        species_names=[],
        distribution=distribution,
        formula=formula,
        chains=chains,
        draws=draws,
        seed=seed,
        metadata=gamma_metadata,
        calibration=None,
    )
    posterior_metadata["inference"]["parameter"] = "Gamma"

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to write Neural-HMSC posterior files") from exc
    with h5py.File(output, "w") as handle:
        handle.create_dataset("Gamma", data=gamma)
        handle.attrs["nChains"] = int(chains)
        handle.attrs["nDraws"] = int(draws)
        handle.attrs["pyhmsc_metadata"] = json.dumps(posterior_metadata)
    return output


def write_iid_latent_posterior_hdf5(
    posterior: IidLatentPosterior,
    output: str | Path,
    *,
    covariate_names: Sequence[str],
    species_names: Sequence[str],
    group_names: Sequence[str],
    distribution: str = "normal",
    formula: str = "~ x1 + x2",
    random_level_name: str = "plot",
    random_level_type: str = "iid",
    coords: np.ndarray | None = None,
    chains: int = 1,
    draws: int = 100,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write neural iid latent-factor posterior samples to HDF5."""
    if chains <= 0:
        raise ValueError("chains must be positive")
    if draws <= 0:
        raise ValueError("draws must be positive")
    beta_mean = _single_posterior_array(posterior.beta_mean, "beta_mean")
    beta_scale = _single_posterior_array(posterior.beta_scale, "beta_scale")
    eta_mean = _single_posterior_array(posterior.eta_mean, "eta_mean")
    eta_scale = _single_posterior_array(posterior.eta_scale, "eta_scale")
    lambda_mean = _single_posterior_array(posterior.lambda_mean, "lambda_mean")
    lambda_scale = _single_posterior_array(posterior.lambda_scale, "lambda_scale")
    if beta_scale.shape != beta_mean.shape or eta_scale.shape != eta_mean.shape or lambda_scale.shape != lambda_mean.shape:
        raise ValueError("posterior means and scales must have matching shapes")
    if beta_mean.shape != (len(covariate_names), len(species_names)):
        raise ValueError("covariate/species names do not match posterior Beta shape")
    if eta_mean.shape[0] != len(group_names):
        raise ValueError("group names do not match posterior Eta shape")
    if lambda_mean.shape[1] != len(species_names):
        raise ValueError("species names do not match posterior Lambda shape")
    if eta_mean.shape[1] != lambda_mean.shape[0]:
        raise ValueError("Eta and Lambda factor dimensions do not match")

    rng = np.random.default_rng(seed)
    beta = rng.normal(
        loc=beta_mean[None, None, :, :],
        scale=beta_scale[None, None, :, :],
        size=(chains, draws) + beta_mean.shape,
    )
    eta = rng.normal(
        loc=eta_mean[None, None, :, :],
        scale=eta_scale[None, None, :, :],
        size=(chains, draws) + eta_mean.shape,
    )
    loadings = rng.normal(
        loc=lambda_mean[None, None, :, :],
        scale=lambda_scale[None, None, :, :],
        size=(chains, draws) + lambda_mean.shape,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    posterior_metadata = _neural_metadata(
        covariate_names=covariate_names,
        species_names=species_names,
        distribution=distribution,
        formula=formula,
        chains=chains,
        draws=draws,
        seed=seed,
        metadata=metadata,
        calibration=None,
    )
    posterior_metadata["inference"]["parameter"] = "Beta+Eta+Lambda"
    posterior_metadata["random_levels"] = [
        {
            "name": str(random_level_name),
            "column": str(random_level_name),
            "type": str(random_level_type),
            "levels": [str(name) for name in group_names],
            "nf": int(eta_mean.shape[1]),
        }
    ]
    if coords is not None:
        coord_array = np.asarray(coords, dtype=float)
        if coord_array.shape != (len(group_names), 2):
            raise ValueError(f"coords must have shape {(len(group_names), 2)}")
        posterior_metadata["random_levels"][0]["coords"] = ["xcoord", "ycoord"]
        posterior_metadata["random_levels"][0]["coordinate_values"] = coord_array.tolist()

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to write Neural-HMSC posterior files") from exc
    with h5py.File(output, "w") as handle:
        handle.create_dataset("Beta", data=beta)
        random_levels = handle.create_group("random_levels")
        level = random_levels.create_group("0")
        level.create_dataset("Eta", data=eta)
        level.create_dataset("Lambda", data=loadings)
        handle.attrs["nChains"] = int(chains)
        handle.attrs["nDraws"] = int(draws)
        handle.attrs["pyhmsc_metadata"] = json.dumps(posterior_metadata)
    return output


def write_spatial_latent_posterior_hdf5(
    posterior: IidLatentPosterior,
    output: str | Path,
    *,
    covariate_names: Sequence[str],
    species_names: Sequence[str],
    site_names: Sequence[str],
    coords: np.ndarray,
    distribution: str = "normal",
    formula: str = "~ x1 + x2",
    random_level_name: str = "plot",
    chains: int = 1,
    draws: int = 100,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write full-spatial neural latent-factor posterior samples to HDF5."""
    return write_iid_latent_posterior_hdf5(
        posterior,
        output,
        covariate_names=covariate_names,
        species_names=species_names,
        group_names=site_names,
        distribution=distribution,
        formula=formula,
        random_level_name=random_level_name,
        random_level_type="spatial_full",
        coords=coords,
        chains=chains,
        draws=draws,
        seed=seed,
        metadata=metadata,
    )


def _neural_metadata(
    *,
    covariate_names: Sequence[str],
    species_names: Sequence[str],
    distribution: str,
    formula: str,
    chains: int,
    draws: int,
    seed: int | None,
    metadata: dict[str, Any] | None,
    calibration: BetaScaleCalibration | dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "model_type": "neural-hmsc",
        "inference": {
            "engine": "amortized-neural",
            "posterior_family": "diagonal_normal",
            "parameter": "Beta",
            "chains": int(chains),
            "draws": int(draws),
            "seed": None if seed is None else int(seed),
        },
        "names": {
            "covariates": [str(name) for name in covariate_names],
            "species": [str(name) for name in species_names],
            "traits": ["Intercept"],
        },
        "formula": {"X": formula},
        "distribution": str(distribution),
    }
    if calibration is not None:
        base["calibration"] = calibration_metadata(calibration)
    if metadata:
        extra = dict(metadata)
        extra_names = extra.pop("names", {})
        extra_formula = extra.pop("formula", {})
        extra_inference = extra.pop("inference", {})
        base.update(extra)
        if isinstance(extra_names, dict):
            base["names"].update(extra_names)
        base["names"].update(
            {
                "covariates": [str(name) for name in covariate_names],
                "species": [str(name) for name in species_names],
            }
        )
        if isinstance(extra_formula, dict):
            base["formula"].update(extra_formula)
        base["formula"]["X"] = formula
        if isinstance(extra_inference, dict):
            base["inference"].update(extra_inference)
        base["inference"].update(
            {
                "engine": "amortized-neural",
                "posterior_family": "diagonal_normal",
                "parameter": "Beta",
                "chains": int(chains),
                "draws": int(draws),
                "seed": None if seed is None else int(seed),
            }
        )
        if calibration is not None:
            base["calibration"] = calibration_metadata(calibration)
    return base


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _single_posterior_array(value: Any, name: str) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"{name} currently supports one posterior dataset at a time")
    return array[0]
