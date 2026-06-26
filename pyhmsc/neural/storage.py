"""Storage adapters for experimental Neural-HMSC posterior outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyhmsc.neural.calibration import BetaScaleCalibration, calibration_metadata
from pyhmsc.neural.posterior_heads import BetaPosterior


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
