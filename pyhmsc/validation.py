"""Pure-Python validation helpers for native workflows."""

from __future__ import annotations

import json
from json import JSONDecodeError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    details: dict[str, Any]


def validate_compiled_native_model(path: str | Path | dict[str, Any]) -> list[ValidationResult]:
    """Validate a Python-native compiled model without invoking TensorFlow.

    The checks intentionally focus on the no-R workflow contract: the model is
    compiled as JSON+HDF5, the metadata is internally consistent, and any
    unsupported native sampler features are reported before sampling starts.
    """
    metadata = _load_metadata(path)
    results = [
        _check_native_format(metadata),
        _check_required_dimensions(metadata),
        _check_sampler_supported(metadata),
    ]
    return results


def coefficient_sign_recovery(fit: Any, truth: pd.DataFrame, covariate: str = "x") -> ValidationResult:
    beta = fit.beta_mean()
    common_species = [name for name in truth.columns if name in beta.columns]
    signs = {}
    passed = True
    for species in common_species:
        expected = np.sign(truth.loc[covariate, species])
        observed = np.sign(beta.loc[covariate, species])
        signs[species] = {"expected": float(expected), "observed": float(observed)}
        if expected != 0 and observed != expected:
            passed = False
    return ValidationResult("coefficient_sign_recovery", passed, {"signs": signs})


def predictive_interval_contains_observed_mean(fit: Any, X: pd.DataFrame, Y: pd.DataFrame, level: float = 0.95) -> ValidationResult:
    ci = fit.predict_ci(X, level=level)
    observed = Y.mean(axis=0)
    lower = ci["lower"].mean(axis=0)
    upper = ci["upper"].mean(axis=0)
    checks = {}
    passed = True
    for species in observed.index:
        ok = bool(lower[species] <= observed[species] <= upper[species])
        checks[species] = {
            "observed_mean": float(observed[species]),
            "lower_mean": float(lower[species]),
            "upper_mean": float(upper[species]),
            "passed": ok,
        }
        passed = passed and ok
    return ValidationResult("predictive_interval_contains_observed_mean", passed, checks)


def validate_fit(fit: Any, X: pd.DataFrame | None = None, Y: pd.DataFrame | None = None, truth: pd.DataFrame | None = None) -> list[ValidationResult]:
    results = []
    if truth is not None:
        results.append(coefficient_sign_recovery(fit, truth))
    if X is not None and Y is not None:
        results.append(predictive_interval_contains_observed_mean(fit, X, Y))
    return results


def _load_metadata(path_or_metadata: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_metadata, dict):
        return path_or_metadata
    path = Path(path_or_metadata)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not a compiled pyhmsc init.json file. Run "
            "`python -m pyhmsc compile MODEL.yaml --output run` first, then "
            "validate run/init.json."
        ) from exc


def _check_native_format(metadata: dict[str, Any]) -> ValidationResult:
    ok = metadata.get("format") == "pyhmsc-json-hdf5"
    return ValidationResult(
        "native_format",
        ok,
        {"format": metadata.get("format"), "expected": "pyhmsc-json-hdf5"},
    )


def _check_required_dimensions(metadata: dict[str, Any]) -> ValidationResult:
    required = {"n_sites", "n_species", "n_covariates", "n_chains"}
    dims = metadata.get("dimensions", {})
    missing = sorted(required.difference(dims))
    positive = {
        name: isinstance(dims.get(name), int) and dims.get(name) > 0
        for name in required
        if name in dims
    }
    return ValidationResult(
        "required_dimensions",
        not missing and all(positive.values()),
        {"missing": missing, "positive": positive},
    )


def _check_sampler_supported(metadata: dict[str, Any]) -> ValidationResult:
    unsupported = []
    for level in metadata.get("random_levels", []):
        if int(level.get("xDim", 0)) > 0:
            unsupported.append(
                {
                    "feature": "random_slopes",
                    "random_level": level.get("name"),
                    "reason": "compiled and loadable, but native TensorFlow sampling is not enabled",
                }
            )
    return ValidationResult(
        "native_sampler_supported",
        not unsupported,
        {"unsupported": unsupported},
    )
