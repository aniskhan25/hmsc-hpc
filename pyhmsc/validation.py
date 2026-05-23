"""Pure-Python validation helpers for native workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    details: dict[str, Any]


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
