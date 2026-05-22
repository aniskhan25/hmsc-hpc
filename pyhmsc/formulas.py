"""Small formula helpers shared by the Python-facing API."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def normalize_formula(formula: str) -> str:
    """Return an R-compatible one-sided formula string."""
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("x_formula must be a non-empty string")
    formula = formula.strip()
    if not formula.startswith("~"):
        formula = f"~ {formula}"
    return formula


def covariate_names_from_formula(formula: str, data: pd.DataFrame) -> list[str]:
    """Best-effort covariate names for fixed-effect summaries.

    The R bridge is authoritative for fitting. This helper is intentionally
    conservative and covers the simple fixed-effect formulas supported by the
    first Python API milestone.
    """
    formula = normalize_formula(formula)
    rhs = formula[1:].strip()
    include_intercept = "- 1" not in rhs and "+ 0" not in rhs and rhs != "0"
    if rhs in {"", "1"}:
        terms: Iterable[str] = []
    elif rhs == ".":
        terms = data.columns
    else:
        cleaned = rhs.replace("- 1", "").replace("+ 0", "").replace("0 +", "")
        terms = [term.strip() for term in cleaned.split("+")]

    names = ["Intercept"] if include_intercept else []
    for term in terms:
        if not term or term == "1":
            continue
        match = re.fullmatch(r"`([^`]+)`", term)
        names.append(match.group(1) if match else term)
    return names


def build_design_matrix(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    """Build a Python design matrix for summaries and predictions.

    Uses patsy when available; otherwise supports the Phase 1 subset of
    additive numeric covariates with an optional intercept.
    """
    formula = normalize_formula(formula)
    try:
        import patsy  # type: ignore

        rhs = formula[1:].strip()
        return patsy.dmatrix(rhs, data, return_type="dataframe")
    except ImportError:
        names = covariate_names_from_formula(formula, data)
        out = pd.DataFrame(index=data.index)
        if "Intercept" in names:
            out["Intercept"] = 1.0
        for name in names:
            if name == "Intercept":
                continue
            if name not in data:
                raise ValueError(
                    f"Cannot build design matrix without patsy: missing simple column {name!r}"
                )
            out[name] = data[name]
        return out
