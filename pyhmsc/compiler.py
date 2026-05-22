"""Compile raw Python data into a Python-native Hmsc-HPC model artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyhmsc.formulas import build_design_matrix, normalize_formula
from pyhmsc.serialization import write_compiled_model


class CompiledModel:
    """Location and metadata for a compiled JSON+HDF5 model."""

    def __init__(self, init_json: Path, metadata: dict[str, Any]) -> None:
        self.init_json = init_json
        self.metadata = metadata


def compile_hmsc_model(
    Y: Any,
    X: Any,
    formula: str,
    distr: str = "poisson",
    chains: int = 4,
    output: str | Path = "run",
    beta_prior_mean: float = 0.0,
    beta_prior_variance: float = 100.0,
    study_design: Any | None = None,
    random_levels: dict[str, Any] | None = None,
) -> CompiledModel:
    """Compile the fixed-effect Phase 2 target format.

    This does not yet claim sampler compatibility. It creates the stable
    Python-native artifact that the future sampler loader will consume.
    """
    Y_frame = _as_frame(Y, "Y")
    X_frame = _as_frame(X, "X")
    if len(Y_frame) != len(X_frame):
        raise ValueError("Y and X must have the same number of rows")
    if chains <= 0:
        raise ValueError("chains must be positive")

    formula = normalize_formula(formula)
    X_design = build_design_matrix(formula, X_frame)
    beta_init = np.zeros((chains, X_design.shape[1], Y_frame.shape[1]), dtype=float)
    arrays = {
        "Y": Y_frame.to_numpy(dtype=float),
        "X": X_design.to_numpy(dtype=float),
        "Beta_init": beta_init,
    }
    random_meta, random_arrays = _compile_random_levels(
        study_design=study_design,
        random_levels=random_levels,
        chains=chains,
        n_sites=Y_frame.shape[0],
        n_species=Y_frame.shape[1],
    )
    arrays.update(random_arrays)

    metadata = {
        "model_type": "hmsc",
        "format": "pyhmsc-json-hdf5",
        "distribution": distr,
        "formula": {"X": formula},
        "dimensions": {
            "n_sites": int(Y_frame.shape[0]),
            "n_species": int(Y_frame.shape[1]),
            "n_covariates": int(X_design.shape[1]),
            "n_chains": int(chains),
        },
        "names": {
            "sites": [str(value) for value in Y_frame.index],
            "species": [str(value) for value in Y_frame.columns],
            "covariates": [str(value) for value in X_design.columns],
        },
        "priors": {
            "Beta": {
                "mean": float(beta_prior_mean),
                "variance": float(beta_prior_variance),
            }
        },
        "capabilities": {
            "fixed_effects": True,
            "random_levels": bool(random_meta),
            "traits": False,
            "phylogeny": False,
            "spatial": False,
        },
        "random_levels": random_meta,
    }
    init_json = write_compiled_model(metadata, arrays, output)
    return CompiledModel(init_json=init_json, metadata=metadata)


def _as_frame(value: Any, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        frame = pd.DataFrame(value)
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    return frame


def _compile_random_levels(
    study_design: Any | None,
    random_levels: dict[str, Any] | None,
    chains: int,
    n_sites: int,
    n_species: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if not random_levels:
        return [], {}
    if study_design is None:
        raise ValueError("study_design is required when random_levels are provided")
    design = _as_frame(study_design, "study_design")
    if len(design) != n_sites:
        raise ValueError("study_design must have the same number of rows as Y")

    pi_columns = []
    meta = []
    arrays: dict[str, np.ndarray] = {}
    for idx, (name, spec) in enumerate(random_levels.items()):
        if spec.get("type", "iid") != "iid":
            raise NotImplementedError("Only iid random intercepts are currently supported")
        column = spec.get("column", name)
        if column not in design:
            raise ValueError(f"study_design is missing random level column {column!r}")
        codes, levels = pd.factorize(design[column], sort=True)
        pi_columns.append(codes.astype(int))
        n_levels = len(levels)
        nf = int(spec.get("nf", 1))
        prefix = f"RandomLevel_{idx}"
        arrays[f"{prefix}_Eta_init"] = np.zeros((chains, n_levels, nf), dtype=float)
        arrays[f"{prefix}_Lambda_init"] = np.zeros((chains, nf, n_species), dtype=float)
        arrays[f"{prefix}_Psi_init"] = np.ones((chains, nf, n_species), dtype=float)
        arrays[f"{prefix}_Delta_init"] = np.ones((chains, nf, 1), dtype=float)
        arrays[f"{prefix}_Alpha_init"] = np.ones((chains, nf), dtype=int)
        meta.append(
            {
                "name": name,
                "column": column,
                "type": "iid",
                "n_levels": n_levels,
                "levels": [str(level) for level in levels],
                "nf": nf,
                "array_prefix": prefix,
                "nu": float(spec.get("nu", 3.0)),
                "a1": float(spec.get("a1", 2.0)),
                "b1": float(spec.get("b1", 1.0)),
                "a2": float(spec.get("a2", 3.0)),
                "b2": float(spec.get("b2", 1.0)),
                "nfMin": int(spec.get("nfMin", nf)),
                "nfMax": int(spec.get("nfMax", max(nf, 4))),
            }
        )
    arrays["Pi"] = np.column_stack(pi_columns).astype(int)
    return meta, arrays
