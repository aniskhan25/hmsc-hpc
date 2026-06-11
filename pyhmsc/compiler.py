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
    traits: Any | None = None,
    trait_formula: str | None = None,
    phylo_cov: Any | None = None,
    phylo_tree: str | Path | None = None,
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
    T_design = _compile_traits(traits, trait_formula, Y_frame)
    C = _compile_phylo_cov(phylo_cov, phylo_tree, Y_frame)
    beta_init = np.zeros((chains, X_design.shape[1], Y_frame.shape[1]), dtype=float)
    arrays = {
        "Y": Y_frame.to_numpy(dtype=float),
        "X": X_design.to_numpy(dtype=float),
        "Beta_init": beta_init,
        "T": T_design.to_numpy(dtype=float),
    }
    if C is not None:
        arrays["C"] = C
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
            "n_traits": int(T_design.shape[1]),
            "n_chains": int(chains),
        },
        "names": {
            "sites": [str(value) for value in Y_frame.index],
            "species": [str(value) for value in Y_frame.columns],
            "covariates": [str(value) for value in X_design.columns],
            "traits": [str(value) for value in T_design.columns],
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
            "traits": traits is not None,
            "phylogeny": C is not None,
            "spatial": any(level.get("spatial", False) for level in random_meta),
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
        level_type = spec.get("type", "iid")
        if level_type not in {"iid", "spatial_full", "spatial_gpp", "gpp"}:
            raise NotImplementedError("Only iid, spatial_full, and spatial_gpp random levels are currently supported")
        if level_type == "gpp":
            level_type = "spatial_gpp"
        column = spec.get("column", name)
        if column not in design:
            raise ValueError(f"study_design is missing random level column {column!r}")
        codes, levels = pd.factorize(design[column], sort=True)
        pi_columns.append(codes.astype(int))
        n_levels = len(levels)
        nf = int(spec.get("nf", 1))
        prefix = f"RandomLevel_{idx}"
        x_design = _random_level_design(spec, design, codes)
        x_dim = x_design.shape[1]
        arrays[f"{prefix}_Eta_init"] = np.zeros((chains, n_levels, nf), dtype=float)
        lambda_shape = (chains, nf, n_species) if x_dim == 0 else (chains, nf, n_species, x_dim)
        arrays[f"{prefix}_Lambda_init"] = np.zeros(lambda_shape, dtype=float)
        psi_shape = (chains, nf, n_species) if x_dim == 0 else (chains, nf, n_species, x_dim)
        arrays[f"{prefix}_Psi_init"] = np.ones(psi_shape, dtype=float)
        arrays[f"{prefix}_Delta_init"] = np.ones((chains, nf, 1), dtype=float)
        arrays[f"{prefix}_Alpha_init"] = np.ones((chains, nf), dtype=int)
        if x_dim > 0:
            arrays[f"{prefix}_xMat"] = x_design
        level_meta = {
                "name": name,
                "column": column,
                "type": level_type,
                "n_levels": n_levels,
                "levels": [str(level) for level in levels],
                "nf": nf,
                "xDim": x_dim,
                "x_formula": spec.get("x_formula"),
                "array_prefix": prefix,
                "nu": float(spec.get("nu", 3.0)),
                "a1": float(spec.get("a1", 2.0)),
                "b1": float(spec.get("b1", 1.0)),
                "a2": float(spec.get("a2", 3.0)),
                "b2": float(spec.get("b2", 1.0)),
                "nfMin": int(spec.get("nfMin", nf)),
                "nfMax": int(spec.get("nfMax", max(nf, 4))),
                "spatial": level_type in {"spatial_full", "spatial_gpp"},
            }
        if level_type in {"spatial_full", "spatial_gpp"}:
            coord_cols = spec.get("coords", ["x", "y"])
            if len(coord_cols) != 2 or any(col not in design for col in coord_cols):
                raise ValueError(f"{level_type} random levels require coordinate columns via coords")
            coords = (
                design.assign(__code=codes)
                .groupby("__code", sort=True)[coord_cols]
                .mean()
                .to_numpy(dtype=float)
            )
            dist = _pairwise_distances(coords)
            scale = float(spec.get("alpha", np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0))
            level_meta["alphapw"] = [[scale, 1.0]]
            if level_type == "spatial_full":
                arrays[f"{prefix}_distMat"] = dist
            else:
                n_knots = int(spec.get("n_knots", spec.get("nKnots", min(max(2, int(np.sqrt(n_levels))), n_levels))))
                knots = _select_gpp_knots(coords, n_knots)
                arrays[f"{prefix}_distMat12"] = _cross_distances(coords, knots)
                arrays[f"{prefix}_distMat22"] = _pairwise_distances(knots)
                level_meta["nKnots"] = int(knots.shape[0])
        meta.append(level_meta)
    arrays["Pi"] = np.column_stack(pi_columns).astype(int)
    return meta, arrays


def _compile_traits(traits: Any | None, trait_formula: str | None, Y: pd.DataFrame) -> pd.DataFrame:
    if traits is None:
        return pd.DataFrame({"Intercept": np.ones(Y.shape[1])}, index=Y.columns)
    trait_frame = _as_frame(traits, "traits")
    missing = [species for species in Y.columns if species not in trait_frame.index]
    if missing:
        raise ValueError(f"traits missing species rows: {missing}")
    trait_frame = trait_frame.loc[Y.columns]
    return build_design_matrix(trait_formula or "~ .", trait_frame)


def _compile_phylo_cov(phylo_cov: Any | None, phylo_tree: str | Path | None, Y: pd.DataFrame) -> np.ndarray | None:
    if phylo_cov is None and phylo_tree is None:
        return None
    if phylo_tree is not None:
        return _phylo_cov_from_newick(phylo_tree, Y)
    cov = _as_frame(phylo_cov, "phylo_cov")
    missing = [species for species in Y.columns if species not in cov.index or species not in cov.columns]
    if missing:
        raise ValueError(f"phylo_cov missing species rows/columns: {missing}")
    cov = cov.loc[Y.columns, Y.columns].to_numpy(dtype=float)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("phylo_cov must be square")
    return cov


def _random_level_design(spec: dict[str, Any], design: pd.DataFrame, codes: np.ndarray) -> np.ndarray:
    formula = spec.get("x_formula")
    if not formula:
        return np.zeros((len(np.unique(codes)), 0), dtype=float)
    matrix = build_design_matrix(formula, design)
    matrix = matrix.groupby(codes, sort=True).mean()
    return matrix.to_numpy(dtype=float)


def _phylo_cov_from_newick(phylo_tree: str | Path, Y: pd.DataFrame) -> np.ndarray:
    try:
        from Bio import Phylo  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install biopython to use phylo_tree Newick input") from exc
    tree = Phylo.read(str(phylo_tree), "newick")
    terminals = {terminal.name: terminal for terminal in tree.get_terminals()}
    missing = [species for species in Y.columns if species not in terminals]
    if missing:
        raise ValueError(f"phylo_tree missing species: {missing}")
    depths = tree.depths()
    if all(depth == 0 for depth in depths.values()):
        depths = tree.depths(unit_branch_lengths=True)
    cov = np.zeros((len(Y.columns), len(Y.columns)), dtype=float)
    for i, left in enumerate(Y.columns):
        for j, right in enumerate(Y.columns):
            ancestor = tree.common_ancestor(terminals[left], terminals[right])
            cov[i, j] = depths[ancestor]
    diag = np.diag(cov)
    if np.any(diag == 0):
        cov = cov + np.eye(cov.shape[0])
    return cov


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _cross_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _select_gpp_knots(coords: np.ndarray, n_knots: int) -> np.ndarray:
    if n_knots <= 0:
        raise ValueError("spatial_gpp n_knots must be positive")
    if n_knots >= coords.shape[0]:
        return coords.copy()
    center = coords.mean(axis=0)
    first = int(np.argmin(np.sum((coords - center) ** 2, axis=1)))
    selected = [first]
    min_dist = np.sum((coords - coords[first]) ** 2, axis=1)
    while len(selected) < n_knots:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        min_dist = np.minimum(min_dist, np.sum((coords - coords[next_idx]) ** 2, axis=1))
    return coords[selected]
