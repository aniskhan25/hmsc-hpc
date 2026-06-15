"""Analyze a multi-factor NNGP spatial Eta validation run."""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyhmsc.config import model_from_config
from pyhmsc.posterior import HmscFit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default="examples/projects/simulated_spatial_multifactor_eta_validation",
        help="multi-factor spatial Eta validation project directory",
    )
    parser.add_argument("--posterior", required=True)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--ppc-seed", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = build_report(
        project=Path(args.project),
        posterior=Path(args.posterior),
        level=args.level,
        ppc_seed=args.ppc_seed,
    )
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        print(report)


def build_report(project: Path, posterior: Path, level: float = 0.95, ppc_seed: int = 1) -> str:
    metrics = build_metrics(project, posterior, level=level, ppc_seed=ppc_seed)
    rows = pd.DataFrame([metrics])
    lines = [
        "# Simulated Spatial Multi-Factor Eta Validation Report",
        "",
        f"project: {project}",
        "",
        rows.to_string(index=False),
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_metrics(project: Path, posterior: Path, level: float = 0.95, ppc_seed: int = 1) -> dict[str, object]:
    data = _load_project_data(project)
    model, _config = model_from_config(project / "model_spatial_nngp.yaml")
    fit = HmscFit.from_file(posterior, model=model)
    X = _prediction_data(data["X"], model)
    ppc = _ppc_metrics(fit, X, data["Y"], level=level, ppc_seed=ppc_seed)
    raw = _latent_recovery(fit, data["truth_eta"], data["truth_lambda"], align=False)
    aligned = _latent_recovery(fit, data["truth_eta"], data["truth_lambda"], align=True)
    return {
        "model": "spatial_nngp",
        "n_factors": int(data["truth_eta"].shape[1]),
        "n_neighbors": int(model.random_levels["plot"]["n_neighbors"]),
        "beta_sign_recovered": _beta_sign_recovered(fit, data["truth_beta"]),
        **ppc,
        "eta_raw_mean_corr": raw["eta_mean_corr"],
        "eta_aligned_mean_corr": aligned["eta_mean_corr"],
        "lambda_raw_mean_corr": raw["lambda_mean_corr"],
        "lambda_aligned_mean_corr": aligned["lambda_mean_corr"],
        "association_truth_corr": _association_truth_correlation(fit, data["truth_lambda"]),
    }


def _load_project_data(project: Path) -> dict[str, pd.DataFrame]:
    data_dir = project / "data"
    return {
        "Y": pd.read_csv(data_dir / "Y.csv", index_col=0),
        "X": pd.read_csv(data_dir / "X.csv", index_col=0),
        "study_design": pd.read_csv(data_dir / "study_design.csv", index_col=0),
        "truth_beta": pd.read_csv(data_dir / "truth_beta.csv", index_col=0),
        "truth_eta": pd.read_csv(data_dir / "truth_eta.csv", index_col=0),
        "truth_lambda": pd.read_csv(data_dir / "truth_lambda.csv", index_col=0),
    }


def _prediction_data(X: pd.DataFrame, model: object) -> pd.DataFrame:
    output = X.copy()
    study_design = model.study_design
    for level_name, spec in model.random_levels.items():
        column = spec.get("column", level_name)
        if column not in output:
            output[column] = study_design[column].to_numpy()
        for coord in spec.get("coords", []):
            if coord not in output and coord in study_design:
                output[coord] = study_design[coord].to_numpy()
    return output


def _ppc_metrics(fit: HmscFit, X: pd.DataFrame, Y: pd.DataFrame, level: float, ppc_seed: int) -> dict[str, object]:
    species_ppc = fit.ppc_summary(Y, X, level=level, random_effects="known", rng_seed=ppc_seed)
    richness_ppc = fit.richness_ppc_summary(Y, X, level=level, random_effects="known", rng_seed=ppc_seed)
    return {
        "species_covered": f"{int(species_ppc['covered'].sum())} / {len(species_ppc)}",
        "site_richness_covered": f"{int(richness_ppc['covered'].sum())} / {len(richness_ppc)}",
        "species_mae": float((species_ppc["observed_mean"] - species_ppc["replicated_mean"]).abs().mean()),
        "site_richness_mae": float(
            (richness_ppc["observed_richness"] - richness_ppc["replicated_richness"]).abs().mean()
        ),
    }


def _latent_recovery(
    fit: HmscFit,
    truth_eta: pd.DataFrame,
    truth_lambda: pd.DataFrame,
    align: bool,
) -> dict[str, float]:
    eta = fit.eta_mean(level=0, align=align)
    lam = fit.lambda_mean(level=0, align=align)
    units = [name for name in truth_eta.index if name in eta.index]
    species = [name for name in truth_lambda.columns if name in lam.columns]
    estimated_lambda = lam.loc[:, species].to_numpy(dtype=float)
    truth_loadings = truth_lambda.loc[:, species].to_numpy(dtype=float)
    order, signs = _match_estimated_factors_to_truth(estimated_lambda, truth_loadings)
    estimated_lambda = estimated_lambda[order] * signs[:, None]
    estimated_eta = eta.loc[units].to_numpy(dtype=float)[:, order] * signs[None, :]
    truth_eta_values = truth_eta.loc[units].to_numpy(dtype=float)
    eta_corr = [_safe_abs_corr(estimated_eta[:, idx], truth_eta_values[:, idx]) for idx in range(truth_eta_values.shape[1])]
    lambda_corr = [
        _safe_abs_corr(estimated_lambda[idx], truth_loadings[idx])
        for idx in range(truth_loadings.shape[0])
    ]
    return {
        "eta_mean_corr": float(np.nanmean(eta_corr)),
        "lambda_mean_corr": float(np.nanmean(lambda_corr)),
    }


def _association_truth_correlation(fit: HmscFit, truth_lambda: pd.DataFrame) -> float:
    estimated = fit.species_associations(level=0, correlation=False)
    truth = truth_lambda.to_numpy(dtype=float).T @ truth_lambda.to_numpy(dtype=float)
    species = [name for name in truth_lambda.columns if name in estimated.index]
    positions = [list(truth_lambda.columns).index(name) for name in species]
    truth = truth[np.ix_(positions, positions)]
    estimated_values = estimated.loc[species, species].to_numpy(dtype=float)
    upper = np.triu_indices(len(species), k=1)
    return _safe_abs_corr(estimated_values[upper], truth[upper])


def _match_estimated_factors_to_truth(loadings: np.ndarray, reference: np.ndarray) -> tuple[list[int], np.ndarray]:
    n_truth = reference.shape[0]
    n_estimated = loadings.shape[0]
    if n_estimated < n_truth:
        raise ValueError(f"Estimated {n_estimated} factors but truth has {n_truth}")
    scores = _factor_alignment_scores(loadings, reference)
    if n_estimated <= 8:
        best_order = max(
            permutations(range(n_estimated), n_truth),
            key=lambda order: sum(scores[truth_idx, order[truth_idx]] for truth_idx in range(n_truth)),
        )
    else:
        remaining = set(range(n_estimated))
        order = []
        for truth_idx in range(n_truth):
            chosen = max(remaining, key=lambda factor_idx: scores[truth_idx, factor_idx])
            order.append(chosen)
            remaining.remove(chosen)
        best_order = tuple(order)
    signs = []
    for truth_idx, factor_idx in enumerate(best_order):
        dot = float(np.dot(reference[truth_idx], loadings[factor_idx]))
        signs.append(1.0 if dot >= 0 else -1.0)
    return list(best_order), np.asarray(signs, dtype=float)


def _factor_alignment_scores(loadings: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_norm = np.linalg.norm(reference, axis=1)
    load_norm = np.linalg.norm(loadings, axis=1)
    denom = np.maximum(ref_norm[:, None] * load_norm[None, :], np.finfo(float).eps)
    return np.abs(reference @ loadings.T) / denom


def _beta_sign_recovered(fit: HmscFit, truth_beta: pd.DataFrame) -> str:
    beta = fit.beta_mean()
    species = [name for name in truth_beta.columns if name in beta.columns]
    recovered = 0
    checked = 0
    for name in species:
        expected = np.sign(truth_beta.loc["env", name])
        if expected == 0:
            continue
        checked += 1
        recovered += int(np.sign(beta.loc["env", name]) == expected)
    return f"{recovered} / {checked}"


def _safe_abs_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(abs(np.corrcoef(left, right)[0, 1]))


if __name__ == "__main__":
    main()
