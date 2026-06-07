"""Compare fixed, iid, and spatial fits for the simulated spatial project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyhmsc.config import model_from_config
from pyhmsc.posterior import HmscFit


MODEL_CONFIGS = {
    "fixed": "model_fixed.yaml",
    "iid": "model_iid.yaml",
    "spatial": "model_spatial_full.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default="examples/projects/simulated_spatial_validation",
        help="simulated spatial validation project directory",
    )
    parser.add_argument("--fixed-posterior", help="posterior .h5 for model_fixed.yaml")
    parser.add_argument("--iid-posterior", help="posterior .h5 for model_iid.yaml")
    parser.add_argument("--spatial-posterior", help="posterior .h5 for model_spatial_full.yaml")
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--ppc-seed", type=int, default=1)
    parser.add_argument("--output", help="optional report path")
    args = parser.parse_args()

    posteriors = {
        "fixed": Path(args.fixed_posterior) if args.fixed_posterior else None,
        "iid": Path(args.iid_posterior) if args.iid_posterior else None,
        "spatial": Path(args.spatial_posterior) if args.spatial_posterior else None,
    }
    report = build_report(
        project=Path(args.project),
        posteriors=posteriors,
        level=args.level,
        ppc_seed=args.ppc_seed,
    )
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        print(report)


def build_report(
    project: Path,
    posteriors: dict[str, Path | None],
    level: float = 0.95,
    ppc_seed: int = 1,
) -> str:
    summary = build_metrics_table(
        project=project,
        posteriors=posteriors,
        level=level,
        ppc_seed=ppc_seed,
    )
    detail_sections = _build_detail_sections(project, posteriors, level=level)
    lines = [
        "# Simulated Spatial Validation Report",
        "",
        f"project: {project}",
        "",
        "## Model Comparison",
        "",
        summary.to_string(index=False),
        "",
        "## Details",
        "",
        *detail_sections,
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_metrics_table(
    project: Path,
    posteriors: dict[str, Path | None],
    level: float = 0.95,
    ppc_seed: int = 1,
) -> pd.DataFrame:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    data = _load_project_data(project)
    rows = []
    for model_name in ["fixed", "iid", "spatial"]:
        posterior = posteriors.get(model_name)
        if posterior is None:
            continue
        fit, model = _load_fit(project, model_name, posterior)
        random_effects = "known" if model_name in {"iid", "spatial"} else "none"
        x_pred = _prediction_data_for_random_effects(data["X"], model, random_effects)
        metrics = _model_metrics(
            model_name=model_name,
            fit=fit,
            X=x_pred,
            Y=data["Y"],
            study_design=data["study_design"],
            truth_beta=data["truth_beta"],
            truth_site_effect=data["truth_site_effect"],
            truth_lambda=data["truth_lambda"],
            level=level,
            ppc_seed=ppc_seed,
            random_effects=random_effects,
        )
        rows.append(metrics)
    if not rows:
        raise ValueError("At least one posterior path is required")
    return pd.DataFrame(rows)


def _build_detail_sections(
    project: Path,
    posteriors: dict[str, Path | None],
    level: float,
) -> list[str]:
    detail_sections = []
    for model_name in ["fixed", "iid", "spatial"]:
        posterior = posteriors.get(model_name)
        if posterior is None:
            continue
        fit, _model = _load_fit(project, model_name, posterior)
        detail_sections.append(_detail_section(model_name, fit, level=level))
    return detail_sections


def _load_project_data(project: Path) -> dict[str, pd.DataFrame]:
    data_dir = project / "data"
    return {
        "Y": pd.read_csv(data_dir / "Y.csv", index_col=0),
        "X": pd.read_csv(data_dir / "X.csv", index_col=0),
        "study_design": pd.read_csv(data_dir / "study_design.csv", index_col=0),
        "truth_beta": pd.read_csv(data_dir / "truth_beta.csv", index_col=0),
        "truth_site_effect": pd.read_csv(data_dir / "truth_site_effect.csv", index_col=0),
        "truth_lambda": pd.read_csv(data_dir / "truth_lambda.csv", index_col=0),
    }


def _load_fit(project: Path, model_name: str, posterior: Path) -> tuple[HmscFit, object]:
    model, _config = model_from_config(project / MODEL_CONFIGS[model_name])
    return HmscFit.from_file(posterior, model=model), model


def _model_metrics(
    model_name: str,
    fit: HmscFit,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    study_design: pd.DataFrame,
    truth_beta: pd.DataFrame,
    truth_site_effect: pd.DataFrame,
    truth_lambda: pd.DataFrame,
    level: float,
    ppc_seed: int,
    random_effects: str,
) -> dict[str, object]:
    species_ppc = fit.ppc_summary(
        Y,
        X,
        level=level,
        random_effects=random_effects,
        rng_seed=ppc_seed,
    )
    richness_ppc = fit.richness_ppc_summary(
        Y,
        X,
        level=level,
        random_effects=random_effects,
        rng_seed=ppc_seed,
    )
    prediction = fit.predict_mean(X, random_effects=random_effects)
    richness_residual = Y.sum(axis=1) - prediction.sum(axis=1)
    return {
        "model": model_name,
        "random_effects": random_effects,
        "beta_sign_recovered": _beta_sign_recovered(fit, truth_beta),
        "species_covered": f"{int(species_ppc['covered'].sum())} / {len(species_ppc)}",
        "site_richness_covered": f"{int(richness_ppc['covered'].sum())} / {len(richness_ppc)}",
        "species_mae": float((species_ppc["observed_mean"] - species_ppc["replicated_mean"]).abs().mean()),
        "site_richness_mae": float(
            (richness_ppc["observed_richness"] - richness_ppc["replicated_richness"]).abs().mean()
        ),
        "neighbor_residual_corr": _neighbor_correlation(study_design, richness_residual),
        "eta_truth_corr": _eta_truth_correlation(fit, truth_site_effect, truth_lambda),
    }


def _beta_sign_recovered(fit: HmscFit, truth_beta: pd.DataFrame) -> str:
    beta = fit.beta_mean()
    covariate = "env"
    species = [name for name in truth_beta.columns if name in beta.columns]
    recovered = 0
    checked = 0
    for name in species:
        expected = np.sign(truth_beta.loc[covariate, name])
        if expected == 0:
            continue
        checked += 1
        recovered += int(np.sign(beta.loc[covariate, name]) == expected)
    return f"{recovered} / {checked}"


def _neighbor_correlation(study_design: pd.DataFrame, residual: pd.Series) -> float:
    coords = study_design.loc[residual.index, ["xcoord", "ycoord"]].to_numpy(dtype=float)
    distances = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    values = residual.to_numpy(dtype=float)
    return _safe_corr(values, values[nearest])


def _eta_truth_correlation(
    fit: HmscFit,
    truth_site_effect: pd.DataFrame,
    truth_lambda: pd.DataFrame,
) -> float | str:
    try:
        eta = fit.eta_mean(level=0)
        loadings = fit.lambda_mean(level=0)
    except ValueError:
        return "n/a"
    if eta.empty or loadings.empty:
        return "n/a"
    factor = eta.columns[0]
    common_species = [name for name in truth_lambda.columns if name in loadings.columns]
    orientation = 1.0
    if common_species:
        dot = float((loadings.loc[factor, common_species] * truth_lambda.loc["factor_0", common_species]).sum())
        if dot < 0:
            orientation = -1.0
    common_units = [name for name in truth_site_effect.index if name in eta.index]
    if len(common_units) < 2:
        return "n/a"
    return _safe_corr(
        eta.loc[common_units, factor].to_numpy(dtype=float) * orientation,
        truth_site_effect.loc[common_units, "eta"].to_numpy(dtype=float),
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _prediction_data_for_random_effects(
    X: pd.DataFrame,
    model: object,
    random_effects: str,
) -> pd.DataFrame:
    if random_effects != "known":
        return X
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


def _detail_section(model_name: str, fit: HmscFit, level: float) -> str:
    parts = [
        f"### {model_name}",
        "",
        "Beta:",
        fit.beta_summary(level=level).to_string(index=False),
    ]
    try:
        parts.extend(["", "Eta:", fit.eta_summary(level=0, cred_level=level).head(10).to_string(index=False)])
        parts.extend(["", "Lambda:", fit.lambda_summary(level=0, cred_level=level).to_string(index=False)])
    except ValueError:
        pass
    return "\n".join(parts)


if __name__ == "__main__":
    main()
