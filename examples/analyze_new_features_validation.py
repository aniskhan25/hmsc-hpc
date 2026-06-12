"""Analyze iid random-slope and GPP spatial validation runs."""

from __future__ import annotations

import argparse
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
        default="examples/projects/simulated_new_features_validation",
        help="new-feature validation project directory",
    )
    parser.add_argument("--random-fixed-posterior", required=True)
    parser.add_argument("--random-slope-posterior", required=True)
    parser.add_argument("--spatial-full-posterior", required=True)
    parser.add_argument("--spatial-gpp-posterior", required=True)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--ppc-seed", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = build_report(
        project=Path(args.project),
        posteriors={
            "random_fixed": Path(args.random_fixed_posterior),
            "random_slope": Path(args.random_slope_posterior),
            "spatial_full": Path(args.spatial_full_posterior),
            "spatial_gpp": Path(args.spatial_gpp_posterior),
        },
        level=args.level,
        ppc_seed=args.ppc_seed,
    )
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        print(report)


def build_report(
    project: Path,
    posteriors: dict[str, Path],
    level: float = 0.95,
    ppc_seed: int = 1,
) -> str:
    random_metrics = build_random_slope_metrics(project / "random_slope", posteriors, level, ppc_seed)
    spatial_metrics = build_spatial_gpp_metrics(project / "spatial_gpp", posteriors, level, ppc_seed)
    lines = [
        "# Simulated New-Feature Validation Report",
        "",
        f"project: {project}",
        "",
        "## Random Slope",
        "",
        random_metrics.to_string(index=False),
        "",
        "## Spatial GPP",
        "",
        spatial_metrics.to_string(index=False),
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_random_slope_metrics(
    project: Path,
    posteriors: dict[str, Path],
    level: float = 0.95,
    ppc_seed: int = 1,
) -> pd.DataFrame:
    data = _load_project_data(project)
    rows = []
    for model_name, config_name, posterior_key, random_effects in [
        ("fixed", "model_fixed.yaml", "random_fixed", "none"),
        ("random_slope", "model_random_slope.yaml", "random_slope", "known"),
    ]:
        fit, model = _load_fit(project, config_name, posteriors[posterior_key])
        X = _prediction_data(data["X"], model, random_effects=random_effects)
        rows.append(
            {
                "model": model_name,
                "random_effects": random_effects,
                "beta_sign_recovered": _beta_sign_recovered(fit, data["truth_beta"]),
                **_ppc_metrics(fit, X, data["Y"], level, ppc_seed, random_effects),
                "eta_truth_corr": _eta_truth_correlation(fit, data["truth_eta"]),
                "lambda_slope_truth_corr": _lambda_truth_correlation(fit, data["truth_lambda"], x_index=1),
            }
        )
    return pd.DataFrame(rows)


def build_spatial_gpp_metrics(
    project: Path,
    posteriors: dict[str, Path],
    level: float = 0.95,
    ppc_seed: int = 1,
) -> pd.DataFrame:
    data = _load_project_data(project)
    rows = []
    for model_name, config_name, posterior_key in [
        ("spatial_full", "model_spatial_full.yaml", "spatial_full"),
        ("spatial_gpp", "model_spatial_gpp.yaml", "spatial_gpp"),
    ]:
        fit, model = _load_fit(project, config_name, posteriors[posterior_key])
        X = _prediction_data(data["X"], model, random_effects="known")
        rows.append(
            {
                "model": model_name,
                "random_effects": "known",
                "beta_sign_recovered": _beta_sign_recovered(fit, data["truth_beta"]),
                **_ppc_metrics(fit, X, data["Y"], level, ppc_seed, "known"),
                "eta_truth_corr": _eta_truth_correlation(fit, data["truth_eta"]),
                "lambda_truth_corr": _lambda_truth_correlation(fit, data["truth_lambda"]),
            }
        )
    return pd.DataFrame(rows)


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


def _load_fit(project: Path, config_name: str, posterior: Path) -> tuple[HmscFit, object]:
    model, _config = model_from_config(project / config_name)
    return HmscFit.from_file(posterior, model=model), model


def _prediction_data(X: pd.DataFrame, model: object, random_effects: str) -> pd.DataFrame:
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
        for name in ["slope_env"]:
            if name not in output and name in study_design:
                output[name] = study_design[name].to_numpy()
    return output


def _ppc_metrics(
    fit: HmscFit,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    level: float,
    ppc_seed: int,
    random_effects: str,
) -> dict[str, object]:
    species_ppc = fit.ppc_summary(Y, X, level=level, random_effects=random_effects, rng_seed=ppc_seed)
    richness_ppc = fit.richness_ppc_summary(Y, X, level=level, random_effects=random_effects, rng_seed=ppc_seed)
    return {
        "species_covered": f"{int(species_ppc['covered'].sum())} / {len(species_ppc)}",
        "site_richness_covered": f"{int(richness_ppc['covered'].sum())} / {len(richness_ppc)}",
        "species_mae": float((species_ppc["observed_mean"] - species_ppc["replicated_mean"]).abs().mean()),
        "site_richness_mae": float(
            (richness_ppc["observed_richness"] - richness_ppc["replicated_richness"]).abs().mean()
        ),
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


def _eta_truth_correlation(fit: HmscFit, truth_eta: pd.DataFrame) -> float | str:
    try:
        eta = fit.eta_mean(level=0)
    except ValueError:
        return "n/a"
    common = [name for name in truth_eta.index if name in eta.index]
    if len(common) < 2:
        return "n/a"
    return _safe_abs_corr(eta.loc[common].iloc[:, 0].to_numpy(dtype=float), truth_eta.loc[common].iloc[:, 0].to_numpy(dtype=float))


def _lambda_truth_correlation(fit: HmscFit, truth_lambda: pd.DataFrame, x_index: int | None = None) -> float | str:
    try:
        samples = fit.lambda_samples(level=0)
    except ValueError:
        return "n/a"
    if samples.ndim == 5:
        if x_index is None:
            x_index = 0
        values = samples[..., x_index].mean(axis=(0, 1))[0]
        truth_row = truth_lambda.index[min(x_index, len(truth_lambda.index) - 1)]
    else:
        values = samples.mean(axis=(0, 1))[0]
        truth_row = truth_lambda.index[0]
    species = [name for name in truth_lambda.columns if name in fit._species_names(len(values))]
    if len(species) < 2:
        return "n/a"
    species_names = fit._species_names(len(values))
    positions = [species_names.index(name) for name in species]
    return _safe_abs_corr(values[positions], truth_lambda.loc[truth_row, species].to_numpy(dtype=float))


def _safe_abs_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(abs(np.corrcoef(left, right)[0, 1]))


if __name__ == "__main__":
    main()

