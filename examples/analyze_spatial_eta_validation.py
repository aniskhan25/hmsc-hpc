"""Analyze full/GPP/NNGP spatial Eta validation runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyhmsc.config import load_model_config, model_from_config
from pyhmsc.posterior import HmscFit


DEFAULT_MODELS = [
    "spatial_full",
    "spatial_gpp",
    "spatial_nngp_5",
    "spatial_nngp_10",
    "spatial_nngp_20",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default="examples/projects/simulated_spatial_eta_validation",
        help="simulated spatial Eta validation project directory",
    )
    parser.add_argument(
        "--posterior",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="posterior path keyed by model name; may be repeated",
    )
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--ppc-seed", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    posteriors = _parse_posterior_args(args.posterior)
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
    posteriors: dict[str, Path],
    level: float = 0.95,
    ppc_seed: int = 1,
) -> str:
    metrics = build_metrics_table(project, posteriors, level, ppc_seed)
    lines = [
        "# Simulated Spatial Eta Validation Report",
        "",
        f"project: {project}",
        "",
        metrics.to_string(index=False),
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_metrics_table(
    project: Path,
    posteriors: dict[str, Path],
    level: float = 0.95,
    ppc_seed: int = 1,
) -> pd.DataFrame:
    data = _load_project_data(project)
    rows = []
    for model_name in DEFAULT_MODELS:
        if model_name not in posteriors:
            continue
        config_name = _config_name(model_name)
        fit, model, config = _load_fit(project, config_name, posteriors[model_name])
        X = _prediction_data(data["X"], model)
        rows.append(
            {
                "model": model_name,
                **_model_settings(config),
                "beta_sign_recovered": _beta_sign_recovered(fit, data["truth_beta"]),
                **_ppc_metrics(fit, X, data["Y"], level, ppc_seed),
                "eta_raw_truth_corr": _eta_truth_correlation(fit, data["truth_eta"], align=False),
                "eta_aligned_truth_corr": _eta_truth_correlation(fit, data["truth_eta"], align=True),
                "lambda_truth_corr": _lambda_truth_correlation(fit, data["truth_lambda"]),
                "eta_raw_rmse_scaled": _eta_scaled_rmse(fit, data["truth_eta"], align=False),
                "eta_aligned_rmse_scaled": _eta_scaled_rmse(fit, data["truth_eta"], align=True),
            }
        )
    return pd.DataFrame(rows)


def _parse_posterior_args(values: list[str]) -> dict[str, Path]:
    posteriors: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--posterior must be MODEL=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"--posterior must be MODEL=PATH, got {value!r}")
        posteriors[name] = Path(path)
    return posteriors


def _config_name(model_name: str) -> str:
    return f"model_{model_name}.yaml"


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


def _load_fit(project: Path, config_name: str, posterior: Path) -> tuple[HmscFit, object, dict[str, object]]:
    config_path = project / config_name
    model, config = model_from_config(config_path)
    return HmscFit.from_file(posterior, model=model), model, config


def _model_settings(config: dict[str, object]) -> dict[str, object]:
    random_levels = config.get("random_levels", {})
    if not isinstance(random_levels, dict) or not random_levels:
        return {"random_level_type": "none", "n_knots": "n/a", "n_neighbors": "n/a"}
    spec = next(iter(random_levels.values()))
    if not isinstance(spec, dict):
        return {"random_level_type": "unknown", "n_knots": "n/a", "n_neighbors": "n/a"}
    return {
        "random_level_type": spec.get("type", "iid"),
        "n_knots": spec.get("n_knots", "n/a"),
        "n_neighbors": spec.get("n_neighbors", "n/a"),
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


def _ppc_metrics(
    fit: HmscFit,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    level: float,
    ppc_seed: int,
) -> dict[str, object]:
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


def _eta_truth_correlation(fit: HmscFit, truth_eta: pd.DataFrame, align: bool = False) -> float | str:
    try:
        eta = fit.eta_mean(level=0, align=align)
    except ValueError:
        return "n/a"
    common = [name for name in truth_eta.index if name in eta.index]
    if len(common) < 2:
        return "n/a"
    return _safe_abs_corr(
        eta.loc[common].iloc[:, 0].to_numpy(dtype=float),
        truth_eta.loc[common].iloc[:, 0].to_numpy(dtype=float),
    )


def _eta_scaled_rmse(fit: HmscFit, truth_eta: pd.DataFrame, align: bool = False) -> float | str:
    try:
        eta = fit.eta_mean(level=0, align=align)
    except ValueError:
        return "n/a"
    common = [name for name in truth_eta.index if name in eta.index]
    if len(common) < 2:
        return "n/a"
    estimated = _standardize(eta.loc[common].iloc[:, 0].to_numpy(dtype=float))
    truth = _standardize(truth_eta.loc[common].iloc[:, 0].to_numpy(dtype=float))
    if np.corrcoef(estimated, truth)[0, 1] < 0:
        estimated = -estimated
    return float(np.sqrt(np.mean((estimated - truth) ** 2)))


def _lambda_truth_correlation(fit: HmscFit, truth_lambda: pd.DataFrame) -> float | str:
    try:
        samples = fit.lambda_samples(level=0)
    except ValueError:
        return "n/a"
    if samples.ndim != 4:
        return "n/a"
    values = samples.mean(axis=(0, 1))[0]
    species_names = fit._species_names(len(values))
    species = [name for name in truth_lambda.columns if name in species_names]
    if len(species) < 2:
        return "n/a"
    positions = [species_names.index(name) for name in species]
    return _safe_abs_corr(values[positions], truth_lambda.loc[truth_lambda.index[0], species].to_numpy(dtype=float))


def _safe_abs_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(abs(np.corrcoef(left, right)[0, 1]))


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(values.std(ddof=1), np.finfo(float).eps)


if __name__ == "__main__":
    main()
