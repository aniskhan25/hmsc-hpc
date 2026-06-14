"""Analyze full/GPP/NNGP spatial random-slope validation runs."""

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
        default="examples/projects/simulated_spatial_random_slope_validation",
        help="simulated spatial random-slope validation project directory",
    )
    parser.add_argument("--spatial-full-posterior", required=True)
    parser.add_argument("--spatial-gpp-posterior", required=True)
    parser.add_argument("--spatial-nngp-posterior", required=True)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--ppc-seed", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = build_report(
        project=Path(args.project),
        posteriors={
            "spatial_full": Path(args.spatial_full_posterior),
            "spatial_gpp": Path(args.spatial_gpp_posterior),
            "spatial_nngp": Path(args.spatial_nngp_posterior),
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
    metrics = build_metrics_table(project, posteriors, level, ppc_seed)
    lines = [
        "# Simulated Spatial Random-Slope Validation Report",
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
    for model_name, config_name, posterior_key in [
        ("spatial_full", "model_spatial_full.yaml", "spatial_full"),
        ("spatial_gpp", "model_spatial_gpp.yaml", "spatial_gpp"),
        ("spatial_nngp", "model_spatial_nngp.yaml", "spatial_nngp"),
    ]:
        fit, model = _load_fit(project, config_name, posteriors[posterior_key])
        X = _prediction_data(data["X"], model)
        rows.append(
            {
                "model": model_name,
                "beta_sign_recovered": _beta_sign_recovered(fit, data["truth_beta"]),
                **_ppc_metrics(fit, X, data["Y"], level, ppc_seed),
                "eta_truth_corr": _eta_truth_correlation(fit, data["truth_eta"]),
                "lambda_intercept_truth_corr": _lambda_truth_correlation(
                    fit, data["truth_lambda"], x_index=0
                ),
                "lambda_slope_truth_corr": _lambda_truth_correlation(
                    fit, data["truth_lambda"], x_index=1
                ),
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
        if "slope_env" not in output and "slope_env" in study_design:
            output["slope_env"] = study_design["slope_env"].to_numpy()
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


def _eta_truth_correlation(fit: HmscFit, truth_eta: pd.DataFrame) -> float | str:
    try:
        eta = fit.eta_mean(level=0)
    except ValueError:
        return "n/a"
    common = [name for name in truth_eta.index if name in eta.index]
    if len(common) < 2:
        return "n/a"
    return _safe_abs_corr(eta.loc[common].iloc[:, 0].to_numpy(dtype=float), truth_eta.loc[common].iloc[:, 0].to_numpy(dtype=float))


def _lambda_truth_correlation(fit: HmscFit, truth_lambda: pd.DataFrame, x_index: int) -> float | str:
    try:
        samples = fit.lambda_samples(level=0)
    except ValueError:
        return "n/a"
    if samples.ndim != 5:
        return "n/a"
    values = samples[..., x_index].mean(axis=(0, 1))[0]
    truth_row = truth_lambda.index[x_index]
    species_names = fit._species_names(len(values))
    species = [name for name in truth_lambda.columns if name in species_names]
    if len(species) < 2:
        return "n/a"
    positions = [species_names.index(name) for name in species]
    return _safe_abs_corr(values[positions], truth_lambda.loc[truth_row, species].to_numpy(dtype=float))


def _safe_abs_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(abs(np.corrcoef(left, right)[0, 1]))


if __name__ == "__main__":
    main()
