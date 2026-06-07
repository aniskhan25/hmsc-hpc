"""Compare fixed, iid, and spatial fits for the big spatial plant project."""

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
        default="examples/projects/big_spatial_plants_validation",
        help="big spatial plant validation project directory",
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
    metrics = build_metrics_table(
        project=project,
        posteriors=posteriors,
        level=level,
        ppc_seed=ppc_seed,
    )
    detail_sections = _build_detail_sections(project, posteriors, level=level)
    lines = [
        "# Big Spatial Plant Validation Report",
        "",
        f"project: {project}",
        "",
        "## Model Comparison",
        "",
        metrics.to_string(index=False),
        "",
        "## Interpretation Target",
        "",
        "- iid/spatial random effects should improve PPC errors over fixed effects",
        "- full spatial random effects should reduce nearest-neighbor residual correlation",
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
        rows.append(
            _model_metrics(
                model_name=model_name,
                fit=fit,
                X=x_pred,
                Y=data["Y"],
                study_design=data["study_design"],
                level=level,
                ppc_seed=ppc_seed,
                random_effects=random_effects,
            )
        )
    if not rows:
        raise ValueError("At least one posterior path is required")
    return pd.DataFrame(rows)


def _load_project_data(project: Path) -> dict[str, pd.DataFrame]:
    data_dir = project / "data"
    return {
        "Y": pd.read_csv(data_dir / "Y_presence.csv", index_col=0),
        "X": pd.read_csv(data_dir / "X.csv", index_col=0),
        "study_design": pd.read_csv(data_dir / "study_design.csv", index_col=0),
        "taxonomy": pd.read_csv(data_dir / "taxonomy.csv", index_col=0),
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
        "species_covered": f"{int(species_ppc['covered'].sum())} / {len(species_ppc)}",
        "site_richness_covered": f"{int(richness_ppc['covered'].sum())} / {len(richness_ppc)}",
        "species_mae": float((species_ppc["observed_mean"] - species_ppc["replicated_mean"]).abs().mean()),
        "site_richness_mae": float(
            (richness_ppc["observed_richness"] - richness_ppc["replicated_richness"]).abs().mean()
        ),
        "neighbor_residual_corr": _neighbor_correlation(study_design, richness_residual),
    }


def _neighbor_correlation(study_design: pd.DataFrame, residual: pd.Series) -> float:
    coords = study_design.loc[residual.index, ["xcoord", "ycoord"]].to_numpy(dtype=float)
    distances = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    values = residual.to_numpy(dtype=float)
    return _safe_corr(values, values[nearest])


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


def _detail_section(model_name: str, fit: HmscFit, level: float) -> str:
    parts = [
        f"### {model_name}",
        "",
        "Beta:",
        fit.beta_summary(level=level).to_string(index=False),
        "",
        "Beta diagnostics:",
        _diagnostics_report(fit, "Beta"),
    ]
    try:
        parts.extend(["", "Eta:", fit.eta_summary(level=0, cred_level=level).head(10).to_string(index=False)])
        parts.extend(["", "Lambda:", fit.lambda_summary(level=0, cred_level=level).head(20).to_string(index=False)])
    except ValueError:
        pass
    return "\n".join(parts)


def _diagnostics_report(fit: HmscFit, param: str) -> str:
    try:
        overview = fit.diagnostics_overview(param)
    except ValueError:
        return f"{param}: samples unavailable."
    return "\n".join(
        [
            f"{param}:",
            f"  parameters: {overview['n_parameters']}",
            f"  max R-hat: {overview['rhat_max']:.6g}",
            f"  median R-hat: {overview['rhat_median']:.6g}",
            f"  min ESS: {overview['ess_min']:.6g}",
            f"  median ESS: {overview['ess_median']:.6g}",
            f"  R-hat > {overview['rhat_threshold']}: {overview['n_rhat_flagged']}",
            f"  ESS < {overview['ess_threshold']}: {overview['n_ess_flagged']}",
        ]
    )


if __name__ == "__main__":
    main()
