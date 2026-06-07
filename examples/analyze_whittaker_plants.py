"""Summarize the Whittaker plant HMSC book validation run."""

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
    parser.add_argument("--posterior", required=True, help="posterior .h5 from the Whittaker plant run")
    parser.add_argument(
        "--project",
        default="examples/projects/whittaker_plants_hmsc_book",
        help="project directory containing data/X.csv and data/traits.csv",
    )
    parser.add_argument("--gradient-points", type=int, default=25)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--ppc-seed", type=int, default=1)
    parser.add_argument(
        "--model-config",
        help="optional model config for prediction metadata, required for known random-effect PPCs",
    )
    parser.add_argument("--random-effects", choices=["none", "known", "marginal"], default="none")
    parser.add_argument("--unseen-groups", choices=["error", "zero", "sample", "nearest"], default="error")
    parser.add_argument("--output", help="optional path for the text report")
    parser.add_argument("--ppc-output", help="optional path for the posterior predictive section")
    args = parser.parse_args()

    report = build_report(
        posterior=Path(args.posterior),
        project=Path(args.project),
        gradient_points=args.gradient_points,
        level=args.level,
        ppc_seed=args.ppc_seed,
        model_config=Path(args.model_config) if args.model_config else None,
        random_effects=args.random_effects,
        unseen_groups=args.unseen_groups,
    )
    if args.ppc_output:
        Path(args.ppc_output).write_text(
            build_ppc_report(
                posterior=Path(args.posterior),
                project=Path(args.project),
                level=args.level,
                ppc_seed=args.ppc_seed,
                model_config=Path(args.model_config) if args.model_config else None,
                random_effects=args.random_effects,
                unseen_groups=args.unseen_groups,
            ),
            encoding="utf-8",
        )
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        print(report)


def build_report(
    posterior: Path,
    project: Path,
    gradient_points: int = 25,
    level: float = 0.95,
    ppc_seed: int = 1,
    model_config: Path | None = None,
    random_effects: str = "none",
    unseen_groups: str = "error",
) -> str:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    model = _load_model(model_config)
    fit = HmscFit.from_file(posterior, model=model)
    x_data = pd.read_csv(project / "data" / "X.csv", index_col=0)
    y_data = pd.read_csv(project / "data" / "Y_presence.csv", index_col=0)
    traits = pd.read_csv(project / "data" / "traits.csv", index_col=0)
    ppc_x_data = _prediction_data_for_random_effects(x_data, model, random_effects)

    beta = fit.beta_mean()
    beta_ci = fit.beta_ci(level=level)
    tmg = pd.DataFrame(
        {
            "mean": beta.loc["TMG"],
            "lower": beta_ci["lower"].loc["TMG"],
            "upper": beta_ci["upper"].loc["TMG"],
        }
    )

    gradient = fit.gradient("TMG", X_reference=x_data, n=gradient_points)
    prediction = fit.predict_samples(gradient, response=True)
    richness = prediction.sum(axis=-1)
    cn = traits.loc[beta.columns, "CN"].to_numpy(dtype=float)
    weighted_cn = (prediction @ cn) / np.maximum(prediction.sum(axis=-1), 1e-12)
    richness_gradient = _endpoint_summary(richness, level)
    cn_gradient = _endpoint_summary(weighted_cn, level)

    lines = [
        "# Whittaker Plant Validation Report",
        "",
        f"posterior: {posterior}",
        f"project: {project}",
        f"distribution: {fit._distribution()}",
        "",
        "## TMG Species Effects",
        "",
        f"negative mean effects: {int((tmg['mean'] < 0).sum())} / {len(tmg)}",
        f"positive mean effects: {int((tmg['mean'] > 0).sum())} / {len(tmg)}",
        f"{int(level * 100)}% negative effects: {int((tmg['upper'] < 0).sum())}",
        f"{int(level * 100)}% positive effects: {int((tmg['lower'] > 0).sum())}",
        "",
        "Most negative TMG effects:",
        tmg.sort_values("mean").head(10).to_string(),
        "",
        "Most positive TMG effects:",
        tmg.sort_values("mean", ascending=False).head(10).to_string(),
        "",
        "## Gamma Trait Effects",
        "",
        _gamma_report(fit, level=level),
        "",
        "## Diagnostics",
        "",
        _diagnostics_report(fit, "Beta"),
        "",
        _diagnostics_report(fit, "Gamma"),
        "",
        "## Posterior Predictive Checks",
        "",
        _ppc_report(
            fit,
            y_data,
            ppc_x_data,
            level=level,
            ppc_seed=ppc_seed,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
        ),
        "",
        "## TMG Gradient",
        "",
        _format_gradient("predicted richness", richness_gradient),
        "",
        _format_gradient("community-weighted CN", cn_gradient),
        "",
        "## Expected Literature Pattern",
        "",
        "- many species respond negatively to TMG",
        "- typical species responds negatively to TMG",
        "- higher-CN species respond less negatively or positively to TMG",
        "- species richness decreases along TMG",
        "- community-weighted CN increases along TMG",
    ]
    return "\n".join(lines) + "\n"


def build_ppc_report(
    posterior: Path,
    project: Path,
    level: float = 0.95,
    ppc_seed: int = 1,
    model_config: Path | None = None,
    random_effects: str = "none",
    unseen_groups: str = "error",
) -> str:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    model = _load_model(model_config)
    fit = HmscFit.from_file(posterior, model=model)
    x_data = pd.read_csv(project / "data" / "X.csv", index_col=0)
    y_data = pd.read_csv(project / "data" / "Y_presence.csv", index_col=0)
    ppc_x_data = _prediction_data_for_random_effects(x_data, model, random_effects)
    return (
        _ppc_report(
            fit,
            y_data,
            ppc_x_data,
            level=level,
            ppc_seed=ppc_seed,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
        )
        + "\n"
    )


def _gamma_report(fit: HmscFit, level: float) -> str:
    try:
        gamma = fit.gamma_summary(level=level)
    except ValueError:
        return "Gamma samples unavailable."
    return gamma.to_string(index=False)


def _ppc_report(
    fit: HmscFit,
    y_data: pd.DataFrame,
    x_data: pd.DataFrame,
    level: float,
    ppc_seed: int,
    random_effects: str = "none",
    unseen_groups: str = "error",
) -> str:
    species = fit.ppc_summary(
        y_data,
        x_data,
        level=level,
        random_effects=random_effects,
        unseen_groups=unseen_groups,
        rng_seed=ppc_seed,
    )
    richness = fit.richness_ppc_summary(
        y_data,
        x_data,
        level=level,
        random_effects=random_effects,
        unseen_groups=unseen_groups,
        rng_seed=ppc_seed,
    )
    species = species.assign(abs_error=(species["observed_mean"] - species["replicated_mean"]).abs())
    richness = richness.assign(abs_error=(richness["observed_richness"] - richness["replicated_richness"]).abs())
    species_covered = int(species["covered"].sum())
    richness_covered = int(richness["covered"].sum())
    return "\n".join(
        [
            f"species occupancy covered: {species_covered} / {len(species)}",
            f"site richness covered: {richness_covered} / {len(richness)}",
            f"random effects: {random_effects}",
            f"mean absolute species occupancy error: {species['abs_error'].mean():.6g}",
            f"mean absolute site richness error: {richness['abs_error'].mean():.6g}",
            "",
            "Largest species occupancy errors:",
            species.sort_values("abs_error", ascending=False).head(10).to_string(index=False),
            "",
            "Largest site richness errors:",
            richness.sort_values("abs_error", ascending=False).head(10).to_string(index=False),
        ]
    )


def _load_model(model_config: Path | None):
    if model_config is None:
        return None
    model, _config = model_from_config(model_config)
    return model


def _prediction_data_for_random_effects(
    x_data: pd.DataFrame,
    model: object | None,
    random_effects: str,
) -> pd.DataFrame:
    if random_effects != "known":
        return x_data
    if model is None or getattr(model, "study_design", None) is None:
        raise ValueError("--model-config with study_design is required for known random-effect PPCs")
    output = x_data.copy()
    study_design = model.study_design
    for level_name, spec in getattr(model, "random_levels", {}).items():
        column = spec.get("column", level_name)
        if column not in output:
            output[column] = study_design[column].to_numpy()
    return output


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


def _endpoint_summary(samples: np.ndarray, level: float) -> dict[str, float]:
    first = samples[:, :, 0].reshape(-1)
    last = samples[:, :, -1].reshape(-1)
    delta = last - first
    alpha = (1 - level) / 2
    return {
        "first_mean": float(first.mean()),
        "first_lower": float(np.quantile(first, alpha)),
        "first_upper": float(np.quantile(first, 1 - alpha)),
        "last_mean": float(last.mean()),
        "last_lower": float(np.quantile(last, alpha)),
        "last_upper": float(np.quantile(last, 1 - alpha)),
        "delta_mean": float(delta.mean()),
        "delta_lower": float(np.quantile(delta, alpha)),
        "delta_upper": float(np.quantile(delta, 1 - alpha)),
        "p_delta_positive": float((delta > 0).mean()),
        "p_delta_negative": float((delta < 0).mean()),
    }


def _format_gradient(name: str, values: dict[str, float]) -> str:
    return "\n".join(
        [
            f"{name}:",
            f"  low TMG mean: {values['first_mean']:.6g} [{values['first_lower']:.6g}, {values['first_upper']:.6g}]",
            f"  high TMG mean: {values['last_mean']:.6g} [{values['last_lower']:.6g}, {values['last_upper']:.6g}]",
            f"  delta mean: {values['delta_mean']:.6g} [{values['delta_lower']:.6g}, {values['delta_upper']:.6g}]",
            f"  P(delta > 0): {values['p_delta_positive']:.3f}",
            f"  P(delta < 0): {values['p_delta_negative']:.3f}",
        ]
    )


if __name__ == "__main__":
    main()
