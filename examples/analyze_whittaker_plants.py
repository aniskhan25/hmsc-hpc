"""Summarize the Whittaker plant HMSC book validation run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    parser.add_argument("--output", help="optional path for the text report")
    args = parser.parse_args()

    report = build_report(
        posterior=Path(args.posterior),
        project=Path(args.project),
        gradient_points=args.gradient_points,
        level=args.level,
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
) -> str:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    fit = HmscFit.from_file(posterior)
    x_data = pd.read_csv(project / "data" / "X.csv", index_col=0)
    traits = pd.read_csv(project / "data" / "traits.csv", index_col=0)

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


def _gamma_report(fit: HmscFit, level: float) -> str:
    try:
        gamma = fit.gamma_summary(level=level)
    except ValueError:
        return "Gamma samples unavailable."
    return gamma.to_string(index=False)


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
