"""Analyze held-out spatial predictions against deterministic truth."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.config import model_from_config
from pyhmsc.posterior import HmscFit


MODEL_CONFIGS = {
    "fixed": "model_fixed.yaml",
    "spatial_full": "model_spatial_full.yaml",
    "spatial_gpp": "model_spatial_gpp.yaml",
    "spatial_nngp": "model_spatial_nngp.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("examples/projects/simulated_spatial_holdout_validation"),
    )
    parser.add_argument("--prediction", action="append", default=[], help="MODEL=predictions.csv")
    parser.add_argument("--posterior", action="append", default=[], help="MODEL=posterior.h5")
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    predictions = _parse_named_paths(args.prediction, "--prediction")
    posteriors = _parse_named_paths(args.posterior, "--posterior")
    metrics = build_metrics_table(args.project, predictions, posteriors, level=args.level)
    report = _build_report(args.project, metrics, args.level)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        metrics.to_csv(args.output.with_suffix(".csv"), index=False)
    print(report)


def build_metrics_table(
    project: Path,
    predictions: dict[str, Path],
    posteriors: dict[str, Path] | None = None,
    level: float = 0.95,
) -> pd.DataFrame:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    posteriors = posteriors or {}
    truth = pd.read_csv(project / "data/test/truth_linear_predictor.csv", index_col=0)
    rows = []
    for name in MODEL_CONFIGS:
        if name not in predictions:
            continue
        prediction = pd.read_csv(predictions[name], index_col=0)
        prediction = prediction.loc[truth.index, truth.columns]
        error = prediction - truth
        values = prediction.to_numpy(dtype=float)
        truth_values = truth.to_numpy(dtype=float)
        row = {
            "model": name,
            "correlation": _safe_corr(values.ravel(), truth_values.ravel()),
            "rmse": float(np.sqrt(np.mean(error.to_numpy(dtype=float) ** 2))),
            "mae": float(np.mean(np.abs(error.to_numpy(dtype=float)))),
            "interval_coverage": "n/a",
            "mean_interval_width": "n/a",
        }
        if name in posteriors:
            coverage, width = _interval_metrics(project, name, posteriors[name], truth, level)
            row["interval_coverage"] = coverage
            row["mean_interval_width"] = width
        rows.append(row)
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No recognized prediction files were provided")
    fixed = metrics.loc[metrics["model"] == "fixed", "rmse"]
    baseline = float(fixed.iloc[0]) if not fixed.empty else float("nan")
    metrics["rmse_improvement_vs_fixed"] = baseline - metrics["rmse"]
    return metrics


def _interval_metrics(
    project: Path,
    name: str,
    posterior: Path,
    truth: pd.DataFrame,
    level: float,
) -> tuple[float, float]:
    model, _config = model_from_config(project / MODEL_CONFIGS[name])
    fit = HmscFit.from_file(posterior, model=model)
    X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    kwargs: dict[str, object] = {}
    if name != "fixed":
        kwargs = {
            "study_design": pd.read_csv(project / "data/test/study_design.csv", index_col=0),
            "coords": pd.read_csv(project / "data/test/coords.csv", index_col=0),
            "random_effects": "known",
            "unseen_groups": "nearest",
        }
    interval = fit.predict_ci(X, level=level, response=False, **kwargs)
    lower = interval["lower"].loc[truth.index, truth.columns]
    upper = interval["upper"].loc[truth.index, truth.columns]
    covered = (lower <= truth) & (truth <= upper)
    return float(covered.to_numpy().mean()), float((upper - lower).to_numpy().mean())


def _parse_named_paths(values: list[str], flag: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} must be MODEL=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if name not in MODEL_CONFIGS or not path:
            raise ValueError(f"{flag} has unknown model or empty path: {value!r}")
        parsed[name] = Path(path)
    return parsed


def _build_report(project: Path, metrics: pd.DataFrame, level: float) -> str:
    best = metrics.loc[metrics["rmse"].idxmin()]
    lines = [
        "# Simulated Spatial Hold-Out Prediction Validation Report",
        "",
        f"Project: {project}",
        "Training sites: 80",
        "Held-out sites: 20",
        f"Credible interval level: {level:.3f}",
        "",
        metrics.to_string(index=False),
        "",
        f"Lowest held-out RMSE: {best['model']} ({float(best['rmse']):.6f})",
        "",
        "Nearest spatial predictions reuse the closest sampled random-effect unit; "
        "they are a baseline rather than conditional spatial interpolation.",
    ]
    return "\n".join(lines) + "\n"


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


if __name__ == "__main__":
    main()
