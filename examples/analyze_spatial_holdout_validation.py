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
    "spatial_full_conditional": "model_spatial_full.yaml",
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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    predictions = _parse_named_paths(args.prediction, "--prediction")
    posteriors = _parse_named_paths(args.posterior, "--posterior")
    metrics = build_metrics_table(args.project, predictions, posteriors, level=args.level, seed=args.seed)
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
    seed: int = 17,
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
            coverage, width = _interval_metrics(project, name, posteriors[name], truth, level, seed)
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
    seed: int,
) -> tuple[float, float]:
    model, _config = model_from_config(project / MODEL_CONFIGS[name])
    fit = HmscFit.from_file(posterior, model=model)
    X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    study_design = None
    coords = None
    random_effects = "none"
    unseen_groups = "error"
    spatial_prediction = "nearest"
    if name != "fixed":
        study_design = pd.read_csv(project / "data/test/study_design.csv", index_col=0)
        coords = pd.read_csv(project / "data/test/coords.csv", index_col=0)
        random_effects = "known"
        unseen_groups = "nearest"
        if name == "spatial_full_conditional":
            spatial_prediction = "conditional"
    interval = _predict_interval_compat(
        fit,
        X,
        level=level,
        study_design=study_design,
        coords=coords,
        random_effects=random_effects,
        unseen_groups=unseen_groups,
        spatial_prediction=spatial_prediction,
        rng_seed=seed,
    )
    lower = interval["lower"].loc[truth.index, truth.columns]
    upper = interval["upper"].loc[truth.index, truth.columns]
    covered = (lower <= truth) & (truth <= upper)
    return float(covered.to_numpy().mean()), float((upper - lower).to_numpy().mean())


def _predict_interval_compat(
    fit: HmscFit,
    X: pd.DataFrame,
    level: float,
    study_design: pd.DataFrame | None,
    coords: pd.DataFrame | None,
    random_effects: str,
    unseen_groups: str,
    spatial_prediction: str = "nearest",
    rng_seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    try:
        return fit.predict_ci(
            X,
            level=level,
            response=False,
            study_design=study_design,
            coords=coords,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
            spatial_prediction=spatial_prediction,
            rng_seed=rng_seed,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        if spatial_prediction == "conditional":
            raise RuntimeError(
                "Conditional spatial prediction requires the current pyhmsc posterior API"
            ) from exc
        merged = X.copy()
        for extra in [study_design, coords]:
            if extra is None:
                continue
            aligned = extra.copy()
            aligned.index = merged.index
            for column in aligned:
                if column not in merged:
                    merged[column] = aligned[column]
        return fit.predict_ci(
            merged,
            level=level,
            response=False,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
        )


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
        "Full conditional spatial predictions sample held-out Eta values from the "
        "Gaussian conditional distribution for each posterior draw.",
    ]
    return "\n".join(lines) + "\n"


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


if __name__ == "__main__":
    main()
