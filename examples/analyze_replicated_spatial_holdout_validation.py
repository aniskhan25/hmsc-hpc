"""Aggregate replicated spatial hold-out and NNGP ordering validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.analyze_spatial_holdout_validation import build_metrics_table


METRICS = ["correlation", "rmse", "mae", "interval_coverage", "mean_interval_width"]


def analyze_replicates(
    manifest_path: Path,
    run_root: Path,
    level: float = 0.95,
    prediction_seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    required = {"task_id", "seed", "ordering", "model", "project"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    rows = []
    for task in manifest.itertuples(index=False):
        task_root = run_root / "tasks" / f"task_{int(task.task_id):03d}"
        prediction = task_root / "prediction.csv"
        posterior = task_root / "posterior.h5"
        if not prediction.exists() or not posterior.exists():
            raise FileNotFoundError(f"task {task.task_id} is missing prediction or posterior output")
        model_key = str(task.model)
        if model_key != "fixed":
            model_key = f"{model_key}_conditional"
        metrics = build_metrics_table(
            Path(task.project),
            {model_key: prediction},
            {model_key: posterior},
            level=level,
            seed=prediction_seed,
        ).iloc[0]
        row = {
            "task_id": int(task.task_id),
            "seed": int(task.seed),
            "ordering": str(task.ordering),
            "model": str(task.model),
        }
        row.update({metric: float(metrics[metric]) for metric in METRICS})
        rows.append(row)
    raw = pd.DataFrame(rows).sort_values(["model", "ordering", "seed"]).reset_index(drop=True)
    summary = _aggregate_metrics(raw, nominal_level=level)
    ordering = _nngp_ordering_deltas(raw)
    return raw, summary, ordering


def _aggregate_metrics(raw: pd.DataFrame, nominal_level: float = 0.95) -> pd.DataFrame:
    aggregations = {}
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sd"] = (metric, "std")
    summary = raw.groupby(["model", "ordering"], sort=True).agg(**aggregations).reset_index()
    summary.insert(2, "replicates", raw.groupby(["model", "ordering"]).size().to_numpy())
    coverage_bounds = (
        raw.groupby(["model", "ordering"], sort=True)["interval_coverage"]
        .agg(["min", "max"])
        .reset_index(drop=True)
    )
    summary["coverage_bias"] = summary["interval_coverage_mean"] - nominal_level
    summary["coverage_min"] = coverage_bounds["min"].to_numpy()
    summary["coverage_max"] = coverage_bounds["max"].to_numpy()
    return summary


def _nngp_ordering_deltas(raw: pd.DataFrame) -> pd.DataFrame:
    nngp = raw.loc[raw["model"] == "spatial_nngp"].copy()
    canonical = nngp.loc[nngp["ordering"] == "canonical"].set_index("seed")
    if canonical.empty:
        raise ValueError("NNGP ordering analysis requires canonical rows")
    rows = []
    for row in nngp.itertuples(index=False):
        if row.seed not in canonical.index:
            raise ValueError(f"NNGP seed {row.seed} has no canonical baseline")
        baseline = canonical.loc[row.seed]
        rows.append(
            {
                "seed": int(row.seed),
                "ordering": row.ordering,
                "correlation_delta": float(row.correlation - baseline["correlation"]),
                "rmse_delta": float(row.rmse - baseline["rmse"]),
                "coverage_delta": float(row.interval_coverage - baseline["interval_coverage"]),
                "width_delta": float(row.mean_interval_width - baseline["mean_interval_width"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["ordering", "seed"]).reset_index(drop=True)


def _build_report(summary: pd.DataFrame, ordering: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Replicated Spatial Hold-Out Validation Report",
            "",
            "## Across-Seed Metrics",
            "",
            summary.to_string(index=False),
            "",
            "## NNGP Ordering Deltas From Canonical",
            "",
            ordering.to_string(index=False),
            "",
            "Positive RMSE deltas indicate worse prediction than canonical ordering.",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--prediction-seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, summary, ordering = analyze_replicates(
        args.manifest,
        args.run_root,
        level=args.level,
        prediction_seed=args.prediction_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_build_report(summary, ordering), encoding="utf-8")
    raw.to_csv(args.output.with_name(f"{args.output.stem}_raw.csv"), index=False)
    summary.to_csv(args.output.with_name(f"{args.output.stem}_summary.csv"), index=False)
    ordering.to_csv(args.output.with_name(f"{args.output.stem}_ordering.csv"), index=False)
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
