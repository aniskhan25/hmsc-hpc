"""Compare Neural-HMSC predictive-only artifacts on held-out benchmark data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.posterior import HmscFit


PREDICTIVE_FILE = "neural_predictive_distribution.h5"
SUMMARY_CSV = "predictive_score_summary.csv"
DETAIL_CSV = "predictive_score_details.csv"
REPORT_JSON = "predictive_score_comparison.json"
REPORT_MD = "predictive_score_comparison.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="benchmark output directory to compare; repeat for each candidate",
    )
    parser.add_argument("--baseline", required=True, help="baseline label")
    parser.add_argument("--output", required=True, help="report output directory")
    args = parser.parse_args()

    runs = [_load_run(spec) for spec in args.run]
    result = compare_predictive_runs(runs, baseline_label=args.baseline)
    paths = write_predictive_score_report(result, args.output)
    print(f"Wrote {paths['summary_csv']}")
    print(f"Wrote {paths['detail_csv']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")


def compare_predictive_runs(
    runs: Sequence[dict[str, Any]],
    *,
    baseline_label: str,
) -> dict[str, Any]:
    """Return predictive score rows and deltas against a baseline."""
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    labels = [str(run["label"]) for run in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")
    if baseline_label not in labels:
        raise ValueError(f"baseline {baseline_label!r} is not one of {labels}")
    baseline = next(run for run in runs if run["label"] == baseline_label)
    baseline_row = _score_run(baseline)
    rows = []
    for run in runs:
        row = _score_run(run)
        rows.append({**row, **_deltas(row, baseline_row)})
    return {
        "kind": "predictive_score_comparison",
        "baseline": baseline_label,
        "summary": rows,
        "details": [
            {
                "run": row["run"],
                "path": row["path"],
                "predictive_artifact": row["predictive_artifact"],
                "data_dir": row["data_dir"],
                "predictive_mean_calibration": row["predictive_mean_calibration"],
            }
            for row in rows
        ],
    }


def write_predictive_score_report(
    result: dict[str, Any],
    output: str | Path,
) -> dict[str, Path]:
    """Write predictive score comparison outputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(result["summary"])
    details = pd.DataFrame(result["details"])
    summary_csv = output / SUMMARY_CSV
    detail_csv = output / DETAIL_CSV
    json_path = output / REPORT_JSON
    markdown = output / REPORT_MD
    csv_summary = summary.drop(columns=["predictive_mean_calibration"], errors="ignore")
    csv_summary.to_csv(summary_csv, index=False)
    csv_details = details.copy()
    if "predictive_mean_calibration" in csv_details:
        csv_details["predictive_mean_calibration"] = csv_details[
            "predictive_mean_calibration"
        ].map(lambda value: json.dumps(value) if isinstance(value, dict) else value)
    csv_details.to_csv(detail_csv, index=False)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_render_markdown(result), encoding="utf-8")
    return {
        "summary_csv": summary_csv,
        "detail_csv": detail_csv,
        "json": json_path,
        "markdown": markdown,
    }


def _load_run(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError("--run must use LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("run label must not be empty")
    path = Path(raw_path).expanduser().resolve()
    probit = path / "probit"
    run_root = probit if probit.exists() else path
    predictive = run_root / PREDICTIVE_FILE
    data_dir = run_root / "data"
    if not predictive.exists():
        raise FileNotFoundError(f"missing predictive artifact: {predictive}")
    if not data_dir.exists():
        raise FileNotFoundError(f"missing data directory: {data_dir}")
    return {
        "label": label,
        "path": str(path),
        "predictive": predictive,
        "data_dir": data_dir,
    }


def _score_run(run: dict[str, Any]) -> dict[str, Any]:
    fit = HmscFit.from_file(run["predictive"])
    X = pd.read_csv(Path(run["data_dir"]) / "X.csv", index_col=0)
    Y = pd.read_csv(Path(run["data_dir"]) / "Y.csv", index_col=0)
    prediction = fit.predict_mean(X).loc[Y.index, Y.columns].clip(1e-9, 1.0 - 1e-9)
    probability = prediction.to_numpy(dtype=float)
    observed = Y.to_numpy(dtype=float)
    prevalence = observed.mean(axis=0)
    predicted_prevalence = probability.mean(axis=0)
    richness = observed.sum(axis=1)
    predicted_richness = probability.sum(axis=1)
    calibration = (
        fit.metadata.get("predictive_mean_calibration")
        if isinstance(fit.metadata, dict)
        else None
    )
    return {
        "run": run["label"],
        "path": run["path"],
        "predictive_artifact": str(run["predictive"]),
        "data_dir": str(run["data_dir"]),
        "brier_score": float(np.mean(np.square(probability - observed))),
        "log_loss": float(
            -np.mean(
                observed * np.log(probability)
                + (1.0 - observed) * np.log1p(-probability)
            )
        ),
        "predictive_rmse": float(np.sqrt(np.mean(np.square(probability - observed)))),
        "prevalence_mae": float(np.mean(np.abs(predicted_prevalence - prevalence))),
        "richness_mae": float(np.mean(np.abs(predicted_richness - richness))),
        "predictive_mean_calibration": calibration,
    }


def _deltas(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    metrics = [
        "brier_score",
        "log_loss",
        "predictive_rmse",
        "prevalence_mae",
        "richness_mae",
    ]
    result = {}
    for metric in metrics:
        result[f"{metric}_delta_vs_baseline"] = float(row[metric] - baseline[metric])
        result[f"{metric}_ratio_vs_baseline"] = float(
            row[metric] / max(float(baseline[metric]), np.finfo(float).eps)
        )
    return result


def _render_markdown(result: dict[str, Any]) -> str:
    frame = pd.DataFrame(result["summary"]).drop(
        columns=["predictive_mean_calibration"],
        errors="ignore",
    )
    preferred = [
        "run",
        "brier_score",
        "brier_score_ratio_vs_baseline",
        "log_loss",
        "log_loss_ratio_vs_baseline",
        "predictive_rmse",
        "predictive_rmse_ratio_vs_baseline",
        "prevalence_mae",
        "richness_mae",
    ]
    columns = [column for column in preferred if column in frame.columns]
    table = frame.loc[:, columns].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: f"{value:.6g}" if pd.notna(value) else ""
            )
    lines = [
        "# Neural-HMSC Predictive Score Comparison",
        "",
        f"Baseline: `{result['baseline']}`",
        "",
        _markdown_table(table),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = [
            "" if pd.isna(row[column]) else str(row[column]) for column in frame.columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
