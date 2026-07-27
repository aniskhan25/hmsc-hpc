"""Promotion gates for predictive-only Neural-HMSC competitors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictiveNoDegradationThresholds:
    """Thresholds for cross-dataset predictive promotion checks."""

    max_brier_ratio: float = 1.0
    max_log_loss_ratio: float = 1.0
    max_predictive_rmse_ratio: float = 1.0
    max_richness_mae_ratio: float = 1.0
    min_mean_brier_gain: float = 0.0
    min_mean_log_loss_gain: float = 0.0
    min_simulated_brier_gain: float = 0.0
    min_simulated_log_loss_gain: float = 0.0

    def validate(self) -> None:
        if self.max_brier_ratio < 1.0:
            raise ValueError("max_brier_ratio must be at least one")
        if self.max_log_loss_ratio < 1.0:
            raise ValueError("max_log_loss_ratio must be at least one")
        if self.max_predictive_rmse_ratio < 1.0:
            raise ValueError("max_predictive_rmse_ratio must be at least one")
        if self.max_richness_mae_ratio < 1.0:
            raise ValueError("max_richness_mae_ratio must be at least one")
        if self.min_mean_brier_gain < 0.0:
            raise ValueError("min_mean_brier_gain must be non-negative")
        if self.min_mean_log_loss_gain < 0.0:
            raise ValueError("min_mean_log_loss_gain must be non-negative")
        if self.min_simulated_brier_gain < 0.0:
            raise ValueError("min_simulated_brier_gain must be non-negative")
        if self.min_simulated_log_loss_gain < 0.0:
            raise ValueError("min_simulated_log_loss_gain must be non-negative")


def evaluate_cross_dataset_predictive_gate(
    datasets: Iterable[Mapping[str, Any]],
    *,
    baseline_model: str = "neural_predictive_only_calibrated",
    candidate_model: str = "neural_predictive_mean_calibrated",
    thresholds: PredictiveNoDegradationThresholds | None = None,
    simulated_summary: Iterable[Mapping[str, Any]] | None = None,
    simulated_baseline_run: str = "external_monotone",
    simulated_candidate_run: str = "external_monotone_response",
) -> dict[str, Any]:
    """Evaluate whether a predictive-mean candidate is promotable.

    Real-data held-out metrics are decisive: every supplied dataset must stay
    within the configured Brier, log-loss, predictive-RMSE, and richness-MAE
    limits relative to the scale-only baseline. An optional simulated summary
    can additionally require positive simulated proper-score gains, but it
    cannot override a real-data degradation.
    """
    thresholds = thresholds or PredictiveNoDegradationThresholds()
    thresholds.validate()

    dataset_rows = []
    for dataset in datasets:
        label = str(dataset["label"])
        metrics = _metrics_frame(dataset["metrics"])
        baseline = _single_model_row(metrics, baseline_model, label)
        candidate = _single_model_row(metrics, candidate_model, label)
        row = _dataset_gate_row(
            label,
            baseline,
            candidate,
            thresholds=thresholds,
        )
        dataset_rows.append(row)

    if not dataset_rows:
        raise ValueError("at least one dataset is required")

    simulated_gate = None
    if simulated_summary is not None:
        simulated_gate = _simulated_gate(
            simulated_summary,
            baseline_run=simulated_baseline_run,
            candidate_run=simulated_candidate_run,
            thresholds=thresholds,
        )

    failed = [
        reason
        for row in dataset_rows
        for reason in row["failure_reasons"]
    ]
    mean_brier_gain = float(
        np.mean(
            [
                row["brier_score_baseline"] - row["brier_score_candidate"]
                for row in dataset_rows
            ]
        )
    )
    mean_log_loss_gain = float(
        np.mean(
            [
                row["log_loss_baseline"] - row["log_loss_candidate"]
                for row in dataset_rows
            ]
        )
    )
    if mean_brier_gain < thresholds.min_mean_brier_gain:
        failed.append(
            f"real-data mean Brier gain {mean_brier_gain:.6g} below "
            f"{thresholds.min_mean_brier_gain:.6g}"
        )
    if mean_log_loss_gain < thresholds.min_mean_log_loss_gain:
        failed.append(
            f"real-data mean log-loss gain {mean_log_loss_gain:.6g} below "
            f"{thresholds.min_mean_log_loss_gain:.6g}"
        )
    if simulated_gate is not None and not simulated_gate["passed"]:
        failed.extend(simulated_gate["failure_reasons"])

    return {
        "kind": "cross_dataset_predictive_no_degradation_gate",
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "thresholds": {
            "max_brier_ratio": float(thresholds.max_brier_ratio),
            "max_log_loss_ratio": float(thresholds.max_log_loss_ratio),
            "max_predictive_rmse_ratio": float(
                thresholds.max_predictive_rmse_ratio
            ),
            "max_richness_mae_ratio": float(thresholds.max_richness_mae_ratio),
            "min_mean_brier_gain": float(thresholds.min_mean_brier_gain),
            "min_mean_log_loss_gain": float(thresholds.min_mean_log_loss_gain),
            "min_simulated_brier_gain": float(thresholds.min_simulated_brier_gain),
            "min_simulated_log_loss_gain": float(
                thresholds.min_simulated_log_loss_gain
            ),
        },
        "datasets": dataset_rows,
        "mean_brier_gain": mean_brier_gain,
        "mean_log_loss_gain": mean_log_loss_gain,
        "simulated_gate": simulated_gate,
        "promotion_gate_passed": not failed,
        "failure_reasons": failed,
    }


def render_cross_dataset_predictive_gate_markdown(result: Mapping[str, Any]) -> str:
    """Render a gate result as Markdown."""
    rows = pd.DataFrame(result["datasets"])
    display = rows[
        [
            "dataset",
            "passed",
            "brier_score_baseline",
            "brier_score_candidate",
            "brier_score_ratio",
            "log_loss_baseline",
            "log_loss_candidate",
            "log_loss_ratio",
            "predictive_rmse_ratio",
            "richness_mae_ratio",
        ]
    ].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")

    lines = [
        "# Predictive Mean Promotion Gate",
        "",
        f"Baseline model: `{result['baseline_model']}`",
        f"Candidate model: `{result['candidate_model']}`",
        f"Promotion gate passed: {bool(result['promotion_gate_passed'])}",
        f"Mean Brier gain: `{float(result['mean_brier_gain']):.6g}`",
        f"Mean log-loss gain: `{float(result['mean_log_loss_gain']):.6g}`",
        "",
        _markdown_table(display),
        "",
    ]
    simulated = result.get("simulated_gate")
    if isinstance(simulated, dict):
        lines.extend(
            [
                "## Simulated Gate",
                "",
                f"Passed: {bool(simulated['passed'])}",
                f"Brier gain: `{float(simulated['brier_gain']):.6g}`",
                f"Log-loss gain: `{float(simulated['log_loss_gain']):.6g}`",
                "",
            ]
        )
    failures = result.get("failure_reasons", [])
    if failures:
        lines.extend(["## Failure Reasons", ""])
        lines.extend(f"- {reason}" for reason in failures)
        lines.append("")
    return "\n".join(lines)


def _metrics_frame(metrics: Any) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        frame = metrics.copy()
    else:
        frame = pd.DataFrame(metrics)
    required = {"model", "brier_score", "log_loss"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"metrics are missing required columns: {missing}")
    return frame


def _single_model_row(frame: pd.DataFrame, model: str, dataset: str) -> dict[str, Any]:
    matches = frame.loc[frame["model"] == model]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {model!r} row for dataset {dataset!r}; "
            f"found {len(matches)}"
        )
    return matches.iloc[0].to_dict()


def _dataset_gate_row(
    dataset: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    thresholds: PredictiveNoDegradationThresholds,
) -> dict[str, Any]:
    brier_base = _finite_metric(baseline, "brier_score", dataset)
    brier_candidate = _finite_metric(candidate, "brier_score", dataset)
    log_base = _finite_metric(baseline, "log_loss", dataset)
    log_candidate = _finite_metric(candidate, "log_loss", dataset)
    rmse_base = _finite_metric(baseline, "predictive_rmse", dataset)
    rmse_candidate = _finite_metric(candidate, "predictive_rmse", dataset)
    richness_base = _finite_metric(baseline, "richness_mae", dataset)
    richness_candidate = _finite_metric(candidate, "richness_mae", dataset)
    brier_ratio = brier_candidate / max(brier_base, np.finfo(float).eps)
    log_ratio = log_candidate / max(log_base, np.finfo(float).eps)
    rmse_ratio = rmse_candidate / max(rmse_base, np.finfo(float).eps)
    richness_ratio = richness_candidate / max(richness_base, np.finfo(float).eps)
    failures = []
    if brier_ratio > thresholds.max_brier_ratio:
        failures.append(
            f"{dataset}: Brier ratio {brier_ratio:.6g} exceeds "
            f"{thresholds.max_brier_ratio:.6g}"
        )
    if log_ratio > thresholds.max_log_loss_ratio:
        failures.append(
            f"{dataset}: log-loss ratio {log_ratio:.6g} exceeds "
            f"{thresholds.max_log_loss_ratio:.6g}"
        )
    if rmse_ratio > thresholds.max_predictive_rmse_ratio:
        failures.append(
            f"{dataset}: predictive RMSE ratio {rmse_ratio:.6g} exceeds "
            f"{thresholds.max_predictive_rmse_ratio:.6g}"
        )
    if richness_ratio > thresholds.max_richness_mae_ratio:
        failures.append(
            f"{dataset}: richness MAE ratio {richness_ratio:.6g} exceeds "
            f"{thresholds.max_richness_mae_ratio:.6g}"
        )
    return {
        "dataset": dataset,
        "passed": not failures,
        "brier_score_baseline": brier_base,
        "brier_score_candidate": brier_candidate,
        "brier_score_delta": brier_candidate - brier_base,
        "brier_score_ratio": brier_ratio,
        "log_loss_baseline": log_base,
        "log_loss_candidate": log_candidate,
        "log_loss_delta": log_candidate - log_base,
        "log_loss_ratio": log_ratio,
        "predictive_rmse_baseline": rmse_base,
        "predictive_rmse_candidate": rmse_candidate,
        "predictive_rmse_ratio": rmse_ratio,
        "richness_mae_baseline": richness_base,
        "richness_mae_candidate": richness_candidate,
        "richness_mae_ratio": richness_ratio,
        "failure_reasons": failures,
    }


def _simulated_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_run: str,
    candidate_run: str,
    thresholds: PredictiveNoDegradationThresholds,
) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    brier_column = (
        "brier_score" if "brier_score" in frame.columns else "brier_score_mean"
    )
    log_loss_column = (
        "log_loss" if "log_loss" in frame.columns else "log_loss_mean"
    )
    required = {"run", brier_column, log_loss_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"simulated summary is missing columns: {missing}")
    baseline = _single_run_row(frame, baseline_run)
    candidate = _single_run_row(frame, candidate_run)
    brier_gain = float(baseline[brier_column] - candidate[brier_column])
    log_loss_gain = float(baseline[log_loss_column] - candidate[log_loss_column])
    failures = []
    if brier_gain < thresholds.min_simulated_brier_gain:
        failures.append(
            "simulated: Brier gain "
            f"{brier_gain:.6g} below {thresholds.min_simulated_brier_gain:.6g}"
        )
    if log_loss_gain < thresholds.min_simulated_log_loss_gain:
        failures.append(
            "simulated: log-loss gain "
            f"{log_loss_gain:.6g} below {thresholds.min_simulated_log_loss_gain:.6g}"
        )
    return {
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "brier_baseline": float(baseline[brier_column]),
        "brier_candidate": float(candidate[brier_column]),
        "brier_gain": brier_gain,
        "log_loss_baseline": float(baseline[log_loss_column]),
        "log_loss_candidate": float(candidate[log_loss_column]),
        "log_loss_gain": log_loss_gain,
        "passed": not failures,
        "failure_reasons": failures,
    }


def _single_run_row(frame: pd.DataFrame, run: str) -> dict[str, Any]:
    matches = frame.loc[frame["run"] == run]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one simulated run {run!r}; found {len(matches)}")
    return matches.iloc[0].to_dict()


def _finite_metric(row: Mapping[str, Any], metric: str, dataset: str) -> float:
    value = float(row[metric])
    if not np.isfinite(value):
        raise ValueError(f"{dataset}: metric {metric!r} is not finite")
    return value


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
