"""Aggregate bounded real-data Neural-HMSC sensitivity runs.

The expected layout is produced by
``docs/lumi_neural_hmsc_realdata_sensitivity_sbatch.sh``:

```
RUN_ROOT/
  seed_<seed>/
    whittaker/
    big_spatial/
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASETS = ("whittaker", "big_spatial")
PREDICTIVE_MODEL = "neural_predictive_only_calibrated"
PREDICTIVE_MEAN_MODEL = "neural_predictive_mean_calibrated"
UNCALIBRATED_MODEL = "neural_uncalibrated"
REFERENCE_MODEL = "qualified_python_mcmc_fixed"
MIN_TRANSFER_IMPROVEMENT_SEEDS = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="output prefix without extension; defaults to RUN_ROOT/realdata_sensitivity",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    prefix = args.output_prefix or args.run_root / "realdata_sensitivity"
    rows = aggregate_sensitivity(args.run_root, args.seeds, strict=args.strict)
    frame = pd.DataFrame(rows)
    summary = summarize_sensitivity(frame)

    prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    prefix.with_suffix(".md").write_text(
        render_report(frame, summary),
        encoding="utf-8",
    )
    print(prefix.with_suffix(".md").read_text(encoding="utf-8"))


def aggregate_sensitivity(
    run_root: Path,
    seeds: list[int],
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        seed_root = run_root / f"seed_{seed}"
        for dataset in DATASETS:
            dataset_root = seed_root / dataset
            try:
                rows.append(_dataset_row(seed, dataset, dataset_root))
            except FileNotFoundError:
                if strict:
                    raise
                rows.append(
                    {
                        "seed": int(seed),
                        "dataset": dataset,
                        "status": "missing",
                        "run_root": str(dataset_root),
                    }
                )
    return rows


def summarize_sensitivity(frame: pd.DataFrame) -> dict[str, Any]:
    completed = frame.loc[frame["status"] == "completed"].copy()
    summary: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_completed_rows": int(len(completed)),
        "seeds": sorted(int(seed) for seed in frame["seed"].dropna().unique()),
        "datasets": sorted(str(value) for value in frame["dataset"].dropna().unique()),
    }
    if completed.empty:
        summary["decision"] = "no_completed_rows"
        return summary

    for dataset in DATASETS:
        dataset_rows = completed.loc[completed["dataset"] == dataset]
        summary[dataset] = _summarize_dataset(dataset_rows)

    paired = []
    for seed in summary["seeds"]:
        seed_rows = completed.loc[completed["seed"] == seed]
        whittaker = _single_row(seed_rows, "whittaker")
        big_spatial = _single_row(seed_rows, "big_spatial")
        if whittaker is None or big_spatial is None:
            continue
        paired.append(
            {
                "seed": int(seed),
                "both_passed": bool(
                    whittaker["acceptance_passed"] and big_spatial["acceptance_passed"]
                ),
                "mcmc_advantage_persisted": bool(
                    whittaker["mcmc_brier_advantage"]
                    and whittaker["mcmc_log_loss_advantage"]
                    and big_spatial["mcmc_brier_advantage"]
                    and big_spatial["mcmc_log_loss_advantage"]
                ),
                "promotion_gate_passed": bool(
                    whittaker.get("promotion_gate_passed", False)
                    and big_spatial.get("promotion_gate_passed", False)
                ),
                "big_spatial_transfer_outcome": big_spatial.get(
                    "predictive_mean_transfer_outcome"
                ),
                "big_spatial_genuine_transfer_improvement": bool(
                    big_spatial.get("predictive_mean_genuine_transfer_improvement")
                ),
                "big_spatial_safe_identity_fallback": bool(
                    big_spatial.get("predictive_mean_safe_identity_fallback")
                ),
            }
        )
    summary["paired_seeds"] = paired
    summary["paired_pass_count"] = int(sum(row["both_passed"] for row in paired))
    summary["paired_mcmc_advantage_count"] = int(
        sum(row["mcmc_advantage_persisted"] for row in paired)
    )
    summary["paired_promotion_gate_pass_count"] = int(
        sum(row["promotion_gate_passed"] for row in paired)
    )
    summary["paired_genuine_transfer_improvement_count"] = int(
        sum(row["big_spatial_genuine_transfer_improvement"] for row in paired)
    )
    summary["paired_safe_identity_fallback_count"] = int(
        sum(row["big_spatial_safe_identity_fallback"] for row in paired)
    )
    if summary["paired_pass_count"] < len(summary["seeds"]):
        summary["decision"] = "inspect_seed_level_dataset_acceptance"
    elif summary["paired_promotion_gate_pass_count"] < len(summary["seeds"]):
        summary["decision"] = "inspect_seed_level_no_degradation"
    elif (
        summary["paired_genuine_transfer_improvement_count"]
        >= MIN_TRANSFER_IMPROVEMENT_SEEDS
    ):
        summary["decision"] = "stable_guarded_selector_promotion_candidate"
    else:
        summary["decision"] = "safe_identity_fallback_not_promotable"
    return summary


def render_report(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    columns = [
        "seed",
        "dataset",
        "acceptance_passed",
        "selector_action",
        "selector_context_family",
        "predictive_mean_transfer_outcome",
        "promotion_gate_passed",
        "promotion_brier_ratio",
        "promotion_log_loss_ratio",
        "predictive_mean_vs_predictive_brier_ratio",
        "predictive_mean_vs_predictive_log_loss_ratio",
        "predictive_vs_uncalibrated_brier_ratio",
        "predictive_vs_reference_brier_ratio",
        "predictive_vs_uncalibrated_log_loss_ratio",
        "predictive_vs_reference_log_loss_ratio",
        "predictive_minus_reference_macro_auc",
        "predictive_vs_reference_prevalence_mae_ratio",
        "predictive_vs_reference_richness_mae_ratio",
        "sbc_coverage_95",
        "sbc_rank_mean",
        "sbc_rank_variance",
        "neural_inference_seconds",
        "mcmc_seconds",
    ]
    available = [column for column in columns if column in frame.columns]
    table = frame.loc[:, available].to_string(index=False) if available else frame.to_string(index=False)
    return "\n".join(
        [
            "# Neural-HMSC Real-Data Sensitivity",
            "",
            "This report aggregates the bounded three-seed real-data sensitivity check.",
            "It does not tune calibration objectives or acceptance thresholds.",
            "",
            f"Decision: `{summary.get('decision')}`",
            f"Completed rows: {summary.get('n_completed_rows')} / {summary.get('n_rows')}",
            f"Paired pass count: {summary.get('paired_pass_count', 0)}",
            f"Paired promotion-gate pass count: {summary.get('paired_promotion_gate_pass_count', 0)}",
            f"Paired genuine transfer-improvement count: {summary.get('paired_genuine_transfer_improvement_count', 0)}",
            f"Paired safe identity-fallback count: {summary.get('paired_safe_identity_fallback_count', 0)}",
            f"Paired MCMC advantage count: {summary.get('paired_mcmc_advantage_count', 0)}",
            "",
            "## Per-Seed Metrics",
            "",
            "```text",
            table,
            "```",
            "",
            "## Summary JSON",
            "",
            "```json",
            json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
            "```",
            "",
        ]
    )


def _dataset_row(seed: int, dataset: str, root: Path) -> dict[str, Any]:
    if dataset == "whittaker":
        acceptance = _read_json(root / "whittaker_acceptance.json")
        heldout = pd.read_csv(root / "whittaker_heldout_metrics.csv")
        metadata = _read_json(root / "run_metadata.json")
        sbc = _read_sbc(root / "whittaker_neural_sbc_diagnostics.csv")
        acceptance_passed = bool(acceptance.get("qualification_acceptance_passed"))
    elif dataset == "big_spatial":
        acceptance = _read_json(root / "big_spatial_transfer_acceptance.json")
        heldout = pd.read_csv(root / "big_spatial_transfer_heldout_metrics.csv")
        metadata = _read_json(root / "run_metadata.json")
        sbc = {}
        acceptance_passed = bool(acceptance.get("predictive_transfer_acceptance_passed"))
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    predictive = _metric_row(heldout, PREDICTIVE_MODEL)
    predictive_mean = _optional_metric_row(heldout, PREDICTIVE_MEAN_MODEL)
    uncalibrated = _metric_row(heldout, UNCALIBRATED_MODEL)
    reference = _metric_row(heldout, REFERENCE_MODEL)
    promotion = _promotion_dataset_row(root.parent / "promotion_gate", dataset)
    selector_decision = _selector_decision(acceptance, metadata)
    row = {
        "seed": int(seed),
        "dataset": dataset,
        "status": "completed",
        "run_root": str(root),
        "acceptance_passed": acceptance_passed,
        "reference_parity_qualified": bool(
            acceptance.get("reference_parity_qualified")
        ),
        "predictive_brier_score": float(predictive["brier_score"]),
        "uncalibrated_brier_score": float(uncalibrated["brier_score"]),
        "reference_brier_score": float(reference["brier_score"]),
        "predictive_log_loss": float(predictive["log_loss"]),
        "uncalibrated_log_loss": float(uncalibrated["log_loss"]),
        "reference_log_loss": float(reference["log_loss"]),
        "predictive_macro_auc": float(predictive["macro_auc"]),
        "reference_macro_auc": float(reference["macro_auc"]),
        "predictive_prevalence_mae": float(predictive["prevalence_mae"]),
        "reference_prevalence_mae": float(reference["prevalence_mae"]),
        "predictive_richness_mae": float(predictive["richness_mae"]),
        "reference_richness_mae": float(reference["richness_mae"]),
        "predictive_mean_brier_score": _metric_float(predictive_mean, "brier_score"),
        "predictive_mean_log_loss": _metric_float(predictive_mean, "log_loss"),
        "selector_action": selector_decision.get("action"),
        "selector_selected": selector_decision.get("selected"),
        "selector_context_family": selector_decision.get("context_family"),
        "selector_reason": selector_decision.get("reason"),
        "promotion_gate_passed": promotion.get("promotion_gate_passed"),
        "promotion_brier_ratio": promotion.get("brier_ratio"),
        "promotion_log_loss_ratio": promotion.get("log_loss_ratio"),
        "promotion_failure_reasons": promotion.get("failure_reasons"),
        "neural_inference_seconds": _optional_float(
            metadata.get("neural_inference_seconds")
        ),
        "mcmc_seconds": _optional_float(metadata.get("mcmc_seconds")),
    }
    row.update(
        {
            "predictive_vs_uncalibrated_brier_ratio": _ratio(
                row["predictive_brier_score"], row["uncalibrated_brier_score"]
            ),
            "predictive_vs_reference_brier_ratio": _ratio(
                row["predictive_brier_score"], row["reference_brier_score"]
            ),
            "predictive_vs_uncalibrated_log_loss_ratio": _ratio(
                row["predictive_log_loss"], row["uncalibrated_log_loss"]
            ),
            "predictive_vs_reference_log_loss_ratio": _ratio(
                row["predictive_log_loss"], row["reference_log_loss"]
            ),
            "predictive_minus_reference_macro_auc": row["predictive_macro_auc"]
            - row["reference_macro_auc"],
            "predictive_vs_reference_prevalence_mae_ratio": _ratio(
                row["predictive_prevalence_mae"], row["reference_prevalence_mae"]
            ),
            "predictive_vs_reference_richness_mae_ratio": _ratio(
                row["predictive_richness_mae"], row["reference_richness_mae"]
            ),
            "mcmc_brier_advantage": row["reference_brier_score"]
            < row["predictive_brier_score"],
            "mcmc_log_loss_advantage": row["reference_log_loss"]
            < row["predictive_log_loss"],
        }
    )
    if predictive_mean is not None:
        row.update(
            {
                "predictive_mean_vs_predictive_brier_ratio": _ratio(
                    row["predictive_mean_brier_score"], row["predictive_brier_score"]
                ),
                "predictive_mean_vs_predictive_log_loss_ratio": _ratio(
                    row["predictive_mean_log_loss"], row["predictive_log_loss"]
                ),
            }
        )
    row.update(_predictive_mean_transfer_outcome(row))
    row.update(sbc)
    return row


def _summarize_dataset(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"n": 0}
    metrics = [
        "acceptance_passed",
        "promotion_gate_passed",
        "promotion_brier_ratio",
        "promotion_log_loss_ratio",
        "predictive_mean_genuine_transfer_improvement",
        "predictive_mean_safe_identity_fallback",
        "predictive_mean_vs_predictive_brier_ratio",
        "predictive_mean_vs_predictive_log_loss_ratio",
        "predictive_vs_uncalibrated_brier_ratio",
        "predictive_vs_reference_brier_ratio",
        "predictive_vs_uncalibrated_log_loss_ratio",
        "predictive_vs_reference_log_loss_ratio",
        "predictive_minus_reference_macro_auc",
        "predictive_vs_reference_prevalence_mae_ratio",
        "predictive_vs_reference_richness_mae_ratio",
        "sbc_coverage_95",
        "sbc_rank_mean",
        "sbc_rank_variance",
        "neural_inference_seconds",
        "mcmc_seconds",
    ]
    out: dict[str, Any] = {"n": int(len(rows))}
    out["acceptance_pass_count"] = int(rows["acceptance_passed"].sum())
    if "promotion_gate_passed" in rows:
        out["promotion_gate_pass_count"] = int(rows["promotion_gate_passed"].fillna(False).sum())
    if "predictive_mean_genuine_transfer_improvement" in rows:
        out["genuine_transfer_improvement_count"] = int(
            rows["predictive_mean_genuine_transfer_improvement"]
            .fillna(False)
            .sum()
        )
    if "predictive_mean_safe_identity_fallback" in rows:
        out["safe_identity_fallback_count"] = int(
            rows["predictive_mean_safe_identity_fallback"].fillna(False).sum()
        )
    out["mcmc_brier_advantage_count"] = int(rows["mcmc_brier_advantage"].sum())
    out["mcmc_log_loss_advantage_count"] = int(rows["mcmc_log_loss_advantage"].sum())
    for metric in metrics:
        if metric not in rows:
            continue
        values = pd.to_numeric(rows[metric], errors="coerce").dropna()
        if values.empty:
            continue
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_min"] = float(values.min())
        out[f"{metric}_max"] = float(values.max())
    return out


def _single_row(rows: pd.DataFrame, dataset: str) -> pd.Series | None:
    matches = rows.loc[rows["dataset"] == dataset]
    if len(matches) != 1:
        return None
    return matches.iloc[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_sbc(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    row = frame.loc[
        (frame["posterior_variant"] == "coefficient_calibrated")
        & (frame["sbc_stratum_kind"] == "overall")
    ]
    if len(row) != 1:
        raise ValueError(f"expected one calibrated overall SBC row in {path}")
    values = row.iloc[0]
    return {
        "sbc_coverage_95": float(values["sbc_beta_interval_coverage_95"]),
        "sbc_rank_mean": float(values["sbc_rank_mean"]),
        "sbc_rank_variance": float(values["sbc_rank_variance"]),
        "sbc_beta_mean_rmse": float(values["sbc_beta_mean_rmse"]),
    }


def _metric_row(frame: pd.DataFrame, model: str) -> pd.Series:
    rows = frame.loc[frame["model"] == model]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row for model {model!r}")
    return rows.iloc[0]


def _optional_metric_row(frame: pd.DataFrame, model: str) -> pd.Series | None:
    rows = frame.loc[frame["model"] == model]
    if len(rows) == 0:
        return None
    if len(rows) != 1:
        raise ValueError(f"expected at most one row for model {model!r}")
    return rows.iloc[0]


def _metric_float(row: pd.Series | None, column: str) -> float | None:
    if row is None:
        return None
    return float(row[column])


def _selector_decision(
    acceptance: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    for source in (acceptance, metadata):
        decision = source.get("predictive_mean_selector_decision")
        if isinstance(decision, dict):
            return decision
    return {}


def _promotion_dataset_row(path: Path, dataset: str) -> dict[str, Any]:
    gate_path = path / "predictive_mean_promotion_gate.json"
    if not gate_path.exists():
        return {
            "promotion_gate_passed": None,
            "brier_ratio": None,
            "log_loss_ratio": None,
            "failure_reasons": None,
        }
    payload = _read_json(gate_path)
    for row in payload.get("datasets", []):
        if row.get("dataset", row.get("label")) == dataset:
            return {
                "promotion_gate_passed": bool(row.get("passed")),
                "brier_ratio": _optional_float(row.get("brier_score_ratio")),
                "log_loss_ratio": _optional_float(row.get("log_loss_ratio")),
                "failure_reasons": row.get("failure_reasons", []),
            }
    raise ValueError(f"missing dataset {dataset!r} in {gate_path}")


def _predictive_mean_transfer_outcome(row: dict[str, Any]) -> dict[str, Any]:
    brier_ratio = _optional_float(row.get("predictive_mean_vs_predictive_brier_ratio"))
    log_loss_ratio = _optional_float(
        row.get("predictive_mean_vs_predictive_log_loss_ratio")
    )
    action = row.get("selector_action")
    is_transfer = row.get("selector_context_family") == "transfer_like"
    applied = bool(is_transfer and action == "apply_candidate")
    identity = bool(is_transfer and action == "identity")
    strict_improvement = bool(
        applied
        and brier_ratio is not None
        and log_loss_ratio is not None
        and brier_ratio < 1.0
        and log_loss_ratio < 1.0
    )
    no_degradation = bool(
        brier_ratio is not None
        and log_loss_ratio is not None
        and brier_ratio <= 1.0
        and log_loss_ratio <= 1.0
    )
    safe_identity = bool(identity and no_degradation)
    if not is_transfer:
        outcome = "not_transfer_context"
    elif strict_improvement:
        outcome = "genuine_transfer_improvement"
    elif safe_identity:
        outcome = "safe_identity_fallback"
    elif applied and no_degradation:
        outcome = "applied_no_degradation_without_strict_improvement"
    elif applied:
        outcome = "applied_degradation"
    else:
        outcome = "transfer_no_signal"
    return {
        "predictive_mean_transfer_outcome": outcome,
        "predictive_mean_candidate_applied": applied,
        "predictive_mean_genuine_transfer_improvement": strict_improvement,
        "predictive_mean_safe_identity_fallback": safe_identity,
        "predictive_mean_no_degradation": no_degradation if is_transfer else None,
    }


def _ratio(left: float, right: float) -> float:
    if left is None or right is None:
        return float("nan")
    return float(left / right) if np.isfinite(right) and right != 0.0 else float("nan")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
