"""Compare Neural-HMSC runs on a fixed independent evaluation bundle.

The harness consumes benchmark output directories that already contain
``neural_hmsc_sbc_diagnostics.json``. It does not refit or resimulate anything;
all runs must expose the same SBC/OOD row keys. The report compares final
calibrated rows against a named frozen baseline so candidate improvements cannot
be confused with internal calibration-batch diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SBC_FILE = "neural_hmsc_sbc_diagnostics.json"
SUMMARY_CSV = "fixed_evaluation_summary.csv"
DETAIL_CSV = "fixed_evaluation_deltas.csv"
REPORT_JSON = "fixed_evaluation_comparison.json"
REPORT_MD = "fixed_evaluation_comparison.md"


@dataclass(frozen=True)
class EvaluationRun:
    """One already-generated benchmark run."""

    label: str
    path: Path
    rows: list[dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="benchmark output directory to compare; repeat for each candidate",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="label from --run to use as the frozen baseline",
    )
    parser.add_argument("--output", required=True, help="report output directory")
    parser.add_argument(
        "--posterior-variant",
        default="calibrated",
        help="posterior_variant to compare; default: calibrated",
    )
    parser.add_argument(
        "--min-mean-ood-delta",
        type=float,
        default=0.010,
        help="minimum mean OOD coverage gain over baseline",
    )
    parser.add_argument(
        "--min-combined-delta",
        type=float,
        default=0.010,
        help="minimum combined-shift coverage gain over baseline",
    )
    parser.add_argument(
        "--max-worst-ood-loss",
        type=float,
        default=0.005,
        help="maximum allowed worst-domain OOD coverage loss versus baseline",
    )
    parser.add_argument(
        "--max-in-domain-coverage",
        type=float,
        default=0.990,
        help="maximum accepted in-domain overall coverage",
    )
    parser.add_argument(
        "--min-in-domain-coverage",
        type=float,
        default=0.900,
        help="minimum accepted in-domain overall coverage",
    )
    args = parser.parse_args()

    runs = [_load_run(spec) for spec in args.run]
    result = compare_fixed_evaluation_runs(
        runs,
        baseline_label=args.baseline,
        posterior_variant=args.posterior_variant,
        min_mean_ood_delta=args.min_mean_ood_delta,
        min_combined_delta=args.min_combined_delta,
        max_worst_ood_loss=args.max_worst_ood_loss,
        min_in_domain_coverage=args.min_in_domain_coverage,
        max_in_domain_coverage=args.max_in_domain_coverage,
    )
    paths = write_fixed_evaluation_report(result, args.output)
    print(f"Wrote {paths['summary_csv']}")
    print(f"Wrote {paths['detail_csv']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")


def compare_fixed_evaluation_runs(
    runs: Sequence[EvaluationRun],
    *,
    baseline_label: str,
    posterior_variant: str = "calibrated",
    min_mean_ood_delta: float = 0.010,
    min_combined_delta: float = 0.010,
    max_worst_ood_loss: float = 0.005,
    min_in_domain_coverage: float = 0.900,
    max_in_domain_coverage: float = 0.990,
) -> dict[str, Any]:
    """Return summary and detailed deltas for fixed-key evaluation runs."""
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")
    by_label = {run.label: run for run in runs}
    if baseline_label not in by_label:
        raise ValueError(f"baseline {baseline_label!r} is not one of {labels}")

    row_keys = {_row_key(row) for row in runs[0].rows}
    for run in runs[1:]:
        current = {_row_key(row) for row in run.rows}
        missing = sorted(row_keys.difference(current))
        extra = sorted(current.difference(row_keys))
        if missing or extra:
            raise ValueError(
                f"run {run.label!r} does not match fixed evaluation keys; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

    baseline_rows = _indexed_rows(by_label[baseline_label].rows)
    summary_rows = []
    detail_rows = []
    baseline_summary = _summarize_run(
        by_label[baseline_label],
        posterior_variant=posterior_variant,
    )

    for run in runs:
        run_rows = _indexed_rows(run.rows)
        summary = _summarize_run(run, posterior_variant=posterior_variant)
        deltas = _summary_deltas(summary, baseline_summary)
        accepted = _acceptance_flags(
            summary,
            deltas,
            baseline_label=baseline_label,
            run_label=run.label,
            min_mean_ood_delta=min_mean_ood_delta,
            min_combined_delta=min_combined_delta,
            max_worst_ood_loss=max_worst_ood_loss,
            min_in_domain_coverage=min_in_domain_coverage,
            max_in_domain_coverage=max_in_domain_coverage,
        )
        summary_rows.append({**summary, **deltas, **accepted})
        for key in sorted(row_keys):
            row = run_rows[key]
            baseline = baseline_rows[key]
            detail_rows.append(
                {
                    "run": run.label,
                    "baseline": baseline_label,
                    "distribution": key[0],
                    "simulation_domain": key[1],
                    "ood_regime": key[2],
                    "posterior_variant": key[3],
                    "sbc_stratum_kind": key[4],
                    "sbc_stratum_label": key[5],
                    "coverage_95": _float_or_none(
                        row.get("sbc_beta_interval_coverage_95")
                    ),
                    "coverage_95_baseline": _float_or_none(
                        baseline.get("sbc_beta_interval_coverage_95")
                    ),
                    "coverage_95_delta": _delta(
                        row,
                        baseline,
                        "sbc_beta_interval_coverage_95",
                    ),
                    "rank_mean": _float_or_none(row.get("sbc_rank_mean")),
                    "rank_mean_baseline": _float_or_none(baseline.get("sbc_rank_mean")),
                    "rank_mean_delta": _delta(row, baseline, "sbc_rank_mean"),
                    "rank_variance": _float_or_none(row.get("sbc_rank_variance")),
                    "rank_variance_baseline": _float_or_none(
                        baseline.get("sbc_rank_variance")
                    ),
                    "rank_variance_delta": _delta(
                        row,
                        baseline,
                        "sbc_rank_variance",
                    ),
                    "beta_rmse": _float_or_none(row.get("sbc_beta_mean_rmse")),
                    "beta_rmse_baseline": _float_or_none(
                        baseline.get("sbc_beta_mean_rmse")
                    ),
                    "beta_rmse_delta": _delta(row, baseline, "sbc_beta_mean_rmse"),
                }
            )

    return {
        "kind": "fixed_independent_evaluation_comparison",
        "baseline": baseline_label,
        "posterior_variant": posterior_variant,
        "fixed_row_key_count": len(row_keys),
        "acceptance_thresholds": {
            "min_mean_ood_delta": float(min_mean_ood_delta),
            "min_combined_delta": float(min_combined_delta),
            "max_worst_ood_loss": float(max_worst_ood_loss),
            "min_in_domain_coverage": float(min_in_domain_coverage),
            "max_in_domain_coverage": float(max_in_domain_coverage),
        },
        "summary": summary_rows,
        "details": detail_rows,
    }


def write_fixed_evaluation_report(
    result: dict[str, Any],
    output: str | Path,
) -> dict[str, Path]:
    """Write fixed evaluation comparison JSON, CSV, and Markdown reports."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(result["summary"])
    details = pd.DataFrame(result["details"])
    summary_csv = output / SUMMARY_CSV
    detail_csv = output / DETAIL_CSV
    json_path = output / REPORT_JSON
    markdown = output / REPORT_MD
    summary.to_csv(summary_csv, index=False)
    details.to_csv(detail_csv, index=False)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_render_markdown(result), encoding="utf-8")
    return {
        "summary_csv": summary_csv,
        "detail_csv": detail_csv,
        "json": json_path,
        "markdown": markdown,
    }


def _load_run(spec: str) -> EvaluationRun:
    if "=" not in spec:
        raise ValueError("--run must use LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("run label must not be empty")
    path = Path(raw_path).expanduser().resolve()
    sbc_path = path / SBC_FILE
    if not sbc_path.exists():
        raise FileNotFoundError(f"missing {sbc_path}")
    rows = json.loads(sbc_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{sbc_path} must contain a non-empty JSON list")
    return EvaluationRun(label=label, path=path, rows=rows)


def _indexed_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = _row_key(row)
        if key in indexed:
            raise ValueError(f"duplicate SBC row key: {key}")
        indexed[key] = row
    return indexed


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("distribution")),
        str(row.get("simulation_domain")),
        _none_key(row.get("ood_regime")),
        str(row.get("posterior_variant")),
        str(row.get("sbc_stratum_kind")),
        str(row.get("sbc_stratum_label")),
    )


def _summarize_run(
    run: EvaluationRun,
    *,
    posterior_variant: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in run.rows
        if str(row.get("posterior_variant")) == posterior_variant
    ]
    overall = [
        row
        for row in rows
        if str(row.get("sbc_stratum_kind")) == "overall"
        and str(row.get("sbc_stratum_label")) == "overall"
    ]
    in_domain = [
        row for row in overall if str(row.get("simulation_domain")) == "in_distribution"
    ]
    ood = [row for row in overall if str(row.get("simulation_domain")) == "ood"]
    if len(in_domain) != 1:
        raise ValueError(
            f"run {run.label!r} must have exactly one calibrated in-domain overall row"
        )
    ood_coverages = [
        _float_or_none(row.get("sbc_beta_interval_coverage_95")) for row in ood
    ]
    ood_coverages = [value for value in ood_coverages if value is not None]
    by_regime = {
        str(row.get("ood_regime")): _float_or_none(
            row.get("sbc_beta_interval_coverage_95")
        )
        for row in ood
    }
    rare_row = _find_row(rows, "in_distribution", None, "prevalence", "rare")
    intermediate_design_row = _find_row(
        rows,
        "in_distribution",
        None,
        "design_information",
        "intermediate",
    )
    high_design_row = _find_row(
        rows,
        "in_distribution",
        None,
        "design_information",
        "high",
    )
    return {
        "run": run.label,
        "path": str(run.path),
        "posterior_variant": posterior_variant,
        "in_domain_coverage_95": _float_or_none(
            in_domain[0].get("sbc_beta_interval_coverage_95")
        ),
        "in_domain_rank_mean": _float_or_none(in_domain[0].get("sbc_rank_mean")),
        "in_domain_rank_variance": _float_or_none(
            in_domain[0].get("sbc_rank_variance")
        ),
        "rare_prevalence_coverage_95": _row_float(
            rare_row,
            "sbc_beta_interval_coverage_95",
        ),
        "rare_prevalence_rank_mean": _row_float(rare_row, "sbc_rank_mean"),
        "intermediate_design_coverage_95": _row_float(
            intermediate_design_row,
            "sbc_beta_interval_coverage_95",
        ),
        "high_design_coverage_95": _row_float(
            high_design_row,
            "sbc_beta_interval_coverage_95",
        ),
        "mean_ood_coverage_95": (
            float(sum(ood_coverages) / len(ood_coverages)) if ood_coverages else None
        ),
        "worst_ood_coverage_95": min(ood_coverages) if ood_coverages else None,
        "effect_size_shift_coverage_95": by_regime.get("effect_size_shift"),
        "combined_shift_coverage_95": by_regime.get("combined_shift"),
        "covariate_shift_coverage_95": by_regime.get("covariate_shift"),
    }


def _summary_deltas(
    summary: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "in_domain_coverage_95",
        "rare_prevalence_coverage_95",
        "rare_prevalence_rank_mean",
        "intermediate_design_coverage_95",
        "high_design_coverage_95",
        "mean_ood_coverage_95",
        "worst_ood_coverage_95",
        "effect_size_shift_coverage_95",
        "combined_shift_coverage_95",
        "covariate_shift_coverage_95",
    ]
    return {
        f"{key}_delta_vs_baseline": _number_delta(summary.get(key), baseline.get(key))
        for key in keys
    }


def _acceptance_flags(
    summary: dict[str, Any],
    deltas: dict[str, Any],
    *,
    baseline_label: str,
    run_label: str,
    min_mean_ood_delta: float,
    min_combined_delta: float,
    max_worst_ood_loss: float,
    min_in_domain_coverage: float,
    max_in_domain_coverage: float,
) -> dict[str, Any]:
    is_baseline = run_label == baseline_label
    mean_delta = deltas.get("mean_ood_coverage_95_delta_vs_baseline")
    worst_delta = deltas.get("worst_ood_coverage_95_delta_vs_baseline")
    combined_delta = deltas.get("combined_shift_coverage_95_delta_vs_baseline")
    in_domain = summary.get("in_domain_coverage_95")
    rare_delta = deltas.get("rare_prevalence_coverage_95_delta_vs_baseline")
    mean_ok = _ge(mean_delta, min_mean_ood_delta)
    worst_ok = _ge(worst_delta, -float(max_worst_ood_loss))
    combined_ok = _ge(combined_delta, min_combined_delta)
    in_domain_ok = (
        isinstance(in_domain, float)
        and min_in_domain_coverage <= in_domain <= max_in_domain_coverage
    )
    rare_ok = rare_delta is None or _ge(rare_delta, -0.005)
    passed = bool(
        not is_baseline
        and mean_ok
        and worst_ok
        and combined_ok
        and in_domain_ok
        and rare_ok
    )
    return {
        "is_baseline": is_baseline,
        "mean_ood_gate_passed": bool(mean_ok),
        "worst_ood_gate_passed": bool(worst_ok),
        "combined_shift_gate_passed": bool(combined_ok),
        "in_domain_gate_passed": bool(in_domain_ok),
        "rare_gate_passed": bool(rare_ok),
        "fixed_evaluation_acceptance_passed": passed,
    }


def _find_row(
    rows: Sequence[dict[str, Any]],
    simulation_domain: str,
    ood_regime: str | None,
    stratum_kind: str,
    stratum_label: str,
) -> dict[str, Any] | None:
    regime_key = _none_key(ood_regime)
    for row in rows:
        if (
            str(row.get("simulation_domain")) == simulation_domain
            and _none_key(row.get("ood_regime")) == regime_key
            and str(row.get("sbc_stratum_kind")) == stratum_kind
            and str(row.get("sbc_stratum_label")) == stratum_label
        ):
            return row
    return None


def _row_float(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    return _float_or_none(row.get(key))


def _delta(
    row: dict[str, Any],
    baseline: dict[str, Any],
    key: str,
) -> float | None:
    return _number_delta(row.get(key), baseline.get(key))


def _number_delta(value: Any, baseline: Any) -> float | None:
    left = _float_or_none(value)
    right = _float_or_none(baseline)
    if left is None or right is None:
        return None
    return float(left - right)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _none_key(value: Any) -> str:
    return "<none>" if value is None else str(value)


def _ge(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return bool(numeric is not None and numeric >= threshold)


def _render_markdown(result: dict[str, Any]) -> str:
    summary = pd.DataFrame(result["summary"])
    preferred = [
        "run",
        "is_baseline",
        "fixed_evaluation_acceptance_passed",
        "in_domain_coverage_95",
        "rare_prevalence_coverage_95",
        "mean_ood_coverage_95",
        "mean_ood_coverage_95_delta_vs_baseline",
        "worst_ood_coverage_95",
        "worst_ood_coverage_95_delta_vs_baseline",
        "effect_size_shift_coverage_95",
        "combined_shift_coverage_95",
        "combined_shift_coverage_95_delta_vs_baseline",
    ]
    columns = [column for column in preferred if column in summary.columns]
    table = summary.loc[:, columns].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: f"{value:.4f}" if pd.notna(value) else ""
            )
    lines = [
        "# Neural-HMSC Fixed Evaluation Comparison",
        "",
        f"Baseline: `{result['baseline']}`",
        "",
        f"Posterior variant: `{result['posterior_variant']}`",
        "",
        f"Fixed SBC row keys: `{result['fixed_row_key_count']}`",
        "",
        _markdown_table(table),
        "",
        "## Acceptance Thresholds",
        "",
    ]
    for key, value in result["acceptance_thresholds"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
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
