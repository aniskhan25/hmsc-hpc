"""Evaluate cross-dataset promotion gates for predictive-mean competitors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.predictive_selection import (
    PredictiveNoDegradationThresholds,
    evaluate_cross_dataset_predictive_gate,
    render_cross_dataset_predictive_gate_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="held-out metrics CSV; repeat for each real-data dataset",
    )
    parser.add_argument(
        "--baseline-model",
        default="neural_predictive_only_calibrated",
        help="scale-only predictive model row used as the no-degradation baseline",
    )
    parser.add_argument(
        "--candidate-model",
        default="neural_predictive_mean_calibrated",
        help="predictive-mean candidate model row",
    )
    parser.add_argument("--max-brier-ratio", type=float, default=1.0)
    parser.add_argument("--max-log-loss-ratio", type=float, default=1.0)
    parser.add_argument("--max-predictive-rmse-ratio", type=float, default=1.0)
    parser.add_argument("--max-richness-mae-ratio", type=float, default=1.0)
    parser.add_argument("--min-mean-brier-gain", type=float, default=0.0)
    parser.add_argument("--min-mean-log-loss-gain", type=float, default=0.0)
    parser.add_argument(
        "--simulated-summary",
        type=Path,
        help="optional simulated predictive summary CSV or JSON",
    )
    parser.add_argument("--simulated-baseline-run", default="external_monotone")
    parser.add_argument(
        "--simulated-candidate-run",
        default="external_monotone_response",
    )
    parser.add_argument("--min-simulated-brier-gain", type=float, default=0.0)
    parser.add_argument("--min-simulated-log-loss-gain", type=float, default=0.0)
    parser.add_argument("--output", required=True, help="output directory")
    args = parser.parse_args()

    thresholds = PredictiveNoDegradationThresholds(
        max_brier_ratio=args.max_brier_ratio,
        max_log_loss_ratio=args.max_log_loss_ratio,
        max_predictive_rmse_ratio=args.max_predictive_rmse_ratio,
        max_richness_mae_ratio=args.max_richness_mae_ratio,
        min_mean_brier_gain=args.min_mean_brier_gain,
        min_mean_log_loss_gain=args.min_mean_log_loss_gain,
        min_simulated_brier_gain=args.min_simulated_brier_gain,
        min_simulated_log_loss_gain=args.min_simulated_log_loss_gain,
    )
    datasets = [_load_dataset(spec) for spec in args.dataset]
    simulated = (
        None if args.simulated_summary is None else _load_simulated(args.simulated_summary)
    )
    result = evaluate_cross_dataset_predictive_gate(
        datasets,
        baseline_model=args.baseline_model,
        candidate_model=args.candidate_model,
        thresholds=thresholds,
        simulated_summary=simulated,
        simulated_baseline_run=args.simulated_baseline_run,
        simulated_candidate_run=args.simulated_candidate_run,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "predictive_mean_promotion_gate.json"
    csv_path = output / "predictive_mean_promotion_gate_datasets.csv"
    md_path = output / "predictive_mean_promotion_gate.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(result["datasets"]).drop(columns=["failure_reasons"]).to_csv(
        csv_path,
        index=False,
    )
    md_path.write_text(
        render_cross_dataset_predictive_gate_markdown(result),
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if not result["promotion_gate_passed"]:
        raise SystemExit(1)


def _load_dataset(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError("--dataset must use LABEL=CSV")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("dataset label must not be empty")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"missing held-out metrics CSV: {path}")
    return {"label": label, "metrics": pd.read_csv(path), "path": str(path)}


def _load_simulated(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing simulated summary: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ["predictive_summary", "summary"]:
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
    if isinstance(payload, list):
        return payload
    raise ValueError(f"could not find simulated summary rows in {path}")


if __name__ == "__main__":
    main()
