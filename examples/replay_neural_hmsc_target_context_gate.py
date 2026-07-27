"""Replay target-context selection against frozen real-data checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from examples.generate_neural_hmsc_big_spatial_transfer import generate_project
from examples.run_neural_hmsc_big_spatial_transfer import (
    _target_context_conditioned_selector,
)
from pyhmsc.neural.inference import NeuralHmscInference
from pyhmsc.neural.mean_calibration import (
    select_predictive_mean_calibration_for_context,
)
from pyhmsc.posterior import HmscFit


METRICS = ("brier_score", "log_loss", "predictive_rmse", "richness_mae")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-matrix", type=Path, default=Path("examples/big_spatial/data")
    )
    parser.add_argument(
        "--source-project",
        type=Path,
        default=Path("examples/projects/big_spatial_plants_validation"),
    )
    parser.add_argument("--datasets", type=int, default=32)
    parser.add_argument("--max-brier-ratio", type=float, default=1.0)
    parser.add_argument("--max-log-loss-ratio", type=float, default=1.0)
    parser.add_argument("--min-score-improvement", type=float, default=0.0001)
    args = parser.parse_args()
    _validate_args(parser, args)

    args.output.mkdir(parents=True, exist_ok=True)
    project = args.output / "target_context_project"
    generate_project(args.source_matrix, args.source_project, project)
    train_Y = pd.read_csv(project / "data/train/Y.csv", index_col=0)
    train_X = pd.read_csv(project / "data/train/X.csv", index_col=0)
    test_X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    species_names = [str(value) for value in train_Y.columns]

    rows = []
    for seed in args.seeds:
        rows.append(
            replay_seed(
                run_root=args.run_root,
                seed=seed,
                train_X=train_X,
                test_X=test_X,
                species_names=species_names,
                datasets=args.datasets,
                max_brier_ratio=args.max_brier_ratio,
                max_log_loss_ratio=args.max_log_loss_ratio,
                min_score_improvement=args.min_score_improvement,
                output=args.output,
            )
        )
    frame = pd.DataFrame(rows)
    summary = summarize_replay(frame)
    frame.to_csv(args.output / "target_context_gate_replay.csv", index=False)
    (args.output / "target_context_gate_replay.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_report(frame, summary)
    (args.output / "target_context_gate_replay.md").write_text(
        report,
        encoding="utf-8",
    )
    print(report)


def replay_seed(
    *,
    run_root: Path,
    seed: int,
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    species_names: list[str],
    datasets: int,
    max_brier_ratio: float,
    max_log_loss_ratio: float,
    min_score_improvement: float,
    output: Path,
) -> dict[str, Any]:
    seed_root = run_root / f"seed_{seed}"
    whittaker_root = seed_root / "whittaker"
    big_spatial_root = seed_root / "big_spatial"
    predictive_fit = HmscFit.from_file(
        whittaker_root / "neural_predictive_distribution.h5"
    )
    frozen_selector = predictive_fit.metadata.get("predictive_mean_selector")
    engine = NeuralHmscInference.load(whittaker_root / "neural_checkpoint")
    selector, gate = _target_context_conditioned_selector(
        engine=engine,
        selector_metadata=frozen_selector,
        train_X=train_X,
        test_X=test_X,
        species_names=species_names,
        seed=seed,
        datasets=datasets,
        max_brier_ratio=max_brier_ratio,
        max_log_loss_ratio=max_log_loss_ratio,
        min_score_improvement=min_score_improvement,
    )
    active, decision = select_predictive_mean_calibration_for_context(
        selector,
        context="big_spatial_transfer",
        distribution="probit",
        n_covariates=2,
        n_species=len(species_names),
    )

    # Existing held-out summaries are opened only after the simulation-only decision.
    heldout = pd.read_csv(
        big_spatial_root / "big_spatial_transfer_heldout_metrics.csv"
    )
    baseline = _model_row(heldout, "neural_predictive_only_calibrated")
    candidate = _model_row(heldout, "neural_predictive_mean_calibrated")
    selected = baseline if active is None else candidate
    ratios = {
        metric: float(selected[metric]) / float(baseline[metric])
        for metric in METRICS
    }
    no_degradation = bool(all(value <= 1.0 + 1.0e-12 for value in ratios.values()))
    genuine_improvement = bool(
        active is not None
        and ratios["brier_score"] < 1.0
        and ratios["log_loss"] < 1.0
        and no_degradation
    )
    row = {
        "seed": int(seed),
        "target_gate_passed": bool(gate["passed"]),
        "selector_action": decision["action"],
        "selector_reason": decision["reason"],
        "candidate_slope": float(gate["candidate"]["slope"]),
        "candidate_intercept": float(gate["candidate"]["intercept"]),
        "target_calibration_brier_ratio": float(
            gate["target_calibration"]["brier_ratio"]
        ),
        "target_calibration_log_loss_ratio": float(
            gate["target_calibration"]["log_loss_ratio"]
        ),
        "target_validation_brier_ratio": float(
            gate["target_validation"]["brier_ratio"]
        ),
        "target_validation_log_loss_ratio": float(
            gate["target_validation"]["log_loss_ratio"]
        ),
        "replay_brier_ratio": ratios["brier_score"],
        "replay_log_loss_ratio": ratios["log_loss"],
        "replay_rmse_ratio": ratios["predictive_rmse"],
        "replay_richness_mae_ratio": ratios["richness_mae"],
        "no_degradation_passed": no_degradation,
        "genuine_big_spatial_improvement": genuine_improvement,
        "target_response_used_for_selection": False,
    }
    seed_output = output / f"seed_{seed}"
    seed_output.mkdir(parents=True, exist_ok=True)
    (seed_output / "target_context_gate.json").write_text(
        json.dumps(
            {
                "seed": int(seed),
                "gate": gate,
                "selector_decision": decision,
                "replay": row,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return row


def summarize_replay(frame: pd.DataFrame) -> dict[str, Any]:
    n = int(len(frame))
    no_degradation_count = int(frame["no_degradation_passed"].sum())
    genuine_count = int(frame["genuine_big_spatial_improvement"].sum())
    gate_count = int(frame["target_gate_passed"].sum())
    if no_degradation_count == n and genuine_count >= 2:
        decision = "target_context_gate_promotion_candidate"
    elif no_degradation_count < n:
        decision = "target_context_gate_failed_no_degradation"
    else:
        decision = "target_context_gate_safe_but_insufficient_improvement"
    return {
        "kind": "frozen_target_context_gate_replay",
        "seeds": [int(value) for value in frame["seed"]],
        "target_gate_pass_count": gate_count,
        "no_degradation_pass_count": no_degradation_count,
        "genuine_big_spatial_improvement_count": genuine_count,
        "promotion_requires": {
            "no_degradation_pass_count": n,
            "minimum_genuine_big_spatial_improvement_count": 2,
        },
        "target_response_used_for_selection": False,
        "decision": decision,
        "rows": frame.to_dict(orient="records"),
    }


def render_report(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    columns = [
        "seed",
        "target_gate_passed",
        "selector_action",
        "candidate_slope",
        "candidate_intercept",
        "target_calibration_brier_ratio",
        "target_calibration_log_loss_ratio",
        "target_validation_brier_ratio",
        "target_validation_log_loss_ratio",
        "replay_brier_ratio",
        "replay_log_loss_ratio",
        "replay_rmse_ratio",
        "replay_richness_mae_ratio",
        "no_degradation_passed",
        "genuine_big_spatial_improvement",
    ]
    return "\n".join(
        [
            "# Target-Context Gate Frozen Replay",
            "",
            "Target responses were unavailable during simulation-gate evaluation and selection.",
            "Existing held-out summaries were opened only for the final frozen replay score.",
            "",
            f"Decision: `{summary['decision']}`",
            f"Target-gate passes: {summary['target_gate_pass_count']} / {len(frame)}",
            f"No-degradation passes: {summary['no_degradation_pass_count']} / {len(frame)}",
            "Genuine Big Spatial improvements: "
            f"{summary['genuine_big_spatial_improvement_count']} / {len(frame)}",
            "",
            "```text",
            frame.loc[:, columns].to_string(index=False),
            "```",
            "",
        ]
    )


def _model_row(frame: pd.DataFrame, model: str) -> pd.Series:
    rows = frame.loc[frame["model"] == model]
    if len(rows) != 1:
        raise ValueError(f"expected one held-out row for {model!r}")
    return rows.iloc[0]


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.datasets <= 0:
        parser.error("--datasets must be positive")
    if args.max_brier_ratio < 1.0:
        parser.error("--max-brier-ratio must be at least one")
    if args.max_log_loss_ratio < 1.0:
        parser.error("--max-log-loss-ratio must be at least one")
    if args.min_score_improvement < 0.0:
        parser.error("--min-score-improvement must be non-negative")
    if not args.run_root.exists():
        parser.error(f"--run-root does not exist: {args.run_root}")


if __name__ == "__main__":
    main()
