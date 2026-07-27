#!/usr/bin/env python3
"""Aggregate frozen Milestone 53 trait-Gamma qualification evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [_read(args.candidate), _read(args.sensitivity)]
    if (
        reports[0].get("seed") != 20260801
        or reports[0].get("candidate_predeclared") is not True
    ):
        raise ValueError("trait-Gamma candidate was not predeclared seed 20260801")
    if (
        reports[1].get("seed") != 20260802
        or reports[1].get("candidate_predeclared") is not False
    ):
        raise ValueError("trait-Gamma sensitivity evidence has the wrong role")
    if any(
        report.get("selection_used_sensitivity_outcomes") is not False
        for report in reports
    ):
        raise ValueError("trait-Gamma checkpoint selection used sensitivity outcomes")
    passed = [bool(report.get("all_gates_passed")) for report in reports]
    failed_gates = {
        str(report["seed"]): [
            name for name, value in report.get("gates", {}).items() if not value
        ]
        for report in reports
    }
    promoted = all(passed)
    payload = {
        "kind": "neural_hmsc_trait_gamma_multiseed_qualification",
        "schema_version": 1,
        "decision": (
            "trait_gamma_probit_promoted"
            if promoted
            else "trait_gamma_probit_not_promoted"
        ),
        "all_gates_passed": promoted,
        "candidate_seed": 20260801,
        "sensitivity_seeds_completed": [20260802],
        "intended_sensitivity_is_independent": False,
        "seed_overlap": {
            "training": "63/64",
            "calibration": "31/32",
            "test": "63/64",
        },
        "additional_sensitivity_stopped_pending_decision_review": not promoted,
        "selection_used_sensitivity_outcomes": False,
        "existing_releases_modified": False,
        "fixed_release_content_sha256": reports[0]["fixed_release_content_sha256"],
        "variable_release_content_sha256": reports[0][
            "variable_release_content_sha256"
        ],
        "runs": [
            {
                "seed": report["seed"],
                "decision": report["decision"],
                "all_gates_passed": report["all_gates_passed"],
                "gamma_coverage_95": report["gamma_metrics"]["coverage_95"],
                "gamma_rank_mean": report["gamma_metrics"]["rank_mean"],
                "gamma_rank_variance": report["gamma_metrics"]["rank_variance"],
                "simulated_gamma_rmse_ratio": report["simulated_reference"][
                    "gamma_rmse_ratio"
                ],
                "whittaker_brier_ratio": report["realdata"]["brier_ratio"],
                "whittaker_log_loss_ratio": report["realdata"]["log_loss_ratio"],
                "failed_gates": failed_gates[str(report["seed"])],
            }
            for report in reports
        ],
        "failure_diagnosis": {
            "failed_gate": "gamma_coverage",
            "threshold": "0.90 <= coverage <= 0.99",
            "sensitivity_observed": reports[1]["gamma_metrics"]["coverage_95"],
            "coverage_by_covariate": {
                "Intercept": 0.890625,
                "TMG": 0.875,
            },
            "bias_by_covariate": {
                "Intercept": 0.0000803084787,
                "TMG": 0.00466838341,
            },
            "interpretation": (
                "the original 32-community calibration was sensitive to a shifted "
                "but overlapping seed window; mean, rank, MCMC, and real-data "
                "predictive gates passed, so a genuinely disjoint evaluation is required"
            ),
        },
        "stop_rule": (
            "the intended sensitivity failed but was not independent; keep the "
            "candidate unpromoted and require a preregistered disjoint evaluation"
        ),
        "artifact_status": "experimental_not_deployable",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "trait_gamma_multiseed_qualification.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "neural_hmsc_trait_gamma_qualification":
        raise ValueError(f"unsupported trait-Gamma report: {path}")
    return payload


if __name__ == "__main__":
    main()
