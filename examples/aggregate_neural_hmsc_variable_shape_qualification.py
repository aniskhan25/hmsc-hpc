#!/usr/bin/env python3
"""Aggregate frozen variable-shape qualification runs for promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-seed", type=int, default=20260730)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for root in args.runs:
        path = root.expanduser().resolve() / "variable_shape_qualification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != "neural_hmsc_variable_shape_qualification":
            raise ValueError(f"unsupported qualification report: {path}")
        base_seed = int(payload["settings"]["seed"])
        rows.append(
            {
                "base_seed": base_seed,
                "decision": payload["decision"],
                "all_gates_passed": payload["all_gates_passed"],
                "shape_range": payload["shape_range"],
                "summary": payload["summary"],
                "mcmc_summary": payload["mcmc"]["summary"],
                "checkpoint": payload["checkpoint"],
                "report_path": str(path),
                "report_sha256": _sha256(path),
            }
        )
    seeds = [row["base_seed"] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("qualification base seeds must be unique")
    if args.candidate_seed not in seeds:
        raise ValueError("predeclared candidate seed is absent")
    if len({json.dumps(row["shape_range"], sort_keys=True) for row in rows}) != 1:
        raise ValueError("qualification shape ranges differ")
    metrics = (
        "neural_coverage_95",
        "rank_mean",
        "rank_variance",
        "neural_rmse",
        "anchor_rmse",
        "neural_brier",
        "anchor_brier",
        "neural_log_loss",
        "anchor_log_loss",
    )
    aggregate = {
        metric: {
            "mean": float(np.mean([row["summary"][metric] for row in rows])),
            "min": float(np.min([row["summary"][metric] for row in rows])),
            "max": float(np.max([row["summary"][metric] for row in rows])),
        }
        for metric in metrics
    }
    aggregate["neural_to_mcmc_brier_ratio"] = {
        "mean": float(
            np.mean([row["mcmc_summary"]["neural_to_mcmc_brier_ratio"] for row in rows])
        ),
        "max": float(
            np.max([row["mcmc_summary"]["neural_to_mcmc_brier_ratio"] for row in rows])
        ),
    }
    aggregate["neural_to_mcmc_log_loss_ratio"] = {
        "mean": float(
            np.mean(
                [row["mcmc_summary"]["neural_to_mcmc_log_loss_ratio"] for row in rows]
            )
        ),
        "max": float(
            np.max(
                [row["mcmc_summary"]["neural_to_mcmc_log_loss_ratio"] for row in rows]
            )
        ),
    }
    promoted = all(
        row["decision"] == "variable_shape_probit_qualified"
        and row["all_gates_passed"] is True
        for row in rows
    )
    candidate = next(row for row in rows if row["base_seed"] == args.candidate_seed)
    result = {
        "schema_version": 1,
        "kind": "neural_hmsc_variable_shape_multiseed_qualification",
        "decision": (
            "variable_shape_probit_promoted"
            if promoted
            else "variable_shape_probit_not_promoted"
        ),
        "predeclared_candidate_seed": args.candidate_seed,
        "candidate_checkpoint": candidate["checkpoint"],
        "candidate_selected_using_sensitivity_outcomes": False,
        "fixed_release_modified": False,
        "fixed_release_id": "neural_hmsc_v0_1",
        "shape_range": rows[0]["shape_range"],
        "runs": rows,
        "aggregate": aggregate,
        "all_runs_passed": promoted,
        "claim_boundary": {
            "qualified": "variable-shape fixed-effect probit Beta approximation within the declared ranges",
            "not_claimed": [
                "joint-posterior MCMC equivalence",
                "traits or random effects",
                "normal or poisson variable-shape qualification",
                "shapes outside the declared ranges",
            ],
        },
    }
    json_path = output / "variable_shape_multiseed_qualification.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "variable_shape_multiseed_qualification.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "all_runs_passed": promoted,
                "candidate_seed": args.candidate_seed,
                "report": str(json_path),
                "report_sha256": _sha256(json_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _markdown(result):
    lines = [
        "# Variable-Shape Neural-HMSC Multi-Seed Qualification",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Seed | Passed | Coverage | Rank mean | Brier/MCMC | Log loss/MCMC |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["runs"]:
        lines.append(
            f"| {row['base_seed']} | {'yes' if row['all_gates_passed'] else 'no'} "
            f"| {row['summary']['neural_coverage_95']:.4f} "
            f"| {row['summary']['rank_mean']:.4f} "
            f"| {row['mcmc_summary']['neural_to_mcmc_brier_ratio']:.4f} "
            f"| {row['mcmc_summary']['neural_to_mcmc_log_loss_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The fixed `neural_hmsc_v0_1` release was not modified. Seed 20260730 was the predeclared variable-shape release candidate; the other runs are sensitivity evidence only.",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
