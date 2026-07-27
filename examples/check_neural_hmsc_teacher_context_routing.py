"""Check MCMC-teacher context routing without opening real-data outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.ensemble import (  # noqa: E402
    ENSEMBLE_AGGREGATION,
    ENSEMBLE_KIND,
    PredictiveProbabilityEnsemble,
    file_sha256,
)
from pyhmsc.neural.teacher_residual import (  # noqa: E402
    McmcTeacherResidualHead,
    response_context_summary,
)


DATASETS = ("whittaker", "big_spatial")
MEMBER_FILENAMES = {
    "whittaker": "neural_predictive_distribution.h5",
    "big_spatial": "big_spatial_neural_predictive_distribution.h5",
}
EXPECTED_ROUTES = {
    "whittaker": "identity",
    "big_spatial": "approved_context",
}
IDENTITY_TOLERANCE = 1.0e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-id", default="neural_predictive_affine_v1")
    parser.add_argument("--member-root", type=Path, required=True)
    parser.add_argument("--teacher-head", type=Path, required=True)
    parser.add_argument("--frozen-run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_context_routing_check(
        baseline_root=args.baseline_root,
        baseline_id=args.baseline_id,
        member_root=args.member_root,
        teacher_head=args.teacher_head,
        frozen_run_root=args.frozen_run_root,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "teacher_context_routing.json"
    csv_path = args.output / "teacher_context_routing.csv"
    report_path = args.output / "teacher_context_routing.md"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(_routing_csv_rows(result)).to_csv(csv_path, index=False)
    report = render_context_routing_report(result)
    report_path.write_text(report, encoding="utf-8")
    print(report)


def run_context_routing_check(
    *,
    baseline_root: Path,
    baseline_id: str,
    member_root: Path,
    teacher_head: Path,
    frozen_run_root: Path,
    seed: int,
) -> dict[str, Any]:
    """Route frozen predictions using covariates and probabilities only."""
    baseline_path = (baseline_root / baseline_id / "baseline.json").resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("kind") != "pyhmsc_predictive_deployment_baseline":
        raise ValueError("unsupported predictive deployment baseline")
    if baseline.get("baseline_id") != baseline_id:
        raise ValueError("predictive deployment baseline identifier differs")
    if baseline.get("default_policy") != "affine_branch":
        raise ValueError("teacher routing requires the frozen affine baseline")

    head = McmcTeacherResidualHead.load(teacher_head)
    head_metadata = head.to_metadata()
    if head.baseline_id != baseline_id:
        raise ValueError("teacher head baseline identifier differs")
    if not head.selected or head.context_gate is None:
        raise ValueError("teacher head lacks a selected context gate")
    if head_metadata["real_outcomes_used_for_training_or_selection"] is not False:
        raise ValueError("teacher head used real outcomes")

    rows = []
    for dataset in DATASETS:
        manifest_entry = baseline["datasets"][dataset]["affine_branch"]
        manifest_path = (
            baseline_path.parent / str(manifest_entry["path"])
        ).resolve()
        manifest_hash = file_sha256(manifest_path)
        if manifest_hash != manifest_entry["sha256"]:
            raise ValueError(f"{dataset} frozen manifest hash differs")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest_contract(manifest, dataset=dataset)

        member_rows = manifest["members"]
        member_paths = [
            (member_root / str(int(row["seed"])) / MEMBER_FILENAMES[dataset]).resolve()
            for row in member_rows
        ]
        for row, path in zip(member_rows, member_paths):
            observed = file_sha256(path)
            if observed != row["sha256"]:
                raise ValueError(
                    f"{dataset} local member hash differs for seed {row['seed']}"
                )
        ensemble = PredictiveProbabilityEnsemble.create(
            member_paths,
            seeds=[int(row["seed"]) for row in member_rows],
            calibration_role="affine_branch",
            provenance={
                "dataset": dataset,
                "source_manifest_sha256": manifest_hash,
                "response_semantics": "predictive_only",
            },
        )
        if ensemble.compatibility != manifest["compatibility"]:
            raise ValueError(f"{dataset} local member compatibility differs")

        X_path = _dataset_X_path(frozen_run_root, seed=seed, dataset=dataset)
        _validate_covariate_only_path(X_path)
        X = pd.read_csv(X_path, index_col=0)
        prediction = ensemble.predict_mean(X)
        probability = prediction.to_numpy(dtype=float)
        design = _context_design(X, manifest["compatibility"])
        summary = response_context_summary(
            probability,
            design,
            representation_version=head.representation_version,
        )
        routing = head.context_gate.decision(probability, design)
        candidate = head.predict_mean(probability, design)
        max_delta = float(np.max(np.abs(candidate - probability)))

        expected_route = EXPECTED_ROUTES[dataset]
        route_passed = _route_requirement_passed(
            expected_route,
            routing,
            approved_labels=head.context_gate.approved_labels,
            max_delta=max_delta,
        )
        rows.append(
            {
                "dataset": dataset,
                "expected_route": expected_route,
                "route_passed": route_passed,
                "X_path": str(X_path),
                "X_sha256": file_sha256(X_path),
                "prediction_shape": list(probability.shape),
                "probability_min": float(np.min(probability)),
                "probability_max": float(np.max(probability)),
                "context_summary": dict(
                    zip(
                        head.context_gate.feature_names,
                        summary.astype(float).tolist(),
                    )
                ),
                "routing": routing,
                "candidate_max_abs_delta": max_delta,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_hash,
                "ordered_seeds": list(ensemble.seeds),
                "ordered_member_hashes": [member.sha256 for member in ensemble.members],
                "member_hashes_match_frozen_manifest": True,
                "compatibility_matches_frozen_manifest": True,
                "manifest_parity_provenance_qualified": (
                    _manifest_parity_provenance_qualified(manifest)
                ),
                "target_response_opened": False,
                "proper_scores_computed": False,
            }
        )

    all_provenance_qualified = all(
        row["manifest_parity_provenance_qualified"] for row in rows
    )
    passed = bool(all(row["route_passed"] for row in rows) and all_provenance_qualified)
    return {
        "decision": (
            "teacher_context_routing_passed"
            if passed
            else "teacher_context_routing_failed_closed"
        ),
        "passed": passed,
        "baseline_id": baseline_id,
        "baseline_path": str(baseline_path),
        "baseline_sha256": file_sha256(baseline_path),
        "teacher_head_path": str(Path(teacher_head).resolve()),
        "teacher_head_metadata_sha256": file_sha256(
            Path(teacher_head).resolve() / "teacher_residual.json"
            if Path(teacher_head).is_dir()
            else Path(teacher_head).resolve()
        ),
        "teacher_selected_shrinkage": head.selected_shrinkage,
        "approved_context_labels": list(head.context_gate.approved_labels),
        "identity_tolerance": IDENTITY_TOLERANCE,
        "datasets": rows,
        "all_manifest_parity_provenance_qualified": all_provenance_qualified,
        "input_contract": {
            "allowed_real_data_input": "covariate X.csv only",
            "ensemble_probabilities_used": True,
            "design_summaries_used": True,
            "prototype_distances_used": True,
            "support_caps_used": True,
            "target_responses_opened": False,
            "proper_scores_computed": False,
            "mcmc_predictions_opened": False,
        },
    }


def _validate_manifest_contract(manifest: dict[str, Any], *, dataset: str) -> None:
    if manifest.get("kind") != ENSEMBLE_KIND:
        raise ValueError(f"{dataset} manifest kind differs")
    if manifest.get("aggregation") != ENSEMBLE_AGGREGATION:
        raise ValueError(f"{dataset} manifest aggregation differs")
    if manifest.get("ordered_members") is not True:
        raise ValueError(f"{dataset} manifest member order is not frozen")
    if manifest.get("calibration_role") != "affine_branch":
        raise ValueError(f"{dataset} manifest is not affine_branch")
    provenance = manifest.get("provenance", {})
    if provenance.get("dataset") != dataset:
        raise ValueError(f"{dataset} manifest dataset provenance differs")
    if provenance.get("response_semantics") != "predictive_only":
        raise ValueError(f"{dataset} manifest is not predictive-only")
    if provenance.get("selection_outcomes_used") is not False:
        raise ValueError(f"{dataset} manifest selection used outcomes")


def _manifest_parity_provenance_qualified(manifest: dict[str, Any]) -> bool:
    required_files = ("acceptance", "run_metadata", "parity_metrics", "mcmc_reference")
    for member in manifest.get("members", []):
        provenance = member.get("provenance", {})
        if not (
            provenance.get("reference_parity_qualified", False)
            and provenance.get("dataset_acceptance_passed", False)
        ):
            return False
        if any(
            not provenance.get(f"{name}_path")
            or not provenance.get(f"{name}_sha256")
            for name in required_files
        ):
            return False
    return True


def _dataset_X_path(frozen_run_root: Path, *, seed: int, dataset: str) -> Path:
    root = Path(frozen_run_root) / f"seed_{int(seed)}"
    if dataset == "whittaker":
        relative = Path("whittaker/whittaker_holdout/data/test/X.csv")
    elif dataset == "big_spatial":
        relative = Path("big_spatial/big_spatial_transfer_project/data/test/X.csv")
    else:
        raise ValueError(f"unsupported routing dataset: {dataset}")
    return (root / relative).resolve()


def _validate_covariate_only_path(path: Path) -> None:
    if path.name != "X.csv" or path.parent.name != "test":
        raise ValueError("routing input must be a held-out covariate X.csv")
    if not path.is_file():
        raise FileNotFoundError(f"routing covariate input does not exist: {path}")


def _context_design(X: pd.DataFrame, compatibility: dict[str, Any]) -> np.ndarray:
    expected = [
        str(value)
        for value in compatibility["covariates"]
        if str(value) != "Intercept"
    ]
    missing = [column for column in expected if column not in X.columns]
    if missing:
        raise ValueError(f"routing covariates are missing: {missing}")
    parts = [np.ones((len(X), 1), dtype=float)]
    if expected:
        parts.append(X.loc[:, expected].to_numpy(dtype=float))
    return np.column_stack(parts)


def _route_requirement_passed(
    expected_route: str,
    routing: dict[str, Any],
    *,
    approved_labels: tuple[str, ...],
    max_delta: float,
) -> bool:
    if expected_route == "identity":
        return bool(not routing["active"] and max_delta <= IDENTITY_TOLERANCE)
    if expected_route != "approved_context":
        raise ValueError(f"unsupported expected route: {expected_route}")
    return bool(
        routing["active"]
        and routing["selected_label"] in approved_labels
        and routing["within_approved_distance_cap"]
        and max_delta > IDENTITY_TOLERANCE
    )


def _routing_csv_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in result["datasets"]:
        routing = dataset["routing"]
        rows.append(
            {
                "dataset": dataset["dataset"],
                "expected_route": dataset["expected_route"],
                "route_passed": dataset["route_passed"],
                "active": routing["active"],
                "selected_label": routing["selected_label"],
                "approved_label": routing["approved_label"],
                "approved_distance": routing["approved_distance"],
                "fallback_label": routing["fallback_label"],
                "fallback_distance": routing["fallback_distance"],
                "distance_margin": routing["distance_margin"],
                "approved_distance_cap": routing["approved_distance_cap"],
                "within_approved_distance_cap": routing[
                    "within_approved_distance_cap"
                ],
                "candidate_max_abs_delta": dataset["candidate_max_abs_delta"],
            }
        )
    return rows


def render_context_routing_report(result: dict[str, Any]) -> str:
    lines = [
        "# Outcome-blind MCMC-teacher context routing",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "The check opened frozen affine ensemble members and held-out `X.csv` "
        "covariates only. It did not open target responses, MCMC predictions, or "
        "compute proper scores.",
        "",
        "| Dataset | Required | Selected | Active | Approved distance / cap | "
        "Max movement | Pass |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["datasets"]:
        routing = row["routing"]
        lines.append(
            f"| {row['dataset']} | {row['expected_route']} | "
            f"{routing['selected_label']} | {routing['active']} | "
            f"{routing['approved_distance']:.4f} / "
            f"{routing['approved_distance_cap']:.4f} | "
            f"{row['candidate_max_abs_delta']:.3e} | {row['route_passed']} |"
        )
    lines.extend(
        [
            "",
            "## Outcome-blind diagnostics",
            "",
        ]
    )
    for row in result["datasets"]:
        routing = row["routing"]
        design_information_key = next(
            key for key in row["context_summary"] if "information" in key
        )
        design_information = row["context_summary"][design_information_key]
        lines.extend(
            [
                f"- `{row['dataset']}`: nearest approved "
                f"`{routing['approved_label']}` at {routing['approved_distance']:.4f}; "
                f"nearest fallback `{routing['fallback_label']}` at "
                f"{routing['fallback_distance']:.4f}; mean log design information "
                f"{design_information:.4f}.",
            ]
        )
    lines.extend(
        [
            "",
            "Real-data scoring remains blocked unless both routing requirements pass.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
