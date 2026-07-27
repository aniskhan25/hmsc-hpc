"""Versioned deployment baselines for predictive Neural-HMSC ensembles."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from pyhmsc.neural.ensemble import (
    DEFAULT_PREDICTIVE_MEAN_POLICY,
    PREDICTIVE_MEAN_POLICIES,
    PredictiveProbabilityEnsemble,
    file_sha256,
    load_predictive_mean_ensemble,
)


PREDICTIVE_BASELINE_KIND = "pyhmsc_predictive_deployment_baseline"
PREDICTIVE_BASELINE_SCHEMA_VERSION = 1
PROMOTED_PREDICTIVE_BASELINE_ID = "neural_predictive_affine_v1"
PREDICTIVE_BASELINE_DATASETS = ("whittaker", "big_spatial")
FROZEN_COMPETITOR_GATES = (
    "coefficient_sbc",
    "ood_regimes",
    "rare_validation",
    "whittaker_no_degradation",
    "big_spatial_no_degradation",
    "full_and_leave_one_out_stability",
    "manifest_and_parity_provenance",
)


def freeze_predictive_deployment_baseline(
    *,
    registry_root: str | Path,
    requalification_root: str | Path,
    smoke_root: str | Path,
    baseline_id: str = PROMOTED_PREDICTIVE_BASELINE_ID,
) -> Path:
    """Atomically freeze a qualified manifest bundle under a stable ID."""
    _validate_baseline_id(baseline_id)
    registry = Path(registry_root).expanduser().resolve()
    destination = registry / baseline_id
    if destination.exists():
        raise FileExistsError(f"predictive baseline already exists: {destination}")
    requalification = Path(requalification_root).expanduser().resolve()
    smoke = Path(smoke_root).expanduser().resolve()
    requalification_json = requalification / "probability_ensemble_comparison.json"
    smoke_json = smoke / "predictive_deployment_smoke.json"
    qualification = json.loads(requalification_json.read_text(encoding="utf-8"))
    smoke_result = json.loads(smoke_json.read_text(encoding="utf-8"))
    _validate_qualification_evidence(qualification, smoke_result)

    source_manifests: dict[str, dict[str, Path]] = {}
    for dataset in PREDICTIVE_BASELINE_DATASETS:
        source_manifests[dataset] = {}
        for policy in PREDICTIVE_MEAN_POLICIES:
            source = requalification / "manifests" / (
                f"{dataset}_{policy}_ensemble.json"
            )
            ensemble = load_predictive_mean_ensemble(
                source.parent,
                dataset=dataset,
                policy=policy,
            )
            if ensemble.calibration_role != policy:
                raise ValueError("source deployment manifest role mismatch")
            source_manifests[dataset][policy] = source

    registry.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{baseline_id}.", dir=registry))
    try:
        manifests_dir = staging / "manifests"
        evidence_dir = staging / "evidence"
        manifests_dir.mkdir()
        evidence_dir.mkdir()
        manifest_records: dict[str, dict[str, dict[str, str]]] = {}
        for dataset, policies in source_manifests.items():
            manifest_records[dataset] = {}
            for policy, source in policies.items():
                target = manifests_dir / source.name
                shutil.copy2(source, target)
                manifest_records[dataset][policy] = {
                    "path": str(target.relative_to(staging)),
                    "sha256": file_sha256(target),
                }
        qualification_target = evidence_dir / "api_requalification.json"
        smoke_target = evidence_dir / "default_wiring_smoke.json"
        shutil.copy2(requalification_json, qualification_target)
        shutil.copy2(smoke_json, smoke_target)
        payload = {
            "schema_version": PREDICTIVE_BASELINE_SCHEMA_VERSION,
            "kind": PREDICTIVE_BASELINE_KIND,
            "baseline_id": baseline_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact_role": "predictive_only_deployment_baseline",
            "default_policy": DEFAULT_PREDICTIVE_MEAN_POLICY,
            "fallback_policy": "scale_only",
            "qualified_python_mcmc_role": "statistical_reference_only",
            "datasets": manifest_records,
            "evidence": {
                "api_requalification": {
                    "path": str(qualification_target.relative_to(staging)),
                    "sha256": file_sha256(qualification_target),
                    "required_decision": "predictive_ensemble_api_requalification_passed",
                },
                "default_wiring_smoke": {
                    "path": str(smoke_target.relative_to(staging)),
                    "sha256": file_sha256(smoke_target),
                    "required_decision": "predictive_deployment_smoke_passed",
                },
            },
            "competitor_contract": {
                "baseline_id": baseline_id,
                "frozen_gates": list(FROZEN_COMPETITOR_GATES),
                "target": "reduce_big_spatial_gap_to_qualified_python_mcmc",
                "full_ensemble_mcmc_gap": _full_mcmc_gap(qualification),
                "candidate_must_report_against_exact_manifest_hashes": True,
            },
            "source_provenance": {
                "requalification_root": str(requalification),
                "smoke_root": str(smoke),
            },
        }
        bundle_path = staging / "baseline.json"
        bundle_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_predictive_deployment_baseline(bundle_path)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "baseline.json"


def validate_predictive_deployment_baseline(
    baseline_path: str | Path,
    *,
    expected_baseline_id: str | None = None,
) -> dict[str, Any]:
    """Validate bundle structure, hashes, qualification evidence, and gates."""
    path = _baseline_json_path(baseline_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != PREDICTIVE_BASELINE_KIND:
        raise ValueError("unsupported predictive deployment baseline kind")
    if int(payload.get("schema_version", -1)) != PREDICTIVE_BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported predictive deployment baseline schema")
    baseline_id = str(payload.get("baseline_id", ""))
    _validate_baseline_id(baseline_id)
    if expected_baseline_id is not None and baseline_id != expected_baseline_id:
        raise ValueError("predictive deployment baseline identifier differs")
    if payload.get("default_policy") != DEFAULT_PREDICTIVE_MEAN_POLICY:
        raise ValueError("predictive deployment baseline default policy differs")
    if payload.get("fallback_policy") != "scale_only":
        raise ValueError("predictive deployment baseline fallback policy differs")
    if payload.get("qualified_python_mcmc_role") != "statistical_reference_only":
        raise ValueError("predictive deployment baseline MCMC role differs")

    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(
        PREDICTIVE_BASELINE_DATASETS
    ):
        raise ValueError("predictive deployment baseline datasets differ")
    for dataset in PREDICTIVE_BASELINE_DATASETS:
        policies = datasets[dataset]
        if not isinstance(policies, dict) or set(policies) != set(
            PREDICTIVE_MEAN_POLICIES
        ):
            raise ValueError(f"{dataset} predictive baseline policies differ")
        for policy in PREDICTIVE_MEAN_POLICIES:
            _validate_record_hash(path.parent, policies[policy])

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("predictive deployment baseline lacks evidence")
    qualification_path = _validate_record_hash(
        path.parent, evidence.get("api_requalification")
    )
    smoke_path = _validate_record_hash(
        path.parent, evidence.get("default_wiring_smoke")
    )
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    smoke_result = json.loads(smoke_path.read_text(encoding="utf-8"))
    _validate_qualification_evidence(qualification, smoke_result)
    contract = payload.get("competitor_contract")
    if not isinstance(contract, dict):
        raise ValueError("predictive deployment baseline lacks competitor contract")
    if contract.get("baseline_id") != baseline_id:
        raise ValueError("competitor contract baseline identifier differs")
    if tuple(contract.get("frozen_gates", ())) != FROZEN_COMPETITOR_GATES:
        raise ValueError("competitor contract frozen gates differ")
    if contract.get("full_ensemble_mcmc_gap") != _full_mcmc_gap(qualification):
        raise ValueError("competitor contract MCMC gap differs from evidence")
    return payload


def load_predictive_deployment_baseline(
    registry_root: str | Path,
    *,
    dataset: str,
    baseline_id: str = PROMOTED_PREDICTIVE_BASELINE_ID,
    policy: str | None = None,
) -> PredictiveProbabilityEnsemble:
    """Resolve a stable baseline identifier and load its selected ensemble."""
    selected_policy = DEFAULT_PREDICTIVE_MEAN_POLICY if policy is None else policy
    if dataset not in PREDICTIVE_BASELINE_DATASETS:
        raise ValueError(f"unsupported predictive baseline dataset: {dataset!r}")
    if selected_policy not in PREDICTIVE_MEAN_POLICIES:
        raise ValueError(f"unsupported predictive baseline policy: {selected_policy!r}")
    baseline_dir = Path(registry_root).expanduser().resolve() / baseline_id
    payload = validate_predictive_deployment_baseline(
        baseline_dir,
        expected_baseline_id=baseline_id,
    )
    record = payload["datasets"][dataset][selected_policy]
    expected_path = (baseline_dir / record["path"]).resolve()
    conventional_path = baseline_dir / "manifests" / (
        f"{dataset}_{selected_policy}_ensemble.json"
    )
    if expected_path != conventional_path.resolve():
        raise ValueError("predictive baseline manifest location is not conventional")
    return load_predictive_mean_ensemble(
        conventional_path.parent,
        dataset=dataset,
        policy=selected_policy,
    )


def _validate_qualification_evidence(
    qualification: dict[str, Any],
    smoke: dict[str, Any],
) -> None:
    required_qualification = {
        "decision": "predictive_ensemble_api_requalification_passed",
        "api_requalification_passed": True,
        "all_full_and_leave_one_out_no_degradation": True,
        "full_big_spatial_genuine_proper_score_improvement": True,
        "manifest_validation_passed": True,
        "provenance_passed": True,
        "target_response_used_for_selection": False,
    }
    for key, expected in required_qualification.items():
        if qualification.get(key) != expected:
            raise ValueError(f"requalification evidence failed {key}")
    required_smoke = {
        "decision": "predictive_deployment_smoke_passed",
        "all_datasets_passed": True,
        "default_policy": DEFAULT_PREDICTIVE_MEAN_POLICY,
        "fallback_policy": "scale_only",
        "qualified_python_mcmc_role": "statistical_reference_only",
        "response_semantics": "predictive_only",
    }
    for key, expected in required_smoke.items():
        if smoke.get(key) != expected:
            raise ValueError(f"deployment smoke evidence failed {key}")
    rows = smoke.get("datasets")
    if not isinstance(rows, list) or {row.get("dataset") for row in rows} != set(
        PREDICTIVE_BASELINE_DATASETS
    ):
        raise ValueError("deployment smoke dataset evidence differs")
    if any(
        not row.get("passed", False)
        or row.get("target_response_opened") is not False
        or row.get("mcmc_used_for_neural_prediction") is not False
        or not row.get("parity_provenance_qualified", False)
        for row in rows
    ):
        raise ValueError("deployment smoke dataset gate failed")


def _full_mcmc_gap(qualification: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = qualification.get("rows")
    if not isinstance(rows, list):
        raise ValueError("requalification evidence lacks rows")
    result = {}
    for dataset in PREDICTIVE_BASELINE_DATASETS:
        matches = [
            row
            for row in rows
            if row.get("dataset") == dataset and row.get("ensemble") == "full"
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one full {dataset} requalification row")
        row = matches[0]
        result[dataset] = {
            "affine_vs_mcmc_brier_score_ratio": float(
                row["affine_vs_mcmc_brier_score_ratio"]
            ),
            "affine_vs_mcmc_log_loss_ratio": float(
                row["affine_vs_mcmc_log_loss_ratio"]
            ),
        }
    return result


def _validate_record_hash(root: Path, record: Any) -> Path:
    if not isinstance(record, dict) or not record.get("path") or not record.get(
        "sha256"
    ):
        raise ValueError("predictive baseline file record is incomplete")
    path = (root / str(record["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("predictive baseline file escapes bundle root") from exc
    observed = file_sha256(path)
    if observed != str(record["sha256"]):
        raise ValueError(f"predictive baseline hash mismatch for {path}")
    return path


def _baseline_json_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    return resolved / "baseline.json" if resolved.is_dir() else resolved


def _validate_baseline_id(value: str) -> None:
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value
    ):
        raise ValueError(
            "baseline_id must contain only lowercase letters, digits, or '_'"
        )
