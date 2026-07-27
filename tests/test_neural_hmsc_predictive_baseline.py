import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyhmsc import PROMOTED_PREDICTIVE_BASELINE_ID
from pyhmsc.neural.deployment import (
    FROZEN_COMPETITOR_GATES,
    freeze_predictive_deployment_baseline,
    validate_predictive_deployment_baseline,
)


def test_freeze_baseline_pins_manifests_evidence_and_competitor_gates(
    tmp_path, monkeypatch
):
    requalification, smoke = _write_source_evidence(tmp_path)
    monkeypatch.setattr(
        "pyhmsc.neural.deployment.load_predictive_mean_ensemble",
        lambda _root, *, dataset, policy: SimpleNamespace(
            calibration_role=policy,
            dataset=dataset,
        ),
    )

    path = freeze_predictive_deployment_baseline(
        registry_root=tmp_path / "registry",
        requalification_root=requalification,
        smoke_root=smoke,
    )
    payload = validate_predictive_deployment_baseline(
        path,
        expected_baseline_id=PROMOTED_PREDICTIVE_BASELINE_ID,
    )

    assert path.parent.name == PROMOTED_PREDICTIVE_BASELINE_ID
    assert payload["default_policy"] == "affine_branch"
    assert payload["fallback_policy"] == "scale_only"
    assert payload["qualified_python_mcmc_role"] == "statistical_reference_only"
    assert tuple(payload["competitor_contract"]["frozen_gates"]) == (
        FROZEN_COMPETITOR_GATES
    )
    assert payload["competitor_contract"]["full_ensemble_mcmc_gap"][
        "big_spatial"
    ]["affine_vs_mcmc_brier_score_ratio"] == pytest.approx(1.079)

    with pytest.raises(FileExistsError, match="already exists"):
        freeze_predictive_deployment_baseline(
            registry_root=tmp_path / "registry",
            requalification_root=requalification,
            smoke_root=smoke,
        )


def test_frozen_baseline_rejects_manifest_mutation(tmp_path, monkeypatch):
    requalification, smoke = _write_source_evidence(tmp_path)
    monkeypatch.setattr(
        "pyhmsc.neural.deployment.load_predictive_mean_ensemble",
        lambda _root, *, dataset, policy: SimpleNamespace(
            calibration_role=policy,
            dataset=dataset,
        ),
    )
    path = freeze_predictive_deployment_baseline(
        registry_root=tmp_path / "registry",
        requalification_root=requalification,
        smoke_root=smoke,
    )
    manifest = path.parent / "manifests/whittaker_affine_branch_ensemble.json"
    manifest.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="baseline hash mismatch"):
        validate_predictive_deployment_baseline(path)


def test_baseline_freeze_scheduler_uses_stable_identifier_and_registry():
    text = Path(
        "docs/lumi_neural_hmsc_predictive_baseline_freeze_sbatch.sh"
    ).read_text(encoding="utf-8")

    assert 'BASELINE_ID="${BASELINE_ID:-neural_predictive_affine_v1}"' in text
    assert 'REGISTRY_ROOT="${REGISTRY_ROOT:-${USER_WORK}/hmsc-hpc-deployments}"' in text
    assert "--baseline-root \"${REGISTRY_ROOT}\"" in text
    assert "--baseline-id \"${BASELINE_ID}\"" in text


def _write_source_evidence(tmp_path):
    requalification = tmp_path / "requalification"
    manifests = requalification / "manifests"
    manifests.mkdir(parents=True)
    for dataset in ("whittaker", "big_spatial"):
        for policy in ("affine_branch", "scale_only"):
            (manifests / f"{dataset}_{policy}_ensemble.json").write_text(
                json.dumps({"dataset": dataset, "policy": policy}) + "\n",
                encoding="utf-8",
            )
    rows = []
    for dataset, brier, log_loss in (
        ("whittaker", 1.0213, 1.0327),
        ("big_spatial", 1.079, 1.0731),
    ):
        rows.append(
            {
                "dataset": dataset,
                "ensemble": "full",
                "affine_vs_mcmc_brier_score_ratio": brier,
                "affine_vs_mcmc_log_loss_ratio": log_loss,
            }
        )
    qualification = {
        "decision": "predictive_ensemble_api_requalification_passed",
        "api_requalification_passed": True,
        "all_full_and_leave_one_out_no_degradation": True,
        "full_big_spatial_genuine_proper_score_improvement": True,
        "manifest_validation_passed": True,
        "provenance_passed": True,
        "target_response_used_for_selection": False,
        "rows": rows,
    }
    (requalification / "probability_ensemble_comparison.json").write_text(
        json.dumps(qualification) + "\n", encoding="utf-8"
    )
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    smoke_result = {
        "decision": "predictive_deployment_smoke_passed",
        "all_datasets_passed": True,
        "default_policy": "affine_branch",
        "fallback_policy": "scale_only",
        "qualified_python_mcmc_role": "statistical_reference_only",
        "response_semantics": "predictive_only",
        "datasets": [
            {
                "dataset": dataset,
                "passed": True,
                "target_response_opened": False,
                "mcmc_used_for_neural_prediction": False,
                "parity_provenance_qualified": True,
            }
            for dataset in ("whittaker", "big_spatial")
        ],
    }
    (smoke / "predictive_deployment_smoke.json").write_text(
        json.dumps(smoke_result) + "\n", encoding="utf-8"
    )
    return requalification, smoke
