import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyhmsc import NEURAL_HMSC_RELEASE_ID, load_neural_hmsc_release
from pyhmsc.neural.release import (
    NEURAL_HMSC_RELEASE_DATASETS,
    NEURAL_HMSC_RELEASE_SEEDS,
    freeze_neural_hmsc_release,
    validate_neural_hmsc_release,
)


def test_freeze_release_is_complete_immutable_and_loadable(tmp_path, monkeypatch):
    packaged, baseline, audit = _write_release_sources(tmp_path)
    _stub_runtime_validation(monkeypatch, baseline)

    path = freeze_neural_hmsc_release(
        registry_root=tmp_path / "registry",
        packaged_members_root=packaged,
        predictive_baseline_root=baseline,
        audit_root=audit,
    )
    payload = validate_neural_hmsc_release(
        path, expected_release_id=NEURAL_HMSC_RELEASE_ID
    )
    release = load_neural_hmsc_release(tmp_path / "registry")

    assert path.parent.name == NEURAL_HMSC_RELEASE_ID
    assert payload["release_status"] == "release_ready"
    assert payload["default_predictive_policy"] == "affine_branch"
    assert payload["fallback_predictive_policy"] == "scale_only"
    assert (
        len(payload["inventory"])
        == len([file for file in path.parent.rglob("*") if file.is_file()]) - 1
    )
    assert release.seeds == NEURAL_HMSC_RELEASE_SEEDS
    assert release.load_checkpoint().distribution == "probit"
    assert (
        release.load_predictive_ensemble(dataset="big_spatial").calibration_role
        == "affine_branch"
    )
    local_manifest = json.loads(
        (
            path.parent / "predictive/manifests/whittaker_scale_only_ensemble.json"
        ).read_text(encoding="utf-8")
    )
    assert all(not Path(row["path"]).is_absolute() for row in local_manifest["members"])

    with pytest.raises(FileExistsError, match="already exists"):
        freeze_neural_hmsc_release(
            registry_root=tmp_path / "registry",
            packaged_members_root=packaged,
            predictive_baseline_root=baseline,
            audit_root=audit,
        )


def test_release_validation_rejects_member_mutation(tmp_path, monkeypatch):
    packaged, baseline, audit = _write_release_sources(tmp_path)
    _stub_runtime_validation(monkeypatch, baseline)
    path = freeze_neural_hmsc_release(
        registry_root=tmp_path / "registry",
        packaged_members_root=packaged,
        predictive_baseline_root=baseline,
        audit_root=audit,
    )
    member = (
        path.parent
        / "checkpoints/20260721/neural_predictive_distribution_scale_only.h5"
    )
    member.write_bytes(b"changed")

    with pytest.raises(ValueError, match="release hash mismatch"):
        validate_neural_hmsc_release(path)


def test_release_validation_rejects_extra_file(tmp_path, monkeypatch):
    packaged, baseline, audit = _write_release_sources(tmp_path)
    _stub_runtime_validation(monkeypatch, baseline)
    path = freeze_neural_hmsc_release(
        registry_root=tmp_path / "registry",
        packaged_members_root=packaged,
        predictive_baseline_root=baseline,
        audit_root=audit,
    )
    (path.parent / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory file set differs"):
        validate_neural_hmsc_release(path)


def _write_release_sources(tmp_path):
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    package_rows = []
    member_hashes = {}
    filenames = {
        ("whittaker", "affine_branch"): "neural_predictive_distribution.h5",
        ("big_spatial", "affine_branch"): (
            "big_spatial_neural_predictive_distribution.h5"
        ),
        ("whittaker", "scale_only"): ("neural_predictive_distribution_scale_only.h5"),
        ("big_spatial", "scale_only"): (
            "big_spatial_neural_predictive_distribution_scale_only.h5"
        ),
    }
    for seed in NEURAL_HMSC_RELEASE_SEEDS:
        member = packaged / str(seed)
        checkpoint = member / "neural_checkpoint"
        checkpoint.mkdir(parents=True)
        for name in (
            "neural_checkpoint.json",
            "weights.weights.h5",
            "coefficient_calibration.json",
        ):
            (checkpoint / name).write_bytes(f"{seed}:{name}".encode())
        artifacts = []
        for (dataset, policy), name in filenames.items():
            file = member / name
            file.write_bytes(f"{seed}:{dataset}:{policy}".encode())
            digest = _sha256(file)
            member_hashes[(dataset, policy, seed)] = digest
            artifacts.append(
                {
                    "dataset": dataset,
                    "policy": policy,
                    "name": name,
                    "source_sha256": digest,
                    "packaged_sha256": digest,
                }
            )
        package_rows.append(
            {
                "seed": seed,
                "checkpoint_version": "0.4",
                "distribution": "probit",
                "calibration_method": "external_context_monotone_scale",
                "checkpoint_manifest_sha256": _sha256(
                    checkpoint / "neural_checkpoint.json"
                ),
                "calibration_artifact_sha256": _sha256(
                    checkpoint / "coefficient_calibration.json"
                ),
                "predictive_artifacts": artifacts,
            }
        )
    (packaged / "package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "neural_hmsc_v0_1_packaged_checkpoint_bundle",
                "release_scope": "fixed_shape_fixed_effect_probit_beta",
                "seeds": list(NEURAL_HMSC_RELEASE_SEEDS),
                "model_fitting_performed": False,
                "calibration_selection_performed": False,
                "all_weights_unchanged": True,
                "members": package_rows,
            }
        ),
        encoding="utf-8",
    )

    baseline = tmp_path / "baseline"
    manifests = baseline / "manifests"
    manifests.mkdir(parents=True)
    baseline_records = {}
    for dataset in NEURAL_HMSC_RELEASE_DATASETS:
        baseline_records[dataset] = {}
        for policy in ("affine_branch", "scale_only"):
            source = manifests / f"{dataset}_{policy}_ensemble.json"
            rows = [
                {
                    "path": f"/qualified/source/{dataset}/{policy}/{seed}.h5",
                    "sha256": member_hashes[(dataset, policy, seed)],
                    "seed": seed,
                    "calibration_role": policy,
                    "provenance": {
                        "reference_parity_qualified": True,
                        "dataset_acceptance_passed": True,
                        "acceptance_path": "/qualified/acceptance.json",
                        "acceptance_sha256": "a",
                        "run_metadata_path": "/qualified/run.json",
                        "run_metadata_sha256": "b",
                        "parity_metrics_path": "/qualified/parity.json",
                        "parity_metrics_sha256": "c",
                        "mcmc_reference_path": "/qualified/mcmc.h5",
                        "mcmc_reference_sha256": "d",
                    },
                }
                for seed in NEURAL_HMSC_RELEASE_SEEDS
            ]
            source.write_text(
                json.dumps(
                    {
                        "kind": "pyhmsc_predictive_probability_ensemble",
                        "schema_version": 1,
                        "aggregation": "arithmetic_mean_response_probability",
                        "calibration_role": policy,
                        "members": rows,
                        "provenance": {
                            "dataset": dataset,
                            "response_semantics": "predictive_only",
                            "selection_outcomes_used": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_records[dataset][policy] = {
                "path": f"manifests/{source.name}",
                "sha256": _sha256(source),
            }
    baseline_payload = {
        "baseline_id": "neural_predictive_affine_v1",
        "datasets": baseline_records,
    }
    (baseline / "baseline.json").write_text(
        json.dumps(baseline_payload), encoding="utf-8"
    )

    audit = tmp_path / "audit"
    audit.mkdir()
    audit_payload = {
        "kind": "neural_hmsc_v0_1_release_readiness_audit",
        "decision": "release_ready",
        "release_blockers": [],
        "gates": {"qualified": True},
        "release_required_gates": ["qualified"],
        "model_fitting_performed": False,
        "calibration_selection_performed": False,
        "support_matrix": [
            {
                "capability": "fixed probit",
                "implementation": "public",
                "qualification": "release-qualified",
            }
        ],
        "claim_boundary": {"qualified_claim": "bounded approximation"},
        "release_scope_exclusions": {"normal": "experimental"},
    }
    (audit / "neural_hmsc_v0_1_release_audit.json").write_text(
        json.dumps(audit_payload), encoding="utf-8"
    )
    (audit / "neural_hmsc_v0_1_release_audit.md").write_text(
        "# Release audit\n", encoding="utf-8"
    )
    return packaged, baseline, audit


def _stub_runtime_validation(monkeypatch, baseline):
    baseline_payload = json.loads(
        (baseline / "baseline.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        "pyhmsc.neural.release.validate_predictive_deployment_baseline",
        lambda *_args, **_kwargs: baseline_payload,
    )
    monkeypatch.setattr(
        "pyhmsc.neural.release.NeuralHmscInference.load",
        lambda _path: SimpleNamespace(
            distribution="probit", coefficient_calibration=object()
        ),
    )

    def fake_ensemble(path, verify_hashes=False):
        del verify_hashes
        name = Path(path).stem
        dataset = "big_spatial" if name.startswith("big_spatial") else "whittaker"
        policy = "scale_only" if "scale_only" in name else "affine_branch"
        return SimpleNamespace(
            calibration_role=policy,
            provenance={
                "dataset": dataset,
                "response_semantics": "predictive_only",
                "selection_outcomes_used": False,
            },
            parity_provenance_qualified=True,
            qualified_mcmc_reference={},
        )

    monkeypatch.setattr(
        "pyhmsc.neural.release.PredictiveProbabilityEnsemble.from_manifest",
        fake_ensemble,
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
