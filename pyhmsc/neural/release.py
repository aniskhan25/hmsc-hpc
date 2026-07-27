"""Immutable release bundles for qualified Neural-HMSC artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from pyhmsc.neural.deployment import (
    PROMOTED_PREDICTIVE_BASELINE_ID,
    validate_predictive_deployment_baseline,
)
from pyhmsc.neural.ensemble import (
    DEFAULT_PREDICTIVE_MEAN_POLICY,
    PREDICTIVE_MEAN_POLICIES,
    PredictiveProbabilityEnsemble,
    file_sha256,
)
from pyhmsc.neural.inference import NeuralHmscInference


NEURAL_HMSC_RELEASE_ID = "neural_hmsc_v0_1"
NEURAL_HMSC_RELEASE_KIND = "pyhmsc_neural_hmsc_release"
NEURAL_HMSC_RELEASE_SCHEMA_VERSION = 1
NEURAL_HMSC_RELEASE_SEEDS = (20260721, 20260722, 20260723)
NEURAL_HMSC_RELEASE_DATASETS = ("whittaker", "big_spatial")
NEURAL_HMSC_RELEASE_SCOPE = "fixed_shape_fixed_effect_probit_beta"

_PREDICTIVE_FILENAMES = {
    ("whittaker", "affine_branch"): "neural_predictive_distribution.h5",
    ("big_spatial", "affine_branch"): ("big_spatial_neural_predictive_distribution.h5"),
    ("whittaker", "scale_only"): ("neural_predictive_distribution_scale_only.h5"),
    ("big_spatial", "scale_only"): (
        "big_spatial_neural_predictive_distribution_scale_only.h5"
    ),
}


@dataclass(frozen=True)
class NeuralHmscRelease:
    """A validated, self-contained Neural-HMSC release registry entry."""

    path: Path
    manifest: dict[str, Any]

    @classmethod
    def load(
        cls,
        registry_root: str | Path,
        *,
        release_id: str = NEURAL_HMSC_RELEASE_ID,
    ) -> "NeuralHmscRelease":
        root = Path(registry_root).expanduser().resolve() / release_id
        manifest = validate_neural_hmsc_release(root, expected_release_id=release_id)
        return cls(path=root, manifest=manifest)

    @property
    def release_id(self) -> str:
        return str(self.manifest["release_id"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.manifest["checkpoint_seeds"])

    def load_checkpoint(self, seed: int | None = None) -> NeuralHmscInference:
        """Load one calibrated coefficient-posterior checkpoint."""
        selected = self.seeds[0] if seed is None else int(seed)
        if selected not in self.seeds:
            raise ValueError(
                f"release checkpoint seed must be one of {self.seeds}, got {selected}"
            )
        checkpoint = self.path / "checkpoints" / str(selected) / "neural_checkpoint"
        return NeuralHmscInference.load(checkpoint)

    def load_predictive_ensemble(
        self,
        *,
        dataset: str,
        policy: str | None = None,
    ) -> PredictiveProbabilityEnsemble:
        """Load a release-local predictive-only probability ensemble."""
        selected_policy = (
            self.manifest["default_predictive_policy"]
            if policy is None
            else str(policy)
        )
        if dataset not in NEURAL_HMSC_RELEASE_DATASETS:
            raise ValueError(f"unsupported release dataset: {dataset!r}")
        if selected_policy not in PREDICTIVE_MEAN_POLICIES:
            raise ValueError(f"unsupported predictive policy: {selected_policy!r}")
        path = (
            self.path
            / "predictive"
            / "manifests"
            / f"{dataset}_{selected_policy}_ensemble.json"
        )
        # Whole-bundle validation has already checked every member hash. Disabling
        # per-manifest provenance-file I/O keeps the release portable while the
        # original qualified provenance paths and hashes remain frozen verbatim.
        ensemble = PredictiveProbabilityEnsemble.from_manifest(
            path, verify_hashes=False
        )
        _validate_loaded_ensemble(ensemble, dataset=dataset, policy=selected_policy)
        return ensemble


def freeze_neural_hmsc_release(
    *,
    registry_root: str | Path,
    packaged_members_root: str | Path,
    predictive_baseline_root: str | Path,
    audit_root: str | Path,
    release_id: str = NEURAL_HMSC_RELEASE_ID,
) -> Path:
    """Atomically freeze all qualified v0.1 artifacts under a stable ID."""
    _validate_release_id(release_id)
    registry = Path(registry_root).expanduser().resolve()
    destination = registry / release_id
    if destination.exists():
        raise FileExistsError(f"Neural-HMSC release already exists: {destination}")

    packaged = Path(packaged_members_root).expanduser().resolve()
    predictive = Path(predictive_baseline_root).expanduser().resolve()
    audit = Path(audit_root).expanduser().resolve()
    package_manifest = _load_and_validate_package_manifest(packaged)
    predictive_manifest = validate_predictive_deployment_baseline(
        predictive,
        expected_baseline_id=PROMOTED_PREDICTIVE_BASELINE_ID,
    )
    audit_json = audit / "neural_hmsc_v0_1_release_audit.json"
    audit_markdown = audit / "neural_hmsc_v0_1_release_audit.md"
    audit_payload = _load_and_validate_audit(audit_json)
    if not audit_markdown.is_file():
        raise FileNotFoundError(f"release audit Markdown not found: {audit_markdown}")

    registry.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=registry))
    try:
        shutil.copytree(packaged, staging / "checkpoints")
        shutil.copytree(predictive, staging / "predictive" / "source_baseline")
        evidence = staging / "evidence"
        evidence.mkdir()
        shutil.copy2(audit_json, evidence / audit_json.name)
        shutil.copy2(audit_markdown, evidence / audit_markdown.name)
        (evidence / "support_matrix.json").write_text(
            json.dumps(audit_payload["support_matrix"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        localized = _localize_predictive_manifests(
            staging,
            package_manifest=package_manifest,
            predictive_manifest=predictive_manifest,
            release_id=release_id,
        )
        inventory = _build_inventory(staging)
        payload = {
            "schema_version": NEURAL_HMSC_RELEASE_SCHEMA_VERSION,
            "kind": NEURAL_HMSC_RELEASE_KIND,
            "release_id": release_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "release_status": "release_ready",
            "release_scope": NEURAL_HMSC_RELEASE_SCOPE,
            "checkpoint_seeds": list(NEURAL_HMSC_RELEASE_SEEDS),
            "coefficient_calibration": "external_context_monotone_scale",
            "predictive_baseline_id": PROMOTED_PREDICTIVE_BASELINE_ID,
            "default_predictive_policy": DEFAULT_PREDICTIVE_MEAN_POLICY,
            "fallback_predictive_policy": "scale_only",
            "qualified_python_mcmc_role": "statistical_reference_only",
            "artifacts": {
                "checkpoint_package_manifest": _record(
                    staging / "checkpoints" / "package_manifest.json", staging
                ),
                "predictive_source_baseline": _record(
                    staging / "predictive" / "source_baseline" / "baseline.json",
                    staging,
                ),
                "predictive_local_manifests": localized,
                "release_audit_json": _record(evidence / audit_json.name, staging),
                "release_audit_markdown": _record(
                    evidence / audit_markdown.name, staging
                ),
                "support_matrix": _record(evidence / "support_matrix.json", staging),
            },
            "claim_boundary": audit_payload["claim_boundary"],
            "release_scope_exclusions": audit_payload["release_scope_exclusions"],
            "amortized_use_assumption": (
                "checkpoint training may cost more than one compact MCMC fit; "
                "the qualified speed advantage assumes repeated inference"
            ),
            "model_fitting_performed": False,
            "calibration_selection_performed": False,
            "inventory": inventory,
            "content_sha256": _inventory_digest(inventory),
            "source_provenance": {
                "packaged_members_root": str(packaged),
                "predictive_baseline_root": str(predictive),
                "audit_root": str(audit),
            },
        }
        release_path = staging / "release.json"
        release_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_neural_hmsc_release(release_path, expected_release_id=release_id)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "release.json"


def validate_neural_hmsc_release(
    release_path: str | Path,
    *,
    expected_release_id: str | None = None,
) -> dict[str, Any]:
    """Validate release identity, complete inventory, semantics, and loaders."""
    path = _release_json_path(release_path)
    root = path.parent
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != NEURAL_HMSC_RELEASE_KIND:
        raise ValueError("unsupported Neural-HMSC release kind")
    if int(payload.get("schema_version", -1)) != NEURAL_HMSC_RELEASE_SCHEMA_VERSION:
        raise ValueError("unsupported Neural-HMSC release schema")
    release_id = str(payload.get("release_id", ""))
    _validate_release_id(release_id)
    if expected_release_id is not None and release_id != expected_release_id:
        raise ValueError("Neural-HMSC release identifier differs")
    if payload.get("release_status") != "release_ready":
        raise ValueError("Neural-HMSC release is not release-ready")
    if payload.get("release_scope") != NEURAL_HMSC_RELEASE_SCOPE:
        raise ValueError("Neural-HMSC release scope differs")
    if tuple(payload.get("checkpoint_seeds", ())) != NEURAL_HMSC_RELEASE_SEEDS:
        raise ValueError("Neural-HMSC release checkpoint seeds differ")
    if payload.get("coefficient_calibration") != ("external_context_monotone_scale"):
        raise ValueError("Neural-HMSC release coefficient calibration differs")
    if payload.get("default_predictive_policy") != (DEFAULT_PREDICTIVE_MEAN_POLICY):
        raise ValueError("Neural-HMSC release default predictive policy differs")
    if payload.get("fallback_predictive_policy") != "scale_only":
        raise ValueError("Neural-HMSC release fallback predictive policy differs")
    if payload.get("qualified_python_mcmc_role") != "statistical_reference_only":
        raise ValueError("Neural-HMSC release MCMC role differs")
    if payload.get("model_fitting_performed") is not False:
        raise ValueError("release freeze must not fit a model")
    if payload.get("calibration_selection_performed") is not False:
        raise ValueError("release freeze must not select calibration")

    inventory = payload.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Neural-HMSC release inventory is missing")
    if payload.get("content_sha256") != _inventory_digest(inventory):
        raise ValueError("Neural-HMSC release inventory digest mismatch")
    expected_paths = {str(row["path"]) for row in inventory}
    if len(expected_paths) != len(inventory):
        raise ValueError("Neural-HMSC release inventory contains duplicate paths")
    actual_paths = {
        str(file.relative_to(root))
        for file in root.rglob("*")
        if file.is_file() and file != path
    }
    if expected_paths != actual_paths:
        raise ValueError("Neural-HMSC release inventory file set differs")
    for row in inventory:
        file = _contained_path(root, str(row["path"]))
        if file_sha256(file) != str(row["sha256"]):
            raise ValueError(f"Neural-HMSC release hash mismatch for {file}")
        if file.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Neural-HMSC release size mismatch for {file}")

    package = _load_and_validate_package_manifest(root / "checkpoints")
    source_baseline = validate_predictive_deployment_baseline(
        root / "predictive" / "source_baseline",
        expected_baseline_id=PROMOTED_PREDICTIVE_BASELINE_ID,
    )
    audit = _load_and_validate_audit(
        root / "evidence" / "neural_hmsc_v0_1_release_audit.json"
    )
    support_matrix = json.loads(
        (root / "evidence" / "support_matrix.json").read_text(encoding="utf-8")
    )
    if support_matrix != audit["support_matrix"]:
        raise ValueError("release support matrix differs from qualified audit")
    if payload.get("claim_boundary") != audit["claim_boundary"]:
        raise ValueError("release claim boundary differs from qualified audit")
    _validate_artifact_records(root, payload.get("artifacts"))

    package_rows = {int(row["seed"]): row for row in package["members"]}
    for seed in NEURAL_HMSC_RELEASE_SEEDS:
        row = package_rows[seed]
        checkpoint = root / "checkpoints" / str(seed) / "neural_checkpoint"
        engine = NeuralHmscInference.load(checkpoint)
        if engine.distribution != "probit" or engine.coefficient_calibration is None:
            raise ValueError(f"release checkpoint {seed} is not calibrated probit")
        if (
            file_sha256(checkpoint / "neural_checkpoint.json")
            != row["checkpoint_manifest_sha256"]
        ):
            raise ValueError(f"release checkpoint {seed} manifest hash differs")
        if (
            file_sha256(checkpoint / "coefficient_calibration.json")
            != row["calibration_artifact_sha256"]
        ):
            raise ValueError(f"release checkpoint {seed} calibration hash differs")

    for dataset in NEURAL_HMSC_RELEASE_DATASETS:
        for policy in PREDICTIVE_MEAN_POLICIES:
            source_record = source_baseline["datasets"][dataset][policy]
            local_record = payload["artifacts"]["predictive_local_manifests"][dataset][
                policy
            ]
            if local_record["source_sha256"] != source_record["sha256"]:
                raise ValueError("release predictive source manifest hash differs")
            local_path = _validate_record(root, local_record)
            _validate_localized_manifest(
                root,
                local_path,
                dataset=dataset,
                policy=policy,
                release_id=release_id,
                source_manifest=(
                    root / "predictive" / "source_baseline" / source_record["path"]
                ),
            )
    return payload


def load_neural_hmsc_release(
    registry_root: str | Path,
    *,
    release_id: str = NEURAL_HMSC_RELEASE_ID,
) -> NeuralHmscRelease:
    """Resolve and validate an immutable Neural-HMSC release identifier."""
    return NeuralHmscRelease.load(registry_root, release_id=release_id)


def _load_and_validate_package_manifest(root: Path) -> dict[str, Any]:
    path = root / "package_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "neural_hmsc_v0_1_packaged_checkpoint_bundle":
        raise ValueError("unsupported packaged checkpoint bundle kind")
    if payload.get("release_scope") != NEURAL_HMSC_RELEASE_SCOPE:
        raise ValueError("packaged checkpoint release scope differs")
    if tuple(payload.get("seeds", ())) != NEURAL_HMSC_RELEASE_SEEDS:
        raise ValueError("packaged checkpoint seeds differ")
    if payload.get("model_fitting_performed") is not False:
        raise ValueError("packaged checkpoint bundle refit a model")
    if payload.get("calibration_selection_performed") is not False:
        raise ValueError("packaged checkpoint bundle reselected calibration")
    if payload.get("all_weights_unchanged") is not True:
        raise ValueError("packaged checkpoint weights are not unchanged")
    rows = payload.get("members")
    if not isinstance(rows, list) or len(rows) != len(NEURAL_HMSC_RELEASE_SEEDS):
        raise ValueError("packaged checkpoint member count differs")
    expected_roles = set(_PREDICTIVE_FILENAMES)
    for row in rows:
        seed = int(row.get("seed", -1))
        if seed not in NEURAL_HMSC_RELEASE_SEEDS:
            raise ValueError("packaged checkpoint member seed differs")
        if row.get("checkpoint_version") != "0.4":
            raise ValueError("packaged checkpoint version differs")
        if row.get("distribution") != "probit":
            raise ValueError("packaged checkpoint distribution differs")
        if row.get("calibration_method") != "external_context_monotone_scale":
            raise ValueError("packaged checkpoint calibration method differs")
        artifacts = row.get("predictive_artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("packaged predictive artifacts are missing")
        roles = {(item.get("dataset"), item.get("policy")) for item in artifacts}
        if roles != expected_roles:
            raise ValueError("packaged predictive artifact roles differ")
        for item in artifacts:
            file = root / str(seed) / str(item["name"])
            observed = file_sha256(file)
            if observed != item["packaged_sha256"]:
                raise ValueError(f"packaged predictive artifact hash mismatch: {file}")
            if observed != item["source_sha256"]:
                raise ValueError(f"packaged predictive artifact changed: {file}")
    return payload


def _load_and_validate_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "neural_hmsc_v0_1_release_readiness_audit":
        raise ValueError("unsupported Neural-HMSC release audit kind")
    if payload.get("decision") != "release_ready":
        raise ValueError("Neural-HMSC release audit did not pass")
    if payload.get("release_blockers") != []:
        raise ValueError("Neural-HMSC release audit retains blockers")
    gates = payload.get("gates", {})
    required = payload.get("release_required_gates", ())
    if not required or not all(gates.get(name) is True for name in required):
        raise ValueError("Neural-HMSC release audit required gates did not pass")
    if payload.get("model_fitting_performed") is not False:
        raise ValueError("release audit fitted a model")
    if payload.get("calibration_selection_performed") is not False:
        raise ValueError("release audit reselected calibration")
    if not isinstance(payload.get("support_matrix"), list):
        raise ValueError("release audit support matrix is missing")
    return payload


def _localize_predictive_manifests(
    release_root: Path,
    *,
    package_manifest: dict[str, Any],
    predictive_manifest: dict[str, Any],
    release_id: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    output = release_root / "predictive" / "manifests"
    output.mkdir()
    records: dict[str, dict[str, dict[str, Any]]] = {}
    package_rows = {int(row["seed"]): row for row in package_manifest["members"]}
    for dataset in NEURAL_HMSC_RELEASE_DATASETS:
        records[dataset] = {}
        for policy in PREDICTIVE_MEAN_POLICIES:
            source_record = predictive_manifest["datasets"][dataset][policy]
            source_path = (
                release_root / "predictive" / "source_baseline" / source_record["path"]
            )
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            member_rows = payload.get("members")
            if not isinstance(member_rows, list):
                raise ValueError("predictive source manifest members are missing")
            for member in member_rows:
                seed = int(member["seed"])
                package_row = package_rows[seed]
                artifact = _packaged_predictive_artifact(
                    package_row, dataset=dataset, policy=policy
                )
                if member["sha256"] != artifact["packaged_sha256"]:
                    raise ValueError("predictive source/member hash mismatch")
                target = (
                    release_root / "checkpoints" / str(seed) / str(artifact["name"])
                )
                member["path"] = os.path.relpath(target, output)
            payload["release_relocation"] = {
                "release_id": release_id,
                "source_manifest_sha256": source_record["sha256"],
                "member_content_unchanged": True,
            }
            local_path = output / f"{dataset}_{policy}_ensemble.json"
            local_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records[dataset][policy] = {
                **_record(local_path, release_root),
                "source_sha256": source_record["sha256"],
            }
    return records


def _packaged_predictive_artifact(
    package_row: dict[str, Any], *, dataset: str, policy: str
) -> dict[str, Any]:
    matches = [
        item
        for item in package_row["predictive_artifacts"]
        if item.get("dataset") == dataset and item.get("policy") == policy
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one packaged {dataset}/{policy} artifact")
    expected_name = _PREDICTIVE_FILENAMES[(dataset, policy)]
    if matches[0].get("name") != expected_name:
        raise ValueError(f"packaged {dataset}/{policy} filename differs")
    return matches[0]


def _validate_localized_manifest(
    root: Path,
    path: Path,
    *,
    dataset: str,
    policy: str,
    release_id: str,
    source_manifest: Path,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    relocation = payload.pop("release_relocation", None)
    if not isinstance(relocation, dict):
        raise ValueError("release predictive manifest lacks relocation provenance")
    if relocation.get("release_id") != release_id:
        raise ValueError("release predictive relocation identifier differs")
    if relocation.get("source_manifest_sha256") != file_sha256(source_manifest):
        raise ValueError("release predictive relocation source hash differs")
    source_members = source.get("members", [])
    local_members = payload.get("members", [])
    if len(source_members) != len(local_members):
        raise ValueError("release predictive member count differs")
    for original, local in zip(source_members, local_members):
        expected = dict(original)
        expected.pop("path", None)
        observed = dict(local)
        local_member_path = observed.pop("path", None)
        if observed != expected:
            raise ValueError("release predictive member metadata changed")
        member_path = _contained_path(path.parent, str(local_member_path), root=root)
        if file_sha256(member_path) != local["sha256"]:
            raise ValueError("release predictive member hash differs")
    source_without_paths = json.loads(json.dumps(source))
    for row in source_without_paths["members"]:
        row.pop("path", None)
    for row in payload["members"]:
        row.pop("path", None)
    if payload != source_without_paths:
        raise ValueError("release predictive manifest semantics changed")
    ensemble = PredictiveProbabilityEnsemble.from_manifest(path, verify_hashes=False)
    _validate_loaded_ensemble(ensemble, dataset=dataset, policy=policy)


def _validate_loaded_ensemble(
    ensemble: PredictiveProbabilityEnsemble, *, dataset: str, policy: str
) -> None:
    if ensemble.calibration_role != policy:
        raise ValueError("release predictive ensemble policy differs")
    if ensemble.provenance.get("dataset") != dataset:
        raise ValueError("release predictive ensemble dataset differs")
    if ensemble.provenance.get("response_semantics") != "predictive_only":
        raise ValueError("release ensemble is not predictive-only")
    if ensemble.provenance.get("selection_outcomes_used") is not False:
        raise ValueError("release ensemble used target outcomes for selection")
    if not ensemble.parity_provenance_qualified:
        raise ValueError("release ensemble parity provenance is not qualified")
    ensemble.qualified_mcmc_reference


def _validate_artifact_records(root: Path, records: Any) -> None:
    if not isinstance(records, dict):
        raise ValueError("Neural-HMSC release artifact records are missing")
    for name in (
        "checkpoint_package_manifest",
        "predictive_source_baseline",
        "release_audit_json",
        "release_audit_markdown",
        "support_matrix",
    ):
        _validate_record(root, records.get(name))
    manifests = records.get("predictive_local_manifests")
    if not isinstance(manifests, dict):
        raise ValueError("release predictive manifest records are missing")


def _validate_record(root: Path, record: Any) -> Path:
    if (
        not isinstance(record, dict)
        or not record.get("path")
        or not record.get("sha256")
    ):
        raise ValueError("Neural-HMSC release file record is incomplete")
    path = _contained_path(root, str(record["path"]))
    if file_sha256(path) != str(record["sha256"]):
        raise ValueError(f"Neural-HMSC release record hash mismatch for {path}")
    return path


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "sha256": file_sha256(path),
    }


def _build_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
        if path.name != "release.json"
    ]


def _inventory_digest(inventory: list[dict[str, Any]]) -> str:
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _contained_path(base: Path, relative: str, *, root: Path | None = None) -> Path:
    path = (base / relative).resolve()
    boundary = base.resolve() if root is None else root.resolve()
    try:
        path.relative_to(boundary)
    except ValueError as exc:
        raise ValueError("Neural-HMSC release path escapes bundle root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Neural-HMSC release file not found: {path}")
    return path


def _release_json_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    return resolved / "release.json" if resolved.is_dir() else resolved


def _validate_release_id(value: str) -> None:
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value
    ):
        raise ValueError(
            "release_id must contain only lowercase letters, digits, or '_'"
        )
