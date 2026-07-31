"""Immutable artifact contract for generative Neural-HMSC iid probit v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pyhmsc.neural.generative_iid import (
    GENERATIVE_IID_MAX_SITES,
    GENERATIVE_IID_MAX_SPECIES,
    GENERATIVE_IID_MIN_SITES,
    GENERATIVE_IID_MIN_SPECIES,
)
from pyhmsc.neural.generative_iid_v2 import (
    GENERATIVE_IID_V1_ARTIFACT_SHA256,
    GENERATIVE_IID_V1_SOURCE_SHA256,
    GENERATIVE_IID_V2_ATTENTION_BLOCKS,
    GENERATIVE_IID_V2_ATTENTION_HEADS,
    GENERATIVE_IID_V2_FEEDFORWARD_WIDTH,
    GENERATIVE_IID_V2_HIDDEN_WIDTH,
    GENERATIVE_IID_V2_POSTERIOR_RANK,
    GENERATIVE_IID_V2_PREREGISTRATION_SHA256,
    GENERATIVE_IID_V2_PROTOCOL,
    GENERATIVE_IID_V2_REFINEMENT_DRAWS,
    GENERATIVE_IID_V2_REFINEMENT_STEPS,
    GENERATIVE_IID_V2_SCHEMA_VERSION,
    GENERATIVE_IID_V2_SEED_AUDIT_SHA256,
    GenerativeIidOrbitPosteriorModel,
)


GENERATIVE_IID_V2_ARTIFACT_KIND = "pyhmsc_generative_iid_orbit_checkpoint"
GENERATIVE_IID_V2_MODEL_FAMILY = "generative_iid_latent_probit"
GENERATIVE_IID_V2_MANIFEST = "generative_iid_orbit_checkpoint.json"
GENERATIVE_IID_V2_WEIGHTS = "weights.weights.h5"

_FORBIDDEN_DEPENDENCY_TOKENS = (
    "neural_hmsc_v0_1",
    "neural_hmsc_variable_probit_v1",
    "irls",
    "laplace",
    "mcmc_teacher",
    "calibration",
    "ensemble",
    "fallback",
    "router",
)


@dataclass(frozen=True)
class GenerativeIidOrbitInference:
    """Loaded v2 posterior with a validated immutable manifest."""

    model: GenerativeIidOrbitPosteriorModel
    manifest: dict[str, Any]
    checkpoint_root: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        max_sites: int = GENERATIVE_IID_MAX_SITES,
        max_species: int = GENERATIVE_IID_MAX_SPECIES,
    ) -> "GenerativeIidOrbitInference":
        model = GenerativeIidOrbitPosteriorModel(
            max_sites=max_sites, max_species=max_species
        )
        _build_model(model)
        return cls(
            model=model,
            manifest=_base_manifest(
                model,
                source_commit="unfrozen",
                source_provenance=None,
                training_manifest=None,
            ),
        )

    @classmethod
    def load(cls, checkpoint: str | Path) -> "GenerativeIidOrbitInference":
        root = Path(checkpoint).expanduser().resolve()
        manifest = validate_generative_iid_v2_checkpoint(root)
        config = manifest["model_config"]
        model = GenerativeIidOrbitPosteriorModel(
            max_sites=int(config["max_sites"]),
            max_species=int(config["max_species"]),
            hidden_width=int(config["hidden_width"]),
            attention_heads=int(config["attention_heads"]),
            attention_blocks=int(config["attention_blocks"]),
            feedforward_width=int(config["feedforward_width"]),
            posterior_rank=int(config["posterior_rank"]),
        )
        _build_model(model)
        model.load_weights(root / GENERATIVE_IID_V2_WEIGHTS)
        return cls(model=model, manifest=manifest, checkpoint_root=root)

    def save(
        self,
        checkpoint: str | Path,
        *,
        source_commit: str,
        source_provenance: dict[str, Any],
        training_manifest: dict[str, Any] | None,
    ) -> Path:
        root = Path(checkpoint).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"v2 checkpoint is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        _build_model(self.model)
        weights = root / GENERATIVE_IID_V2_WEIGHTS
        self.model.save_weights(weights)
        manifest = _base_manifest(
            self.model,
            source_commit=source_commit,
            source_provenance=source_provenance,
            training_manifest=training_manifest,
        )
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts"] = {
            "weights": {
                "path": GENERATIVE_IID_V2_WEIGHTS,
                "sha256": file_sha256(weights),
                "bytes": weights.stat().st_size,
            }
        }
        manifest["content_sha256"] = _content_sha256(manifest)
        (root / GENERATIVE_IID_V2_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_generative_iid_v2_checkpoint(root)
        return root


def validate_generative_iid_v2_checkpoint(
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Validate v2 identity, representation, provenance, and exact files."""
    root = Path(checkpoint).expanduser().resolve()
    manifest_path = root / GENERATIVE_IID_V2_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"v2 manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("kind") != GENERATIVE_IID_V2_ARTIFACT_KIND:
        raise ValueError("unsupported generative iid v2 artifact kind")
    if int(payload.get("schema_version", -1)) != (GENERATIVE_IID_V2_SCHEMA_VERSION):
        raise ValueError("unsupported generative iid v2 artifact schema")
    if payload.get("protocol") != GENERATIVE_IID_V2_PROTOCOL:
        raise ValueError("generative iid v2 protocol differs")
    if payload.get("model_family") != GENERATIVE_IID_V2_MODEL_FAMILY:
        raise ValueError("generative iid v2 model family differs")
    if payload.get("distribution") != "probit":
        raise ValueError("generative iid v2 distribution differs")
    if payload.get("formula") != "~ x1":
        raise ValueError("generative iid v2 formula differs")
    if payload.get("random_level") != {
        "type": "iid_site",
        "n_factors": 2,
        "one_unit_per_site": True,
    }:
        raise ValueError("generative iid v2 random level differs")
    if payload.get("posterior_state") != [
        "alpha",
        "Beta",
        "Eta",
        "Lambda",
        "log_tau",
    ]:
        raise ValueError("generative iid v2 posterior state differs")
    if payload.get("identifiable_summaries") != [
        "Eta@Lambda",
        "Lambda.T@Lambda",
        "association_correlation",
    ]:
        raise ValueError("generative iid v2 summaries differ")
    if payload.get("posterior_family") != (
        "student_t_global_plus_exact_o2_orbit_matrix_normal"
    ):
        raise ValueError("generative iid v2 posterior family differs")
    if payload.get("training_objective") != "importance_weighted_elbo_k8":
        raise ValueError("generative iid v2 objective differs")
    if payload.get("calibration") is not None:
        raise ValueError("generative iid v2 calibration must be absent")
    if payload.get("preregistration_sha256") != (
        GENERATIVE_IID_V2_PREREGISTRATION_SHA256
    ):
        raise ValueError("generative iid v2 preregistration hash differs")
    if payload.get("seed_audit_sha256") != (GENERATIVE_IID_V2_SEED_AUDIT_SHA256):
        raise ValueError("generative iid v2 seed-audit hash differs")
    if payload.get("v1_regression_hashes") != {
        "pyhmsc/neural/generative_iid.py": GENERATIVE_IID_V1_SOURCE_SHA256,
        "pyhmsc/neural/generative_iid_artifact.py": (GENERATIVE_IID_V1_ARTIFACT_SHA256),
    }:
        raise ValueError("generative iid v1 regression hashes differ")
    config = payload.get("model_config")
    expected_architecture = {
        "hidden_width": GENERATIVE_IID_V2_HIDDEN_WIDTH,
        "attention_heads": GENERATIVE_IID_V2_ATTENTION_HEADS,
        "attention_blocks": GENERATIVE_IID_V2_ATTENTION_BLOCKS,
        "feedforward_width": GENERATIVE_IID_V2_FEEDFORWARD_WIDTH,
        "posterior_rank": GENERATIVE_IID_V2_POSTERIOR_RANK,
    }
    if not isinstance(config, dict) or any(
        int(config.get(key, -1)) != value
        for key, value in expected_architecture.items()
    ):
        raise ValueError("generative iid v2 frozen architecture differs")
    if not (
        GENERATIVE_IID_MIN_SITES
        <= int(config.get("max_sites", -1))
        <= GENERATIVE_IID_MAX_SITES
    ):
        raise ValueError("generative iid v2 max_sites differs")
    if not (
        GENERATIVE_IID_MIN_SPECIES
        <= int(config.get("max_species", -1))
        <= GENERATIVE_IID_MAX_SPECIES
    ):
        raise ValueError("generative iid v2 max_species differs")
    if payload.get("support") != {
        "n_sites": [
            GENERATIVE_IID_MIN_SITES,
            int(config["max_sites"]),
        ],
        "n_species": [
            GENERATIVE_IID_MIN_SPECIES,
            int(config["max_species"]),
        ],
        "n_covariates": 2,
        "n_factors": 2,
    }:
        raise ValueError("generative iid v2 support differs")
    refinement = payload.get("refinement")
    if refinement != {
        "draws": GENERATIVE_IID_V2_REFINEMENT_DRAWS,
        "steps": list(GENERATIVE_IID_V2_REFINEMENT_STEPS),
        "backtracks": 3,
        "common_random_acceptance": True,
        "first_order_stop_gradient": True,
    }:
        raise ValueError("generative iid v2 refinement contract differs")
    if payload.get("dependency_inventory") != []:
        raise ValueError("generative iid v2 dependency inventory must be empty")
    dependencies = payload.get("source_dependencies")
    if dependencies != {
        "raw_inputs": [
            "X",
            "Y",
            "response_mask",
            "site_mask",
            "species_mask",
        ],
        "unchanged_target": [
            "generative_log_prior",
            "probit_log_likelihood",
        ],
    }:
        raise ValueError("generative iid v2 source dependencies differ")
    serialized = json.dumps(dependencies).lower()
    forbidden = [token for token in _FORBIDDEN_DEPENDENCY_TOKENS if token in serialized]
    if forbidden:
        raise ValueError(
            "generative iid v2 contains forbidden dependencies: " + ", ".join(forbidden)
        )
    if payload.get("content_sha256") != _content_sha256(payload):
        raise ValueError("generative iid v2 content hash differs")
    if (
        not isinstance(payload.get("source_commit"), str)
        or not payload["source_commit"]
    ):
        raise ValueError("generative iid v2 source commit is missing")
    provenance = payload.get("source_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("generative iid v2 source provenance is missing")
    if not isinstance(provenance.get("branch"), str) or not provenance["branch"]:
        raise ValueError("generative iid v2 source branch is missing")
    if not isinstance(provenance.get("worktree_dirty"), bool):
        raise ValueError("generative iid v2 worktree state is missing")
    source_files = provenance.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("generative iid v2 source-file inventory is missing")
    paths: set[str] = set()
    for record in source_files:
        if not isinstance(record, dict):
            raise ValueError("generative iid v2 source record is invalid")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path or path in paths:
            raise ValueError("generative iid v2 source paths differ")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("generative iid v2 source hash differs")
        paths.add(path)
    environment = provenance.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("generative iid v2 runtime environment is missing")
    for key in (
        "python",
        "tensorflow",
        "tensorflow_probability",
        "numpy",
        "platform",
    ):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"generative iid v2 runtime environment lacks {key}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"weights"}:
        raise ValueError("generative iid v2 artifact inventory differs")
    record = artifacts["weights"]
    weights = root / str(record.get("path", ""))
    if weights.name != GENERATIVE_IID_V2_WEIGHTS or not weights.is_file():
        raise ValueError("generative iid v2 weights are missing")
    if file_sha256(weights) != record.get("sha256"):
        raise ValueError("generative iid v2 weights hash differs")
    if int(record.get("bytes", -1)) != weights.stat().st_size:
        raise ValueError("generative iid v2 weights size differs")
    files = {path.name for path in root.iterdir() if path.is_file()}
    if files != {GENERATIVE_IID_V2_MANIFEST, GENERATIVE_IID_V2_WEIGHTS}:
        raise ValueError("generative iid v2 checkpoint file set differs")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_manifest(
    model: GenerativeIidOrbitPosteriorModel,
    *,
    source_commit: str,
    source_provenance: dict[str, Any] | None,
    training_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": GENERATIVE_IID_V2_SCHEMA_VERSION,
        "kind": GENERATIVE_IID_V2_ARTIFACT_KIND,
        "protocol": GENERATIVE_IID_V2_PROTOCOL,
        "model_family": GENERATIVE_IID_V2_MODEL_FAMILY,
        "distribution": "probit",
        "formula": "~ x1",
        "random_level": {
            "type": "iid_site",
            "n_factors": 2,
            "one_unit_per_site": True,
        },
        "posterior_state": ["alpha", "Beta", "Eta", "Lambda", "log_tau"],
        "identifiable_summaries": [
            "Eta@Lambda",
            "Lambda.T@Lambda",
            "association_correlation",
        ],
        "posterior_family": ("student_t_global_plus_exact_o2_orbit_matrix_normal"),
        "training_objective": "importance_weighted_elbo_k8",
        "calibration": None,
        "model_config": model.get_config(),
        "refinement": {
            "draws": GENERATIVE_IID_V2_REFINEMENT_DRAWS,
            "steps": list(GENERATIVE_IID_V2_REFINEMENT_STEPS),
            "backtracks": 3,
            "common_random_acceptance": True,
            "first_order_stop_gradient": True,
        },
        "support": {
            "n_sites": [GENERATIVE_IID_MIN_SITES, int(model.max_sites)],
            "n_species": [
                GENERATIVE_IID_MIN_SPECIES,
                int(model.max_species),
            ],
            "n_covariates": 2,
            "n_factors": 2,
        },
        "preregistration_sha256": GENERATIVE_IID_V2_PREREGISTRATION_SHA256,
        "seed_audit_sha256": GENERATIVE_IID_V2_SEED_AUDIT_SHA256,
        "v1_regression_hashes": {
            "pyhmsc/neural/generative_iid.py": GENERATIVE_IID_V1_SOURCE_SHA256,
            "pyhmsc/neural/generative_iid_artifact.py": (
                GENERATIVE_IID_V1_ARTIFACT_SHA256
            ),
        },
        "source_commit": str(source_commit),
        "source_provenance": source_provenance,
        "training_manifest": training_manifest,
        "dependency_inventory": [],
        "source_dependencies": {
            "raw_inputs": [
                "X",
                "Y",
                "response_mask",
                "site_mask",
                "species_mask",
            ],
            "unchanged_target": [
                "generative_log_prior",
                "probit_log_likelihood",
            ],
        },
        "claim_boundary": (
            "bounded orbit-symmetrized approximate posterior for the "
            "two-factor iid Bernoulli-probit family; not full HMSC or MCMC "
            "equivalence"
        ),
    }


def _content_sha256(payload: dict[str, Any]) -> str:
    canonical = {
        key: value
        for key, value in payload.items()
        if key not in {"content_sha256", "created_at"}
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _build_model(model: GenerativeIidOrbitPosteriorModel) -> None:
    X = np.zeros((1, model.max_sites, 2), dtype=np.float32)
    X[:, :GENERATIVE_IID_MIN_SITES, 0] = 1.0
    Y = np.zeros((1, model.max_sites, model.max_species), dtype=np.float32)
    site_mask = np.zeros((1, model.max_sites), dtype=bool)
    species_mask = np.zeros((1, model.max_species), dtype=bool)
    site_mask[:, :GENERATIVE_IID_MIN_SITES] = True
    species_mask[:, :GENERATIVE_IID_MIN_SPECIES] = True
    response_mask = site_mask[:, :, None] & species_mask[:, None, :]
    model(
        {
            "X": X,
            "Y": Y,
            "response_mask": response_mask,
            "site_mask": site_mask,
            "species_mask": species_mask,
        },
        training=False,
        refine=False,
    )
