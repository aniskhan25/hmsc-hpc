"""Immutable artifacts for generative Neural-HMSC iid probit v1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from pyhmsc.neural.generative_iid import (
    GENERATIVE_IID_DESIGN_REVIEW_SHA256,
    GENERATIVE_IID_MAX_SITES,
    GENERATIVE_IID_MAX_SPECIES,
    GENERATIVE_IID_MIN_SITES,
    GENERATIVE_IID_MIN_SPECIES,
    GENERATIVE_IID_PREREGISTRATION_SHA256,
    GENERATIVE_IID_PROTOCOL,
    GENERATIVE_IID_SCHEMA_VERSION,
    GENERATIVE_IID_SEED_AUDIT_SHA256,
    GenerativeIidPosteriorModel,
)


GENERATIVE_IID_ARTIFACT_KIND = "pyhmsc_generative_iid_checkpoint"
GENERATIVE_IID_MODEL_FAMILY = "generative_iid_latent_probit"
GENERATIVE_IID_MANIFEST = "generative_iid_checkpoint.json"
GENERATIVE_IID_WEIGHTS = "weights.weights.h5"

_FORBIDDEN_DEPENDENCY_TOKENS = (
    "neural_hmsc_v0_1",
    "neural_hmsc_variable_probit_v1",
    "irls",
    "laplace",
    "mcmc_teacher",
    "calibration",
    "ensemble",
)


@dataclass(frozen=True)
class GenerativeIidInference:
    """Loaded structural neural posterior with a validated immutable manifest."""

    model: GenerativeIidPosteriorModel
    manifest: dict[str, Any]
    checkpoint_root: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        max_sites: int = GENERATIVE_IID_MAX_SITES,
        max_species: int = GENERATIVE_IID_MAX_SPECIES,
    ) -> "GenerativeIidInference":
        model = GenerativeIidPosteriorModel(
            max_sites=max_sites, max_species=max_species
        )
        _build_model(model)
        return cls(
            model=model,
            manifest=_base_manifest(
                model,
                source_commit="unfrozen",
                training_manifest=None,
            ),
            checkpoint_root=None,
        )

    @classmethod
    def load(cls, checkpoint: str | Path) -> "GenerativeIidInference":
        root = Path(checkpoint).expanduser().resolve()
        manifest = validate_generative_iid_checkpoint(root)
        config = manifest["model_config"]
        model = GenerativeIidPosteriorModel(
            max_sites=int(config["max_sites"]),
            max_species=int(config["max_species"]),
            hidden_width=int(config["hidden_width"]),
            message_rounds=int(config["message_rounds"]),
            posterior_rank=int(config["posterior_rank"]),
        )
        _build_model(model)
        model.load_weights(root / GENERATIVE_IID_WEIGHTS)
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
            raise FileExistsError(
                f"generative iid checkpoint is not empty: {root}"
            )
        root.mkdir(parents=True, exist_ok=True)
        _build_model(self.model)
        weights = root / GENERATIVE_IID_WEIGHTS
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
                "path": GENERATIVE_IID_WEIGHTS,
                "sha256": file_sha256(weights),
                "bytes": weights.stat().st_size,
            }
        }
        manifest["content_sha256"] = _content_sha256(manifest)
        path = root / GENERATIVE_IID_MANIFEST
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_generative_iid_checkpoint(root)
        return root


def validate_generative_iid_checkpoint(
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Validate identity, scope, hashes, dependencies, and exact file set."""
    root = Path(checkpoint).expanduser().resolve()
    manifest_path = root / GENERATIVE_IID_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"generative iid manifest not found: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("kind") != GENERATIVE_IID_ARTIFACT_KIND:
        raise ValueError("unsupported generative iid artifact kind")
    if int(payload.get("schema_version", -1)) != GENERATIVE_IID_SCHEMA_VERSION:
        raise ValueError("unsupported generative iid artifact schema")
    if payload.get("protocol") != GENERATIVE_IID_PROTOCOL:
        raise ValueError("generative iid protocol differs")
    if payload.get("model_family") != GENERATIVE_IID_MODEL_FAMILY:
        raise ValueError("generative iid model family differs")
    if payload.get("posterior_family") != "joint_low_rank_normal_rank_16":
        raise ValueError("generative iid posterior family differs")
    if payload.get("training_objective") != "importance_weighted_elbo_k8":
        raise ValueError("generative iid training objective differs")
    if payload.get("distribution") != "probit":
        raise ValueError("generative iid distribution differs")
    if payload.get("formula") != "~ x1":
        raise ValueError("generative iid formula differs")
    if payload.get("preregistration_sha256") != (
        GENERATIVE_IID_PREREGISTRATION_SHA256
    ):
        raise ValueError("generative iid preregistration hash differs")
    if payload.get("seed_audit_sha256") != GENERATIVE_IID_SEED_AUDIT_SHA256:
        raise ValueError("generative iid seed-audit hash differs")
    if payload.get("design_review_sha256") != (
        GENERATIVE_IID_DESIGN_REVIEW_SHA256
    ):
        raise ValueError("generative iid design-review hash differs")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("generative iid model config is missing")
    if (
        int(config.get("hidden_width", -1)) != 64
        or int(config.get("message_rounds", -1)) != 3
        or int(config.get("posterior_rank", -1)) != 16
    ):
        raise ValueError("generative iid frozen architecture differs")
    if not (
        GENERATIVE_IID_MIN_SITES
        <= int(config.get("max_sites", -1))
        <= GENERATIVE_IID_MAX_SITES
    ):
        raise ValueError("generative iid max_sites differs")
    if not (
        GENERATIVE_IID_MIN_SPECIES
        <= int(config.get("max_species", -1))
        <= GENERATIVE_IID_MAX_SPECIES
    ):
        raise ValueError("generative iid max_species differs")
    dependencies = payload.get("dependency_inventory")
    if dependencies != []:
        raise ValueError(
            "generative iid candidate dependency inventory must be empty"
        )
    serialized = json.dumps(payload.get("source_dependencies", {})).lower()
    forbidden = [
        token for token in _FORBIDDEN_DEPENDENCY_TOKENS if token in serialized
    ]
    if forbidden:
        raise ValueError(
            "generative iid artifact contains forbidden dependencies: "
            + ", ".join(forbidden)
        )
    if payload.get("content_sha256") != _content_sha256(payload):
        raise ValueError("generative iid content hash differs")
    provenance = payload.get("source_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("generative iid source provenance is missing")
    if not isinstance(provenance.get("branch"), str) or not provenance["branch"]:
        raise ValueError("generative iid source branch is missing")
    if not isinstance(provenance.get("worktree_dirty"), bool):
        raise ValueError("generative iid worktree state is missing")
    source_files = provenance.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("generative iid source-file inventory is missing")
    source_paths = set()
    for source_record in source_files:
        if not isinstance(source_record, dict):
            raise ValueError("generative iid source-file record is invalid")
        path = source_record.get("path")
        sha256 = source_record.get("sha256")
        if not isinstance(path, str) or not path or path in source_paths:
            raise ValueError("generative iid source-file paths differ")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError("generative iid source-file hash differs")
        source_paths.add(path)
    environment = provenance.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("generative iid runtime environment is missing")
    for key in ("python", "tensorflow", "tensorflow_probability", "numpy", "platform"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(
                f"generative iid runtime environment lacks {key}"
            )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"weights"}:
        raise ValueError("generative iid artifact inventory differs")
    record = artifacts["weights"]
    weights = root / str(record.get("path", ""))
    if weights.name != GENERATIVE_IID_WEIGHTS or not weights.is_file():
        raise ValueError("generative iid weights are missing")
    if file_sha256(weights) != record.get("sha256"):
        raise ValueError("generative iid weights hash differs")
    if int(record.get("bytes", -1)) != weights.stat().st_size:
        raise ValueError("generative iid weights size differs")
    actual_files = {
        path.name for path in root.iterdir() if path.is_file()
    }
    if actual_files != {GENERATIVE_IID_MANIFEST, GENERATIVE_IID_WEIGHTS}:
        raise ValueError("generative iid checkpoint file set differs")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_manifest(
    model: GenerativeIidPosteriorModel,
    *,
    source_commit: str,
    training_manifest: dict[str, Any] | None,
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": GENERATIVE_IID_SCHEMA_VERSION,
        "kind": GENERATIVE_IID_ARTIFACT_KIND,
        "protocol": GENERATIVE_IID_PROTOCOL,
        "model_family": GENERATIVE_IID_MODEL_FAMILY,
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
        "posterior_family": "joint_low_rank_normal_rank_16",
        "training_objective": "importance_weighted_elbo_k8",
        "calibration": None,
        "model_config": model.get_config(),
        "support": {
            "n_sites": [
                GENERATIVE_IID_MIN_SITES,
                int(model.max_sites),
            ],
            "n_species": [
                GENERATIVE_IID_MIN_SPECIES,
                int(model.max_species),
            ],
            "n_covariates": 2,
            "n_factors": 2,
        },
        "preregistration_sha256": GENERATIVE_IID_PREREGISTRATION_SHA256,
        "seed_audit_sha256": GENERATIVE_IID_SEED_AUDIT_SHA256,
        "design_review_sha256": GENERATIVE_IID_DESIGN_REVIEW_SHA256,
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
            ]
        },
        "claim_boundary": (
            "bounded approximate posterior for the two-factor iid latent "
            "Bernoulli-probit family; not full HMSC or MCMC equivalence"
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


def _build_model(model: GenerativeIidPosteriorModel) -> None:
    max_sites = model.max_sites
    max_species = model.max_species
    X = np.zeros((1, max_sites, 2), dtype=np.float32)
    X[:, :GENERATIVE_IID_MIN_SITES, 0] = 1.0
    Y = np.zeros((1, max_sites, max_species), dtype=np.float32)
    site_mask = np.zeros((1, max_sites), dtype=bool)
    species_mask = np.zeros((1, max_species), dtype=bool)
    site_mask[:, :GENERATIVE_IID_MIN_SITES] = True
    species_mask[:, :GENERATIVE_IID_MIN_SPECIES] = True
    response_mask = (
        site_mask[:, :, None] & species_mask[:, None, :]
    )
    model(
        {
            "X": X,
            "Y": Y,
            "response_mask": response_mask,
            "site_mask": site_mask,
            "species_mask": species_mask,
        },
        training=False,
    )
