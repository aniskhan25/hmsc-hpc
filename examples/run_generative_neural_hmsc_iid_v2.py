#!/usr/bin/env python3
"""Sealed disposable harness for generative Neural-HMSC iid probit v2."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache")
)

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.generative_iid import (
    batch_generative_iid_datasets,
    make_stratified_response_mask,
    simulate_generative_iid_dataset,
)
from pyhmsc.neural.generative_iid_mcmc import exact_model_log_joint
from pyhmsc.neural.generative_iid_v2 import (
    GENERATIVE_IID_V2_PREREGISTRATION_SHA256,
    GENERATIVE_IID_V2_PROTOCOL,
    GENERATIVE_IID_V2_SEED_AUDIT_SHA256,
    GenerativeIidOrbitPosteriorModel,
    importance_weighted_orbit_loss,
    train_generative_iid_orbit_model,
)
from pyhmsc.neural.generative_iid_v2_artifact import (
    GENERATIVE_IID_V2_MANIFEST,
    GENERATIVE_IID_V2_WEIGHTS,
    GenerativeIidOrbitInference,
    file_sha256,
    validate_generative_iid_v2_checkpoint,
)


CONFIRMATION_ENV = "OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE"
CONFIRMATION_VALUE = "GENERATE_593M_594M_DISPOSABLE_ONLY"
HOST_SOURCE_COMMIT_ENV = "GENERATIVE_IID_V2_HOST_SOURCE_COMMIT"
HOST_SOURCE_BRANCH_ENV = "GENERATIVE_IID_V2_HOST_SOURCE_BRANCH"
HOST_WORKTREE_CLEAN_ENV = "GENERATIVE_IID_V2_HOST_WORKTREE_CLEAN"

TRAINING_SEEDS = tuple(range(593_000_001, 593_000_019))
VALIDATION_SEEDS = tuple(range(594_000_001, 594_000_019))
MODEL_SEED = 511_900_001
SMOKE_EPOCHS = 2

PREREGISTRATION = (
    ROOT / "docs" / "generative_neural_hmsc_iid_v2_orbit_preregistration_2026-07-31.md"
)
SEED_AUDIT = (
    ROOT / "docs" / "generative_neural_hmsc_iid_v2_seed_reaudit_2026-07-31.json.md"
)
REPRESENTATION_DECISION = (
    ROOT
    / "docs"
    / "generative_neural_hmsc_iid_v2_representation_decision_2026-07-31.md"
)
IMPLEMENTATION_EVIDENCE = (
    ROOT / "docs" / "generative_neural_hmsc_iid_v2_implementation_2026-07-31.md"
)
REPRESENTATION_DECISION_SHA256 = (
    "13041f6368eeaa64d4eae4446782c99c7a0b8af2a13bb13be9a69bec040df7ea"
)
IMPLEMENTATION_EVIDENCE_SHA256 = (
    "0d54f04ea5ec5c654df73594b7ff6614157152ec87bdfc3ecfd09c2401550cab"
)

SOURCE_PATHS = (
    "pyhmsc/neural/generative_iid.py",
    "pyhmsc/neural/generative_iid_mcmc.py",
    "pyhmsc/neural/generative_iid_v2.py",
    "pyhmsc/neural/generative_iid_v2_artifact.py",
    "pyhmsc/neural/__init__.py",
    "examples/run_generative_neural_hmsc_iid_v2.py",
    "docs/generative_neural_hmsc_iid_v2_orbit_preregistration_2026-07-31.md",
    "docs/generative_neural_hmsc_iid_v2_seed_reaudit_2026-07-31.json.md",
    "docs/generative_neural_hmsc_iid_v2_representation_decision_2026-07-31.md",
    "docs/generative_neural_hmsc_iid_v2_implementation_2026-07-31.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "check-seal",
            "preflight",
            "disposable-smoke",
            "validate-disposable",
        ),
        default="check-seal",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-source-commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "check-seal":
        _require_no_opening_tokens()
        _validate_frozen_documents()
        print(json.dumps(seal_status(), indent=2, sort_keys=True))
        return
    if args.expected_source_commit is None:
        raise ValueError("--expected-source-commit is required")
    if args.mode == "preflight":
        print(
            json.dumps(
                preflight_disposable_smoke(
                    expected_source_commit=args.expected_source_commit
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output is None:
        raise ValueError(f"--output is required for {args.mode}")
    if args.mode == "disposable-smoke":
        run_disposable_smoke(
            args.output,
            expected_source_commit=args.expected_source_commit,
        )
        return
    print(
        json.dumps(
            validate_disposable_smoke(
                args.output,
                expected_source_commit=args.expected_source_commit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


def seal_status() -> dict[str, object]:
    return {
        "status": "generative_iid_v2_disposable_sealed",
        "protocol": GENERATIVE_IID_V2_PROTOCOL,
        "disposable_training_seed_range": [
            TRAINING_SEEDS[0],
            TRAINING_SEEDS[-1],
        ],
        "disposable_validation_seed_range": [
            VALIDATION_SEEDS[0],
            VALIDATION_SEEDS[-1],
        ],
        "disposable_seed_ranges_opened": False,
        "production_511m_opened": False,
        "fixed_validation_512m_opened": False,
        "reserved_513m_515m_opened": False,
        "confirmation_env": CONFIRMATION_ENV,
        "confirmation_present": False,
    }


def preflight_disposable_smoke(*, expected_source_commit: str) -> dict[str, object]:
    """Read-only preflight that refuses every opening token."""
    _require_no_opening_tokens()
    _validate_frozen_documents()
    source_commit = _require_clean_pinned_source(expected_source_commit)
    cells = _factorial_cells()
    if len(cells) != 18:
        raise AssertionError("v2 disposable factorial differs")
    source_files = _source_file_inventory()
    return {
        "status": "generative_iid_v2_disposable_preflight_sealed",
        "protocol": GENERATIVE_IID_V2_PROTOCOL,
        "source_commit": source_commit,
        "source_files": source_files,
        "factorial_cell_count": len(cells),
        "factorial_unique_cell_count": len(
            {
                (
                    cell["n_sites"],
                    cell["n_species"],
                    cell["covariate_shape"],
                    cell["loading_stratum"],
                    cell["prevalence_stratum"],
                )
                for cell in cells
            }
        ),
        "smoke_epochs": SMOKE_EPOCHS,
        "model_seed": MODEL_SEED,
        "simulation_generation_called": False,
        "output_created": False,
        "disposable_seed_ranges_opened": False,
        "production_511m_opened": False,
        "fixed_validation_512m_opened": False,
        "reserved_513m_515m_opened": False,
        "authorization_required": True,
        "confirmation_env": CONFIRMATION_ENV,
    }


def run_disposable_smoke(
    output: Path,
    *,
    expected_source_commit: str,
) -> Path:
    """Generate exactly the authorized 593M-594M smoke."""
    _require_disposable_token_only()
    _validate_frozen_documents()
    source_commit = _require_clean_pinned_source(expected_source_commit)
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"v2 smoke output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train = _generate_block(TRAINING_SEEDS, masked=False)
    validation = _generate_block(VALIDATION_SEEDS, masked=True)
    corpus_path = output / "corpus_manifest.json"
    corpus_path.write_text(
        json.dumps(
            _corpus_manifest(train, validation),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    train_batch = batch_generative_iid_datasets(train, max_sites=96, max_species=75)
    validation_batch = batch_generative_iid_datasets(
        validation, max_sites=96, max_species=75
    )
    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = GenerativeIidOrbitPosteriorModel()
    history = train_generative_iid_orbit_model(
        model,
        train_batch,
        epochs=SMOKE_EPOCHS,
        model_seed=MODEL_SEED,
    )
    posterior = model(
        validation_batch.model_inputs(),
        training=False,
        refine=True,
        refinement_seed=MODEL_SEED,
    )
    loss, diagnostics = importance_weighted_orbit_loss(
        posterior,
        validation_batch.model_inputs(),
        draws=8,
        kl_weight=1.0,
        seed=MODEL_SEED,
    )
    invariants = posterior.invariant_moments()
    exact_truth_log_joint = float(
        exact_model_log_joint(
            _truth_state_for_dataset(validation[0]),
            validation[0],
        )[0]
    )
    source_provenance = _source_provenance(source_commit)
    checkpoint = GenerativeIidOrbitInference(
        model=model,
        manifest={},
    ).save(
        output / "checkpoint",
        source_commit=source_commit,
        source_provenance=source_provenance,
        training_manifest={
            "role": "v2_disposable_smoke_only",
            "training_seed_range": [
                TRAINING_SEEDS[0],
                TRAINING_SEEDS[-1],
            ],
            "validation_seed_range": [
                VALIDATION_SEEDS[0],
                VALIDATION_SEEDS[-1],
            ],
            "epochs": SMOKE_EPOCHS,
            "model_seed": MODEL_SEED,
            "production_511m_opened": False,
            "fixed_validation_512m_opened": False,
            "reserved_513m_515m_opened": False,
        },
    )
    checkpoint_manifest = validate_generative_iid_v2_checkpoint(checkpoint)
    report = {
        "status": "generative_iid_v2_disposable_smoke_complete",
        "evidence_role": "plumbing_and_optimization_only",
        "training_seed_range": [
            TRAINING_SEEDS[0],
            TRAINING_SEEDS[-1],
        ],
        "validation_seed_range": [
            VALIDATION_SEEDS[0],
            VALIDATION_SEEDS[-1],
        ],
        "training_contexts": len(train),
        "validation_contexts": len(validation),
        "epochs": SMOKE_EPOCHS,
        "model_seed": MODEL_SEED,
        "final_training_loss": history.loss[-1],
        "final_training_iwelbo": history.iwelbo[-1],
        "final_training_gradient_norm": history.gradient_norm[-1],
        "validation_loss": float(loss),
        "validation_iwelbo": float(diagnostics["iwelbo"]),
        "exact_truth_log_joint_first_validation": exact_truth_log_joint,
        "all_finite": bool(
            np.isfinite(history.loss).all()
            and np.isfinite(history.iwelbo).all()
            and np.isfinite(history.gradient_norm).all()
            and np.isfinite(float(loss))
            and all(
                np.all(np.isfinite(np.asarray(value))) for value in invariants.values()
            )
        ),
        "refinement_steps": len(posterior.refinement_trace),
        "checkpoint_content_sha256": checkpoint_manifest["content_sha256"],
        "checkpoint_weights_sha256": checkpoint_manifest["artifacts"]["weights"][
            "sha256"
        ],
        "disposable_seed_ranges_opened": True,
        "production_511m_opened": False,
        "fixed_validation_512m_opened": False,
        "reserved_513m_515m_opened": False,
        "source_commit": source_commit,
    }
    report_path = output / "disposable_smoke_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze = {
        "schema_version": 1,
        "kind": "generative_iid_v2_disposable_freeze",
        "protocol": GENERATIVE_IID_V2_PROTOCOL,
        "source_commit": source_commit,
        "preregistration_sha256": (GENERATIVE_IID_V2_PREREGISTRATION_SHA256),
        "seed_audit_sha256": GENERATIVE_IID_V2_SEED_AUDIT_SHA256,
        "representation_decision_sha256": REPRESENTATION_DECISION_SHA256,
        "implementation_evidence_sha256": IMPLEMENTATION_EVIDENCE_SHA256,
        "artifacts": {
            "corpus_manifest": _artifact_record(corpus_path, output=output),
            "report": _artifact_record(report_path, output=output),
            "checkpoint_manifest": _artifact_record(
                checkpoint / GENERATIVE_IID_V2_MANIFEST,
                output=output,
            ),
            "checkpoint_weights": _artifact_record(
                checkpoint / GENERATIVE_IID_V2_WEIGHTS,
                output=output,
            ),
        },
        "disposable_seed_ranges_opened": True,
        "production_511m_opened": False,
        "fixed_validation_512m_opened": False,
        "reserved_513m_515m_opened": False,
    }
    freeze_path = output / "freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return freeze_path


def validate_disposable_smoke(
    output: Path,
    *,
    expected_source_commit: str,
) -> dict[str, object]:
    """Independently validate a completed authorized disposable run."""
    _require_disposable_token_only()
    _validate_frozen_documents()
    _require_clean_pinned_source(expected_source_commit)
    output = output.expanduser().resolve()
    freeze = _read_json(output / "freeze.json")
    report = _read_json(output / "disposable_smoke_report.json")
    corpus = _read_json(output / "corpus_manifest.json")
    checkpoint = output / "checkpoint"
    manifest = validate_generative_iid_v2_checkpoint(checkpoint)
    _validate_freeze_inventory(output, freeze)
    _require_exact_seed_flags(
        freeze,
        disposable=True,
        context="v2 disposable freeze",
    )
    _require_exact_seed_flags(
        report,
        disposable=True,
        context="v2 disposable report",
    )
    if freeze.get("source_commit") != expected_source_commit:
        raise ValueError("v2 disposable freeze source commit differs")
    if report.get("source_commit") != expected_source_commit:
        raise ValueError("v2 disposable report source commit differs")
    if report.get("all_finite") is not True:
        raise ValueError("v2 disposable report contains non-finite values")
    if report.get("refinement_steps") != 4:
        raise ValueError("v2 disposable refinement count differs")
    if manifest.get("training_manifest", {}).get("role") != (
        "v2_disposable_smoke_only"
    ):
        raise ValueError("v2 disposable checkpoint role differs")
    _validate_corpus_manifest(corpus)

    train = _generate_block(TRAINING_SEEDS, masked=False)
    validation = _generate_block(VALIDATION_SEEDS, masked=True)
    if _corpus_records(train, role="training") != corpus["training"]:
        raise ValueError("v2 disposable training corpus differs")
    if _corpus_records(validation, role="validation") != corpus["validation"]:
        raise ValueError("v2 disposable validation corpus differs")
    validation_batch = batch_generative_iid_datasets(
        validation, max_sites=96, max_species=75
    )
    loaded = GenerativeIidOrbitInference.load(checkpoint)
    posterior = loaded.model(
        validation_batch.model_inputs(),
        training=False,
        refine=True,
        refinement_seed=MODEL_SEED,
    )
    loss, diagnostics = importance_weighted_orbit_loss(
        posterior,
        validation_batch.model_inputs(),
        draws=8,
        kl_weight=1.0,
        seed=MODEL_SEED,
    )
    exact = float(
        exact_model_log_joint(
            _truth_state_for_dataset(validation[0]),
            validation[0],
        )[0]
    )
    _require_close(
        float(loss),
        report["validation_loss"],
        "v2 disposable validation loss",
    )
    _require_close(
        float(diagnostics["iwelbo"]),
        report["validation_iwelbo"],
        "v2 disposable validation IWELBO",
    )
    _require_close(
        exact,
        report["exact_truth_log_joint_first_validation"],
        "v2 disposable exact target",
        rtol=1e-10,
        atol=1e-8,
    )

    tf.keras.utils.set_random_seed(MODEL_SEED)
    untrained = GenerativeIidOrbitPosteriorModel()
    untrained(validation_batch.model_inputs(), training=False, refine=False)
    if len(loaded.model.weights) != len(untrained.weights):
        raise ValueError("v2 disposable checkpoint weight count differs")
    maximum_weight_change = max(
        float(np.max(np.abs(np.asarray(trained) - np.asarray(initial))))
        for trained, initial in zip(loaded.model.weights, untrained.weights)
    )
    if maximum_weight_change <= 1e-8:
        raise ValueError("v2 disposable optimizer did not change weights")
    validation_record = {
        "status": "generative_iid_v2_disposable_validation_passed",
        "freeze_sha256": file_sha256(output / "freeze.json"),
        "checkpoint_content_sha256": manifest["content_sha256"],
        "weights_sha256": manifest["artifacts"]["weights"]["sha256"],
        "recomputed_validation_loss": float(loss),
        "recomputed_validation_iwelbo": float(diagnostics["iwelbo"]),
        "recomputed_exact_truth_log_joint": exact,
        "maximum_weight_change_from_seeded_initialization": (maximum_weight_change),
        "source_commit": expected_source_commit,
        "disposable_seed_ranges_opened": True,
        "production_511m_opened": False,
        "fixed_validation_512m_opened": False,
        "reserved_513m_515m_opened": False,
    }
    (output / "postfreeze_validation.json").write_text(
        json.dumps(validation_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation_record


def _factorial_cells() -> list[dict[str, object]]:
    combinations = list(
        itertools.product(
            ("normal", "right_skewed"),
            ("weak", "medium", "strong"),
            ("rare", "moderate", "common"),
        )
    )
    shapes = ((24, 12), (40, 36), (96, 75))
    return [
        {
            "index": index,
            "n_sites": shapes[index % len(shapes)][0],
            "n_species": shapes[index % len(shapes)][1],
            "covariate_shape": combination[0],
            "loading_stratum": combination[1],
            "prevalence_stratum": combination[2],
        }
        for index, combination in enumerate(combinations)
    ]


def _generate_block(seeds: tuple[int, ...], *, masked: bool) -> list[Any]:
    cells = _factorial_cells()
    if len(cells) != len(seeds):
        raise AssertionError("v2 disposable factorial differs")
    datasets = []
    for seed, cell in zip(seeds, cells):
        n_sites = int(cell["n_sites"])
        n_species = int(cell["n_species"])
        response_mask = (
            make_stratified_response_mask(n_sites, n_species, seed=seed)
            if masked
            else None
        )
        datasets.append(
            simulate_generative_iid_dataset(
                n_sites=n_sites,
                n_species=n_species,
                covariate_shape=str(cell["covariate_shape"]),
                loading_stratum=str(cell["loading_stratum"]),
                prevalence_stratum=str(cell["prevalence_stratum"]),
                seed=seed,
                response_mask=response_mask,
            )
        )
    return datasets


def _corpus_manifest(train: list[Any], validation: list[Any]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "generative_iid_v2_disposable_corpus",
        "protocol": GENERATIVE_IID_V2_PROTOCOL,
        "training_seed_range": [
            TRAINING_SEEDS[0],
            TRAINING_SEEDS[-1],
        ],
        "validation_seed_range": [
            VALIDATION_SEEDS[0],
            VALIDATION_SEEDS[-1],
        ],
        "training": _corpus_records(train, role="training"),
        "validation": _corpus_records(validation, role="validation"),
        "disposable_seed_ranges_opened": True,
        "production_511m_opened": False,
        "fixed_validation_512m_opened": False,
        "reserved_513m_515m_opened": False,
    }


def _corpus_records(datasets: list[Any], *, role: str) -> list[dict[str, object]]:
    return [
        {
            "role": role,
            "index": index,
            "seed": int(dataset.metadata["seed"]),
            "n_sites": int(dataset.X.shape[0]),
            "n_species": int(dataset.Y.shape[1]),
            "covariate_shape": dataset.metadata["covariate_shape"],
            "loading_stratum": dataset.metadata["loading_stratum"],
            "prevalence_stratum": dataset.metadata["prevalence_stratum"],
            "dataset_sha256": _dataset_sha256(dataset),
        }
        for index, dataset in enumerate(datasets)
    ]


def _dataset_sha256(dataset: Any) -> str:
    digest = hashlib.sha256()
    for name in (
        "X",
        "Y",
        "response_mask",
        "truth_beta",
        "truth_eta",
        "truth_lambda",
        "probabilities",
    ):
        value = np.ascontiguousarray(getattr(dataset, name))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    for value in (
        dataset.truth_alpha,
        dataset.truth_log_tau,
    ):
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _validate_corpus_manifest(corpus: dict[str, object]) -> None:
    if corpus.get("kind") != "generative_iid_v2_disposable_corpus":
        raise ValueError("v2 disposable corpus kind differs")
    if corpus.get("protocol") != GENERATIVE_IID_V2_PROTOCOL:
        raise ValueError("v2 disposable corpus protocol differs")
    if corpus.get("training_seed_range") != [
        TRAINING_SEEDS[0],
        TRAINING_SEEDS[-1],
    ]:
        raise ValueError("v2 disposable training seed range differs")
    if corpus.get("validation_seed_range") != [
        VALIDATION_SEEDS[0],
        VALIDATION_SEEDS[-1],
    ]:
        raise ValueError("v2 disposable validation seed range differs")
    if len(corpus.get("training", [])) != 18:
        raise ValueError("v2 disposable training corpus count differs")
    if len(corpus.get("validation", [])) != 18:
        raise ValueError("v2 disposable validation corpus count differs")
    _require_exact_seed_flags(corpus, disposable=True, context="v2 disposable corpus")


def _validate_freeze_inventory(output: Path, freeze: dict[str, object]) -> None:
    if freeze.get("kind") != "generative_iid_v2_disposable_freeze":
        raise ValueError("v2 disposable freeze kind differs")
    if freeze.get("protocol") != GENERATIVE_IID_V2_PROTOCOL:
        raise ValueError("v2 disposable freeze protocol differs")
    expected = {
        "corpus_manifest",
        "report",
        "checkpoint_manifest",
        "checkpoint_weights",
    }
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("v2 disposable freeze inventory differs")
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError(f"v2 disposable {name} record differs")
        path = output / str(record.get("path", ""))
        if not path.is_file():
            raise ValueError(f"v2 disposable {name} is missing")
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"v2 disposable {name} hash differs")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"v2 disposable {name} size differs")
    allowed_root = {
        "checkpoint",
        "corpus_manifest.json",
        "disposable_smoke_report.json",
        "freeze.json",
        "postfreeze_validation.json",
    }
    if not {path.name for path in output.iterdir()}.issubset(allowed_root):
        raise ValueError("v2 disposable output contains unknown artifacts")


def _artifact_record(path: Path, *, output: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(output)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _require_exact_seed_flags(
    payload: dict[str, object],
    *,
    disposable: bool | None,
    context: str,
) -> None:
    expected = {
        "production_511m_opened": False,
        "fixed_validation_512m_opened": False,
        "reserved_513m_515m_opened": False,
    }
    if disposable is not None:
        expected["disposable_seed_ranges_opened"] = disposable
    for key, value in expected.items():
        if payload.get(key) is not value:
            raise ValueError(f"{context} has invalid {key}")


def _truth_state_for_dataset(dataset: Any) -> tf.Tensor:
    return tf.convert_to_tensor(
        np.concatenate(
            [
                [dataset.truth_alpha],
                dataset.truth_beta.ravel(),
                dataset.truth_eta.ravel(),
                dataset.truth_lambda.ravel(),
                [dataset.truth_log_tau],
            ]
        )[None, :],
        dtype=tf.float32,
    )


def _validate_frozen_documents() -> None:
    expected = {
        PREREGISTRATION: GENERATIVE_IID_V2_PREREGISTRATION_SHA256,
        SEED_AUDIT: GENERATIVE_IID_V2_SEED_AUDIT_SHA256,
        REPRESENTATION_DECISION: REPRESENTATION_DECISION_SHA256,
        IMPLEMENTATION_EVIDENCE: IMPLEMENTATION_EVIDENCE_SHA256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"frozen v2 document hash differs: {path}")


def _require_no_opening_tokens() -> None:
    present = sorted(
        name
        for name, value in os.environ.items()
        if name.startswith("OPEN_GENERATIVE_IID") and value
    )
    if present:
        raise RuntimeError(
            "all generative opening tokens must remain unset during "
            f"preflight: {present}"
        )


def _require_disposable_token_only() -> None:
    present = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("OPEN_GENERATIVE_IID") and value
    }
    if present.get(CONFIRMATION_ENV) != CONFIRMATION_VALUE:
        raise RuntimeError(f"{CONFIRMATION_ENV} must equal {CONFIRMATION_VALUE!r}")
    extra = sorted(set(present).difference({CONFIRMATION_ENV}))
    if extra:
        raise RuntimeError(f"unrelated generative opening tokens are present: {extra}")


def _require_clean_pinned_source(expected_source_commit: str) -> str:
    if len(expected_source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_source_commit
    ):
        raise RuntimeError("expected source commit must be a full SHA-1")
    commit, _, worktree_dirty = _source_control_state()
    if commit != expected_source_commit:
        raise RuntimeError("v2 disposable source commit differs")
    if worktree_dirty:
        raise RuntimeError("v2 disposable source worktree must be clean")
    return commit


def _source_file_inventory() -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "sha256": file_sha256(ROOT / path),
            "bytes": (ROOT / path).stat().st_size,
        }
        for path in SOURCE_PATHS
    ]


def _source_provenance(source_commit: str) -> dict[str, object]:
    observed_commit, branch, worktree_dirty = _source_control_state()
    if observed_commit != source_commit or worktree_dirty:
        raise RuntimeError("v2 source changed after authorization")
    try:
        import tensorflow_probability as tfp

        tfp_version = str(tfp.__version__)
    except ImportError:
        tfp_version = "unavailable"
    return {
        "commit": source_commit,
        "branch": branch,
        "worktree_dirty": False,
        "source_files": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in _source_file_inventory()
        ],
        "environment": {
            "python": sys.version.split()[0],
            "tensorflow": str(tf.__version__),
            "tensorflow_probability": tfp_version,
            "numpy": str(np.__version__),
            "platform": platform.platform(),
        },
    }


def _source_control_state() -> tuple[str, str, bool]:
    """Read Git state, or consume a strict clean host attestation."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, branch, bool(status.strip())
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        commit = os.environ.get(HOST_SOURCE_COMMIT_ENV, "")
        branch = os.environ.get(HOST_SOURCE_BRANCH_ENV, "")
        clean = os.environ.get(HOST_WORKTREE_CLEAN_ENV)
        if (
            len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or clean != "1"
        ):
            raise RuntimeError(
                "Git is unavailable and the clean v2 host-source "
                "attestation is absent or invalid"
            ) from error
        return commit, branch, False


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require_close(
    observed: float,
    expected: object,
    label: str,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-4,
) -> None:
    if not np.isclose(observed, float(expected), rtol=rtol, atol=atol):
        raise ValueError(f"{label} differs")


if __name__ == "__main__":
    main()
