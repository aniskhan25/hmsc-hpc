#!/usr/bin/env python3
"""Sealed production harness for generative Neural-HMSC iid probit v1."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "pyhmsc-cache"),
)

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.generative_iid import (  # noqa: E402
    GENERATIVE_IID_DESIGN_REVIEW_SHA256,
    GENERATIVE_IID_PREREGISTRATION_SHA256,
    GENERATIVE_IID_SEED_AUDIT_SHA256,
    GenerativeIidPosteriorModel,
    batch_generative_iid_datasets,
    simulate_generative_iid_dataset,
    train_generative_iid_model,
    train_generative_iid_no_latent_ablation,
)
from pyhmsc.neural.generative_iid_artifact import (  # noqa: E402
    GenerativeIidInference,
    file_sha256,
    validate_generative_iid_checkpoint,
)
from pyhmsc.neural.generative_iid import (  # noqa: E402
    make_stratified_response_mask,
    posterior_mean_invariants,
)
from examples.run_generative_neural_hmsc_iid_v1 import (  # noqa: E402
    _source_control_state,
    _source_provenance,
    _validate_frozen_documents,
)


PROTOCOL = "generative_neural_hmsc_iid_probit_v1"
TRAIN_CONFIRMATION_ENV = "OPEN_GENERATIVE_IID_501M_TRAINING"
TRAIN_CONFIRMATION = "GENERATE_501M_CANDIDATE_TRAINING_ONLY"
VALIDATION_CONFIRMATION_ENV = "OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION"
VALIDATION_CONFIRMATION = "EVALUATE_502M_FIXED_VALIDATION_ONCE"

TRAINING_SEEDS = tuple(range(501_000_001, 501_000_325))
FIXED_VALIDATION_SEEDS = tuple(range(502_000_001, 502_000_325))
RESERVED_SEED_RANGES = (
    (503_000_001, 503_000_324),
    (504_000_001, 504_000_324),
    (505_000_001, 505_000_324),
)
REDESIGN_SEED_RANGES = (
    (511_000_001, 511_000_324),
    (512_000_001, 512_000_324),
    (513_000_001, 513_000_324),
    (514_000_001, 514_000_324),
    (515_000_001, 515_000_324),
)
MODEL_SEED = 501_900_001
TRAINING_EPOCHS = 200
TRAINING_BATCH_SIZE = 4
TRAINING_RESPONSES_PER_CONTEXT = 2

PRODUCTION_SOURCE_PATHS = (
    "pyhmsc/neural/generative_iid.py",
    "pyhmsc/neural/generative_iid_mcmc.py",
    "pyhmsc/neural/generative_iid_artifact.py",
    "pyhmsc/neural/generative_iid_evaluation.py",
    "pyhmsc/neural/generative_iid_comparators.py",
    "pyhmsc/neural/__init__.py",
    "examples/run_generative_neural_hmsc_iid_v1.py",
    "examples/run_generative_neural_hmsc_iid_v1_production.py",
    "docs/lumi_generative_neural_hmsc_iid_v1_training_sbatch.sh",
    "docs/lumi_generative_neural_hmsc_iid_v1_fixed_validation_sbatch.sh",
    "docs/generative_neural_hmsc_iid_v1_preregistration_2026-07-27.md",
    "docs/generative_neural_hmsc_iid_v1_seed_audit_2026-07-27.json.md",
    "docs/generative_neural_hmsc_iid_v1_design_review_2026-07-27.md",
)

REQUIRED_502_EVALUATOR_COMPONENTS = (
    "candidate_256_draw_posterior_metrics",
    "fixed_no_latent_ablation_trained_on_501m",
    "exact_model_mcmc_36_context_comparator",
    "qualified_python_hmsc_hpc_36_context_comparator",
    "immutable_neural_hmsc_v0_1_matched_cell_comparator",
    "permutation_and_padding_invariance",
    "masked_cell_and_new_site_predictive_scores",
    "posterior_predictive_richness_and_prevalence",
    "runtime_and_peak_device_memory",
    "all_preregistered_aggregate_and_stratum_gates",
)
FIXED_VALIDATION_EVALUATOR_VERSION = "generative_iid_v1_502_evaluator_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-seal")

    preflight_training = subparsers.add_parser("preflight-training")
    preflight_training.add_argument("--expected-source-commit", required=True)

    train = subparsers.add_parser("train-candidate")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--expected-source-commit", required=True)

    validate = subparsers.add_parser("validate-training")
    validate.add_argument("--freeze-root", type=Path, required=True)
    validate.add_argument("--expected-source-commit", required=True)

    preflight = subparsers.add_parser("preflight-fixed-validation")
    preflight.add_argument("--freeze-root", type=Path, required=True)
    preflight.add_argument("--expected-source-commit", required=True)
    preflight.add_argument("--expected-checkpoint-content-sha256", required=True)
    preflight.add_argument("--expected-ablation-content-sha256", required=True)

    fixed = subparsers.add_parser("fixed-validation")
    fixed.add_argument("--freeze-root", type=Path, required=True)
    fixed.add_argument("--output", type=Path, required=True)
    fixed.add_argument("--expected-source-commit", required=True)
    fixed.add_argument("--expected-checkpoint-content-sha256", required=True)
    fixed.add_argument("--expected-ablation-content-sha256", required=True)
    fixed.add_argument("--release-registry", type=Path, required=True)
    fixed.add_argument("--python", default=sys.executable)

    validate_fixed = subparsers.add_parser("validate-fixed-validation")
    validate_fixed.add_argument("--root", type=Path, required=True)
    validate_fixed.add_argument("--expected-source-commit", required=True)
    validate_fixed.add_argument(
        "--expected-training-freeze-sha256",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_frozen_documents()
    if args.command == "check-seal":
        result = production_seal_status()
    elif args.command == "preflight-training":
        result = preflight_candidate_training(
            expected_source_commit=args.expected_source_commit,
        )
    elif args.command == "train-candidate":
        result = train_candidate(
            args.output,
            expected_source_commit=args.expected_source_commit,
        )
    elif args.command == "validate-training":
        result = validate_training_freeze(
            args.freeze_root,
            expected_source_commit=args.expected_source_commit,
        )
    elif args.command == "preflight-fixed-validation":
        result = preflight_fixed_validation(
            args.freeze_root,
            expected_source_commit=args.expected_source_commit,
            expected_checkpoint_content_sha256=(
                args.expected_checkpoint_content_sha256
            ),
            expected_ablation_content_sha256=(
                args.expected_ablation_content_sha256
            ),
        )
    elif args.command == "fixed-validation":
        result = run_fixed_validation(
            args.freeze_root,
            args.output,
            expected_source_commit=args.expected_source_commit,
            expected_checkpoint_content_sha256=(
                args.expected_checkpoint_content_sha256
            ),
            expected_ablation_content_sha256=(
                args.expected_ablation_content_sha256
            ),
            release_registry=args.release_registry,
            python=args.python,
        )
    else:
        result = validate_fixed_validation_freeze(
            args.root,
            expected_source_commit=args.expected_source_commit,
            expected_training_freeze_sha256=(
                args.expected_training_freeze_sha256
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def production_seal_status() -> dict[str, Any]:
    """Describe all production barriers without reading a production seed."""
    return {
        "protocol": PROTOCOL,
        "candidate_training_seed_range": [
            TRAINING_SEEDS[0],
            TRAINING_SEEDS[-1],
        ],
        "fixed_validation_seed_range": [
            FIXED_VALIDATION_SEEDS[0],
            FIXED_VALIDATION_SEEDS[-1],
        ],
        "reserved_seed_ranges": [list(value) for value in RESERVED_SEED_RANGES],
        "redesign_seed_ranges": [list(value) for value in REDESIGN_SEED_RANGES],
        "candidate_training_opened": False,
        "fixed_validation_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
        "training_confirmation_env": TRAIN_CONFIRMATION_ENV,
        "training_confirmation_value": TRAIN_CONFIRMATION,
        "fixed_validation_confirmation_env": VALIDATION_CONFIRMATION_ENV,
        "fixed_validation_confirmation_value": VALIDATION_CONFIRMATION,
        "fixed_validation_executable": True,
        "fixed_validation_evaluator_version": (
            FIXED_VALIDATION_EVALUATOR_VERSION
        ),
        "fixed_validation_requires_separate_confirmation": True,
    }


def preflight_candidate_training(
    *,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Validate the 501M boundary without reading a production seed."""
    for name in (TRAIN_CONFIRMATION_ENV, VALIDATION_CONFIRMATION_ENV):
        if os.environ.get(name):
            raise RuntimeError(f"{name} must remain unset during preflight")
    source_commit = _require_clean_pinned_source(expected_source_commit)
    source_files = [
        {"path": path, "sha256": file_sha256(ROOT / path)}
        for path in PRODUCTION_SOURCE_PATHS
    ]
    return {
        "status": "candidate_training_preflight_sealed",
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "source_files": source_files,
        "candidate_training_seed_range": [
            TRAINING_SEEDS[0],
            TRAINING_SEEDS[-1],
        ],
        "candidate_training_context_count": len(TRAINING_SEEDS),
        "responses_per_context": TRAINING_RESPONSES_PER_CONTEXT,
        "training_realization_count": (
            len(TRAINING_SEEDS) * TRAINING_RESPONSES_PER_CONTEXT
        ),
        "candidate_and_ablation_epochs": TRAINING_EPOCHS,
        "batch_size": TRAINING_BATCH_SIZE,
        "model_seed": MODEL_SEED,
        "candidate_training_opened": False,
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
        "fixed_validation_executable": True,
        "fixed_validation_evaluator_version": (
            FIXED_VALIDATION_EVALUATOR_VERSION
        ),
        "reviewed_fixed_validation_components": list(
            REQUIRED_502_EVALUATOR_COMPONENTS
        ),
        "decision": (
            "501M may be explicitly authorized from this exact clean commit; "
            "502M-515M remain sealed"
        ),
    }


def train_candidate(
    output: Path,
    *,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Open only 501M and freeze the final-epoch production checkpoint."""
    _require_confirmation(
        TRAIN_CONFIRMATION_ENV,
        TRAIN_CONFIRMATION,
        action="501M candidate training",
    )
    source_commit = _require_clean_pinned_source(expected_source_commit)
    output = _empty_output(output)
    started = time.perf_counter()

    owners = _generate_production_block(TRAINING_SEEDS, masked=False)
    training = [
        simulate_generative_iid_dataset(
            n_sites=int(owner.metadata["n_sites"]),
            n_species=int(owner.metadata["n_species"]),
            covariate_shape=str(owner.metadata["covariate_shape"]),
            loading_stratum=str(owner.metadata["loading_stratum"]),
            prevalence_stratum=str(owner.metadata["prevalence_stratum"]),
            seed=int(owner.metadata["seed"]),
            response_realization=response_realization,
        )
        for owner in owners
        for response_realization in range(TRAINING_RESPONSES_PER_CONTEXT)
    ]
    _validate_training_corpus(owners, training)
    batch = batch_generative_iid_datasets(
        training,
        max_sites=96,
        max_species=75,
    )

    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = GenerativeIidPosteriorModel()
    history = train_generative_iid_model(
        model,
        batch,
        epochs=TRAINING_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        model_seed=MODEL_SEED,
    )
    tf.keras.utils.set_random_seed(MODEL_SEED)
    ablation_model = GenerativeIidPosteriorModel()
    ablation_history = train_generative_iid_no_latent_ablation(
        ablation_model,
        batch,
        epochs=TRAINING_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        model_seed=MODEL_SEED,
    )
    elapsed = time.perf_counter() - started
    if not all(
        np.isfinite(np.asarray(values)).all()
        for values in (
            history.loss,
            history.iwelbo,
            history.gradient_norm,
            ablation_history.loss,
            ablation_history.iwelbo,
            ablation_history.gradient_norm,
        )
    ):
        raise FloatingPointError("501M training history is non-finite")

    source_provenance = _source_provenance(source_commit)
    source_provenance["source_files"] = [
        {"path": path, "sha256": file_sha256(ROOT / path)}
        for path in PRODUCTION_SOURCE_PATHS
    ]
    if source_provenance["worktree_dirty"] is not False:
        raise RuntimeError("production source became dirty during training")

    training_manifest = {
        "role": "candidate_production_training_501m_only",
        "owning_seed_range": [TRAINING_SEEDS[0], TRAINING_SEEDS[-1]],
        "owning_context_count": len(owners),
        "responses_per_context": TRAINING_RESPONSES_PER_CONTEXT,
        "training_realization_count": len(training),
        "model_seed": MODEL_SEED,
        "epochs": TRAINING_EPOCHS,
        "batch_size": TRAINING_BATCH_SIZE,
        "importance_draws": 8,
        "final_epoch_weights": True,
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
    }
    inference = GenerativeIidInference(
        model=model,
        manifest={},
        checkpoint_root=None,
    )
    checkpoint = inference.save(
        output / "checkpoint",
        source_commit=source_commit,
        source_provenance=source_provenance,
        training_manifest=training_manifest,
    )
    manifest = validate_generative_iid_checkpoint(checkpoint)
    ablation_training_manifest = {
        **training_manifest,
        "role": "no_latent_ablation_501m_only",
        "response_likelihood_random_effect": "R_equals_zero",
        "same_architecture_as_candidate": True,
    }
    ablation_inference = GenerativeIidInference(
        model=ablation_model,
        manifest={},
        checkpoint_root=None,
    )
    ablation_checkpoint = ablation_inference.save(
        output / "no_latent_ablation_checkpoint",
        source_commit=source_commit,
        source_provenance=source_provenance,
        training_manifest=ablation_training_manifest,
    )
    ablation_manifest = validate_generative_iid_checkpoint(
        ablation_checkpoint
    )

    corpus_manifest = {
        "protocol": PROTOCOL,
        "role": "candidate_production_training_501m",
        "owning_seed_range": [TRAINING_SEEDS[0], TRAINING_SEEDS[-1]],
        "factorial": _factorial_summary(),
        "owning_context_count": len(owners),
        "responses_per_context": TRAINING_RESPONSES_PER_CONTEXT,
        "training_realization_count": len(training),
        "metadata_sha256": _metadata_sha256(owners),
        "response_metadata_sha256": _metadata_sha256(training),
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
    }
    corpus_path = output / "training_corpus_manifest.json"
    _write_json(corpus_path, corpus_manifest)

    report = {
        "protocol": PROTOCOL,
        "status": "candidate_501m_training_complete",
        "promotion_evidence": False,
        "source_commit": source_commit,
        "checkpoint_content_sha256": manifest["content_sha256"],
        "checkpoint_manifest_sha256": file_sha256(
            checkpoint / "generative_iid_checkpoint.json"
        ),
        "weights_sha256": manifest["artifacts"]["weights"]["sha256"],
        "no_latent_ablation_content_sha256": ablation_manifest[
            "content_sha256"
        ],
        "no_latent_ablation_manifest_sha256": file_sha256(
            ablation_checkpoint / "generative_iid_checkpoint.json"
        ),
        "no_latent_ablation_weights_sha256": ablation_manifest["artifacts"][
            "weights"
        ]["sha256"],
        "training_corpus_manifest_sha256": file_sha256(corpus_path),
        "training": training_manifest,
        "final_training_loss": history.loss[-1],
        "final_training_iwelbo": history.iwelbo[-1],
        "final_gradient_norm": history.gradient_norm[-1],
        "no_latent_ablation_final_loss": ablation_history.loss[-1],
        "no_latent_ablation_final_iwelbo": ablation_history.iwelbo[-1],
        "no_latent_ablation_final_gradient_norm": (
            ablation_history.gradient_norm[-1]
        ),
        "wall_time_seconds": elapsed,
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
    }
    report_path = output / "training_report.json"
    _write_json(report_path, report)
    freeze = {
        "protocol": PROTOCOL,
        "kind": "generative_iid_v1_501m_training_freeze",
        "source_commit": source_commit,
        "preregistration_sha256": GENERATIVE_IID_PREREGISTRATION_SHA256,
        "seed_audit_sha256": GENERATIVE_IID_SEED_AUDIT_SHA256,
        "design_review_sha256": GENERATIVE_IID_DESIGN_REVIEW_SHA256,
        "checkpoint": {
            "path": "checkpoint",
            "manifest_sha256": report["checkpoint_manifest_sha256"],
            "content_sha256": report["checkpoint_content_sha256"],
            "weights_sha256": report["weights_sha256"],
        },
        "no_latent_ablation_checkpoint": {
            "path": "no_latent_ablation_checkpoint",
            "manifest_sha256": report[
                "no_latent_ablation_manifest_sha256"
            ],
            "content_sha256": report[
                "no_latent_ablation_content_sha256"
            ],
            "weights_sha256": report[
                "no_latent_ablation_weights_sha256"
            ],
        },
        "training_corpus_manifest": {
            "path": corpus_path.name,
            "sha256": file_sha256(corpus_path),
        },
        "training_report": {
            "path": report_path.name,
            "sha256": file_sha256(report_path),
        },
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
    }
    freeze_path = output / "freeze.json"
    _write_json(freeze_path, freeze)
    (output / "freeze.sha256").write_text(
        file_sha256(freeze_path) + "\n",
        encoding="utf-8",
    )
    return validate_training_freeze(
        output,
        expected_source_commit=source_commit,
    )


def validate_training_freeze(
    freeze_root: Path,
    *,
    expected_source_commit: str,
    write_validation: bool = True,
) -> dict[str, Any]:
    """Independently validate a 501M freeze without generating any dataset."""
    root = freeze_root.expanduser().resolve()
    freeze_path = root / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("protocol") != PROTOCOL:
        raise ValueError("501M freeze protocol differs")
    if freeze.get("kind") != "generative_iid_v1_501m_training_freeze":
        raise ValueError("501M freeze kind differs")
    if freeze.get("source_commit") != expected_source_commit:
        raise ValueError("501M freeze source commit differs")
    _require_false_seed_flags(freeze)
    freeze_sha256 = file_sha256(freeze_path)
    recorded_freeze_sha256 = (
        root / "freeze.sha256"
    ).read_text(encoding="utf-8").strip()
    if recorded_freeze_sha256 != freeze_sha256:
        raise ValueError("501M freeze sidecar hash differs")
    expected_document_hashes = {
        "preregistration_sha256": GENERATIVE_IID_PREREGISTRATION_SHA256,
        "seed_audit_sha256": GENERATIVE_IID_SEED_AUDIT_SHA256,
        "design_review_sha256": GENERATIVE_IID_DESIGN_REVIEW_SHA256,
    }
    for key, expected in expected_document_hashes.items():
        if freeze.get(key) != expected:
            raise ValueError(f"501M frozen document binding differs: {key}")

    checkpoint = root / str(freeze["checkpoint"]["path"])
    manifest_path = checkpoint / "generative_iid_checkpoint.json"
    manifest = validate_generative_iid_checkpoint(checkpoint)
    if file_sha256(manifest_path) != freeze["checkpoint"]["manifest_sha256"]:
        raise ValueError("501M checkpoint manifest hash differs")
    if manifest["content_sha256"] != freeze["checkpoint"]["content_sha256"]:
        raise ValueError("501M checkpoint content hash differs")
    if manifest["artifacts"]["weights"]["sha256"] != (
        freeze["checkpoint"]["weights_sha256"]
    ):
        raise ValueError("501M checkpoint weight hash differs")
    ablation_checkpoint = root / str(
        freeze["no_latent_ablation_checkpoint"]["path"]
    )
    ablation_manifest_path = (
        ablation_checkpoint / "generative_iid_checkpoint.json"
    )
    ablation_manifest = validate_generative_iid_checkpoint(
        ablation_checkpoint
    )
    ablation_record = freeze["no_latent_ablation_checkpoint"]
    if file_sha256(ablation_manifest_path) != (
        ablation_record["manifest_sha256"]
    ):
        raise ValueError("501M ablation checkpoint manifest hash differs")
    if ablation_manifest["content_sha256"] != (
        ablation_record["content_sha256"]
    ):
        raise ValueError("501M ablation checkpoint content hash differs")
    if ablation_manifest["artifacts"]["weights"]["sha256"] != (
        ablation_record["weights_sha256"]
    ):
        raise ValueError("501M ablation checkpoint weight hash differs")
    training = manifest.get("training_manifest")
    expected_training = {
        "role": "candidate_production_training_501m_only",
        "owning_seed_range": [TRAINING_SEEDS[0], TRAINING_SEEDS[-1]],
        "owning_context_count": 324,
        "responses_per_context": 2,
        "training_realization_count": 648,
        "model_seed": MODEL_SEED,
        "epochs": TRAINING_EPOCHS,
        "batch_size": TRAINING_BATCH_SIZE,
        "importance_draws": 8,
        "final_epoch_weights": True,
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
    }
    if training != expected_training:
        raise ValueError("501M checkpoint training contract differs")
    expected_ablation_training = {
        **expected_training,
        "role": "no_latent_ablation_501m_only",
        "response_likelihood_random_effect": "R_equals_zero",
        "same_architecture_as_candidate": True,
    }
    if ablation_manifest.get("training_manifest") != (
        expected_ablation_training
    ):
        raise ValueError("501M no-latent ablation contract differs")
    provenance = manifest["source_provenance"]
    if provenance.get("commit") != expected_source_commit:
        raise ValueError("501M source-provenance commit differs")
    if provenance.get("worktree_dirty") is not False:
        raise ValueError("501M checkpoint was not produced from clean source")
    ablation_provenance = ablation_manifest["source_provenance"]
    if ablation_provenance.get("commit") != expected_source_commit:
        raise ValueError("501M ablation source-provenance commit differs")
    if ablation_provenance.get("worktree_dirty") is not False:
        raise ValueError(
            "501M ablation checkpoint was not produced from clean source"
        )
    source_inventory = {
        record["path"]: record["sha256"]
        for record in provenance["source_files"]
    }
    if set(source_inventory) != set(PRODUCTION_SOURCE_PATHS):
        raise ValueError("501M source inventory differs")
    ablation_source_inventory = {
        record["path"]: record["sha256"]
        for record in ablation_provenance["source_files"]
    }
    if ablation_source_inventory != source_inventory:
        raise ValueError("501M candidate and ablation source inventories differ")

    for key in ("training_corpus_manifest", "training_report"):
        record = freeze[key]
        path = root / str(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"501M {key} hash differs")
    corpus = json.loads(
        (root / freeze["training_corpus_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if corpus.get("owning_seed_range") != [
        TRAINING_SEEDS[0],
        TRAINING_SEEDS[-1],
    ]:
        raise ValueError("501M corpus seed range differs")
    _require_false_seed_flags(corpus)
    expected_corpus = {
        "protocol": PROTOCOL,
        "role": "candidate_production_training_501m",
        "factorial": _factorial_summary(),
        "owning_context_count": len(TRAINING_SEEDS),
        "responses_per_context": TRAINING_RESPONSES_PER_CONTEXT,
        "training_realization_count": (
            len(TRAINING_SEEDS) * TRAINING_RESPONSES_PER_CONTEXT
        ),
    }
    for key, expected in expected_corpus.items():
        if corpus.get(key) != expected:
            raise ValueError(f"501M corpus contract differs: {key}")
    for key in ("metadata_sha256", "response_metadata_sha256"):
        digest = corpus.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"501M corpus digest differs: {key}")
    report = json.loads(
        (root / freeze["training_report"]["path"]).read_text(encoding="utf-8")
    )
    _require_false_seed_flags(report)
    expected_report = {
        "protocol": PROTOCOL,
        "status": "candidate_501m_training_complete",
        "promotion_evidence": False,
        "source_commit": expected_source_commit,
        "checkpoint_content_sha256": manifest["content_sha256"],
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "weights_sha256": manifest["artifacts"]["weights"]["sha256"],
        "no_latent_ablation_content_sha256": ablation_manifest[
            "content_sha256"
        ],
        "no_latent_ablation_manifest_sha256": file_sha256(
            ablation_manifest_path
        ),
        "no_latent_ablation_weights_sha256": ablation_manifest["artifacts"][
            "weights"
        ]["sha256"],
        "training_corpus_manifest_sha256": file_sha256(
            root / freeze["training_corpus_manifest"]["path"]
        ),
        "training": expected_training,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise ValueError(f"501M training report binding differs: {key}")
    metric_keys = (
        "final_training_loss",
        "final_training_iwelbo",
        "final_gradient_norm",
        "no_latent_ablation_final_loss",
        "no_latent_ablation_final_iwelbo",
        "no_latent_ablation_final_gradient_norm",
        "wall_time_seconds",
    )
    for key in metric_keys:
        value = report.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"501M training report metric differs: {key}")
    if float(report["wall_time_seconds"]) <= 0.0:
        raise ValueError("501M training wall time must be positive")

    result = {
        "status": "candidate_501m_training_freeze_valid",
        "freeze_sha256": freeze_sha256,
        "freeze_sidecar_sha256": recorded_freeze_sha256,
        "source_commit": expected_source_commit,
        "checkpoint_content_sha256": manifest["content_sha256"],
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "weights_sha256": manifest["artifacts"]["weights"]["sha256"],
        "no_latent_ablation_content_sha256": ablation_manifest[
            "content_sha256"
        ],
        "no_latent_ablation_manifest_sha256": file_sha256(
            ablation_manifest_path
        ),
        "no_latent_ablation_weights_sha256": ablation_manifest["artifacts"][
            "weights"
        ]["sha256"],
        "training_corpus_manifest_sha256": file_sha256(
            root / freeze["training_corpus_manifest"]["path"]
        ),
        "training_report_sha256": file_sha256(
            root / freeze["training_report"]["path"]
        ),
        "training_metrics_finite": True,
        "candidate_and_ablation_source_inventory_match": True,
        "training_seed_range": [TRAINING_SEEDS[0], TRAINING_SEEDS[-1]],
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
    }
    if write_validation:
        _write_json(root / "postfreeze_validation.json", result)
    return result


def preflight_fixed_validation(
    freeze_root: Path,
    *,
    expected_source_commit: str,
    expected_checkpoint_content_sha256: str,
    expected_ablation_content_sha256: str,
) -> dict[str, Any]:
    """Validate the 502M boundary while leaving its seeds inaccessible."""
    training = validate_training_freeze(
        freeze_root,
        expected_source_commit=expected_source_commit,
        write_validation=False,
    )
    if training["checkpoint_content_sha256"] != (
        expected_checkpoint_content_sha256
    ):
        raise ValueError("502M preflight checkpoint content hash differs")
    if training["no_latent_ablation_content_sha256"] != (
        expected_ablation_content_sha256
    ):
        raise ValueError("502M preflight ablation content hash differs")
    if os.environ.get(VALIDATION_CONFIRMATION_ENV):
        raise RuntimeError(
            f"{VALIDATION_CONFIRMATION_ENV} must remain unset during preflight"
        )
    return {
        "status": "fixed_validation_preflight_sealed",
        "training_freeze_sha256": training["freeze_sha256"],
        "checkpoint_content_sha256": training["checkpoint_content_sha256"],
        "no_latent_ablation_content_sha256": training[
            "no_latent_ablation_content_sha256"
        ],
        "fixed_validation_seed_range": [
            FIXED_VALIDATION_SEEDS[0],
            FIXED_VALIDATION_SEEDS[-1],
        ],
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
        "fixed_validation_executable": True,
        "evaluator_version": FIXED_VALIDATION_EVALUATOR_VERSION,
        "reviewed_components": list(REQUIRED_502_EVALUATOR_COMPONENTS),
        "decision": (
            "502M may be separately authorized only after this preflight, "
            "an exact clean evaluator commit, and explicit user approval"
        ),
    }


def run_fixed_validation(
    freeze_root: Path,
    output: Path,
    *,
    expected_source_commit: str,
    expected_checkpoint_content_sha256: str,
    expected_ablation_content_sha256: str,
    release_registry: Path,
    python: str,
) -> dict[str, Any]:
    """Open 502M once and execute the complete preregistered gate report."""
    from pyhmsc.neural.generative_iid_comparators import (
        evaluate_exact_mcmc_contexts,
        evaluate_neural_contexts,
        evaluate_python_hmsc_contexts,
        evaluate_v0_1_contexts,
    )
    from pyhmsc.neural.generative_iid_evaluation import (
        fixed_mcmc_subset_seeds,
        fixed_validation_gates,
        qualification_report,
    )
    from pyhmsc.neural.release import NeuralHmscRelease

    _require_confirmation(
        VALIDATION_CONFIRMATION_ENV,
        VALIDATION_CONFIRMATION,
        action="502M fixed validation",
    )
    source_commit = _require_clean_pinned_source(expected_source_commit)
    training = validate_training_freeze(
        freeze_root,
        expected_source_commit=source_commit,
        write_validation=False,
    )
    if training["checkpoint_content_sha256"] != (
        expected_checkpoint_content_sha256
    ):
        raise ValueError("502M candidate checkpoint hash differs")
    if training["no_latent_ablation_content_sha256"] != (
        expected_ablation_content_sha256
    ):
        raise ValueError("502M ablation checkpoint hash differs")
    output = _empty_output(output)
    freeze_root = freeze_root.expanduser().resolve()
    training_freeze_path = freeze_root / "freeze.json"
    training_freeze = json.loads(
        training_freeze_path.read_text(encoding="utf-8")
    )

    validation_datasets = _generate_fixed_validation_block()
    candidate_inference = GenerativeIidInference.load(
        freeze_root / training_freeze["checkpoint"]["path"]
    )
    ablation_inference = GenerativeIidInference.load(
        freeze_root
        / training_freeze["no_latent_ablation_checkpoint"]["path"]
    )
    candidate_rows, candidate_operational = evaluate_neural_contexts(
        candidate_inference,
        validation_datasets,
        draws=256,
        zero_latent=False,
        method="generative_neural_hmsc_iid_v1",
    )
    ablation_rows, _ = evaluate_neural_contexts(
        ablation_inference,
        validation_datasets,
        draws=256,
        zero_latent=True,
        method="same_architecture_no_latent_ablation",
    )
    subset_seeds = set(fixed_mcmc_subset_seeds(candidate_rows))
    subset = [
        dataset
        for dataset in validation_datasets
        if int(dataset.metadata["seed"]) in subset_seeds
    ]
    exact_rows, mcmc_diagnostics, exact_seconds = (
        evaluate_exact_mcmc_contexts(
            subset,
            output_root=output / "exact_mcmc",
        )
    )
    python_rows, python_seconds = evaluate_python_hmsc_contexts(
        subset,
        output_root=output / "python_hmsc",
        python=python,
    )
    matched_v0 = [
        dataset
        for dataset in validation_datasets
        if dataset.Y.shape == (40, 75)
    ]
    release = NeuralHmscRelease.load(release_registry)
    v0_rows = evaluate_v0_1_contexts(release, matched_v0, draws=256)
    invariance = _production_invariance_checks(
        candidate_inference,
        validation_datasets,
    )

    training_report = json.loads(
        (
            freeze_root / training_freeze["training_report"]["path"]
        ).read_text(encoding="utf-8")
    )
    max_exact_seconds = max(
        (
            float(row["inference_seconds"])
            for row in exact_rows
            if int(row["n_sites"]) == 96 and int(row["n_species"]) == 75
        ),
        default=max(
            (float(row["inference_seconds"]) for row in exact_rows),
            default=0.0,
        ),
    )
    max_neural_seconds = max(
        float(candidate_operational["max_shape_inference_seconds"]),
        np.finfo(float).eps,
    )
    runtime = {
        "training_dev_gpu_hours": (
            float(training_report["wall_time_seconds"]) / 3600.0
        ),
        "max_shape_inference_seconds": candidate_operational[
            "max_shape_inference_seconds"
        ],
        "peak_device_memory_bytes": candidate_operational[
            "peak_device_memory_bytes"
        ],
        "speedup_vs_exact_mcmc": max_exact_seconds / max_neural_seconds,
        "exact_mcmc_total_seconds": exact_seconds,
        "python_hmsc_total_seconds": python_seconds,
    }
    operational = {
        "checkpoint_roundtrip": True,
        "permutation_invariance": (
            invariance["permutation_max_abs_delta"] <= 2e-5
        ),
        "padding_invariance": (
            invariance["padding_max_abs_delta"] <= 2e-5
        ),
        "dependency_inventory_clean": (
            candidate_inference.manifest["dependency_inventory"] == []
        ),
        "covariance_jitter_fraction": candidate_operational[
            "covariance_jitter_fraction"
        ],
        "covariance_condition_max": candidate_operational[
            "covariance_condition_max"
        ],
    }
    gates = fixed_validation_gates(
        candidate_rows,
        ablation_rows=ablation_rows,
        exact_rows=exact_rows,
        python_rows=python_rows,
        v0_rows=v0_rows,
        operational=operational,
        mcmc_diagnostics=mcmc_diagnostics,
        runtime=runtime,
    )

    metrics_path = output / "context_metrics.json.gz"
    with gzip.open(metrics_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate": candidate_rows,
                "no_latent_ablation": ablation_rows,
                "exact_model_mcmc": exact_rows,
                "qualified_python_hmsc_hpc": python_rows,
                "immutable_neural_hmsc_v0_1": v0_rows,
                "mcmc_diagnostics": mcmc_diagnostics,
                "invariance": invariance,
                "runtime": runtime,
            },
            handle,
            sort_keys=True,
        )
    report = qualification_report(
        gates=gates,
        freeze_binding={
            "training_freeze_sha256": file_sha256(training_freeze_path),
            "source_commit": source_commit,
            "candidate_checkpoint_content_sha256": (
                expected_checkpoint_content_sha256
            ),
            "ablation_checkpoint_content_sha256": (
                expected_ablation_content_sha256
            ),
            "evaluator_version": FIXED_VALIDATION_EVALUATOR_VERSION,
            "v0_1_release_id": release.release_id,
        },
        seed_roles={
            "fixed_validation": [
                FIXED_VALIDATION_SEEDS[0],
                FIXED_VALIDATION_SEEDS[-1],
            ],
            "context_count": len(validation_datasets),
            "exact_mcmc_subset_count": len(subset),
            "reserved_seed_ranges_opened": False,
            "redesign_seed_ranges_opened": False,
        },
        artifacts={
            "context_metrics": {
                "path": metrics_path.name,
                "sha256": file_sha256(metrics_path),
            },
            "exact_mcmc": _artifact_inventory(
                output / "exact_mcmc",
                relative_to=output,
            ),
            "python_hmsc": _artifact_inventory(
                output / "python_hmsc",
                relative_to=output,
            ),
            "immutable_v0_1_release": {
                "release_id": release.release_id,
                "content_sha256": release.manifest["content_sha256"],
            },
        },
    )
    report_path = output / "fixed_validation_report.json"
    _write_json(report_path, report)
    freeze = {
        "protocol": PROTOCOL,
        "kind": "generative_iid_v1_502m_fixed_validation_freeze",
        "source_commit": source_commit,
        "training_freeze_sha256": file_sha256(training_freeze_path),
        "report": {
            "path": report_path.name,
            "sha256": file_sha256(report_path),
        },
        "context_metrics": {
            "path": metrics_path.name,
            "sha256": file_sha256(metrics_path),
        },
        "all_gates_passed": report["all_gates_passed"],
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
    }
    freeze_path = output / "freeze.json"
    _write_json(freeze_path, freeze)
    return validate_fixed_validation_freeze(
        output,
        expected_source_commit=source_commit,
        expected_training_freeze_sha256=file_sha256(training_freeze_path),
    )


def validate_fixed_validation_freeze(
    root: Path,
    *,
    expected_source_commit: str,
    expected_training_freeze_sha256: str,
) -> dict[str, Any]:
    """Validate a completed 502M report without regenerating any simulation."""
    root = root.expanduser().resolve()
    freeze_path = root / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("protocol") != PROTOCOL:
        raise ValueError("502M freeze protocol differs")
    if freeze.get("kind") != (
        "generative_iid_v1_502m_fixed_validation_freeze"
    ):
        raise ValueError("502M freeze kind differs")
    if freeze.get("source_commit") != expected_source_commit:
        raise ValueError("502M source commit differs")
    if freeze.get("training_freeze_sha256") != (
        expected_training_freeze_sha256
    ):
        raise ValueError("502M training freeze binding differs")
    for key in ("reserved_seed_ranges_opened", "redesign_seed_ranges_opened"):
        if freeze.get(key) is not False:
            raise ValueError(f"502M freeze opened {key}")
    for key in ("report", "context_metrics"):
        record = freeze[key]
        if file_sha256(root / record["path"]) != record["sha256"]:
            raise ValueError(f"502M {key} hash differs")
    report = json.loads(
        (root / freeze["report"]["path"]).read_text(encoding="utf-8")
    )
    for name in ("exact_mcmc", "python_hmsc"):
        _validate_artifact_inventory(root, report["artifacts"][name])
    if report["all_gates_passed"] != all(report["gates"].values()):
        raise ValueError("502M gate decision differs")
    expected_decision = (
        "eligible_to_authorize_503m_505m"
        if report["all_gates_passed"]
        else "stop_before_reserved_evaluation"
    )
    if report["decision"] != expected_decision:
        raise ValueError("502M report decision differs")
    return {
        "status": "fixed_validation_freeze_valid",
        "freeze_sha256": file_sha256(freeze_path),
        "report_sha256": freeze["report"]["sha256"],
        "context_metrics_sha256": freeze["context_metrics"]["sha256"],
        "all_gates_passed": report["all_gates_passed"],
        "failed_gates": report["failed_gates"],
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
    }


def _generate_fixed_validation_block() -> list:
    _require_confirmation(
        VALIDATION_CONFIRMATION_ENV,
        VALIDATION_CONFIRMATION,
        action="502M fixed validation",
    )
    datasets = []
    for seed, cell in zip(FIXED_VALIDATION_SEEDS, _factorial_cells()):
        mask = make_stratified_response_mask(
            cell["n_sites"],
            cell["n_species"],
            seed=seed,
        )
        datasets.append(
            simulate_generative_iid_dataset(
                n_sites=cell["n_sites"],
                n_species=cell["n_species"],
                covariate_shape=cell["covariate_shape"],
                loading_stratum=cell["loading_stratum"],
                prevalence_stratum=cell["prevalence_stratum"],
                seed=seed,
                response_realization=0,
                response_mask=mask,
            )
        )
    if len(datasets) != 324:
        raise AssertionError("502M fixed validation must contain 324 contexts")
    return datasets


def _production_invariance_checks(
    inference: GenerativeIidInference,
    datasets: list,
) -> dict[str, float]:
    representatives = {}
    for dataset in datasets:
        representatives.setdefault(dataset.Y.shape, dataset)
    permutation_deltas = []
    for dataset in representatives.values():
        batch = batch_generative_iid_datasets(
            [dataset],
            max_sites=inference.model.max_sites,
            max_species=inference.model.max_species,
        )
        n_sites, n_species = dataset.Y.shape
        site_order = np.random.default_rng(
            int(dataset.metadata["seed"]) + 11
        ).permutation(n_sites)
        species_order = np.random.default_rng(
            int(dataset.metadata["seed"]) + 12
        ).permutation(n_species)
        full_site_order = np.concatenate(
            [site_order, np.arange(n_sites, inference.model.max_sites)]
        )
        full_species_order = np.concatenate(
            [
                species_order,
                np.arange(n_species, inference.model.max_species),
            ]
        )
        permuted = {
            "X": batch.X[:, full_site_order],
            "Y": batch.Y[:, full_site_order][:, :, full_species_order],
            "response_mask": batch.response_mask[:, full_site_order][
                :, :, full_species_order
            ],
            "site_mask": batch.site_mask[:, full_site_order],
            "species_mask": batch.species_mask[:, full_species_order],
        }
        original = posterior_mean_invariants(
            inference.model(batch.model_inputs(), training=False)
        )
        moved = posterior_mean_invariants(
            inference.model(permuted, training=False)
        )
        site_inverse = np.argsort(site_order)
        species_inverse = np.argsort(species_order)
        permutation_deltas.extend(
            [
                float(
                    np.max(
                        np.abs(
                            np.asarray(moved["Beta"])[
                                0, :, :n_species
                            ][:, species_inverse]
                            - np.asarray(original["Beta"])[
                                0, :, :n_species
                            ]
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(moved["R"])[
                                0, :n_sites, :n_species
                            ][site_inverse][:, species_inverse]
                            - np.asarray(original["R"])[
                                0, :n_sites, :n_species
                            ]
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(moved["C"])[
                                0, :n_species, :n_species
                            ][species_inverse][:, species_inverse]
                            - np.asarray(original["C"])[
                                0, :n_species, :n_species
                            ]
                        )
                    )
                ),
            ]
        )

    smallest = min(datasets, key=lambda value: value.Y.size)
    largest = max(datasets, key=lambda value: value.Y.size)
    alone = batch_generative_iid_datasets(
        [smallest],
        max_sites=inference.model.max_sites,
        max_species=inference.model.max_species,
    )
    together = batch_generative_iid_datasets(
        [smallest, largest],
        max_sites=inference.model.max_sites,
        max_species=inference.model.max_species,
    )
    alone_posterior = inference.model(alone.model_inputs(), training=False)
    together_posterior = inference.model(
        together.model_inputs(),
        training=False,
    )
    padding_delta = max(
        float(
            np.max(
                np.abs(
                    np.asarray(alone_posterior.mean)[0]
                    - np.asarray(together_posterior.mean)[0]
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    np.asarray(alone_posterior.low_rank_factor)[0]
                    - np.asarray(together_posterior.low_rank_factor)[0]
                )
            )
        ),
    )
    return {
        "permutation_max_abs_delta": max(permutation_deltas, default=0.0),
        "padding_max_abs_delta": padding_delta,
        "representative_shape_count": len(representatives),
    }


def _factorial_cells() -> tuple[dict[str, Any], ...]:
    cells = []
    for (
        n_sites,
        n_species,
        covariate_shape,
        loading_stratum,
        prevalence_stratum,
        replicate,
    ) in itertools.product(
        (24, 40, 96),
        (12, 36, 75),
        ("normal", "right_skewed"),
        ("weak", "medium", "strong"),
        ("rare", "moderate", "common"),
        (0, 1),
    ):
        cells.append(
            {
                "n_sites": n_sites,
                "n_species": n_species,
                "covariate_shape": covariate_shape,
                "loading_stratum": loading_stratum,
                "prevalence_stratum": prevalence_stratum,
                "replicate": replicate,
            }
        )
    if len(cells) != 324:
        raise AssertionError("production factorial must contain 324 cells")
    return tuple(cells)


def _generate_production_block(
    seeds: tuple[int, ...],
    *,
    masked: bool,
) -> list:
    if seeds != TRAINING_SEEDS:
        raise ValueError("501M generator accepts only the training seed block")
    if masked:
        raise ValueError("501M training contexts must be fully observed")
    datasets = []
    for seed, cell in zip(seeds, _factorial_cells()):
        datasets.append(
            simulate_generative_iid_dataset(
                n_sites=cell["n_sites"],
                n_species=cell["n_species"],
                covariate_shape=cell["covariate_shape"],
                loading_stratum=cell["loading_stratum"],
                prevalence_stratum=cell["prevalence_stratum"],
                seed=seed,
                response_realization=0,
            )
        )
    return datasets


def _validate_training_corpus(owners: list, training: list) -> None:
    if len(owners) != 324 or len(training) != 648:
        raise ValueError("501M corpus size differs")
    expected_seeds = list(TRAINING_SEEDS)
    if [int(value.metadata["seed"]) for value in owners] != expected_seeds:
        raise ValueError("501M owning seed order differs")
    for index, seed in enumerate(expected_seeds):
        pair = training[2 * index : 2 * index + 2]
        if [int(value.metadata["seed"]) for value in pair] != [seed, seed]:
            raise ValueError("501M paired owning seeds differ")
        if [int(value.metadata["response_realization"]) for value in pair] != [
            0,
            1,
        ]:
            raise ValueError("501M response realization roles differ")
        if not np.array_equal(pair[0].X, pair[1].X):
            raise ValueError("501M paired response design differs")
        if not np.array_equal(pair[0].truth_beta, pair[1].truth_beta):
            raise ValueError("501M paired response parameters differ")


def _factorial_summary() -> dict[str, Any]:
    return {
        "n_sites": [24, 40, 96],
        "n_species": [12, 36, 75],
        "covariate_shape": ["normal", "right_skewed"],
        "loading_stratum": ["weak", "medium", "strong"],
        "prevalence_stratum": ["rare", "moderate", "common"],
        "replicates": 2,
        "owning_contexts": 324,
    }


def _metadata_sha256(datasets: list) -> str:
    import hashlib

    payload = [dict(dataset.metadata) for dataset in datasets]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact_inventory(
    root: Path,
    *,
    relative_to: Path,
) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        "root": str(root.resolve().relative_to(relative_to.resolve())),
        "file_count": len(files),
        "files": [
            {
                "path": str(path.resolve().relative_to(relative_to.resolve())),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }


def _validate_artifact_inventory(
    root: Path,
    inventory: dict[str, Any],
) -> None:
    records = inventory.get("files")
    if not isinstance(records, list):
        raise ValueError("502M comparator inventory is missing")
    if int(inventory.get("file_count", -1)) != len(records):
        raise ValueError("502M comparator inventory count differs")
    for record in records:
        path = (root / str(record["path"])).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("502M comparator path escapes run root") from error
        if not path.is_file():
            raise FileNotFoundError(f"502M comparator artifact missing: {path}")
        if file_sha256(path) != record["sha256"]:
            raise ValueError("502M comparator artifact hash differs")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError("502M comparator artifact size differs")


def _require_confirmation(name: str, value: str, *, action: str) -> None:
    if os.environ.get(name) != value:
        raise RuntimeError(f"{name} must equal {value!r} before {action}")


def _require_clean_pinned_source(expected_source_commit: str) -> str:
    head, _, worktree_dirty = _source_control_state()
    if head != expected_source_commit:
        raise RuntimeError(
            f"source HEAD {head} differs from pinned {expected_source_commit}"
        )
    if worktree_dirty:
        raise RuntimeError("production training requires a clean worktree")
    for path in PRODUCTION_SOURCE_PATHS:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(f"production source file is missing: {path}")
    return head


def _require_false_seed_flags(payload: dict[str, Any]) -> None:
    fixed_validation_keys = (
        "fixed_validation_seed_ranges_opened",
        "fixed_validation_opened",
    )
    fixed_validation_values = [
        payload[key] for key in fixed_validation_keys if key in payload
    ]
    if not fixed_validation_values or any(
        value is not False for value in fixed_validation_values
    ):
        raise ValueError(
            "production seed-seal flag differs: "
            "fixed_validation_seed_ranges_opened"
        )
    for key in (
        "reserved_seed_ranges_opened",
        "redesign_seed_ranges_opened",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"production seed-seal flag differs: {key}")


def _empty_output(output: Path) -> Path:
    root = output.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"production output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
