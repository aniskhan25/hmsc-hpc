#!/usr/bin/env python3
"""Sealed production harness for generative Neural-HMSC iid probit v1."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
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
)
from pyhmsc.neural.generative_iid_artifact import (  # noqa: E402
    GenerativeIidInference,
    file_sha256,
    validate_generative_iid_checkpoint,
)
from examples.run_generative_neural_hmsc_iid_v1 import (  # noqa: E402
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
    "pyhmsc/neural/__init__.py",
    "examples/run_generative_neural_hmsc_iid_v1.py",
    "examples/run_generative_neural_hmsc_iid_v1_production.py",
    "docs/lumi_generative_neural_hmsc_iid_v1_training_sbatch.sh",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-seal")

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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_frozen_documents()
    if args.command == "check-seal":
        result = production_seal_status()
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
    else:
        result = preflight_fixed_validation(
            args.freeze_root,
            expected_source_commit=args.expected_source_commit,
            expected_checkpoint_content_sha256=(
                args.expected_checkpoint_content_sha256
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
        "fixed_validation_executable": False,
        "fixed_validation_blocker": (
            "the complete preregistered 502M comparator evaluator must be "
            "implemented, tested, and hash-frozen before this one-shot block"
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
    elapsed = time.perf_counter() - started
    if not all(
        np.isfinite(np.asarray(values)).all()
        for values in (history.loss, history.iwelbo, history.gradient_norm)
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
        "fixed_validation_opened": False,
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
        "training_corpus_manifest_sha256": file_sha256(corpus_path),
        "training": training_manifest,
        "final_training_loss": history.loss[-1],
        "final_training_iwelbo": history.iwelbo[-1],
        "final_gradient_norm": history.gradient_norm[-1],
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
    provenance = manifest["source_provenance"]
    if provenance.get("commit") != expected_source_commit:
        raise ValueError("501M source-provenance commit differs")
    if provenance.get("worktree_dirty") is not False:
        raise ValueError("501M checkpoint was not produced from clean source")
    source_inventory = {
        record["path"]: record["sha256"]
        for record in provenance["source_files"]
    }
    if set(source_inventory) != set(PRODUCTION_SOURCE_PATHS):
        raise ValueError("501M source inventory differs")

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
    report = json.loads(
        (root / freeze["training_report"]["path"]).read_text(encoding="utf-8")
    )
    _require_false_seed_flags(report)
    if report.get("checkpoint_content_sha256") != manifest["content_sha256"]:
        raise ValueError("501M report checkpoint binding differs")

    result = {
        "status": "candidate_501m_training_freeze_valid",
        "freeze_sha256": file_sha256(freeze_path),
        "source_commit": expected_source_commit,
        "checkpoint_content_sha256": manifest["content_sha256"],
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "weights_sha256": manifest["artifacts"]["weights"]["sha256"],
        "training_seed_range": [TRAINING_SEEDS[0], TRAINING_SEEDS[-1]],
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
    }
    _write_json(root / "postfreeze_validation.json", result)
    return result


def preflight_fixed_validation(
    freeze_root: Path,
    *,
    expected_source_commit: str,
    expected_checkpoint_content_sha256: str,
) -> dict[str, Any]:
    """Validate the 502M boundary while leaving its seeds inaccessible."""
    training = validate_training_freeze(
        freeze_root,
        expected_source_commit=expected_source_commit,
    )
    if training["checkpoint_content_sha256"] != (
        expected_checkpoint_content_sha256
    ):
        raise ValueError("502M preflight checkpoint content hash differs")
    if os.environ.get(VALIDATION_CONFIRMATION_ENV):
        raise RuntimeError(
            f"{VALIDATION_CONFIRMATION_ENV} must remain unset during preflight"
        )
    return {
        "status": "fixed_validation_preflight_sealed",
        "training_freeze_sha256": training["freeze_sha256"],
        "checkpoint_content_sha256": training["checkpoint_content_sha256"],
        "fixed_validation_seed_range": [
            FIXED_VALIDATION_SEEDS[0],
            FIXED_VALIDATION_SEEDS[-1],
        ],
        "fixed_validation_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
        "fixed_validation_executable": False,
        "missing_reviewed_components": list(REQUIRED_502_EVALUATOR_COMPONENTS),
        "decision": (
            "do not authorize 502M until all listed evaluator components are "
            "implemented, tested, and pinned to a clean source commit"
        ),
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
    if seeds not in (TRAINING_SEEDS, FIXED_VALIDATION_SEEDS):
        raise ValueError("unregistered generative iid production seed block")
    if seeds == FIXED_VALIDATION_SEEDS:
        _require_confirmation(
            VALIDATION_CONFIRMATION_ENV,
            VALIDATION_CONFIRMATION,
            action="502M fixed validation",
        )
        raise RuntimeError(
            "502M remains sealed until its complete comparator evaluator is "
            "implemented and reviewed"
        )
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


def _require_confirmation(name: str, value: str, *, action: str) -> None:
    if os.environ.get(name) != value:
        raise RuntimeError(f"{name} must equal {value!r} before {action}")


def _require_clean_pinned_source(expected_source_commit: str) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_source_commit:
        raise RuntimeError(
            f"source HEAD {head} differs from pinned {expected_source_commit}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("production training requires a clean worktree")
    for path in PRODUCTION_SOURCE_PATHS:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(f"production source file is missing: {path}")
    return head


def _require_false_seed_flags(payload: dict[str, Any]) -> None:
    for key in (
        "fixed_validation_seed_ranges_opened",
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
