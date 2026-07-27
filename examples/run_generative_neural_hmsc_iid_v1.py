#!/usr/bin/env python3
"""Sealed disposable harness for generative Neural-HMSC iid probit v1."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.generative_iid import (
    GENERATIVE_IID_DESIGN_REVIEW_SHA256,
    GENERATIVE_IID_PREREGISTRATION_SHA256,
    GENERATIVE_IID_SEED_AUDIT_SHA256,
    GenerativeIidPosteriorModel,
    batch_generative_iid_datasets,
    importance_weighted_variational_loss,
    make_stratified_response_mask,
    posterior_mean_invariants,
    simulate_generative_iid_dataset,
    train_generative_iid_model,
)
from pyhmsc.neural.generative_iid_artifact import (
    GenerativeIidInference,
    file_sha256,
    validate_generative_iid_checkpoint,
)
from pyhmsc.neural.generative_iid_mcmc import exact_model_log_joint


CONFIRMATION_ENV = "OPEN_GENERATIVE_IID_DISPOSABLE_SMOKE"
CONFIRMATION_VALUE = "GENERATE_591M_592M_DISPOSABLE_ONLY"
HOST_SOURCE_COMMIT_ENV = "GENERATIVE_IID_HOST_SOURCE_COMMIT"
HOST_SOURCE_BRANCH_ENV = "GENERATIVE_IID_HOST_SOURCE_BRANCH"
HOST_WORKTREE_CLEAN_ENV = "GENERATIVE_IID_HOST_WORKTREE_CLEAN"
TRAINING_SEEDS = tuple(range(591000001, 591000019))
VALIDATION_SEEDS = tuple(range(592000001, 592000019))
PREREGISTRATION = (
    ROOT
    / "docs"
    / "generative_neural_hmsc_iid_v1_preregistration_2026-07-27.md"
)
SEED_AUDIT = (
    ROOT
    / "docs"
    / "generative_neural_hmsc_iid_v1_seed_audit_2026-07-27.json.md"
)
DESIGN_REVIEW = (
    ROOT
    / "docs"
    / "generative_neural_hmsc_iid_v1_design_review_2026-07-27.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("check-seal", "disposable-smoke", "validate-disposable"),
        default="check-seal",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_frozen_documents()
    if args.mode == "check-seal":
        print(
            json.dumps(
                {
                    "sealed": True,
                    "disposable_seed_ranges": [
                        [TRAINING_SEEDS[0], TRAINING_SEEDS[-1]],
                        [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1]],
                    ],
                    "production_seed_ranges_opened": False,
                    "confirmation_env": CONFIRMATION_ENV,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.mode == "validate-disposable":
        if args.output is None:
            raise ValueError("--output is required for validate-disposable")
        print(
            json.dumps(
                validate_disposable_smoke(args.output),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if os.environ.get(CONFIRMATION_ENV) != CONFIRMATION_VALUE:
        raise RuntimeError(
            f"{CONFIRMATION_ENV} must equal {CONFIRMATION_VALUE!r} "
            "before disposable seeds may open"
        )
    if args.output is None:
        raise ValueError("--output is required for disposable-smoke")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    run_disposable_smoke(args.output, epochs=args.epochs)


def run_disposable_smoke(output: Path, *, epochs: int) -> Path:
    """Generate exactly the sealed 591M-592M smoke and no production data."""
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"smoke output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train = _generate_block(TRAINING_SEEDS, masked=False)
    validation = _generate_block(VALIDATION_SEEDS, masked=True)
    train_batch = batch_generative_iid_datasets(
        train, max_sites=96, max_species=75
    )
    validation_batch = batch_generative_iid_datasets(
        validation, max_sites=96, max_species=75
    )
    tf.keras.utils.set_random_seed(501900001)
    model = GenerativeIidPosteriorModel()
    history = train_generative_iid_model(
        model,
        train_batch,
        epochs=epochs,
        model_seed=501900001,
    )
    posterior = model(validation_batch.model_inputs(), training=False)
    loss, diagnostics = importance_weighted_variational_loss(
        posterior,
        validation_batch.model_inputs(),
        draws=8,
        kl_weight=1.0,
        seed=592900001,
    )
    invariants = posterior_mean_invariants(posterior)
    exact_truth_log_joint = float(
        exact_model_log_joint(
            _truth_state_for_dataset(validation[0]),
            validation[0],
        )[0]
    )
    source_commit = _source_commit()
    source_provenance = _source_provenance(source_commit)
    inference = GenerativeIidInference(
        model=model,
        manifest={},
        checkpoint_root=None,
    )
    checkpoint = inference.save(
        output / "checkpoint",
        source_commit=source_commit,
        source_provenance=source_provenance,
        training_manifest={
            "role": "disposable_smoke_only",
            "training_seed_range": [
                TRAINING_SEEDS[0],
                TRAINING_SEEDS[-1],
            ],
            "validation_seed_range": [
                VALIDATION_SEEDS[0],
                VALIDATION_SEEDS[-1],
            ],
            "epochs": int(epochs),
            "production_seed_ranges_opened": False,
        },
    )
    checkpoint_manifest = validate_generative_iid_checkpoint(checkpoint)
    report = {
        "status": "disposable_smoke_complete",
        "evidence_role": "plumbing_and_optimization_only",
        "training_seed_range": [TRAINING_SEEDS[0], TRAINING_SEEDS[-1]],
        "validation_seed_range": [
            VALIDATION_SEEDS[0],
            VALIDATION_SEEDS[-1],
        ],
        "training_contexts": len(train),
        "validation_contexts": len(validation),
        "epochs": int(epochs),
        "final_training_loss": history.loss[-1],
        "validation_loss": float(loss),
        "validation_iwelbo": float(diagnostics["iwelbo"]),
        "exact_truth_log_joint_first_validation": exact_truth_log_joint,
        "all_finite": bool(
            np.isfinite(history.loss).all()
            and np.isfinite(float(loss))
            and all(
                np.all(np.isfinite(np.asarray(value)))
                for value in invariants.values()
            )
        ),
        "checkpoint_content_sha256": checkpoint_manifest["content_sha256"],
        "checkpoint_weights_sha256": checkpoint_manifest["artifacts"]["weights"][
            "sha256"
        ],
        "production_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
        "source_commit": source_commit,
    }
    report_path = output / "disposable_smoke_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze = {
        "protocol": "generative_neural_hmsc_iid_probit_v1",
        "preregistration_sha256": GENERATIVE_IID_PREREGISTRATION_SHA256,
        "seed_audit_sha256": GENERATIVE_IID_SEED_AUDIT_SHA256,
        "design_review_sha256": GENERATIVE_IID_DESIGN_REVIEW_SHA256,
        "report": {
            "path": report_path.name,
            "sha256": file_sha256(report_path),
        },
        "checkpoint_manifest": {
            "path": str(
                Path("checkpoint") / "generative_iid_checkpoint.json"
            ),
            "sha256": file_sha256(
                checkpoint / "generative_iid_checkpoint.json"
            ),
        },
        "production_seed_ranges_opened": False,
    }
    (output / "freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def validate_disposable_smoke(output: Path) -> dict[str, object]:
    """Independently validate a completed disposable run from frozen files."""
    output = output.expanduser().resolve()
    freeze_path = output / "freeze.json"
    report_path = output / "disposable_smoke_report.json"
    checkpoint = output / "checkpoint"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = validate_generative_iid_checkpoint(checkpoint)
    if file_sha256(report_path) != freeze["report"]["sha256"]:
        raise ValueError("disposable report hash differs")
    checkpoint_manifest = checkpoint / "generative_iid_checkpoint.json"
    if file_sha256(checkpoint_manifest) != (
        freeze["checkpoint_manifest"]["sha256"]
    ):
        raise ValueError("disposable checkpoint-manifest hash differs")
    if freeze.get("production_seed_ranges_opened") is not False:
        raise ValueError("disposable freeze opened production seeds")
    if report.get("production_seed_ranges_opened") is not False:
        raise ValueError("disposable report opened production seeds")
    if report.get("reserved_seed_ranges_opened") is not False:
        raise ValueError("disposable report opened reserved seeds")
    if report.get("training_seed_range") != [
        TRAINING_SEEDS[0],
        TRAINING_SEEDS[-1],
    ]:
        raise ValueError("disposable training seed range differs")
    if report.get("validation_seed_range") != [
        VALIDATION_SEEDS[0],
        VALIDATION_SEEDS[-1],
    ]:
        raise ValueError("disposable validation seed range differs")
    if report.get("all_finite") is not True:
        raise ValueError("disposable report contains non-finite values")
    if manifest["training_manifest"]["role"] != "disposable_smoke_only":
        raise ValueError("disposable checkpoint role differs")

    validation = _generate_block(VALIDATION_SEEDS, masked=True)
    validation_batch = batch_generative_iid_datasets(
        validation, max_sites=96, max_species=75
    )
    loaded = GenerativeIidInference.load(checkpoint)
    posterior = loaded.model(validation_batch.model_inputs(), training=False)
    loss, diagnostics = importance_weighted_variational_loss(
        posterior,
        validation_batch.model_inputs(),
        draws=8,
        kl_weight=1.0,
        seed=592900001,
    )
    exact = float(
        exact_model_log_joint(
            _truth_state_for_dataset(validation[0]),
            validation[0],
        )[0]
    )
    if not np.isclose(
        float(loss), float(report["validation_loss"]), rtol=1e-6, atol=1e-4
    ):
        raise ValueError("recomputed disposable validation loss differs")
    if not np.isclose(
        float(diagnostics["iwelbo"]),
        float(report["validation_iwelbo"]),
        rtol=1e-6,
        atol=1e-4,
    ):
        raise ValueError("recomputed disposable IWELBO differs")
    if not np.isclose(
        exact,
        float(report["exact_truth_log_joint_first_validation"]),
        rtol=1e-10,
        atol=1e-8,
    ):
        raise ValueError("recomputed disposable exact target differs")

    tf.keras.utils.set_random_seed(501900001)
    untrained = GenerativeIidPosteriorModel()
    untrained(validation_batch.model_inputs(), training=False)
    maximum_weight_change = max(
        float(np.max(np.abs(np.asarray(trained) - np.asarray(initial))))
        for trained, initial in zip(
            loaded.model.weights, untrained.weights
        )
    )
    if maximum_weight_change <= 1e-8:
        raise ValueError("disposable optimizer did not change model weights")

    validation_record = {
        "status": "independent_disposable_validation_passed",
        "report_sha256": file_sha256(report_path),
        "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest),
        "checkpoint_content_sha256": manifest["content_sha256"],
        "weights_sha256": manifest["artifacts"]["weights"]["sha256"],
        "recomputed_validation_loss": float(loss),
        "recomputed_validation_iwelbo": float(diagnostics["iwelbo"]),
        "recomputed_exact_truth_log_joint": exact,
        "maximum_weight_change_from_seeded_initialization": (
            maximum_weight_change
        ),
        "source_worktree_dirty_recorded": manifest["source_provenance"][
            "worktree_dirty"
        ],
        "production_seed_ranges_opened": False,
        "reserved_seed_ranges_opened": False,
    }
    validation_path = output / "independent_validation.json"
    validation_path.write_text(
        json.dumps(validation_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation_record


def _generate_block(
    seeds: tuple[int, ...], *, masked: bool
) -> list:
    combinations = list(
        itertools.product(
            ("normal", "right_skewed"),
            ("weak", "medium", "strong"),
            ("rare", "moderate", "common"),
        )
    )
    if len(combinations) != len(seeds):
        raise AssertionError("disposable factorial differs")
    shapes = ((24, 12), (40, 36), (96, 75))
    datasets = []
    for index, (seed, combination) in enumerate(zip(seeds, combinations)):
        n_sites, n_species = shapes[index % len(shapes)]
        mask = (
            make_stratified_response_mask(
                n_sites, n_species, seed=seed
            )
            if masked
            else None
        )
        datasets.append(
            simulate_generative_iid_dataset(
                n_sites=n_sites,
                n_species=n_species,
                covariate_shape=combination[0],
                loading_stratum=combination[1],
                prevalence_stratum=combination[2],
                seed=seed,
                response_mask=mask,
            )
        )
    return datasets


def _truth_state_for_dataset(dataset) -> tf.Tensor:
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
        PREREGISTRATION: GENERATIVE_IID_PREREGISTRATION_SHA256,
        SEED_AUDIT: GENERATIVE_IID_SEED_AUDIT_SHA256,
        DESIGN_REVIEW: GENERATIVE_IID_DESIGN_REVIEW_SHA256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"frozen document hash differs: {path}")


def _source_commit() -> str:
    commit, _, _ = _source_control_state()
    return commit


def _source_provenance(source_commit: str) -> dict[str, object]:
    source_paths = (
        "pyhmsc/neural/generative_iid.py",
        "pyhmsc/neural/generative_iid_mcmc.py",
        "pyhmsc/neural/generative_iid_artifact.py",
        "pyhmsc/neural/__init__.py",
        "examples/run_generative_neural_hmsc_iid_v1.py",
        "docs/generative_neural_hmsc_iid_v1_preregistration_2026-07-27.md",
        "docs/generative_neural_hmsc_iid_v1_seed_audit_2026-07-27.json.md",
        "docs/generative_neural_hmsc_iid_v1_design_review_2026-07-27.md",
    )
    observed_commit, branch, worktree_dirty = _source_control_state()
    if observed_commit != source_commit:
        raise RuntimeError("source provenance commit changed during execution")
    try:
        import tensorflow_probability as tfp

        tfp_version = str(tfp.__version__)
    except ImportError:
        tfp_version = "unavailable"
    return {
        "commit": source_commit,
        "branch": branch,
        "worktree_dirty": worktree_dirty,
        "source_files": [
            {"path": path, "sha256": file_sha256(ROOT / path)}
            for path in source_paths
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
    """Read Git state, or consume a strict host attestation in a container."""
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
                "Git is unavailable and the clean host-source attestation "
                "is absent or invalid"
            ) from error
        return commit, branch, False


if __name__ == "__main__":
    main()
