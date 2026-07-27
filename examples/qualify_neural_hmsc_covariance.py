#!/usr/bin/env python3
"""Run the sealed Milestone 56 fixed-probit covariance qualification protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache")
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.covariance_inference import (  # noqa: E402
    BOUND_MEMBER_SEED,
    CORRELATION_OVERLAY_ID,
    M56_AUDIT_SHA256,
    M56_PREREGISTRATION_SHA256,
    FixedProbitCovarianceInference,
    bivariate_beta_negative_log_probability,
    fit_fixed_probit_covariance_overlay,
    validate_bound_v0_1_release,
)
from pyhmsc.neural.models import probit_irls_laplace_full_anchor  # noqa: E402
from pyhmsc.neural.release import load_neural_hmsc_release  # noqa: E402
from pyhmsc.neural.simulator import FixedEffectDataset  # noqa: E402
from pyhmsc.neural.train import fixed_shape_training_data  # noqa: E402


PROTOCOL_ID = "neural_hmsc_fixed_probit_covariance_m56_v1"
PREREGISTRATION_PATH = (
    ROOT / "docs/neural_hmsc_m56_covariance_preregistration_2026-07-23.md"
)
AUDIT_PATH = ROOT / "docs/neural_hmsc_m56_artifact_seed_audit_2026-07-23.json.md"
TRAIN_CONFIRMATION = "GENERATE_M56_CORRELATION_TRAIN_VALIDATION"
EVALUATION_CONFIRMATION = "OPEN_M56_RESERVED_COVARIANCE_EVALUATION"
REALDATA_CONFIRMATION = "OPEN_M56_FROZEN_REALDATA_REPLAY"
PRODUCTION_MODEL_SEED = 211_900_001
SMOKE_MODEL_SEED = 291_900_001
PRODUCTION_COUNT = 324
SMOKE_COUNT = 27
PRODUCTION_STARTS = {
    "training": 211_000_001,
    "validation": 212_000_001,
    "evaluation_a": 213_000_001,
    "evaluation_b": 214_000_001,
    "evaluation_c": 215_000_001,
}
SMOKE_STARTS = {"training": 291_000_001, "evaluation": 292_000_001}
LOCATIONS = (-1.5, 0.0, 1.5)
SCALES = (0.5, 1.0, 2.0)
PREVALENCE = (("rare", 0.05), ("balanced", 0.30), ("common", 0.65))
EFFECTS = (("weak", 0.25), ("moderate", 0.75), ("strong", 1.50))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-registry",
        type=Path,
        default=Path("/private/tmp/neural_hmsc_releases"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train-validate")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--confirmation", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--freeze-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--confirmation", required=True)

    realdata = subparsers.add_parser("realdata")
    realdata.add_argument("--evaluation-root", type=Path, required=True)
    realdata.add_argument("--output", type=Path, required=True)
    realdata.add_argument("--confirmation", required=True)

    validate = subparsers.add_parser("validate-freeze")
    validate.add_argument("--freeze-root", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(args)
    elif args.command == "train-validate":
        result = train_and_validate(args)
    elif args.command == "evaluate":
        result = evaluate_reserved(args)
    elif args.command == "realdata":
        result = authorize_realdata_replay(args)
    else:
        result = validate_train_validation_freeze(
            args.freeze_root,
            registry_root=args.release_registry,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output = _empty_output(args.output)
    protocol_hashes = _validate_protocol_documents()
    release = load_neural_hmsc_release(args.release_registry)
    binding = validate_bound_v0_1_release(release)
    train_seeds = tuple(
        range(SMOKE_STARTS["training"], SMOKE_STARTS["training"] + SMOKE_COUNT)
    )
    evaluation_seeds = tuple(
        range(SMOKE_STARTS["evaluation"], SMOKE_STARTS["evaluation"] + SMOKE_COUNT)
    )
    _assert_seed_roles(train_seeds, evaluation_seeds, production=False)
    training = build_m56_corpus(train_seeds, smoke=True)
    evaluation = build_m56_corpus(evaluation_seeds, smoke=True)
    base = release.load_checkpoint(seed=BOUND_MEMBER_SEED)
    engine, history = fit_fixed_probit_covariance_overlay(
        base,
        training,
        base_binding=binding,
        model_seed=SMOKE_MODEL_SEED,
        epochs=1,
        batch_size=9,
        learning_rate=0.001,
    )
    artifact = engine.save(output / "overlay")
    loaded = FixedProbitCovarianceInference.load(
        artifact, registry_root=args.release_registry
    )
    metrics = evaluate_corpus(loaded, evaluation, draws=32)
    roundtrip = _roundtrip_deltas(engine, loaded, evaluation[0])
    smoke_checks = {
        "protocol_hashes_valid": True,
        "bound_release_hashes_valid": True,
        "exact_disposable_seed_roles": True,
        "production_seeds_remain_unopened": True,
        "training_finite": bool(np.isfinite(history["loss"][-1])),
        "artifact_roundtrip": all(value <= 1e-7 for value in roundtrip.values()),
        "base_mean_parity": metrics["max_abs_mean_delta"] <= 1e-7,
        "base_scale_parity": metrics["max_abs_scale_delta"] <= 1e-7,
        "positive_definite": metrics["minimum_covariance_eigenvalue"] > 1e-8,
        "correlation_bounded": metrics["maximum_absolute_correlation"] <= 0.98,
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "mode": "disposable_smoke",
        "promotion_evidence": False,
        "production_seed_opened": False,
        "protocol_hashes": protocol_hashes,
        "base_binding": binding,
        "seed_roles": {
            "training": [train_seeds[0], train_seeds[-1]],
            "evaluation": [evaluation_seeds[0], evaluation_seeds[-1]],
        },
        "training": engine.training_record,
        "metrics": metrics,
        "roundtrip_max_abs_delta": roundtrip,
        "checks": smoke_checks,
        "passed": all(smoke_checks.values()),
        "artifact": {
            "id": CORRELATION_OVERLAY_ID,
            "path": str(artifact),
            "manifest_sha256": _file_sha256(artifact / "correlation_overlay.json"),
            "weights_sha256": _file_sha256(artifact / "correlation_head.weights.h5"),
        },
    }
    _write_json(output / "smoke_report.json", result)
    return result


def train_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    _require_confirmation(args.confirmation, TRAIN_CONFIRMATION)
    output = _empty_output(args.output)
    protocol_hashes = _validate_protocol_documents()
    release = load_neural_hmsc_release(args.release_registry)
    binding = validate_bound_v0_1_release(release)
    training_seeds = tuple(
        range(
            PRODUCTION_STARTS["training"],
            PRODUCTION_STARTS["training"] + PRODUCTION_COUNT,
        )
    )
    validation_seeds = tuple(
        range(
            PRODUCTION_STARTS["validation"],
            PRODUCTION_STARTS["validation"] + PRODUCTION_COUNT,
        )
    )
    _assert_seed_roles(training_seeds, validation_seeds, production=True)
    training = build_m56_corpus(training_seeds, smoke=False)
    validation = build_m56_corpus(validation_seeds, smoke=False)
    engine, history = fit_fixed_probit_covariance_overlay(
        release.load_checkpoint(seed=BOUND_MEMBER_SEED),
        training,
        base_binding=binding,
        model_seed=PRODUCTION_MODEL_SEED,
        epochs=100,
        batch_size=9,
        learning_rate=0.001,
    )
    artifact = engine.save(output / "overlay")
    metrics = evaluate_corpus(engine, validation)
    gates = fixed_validation_gates(metrics)
    freeze = {
        "protocol_id": PROTOCOL_ID,
        "mode": "production_train_validation",
        "protocol_hashes": protocol_hashes,
        "base_binding": binding,
        "training_seed_range": [training_seeds[0], training_seeds[-1]],
        "validation_seed_range": [validation_seeds[0], validation_seeds[-1]],
        "reserved_seed_blocks_opened": False,
        "training": engine.training_record,
        "history_final": {key: values[-1] for key, values in history.items()},
        "validation_metrics": metrics,
        "validation_gates": gates,
        "validation_passed": all(gates.values()),
        "overlay_manifest_sha256": _file_sha256(artifact / "correlation_overlay.json"),
        "overlay_weights_sha256": _file_sha256(
            artifact / "correlation_head.weights.h5"
        ),
    }
    _write_json(output / "freeze.json", freeze)
    return freeze


def evaluate_reserved(args: argparse.Namespace) -> dict[str, Any]:
    _require_confirmation(args.confirmation, EVALUATION_CONFIRMATION)
    freeze = _load_qualified_freeze(args.freeze_root)
    output = _empty_output(args.output)
    engine = FixedProbitCovarianceInference.load(
        args.freeze_root / "overlay", registry_root=args.release_registry
    )
    reports = []
    for role in ("evaluation_a", "evaluation_b", "evaluation_c"):
        start = PRODUCTION_STARTS[role]
        seeds = tuple(range(start, start + PRODUCTION_COUNT))
        corpus = build_m56_corpus(seeds, smoke=False)
        reports.append(
            {
                "role": role,
                "seed_range": [seeds[0], seeds[-1]],
                "metrics": evaluate_corpus(engine, corpus),
            }
        )
    result = {
        "protocol_id": PROTOCOL_ID,
        "mode": "reserved_evaluation",
        "freeze_sha256": _file_sha256(args.freeze_root / "freeze.json"),
        "frozen_validation_passed": freeze["validation_passed"],
        "blocks": reports,
        "mcmc_subsets_required_before_promotion": True,
        "realdata_authorized": False,
        "promotion_evidence_complete": False,
    }
    _write_json(output / "reserved_evaluation.json", result)
    return result


def authorize_realdata_replay(args: argparse.Namespace) -> dict[str, Any]:
    _require_confirmation(args.confirmation, REALDATA_CONFIRMATION)
    evaluation = json.loads(
        (args.evaluation_root / "reserved_evaluation.json").read_text(encoding="utf-8")
    )
    if evaluation.get("promotion_evidence_complete") is not True:
        raise RuntimeError(
            "reserved simulation and MCMC evidence is incomplete; real-data seeds remain sealed"
        )
    output = _empty_output(args.output)
    result = {
        "protocol_id": PROTOCOL_ID,
        "mode": "realdata_replay_authorization",
        "target_outcomes_may_fit_or_select": False,
        "evaluation_sha256": _file_sha256(
            args.evaluation_root / "reserved_evaluation.json"
        ),
    }
    _write_json(output / "realdata_authorization.json", result)
    return result


def validate_train_validation_freeze(
    freeze_root: str | Path,
    *,
    registry_root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Independently validate a completed train-validation bundle."""
    root = Path(freeze_root).expanduser().resolve()
    freeze_path = root / "freeze.json"
    postfreeze_path = root / "postfreeze_validation.json"
    overlay = root / "overlay"
    required = {
        freeze_path,
        postfreeze_path,
        overlay / "correlation_overlay.json",
        overlay / "correlation_head.weights.h5",
    }
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Milestone 56 freeze files are missing: {missing}")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    postfreeze = json.loads(postfreeze_path.read_text(encoding="utf-8"))
    protocol_hashes = _validate_protocol_documents()
    release = load_neural_hmsc_release(registry_root)
    binding = validate_bound_v0_1_release(release)
    loaded = FixedProbitCovarianceInference.load(overlay, registry_root=registry_root)
    recomputed_gates = fixed_validation_gates(freeze["validation_metrics"])
    recorded_gates = {
        str(name): bool(value) for name, value in freeze["validation_gates"].items()
    }
    failed_gates = [name for name, value in recomputed_gates.items() if not bool(value)]

    disposable_fixture = simulate_m56_community(
        seed=292_000_001,
        predictor_location=-1.5,
        predictor_scale=1.0,
        prevalence_name="rare",
        target_prevalence=0.05,
        effect_name="weak",
        effect_magnitude=0.25,
        replicate=0,
    )
    prediction = loaded.predict_details(disposable_fixture)
    covariance = np.asarray(tf_matmul(prediction.posterior.scale_tril))
    training = freeze["training"]
    checks = {
        "protocol_id": freeze.get("protocol_id") == PROTOCOL_ID,
        "mode": freeze.get("mode") == "production_train_validation",
        "protocol_hashes": freeze.get("protocol_hashes") == protocol_hashes,
        "base_binding": freeze.get("base_binding") == binding,
        "training_seed_range": freeze.get("training_seed_range")
        == [211_000_001, 211_000_324],
        "validation_seed_range": freeze.get("validation_seed_range")
        == [212_000_001, 212_000_324],
        "reserved_seed_blocks_opened": (
            freeze.get("reserved_seed_blocks_opened") is False
            and postfreeze.get("reserved_seed_blocks_opened") is False
        ),
        "training_record": (
            training["epochs"] == 100
            and training["batch_size"] == 9
            and training["learning_rate"] == 0.001
            and training["seed"] == PRODUCTION_MODEL_SEED
            and training["community_count"] == PRODUCTION_COUNT
        ),
        "training_values_finite": _all_numeric_values_finite(training),
        "gate_recomputation": recomputed_gates == recorded_gates,
        "gate_decision": freeze.get("validation_passed")
        == all(recomputed_gates.values()),
        "postfreeze_gate_decision": postfreeze.get("validation_passed")
        == freeze.get("validation_passed"),
        "postfreeze_failed_gates": sorted(postfreeze.get("failed_gates", ()))
        == sorted(failed_gates),
        "postfreeze_checks": all(postfreeze.get("checks", {}).values()),
        "freeze_sha256": postfreeze.get("freeze_sha256") == _file_sha256(freeze_path),
        "overlay_manifest_sha256": (
            freeze.get("overlay_manifest_sha256")
            == postfreeze.get("overlay_manifest_sha256")
            == _file_sha256(overlay / "correlation_overlay.json")
        ),
        "overlay_weights_sha256": (
            freeze.get("overlay_weights_sha256")
            == postfreeze.get("overlay_weights_sha256")
            == _file_sha256(overlay / "correlation_head.weights.h5")
        ),
        "overlay_loadable": True,
        "disposable_prediction_finite": (
            np.all(np.isfinite(np.asarray(prediction.posterior.mean)))
            and np.all(np.isfinite(np.asarray(prediction.posterior.scale)))
            and np.all(np.isfinite(covariance))
        ),
        "disposable_prediction_positive_definite": (
            float(np.min(np.linalg.eigvalsh(covariance))) > 1e-8
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed_checks = [name for name, value in checks.items() if not value]
        raise ValueError(
            f"Milestone 56 independent freeze validation failed: {failed_checks}"
        )

    key_metric_names = (
        "marginal_coverage_95",
        "joint_ellipse_coverage_95",
        "candidate_diagonal_joint_nll_ratio",
        "candidate_raw_laplace_joint_nll_ratio",
        "marginal_rank_mean",
        "marginal_rank_variance",
        "radial_rank_mean",
        "radial_rank_variance",
        "candidate_diagonal_brier_ratio",
        "candidate_diagonal_log_loss_ratio",
        "mean_absolute_fisher_z_movement",
        "max_abs_mean_delta",
        "max_abs_scale_delta",
        "minimum_covariance_eigenvalue",
        "maximum_absolute_correlation",
    )
    report = {
        "schema_version": 1,
        "kind": "neural_hmsc_m56_independent_train_validation",
        "validated": True,
        "source_root": str(root),
        "checks": checks,
        "freeze_sha256": _file_sha256(freeze_path),
        "postfreeze_validation_sha256": _file_sha256(postfreeze_path),
        "overlay_manifest_sha256": _file_sha256(overlay / "correlation_overlay.json"),
        "overlay_weights_sha256": _file_sha256(overlay / "correlation_head.weights.h5"),
        "gate_count": len(recomputed_gates),
        "failed_gate_count": len(failed_gates),
        "failed_gates": failed_gates,
        "validation_passed": bool(freeze["validation_passed"]),
        "reserved_evaluation_authorized": bool(
            freeze["validation_passed"] and all(recomputed_gates.values())
        ),
        "decision": (
            "m56_fixed_validation_passed_reserved_evaluation_pending"
            if freeze["validation_passed"]
            else "m56_terminal_failure_reserved_evaluation_sealed"
        ),
        "key_metrics": {
            name: freeze["validation_metrics"][name] for name in key_metric_names
        },
    }
    if output is not None:
        destination = Path(output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(
                f"independent validation output already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json(destination, report)
    return report


def build_m56_corpus(seeds: Sequence[int], *, smoke: bool) -> list[FixedEffectDataset]:
    levels = (
        (
            (
                location,
                1.0,
                prevalence_name,
                prevalence,
                effect_name,
                effect,
                0,
            )
            for location in LOCATIONS
            for prevalence_name, prevalence in PREVALENCE
            for effect_name, effect in EFFECTS
        )
        if smoke
        else (
            (
                location,
                scale,
                prevalence_name,
                prevalence,
                effect_name,
                effect,
                replicate,
            )
            for location in LOCATIONS
            for scale in SCALES
            for prevalence_name, prevalence in PREVALENCE
            for effect_name, effect in EFFECTS
            for replicate in range(4)
        )
    )
    cells = list(levels)
    if len(cells) != len(seeds):
        raise ValueError(
            f"M56 seed count {len(seeds)} does not match factorial count {len(cells)}"
        )
    return [
        simulate_m56_community(
            seed=int(seed),
            predictor_location=location,
            predictor_scale=scale,
            prevalence_name=prevalence_name,
            target_prevalence=prevalence,
            effect_name=effect_name,
            effect_magnitude=effect,
            replicate=replicate,
        )
        for seed, (
            location,
            scale,
            prevalence_name,
            prevalence,
            effect_name,
            effect,
            replicate,
        ) in zip(seeds, cells)
    ]


def simulate_m56_community(
    *,
    seed: int,
    predictor_location: float,
    predictor_scale: float,
    prevalence_name: str,
    target_prevalence: float,
    effect_name: str,
    effect_magnitude: float,
    replicate: int,
) -> FixedEffectDataset:
    rng = np.random.default_rng(int(seed))
    n_sites, n_species = 40, 75
    z = rng.normal(size=n_sites)
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.square(z)))
    tmg = float(predictor_location) + float(predictor_scale) * z
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=n_species)
    slopes = signs * (
        float(effect_magnitude)
        + rng.normal(scale=0.10 * float(effect_magnitude), size=n_species)
    )
    intercepts = (
        ndtri(float(target_prevalence))
        - slopes * float(predictor_location)
        + rng.normal(scale=0.15, size=n_species)
    )
    beta = np.vstack([intercepts, slopes])
    design = np.column_stack([np.ones(n_sites), tmg])
    linear = design @ beta
    response = rng.binomial(1, ndtr(linear))
    sites = [f"site_{index + 1:03d}" for index in range(n_sites)]
    species = [f"sp{index + 1}" for index in range(n_species)]
    return FixedEffectDataset(
        Y=pd.DataFrame(response, index=sites, columns=species),
        X=pd.DataFrame({"TMG": tmg}, index=sites),
        truth_beta=pd.DataFrame(beta, index=["Intercept", "TMG"], columns=species),
        linear_predictor=pd.DataFrame(linear, index=sites, columns=species),
        metadata={
            "distribution": "probit",
            "formula": "~ TMG",
            "seed": int(seed),
            "n_sites": n_sites,
            "n_species": n_species,
            "n_covariates": 2,
            "predictor_location": float(predictor_location),
            "predictor_scale": float(predictor_scale),
            "prevalence": prevalence_name,
            "target_prevalence": float(target_prevalence),
            "effect": effect_name,
            "effect_magnitude": float(effect_magnitude),
            "replicate": int(replicate),
        },
    )


def evaluate_corpus(
    engine: FixedProbitCovarianceInference,
    corpus: Sequence[FixedEffectDataset],
    *,
    draws: int = 256,
) -> dict[str, Any]:
    if draws <= 0:
        raise ValueError("evaluation draws must be positive")
    data = fixed_shape_training_data(corpus)
    base = engine.base.predict_beta_posterior(data, calibrated=True)
    prediction = engine.predict_details(data)
    candidate = prediction.posterior
    covariance = np.asarray(tf_matmul(candidate.scale_tril), dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariance)
    truth = np.transpose(data.Beta, (0, 2, 1))
    mean = np.transpose(np.asarray(base.mean), (0, 2, 1))
    scale = np.transpose(np.asarray(base.scale), (0, 2, 1))
    candidate_nll = np.asarray(
        bivariate_beta_negative_log_probability(
            truth, mean, scale, prediction.correlation
        )
    )
    diagonal_nll = np.asarray(
        bivariate_beta_negative_log_probability(
            truth, mean, scale, np.zeros_like(prediction.correlation)
        )
    )
    raw_nll = np.asarray(
        bivariate_beta_negative_log_probability(
            truth, mean, scale, prediction.anchor_correlation
        )
    )
    records = [
        _community_diagnostics(
            dataset=dataset,
            truth=truth[index],
            mean=mean[index],
            scale=scale[index],
            covariance=covariance[index],
            candidate_nll=candidate_nll[index],
            diagonal_nll=diagonal_nll[index],
            raw_nll=raw_nll[index],
            draws=draws,
        )
        for index, dataset in enumerate(corpus)
    ]
    aggregate = _summarize_diagnostic_records(records)
    base_again = engine.base.predict_beta_posterior(data, calibrated=True)
    metrics: dict[str, Any] = {
        "max_abs_mean_delta": float(
            np.max(np.abs(np.asarray(candidate.mean) - np.asarray(base_again.mean)))
        ),
        "max_abs_scale_delta": float(
            np.max(np.abs(np.asarray(candidate.scale) - np.asarray(base_again.scale)))
        ),
        "minimum_covariance_eigenvalue": float(np.min(eigenvalues)),
        "maximum_absolute_correlation": float(
            np.max(np.abs(np.asarray(prediction.correlation)))
        ),
        "mean_absolute_fisher_z_movement": float(
            np.mean(
                np.abs(
                    np.arctanh(np.asarray(prediction.correlation) / 0.98)
                    - np.arctanh(
                        np.clip(
                            np.asarray(prediction.anchor_correlation),
                            -0.979,
                            0.979,
                        )
                        / 0.98
                    )
                )
            )
        ),
        "candidate_joint_nll": float(np.mean(candidate_nll)),
        "diagonal_joint_nll": float(np.mean(diagonal_nll)),
        "raw_laplace_joint_nll": float(np.mean(raw_nll)),
        "candidate_diagonal_joint_nll_ratio": float(
            np.mean(candidate_nll) / np.mean(diagonal_nll)
        ),
        "candidate_raw_laplace_joint_nll_ratio": float(
            np.mean(candidate_nll) / np.mean(raw_nll)
        ),
        **aggregate,
    }
    metrics["strata"] = {}
    for field in (
        "predictor_location",
        "predictor_scale",
        "prevalence",
        "effect",
    ):
        values = []
        for record in records:
            value = record["metadata"][field]
            if value not in values:
                values.append(value)
        metrics["strata"][field] = {
            str(value): _summarize_diagnostic_records(
                [record for record in records if record["metadata"][field] == value]
            )
            for value in values
        }
    return metrics


def fixed_validation_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    """Apply every preregistered 212M aggregate and non-MCMC stratum gate."""
    gates = {
        "mean_parity": metrics["max_abs_mean_delta"] <= 1e-7,
        "scale_parity": metrics["max_abs_scale_delta"] <= 1e-7,
        "positive_definite": metrics["minimum_covariance_eigenvalue"] > 1e-8,
        "correlation_bound": metrics["maximum_absolute_correlation"] <= 0.98,
        "marginal_coverage": 0.925 <= metrics["marginal_coverage_95"] <= 0.975,
        "marginal_rank_mean": abs(metrics["marginal_rank_mean"] - 0.5) <= 0.025,
        "marginal_rank_variance": (
            abs(metrics["marginal_rank_variance"] - (1.0 / 12.0)) <= 0.025
        ),
        "joint_ellipse_coverage": 0.925
        <= metrics["joint_ellipse_coverage_95"]
        <= 0.975,
        "radial_rank_mean": abs(metrics["radial_rank_mean"] - 0.5) <= 0.025,
        "radial_rank_variance": (
            abs(metrics["radial_rank_variance"] - (1.0 / 12.0)) <= 0.025
        ),
        "joint_nll_vs_diagonal": (
            metrics["candidate_diagonal_joint_nll_ratio"] <= 0.99
        ),
        "joint_nll_vs_raw_laplace": (
            metrics["candidate_raw_laplace_joint_nll_ratio"] <= 0.995
        ),
        "nonzero_movement": metrics["mean_absolute_fisher_z_movement"] >= 0.01,
        "heldout_brier": metrics["candidate_diagonal_brier_ratio"] <= 1.02,
        "heldout_log_loss": metrics["candidate_diagonal_log_loss_ratio"] <= 1.02,
    }
    for field, groups in metrics["strata"].items():
        for value, row in groups.items():
            prefix = f"stratum_{field}_{value}"
            gates[f"{prefix}_marginal_coverage"] = (
                0.90 <= row["marginal_coverage_95"] <= 0.99
            )
            gates[f"{prefix}_joint_coverage"] = (
                0.90 <= row["joint_ellipse_coverage_95"] <= 0.99
            )
            gates[f"{prefix}_marginal_rank_mean"] = (
                abs(row["marginal_rank_mean"] - 0.5) <= 0.05
            )
            gates[f"{prefix}_marginal_rank_variance"] = (
                abs(row["marginal_rank_variance"] - (1.0 / 12.0)) <= 0.04
            )
            gates[f"{prefix}_radial_rank_mean"] = (
                abs(row["radial_rank_mean"] - 0.5) <= 0.05
            )
            gates[f"{prefix}_radial_rank_variance"] = (
                abs(row["radial_rank_variance"] - (1.0 / 12.0)) <= 0.04
            )
            gates[f"{prefix}_joint_nll"] = (
                row["candidate_diagonal_joint_nll_ratio"] <= 1.02
            )
            gates[f"{prefix}_brier"] = row["candidate_diagonal_brier_ratio"] <= 1.02
            gates[f"{prefix}_log_loss"] = (
                row["candidate_diagonal_log_loss_ratio"] <= 1.02
            )
    return gates


def _community_diagnostics(
    *,
    dataset: FixedEffectDataset,
    truth: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    covariance: np.ndarray,
    candidate_nll: np.ndarray,
    diagonal_nll: np.ndarray,
    raw_nll: np.ndarray,
    draws: int,
) -> dict[str, Any]:
    seed = int(dataset.metadata["seed"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4D56, 1]))
    standard = rng.normal(size=(draws, truth.shape[0], 2))
    cholesky = np.linalg.cholesky(covariance)
    sampled = mean[None, :, :] + np.einsum("sij,dsj->dsi", cholesky, standard)
    marginal_ranks = np.mean(sampled < truth[None, :, :], axis=0).reshape(-1)
    marginal_covered = np.abs(truth - mean) <= 1.959963984540054 * scale
    residual = truth - mean
    truth_radius = np.einsum(
        "si,sij,sj->s", residual, np.linalg.inv(covariance), residual
    )
    draw_radius = np.sum(np.square(standard), axis=-1)
    radial_ranks = np.mean(draw_radius < truth_radius[None, :], axis=0)
    joint_covered = truth_radius <= 5.991464547107979

    heldout_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4D56, 2]))
    z = heldout_rng.normal(size=40)
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.square(z)))
    tmg = (
        float(dataset.metadata["predictor_location"])
        + float(dataset.metadata["predictor_scale"]) * z
    )
    design = np.column_stack([np.ones(40), tmg])
    linear_truth = design @ truth.T
    heldout_y = heldout_rng.binomial(1, ndtr(linear_truth))
    linear_mean = design @ mean.T
    candidate_variance = np.einsum("ni,sij,nj->ns", design, covariance, design)
    diagonal_covariance = np.zeros_like(covariance)
    diagonal_covariance[:, 0, 0] = np.square(scale[:, 0])
    diagonal_covariance[:, 1, 1] = np.square(scale[:, 1])
    diagonal_variance = np.einsum("ni,sij,nj->ns", design, diagonal_covariance, design)
    candidate_probability = ndtr(linear_mean / np.sqrt(1.0 + candidate_variance))
    diagonal_probability = ndtr(linear_mean / np.sqrt(1.0 + diagonal_variance))
    return {
        "metadata": {
            "predictor_location": float(dataset.metadata["predictor_location"]),
            "predictor_scale": float(dataset.metadata["predictor_scale"]),
            "prevalence": str(dataset.metadata["prevalence"]),
            "effect": str(dataset.metadata["effect"]),
        },
        "marginal_covered": marginal_covered.reshape(-1),
        "marginal_ranks": marginal_ranks,
        "joint_covered": joint_covered,
        "radial_ranks": radial_ranks,
        "candidate_nll": np.asarray(candidate_nll),
        "diagonal_nll": np.asarray(diagonal_nll),
        "raw_nll": np.asarray(raw_nll),
        "heldout_y": heldout_y.reshape(-1),
        "candidate_probability": candidate_probability.reshape(-1),
        "diagonal_probability": diagonal_probability.reshape(-1),
    }


def _summarize_diagnostic_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, float]:
    if not records:
        raise ValueError("diagnostic records must not be empty")
    covered = np.concatenate([row["marginal_covered"] for row in records])
    marginal_ranks = np.concatenate([row["marginal_ranks"] for row in records])
    joint = np.concatenate([row["joint_covered"] for row in records])
    radial_ranks = np.concatenate([row["radial_ranks"] for row in records])
    candidate_nll = np.concatenate([row["candidate_nll"] for row in records])
    diagonal_nll = np.concatenate([row["diagonal_nll"] for row in records])
    raw_nll = np.concatenate([row["raw_nll"] for row in records])
    response = np.concatenate([row["heldout_y"] for row in records])
    candidate_probability = np.clip(
        np.concatenate([row["candidate_probability"] for row in records]),
        1e-7,
        1.0 - 1e-7,
    )
    diagonal_probability = np.clip(
        np.concatenate([row["diagonal_probability"] for row in records]),
        1e-7,
        1.0 - 1e-7,
    )
    candidate_brier = float(np.mean(np.square(candidate_probability - response)))
    diagonal_brier = float(np.mean(np.square(diagonal_probability - response)))
    candidate_log_loss = float(
        -np.mean(
            response * np.log(candidate_probability)
            + (1.0 - response) * np.log(1.0 - candidate_probability)
        )
    )
    diagonal_log_loss = float(
        -np.mean(
            response * np.log(diagonal_probability)
            + (1.0 - response) * np.log(1.0 - diagonal_probability)
        )
    )
    return {
        "marginal_coverage_95": float(np.mean(covered)),
        "marginal_rank_mean": float(np.mean(marginal_ranks)),
        "marginal_rank_variance": float(np.var(marginal_ranks)),
        "joint_ellipse_coverage_95": float(np.mean(joint)),
        "radial_rank_mean": float(np.mean(radial_ranks)),
        "radial_rank_variance": float(np.var(radial_ranks)),
        "candidate_diagonal_joint_nll_ratio": float(
            np.mean(candidate_nll) / np.mean(diagonal_nll)
        ),
        "candidate_raw_laplace_joint_nll_ratio": float(
            np.mean(candidate_nll) / np.mean(raw_nll)
        ),
        "candidate_brier": candidate_brier,
        "diagonal_brier": diagonal_brier,
        "candidate_diagonal_brier_ratio": candidate_brier / diagonal_brier,
        "candidate_log_loss": candidate_log_loss,
        "diagonal_log_loss": diagonal_log_loss,
        "candidate_diagonal_log_loss_ratio": (candidate_log_loss / diagonal_log_loss),
    }


def _roundtrip_deltas(
    first: FixedProbitCovarianceInference,
    second: FixedProbitCovarianceInference,
    dataset: FixedEffectDataset,
) -> dict[str, float]:
    left = first.predict_details(dataset)
    right = second.predict_details(dataset)
    return {
        "mean": float(
            np.max(
                np.abs(
                    np.asarray(left.posterior.mean) - np.asarray(right.posterior.mean)
                )
            )
        ),
        "scale": float(
            np.max(
                np.abs(
                    np.asarray(left.posterior.scale) - np.asarray(right.posterior.scale)
                )
            )
        ),
        "correlation": float(
            np.max(np.abs(np.asarray(left.correlation) - np.asarray(right.correlation)))
        ),
    }


def tf_matmul(scale_tril: Any) -> Any:
    import tensorflow as tf

    return tf.matmul(scale_tril, scale_tril, transpose_b=True)


def _assert_seed_roles(
    first: Sequence[int], second: Sequence[int], *, production: bool
) -> None:
    if set(first).intersection(second):
        raise ValueError("M56 seed roles overlap")
    if not production and any(
        211_000_001 <= seed <= 215_000_324 for seed in (*first, *second)
    ):
        raise ValueError("disposable smoke attempted to open a production seed")
    permitted = (
        range(211_000_001, 212_000_325)
        if production
        else range(291_000_001, 292_000_028)
    )
    if any(seed not in permitted for seed in (*first, *second)):
        raise ValueError("M56 seed is outside the permitted role ledger")


def _validate_protocol_documents() -> dict[str, str]:
    observed = {
        "preregistration_sha256": _file_sha256(PREREGISTRATION_PATH),
        "artifact_seed_audit_sha256": _file_sha256(AUDIT_PATH),
    }
    expected = {
        "preregistration_sha256": M56_PREREGISTRATION_SHA256,
        "artifact_seed_audit_sha256": M56_AUDIT_SHA256,
    }
    if observed != expected:
        raise ValueError(f"frozen M56 protocol document hash differs: {observed!r}")
    return observed


def _require_confirmation(observed: str, expected: str) -> None:
    if observed != expected:
        raise PermissionError(
            f"exact confirmation {expected!r} is required before opening this seed role"
        )


def _load_qualified_freeze(root: Path) -> dict[str, Any]:
    freeze = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    if freeze.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("M56 freeze protocol differs")
    if freeze.get("validation_passed") is not True:
        raise RuntimeError(
            "fixed 212M validation did not pass; evaluation remains sealed"
        )
    if freeze.get("reserved_seed_blocks_opened") is not False:
        raise ValueError("M56 freeze reports prior reserved-seed access")
    _validate_protocol_documents()
    return freeze


def _empty_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"qualification output must be empty: {path}")
        path.rmdir()
    path.mkdir(parents=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _all_numeric_values_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
