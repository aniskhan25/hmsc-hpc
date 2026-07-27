#!/usr/bin/env python3
"""Run the sealed Milestone 57 fixed-probit Student-t qualification protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import f as f_distribution

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache")
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.covariance_inference import (  # noqa: E402
    BOUND_MEMBER_SEED,
    FixedProbitCovarianceInference,
    validate_bound_v0_1_release,
)
from pyhmsc.neural.release import load_neural_hmsc_release  # noqa: E402
from pyhmsc.neural.simulator import FixedEffectDataset  # noqa: E402
from pyhmsc.neural.student_t_inference import (  # noqa: E402
    M57_AUDIT_SHA256,
    M57_DECISION_SHA256,
    M57_PREREGISTRATION_SHA256,
    STUDENT_T_OVERLAY_ID,
    STUDENT_T_OVERLAY_MANIFEST,
    STUDENT_T_OVERLAY_WEIGHTS,
    FixedProbitStudentTInference,
    bivariate_student_t_negative_log_probability,
    fit_fixed_probit_student_t_overlay,
    validate_bound_m56_negative,
    validate_bound_variable_v1,
    write_student_t_beta_posterior_hdf5,
)


PROTOCOL_ID = "neural_hmsc_fixed_probit_student_t_m57_v1"
DECISION_PATH = ROOT / "docs/neural_hmsc_post_m56_capability_decision_2026-07-24.md"
AUDIT_PATH = ROOT / "docs/neural_hmsc_m57_artifact_seed_audit_2026-07-24.json.md"
PREREGISTRATION_PATH = (
    ROOT / "docs/neural_hmsc_m57_student_t_preregistration_2026-07-24.md"
)

TRAIN_CONFIRMATION = "GENERATE_M57_STUDENT_T_TRAIN_VALIDATION"
EVALUATION_CONFIRMATION = "OPEN_M57_RESERVED_STUDENT_T_EVALUATION"
REALDATA_CONFIRMATION = "OPEN_M57_FROZEN_REALDATA_REPLAY"

MODEL_SEED = 321_900_001
PRODUCTION_COUNT = 324
SMOKE_COUNT = 27
PRODUCTION_STARTS = {
    "training": 321_000_001,
    "validation": 322_000_001,
    "evaluation_a": 323_000_001,
    "evaluation_b": 324_000_001,
    "evaluation_c": 325_000_001,
}
SMOKE_STARTS = {"training": 391_000_001, "evaluation": 392_000_001}
CONTEXT_TAG = 0x4D5701
OBSERVED_RESPONSE_TAG = 0x4D5702
HELDOUT_RESPONSE_TAG = 0x4D5703
NEURAL_DRAW_TAG = 0x4D5704
MCMC_TAG = 0x4D5705
REALDATA_TAG = 0x4D5706

LOCATIONS = (-1.5, 0.0, 1.5)
SCALES = (0.5, 1.0, 2.0)
PREVALENCES = (("rare", 0.05), ("balanced", 0.30), ("common", 0.65))
EFFECTS = (("weak", 0.25), ("moderate", 0.75), ("strong", 1.50))
MCMC_OFFSETS = (1, 105, 161, 185, 241, 297)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-registry",
        type=Path,
        default=Path("/private/tmp/neural_hmsc_releases"),
    )
    parser.add_argument(
        "--variable-registry",
        type=Path,
        default=Path("/private/tmp/neural_hmsc_variable_deployments"),
    )
    parser.add_argument(
        "--m56-root",
        type=Path,
        default=Path("/private/tmp/neural_hmsc_m56_train_validation_20192218"),
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

    validate = subparsers.add_parser("validate")
    validate.add_argument("--freeze-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(args)
    elif args.command == "train-validate":
        result = train_and_validate(args)
    elif args.command == "evaluate":
        result = evaluate_reserved(args)
    elif args.command == "realdata":
        result = authorize_realdata(args)
    else:
        result = validate_freeze(
            args.freeze_root,
            release_registry=args.release_registry,
            variable_registry=args.variable_registry,
            m56_root=args.m56_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    protocols, bindings = _validate_prerequisites(args)
    output = _empty_output(args.output)
    train_seeds = _seed_range("smoke_training")
    evaluation_seeds = _seed_range("smoke_evaluation")
    _assert_seed_roles(train_seeds, evaluation_seeds, production=False)
    training = build_m57_corpus(train_seeds, paired=True, smoke=True)
    evaluation = build_m57_corpus(evaluation_seeds, paired=False, smoke=True)
    release = load_neural_hmsc_release(args.release_registry)
    engine, history = fit_fixed_probit_student_t_overlay(
        release.load_checkpoint(seed=BOUND_MEMBER_SEED),
        training,
        base_binding=bindings["v0_1"],
        variable_v1_binding=bindings["variable_v1"],
        m56_negative_binding=bindings["m56_negative"],
        model_seed=MODEL_SEED,
        epochs=1,
        batch_contexts=9,
        learning_rate=0.0005,
    )
    artifact = engine.save(output / "overlay")
    loaded = FixedProbitStudentTInference.load(
        artifact,
        registry_root=args.release_registry,
        variable_registry_root=args.variable_registry,
        m56_root=args.m56_root,
    )
    m56 = FixedProbitCovarianceInference.load(
        Path(args.m56_root) / "overlay", registry_root=args.release_registry
    )
    metrics = evaluate_corpus(loaded, evaluation, draws=32, m56_engine=m56)
    roundtrip = _roundtrip_deltas(engine, loaded, evaluation[0])
    hdf5 = _hdf5_fixture_check(loaded, evaluation[0], output)
    checks = {
        "protocol_hashes_valid": True,
        "immutable_bindings_valid": True,
        "exact_disposable_seed_roles": True,
        "production_seeds_remain_unopened": True,
        "paired_training_realizations": len(training) == 2 * SMOKE_COUNT,
        "training_finite": bool(np.isfinite(history["loss"][-1])),
        "parameters_finite": metrics["all_parameters_finite"],
        "positive_definite": metrics["minimum_covariance_eigenvalue"] > 1e-8,
        "correlation_bounded": metrics["maximum_absolute_correlation"] <= 0.98,
        "degrees_of_freedom_bounded": (
            metrics["minimum_degrees_of_freedom"] >= 2.1 - 1e-6
            and metrics["maximum_degrees_of_freedom"] <= 30.0 + 1e-6
        ),
        "artifact_roundtrip": max(roundtrip.values()) <= 1e-7,
        "hdf5_shape_and_metadata": hdf5["passed"],
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "mode": "disposable_smoke",
        "promotion_evidence": False,
        "production_seed_opened": False,
        "protocol_hashes": protocols,
        "bindings": bindings,
        "seed_roles": {
            "training": [train_seeds[0], train_seeds[-1]],
            "evaluation": [evaluation_seeds[0], evaluation_seeds[-1]],
            "training_realizations": len(training),
            "evaluation_realizations": len(evaluation),
        },
        "training": engine.training_record,
        "metrics": metrics,
        "roundtrip_max_abs_delta": roundtrip,
        "hdf5_fixture": hdf5,
        "checks": checks,
        "passed": all(checks.values()),
        "artifact": {
            "id": STUDENT_T_OVERLAY_ID,
            "path": str(artifact),
            "manifest_sha256": _file_sha256(artifact / STUDENT_T_OVERLAY_MANIFEST),
            "weights_sha256": _file_sha256(artifact / STUDENT_T_OVERLAY_WEIGHTS),
        },
    }
    _write_json(output / "smoke_report.json", report)
    return report


def train_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    _require_confirmation(args.confirmation, TRAIN_CONFIRMATION)
    protocols, bindings = _validate_prerequisites(args)
    output = _empty_output(args.output)
    training_seeds = _seed_range("training")
    validation_seeds = _seed_range("validation")
    _assert_seed_roles(training_seeds, validation_seeds, production=True)
    training = build_m57_corpus(training_seeds, paired=True, smoke=False)
    validation = build_m57_corpus(validation_seeds, paired=False, smoke=False)
    release = load_neural_hmsc_release(args.release_registry)
    engine, history = fit_fixed_probit_student_t_overlay(
        release.load_checkpoint(seed=BOUND_MEMBER_SEED),
        training,
        base_binding=bindings["v0_1"],
        variable_v1_binding=bindings["variable_v1"],
        m56_negative_binding=bindings["m56_negative"],
        model_seed=MODEL_SEED,
        epochs=150,
        batch_contexts=9,
        learning_rate=0.0005,
    )
    artifact = engine.save(output / "overlay")
    m56 = FixedProbitCovarianceInference.load(
        Path(args.m56_root) / "overlay", registry_root=args.release_registry
    )
    metrics = evaluate_corpus(engine, validation, draws=512, m56_engine=m56)
    gates = fixed_validation_gates(metrics)
    freeze = {
        "protocol_id": PROTOCOL_ID,
        "mode": "production_train_validation",
        "protocol_hashes": protocols,
        "bindings": bindings,
        "training_seed_range": [training_seeds[0], training_seeds[-1]],
        "validation_seed_range": [validation_seeds[0], validation_seeds[-1]],
        "reserved_seed_blocks_opened": False,
        "training": engine.training_record,
        "history_final": {key: values[-1] for key, values in history.items()},
        "validation_metrics": metrics,
        "validation_gates": gates,
        "validation_passed": all(gates.values()),
        "overlay_manifest_sha256": _file_sha256(artifact / STUDENT_T_OVERLAY_MANIFEST),
        "overlay_weights_sha256": _file_sha256(artifact / STUDENT_T_OVERLAY_WEIGHTS),
    }
    _write_json(output / "freeze.json", freeze)
    _write_text(output / "freeze.sha256", _file_sha256(output / "freeze.json") + "\n")
    return freeze


def evaluate_reserved(args: argparse.Namespace) -> dict[str, Any]:
    _require_confirmation(args.confirmation, EVALUATION_CONFIRMATION)
    freeze = validate_freeze(
        args.freeze_root,
        release_registry=args.release_registry,
        variable_registry=args.variable_registry,
        m56_root=args.m56_root,
    )
    if freeze.get("validation_passed") is not True:
        raise PermissionError(
            "reserved M57 evaluation requires every 322M gate to pass"
        )
    output = _empty_output(args.output)
    blocks: dict[str, Any] = {}
    engine = FixedProbitStudentTInference.load(
        Path(args.freeze_root) / "overlay",
        registry_root=args.release_registry,
        variable_registry_root=args.variable_registry,
        m56_root=args.m56_root,
    )
    for role in ("evaluation_a", "evaluation_b", "evaluation_c"):
        seeds = _seed_range(role)
        datasets = build_m57_corpus(seeds, paired=False, smoke=False)
        blocks[role] = {
            "seed_range": [seeds[0], seeds[-1]],
            "mcmc_seeds": [seeds[offset - 1] for offset in MCMC_OFFSETS],
            "metrics": evaluate_corpus(engine, datasets, draws=512),
        }
    report = {
        "protocol_id": PROTOCOL_ID,
        "mode": "one_shot_reserved_evaluation",
        "confirmation": EVALUATION_CONFIRMATION,
        "freeze_sha256": _file_sha256(Path(args.freeze_root) / "freeze.json"),
        "blocks": blocks,
        "realdata_opened": False,
    }
    _write_json(output / "reserved_evaluation.json", report)
    return report


def authorize_realdata(args: argparse.Namespace) -> dict[str, Any]:
    _require_confirmation(args.confirmation, REALDATA_CONFIRMATION)
    evaluation = Path(args.evaluation_root) / "reserved_evaluation.json"
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("reserved evaluation protocol differs")
    if payload.get("simulation_and_mcmc_gates_passed") is not True:
        raise PermissionError(
            "real-data replay remains sealed until all simulation and MCMC gates pass"
        )
    output = _empty_output(args.output)
    report = {
        "protocol_id": PROTOCOL_ID,
        "mode": "frozen_realdata_replay_authorized",
        "confirmation": REALDATA_CONFIRMATION,
        "evaluation_sha256": _file_sha256(evaluation),
        "permitted_datasets": ["whittaker", "big_spatial"],
        "target_outcome_fitting_or_selection": False,
    }
    _write_json(output / "realdata_authorization.json", report)
    return report


def validate_freeze(
    freeze_root: str | Path,
    *,
    release_registry: str | Path,
    variable_registry: str | Path,
    m56_root: str | Path,
) -> dict[str, Any]:
    root = Path(freeze_root)
    freeze_path = root / "freeze.json"
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("M57 freeze protocol differs")
    if payload.get("training_seed_range") != [321_000_001, 321_000_324]:
        raise ValueError("M57 training seed range differs")
    if payload.get("validation_seed_range") != [322_000_001, 322_000_324]:
        raise ValueError("M57 validation seed range differs")
    if payload.get("reserved_seed_blocks_opened") is not False:
        raise ValueError("M57 freeze unexpectedly opened reserved blocks")
    args = argparse.Namespace(
        release_registry=Path(release_registry),
        variable_registry=Path(variable_registry),
        m56_root=Path(m56_root),
    )
    protocols, bindings = _validate_prerequisites(args)
    if (
        payload.get("protocol_hashes") != protocols
        or payload.get("bindings") != bindings
    ):
        raise ValueError("M57 freeze prerequisite binding differs")
    overlay = root / "overlay"
    if _file_sha256(overlay / STUDENT_T_OVERLAY_MANIFEST) != payload.get(
        "overlay_manifest_sha256"
    ):
        raise ValueError("M57 overlay manifest hash differs")
    if _file_sha256(overlay / STUDENT_T_OVERLAY_WEIGHTS) != payload.get(
        "overlay_weights_sha256"
    ):
        raise ValueError("M57 overlay weight hash differs")
    FixedProbitStudentTInference.load(
        overlay,
        registry_root=release_registry,
        variable_registry_root=variable_registry,
        m56_root=m56_root,
    )
    return payload


def build_m57_corpus(
    seeds: Sequence[int], *, paired: bool, smoke: bool
) -> list[FixedEffectDataset]:
    seeds = tuple(int(value) for value in seeds)
    expected = _context_grid(smoke=smoke)
    if len(seeds) != len(expected):
        raise ValueError("M57 seed count does not match the frozen context grid")
    datasets: list[FixedEffectDataset] = []
    for seed, context in zip(seeds, expected):
        base = simulate_m57_community(seed=seed, response_replicate=0, **context)
        datasets.append(base)
        if paired:
            datasets.append(
                simulate_m57_community(seed=seed, response_replicate=1, **context)
            )
    return datasets


def simulate_m57_community(
    *,
    seed: int,
    predictor_location: float,
    predictor_scale: float,
    prevalence_name: str,
    target_prevalence: float,
    effect_name: str,
    effect_magnitude: float,
    context_replicate: int,
    response_replicate: int,
) -> FixedEffectDataset:
    context_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), CONTEXT_TAG])
    )
    z = context_rng.normal(size=40)
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.square(z)))
    tmg = predictor_location + predictor_scale * z
    sign = context_rng.choice(np.asarray([-1.0, 1.0]), size=75)
    slope = sign * (
        effect_magnitude + context_rng.normal(scale=0.10 * effect_magnitude, size=75)
    )
    intercept = (
        ndtri(target_prevalence)
        - slope * predictor_location
        + context_rng.normal(scale=0.15, size=75)
    )
    beta = np.stack([intercept, slope], axis=0)
    linear = np.column_stack([np.ones(40), tmg]) @ beta
    probability = ndtr(linear)
    response_rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), OBSERVED_RESPONSE_TAG, int(response_replicate)]
        )
    )
    response = response_rng.binomial(1, probability).astype(np.float32)
    species = [f"sp{index + 1:03d}" for index in range(75)]
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "seed": int(seed),
        "owning_context_seed": int(seed),
        "response_replicate": int(response_replicate),
        "predictor_location": float(predictor_location),
        "predictor_scale": float(predictor_scale),
        "prevalence": prevalence_name,
        "target_prevalence": float(target_prevalence),
        "effect": effect_name,
        "effect_magnitude": float(effect_magnitude),
        "context_replicate": int(context_replicate),
        "rng_tags": {
            "context": CONTEXT_TAG,
            "observed_response": OBSERVED_RESPONSE_TAG,
        },
    }
    return FixedEffectDataset(
        Y=pd.DataFrame(response, columns=species),
        X=pd.DataFrame({"TMG": tmg.astype(np.float32)}),
        truth_beta=pd.DataFrame(
            beta.astype(np.float32),
            index=["Intercept", "TMG"],
            columns=species,
        ),
        linear_predictor=pd.DataFrame(linear.astype(np.float32), columns=species),
        metadata=metadata,
    )


def evaluate_corpus(
    engine: FixedProbitStudentTInference,
    datasets: Sequence[FixedEffectDataset],
    *,
    draws: int,
    m56_engine: FixedProbitCovarianceInference | None = None,
) -> dict[str, Any]:
    from pyhmsc.neural.train import fixed_shape_training_data

    if draws <= 0:
        raise ValueError("evaluation draws must be positive")
    data = fixed_shape_training_data(datasets)
    prediction = engine.predict_details(data)
    posterior = prediction.posterior
    candidate_nll = np.asarray(
        bivariate_student_t_negative_log_probability(posterior, data.Beta)
    )
    base = prediction.base_posterior
    laplace = prediction.laplace_posterior
    truth = np.transpose(np.asarray(data.Beta, dtype=np.float64), (0, 2, 1))
    base_mean = np.transpose(np.asarray(base.mean, dtype=np.float64), (0, 2, 1))
    base_scale = np.transpose(np.asarray(base.scale, dtype=np.float64), (0, 2, 1))
    laplace_mean = np.transpose(np.asarray(laplace.mean, dtype=np.float64), (0, 2, 1))
    laplace_covariance = _covariance_from_tril(laplace.scale_tril)
    diagonal_covariance = np.zeros((len(datasets), 75, 2, 2), dtype=np.float64)
    diagonal_covariance[..., 0, 0] = np.square(base_scale[..., 0])
    diagonal_covariance[..., 1, 1] = np.square(base_scale[..., 1])
    diagonal_nll = _gaussian_nll(truth, base_mean, diagonal_covariance)
    laplace_nll = _gaussian_nll(truth, laplace_mean, laplace_covariance)
    covariance_tril = np.asarray(posterior.covariance_tril)
    covariance = covariance_tril @ np.swapaxes(covariance_tril, -1, -2)
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    correlation = covariance[..., 0, 1] / np.sqrt(
        np.maximum(diagonal[..., 0] * diagonal[..., 1], 1e-12)
    )
    m56_prediction = None
    m56_nll = None
    if m56_engine is not None:
        m56_prediction = m56_engine.predict_details(data)
        m56_covariance = _covariance_from_tril(m56_prediction.posterior.scale_tril)
        m56_mean = np.transpose(
            np.asarray(m56_prediction.posterior.mean, dtype=np.float64),
            (0, 2, 1),
        )
        m56_nll = _gaussian_nll(truth, m56_mean, m56_covariance)
    records = []
    for index, dataset in enumerate(datasets):
        candidate = _slice_student_t_posterior(posterior, index)
        candidate_a = _student_draws_for_context(
            candidate, int(dataset.metadata["seed"]), draws, offset=0
        )
        candidate_b = _student_draws_for_context(
            candidate, int(dataset.metadata["seed"]), draws, offset=draws
        )
        base_a = _gaussian_draws_for_context(
            base_mean[index],
            np.linalg.cholesky(diagonal_covariance[index]),
            int(dataset.metadata["seed"]),
            draws,
            offset=0,
        )
        base_b = _gaussian_draws_for_context(
            base_mean[index],
            np.linalg.cholesky(diagonal_covariance[index]),
            int(dataset.metadata["seed"]),
            draws,
            offset=draws,
        )
        m56_a = m56_b = None
        if m56_prediction is not None:
            m56_covariance_i = _covariance_from_tril(
                m56_prediction.posterior.scale_tril
            )[index]
            m56_mean_i = np.transpose(
                np.asarray(m56_prediction.posterior.mean)[index], (1, 0)
            )
            m56_a = _gaussian_draws_for_context(
                m56_mean_i,
                np.linalg.cholesky(m56_covariance_i),
                int(dataset.metadata["seed"]),
                draws,
                offset=0,
            )
            m56_b = _gaussian_draws_for_context(
                m56_mean_i,
                np.linalg.cholesky(m56_covariance_i),
                int(dataset.metadata["seed"]),
                draws,
                offset=draws,
            )
        records.append(
            _community_diagnostics(
                dataset=dataset,
                truth=truth[index],
                candidate=candidate,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
                base_mean=base_mean[index],
                base_scale=base_scale[index],
                base_a=base_a,
                base_b=base_b,
                candidate_nll=candidate_nll[index],
                diagonal_nll=diagonal_nll[index],
                laplace_nll=laplace_nll[index],
                m56_nll=(None if m56_nll is None else m56_nll[index]),
                m56_a=m56_a,
                m56_b=m56_b,
            )
        )
    aggregate = _summarize_records(records)
    metrics: dict[str, Any] = {
        "community_count": len(datasets),
        "posterior_draws": int(draws),
        "all_parameters_finite": bool(
            np.all(np.isfinite(posterior.mean))
            and np.all(np.isfinite(posterior.marginal_scale))
            and np.all(np.isfinite(covariance_tril))
            and np.all(np.isfinite(posterior.degrees_of_freedom))
        ),
        "minimum_covariance_eigenvalue": float(np.min(np.linalg.eigvalsh(covariance))),
        "maximum_absolute_correlation": float(np.max(np.abs(correlation))),
        "minimum_degrees_of_freedom": float(np.min(posterior.degrees_of_freedom)),
        "maximum_degrees_of_freedom": float(np.max(posterior.degrees_of_freedom)),
        **aggregate,
    }
    metrics["strata"] = {}
    for field in (
        "predictor_location",
        "predictor_scale",
        "prevalence",
        "effect",
    ):
        values = list(dict.fromkeys(row["metadata"][field] for row in records))
        metrics["strata"][field] = {
            str(value): _summarize_records(
                [row for row in records if row["metadata"][field] == value]
            )
            for value in values
        }
    metrics["strata"]["coefficient"] = {
        name: _summarize_records(records, coefficient=index)
        for index, name in enumerate(("Intercept", "TMG"))
    }
    return metrics


def fixed_validation_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    """Apply all frozen non-MCMC aggregate and stratum gates."""
    gates = {
        "all_parameters_finite": bool(metrics["all_parameters_finite"]),
        "positive_definite": metrics["minimum_covariance_eigenvalue"] > 1e-8,
        "correlation_bounded": metrics["maximum_absolute_correlation"] <= 0.98,
        "degrees_of_freedom_bounded": (
            metrics["minimum_degrees_of_freedom"] >= 2.1 - 1e-6
            and metrics["maximum_degrees_of_freedom"] <= 30.0 + 1e-6
        ),
        "marginal_95_coverage": 0.925 <= metrics["marginal_coverage_95"] <= 0.975,
        "marginal_50_coverage": 0.475 <= metrics["marginal_coverage_50"] <= 0.525,
        "marginal_rank_mean": abs(metrics["marginal_rank_mean"] - 0.5) <= 0.025,
        "marginal_rank_variance": (
            abs(metrics["marginal_rank_variance"] - (1.0 / 12.0)) <= 0.025
        ),
        "joint_95_coverage": 0.925 <= metrics["joint_coverage_95"] <= 0.975,
        "radial_rank_mean": abs(metrics["radial_rank_mean"] - 0.5) <= 0.025,
        "radial_rank_variance": (
            abs(metrics["radial_rank_variance"] - (1.0 / 12.0)) <= 0.025
        ),
        "location_rmse": metrics["candidate_base_location_rmse_ratio"] <= 1.05,
        "width_ratio": 0.80 <= metrics["geometric_width_ratio"] <= 2.00,
        "df_saturation": metrics["degrees_of_freedom_bound_fraction"] <= 0.10,
        "scale_saturation": metrics["scale_multiplier_bound_fraction"] <= 0.10,
        "joint_log_score_vs_diagonal": (
            metrics["candidate_diagonal_normalized_log_score_delta"] <= -0.02
        ),
        "joint_log_score_vs_laplace": (
            metrics["candidate_laplace_normalized_log_score_delta"] <= -0.05
        ),
        "joint_log_score_vs_m56": (
            metrics["candidate_m56_normalized_log_score_delta"] <= -0.05
        ),
        "energy_score_vs_diagonal": metrics["candidate_diagonal_energy_ratio"] <= 0.99,
        "energy_score_vs_m56": metrics["candidate_m56_energy_ratio"] <= 0.99,
        "heldout_brier": metrics["candidate_diagonal_brier_ratio"] <= 1.02,
        "heldout_log_loss": metrics["candidate_diagonal_log_loss_ratio"] <= 1.02,
    }
    for field, groups in metrics["strata"].items():
        for value, row in groups.items():
            prefix = f"stratum_{field}_{value}"
            gates[f"{prefix}_marginal_coverage"] = (
                0.90 <= row["marginal_coverage_95"] <= 0.99
            )
            gates[f"{prefix}_joint_coverage"] = 0.90 <= row["joint_coverage_95"] <= 0.99
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
            gates[f"{prefix}_joint_log_score"] = (
                row["candidate_diagonal_normalized_log_score_delta"] <= 0.02
            )
            gates[f"{prefix}_energy_score"] = (
                row["candidate_diagonal_energy_ratio"] <= 1.02
            )
            gates[f"{prefix}_brier"] = row["candidate_diagonal_brier_ratio"] <= 1.02
            gates[f"{prefix}_log_loss"] = (
                row["candidate_diagonal_log_loss_ratio"] <= 1.02
            )
            gates[f"{prefix}_location_rmse"] = (
                row["candidate_base_location_rmse_ratio"] <= 1.10
            )
    return gates


def _community_diagnostics(
    *,
    dataset: FixedEffectDataset,
    truth: np.ndarray,
    candidate: Any,
    candidate_a: np.ndarray,
    candidate_b: np.ndarray,
    base_mean: np.ndarray,
    base_scale: np.ndarray,
    base_a: np.ndarray,
    base_b: np.ndarray,
    candidate_nll: np.ndarray,
    diagonal_nll: np.ndarray,
    laplace_nll: np.ndarray,
    m56_nll: np.ndarray | None,
    m56_a: np.ndarray | None,
    m56_b: np.ndarray | None,
) -> dict[str, Any]:
    candidate_mean = np.transpose(np.asarray(candidate.mean)[0], (1, 0))
    candidate_scale = np.transpose(np.asarray(candidate.marginal_scale)[0], (1, 0))
    candidate_covariance = _covariance_from_tril(candidate.covariance_tril)[0]
    degrees = np.asarray(candidate.degrees_of_freedom)[0]
    marginal_ranks = np.mean(candidate_a < truth[None, ...], axis=0)
    lower_95, upper_95 = np.quantile(candidate_a, [0.025, 0.975], axis=0)
    lower_50, upper_50 = np.quantile(candidate_a, [0.25, 0.75], axis=0)
    marginal_covered_95 = (truth >= lower_95) & (truth <= upper_95)
    marginal_covered_50 = (truth >= lower_50) & (truth <= upper_50)
    residual = truth - candidate_mean
    inverse_covariance = np.linalg.inv(candidate_covariance)
    truth_radius = np.einsum("si,sij,sj->s", residual, inverse_covariance, residual)
    draw_residual = candidate_a - candidate_mean[None, ...]
    draw_radius = np.einsum(
        "dsi,sij,dsj->ds", draw_residual, inverse_covariance, draw_residual
    )
    radial_ranks = np.mean(draw_radius < truth_radius[None, :], axis=0)
    threshold = (
        ((degrees - 2.0) / degrees) * 2.0 * f_distribution.ppf(0.95, dfn=2, dfd=degrees)
    )
    joint_covered = truth_radius <= threshold
    candidate_energy = _energy_score(candidate_a, candidate_b, truth)
    diagonal_energy = _energy_score(base_a, base_b, truth)
    m56_energy = (
        None if m56_a is None or m56_b is None else _energy_score(m56_a, m56_b, truth)
    )
    design = np.column_stack([np.ones(40), dataset.X["TMG"].to_numpy(dtype=np.float64)])
    response_rng = np.random.default_rng(
        np.random.SeedSequence([int(dataset.metadata["seed"]), HELDOUT_RESPONSE_TAG, 0])
    )
    heldout = response_rng.binomial(1, ndtr(design @ truth.T))
    candidate_probability = np.mean(
        ndtr(np.einsum("nk,dsk->dns", design, candidate_a)), axis=0
    )
    diagonal_probability = np.mean(
        ndtr(np.einsum("nk,dsk->dns", design, base_a)), axis=0
    )
    width_ratio = (upper_95 - lower_95) / np.maximum(
        2.0 * 1.959963984540054 * base_scale, 1e-12
    )
    log_scale_multiplier = np.log(
        np.maximum(candidate_scale, 1e-12) / np.maximum(base_scale, 1e-12)
    )
    return {
        "metadata": {
            "predictor_location": float(dataset.metadata["predictor_location"]),
            "predictor_scale": float(dataset.metadata["predictor_scale"]),
            "prevalence": str(dataset.metadata["prevalence"]),
            "effect": str(dataset.metadata["effect"]),
        },
        "marginal_covered_95": marginal_covered_95,
        "marginal_covered_50": marginal_covered_50,
        "marginal_ranks": marginal_ranks,
        "joint_covered": joint_covered,
        "radial_ranks": radial_ranks,
        "candidate_nll": np.asarray(candidate_nll),
        "diagonal_nll": np.asarray(diagonal_nll),
        "laplace_nll": np.asarray(laplace_nll),
        "m56_nll": None if m56_nll is None else np.asarray(m56_nll),
        "candidate_energy": candidate_energy,
        "diagonal_energy": diagonal_energy,
        "m56_energy": m56_energy,
        "heldout": heldout,
        "candidate_probability": candidate_probability,
        "diagonal_probability": diagonal_probability,
        "candidate_location_squared_error": np.square(candidate_mean - truth),
        "base_location_squared_error": np.square(base_mean - truth),
        "width_ratio": width_ratio,
        "df_bound": (degrees <= 2.35) | (degrees >= 29.75),
        "scale_bound": (np.abs(log_scale_multiplier + 1.5) <= 0.02)
        | (np.abs(log_scale_multiplier - 1.5) <= 0.02),
    }


def _summarize_records(
    records: Sequence[dict[str, Any]], coefficient: int | None = None
) -> dict[str, float]:
    if not records:
        raise ValueError("M57 diagnostic records must not be empty")

    def coefficients(name: str) -> np.ndarray:
        values = [np.asarray(row[name]) for row in records]
        if coefficient is not None:
            values = [value[..., coefficient] for value in values]
        return np.concatenate([value.reshape(-1) for value in values])

    def species(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(row[name]).reshape(-1) for row in records])

    candidate_nll = species("candidate_nll")
    diagonal_nll = species("diagonal_nll")
    laplace_nll = species("laplace_nll")
    m56_values = [row["m56_nll"] for row in records]
    m56_nll = (
        np.asarray([np.nan])
        if any(value is None for value in m56_values)
        else np.concatenate([np.asarray(value).reshape(-1) for value in m56_values])
    )
    candidate_energy = species("candidate_energy")
    diagonal_energy = species("diagonal_energy")
    m56_energy_values = [row["m56_energy"] for row in records]
    m56_energy = (
        np.asarray([np.nan])
        if any(value is None for value in m56_energy_values)
        else np.concatenate(
            [np.asarray(value).reshape(-1) for value in m56_energy_values]
        )
    )
    response = species("heldout")
    candidate_probability = np.clip(species("candidate_probability"), 1e-7, 1.0 - 1e-7)
    diagonal_probability = np.clip(species("diagonal_probability"), 1e-7, 1.0 - 1e-7)
    candidate_brier = float(np.mean(np.square(candidate_probability - response)))
    diagonal_brier = float(np.mean(np.square(diagonal_probability - response)))
    candidate_log_loss = _binary_log_loss(response, candidate_probability)
    diagonal_log_loss = _binary_log_loss(response, diagonal_probability)
    candidate_location = coefficients("candidate_location_squared_error")
    base_location = coefficients("base_location_squared_error")
    marginal_ranks = coefficients("marginal_ranks")
    radial_ranks = species("radial_ranks")
    candidate_mean_nll = float(np.mean(candidate_nll))
    diagonal_mean_nll = float(np.mean(diagonal_nll))
    laplace_mean_nll = float(np.mean(laplace_nll))
    m56_mean_nll = float(np.mean(m56_nll))
    return {
        "marginal_coverage_95": float(np.mean(coefficients("marginal_covered_95"))),
        "marginal_coverage_50": float(np.mean(coefficients("marginal_covered_50"))),
        "marginal_rank_mean": float(np.mean(marginal_ranks)),
        "marginal_rank_variance": float(np.var(marginal_ranks)),
        "joint_coverage_95": float(np.mean(species("joint_covered"))),
        "radial_rank_mean": float(np.mean(radial_ranks)),
        "radial_rank_variance": float(np.var(radial_ranks)),
        "candidate_joint_nll": candidate_mean_nll,
        "diagonal_joint_nll": diagonal_mean_nll,
        "laplace_joint_nll": laplace_mean_nll,
        "m56_joint_nll": m56_mean_nll,
        "candidate_diagonal_normalized_log_score_delta": _normalized_delta(
            candidate_mean_nll, diagonal_mean_nll
        ),
        "candidate_laplace_normalized_log_score_delta": _normalized_delta(
            candidate_mean_nll, laplace_mean_nll
        ),
        "candidate_m56_normalized_log_score_delta": _normalized_delta(
            candidate_mean_nll, m56_mean_nll
        ),
        "candidate_energy_score": float(np.mean(candidate_energy)),
        "diagonal_energy_score": float(np.mean(diagonal_energy)),
        "m56_energy_score": float(np.mean(m56_energy)),
        "candidate_diagonal_energy_ratio": float(
            np.mean(candidate_energy) / np.mean(diagonal_energy)
        ),
        "candidate_m56_energy_ratio": float(
            np.mean(candidate_energy) / np.mean(m56_energy)
        ),
        "candidate_brier": candidate_brier,
        "diagonal_brier": diagonal_brier,
        "candidate_diagonal_brier_ratio": candidate_brier / diagonal_brier,
        "candidate_log_loss": candidate_log_loss,
        "diagonal_log_loss": diagonal_log_loss,
        "candidate_diagonal_log_loss_ratio": candidate_log_loss / diagonal_log_loss,
        "candidate_location_rmse": float(np.sqrt(np.mean(candidate_location))),
        "base_location_rmse": float(np.sqrt(np.mean(base_location))),
        "candidate_base_location_rmse_ratio": float(
            np.sqrt(np.mean(candidate_location)) / np.sqrt(np.mean(base_location))
        ),
        "geometric_width_ratio": float(
            np.exp(np.mean(np.log(np.maximum(coefficients("width_ratio"), 1e-12))))
        ),
        "degrees_of_freedom_bound_fraction": float(np.mean(species("df_bound"))),
        "scale_multiplier_bound_fraction": float(np.mean(coefficients("scale_bound"))),
    }


def _slice_student_t_posterior(posterior: Any, index: int) -> Any:
    from pyhmsc.neural.student_t_inference import StudentTBetaPosterior

    return StudentTBetaPosterior(
        mean=posterior.mean[index : index + 1],
        marginal_scale=posterior.marginal_scale[index : index + 1],
        covariance_tril=posterior.covariance_tril[index : index + 1],
        student_t_scale_tril=posterior.student_t_scale_tril[index : index + 1],
        degrees_of_freedom=posterior.degrees_of_freedom[index : index + 1],
    )


def _student_draws_for_context(
    posterior: Any, seed: int, draws: int, *, offset: int
) -> np.ndarray:
    mean = np.transpose(np.asarray(posterior.mean)[0], (1, 0))
    covariance_tril = np.asarray(posterior.covariance_tril)[0]
    degrees = np.asarray(posterior.degrees_of_freedom)[0]
    result = np.empty((draws, mean.shape[0], 2), dtype=np.float64)
    for draw in range(draws):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), NEURAL_DRAW_TAG, offset + draw])
        )
        normal = rng.normal(size=(mean.shape[0], 2))
        chi_square = rng.chisquare(df=degrees)
        correlated = np.einsum("sij,sj->si", covariance_tril, normal)
        result[draw] = (
            mean + np.sqrt((degrees - 2.0) / chi_square)[:, None] * correlated
        )
    return result


def _gaussian_draws_for_context(
    mean: np.ndarray,
    covariance_tril: np.ndarray,
    seed: int,
    draws: int,
    *,
    offset: int,
) -> np.ndarray:
    result = np.empty((draws, mean.shape[0], 2), dtype=np.float64)
    for draw in range(draws):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), NEURAL_DRAW_TAG, offset + draw])
        )
        normal = rng.normal(size=(mean.shape[0], 2))
        result[draw] = mean + np.einsum("sij,sj->si", covariance_tril, normal)
    return result


def _covariance_from_tril(value: Any) -> np.ndarray:
    scale_tril = np.asarray(value, dtype=np.float64)
    return scale_tril @ np.swapaxes(scale_tril, -1, -2)


def _gaussian_nll(
    truth: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    residual = truth - mean
    inverse = np.linalg.inv(covariance)
    quadratic = np.einsum("bsi,bsij,bsj->bs", residual, inverse, residual)
    log_determinant = np.linalg.slogdet(covariance)[1]
    return 0.5 * (2.0 * np.log(2.0 * np.pi) + log_determinant + quadratic)


def _energy_score(
    first: np.ndarray, second: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    return np.mean(np.linalg.norm(first - truth[None, ...], axis=-1), axis=0) - (
        0.5 * np.mean(np.linalg.norm(first - second, axis=-1), axis=0)
    )


def _normalized_delta(candidate: float, comparator: float) -> float:
    return (candidate - comparator) / max(abs(comparator), 1e-6)


def _binary_log_loss(response: np.ndarray, probability: np.ndarray) -> float:
    return float(
        -np.mean(
            response * np.log(probability)
            + (1.0 - response) * np.log(1.0 - probability)
        )
    )


def _context_grid(*, smoke: bool) -> list[dict[str, Any]]:
    scales = (1.0,) if smoke else SCALES
    replicates = range(1) if smoke else range(4)
    return [
        {
            "predictor_location": location,
            "predictor_scale": scale,
            "prevalence_name": prevalence_name,
            "target_prevalence": prevalence,
            "effect_name": effect_name,
            "effect_magnitude": effect,
            "context_replicate": replicate,
        }
        for location in LOCATIONS
        for scale in scales
        for prevalence_name, prevalence in PREVALENCES
        for effect_name, effect in EFFECTS
        for replicate in replicates
    ]


def _seed_range(role: str) -> tuple[int, ...]:
    if role == "smoke_training":
        return tuple(range(SMOKE_STARTS["training"], SMOKE_STARTS["training"] + 27))
    if role == "smoke_evaluation":
        return tuple(range(SMOKE_STARTS["evaluation"], SMOKE_STARTS["evaluation"] + 27))
    start = PRODUCTION_STARTS[role]
    return tuple(range(start, start + PRODUCTION_COUNT))


def _assert_seed_roles(
    first: Iterable[int], second: Iterable[int], *, production: bool
) -> None:
    first = tuple(int(value) for value in first)
    second = tuple(int(value) for value in second)
    expected = (
        (_seed_range("training"), _seed_range("validation"))
        if production
        else (_seed_range("smoke_training"), _seed_range("smoke_evaluation"))
    )
    if (first, second) != expected:
        raise ValueError("M57 seed roles differ from the frozen ledger")
    if not production and any(
        321_000_001 <= value <= 325_000_324 for value in first + second
    ):
        raise ValueError("disposable M57 path may not contain a production seed")


def _validate_prerequisites(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    protocols = {
        "decision_sha256": _validated_hash(DECISION_PATH, M57_DECISION_SHA256),
        "audit_sha256": _validated_hash(AUDIT_PATH, M57_AUDIT_SHA256),
        "preregistration_sha256": _validated_hash(
            PREREGISTRATION_PATH, M57_PREREGISTRATION_SHA256
        ),
    }
    release = load_neural_hmsc_release(args.release_registry)
    bindings = {
        "v0_1": validate_bound_v0_1_release(release),
        "variable_v1": validate_bound_variable_v1(args.variable_registry),
        "m56_negative": validate_bound_m56_negative(args.m56_root),
    }
    return protocols, bindings


def _hdf5_fixture_check(
    engine: FixedProbitStudentTInference,
    dataset: FixedEffectDataset,
    output: Path,
) -> dict[str, Any]:
    posterior = engine.predict_beta_posterior(dataset)
    path = write_student_t_beta_posterior_hdf5(
        posterior,
        output / "student_t_fixture.h5",
        covariate_names=("Intercept", "TMG"),
        species_names=tuple(dataset.Y.columns),
        chains=2,
        draws=16,
        seed=5701,
        metadata={"promotion_evidence": False},
    )
    import h5py

    with h5py.File(path, "r") as handle:
        shape = list(handle["Beta"].shape)
        metadata = json.loads(handle.attrs["pyhmsc_metadata"])
        has_parameters = all(
            name in handle
            for name in (
                "StudentTDegreesOfFreedom",
                "StudentTCovarianceCholesky",
                "StudentTScaleCholesky",
            )
        )
    return {
        "path": str(path),
        "beta_shape": shape,
        "posterior_family": metadata.get("posterior_family"),
        "has_student_t_parameters": has_parameters,
        "passed": shape == [2, 16, 2, 75]
        and metadata.get("posterior_family") == "bivariate_student_t"
        and has_parameters,
    }


def _roundtrip_deltas(
    first: FixedProbitStudentTInference,
    second: FixedProbitStudentTInference,
    dataset: FixedEffectDataset,
) -> dict[str, float]:
    a = first.predict_beta_posterior(dataset)
    b = second.predict_beta_posterior(dataset)
    return {
        "mean": float(np.max(np.abs(np.asarray(a.mean) - np.asarray(b.mean)))),
        "marginal_scale": float(
            np.max(np.abs(np.asarray(a.marginal_scale) - np.asarray(b.marginal_scale)))
        ),
        "covariance_tril": float(
            np.max(
                np.abs(np.asarray(a.covariance_tril) - np.asarray(b.covariance_tril))
            )
        ),
        "student_t_scale_tril": float(
            np.max(
                np.abs(
                    np.asarray(a.student_t_scale_tril)
                    - np.asarray(b.student_t_scale_tril)
                )
            )
        ),
        "degrees_of_freedom": float(
            np.max(
                np.abs(
                    np.asarray(a.degrees_of_freedom) - np.asarray(b.degrees_of_freedom)
                )
            )
        ),
    }


def _require_confirmation(observed: str, expected: str) -> None:
    if observed != expected:
        raise PermissionError(f"exact confirmation required: {expected}")


def _validated_hash(path: Path, expected: str) -> str:
    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(f"frozen M57 document hash changed: {path}")
    return observed


def _empty_output(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"qualification output already exists: {path}")
    path.mkdir(parents=True)
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
