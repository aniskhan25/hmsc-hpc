#!/usr/bin/env python3
"""Run the preregistered Milestone 54 v2.1 gated qualification protocol."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache")
)

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
for path in (ROOT, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import qualify_neural_hmsc_variable_design as v1  # noqa: E402
from pyhmsc.neural import (  # noqa: E402
    GatedVariableDesignNeuralHmscInference,
    variable_design_predictive_auxiliary_data,
)
from pyhmsc.neural.posterior_heads import BetaPosterior  # noqa: E402
from pyhmsc.neural.simulator import FixedEffectDataset  # noqa: E402


PROTOCOL_ID = "neural_hmsc_variable_design_m54_v2_1"
PREREGISTRATION_SHA256 = (
    "900af8719fc73947cd7addf3b7dc9fe2f233eadbbd2bf9f37bac1286fc15e54d"
)
PREREGISTRATION_PATH = (
    ROOT / "docs/neural_hmsc_m54_v2_1_redesign_preregistration_2026-07-22.md"
)
PRODUCTION_COUNT = 243
SMOKE_COUNT = 27
PRODUCTION_STARTS = {
    "coefficient_train": 111_000_001,
    "predictive_context": 112_000_001,
    "predictive_heldout": 113_000_001,
    "calibration": 114_000_001,
    "evaluation": 115_000_001,
}
SMOKE_STARTS = {
    "coefficient_train": 191_000_001,
    "predictive_context": 192_000_001,
    "predictive_heldout": 193_000_001,
    "calibration": 194_000_001,
    "evaluation": 195_000_001,
}
PRODUCTION_MODEL_SEED = 111_900_001
SMOKE_MODEL_SEED = 191_900_001
TRAIN_CONFIRMATION = "GENERATE_M54_V2_1_TRAIN_AUX_CALIBRATION"
EVALUATION_CONFIRMATION = "OPEN_M54_V2_1_RESERVED_EVALUATION"
EXPECTED_MCMC_EVALUATION_SEEDS = (
    115_000_109,
    115_000_148,
    115_000_133,
    115_000_178,
    115_000_211,
    115_000_217,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    v1._add_baseline_arguments(smoke)

    train = subparsers.add_parser("train-calibrate")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--confirmation", required=True)
    v1._add_baseline_arguments(train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--freeze-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--confirmation", required=True)
    v1._add_baseline_arguments(evaluate)

    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(args)
    elif args.command == "train-calibrate":
        result = train_and_freeze(args)
    else:
        result = evaluate_frozen_candidate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output = v1._empty_output(args.output)
    blocks = _seed_blocks(SMOKE_STARTS, SMOKE_COUNT)
    _assert_protocol_seed_roles(blocks, production=False)
    coefficient_train = v1.build_corpus(
        blocks["coefficient_train"], profile="smoke"
    )
    predictive_contexts = v1.build_corpus(
        blocks["predictive_context"], profile="smoke"
    )
    predictive_heldouts = build_predictive_heldouts(
        predictive_contexts, blocks["predictive_heldout"]
    )
    calibration = v1.build_corpus(blocks["calibration"], profile="smoke")
    evaluation = v1.build_corpus(blocks["evaluation"], profile="smoke")
    baseline_hashes = v1._validate_baselines(args)
    preregistration_hash = _validate_preregistration()

    tf.keras.utils.set_random_seed(SMOKE_MODEL_SEED)
    engine = _new_engine()
    engine.training_corpus_version = f"{PROTOCOL_ID}_disposable_smoke"
    auxiliary = variable_design_predictive_auxiliary_data(
        predictive_contexts, predictive_heldouts
    )
    history = engine.fit(
        coefficient_train,
        predictive_auxiliary=auxiliary,
        epochs=1,
        batch_size=9,
        learning_rate=0.001,
        mse_weight=0.25,
        predictive_weight=1.0,
        seed=SMOKE_MODEL_SEED,
    )
    calibration_result = engine.fit_calibration(
        calibration,
        provenance=_calibration_provenance(
            blocks["calibration"], corpus_id="m54_v2_1_disposable_smoke_calibration"
        ),
    )
    checkpoint = engine.save(output / "checkpoint")
    loaded = GatedVariableDesignNeuralHmscInference.load(checkpoint)
    evaluation_result = evaluate_v2_corpus(
        loaded,
        evaluation,
        draws=32,
        draw_seed=195_900_001,
        production=False,
    )
    smoke_checks = {
        "seed_blocks_disjoint": True,
        "production_seeds_remain_unopened": True,
        "marginal_balance": all(
            v1._marginals_balanced(corpus)
            for corpus in (
                coefficient_train,
                predictive_contexts,
                calibration,
                evaluation,
            )
        ),
        "predictive_heldouts_independent": _heldouts_independent(
            predictive_contexts, predictive_heldouts
        ),
        "finite_training_history": v1._all_finite(history),
        "calibration_packaged": (
            loaded.calibration is not None
            and loaded.calibration.method == "split_conformal_scalar_beta_scale"
            and loaded.calibration.n_coefficients
            == sum(dataset.truth_beta.size for dataset in calibration)
        ),
        "checkpoint_roundtrip": (
            evaluation_result["roundtrip_max_delta"] <= 1e-6
        ),
        "support_gate_bounded": evaluation_result["gates"][
            "support_gate_bounded"
        ],
        "finite_evaluation": v1._all_finite(evaluation_result["summary"]),
        "baseline_hashes": baseline_hashes["all_valid"],
        "preregistration_hash": preregistration_hash == PREREGISTRATION_SHA256,
    }
    payload = {
        "schema_version": 1,
        "kind": "neural_hmsc_variable_design_m54_v2_1_smoke",
        "protocol_id": PROTOCOL_ID,
        "decision": "smoke_passed" if all(smoke_checks.values()) else "smoke_failed",
        "smoke_checks": smoke_checks,
        "production_seed_opened": False,
        "promotion_evidence": False,
        "seeds": blocks,
        "model_seed": SMOKE_MODEL_SEED,
        "training": history,
        "calibration": calibration_result.to_metadata(),
        "checkpoint": _checkpoint_record(checkpoint),
        "evaluation": evaluation_result,
        "baseline_hashes": baseline_hashes,
        "preregistration_sha256": preregistration_hash,
    }
    _write_json(output / "m54_v2_1_smoke.json", payload)
    return payload


def train_and_freeze(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirmation != TRAIN_CONFIRMATION:
        raise ValueError("exact v2.1 train/aux/calibration confirmation is required")
    output = v1._empty_output(args.output)
    blocks = _seed_blocks(PRODUCTION_STARTS, PRODUCTION_COUNT)
    _assert_protocol_seed_roles(blocks, production=True)
    coefficient_train = v1.build_corpus(
        blocks["coefficient_train"], profile="production"
    )
    predictive_contexts = v1.build_corpus(
        blocks["predictive_context"], profile="production"
    )
    predictive_heldouts = build_predictive_heldouts(
        predictive_contexts, blocks["predictive_heldout"]
    )
    calibration = v1.build_corpus(blocks["calibration"], profile="production")
    baseline_hashes = v1._validate_baselines(args)
    preregistration_hash = _validate_preregistration()
    auxiliary = variable_design_predictive_auxiliary_data(
        predictive_contexts, predictive_heldouts
    )

    tf.keras.utils.set_random_seed(PRODUCTION_MODEL_SEED)
    engine = _new_engine()
    engine.training_corpus_version = PROTOCOL_ID
    started = time.perf_counter()
    history = engine.fit(
        coefficient_train,
        predictive_auxiliary=auxiliary,
        epochs=40,
        batch_size=9,
        learning_rate=0.001,
        mse_weight=0.25,
        predictive_weight=1.0,
        seed=PRODUCTION_MODEL_SEED,
    )
    training_seconds = time.perf_counter() - started
    calibration_result = engine.fit_calibration(
        calibration,
        provenance=_calibration_provenance(
            blocks["calibration"], corpus_id="m54_v2_1_candidate_calibration"
        ),
    )
    checkpoint = engine.save(output / "checkpoint")
    loaded = GatedVariableDesignNeuralHmscInference.load(checkpoint)
    if loaded.calibration is None:
        raise ValueError("v2.1 production calibration did not roundtrip")
    payload = {
        "schema_version": 1,
        "kind": "neural_hmsc_variable_design_m54_v2_1_freeze",
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_reserved_evaluation",
        "production_seed_opened": True,
        "reserved_evaluation_opened": False,
        "settings": _production_settings(),
        "seeds": {
            "coefficient_train": blocks["coefficient_train"],
            "predictive_context": blocks["predictive_context"],
            "predictive_heldout": blocks["predictive_heldout"],
            "calibration": blocks["calibration"],
            "reserved_evaluation_start": PRODUCTION_STARTS["evaluation"],
            "reserved_evaluation_count": PRODUCTION_COUNT,
            "model": PRODUCTION_MODEL_SEED,
        },
        "corpus_balance": {
            "coefficient_train": v1._corpus_balance(coefficient_train),
            "predictive_context": v1._corpus_balance(predictive_contexts),
            "calibration": v1._corpus_balance(calibration),
        },
        "predictive_heldout_independence": _heldouts_independent(
            predictive_contexts, predictive_heldouts
        ),
        "training": {**history, "seconds": training_seconds},
        "calibration": calibration_result.to_metadata(),
        "checkpoint": _checkpoint_record(checkpoint),
        "baseline_hashes": baseline_hashes,
        "preregistration_sha256": preregistration_hash,
    }
    _write_json(output / "m54_v2_1_freeze.json", payload)
    return payload


def evaluate_frozen_candidate(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirmation != EVALUATION_CONFIRMATION:
        raise ValueError("exact v2.1 reserved-evaluation confirmation is required")
    freeze = validate_freeze(args.freeze_root)
    output = v1._empty_output(args.output)
    evaluation_seeds = v1._seed_block(
        PRODUCTION_STARTS["evaluation"], PRODUCTION_COUNT
    )
    evaluation = v1.build_corpus(evaluation_seeds, profile="production")
    baseline_hashes = v1._validate_baselines(args)
    engine = GatedVariableDesignNeuralHmscInference.load(
        Path(args.freeze_root) / "checkpoint"
    )
    result = evaluate_v2_corpus(
        engine,
        evaluation,
        draws=256,
        draw_seed=115_900_001,
        production=True,
    )
    mcmc = v1.run_mcmc_comparison(engine, evaluation, output=output / "mcmc")
    observed_mcmc_seeds = tuple(int(row["seed"]) for row in mcmc["rows"])
    if observed_mcmc_seeds != EXPECTED_MCMC_EVALUATION_SEEDS:
        raise ValueError("v2.1 MCMC evaluation seeds differ from preregistration")
    gates = dict(result["gates"])
    gates["proper_scores_vs_mcmc"] = (
        mcmc["neural_to_mcmc_brier_ratio"] <= 1.10
        and mcmc["neural_to_mcmc_log_loss_ratio"] <= 1.10
    )
    gates["baseline_hashes"] = baseline_hashes["all_valid"]
    gates["preregistration_hash"] = (
        _validate_preregistration() == PREREGISTRATION_SHA256
    )
    payload = {
        "schema_version": 1,
        "kind": "neural_hmsc_variable_design_m54_v2_1_evaluation",
        "protocol_id": PROTOCOL_ID,
        "decision": (
            "variable_design_v2_1_simulated_passed_realdata_pending"
            if all(gates.values())
            else "variable_design_v2_1_terminal_failure"
        ),
        "all_gates_passed": all(gates.values()),
        "reserved_evaluation_opened": True,
        "freeze_sha256": _sha256(Path(args.freeze_root) / "m54_v2_1_freeze.json"),
        "freeze": freeze,
        "evaluation": result,
        "mcmc": mcmc,
        "gates": gates,
        "baseline_hashes": baseline_hashes,
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }
    _write_json(output / "m54_v2_1_evaluation.json", payload)
    return payload


def build_predictive_heldouts(
    contexts: Sequence[FixedEffectDataset], seeds: Sequence[int]
) -> list[FixedEffectDataset]:
    if len(contexts) != len(seeds):
        raise ValueError("predictive context/heldout seed counts differ")
    return [
        simulate_predictive_heldout(context, seed=int(seed))
        for context, seed in zip(contexts, seeds)
    ]


def simulate_predictive_heldout(
    context: FixedEffectDataset, *, seed: int
) -> FixedEffectDataset:
    """Generate independent X/Y conditional on one context's frozen Beta."""
    rng = np.random.default_rng(seed)
    n_sites = len(context.X)
    n_species = context.Y.shape[1]
    n_covariates = context.truth_beta.shape[0]
    n_predictors = n_covariates - 1
    target_condition = float(context.metadata["target_condition"])
    raw = rng.normal(size=(n_sites, n_predictors))
    raw -= raw.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(raw)
    scales = np.geomspace(1.0 / target_condition, 1.0, n_predictors)
    predictors = q[:, :n_predictors] * np.sqrt(float(n_sites)) * scales[None, :]
    predictor_names = [str(name) for name in context.truth_beta.index[1:]]
    X = pd.DataFrame(predictors, columns=predictor_names)
    design = np.column_stack([np.ones(n_sites), predictors])
    beta = context.truth_beta.to_numpy(dtype=float)
    linear = design @ beta
    probability = ndtr(linear)
    Y = pd.DataFrame(
        rng.binomial(1, probability),
        columns=[str(name) for name in context.Y.columns],
    )
    return FixedEffectDataset(
        Y=Y,
        X=X,
        truth_beta=context.truth_beta.copy(),
        linear_predictor=pd.DataFrame(linear, columns=Y.columns),
        metadata={
            **context.metadata,
            "seed": int(seed),
            "paired_context_seed": int(context.metadata["seed"]),
            "predictive_heldout": True,
            "actual_condition": float(np.linalg.cond(design)),
        },
    )


def evaluate_v2_corpus(
    engine: GatedVariableDesignNeuralHmscInference,
    datasets: Sequence[FixedEffectDataset],
    *,
    draws: int,
    draw_seed: int,
    production: bool,
) -> dict[str, Any]:
    result = v1.evaluate_corpus(
        engine,
        datasets,
        draws=draws,
        draw_seed=draw_seed,
        production=production,
    )
    gate_rows = []
    gate_values = []
    for dataset, dataset_row in zip(datasets, result["dataset_rows"]):
        posterior = engine.predict_beta_posterior(dataset)
        anchor = v1._anchor_posterior(dataset)
        gate = engine.predict_support_gate(dataset).numpy()[0]
        n_covariates, n_species = dataset.truth_beta.shape
        active_gate = gate[:n_covariates, :n_species]
        active_movement = (
            posterior.mean.numpy()[0, :n_covariates, :n_species]
            - anchor.mean.numpy()[0, :n_covariates, :n_species]
        )
        dataset_row["support_gate_median"] = float(np.median(active_gate))
        dataset_row["support_gate_mean"] = float(np.mean(active_gate))
        dataset_row["mean_absolute_movement"] = float(
            np.mean(np.abs(active_movement))
        )
        gate_values.extend(active_gate.ravel().tolist())
        for covariate in range(n_covariates):
            for species in range(n_species):
                gate_rows.append(
                    {
                        "support_gate": float(active_gate[covariate, species]),
                        "absolute_movement": float(
                            abs(active_movement[covariate, species])
                        ),
                        "n_sites": len(dataset.X),
                        "n_covariates": n_covariates,
                        "prevalence": dataset.metadata["prevalence_stratum"],
                        "effect": dataset.metadata["effect_stratum"],
                        "coefficient_role": (
                            "intercept" if covariate == 0 else "non_intercept"
                        ),
                        "design_condition": dataset.metadata["strata"][
                            "design_condition"
                        ],
                    }
                )

    site_scores = _score_ratios(result["dataset_rows"], "n_sites")
    covariate_scores = _score_ratios(result["dataset_rows"], "n_covariates")
    low_support = [
        row["support_gate"]
        for row in gate_rows
        if row["n_sites"] == 12 and row["n_covariates"] == 8
    ]
    high_support = [
        row["support_gate"]
        for row in gate_rows
        if row["n_sites"] == 128 and row["n_covariates"] == 2
    ]
    diagnostics = {
        "support_gate_min": float(np.min(gate_values)),
        "support_gate_max": float(np.max(gate_values)),
        "support_gate_mean": float(np.mean(gate_values)),
        "low_support_median": float(np.median(low_support)),
        "high_support_median": float(np.median(high_support)),
        "by_site": _gate_summary(gate_rows, "n_sites"),
        "by_covariate_count": _gate_summary(gate_rows, "n_covariates"),
        "by_prevalence": _gate_summary(gate_rows, "prevalence"),
        "by_effect": _gate_summary(gate_rows, "effect"),
        "by_coefficient_role": _gate_summary(gate_rows, "coefficient_role"),
        "by_design_condition": _gate_summary(gate_rows, "design_condition"),
    }
    result["proper_scores_by_site"] = site_scores
    result["proper_scores_by_covariate_count"] = covariate_scores
    result["support_gate_diagnostics"] = diagnostics
    result["gates"].update(
        {
            "support_gate_bounded": (
                diagnostics["support_gate_min"] >= 0.0
                and diagnostics["support_gate_max"] <= 1.0
            ),
            "proper_scores_by_site": _score_ratio_gate(site_scores),
            "proper_scores_by_covariate_count": _score_ratio_gate(
                covariate_scores
            ),
            "genuine_beta_rmse_gain": (
                result["summary"]["rmse"]
                <= 0.98 * result["summary"]["anchor_rmse"]
            ),
            "support_gate_ordering": (
                diagnostics["high_support_median"]
                >= diagnostics["low_support_median"]
            ),
        }
    )
    return result


def validate_freeze(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    payload = json.loads((root / "m54_v2_1_freeze.json").read_text(encoding="utf-8"))
    if payload.get("kind") != "neural_hmsc_variable_design_m54_v2_1_freeze":
        raise ValueError("unsupported Milestone 54 v2.1 freeze")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Milestone 54 v2.1 freeze protocol differs")
    if payload.get("status") != "frozen_before_reserved_evaluation":
        raise ValueError("Milestone 54 v2.1 candidate is not frozen")
    if payload.get("reserved_evaluation_opened") is not False:
        raise ValueError("Milestone 54 v2.1 evaluation is already recorded open")
    if payload.get("settings") != _production_settings():
        raise ValueError("Milestone 54 v2.1 settings differ")
    expected = _seed_blocks(PRODUCTION_STARTS, PRODUCTION_COUNT)
    seeds = payload["seeds"]
    for role in (
        "coefficient_train",
        "predictive_context",
        "predictive_heldout",
        "calibration",
    ):
        if seeds[role] != expected[role]:
            raise ValueError(f"Milestone 54 v2.1 {role} seeds differ")
    if (
        seeds["reserved_evaluation_start"] != PRODUCTION_STARTS["evaluation"]
        or seeds["reserved_evaluation_count"] != PRODUCTION_COUNT
        or seeds["model"] != PRODUCTION_MODEL_SEED
    ):
        raise ValueError("Milestone 54 v2.1 reserved/model seeds differ")
    if payload.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("Milestone 54 v2.1 preregistration hash differs")
    checkpoint = root / "checkpoint"
    record = payload["checkpoint"]
    for name, filename in (
        ("manifest_sha256", "neural_checkpoint.json"),
        ("weights_sha256", "weights.weights.h5"),
        ("calibration_sha256", "variable_design_calibration.json"),
    ):
        if record[name] != _sha256(checkpoint / filename):
            raise ValueError(f"Milestone 54 v2.1 checkpoint {name} differs")
    GatedVariableDesignNeuralHmscInference.load(checkpoint)
    return payload


def _new_engine() -> GatedVariableDesignNeuralHmscInference:
    return GatedVariableDesignNeuralHmscInference.for_fixed_effects(
        min_sites=12,
        max_sites=128,
        min_species=2,
        max_species=100,
        min_covariates=2,
        max_covariates=8,
    )


def _seed_blocks(starts: dict[str, int], count: int) -> dict[str, list[int]]:
    return {role: v1._seed_block(start, count) for role, start in starts.items()}


def _assert_protocol_seed_roles(
    blocks: dict[str, list[int]], *, production: bool
) -> None:
    v1._assert_disjoint(*blocks.values())
    flattened = [seed for block in blocks.values() for seed in block]
    if production:
        if not all(_is_production_seed(seed) for seed in flattened):
            raise ValueError("Milestone 54 v2.1 production seed role differs")
    elif any(_is_production_seed(seed) for seed in flattened):
        raise ValueError("Milestone 54 v2.1 smoke intersects production seeds")


def _is_production_seed(seed: int) -> bool:
    return any(
        start <= int(seed) < start + PRODUCTION_COUNT
        for start in PRODUCTION_STARTS.values()
    )


def _heldouts_independent(
    contexts: Sequence[FixedEffectDataset], heldouts: Sequence[FixedEffectDataset]
) -> bool:
    return len(contexts) == len(heldouts) and all(
        int(context.metadata["seed"]) != int(heldout.metadata["seed"])
        and int(heldout.metadata["paired_context_seed"])
        == int(context.metadata["seed"])
        and context.truth_beta.equals(heldout.truth_beta)
        and not context.X.equals(heldout.X)
        for context, heldout in zip(contexts, heldouts)
    )


def _score_ratios(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, dict[str, float]]:
    result = {}
    for value in sorted({row[field] for row in rows}, key=str):
        selected = [row for row in rows if row[field] == value]
        neural_brier = float(np.mean([row["neural_brier"] for row in selected]))
        anchor_brier = float(np.mean([row["anchor_brier"] for row in selected]))
        neural_log_loss = float(
            np.mean([row["neural_log_loss"] for row in selected])
        )
        anchor_log_loss = float(
            np.mean([row["anchor_log_loss"] for row in selected])
        )
        result[str(value)] = {
            "n_datasets": len(selected),
            "neural_brier": neural_brier,
            "anchor_brier": anchor_brier,
            "brier_ratio": neural_brier / anchor_brier,
            "neural_log_loss": neural_log_loss,
            "anchor_log_loss": anchor_log_loss,
            "log_loss_ratio": neural_log_loss / anchor_log_loss,
        }
    return result


def _score_ratio_gate(values: dict[str, dict[str, float]]) -> bool:
    return all(
        row["brier_ratio"] <= 1.02 and row["log_loss_ratio"] <= 1.02
        for row in values.values()
    )


def _gate_summary(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, dict[str, float]]:
    result = {}
    for value in sorted({row[field] for row in rows}, key=str):
        selected = [row for row in rows if row[field] == value]
        gate = np.asarray([row["support_gate"] for row in selected], dtype=float)
        movement = np.asarray(
            [row["absolute_movement"] for row in selected], dtype=float
        )
        result[str(value)] = {
            "n_coefficients": len(selected),
            "support_gate_mean": float(np.mean(gate)),
            "support_gate_median": float(np.median(gate)),
            "mean_absolute_movement": float(np.mean(movement)),
        }
    return result


def _production_settings() -> dict[str, Any]:
    return {
        "corpus_count_per_role": PRODUCTION_COUNT,
        "sites": list(v1.SITE_LEVELS),
        "species": list(v1.SPECIES_LEVELS),
        "covariates": list(v1.COVARIATE_LEVELS),
        "target_conditions": list(v1.CONDITION_LEVELS),
        "epochs": 40,
        "batch_size": 9,
        "learning_rate": 0.001,
        "coefficient_mse_weight": 0.25,
        "predictive_weight": 1.0,
        "predictive_log_loss_weight": 0.5,
        "predictive_brier_weight": 0.5,
        "sbc_draws": 256,
        "mcmc_datasets": 6,
        "mcmc_samples": 200,
        "mcmc_transient": 100,
    }


def _calibration_provenance(
    seeds: Sequence[int], *, corpus_id: str
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "independent_from_training": True,
        "target_ecological_response_used": False,
        "seeds": [int(seed) for seed in seeds],
        "corpus_id": corpus_id,
    }


def _checkpoint_record(checkpoint: Path) -> dict[str, str]:
    return {
        "path": str(checkpoint),
        "manifest_sha256": _sha256(checkpoint / "neural_checkpoint.json"),
        "weights_sha256": _sha256(checkpoint / "weights.weights.h5"),
        "calibration_sha256": _sha256(
            checkpoint / "variable_design_calibration.json"
        ),
    }


def _validate_preregistration() -> str:
    observed = _sha256(PREREGISTRATION_PATH)
    if observed != PREREGISTRATION_SHA256:
        raise ValueError("Milestone 54 v2.1 preregistration hash differs")
    return observed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
