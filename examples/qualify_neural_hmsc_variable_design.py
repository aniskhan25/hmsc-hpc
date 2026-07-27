#!/usr/bin/env python3
"""Run the preregistered Milestone 54 variable-design qualification protocol."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc import HmscModel, load_neural_hmsc_release  # noqa: E402
from pyhmsc.neural import (  # noqa: E402
    VariableDesignNeuralHmscInference,
    validate_variable_shape_baseline,
)
from pyhmsc.neural.models import probit_irls_laplace_anchor  # noqa: E402
from pyhmsc.neural.posterior_heads import BetaPosterior  # noqa: E402
from pyhmsc.neural.simulator import FixedEffectDataset  # noqa: E402


PROTOCOL_ID = "neural_hmsc_variable_design_m54_v1_1"
PRODUCTION_COUNT = 243
SMOKE_COUNT = 27
SITE_LEVELS = (12, 40, 128)
SPECIES_LEVELS = (2, 20, 100)
COVARIATE_LEVELS = (2, 5, 8)
CONDITION_LEVELS = (2.0, 10.0, 50.0)
PREVALENCE_LEVELS = ("rare", "balanced", "common")
EFFECT_LEVELS = ("weak", "moderate", "strong")
INTERCEPT_LOCATION = {"rare": -1.5, "balanced": -0.2, "common": 0.8}
EFFECT_SCALE = {"weak": 0.25, "moderate": 0.6, "strong": 1.0}
FIXED_BASELINE_HASH = "affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8"
VARIABLE_BASELINE_HASH = (
    "badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9"
)
SMOKE_STARTS = {
    "train": 91_000_001,
    "calibration": 92_000_001,
    "evaluation": 93_000_001,
}
ROLE_SPECS = {
    "candidate": {
        "train": 101_000_001,
        "calibration": 102_000_001,
        "evaluation": 103_000_001,
        "model": 101_900_001,
    },
    "sensitivity_a": {
        "train": 104_000_001,
        "calibration": 105_000_001,
        "evaluation": 106_000_001,
        "model": 104_900_001,
    },
    "sensitivity_b": {
        "train": 107_000_001,
        "calibration": 108_000_001,
        "evaluation": 109_000_001,
        "model": 107_900_001,
    },
}
TRAIN_CONFIRMATIONS = {
    role: f"GENERATE_M54_{role.upper()}_TRAIN_CALIBRATION" for role in ROLE_SPECS
}
EVALUATION_CONFIRMATIONS = {
    role: f"OPEN_M54_{role.upper()}_EVALUATION" for role in ROLE_SPECS
}
MCMC_CONTEXTS = (
    (40, 20, 2, 2.0),
    (40, 100, 5, 10.0),
    (40, 20, 8, 50.0),
    (128, 2, 5, 50.0),
    (128, 20, 8, 10.0),
    (128, 100, 2, 2.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    _add_baseline_arguments(smoke)

    train = subparsers.add_parser("train-calibrate")
    train.add_argument("--role", choices=tuple(ROLE_SPECS), required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--confirmation", required=True)
    _add_baseline_arguments(train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--role", choices=tuple(ROLE_SPECS), required=True)
    evaluate.add_argument("--freeze-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--confirmation", required=True)
    _add_baseline_arguments(evaluate)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--candidate", type=Path, required=True)
    aggregate.add_argument("--sensitivity-a", type=Path, required=True)
    aggregate.add_argument("--sensitivity-b", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(args)
    elif args.command == "train-calibrate":
        result = train_and_freeze(args)
    elif args.command == "evaluate":
        result = evaluate_frozen_role(args)
    else:
        result = aggregate_roles(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output = _empty_output(args.output)
    train_seeds = _seed_block(SMOKE_STARTS["train"], SMOKE_COUNT)
    calibration_seeds = _seed_block(SMOKE_STARTS["calibration"], SMOKE_COUNT)
    evaluation_seeds = _seed_block(SMOKE_STARTS["evaluation"], SMOKE_COUNT)
    _assert_disjoint(train_seeds, calibration_seeds, evaluation_seeds)
    if any(
        _is_production_seed(seed)
        for seed in train_seeds + calibration_seeds + evaluation_seeds
    ):
        raise ValueError("disposable smoke intersects a production seed block")

    train = build_corpus(train_seeds, profile="smoke")
    calibration = build_corpus(calibration_seeds, profile="smoke")
    evaluation = build_corpus(evaluation_seeds, profile="smoke")
    baseline_hashes = _validate_baselines(args)
    tf.keras.utils.set_random_seed(91_900_001)
    engine = _new_engine()
    engine.training_corpus_version = f"{PROTOCOL_ID}_disposable_smoke"
    history = engine.fit(
        train,
        epochs=1,
        batch_size=9,
        seed=91_900_001,
    )
    calibration_result = engine.fit_calibration(
        calibration,
        provenance=_calibration_provenance(
            calibration_seeds, corpus_id="m54_disposable_smoke_calibration"
        ),
    )
    checkpoint = engine.save(output / "checkpoint")
    loaded = VariableDesignNeuralHmscInference.load(checkpoint)
    evaluation_result = evaluate_corpus(
        loaded,
        evaluation,
        draws=32,
        draw_seed=93_900_001,
        production=False,
    )
    smoke_checks = {
        "seed_blocks_disjoint": True,
        "production_seeds_remain_unopened": True,
        "marginal_balance": all(
            _marginals_balanced(corpus) for corpus in (train, calibration, evaluation)
        ),
        "finite_training_history": all(
            np.isfinite(value) for values in history.values() for value in values
        ),
        "calibration_packaged": (
            loaded.calibration is not None
            and loaded.calibration.method == "split_conformal_scalar_beta_scale"
            and loaded.calibration.n_coefficients
            == sum(dataset.truth_beta.size for dataset in calibration)
        ),
        "checkpoint_roundtrip": evaluation_result["roundtrip_max_delta"] <= 1e-6,
        "finite_evaluation": _all_finite(evaluation_result["summary"]),
        "baseline_hashes": baseline_hashes["all_valid"],
    }
    payload = {
        "schema_version": 1,
        "kind": "neural_hmsc_variable_design_m54_smoke",
        "protocol_id": PROTOCOL_ID,
        "decision": "smoke_passed" if all(smoke_checks.values()) else "smoke_failed",
        "smoke_checks": smoke_checks,
        "production_seed_opened": False,
        "promotion_evidence": False,
        "seeds": {
            "train": train_seeds,
            "calibration": calibration_seeds,
            "evaluation": evaluation_seeds,
        },
        "training": history,
        "calibration": calibration_result.to_metadata(),
        "checkpoint": _checkpoint_record(checkpoint),
        "evaluation": evaluation_result,
        "baseline_hashes": baseline_hashes,
    }
    _write_json(output / "m54_smoke.json", payload)
    return payload


def train_and_freeze(args: argparse.Namespace) -> dict[str, Any]:
    role = str(args.role)
    if args.confirmation != TRAIN_CONFIRMATIONS[role]:
        raise ValueError(
            "exact role-specific train/calibration confirmation is required"
        )
    output = _empty_output(args.output)
    spec = ROLE_SPECS[role]
    train_seeds = _seed_block(spec["train"], PRODUCTION_COUNT)
    calibration_seeds = _seed_block(spec["calibration"], PRODUCTION_COUNT)
    evaluation_seeds = _seed_block(spec["evaluation"], PRODUCTION_COUNT)
    _assert_disjoint(train_seeds, calibration_seeds, evaluation_seeds)
    train = build_corpus(train_seeds, profile="production")
    calibration = build_corpus(calibration_seeds, profile="production")
    baseline_hashes = _validate_baselines(args)
    tf.keras.utils.set_random_seed(spec["model"])
    engine = _new_engine()
    engine.training_corpus_version = PROTOCOL_ID
    started = time.perf_counter()
    history = engine.fit(
        train,
        epochs=40,
        batch_size=9,
        learning_rate=1e-3,
        mse_weight=0.25,
        seed=spec["model"],
    )
    training_seconds = time.perf_counter() - started
    calibration_result = engine.fit_calibration(
        calibration,
        provenance=_calibration_provenance(
            calibration_seeds, corpus_id=f"m54_{role}_calibration"
        ),
    )
    checkpoint = engine.save(output / "checkpoint")
    loaded = VariableDesignNeuralHmscInference.load(checkpoint)
    if loaded.calibration is None:
        raise ValueError("production checkpoint calibration did not roundtrip")
    payload = {
        "schema_version": 1,
        "kind": "neural_hmsc_variable_design_m54_train_calibration_freeze",
        "protocol_id": PROTOCOL_ID,
        "role": role,
        "status": "frozen_before_reserved_evaluation",
        "production_seed_opened": True,
        "reserved_evaluation_opened": False,
        "settings": _production_settings(),
        "seeds": {
            "train": train_seeds,
            "calibration": calibration_seeds,
            "reserved_evaluation_start": spec["evaluation"],
            "reserved_evaluation_count": PRODUCTION_COUNT,
            "model": spec["model"],
        },
        "corpus_balance": {
            "train": _corpus_balance(train),
            "calibration": _corpus_balance(calibration),
        },
        "training": {**history, "seconds": training_seconds},
        "calibration": calibration_result.to_metadata(),
        "checkpoint": _checkpoint_record(checkpoint),
        "baseline_hashes": baseline_hashes,
    }
    _write_json(output / "m54_train_calibration_freeze.json", payload)
    return payload


def evaluate_frozen_role(args: argparse.Namespace) -> dict[str, Any]:
    role = str(args.role)
    if args.confirmation != EVALUATION_CONFIRMATIONS[role]:
        raise ValueError(
            "exact role-specific reserved-evaluation confirmation is required"
        )
    freeze = validate_freeze(args.freeze_root, role=role)
    baseline_hashes = _validate_baselines(args)
    output = _empty_output(args.output)
    evaluation_seeds = _seed_block(ROLE_SPECS[role]["evaluation"], PRODUCTION_COUNT)
    evaluation = build_corpus(evaluation_seeds, profile="production")
    engine = VariableDesignNeuralHmscInference.load(
        Path(args.freeze_root) / "checkpoint"
    )
    result = evaluate_corpus(
        engine,
        evaluation,
        draws=256,
        draw_seed=ROLE_SPECS[role]["evaluation"] + 900_000,
        production=True,
    )
    mcmc = run_mcmc_comparison(engine, evaluation, output=output / "mcmc")
    gates = dict(result["gates"])
    gates["proper_scores_vs_mcmc"] = (
        mcmc["neural_to_mcmc_brier_ratio"] <= 1.10
        and mcmc["neural_to_mcmc_log_loss_ratio"] <= 1.10
    )
    gates["baseline_hashes"] = baseline_hashes["all_valid"]
    payload = {
        "schema_version": 1,
        "kind": "neural_hmsc_variable_design_m54_role_evaluation",
        "protocol_id": PROTOCOL_ID,
        "role": role,
        "decision": (
            "variable_design_role_passed"
            if all(gates.values())
            else "variable_design_role_failed"
        ),
        "all_gates_passed": all(gates.values()),
        "reserved_evaluation_opened": True,
        "candidate_selected_using_sensitivity_outcomes": False,
        "freeze_sha256": _sha256(
            Path(args.freeze_root) / "m54_train_calibration_freeze.json"
        ),
        "freeze": freeze,
        "evaluation": result,
        "mcmc": mcmc,
        "gates": gates,
        "baseline_hashes": baseline_hashes,
    }
    _write_json(output / "m54_role_evaluation.json", payload)
    return payload


def aggregate_roles(args: argparse.Namespace) -> dict[str, Any]:
    output = _empty_output(args.output)
    reports = {
        "candidate": _read_role_report(args.candidate, "candidate"),
        "sensitivity_a": _read_role_report(args.sensitivity_a, "sensitivity_a"),
        "sensitivity_b": _read_role_report(args.sensitivity_b, "sensitivity_b"),
    }
    passed = all(report["all_gates_passed"] for report in reports.values())
    payload = {
        "schema_version": 1,
        "kind": "neural_hmsc_variable_design_m54_multirole_qualification",
        "protocol_id": PROTOCOL_ID,
        "decision": (
            "simulated_qualification_passed_realdata_pending"
            if passed
            else "variable_design_probit_v2_not_qualified"
        ),
        "all_roles_passed": passed,
        "candidate_selected_using_sensitivity_outcomes": False,
        "realdata_opened": False,
        "roles": {
            role: {
                "report_sha256": _sha256(_report_path(path)),
                "decision": reports[role]["decision"],
            }
            for role, path in {
                "candidate": args.candidate,
                "sensitivity_a": args.sensitivity_a,
                "sensitivity_b": args.sensitivity_b,
            }.items()
        },
    }
    _write_json(output / "m54_multirole_qualification.json", payload)
    return payload


def build_corpus(seeds: Sequence[int], *, profile: str) -> list[FixedEffectDataset]:
    schedule = _corpus_schedule(profile)
    if len(seeds) != len(schedule):
        raise ValueError(f"{profile} corpus must contain exactly {len(schedule)} seeds")
    datasets = []
    for seed, row in zip(seeds, schedule):
        site, species, covariate, condition, prevalence_index, effect_index = row
        datasets.append(
            simulate_stratified_dataset(
                seed=int(seed),
                n_sites=int(site[1]),
                n_species=int(species[1]),
                n_covariates=int(covariate[1]),
                target_condition=float(condition[1]),
                prevalence=PREVALENCE_LEVELS[prevalence_index],
                effect=EFFECT_LEVELS[effect_index],
                strata={
                    "site": f"site_{site[0]}",
                    "species": f"species_{species[0]}",
                    "covariate": f"covariate_{covariate[0]}",
                    "design_condition": f"condition_{condition[0]}",
                },
            )
        )
    return datasets


def _corpus_schedule(profile: str) -> list[tuple[Any, ...]]:
    if profile == "production":
        cells = list(
            itertools.product(
                enumerate(SITE_LEVELS),
                enumerate(SPECIES_LEVELS),
                enumerate(COVARIATE_LEVELS),
                enumerate(CONDITION_LEVELS),
            )
        )
        schedule = []
        for cell_index, cell in enumerate(cells):
            for replicate in range(3):
                prevalence_index = replicate
                effect_index = (replicate + cell_index) % 3
                schedule.append((*cell, prevalence_index, effect_index))
    elif profile == "smoke":
        schedule = []
        for index in range(SMOKE_COUNT):
            site_index = index % 3
            species_index = (index // 3) % 3
            covariate_index = (index // 9) % 3
            condition_index = (site_index + species_index + covariate_index) % 3
            prevalence_index = (site_index + 2 * species_index + covariate_index) % 3
            effect_index = (2 * site_index + species_index + covariate_index) % 3
            schedule.append(
                (
                    (site_index, SITE_LEVELS[site_index]),
                    (species_index, SPECIES_LEVELS[species_index]),
                    (covariate_index, COVARIATE_LEVELS[covariate_index]),
                    (condition_index, CONDITION_LEVELS[condition_index]),
                    prevalence_index,
                    effect_index,
                )
            )
    else:
        raise ValueError(f"unsupported corpus profile {profile!r}")
    return schedule


def simulate_stratified_dataset(
    *,
    seed: int,
    n_sites: int,
    n_species: int,
    n_covariates: int,
    target_condition: float,
    prevalence: str,
    effect: str,
    strata: dict[str, str],
) -> FixedEffectDataset:
    rng = np.random.default_rng(seed)
    n_predictors = n_covariates - 1
    raw = rng.normal(size=(n_sites, n_predictors))
    raw -= raw.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(raw)
    scales = np.geomspace(1.0 / target_condition, 1.0, n_predictors)
    predictors = q[:, :n_predictors] * np.sqrt(float(n_sites)) * scales[None, :]
    predictor_names = [f"x{index}" for index in range(1, n_covariates)]
    X = pd.DataFrame(predictors, columns=predictor_names)
    design = np.column_stack([np.ones(n_sites), predictors])
    beta = np.zeros((n_covariates, n_species), dtype=float)
    beta[0] = INTERCEPT_LOCATION[prevalence] + rng.normal(scale=0.2, size=n_species)
    beta[1:] = rng.normal(
        scale=EFFECT_SCALE[effect] / np.sqrt(max(n_predictors, 1)),
        size=(n_predictors, n_species),
    )
    linear = design @ beta
    probability = ndtr(linear)
    species_names = [f"sp{index + 1}" for index in range(n_species)]
    Y = pd.DataFrame(rng.binomial(1, probability), columns=species_names)
    covariate_names = ["Intercept", *predictor_names]
    return FixedEffectDataset(
        Y=Y,
        X=X,
        truth_beta=pd.DataFrame(beta, index=covariate_names, columns=species_names),
        linear_predictor=pd.DataFrame(linear, columns=species_names),
        metadata={
            "distribution": "probit",
            "formula": "~ " + " + ".join(predictor_names),
            "seed": seed,
            "n_sites": n_sites,
            "n_species": n_species,
            "n_covariates": n_covariates,
            "target_condition": target_condition,
            "actual_condition": float(np.linalg.cond(design)),
            "prevalence_stratum": prevalence,
            "effect_stratum": effect,
            "strata": dict(strata),
        },
    )


def evaluate_corpus(
    engine: VariableDesignNeuralHmscInference,
    datasets: Sequence[FixedEffectDataset],
    *,
    draws: int,
    draw_seed: int,
    production: bool,
) -> dict[str, Any]:
    coefficient_rows = []
    dataset_rows = []
    roundtrip_max_delta = 0.0
    roundtrip_engine = (
        type(engine).load(engine.checkpoint_path)
        if engine.checkpoint_path is not None
        else engine
    )
    for index, dataset in enumerate(datasets):
        posterior = engine.predict_beta_posterior(dataset)
        roundtrip = roundtrip_engine.predict_beta_posterior(dataset)
        roundtrip_max_delta = max(
            roundtrip_max_delta,
            float(np.max(np.abs(posterior.mean.numpy() - roundtrip.mean.numpy()))),
            float(np.max(np.abs(posterior.scale.numpy() - roundtrip.scale.numpy()))),
        )
        anchor = _anchor_posterior(dataset)
        truth = dataset.truth_beta.to_numpy(dtype=float)
        mean = posterior.mean.numpy()[0]
        scale = posterior.scale.numpy()[0]
        anchor_mean = anchor.mean.numpy()[0]
        rng = np.random.default_rng(draw_seed + index)
        samples = rng.normal(loc=mean, scale=scale, size=(draws,) + truth.shape)
        ranks = np.mean(samples < truth[None, ...], axis=0)
        covered = np.abs(mean - truth) <= 1.959963984540054 * scale
        for covariate in range(truth.shape[0]):
            for species in range(truth.shape[1]):
                coefficient_rows.append(
                    {
                        "covered": bool(covered[covariate, species]),
                        "rank": float(ranks[covariate, species]),
                        "squared_error": float(
                            (mean[covariate, species] - truth[covariate, species]) ** 2
                        ),
                        "anchor_squared_error": float(
                            (
                                anchor_mean[covariate, species]
                                - truth[covariate, species]
                            )
                            ** 2
                        ),
                        "coefficient_role": (
                            "intercept" if covariate == 0 else "non_intercept"
                        ),
                        "covariate_count": str(truth.shape[0]),
                        "site_stratum": dataset.metadata["strata"]["site"],
                        "species_stratum": dataset.metadata["strata"]["species"],
                        "design_condition_stratum": dataset.metadata["strata"][
                            "design_condition"
                        ],
                    }
                )
        y = dataset.Y.to_numpy(dtype=float)
        neural_probability = _posterior_probability(posterior, dataset.X)
        anchor_probability = _posterior_probability(anchor, dataset.X)
        dataset_rows.append(
            {
                "seed": int(dataset.metadata["seed"]),
                "n_sites": len(dataset.X),
                "n_species": dataset.Y.shape[1],
                "n_covariates": truth.shape[0],
                "target_condition": dataset.metadata["target_condition"],
                "actual_condition": dataset.metadata["actual_condition"],
                "strata": dataset.metadata["strata"],
                "prevalence_stratum": dataset.metadata["prevalence_stratum"],
                "effect_stratum": dataset.metadata["effect_stratum"],
                "neural_brier": _brier(y, neural_probability),
                "anchor_brier": _brier(y, anchor_probability),
                "neural_log_loss": _log_loss(y, neural_probability),
                "anchor_log_loss": _log_loss(y, anchor_probability),
            }
        )
    summary = _coefficient_summary(coefficient_rows) | {
        "neural_brier": float(np.mean([row["neural_brier"] for row in dataset_rows])),
        "anchor_brier": float(np.mean([row["anchor_brier"] for row in dataset_rows])),
        "neural_log_loss": float(
            np.mean([row["neural_log_loss"] for row in dataset_rows])
        ),
        "anchor_log_loss": float(
            np.mean([row["anchor_log_loss"] for row in dataset_rows])
        ),
    }
    strata = {
        field: _summarize_strata(coefficient_rows, field)
        for field in (
            "covariate_count",
            "coefficient_role",
            "site_stratum",
            "species_stratum",
            "design_condition_stratum",
        )
    }
    statistical_gates = {
        "coverage_95": 0.925 <= summary["coverage_95"] <= 0.975,
        "rank_mean": abs(summary["rank_mean"] - 0.5) <= 0.025,
        "rank_variance": abs(summary["rank_variance"] - 1.0 / 12.0) <= 0.025,
        "mean_rmse_vs_anchor": summary["rmse"] <= 1.05 * summary["anchor_rmse"],
        "brier_vs_anchor": summary["neural_brier"] <= 1.02 * summary["anchor_brier"],
        "log_loss_vs_anchor": summary["neural_log_loss"]
        <= 1.02 * summary["anchor_log_loss"],
        "stratum_calibration": _stratum_gates_pass(strata),
    }
    if production:
        statistical_gates["factorial_balance"] = _production_factorial_balanced(
            datasets
        )
    return {
        "summary": summary,
        "strata": strata,
        "dataset_rows": dataset_rows,
        "roundtrip_max_delta": roundtrip_max_delta,
        "gates": {
            "checkpoint_roundtrip": roundtrip_max_delta <= 1e-6,
            **statistical_gates,
        },
    }


def run_mcmc_comparison(
    engine: VariableDesignNeuralHmscInference,
    datasets: Sequence[FixedEffectDataset],
    *,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    for context in MCMC_CONTEXTS:
        match = next(
            dataset
            for dataset in datasets
            if (
                len(dataset.X),
                dataset.Y.shape[1],
                dataset.truth_beta.shape[0],
                float(dataset.metadata["target_condition"]),
            )
            == context
        )
        selected.append(match)
    rows = []
    for dataset in selected:
        seed = int(dataset.metadata["seed"])
        n_train = int(np.floor(0.75 * len(dataset.X)))
        train = replace(
            dataset,
            X=dataset.X.iloc[:n_train].copy(),
            Y=dataset.Y.iloc[:n_train].copy(),
            linear_predictor=dataset.linear_predictor.iloc[:n_train].copy(),
            metadata={**dataset.metadata, "n_sites": n_train},
        )
        test_X = dataset.X.iloc[n_train:].copy()
        test_Y = dataset.Y.iloc[n_train:].to_numpy(dtype=float)
        fit = HmscModel(
            Y=train.Y,
            X=train.X,
            x_formula=str(dataset.metadata["formula"]),
            distr="probit",
        ).sample(
            samples=200,
            transient=100,
            thin=1,
            chains=1,
            init="python-native",
            workdir=output / f"work_{seed}",
            verbose=200,
            output_file=output / f"mcmc_{seed}.h5",
        )
        neural_probability = _posterior_probability(
            engine.predict_beta_posterior(train), test_X
        )
        mcmc_probability = fit.predict_mean(test_X).to_numpy(dtype=float)
        rows.append(
            {
                "seed": seed,
                "context": list(
                    (
                        len(dataset.X),
                        dataset.Y.shape[1],
                        dataset.truth_beta.shape[0],
                        dataset.metadata["target_condition"],
                    )
                ),
                "neural_brier": _brier(test_Y, neural_probability),
                "mcmc_brier": _brier(test_Y, mcmc_probability),
                "neural_log_loss": _log_loss(test_Y, neural_probability),
                "mcmc_log_loss": _log_loss(test_Y, mcmc_probability),
            }
        )
    return {
        "rows": rows,
        "neural_to_mcmc_brier_ratio": float(
            np.mean([row["neural_brier"] for row in rows])
            / np.mean([row["mcmc_brier"] for row in rows])
        ),
        "neural_to_mcmc_log_loss_ratio": float(
            np.mean([row["neural_log_loss"] for row in rows])
            / np.mean([row["mcmc_log_loss"] for row in rows])
        ),
    }


def validate_freeze(root: str | Path, *, role: str) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    path = root / "m54_train_calibration_freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("kind")
        != "neural_hmsc_variable_design_m54_train_calibration_freeze"
    ):
        raise ValueError("unsupported Milestone 54 freeze")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("role") != role:
        raise ValueError("Milestone 54 freeze protocol or role differs")
    if payload.get("status") != "frozen_before_reserved_evaluation":
        raise ValueError("Milestone 54 role is not frozen before evaluation")
    if payload.get("reserved_evaluation_opened") is not False:
        raise ValueError("Milestone 54 freeze already records opened evaluation")
    if payload.get("settings") != _production_settings():
        raise ValueError("Milestone 54 production settings differ")
    spec = ROLE_SPECS[role]
    seeds = payload["seeds"]
    if seeds["train"] != _seed_block(spec["train"], PRODUCTION_COUNT):
        raise ValueError("Milestone 54 training seeds differ")
    if seeds["calibration"] != _seed_block(spec["calibration"], PRODUCTION_COUNT):
        raise ValueError("Milestone 54 calibration seeds differ")
    if seeds["reserved_evaluation_start"] != spec["evaluation"]:
        raise ValueError("Milestone 54 reserved evaluation start differs")
    checkpoint = root / "checkpoint"
    record = payload["checkpoint"]
    for name, filename in (
        ("manifest_sha256", "neural_checkpoint.json"),
        ("weights_sha256", "weights.weights.h5"),
        ("calibration_sha256", "variable_design_calibration.json"),
    ):
        if record[name] != _sha256(checkpoint / filename):
            raise ValueError(f"Milestone 54 checkpoint {name} differs")
    VariableDesignNeuralHmscInference.load(checkpoint)
    return payload


def _new_engine() -> VariableDesignNeuralHmscInference:
    return VariableDesignNeuralHmscInference.for_fixed_effects(
        min_sites=12,
        max_sites=128,
        min_species=2,
        max_species=100,
        min_covariates=2,
        max_covariates=8,
    )


def _anchor_posterior(dataset: FixedEffectDataset) -> BetaPosterior:
    names = list(dataset.truth_beta.index)
    design = np.column_stack(
        [np.ones(len(dataset.X)), dataset.X[names[1:]].to_numpy(dtype=np.float32)]
    ).astype(np.float32)
    response = dataset.Y.to_numpy(dtype=np.float32)
    mean, scale = probit_irls_laplace_anchor(design[None, ...], response[None, ...])
    return BetaPosterior(mean=mean, scale=scale)


def _posterior_probability(posterior: BetaPosterior, X: pd.DataFrame) -> np.ndarray:
    design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    mean = posterior.mean.numpy()[0]
    scale = posterior.scale.numpy()[0]
    linear_mean = design @ mean
    linear_variance = np.square(design) @ np.square(scale)
    return ndtr(linear_mean / np.sqrt(1.0 + linear_variance))


def _coefficient_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    ranks = np.asarray([row["rank"] for row in rows], dtype=float)
    return {
        "n_coefficients": len(rows),
        "coverage_95": float(np.mean([row["covered"] for row in rows])),
        "rank_mean": float(np.mean(ranks)),
        "rank_variance": float(np.var(ranks)),
        "rmse": float(np.sqrt(np.mean([row["squared_error"] for row in rows]))),
        "anchor_rmse": float(
            np.sqrt(np.mean([row["anchor_squared_error"] for row in rows]))
        ),
    }


def _summarize_strata(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, dict[str, float]]:
    return {
        str(value): _coefficient_summary([row for row in rows if row[field] == value])
        for value in sorted({row[field] for row in rows})
    }


def _stratum_gates_pass(strata: dict[str, dict[str, dict[str, float]]]) -> bool:
    return all(
        0.90 <= summary["coverage_95"] <= 0.99
        and abs(summary["rank_mean"] - 0.5) <= 0.05
        and abs(summary["rank_variance"] - 1.0 / 12.0) <= 0.04
        for field in strata.values()
        for summary in field.values()
    )


def _production_factorial_balanced(datasets: Sequence[FixedEffectDataset]) -> bool:
    counts = _corpus_balance(datasets)["factorial_cell_counts"]
    return len(counts) == 81 and set(counts.values()) == {3}


def _marginals_balanced(datasets: Sequence[FixedEffectDataset]) -> bool:
    balance = _corpus_balance(datasets)["marginal_counts"]
    return all(
        max(counts.values()) - min(counts.values()) <= 1 for counts in balance.values()
    )


def _corpus_balance(datasets: Sequence[FixedEffectDataset]) -> dict[str, Any]:
    fields = {
        "site": [dataset.metadata["strata"]["site"] for dataset in datasets],
        "species": [dataset.metadata["strata"]["species"] for dataset in datasets],
        "covariate": [dataset.metadata["strata"]["covariate"] for dataset in datasets],
        "design_condition": [
            dataset.metadata["strata"]["design_condition"] for dataset in datasets
        ],
        "prevalence": [dataset.metadata["prevalence_stratum"] for dataset in datasets],
        "effect": [dataset.metadata["effect_stratum"] for dataset in datasets],
    }
    marginal_counts = {
        field: {value: values.count(value) for value in sorted(set(values))}
        for field, values in fields.items()
    }
    cells = [
        "|".join(
            (
                dataset.metadata["strata"]["site"],
                dataset.metadata["strata"]["species"],
                dataset.metadata["strata"]["covariate"],
                dataset.metadata["strata"]["design_condition"],
            )
        )
        for dataset in datasets
    ]
    return {
        "marginal_counts": marginal_counts,
        "factorial_cell_counts": {
            value: cells.count(value) for value in sorted(set(cells))
        },
    }


def _production_settings() -> dict[str, Any]:
    return {
        "corpus_count_per_phase": PRODUCTION_COUNT,
        "sites": list(SITE_LEVELS),
        "species": list(SPECIES_LEVELS),
        "covariates": list(COVARIATE_LEVELS),
        "target_conditions": list(CONDITION_LEVELS),
        "epochs": 40,
        "batch_size": 9,
        "learning_rate": 0.001,
        "mse_weight": 0.25,
        "sbc_draws": 256,
        "mcmc_datasets": 6,
        "mcmc_samples": 200,
        "mcmc_transient": 100,
    }


def _calibration_provenance(seeds: Sequence[int], *, corpus_id: str) -> dict[str, Any]:
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
        "calibration_sha256": _sha256(checkpoint / "variable_design_calibration.json"),
    }


def _add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fixed-registry",
        type=Path,
        default=Path("/private/tmp/neural_hmsc_releases"),
    )
    parser.add_argument(
        "--variable-baseline",
        type=Path,
        default=Path(
            "/private/tmp/neural_hmsc_variable_deployments/"
            "neural_hmsc_variable_probit_v1"
        ),
    )


def _validate_baselines(args: argparse.Namespace) -> dict[str, Any]:
    fixed = load_neural_hmsc_release(args.fixed_registry)
    variable = validate_variable_shape_baseline(args.variable_baseline)
    fixed_hash = fixed.manifest["content_sha256"]
    variable_hash = variable["content_sha256"]
    return {
        "fixed": fixed_hash,
        "variable": variable_hash,
        "all_valid": (
            fixed_hash == FIXED_BASELINE_HASH
            and variable_hash == VARIABLE_BASELINE_HASH
        ),
    }


def _read_role_report(path: Path, role: str) -> dict[str, Any]:
    payload = json.loads(_report_path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "neural_hmsc_variable_design_m54_role_evaluation":
        raise ValueError("unsupported Milestone 54 role report")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("role") != role:
        raise ValueError("Milestone 54 aggregate role differs")
    if payload.get("candidate_selected_using_sensitivity_outcomes") is not False:
        raise ValueError("Milestone 54 sensitivity outcomes selected the candidate")
    return payload


def _report_path(path: Path) -> Path:
    return path / "m54_role_evaluation.json" if path.is_dir() else path


def _seed_block(start: int, count: int) -> list[int]:
    return list(range(int(start), int(start) + int(count)))


def _is_production_seed(seed: int) -> bool:
    return any(
        start <= seed < start + PRODUCTION_COUNT
        for spec in ROLE_SPECS.values()
        for start in (spec["train"], spec["calibration"], spec["evaluation"])
    )


def _assert_disjoint(*blocks: Sequence[int]) -> None:
    flattened = [seed for block in blocks for seed in block]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Milestone 54 seed blocks overlap")


def _empty_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Milestone 54 output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _brier(y: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(y - probability)))


def _log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
