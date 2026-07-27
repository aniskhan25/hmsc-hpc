#!/usr/bin/env python3
"""Qualify one fixed-shape probit trait-Gamma Neural-HMSC checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache")
)

import numpy as np
import pandas as pd
from scipy.special import ndtr
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.generate_whittaker_holdout_validation import (
    generate_project,
)  # noqa: E402
from pyhmsc import HmscModel, compile_hmsc_model  # noqa: E402
from pyhmsc.neural import simulate_trait_gamma_boundary_dataset  # noqa: E402
from pyhmsc.neural.trait_inference import (  # noqa: E402
    TraitGammaNeuralHmscInference,
    _sha256,
)


FIXED_RELEASE_DIGEST = (
    "affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8"
)
VARIABLE_RELEASE_DIGEST = (
    "badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--train-datasets", type=int, default=64)
    parser.add_argument("--calibration-datasets", type=int, default=32)
    parser.add_argument("--test-datasets", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sbc-draws", type=int, default=256)
    parser.add_argument("--mcmc-samples", type=int, default=80)
    parser.add_argument("--mcmc-transient", type=int, default=80)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--skip-mcmc", action="store_true")
    parser.add_argument("--run-realdata", action="store_true")
    parser.add_argument(
        "--whittaker-source",
        type=Path,
        default=Path("examples/projects/whittaker_plants_hmsc_book"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)
    training = _corpus(args.train_datasets, args.seed)
    calibration = _corpus(args.calibration_datasets, args.seed + 100_000)
    testing = _corpus(args.test_datasets, args.seed + 200_000)
    engine = TraitGammaNeuralHmscInference.for_trait_gamma(
        n_sites=40,
        n_species=75,
        n_covariates=2,
        n_traits=1,
        hidden_units=(64, 64),
    )
    started = time.perf_counter()
    history = engine.fit(
        training,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    training_seconds = time.perf_counter() - started
    gamma_calibration = engine.fit_calibration(
        calibration,
        provenance={
            "corpus_id": "trait_gamma_boundary_calibration_v1",
            "seeds": [args.seed + 100_000 + index for index in range(len(calibration))],
            "independent_from_training": True,
        },
    )

    raw = _gamma_arrays(engine, testing, calibrated=False)
    calibrated = _gamma_arrays(engine, testing, calibrated=True)
    anchor_engine = TraitGammaNeuralHmscInference.for_trait_gamma(
        n_sites=40,
        n_species=75,
        n_covariates=2,
        n_traits=1,
        hidden_units=(64, 64),
    )
    anchor = _gamma_arrays(anchor_engine, testing, calibrated=False)
    gamma_metrics = _gamma_metrics(
        calibrated[0], calibrated[1], calibrated[2], args.sbc_draws, args.seed + 300_000
    )
    raw_metrics = _gamma_metrics(
        raw[0], raw[1], raw[2], args.sbc_draws, args.seed + 310_000
    )
    anchor_metrics = _gamma_metrics(
        anchor[0], anchor[1], anchor[2], args.sbc_draws, args.seed + 320_000
    )
    checkpoint = engine.save(args.output / "checkpoint")
    loaded = TraitGammaNeuralHmscInference.load(checkpoint)
    parity_before = engine.predict_gamma_posterior(testing[0]).mean.numpy()
    parity_after = loaded.predict_gamma_posterior(testing[0]).mean.numpy()
    checkpoint_parity = float(np.max(np.abs(parity_before - parity_after)))

    simulated_reference = None
    if not args.skip_mcmc:
        simulated_reference = _simulated_mcmc_gate(
            engine=loaded,
            dataset=testing[0],
            output=args.output / "simulated_mcmc",
            seed=args.seed + 400_000,
            samples=args.mcmc_samples,
            transient=args.mcmc_transient,
            thin=args.mcmc_thin,
        )
    realdata = None
    if args.run_realdata and not args.skip_mcmc:
        realdata = _whittaker_gate(
            engine=loaded,
            source=args.whittaker_source,
            output=args.output / "whittaker",
            seed=args.seed + 500_000,
            samples=args.mcmc_samples,
            transient=args.mcmc_transient,
            thin=args.mcmc_thin,
        )

    gates = {
        "gamma_coverage": 0.90 <= gamma_metrics["coverage_95"] <= 0.99,
        "gamma_rank_mean": 0.40 <= gamma_metrics["rank_mean"] <= 0.60,
        "gamma_rank_variance": 0.06 <= gamma_metrics["rank_variance"] <= 0.11,
        "gamma_anchor_no_degradation": gamma_metrics["rmse"]
        <= 1.05 * anchor_metrics["rmse"],
        "checkpoint_parity": checkpoint_parity <= 1e-6,
        "calibration_independent": gamma_calibration.provenance[
            "independent_from_training"
        ]
        is True,
    }
    if simulated_reference is not None:
        gates.update(
            {
                "simulated_gamma_mcmc": simulated_reference["gamma_rmse_ratio"] <= 1.25,
                "simulated_brier_mcmc": simulated_reference["brier_ratio"] <= 1.05,
                "simulated_log_loss_mcmc": simulated_reference["log_loss_ratio"]
                <= 1.05,
            }
        )
    if realdata is not None:
        gates.update(
            {
                "whittaker_brier_mcmc": realdata["brier_ratio"] <= 1.05,
                "whittaker_log_loss_mcmc": realdata["log_loss_ratio"] <= 1.05,
                "whittaker_gamma_agreement": realdata["gamma_mean_mae_mcmc"] <= 0.35,
            }
        )
    all_gates = all(gates.values())
    decision = (
        "trait_gamma_probit_qualified"
        if all_gates and simulated_reference is not None and realdata is not None
        else (
            "trait_gamma_probit_local_gates_passed_reference_pending"
            if all_gates
            else "trait_gamma_probit_not_qualified"
        )
    )
    report = {
        "kind": "neural_hmsc_trait_gamma_qualification",
        "schema_version": 1,
        "decision": decision,
        "all_gates_passed": all_gates,
        "seed": args.seed,
        "candidate_predeclared": args.seed == 20260801,
        "selection_used_sensitivity_outcomes": False,
        "scope": {
            "distribution": "probit",
            "n_sites": 40,
            "n_species": 75,
            "n_covariates": 2,
            "n_traits": 1,
            "formula": "~ TMG",
            "trait_formula": "~ CN",
            "phylogeny": False,
            "random_effects": False,
        },
        "corpus": {
            "training": args.train_datasets,
            "calibration": args.calibration_datasets,
            "test": args.test_datasets,
            "disjoint_seed_windows": True,
        },
        "training_seconds": training_seconds,
        "history_final": {
            "loss": history.loss[-1],
            "gamma_rmse": history.beta_rmse[-1],
            "scale_mean": history.scale_mean[-1],
        },
        "gamma_calibration": gamma_calibration.to_metadata(),
        "gamma_metrics": gamma_metrics,
        "raw_gamma_metrics": raw_metrics,
        "anchor_gamma_metrics": anchor_metrics,
        "checkpoint_parity_max_abs": checkpoint_parity,
        "checkpoint_manifest_sha256": _sha256(checkpoint / "neural_checkpoint.json"),
        "checkpoint_weights_sha256": _sha256(checkpoint / "weights.weights.h5"),
        "simulated_reference": simulated_reference,
        "realdata": realdata,
        "gates": gates,
        "fixed_release_content_sha256": FIXED_RELEASE_DIGEST,
        "variable_release_content_sha256": VARIABLE_RELEASE_DIGEST,
        "existing_releases_modified": False,
        "claim_boundary": (
            "summary-level Beta/Gamma marginal approximation for the declared "
            "trait-probit shape; no coupled joint-posterior equivalence"
        ),
    }
    path = args.output / "trait_gamma_qualification.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _corpus(count: int, seed: int):
    return [
        simulate_trait_gamma_boundary_dataset(
            n_sites=40,
            n_species=75,
            seed=seed + index,
            beta_residual_scale=float([0.1, 0.2, 0.35][index % 3]),
            gamma_scale=float([0.55, 0.8, 1.05][index % 3]),
        )
        for index in range(count)
    ]


def _gamma_arrays(engine, datasets, *, calibrated):
    means = []
    scales = []
    truths = []
    for dataset in datasets:
        posterior = engine.predict_gamma_posterior(dataset, calibrated=calibrated)
        means.append(posterior.mean.numpy()[0])
        scales.append(posterior.scale.numpy()[0])
        truths.append(dataset.truth_gamma.to_numpy(dtype=float))
    return np.stack(means), np.stack(scales), np.stack(truths)


def _gamma_metrics(mean, scale, truth, draws, seed):
    rng = np.random.default_rng(seed)
    samples = rng.normal(mean, scale, size=(draws,) + mean.shape)
    ranks = np.mean(samples < truth[None, ...], axis=0)
    error = mean - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "coverage_95": float(np.mean(np.abs(error) <= 1.959963984540054 * scale)),
        "rank_mean": float(np.mean(ranks)),
        "rank_variance": float(np.var(ranks)),
        "interval_width_mean_95": float(np.mean(2.0 * 1.959963984540054 * scale)),
    }


def _simulated_mcmc_gate(*, engine, dataset, output, seed, samples, transient, thin):
    output.mkdir(parents=True, exist_ok=True)
    model = HmscModel(
        Y=dataset.Y,
        X=dataset.X,
        x_formula="~ TMG",
        distr="probit",
        traits=dataset.traits,
        trait_formula="~ CN",
    )
    mcmc = model.sample(
        samples=samples,
        transient=transient,
        thin=thin,
        chains=1,
        init="python-native",
        workdir=output / "work",
        output_file=output / "mcmc.h5",
        rng_seed=seed,
        verbose=max(int(samples), 1),
    )
    neural = engine.infer(
        {
            "X": np.column_stack([np.ones(40), dataset.X["TMG"]]),
            "Y": dataset.Y,
            "T": dataset.trait_design,
            "distribution": "probit",
            "formula": "~ TMG",
            "trait_formula": "~ CN",
            "covariate_names": ["Intercept", "TMG"],
            "species_names": list(dataset.Y.columns),
            "trait_names": ["CN"],
        },
        chains=1,
        draws=max(samples, 64),
        seed=seed + 1,
        output=output / "neural.h5",
    )
    truth_gamma = dataset.truth_gamma.to_numpy(dtype=float)
    neural_rmse = _rmse(neural.gamma_mean().to_numpy(), truth_gamma)
    mcmc_rmse = _rmse(mcmc.gamma_mean().to_numpy(), truth_gamma)
    test_x = np.random.default_rng(seed + 2).normal(size=12)
    test_x = (test_x - test_x.mean()) / test_x.std(ddof=1)
    design = np.column_stack([np.ones(12), test_x])
    truth_probability = ndtr(design @ dataset.truth_beta.to_numpy(dtype=float))
    observed = np.random.default_rng(seed + 3).binomial(1, truth_probability)
    neural_probability = ndtr(design @ neural.beta_mean().to_numpy())
    mcmc_probability = ndtr(design @ mcmc.beta_mean().to_numpy())
    neural_scores = _scores(observed, neural_probability)
    mcmc_scores = _scores(observed, mcmc_probability)
    return {
        "neural_gamma_rmse_truth": neural_rmse,
        "mcmc_gamma_rmse_truth": mcmc_rmse,
        "gamma_rmse_ratio": neural_rmse / max(mcmc_rmse, 1e-12),
        "gamma_mean_mae_mcmc": float(
            np.mean(
                np.abs(neural.gamma_mean().to_numpy() - mcmc.gamma_mean().to_numpy())
            )
        ),
        "neural_brier": neural_scores["brier"],
        "mcmc_brier": mcmc_scores["brier"],
        "brier_ratio": neural_scores["brier"] / mcmc_scores["brier"],
        "neural_log_loss": neural_scores["log_loss"],
        "mcmc_log_loss": mcmc_scores["log_loss"],
        "log_loss_ratio": neural_scores["log_loss"] / mcmc_scores["log_loss"],
    }


def _whittaker_gate(*, engine, source, output, seed, samples, transient, thin):
    output.mkdir(parents=True, exist_ok=True)
    project = output / "holdout"
    generate_project(source, project, 12)
    train_y = pd.read_csv(project / "data/train/Y.csv", index_col=0)
    train_x = pd.read_csv(project / "data/train/X.csv", index_col=0)
    test_y = pd.read_csv(project / "data/test/Y.csv", index_col=0)
    test_x = pd.read_csv(project / "data/test/X.csv", index_col=0)
    traits = pd.read_csv(source / "data/traits.csv", index_col=0).loc[train_y.columns]
    compiled = compile_hmsc_model(
        Y=train_y,
        X=train_x,
        formula="~ TMG",
        distr="probit",
        output=output / "compiled",
        traits=traits,
        trait_formula="~ CN",
    )
    neural = engine.infer(
        compiled.init_json,
        chains=1,
        draws=max(samples, 64),
        seed=seed,
        output=output / "neural.h5",
    )
    model = HmscModel(
        Y=train_y,
        X=train_x,
        x_formula="~ TMG",
        distr="probit",
        traits=traits,
        trait_formula="~ CN",
    )
    mcmc = model.sample(
        samples=samples,
        transient=transient,
        thin=thin,
        chains=1,
        init="python-native",
        workdir=output / "mcmc_work",
        output_file=output / "mcmc.h5",
        rng_seed=seed + 1,
        verbose=max(int(samples), 1),
    )
    x_scale = compiled.metadata["preprocessing"]["XScalePar"]
    x_index = list(x_scale["columns"]).index("TMG")
    mean = float(x_scale["mean"][x_index])
    sd = float(x_scale["sd"][x_index])
    design = np.column_stack([np.ones(len(test_x)), (test_x["TMG"] - mean) / sd])
    observed = test_y.to_numpy(dtype=float)
    neural_probability = ndtr(design @ neural.beta_mean().to_numpy())
    mcmc_probability = ndtr(design @ mcmc.beta_mean().to_numpy())
    neural_scores = _scores(observed, neural_probability)
    mcmc_scores = _scores(observed, mcmc_probability)
    return {
        "dataset": "whittaker_plants_12_site_holdout",
        "outcomes_used_for_training_or_selection": False,
        "neural_brier": neural_scores["brier"],
        "mcmc_brier": mcmc_scores["brier"],
        "brier_ratio": neural_scores["brier"] / mcmc_scores["brier"],
        "neural_log_loss": neural_scores["log_loss"],
        "mcmc_log_loss": mcmc_scores["log_loss"],
        "log_loss_ratio": neural_scores["log_loss"] / mcmc_scores["log_loss"],
        "gamma_mean_mae_mcmc": float(
            np.mean(
                np.abs(neural.gamma_mean().to_numpy() - mcmc.gamma_mean().to_numpy())
            )
        ),
    }


def _scores(observed, probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    observed = np.asarray(observed, dtype=float)
    return {
        "brier": float(np.mean((observed - probability) ** 2)),
        "log_loss": float(
            -np.mean(
                observed * np.log(probability)
                + (1.0 - observed) * np.log(1.0 - probability)
            )
        ),
    }


def _rmse(left, right):
    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


if __name__ == "__main__":
    main()
