#!/usr/bin/env python3
"""Qualify one probit variable-shape Neural-HMSC checkpoint on simulations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc import HmscModel  # noqa: E402
from pyhmsc.neural import (  # noqa: E402
    VariableShapeNeuralHmscInference,
    simulate_fixed_effect_dataset,
)
from pyhmsc.neural.evaluation import BetaPosteriorMetrics  # noqa: E402
from pyhmsc.neural.models import probit_irls_laplace_anchor  # noqa: E402
from pyhmsc.neural.posterior_heads import BetaPosterior  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-sites", type=int, default=12)
    parser.add_argument("--max-sites", type=int, default=48)
    parser.add_argument("--min-species", type=int, default=2)
    parser.add_argument("--max-species", type=int, default=10)
    parser.add_argument("--train-datasets", type=int, default=64)
    parser.add_argument("--calibration-datasets", type=int, default=32)
    parser.add_argument("--test-datasets", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--run-mcmc", action="store_true")
    parser.add_argument("--mcmc-datasets", type=int, default=2)
    parser.add_argument("--mcmc-samples", type=int, default=200)
    parser.add_argument("--mcmc-transient", type=int, default=100)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_seeds = list(range(args.seed, args.seed + args.train_datasets))
    calibration_seeds = list(
        range(args.seed + 10000, args.seed + 10000 + args.calibration_datasets)
    )
    test_seeds = list(range(args.seed + 20000, args.seed + 20000 + args.test_datasets))
    train = _corpus(args, train_seeds, include_boundaries=True)
    calibration = _corpus(args, calibration_seeds, include_boundaries=True)
    test = _corpus(args, test_seeds, include_boundaries=True)

    engine = VariableShapeNeuralHmscInference.for_fixed_effects(
        min_sites=args.min_sites,
        max_sites=args.max_sites,
        min_species=args.min_species,
        max_species=args.max_species,
    )
    started = time.perf_counter()
    history = engine.fit(
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed + 30000,
    )
    training_seconds = time.perf_counter() - started
    calibration_result = engine.fit_calibration(
        calibration,
        provenance={
            "kind": "independent_variable_shape_simulation_calibration",
            "target_ecological_response_used": False,
            "shape_selection_role": "predeclared_range",
            "seeds": calibration_seeds,
            "corpus_id": "variable_shape_probit_simulation_v1",
        },
    )
    checkpoint = engine.save(output / "checkpoint")
    loaded = VariableShapeNeuralHmscInference.load(checkpoint)

    rows = []
    ranks = []
    roundtrip_max_delta = 0.0
    neural_seconds = 0.0
    for index, dataset in enumerate(test):
        started = time.perf_counter()
        posterior = loaded.predict_beta_posterior(dataset)
        neural_seconds += time.perf_counter() - started
        original = engine.predict_beta_posterior(dataset)
        roundtrip_max_delta = max(
            roundtrip_max_delta,
            float(np.max(np.abs(original.mean.numpy() - posterior.mean.numpy()))),
            float(np.max(np.abs(original.scale.numpy() - posterior.scale.numpy()))),
        )
        anchor = _anchor_posterior(dataset)
        truth = dataset.truth_beta.to_numpy(dtype=float)
        neural_metrics = _metrics(posterior, truth)
        anchor_metrics = _metrics(anchor, truth)
        neural_probability = _posterior_probability(posterior, dataset)
        anchor_probability = _posterior_probability(anchor, dataset)
        y = dataset.Y.to_numpy(dtype=float)
        rng = np.random.default_rng(args.seed + 40000 + index)
        samples = rng.normal(
            loc=posterior.mean.numpy()[0],
            scale=posterior.scale.numpy()[0],
            size=(args.draws,) + truth.shape,
        )
        ranks.append(np.mean(samples < truth[None, ...], axis=0).ravel())
        rows.append(
            {
                "seed": int(dataset.metadata["seed"]),
                "n_sites": len(dataset.X),
                "n_species": dataset.Y.shape[1],
                "neural": asdict(neural_metrics),
                "anchor": asdict(anchor_metrics),
                "neural_brier": _brier(y, neural_probability),
                "anchor_brier": _brier(y, anchor_probability),
                "neural_log_loss": _log_loss(y, neural_probability),
                "anchor_log_loss": _log_loss(y, anchor_probability),
            }
        )

    rank_values = np.concatenate(ranks)
    summary = _summarize(rows, rank_values)
    mcmc_datasets = [
        dataset
        for dataset in test
        if int(np.floor(0.75 * len(dataset.X))) >= args.min_sites
    ][: args.mcmc_datasets]
    mcmc = (
        _run_mcmc(args, mcmc_datasets, output)
        if args.run_mcmc
        else {"enabled": False, "rows": [], "summary": None}
    )
    gates = {
        "checkpoint_roundtrip": roundtrip_max_delta <= 1e-6,
        "boundary_shapes": _boundary_shapes_present(rows, args),
        "coverage_95": 0.925 <= summary["neural_coverage_95"] <= 0.975,
        "rank_mean": abs(summary["rank_mean"] - 0.5) <= 0.025,
        "rank_variance": abs(summary["rank_variance"] - 1.0 / 12.0) <= 0.025,
        "mean_rmse_vs_anchor": summary["neural_rmse"] <= 1.05 * summary["anchor_rmse"],
        "brier_vs_anchor": summary["neural_brier"] <= 1.02 * summary["anchor_brier"],
        "log_loss_vs_anchor": summary["neural_log_loss"]
        <= 1.02 * summary["anchor_log_loss"],
    }
    if args.run_mcmc:
        gates["proper_scores_vs_mcmc"] = (
            mcmc["summary"]["neural_to_mcmc_brier_ratio"] <= 1.10
            and mcmc["summary"]["neural_to_mcmc_log_loss_ratio"] <= 1.10
        )
    decision = (
        "variable_shape_probit_qualified"
        if all(gates.values()) and args.run_mcmc
        else (
            "variable_shape_probit_local_gates_passed_mcmc_pending"
            if all(gates.values())
            else "variable_shape_probit_not_qualified"
        )
    )
    result = {
        "schema_version": 1,
        "kind": "neural_hmsc_variable_shape_qualification",
        "created_from_target_ecological_data": False,
        "decision": decision,
        "shape_range": loaded.shape_range,
        "distribution": "probit",
        "seeds": {
            "train": train_seeds,
            "calibration": calibration_seeds,
            "test": test_seeds,
        },
        "settings": vars(args) | {"output": str(output)},
        "training": {
            "seconds": training_seconds,
            "final_loss": history.loss[-1],
            "final_beta_rmse": history.beta_rmse[-1],
        },
        "calibration": calibration_result.to_metadata(),
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_sha256": _sha256(checkpoint / "neural_checkpoint.json"),
            "weights_sha256": _sha256(checkpoint / "weights.weights.h5"),
            "roundtrip_max_delta": roundtrip_max_delta,
        },
        "test_rows": rows,
        "summary": summary,
        "mcmc": mcmc,
        "runtime": {
            "total_neural_inference_seconds": neural_seconds,
            "mean_neural_inference_seconds": neural_seconds / len(test),
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    json_path = output / "variable_shape_qualification.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "variable_shape_qualification.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "all_gates_passed": result["all_gates_passed"],
                "summary": summary,
                "checkpoint": str(checkpoint),
                "report": str(json_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _corpus(args, seeds, *, include_boundaries):
    rng = np.random.default_rng(seeds[0] - 1)
    shapes = []
    if include_boundaries and len(seeds) >= 3:
        shapes.extend(
            [
                (args.min_sites, args.min_species),
                (args.max_sites, args.max_species),
                (
                    (args.min_sites + args.max_sites) // 2,
                    (args.min_species + args.max_species) // 2,
                ),
            ]
        )
    while len(shapes) < len(seeds):
        shapes.append(
            (
                int(rng.integers(args.min_sites, args.max_sites + 1)),
                int(rng.integers(args.min_species, args.max_species + 1)),
            )
        )
    return [
        simulate_fixed_effect_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution="probit",
            seed=seed,
        )
        for seed, (n_sites, n_species) in zip(seeds, shapes)
    ]


def _anchor_posterior(dataset):
    X = np.column_stack(
        [np.ones(len(dataset.X)), dataset.X[["x1", "x2"]].to_numpy()]
    ).astype(np.float32)
    Y = dataset.Y.to_numpy(dtype=np.float32)
    mean, scale = probit_irls_laplace_anchor(X[None, ...], Y[None, ...])
    return BetaPosterior(mean=mean, scale=scale)


def _metrics(posterior, truth):
    mean = posterior.mean.numpy()[0]
    scale = posterior.scale.numpy()[0]
    error = mean - truth
    covered = np.abs(error) <= 1.959963984540054 * scale
    return BetaPosteriorMetrics(
        beta_mean_rmse_truth=float(np.sqrt(np.mean(error**2))),
        beta_mean_mae_truth=float(np.mean(np.abs(error))),
        beta_interval_coverage_truth_95=float(np.mean(covered)),
        beta_interval_width_mean_95=float(np.mean(2 * 1.959963984540054 * scale)),
        beta_scale_min=float(np.min(scale)),
        beta_scale_mean=float(np.mean(scale)),
        zero_baseline_rmse_truth=float(np.sqrt(np.mean(truth**2))),
    )


def _posterior_probability(posterior, dataset_or_x):
    X_frame = getattr(dataset_or_x, "X", dataset_or_x)
    X = np.column_stack([np.ones(len(X_frame)), X_frame[["x1", "x2"]].to_numpy()])
    mean = posterior.mean.numpy()[0]
    scale = posterior.scale.numpy()[0]
    linear_mean = X @ mean
    linear_variance = np.square(X) @ np.square(scale)
    return ndtr(linear_mean / np.sqrt(1.0 + linear_variance))


def _run_mcmc(args, datasets, output):
    rows = []
    for dataset in datasets:
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
        model = HmscModel(
            Y=train.Y,
            X=train.X,
            x_formula="~ x1 + x2",
            distr="probit",
        )
        started = time.perf_counter()
        fit = model.sample(
            samples=args.mcmc_samples,
            transient=args.mcmc_transient,
            thin=1,
            chains=1,
            init="python-native",
            workdir=output / f"mcmc_{seed}",
            verbose=max(args.mcmc_samples, 1),
            output_file=output / f"mcmc_{seed}.h5",
        )
        mcmc_seconds = time.perf_counter() - started
        neural = VariableShapeNeuralHmscInference.load(output / "checkpoint")
        posterior = neural.predict_beta_posterior(train)
        neural_probability = _posterior_probability(posterior, test_X)
        mcmc_probability = fit.predict_mean(test_X).to_numpy(dtype=float)
        rows.append(
            {
                "seed": seed,
                "n_train_sites": n_train,
                "n_test_sites": len(test_X),
                "mcmc_seconds": mcmc_seconds,
                "neural_brier": _brier(test_Y, neural_probability),
                "mcmc_brier": _brier(test_Y, mcmc_probability),
                "neural_log_loss": _log_loss(test_Y, neural_probability),
                "mcmc_log_loss": _log_loss(test_Y, mcmc_probability),
            }
        )
    return {
        "enabled": True,
        "rows": rows,
        "summary": {
            "neural_to_mcmc_brier_ratio": float(
                np.mean([row["neural_brier"] for row in rows])
                / np.mean([row["mcmc_brier"] for row in rows])
            ),
            "neural_to_mcmc_log_loss_ratio": float(
                np.mean([row["neural_log_loss"] for row in rows])
                / np.mean([row["mcmc_log_loss"] for row in rows])
            ),
            "mean_mcmc_seconds": float(np.mean([row["mcmc_seconds"] for row in rows])),
        },
    }


def _summarize(rows, ranks):
    def average(path):
        first, second = path
        return float(np.mean([row[first][second] for row in rows]))

    return {
        "neural_rmse": average(("neural", "beta_mean_rmse_truth")),
        "anchor_rmse": average(("anchor", "beta_mean_rmse_truth")),
        "neural_coverage_95": average(("neural", "beta_interval_coverage_truth_95")),
        "anchor_coverage_95": average(("anchor", "beta_interval_coverage_truth_95")),
        "neural_brier": float(np.mean([row["neural_brier"] for row in rows])),
        "anchor_brier": float(np.mean([row["anchor_brier"] for row in rows])),
        "neural_log_loss": float(np.mean([row["neural_log_loss"] for row in rows])),
        "anchor_log_loss": float(np.mean([row["anchor_log_loss"] for row in rows])),
        "rank_mean": float(np.mean(ranks)),
        "rank_variance": float(np.var(ranks)),
    }


def _boundary_shapes_present(rows, args):
    shapes = {(row["n_sites"], row["n_species"]) for row in rows}
    return (args.min_sites, args.min_species) in shapes and (
        args.max_sites,
        args.max_species,
    ) in shapes


def _brier(y, probability):
    return float(np.mean(np.square(y - probability)))


def _log_loss(y, probability):
    p = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _markdown(result):
    lines = [
        "# Variable-Shape Neural-HMSC Qualification",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "## Gates",
        "",
        "| Gate | Passed |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| {name} | {'yes' if passed else 'no'} |"
        for name, passed in result["gates"].items()
    )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This evaluation uses disjoint simulated train, calibration, test, and MCMC seeds. No ecological target dataset or target-specific selector is used.",
            "",
            "Qualified Python MCMC remains the statistical reference; this checkpoint is an amortized approximation.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
