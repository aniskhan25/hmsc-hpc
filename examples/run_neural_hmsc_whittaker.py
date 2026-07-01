"""Run Neural-HMSC on the real Whittaker plant held-out-site dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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

from examples.generate_whittaker_holdout_validation import generate_project
from pyhmsc.model import HmscModel
from pyhmsc.neural.benchmark import (
    compare_beta_posteriors,
    occurrence_predictive_acceptance,
    sbc_calibration_acceptance,
    write_benchmark_report,
    write_sbc_report,
)
from pyhmsc.neural.calibration import (
    apply_beta_predictive_calibration,
    apply_beta_scale_calibration,
    fit_beta_scale_calibration,
)
from pyhmsc.neural.diagnostics import beta_sbc_rank_diagnostics
from pyhmsc.neural.inference import NeuralHmscInference
from pyhmsc.neural.posterior_heads import sample_beta_posterior
from pyhmsc.neural.simulator import FixedEffectDataset
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import fixed_shape_training_data
from pyhmsc.posterior import HmscFit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("examples/projects/whittaker_plants_hmsc_book"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-sites", type=int, default=12)
    parser.add_argument("--train-datasets", type=int, default=512)
    parser.add_argument("--calibration-datasets", type=int, default=128)
    parser.add_argument("--sbc-datasets", type=int, default=128)
    parser.add_argument("--sbc-draws", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--neural-chains", type=int, default=4)
    parser.add_argument("--neural-draws", type=int, default=1000)
    parser.add_argument("--mcmc-chains", type=int, default=2)
    parser.add_argument("--mcmc-samples", type=int, default=1000)
    parser.add_argument("--mcmc-transient", type=int, default=500)
    parser.add_argument("--mcmc-thin", type=int, default=5)
    parser.add_argument("--mcmc-verbose", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()
    _validate_args(parser, args)

    args.output.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tf.keras.utils.set_random_seed(args.seed)
    project = args.output / "whittaker_holdout"
    generate_project(args.source, project, args.test_sites)
    train_Y = pd.read_csv(project / "data/train/Y.csv", index_col=0)
    train_X = pd.read_csv(project / "data/train/X.csv", index_col=0)
    test_Y = pd.read_csv(project / "data/test/Y.csv", index_col=0)
    test_X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    if not train_Y.columns.equals(test_Y.columns):
        raise ValueError("training and held-out species must have identical order")

    species_names = [str(name) for name in train_Y.columns]
    simulation_kwargs = {
        "n_sites": len(train_Y),
        "n_species": train_Y.shape[1],
        "tmg": train_X["TMG"].to_numpy(dtype=float),
    }
    simulation_start = time.perf_counter()
    training = _simulate_corpus(
        count=args.train_datasets,
        seed=args.seed,
        species_names=species_names,
        **simulation_kwargs,
    )
    calibration_datasets = _simulate_corpus(
        count=args.calibration_datasets,
        seed=args.seed + 100_000,
        species_names=species_names,
        **simulation_kwargs,
    )
    sbc_datasets = _simulate_corpus(
        count=args.sbc_datasets,
        seed=args.seed + 200_000,
        species_names=species_names,
        **simulation_kwargs,
    )
    simulation_seconds = time.perf_counter() - simulation_start

    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=len(train_Y),
        n_species=train_Y.shape[1],
        n_covariates=2,
        distribution="probit",
        formula="~ TMG",
        covariate_names=["Intercept", "TMG"],
        species_names=species_names,
        hidden_units=(192, 192),
    )
    training_start = time.perf_counter()
    history = engine.fit(
        training,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    training_seconds = time.perf_counter() - training_start
    checkpoint = engine.save(args.output / "neural_checkpoint")

    calibration_data = fixed_shape_training_data(calibration_datasets)
    calibration_posterior = engine.predict_beta_posterior(calibration_data)
    calibration = fit_beta_scale_calibration(
        calibration_posterior,
        calibration_data.Beta,
        distribution="probit",
        predictive_X=calibration_data.X,
        predictive_Y=calibration_data.Y,
        predictive_seed=args.seed + 250_000,
    )

    real_input = {
        "X": _design(train_X),
        "Y": train_Y.to_numpy(dtype=np.float32),
        "distribution": "probit",
        "formula": "~ TMG",
        "covariate_names": ["Intercept", "TMG"],
        "species_names": species_names,
    }
    neural_start = time.perf_counter()
    uncalibrated = engine.predict_beta_posterior(real_input)
    coefficient_calibrated = apply_beta_scale_calibration(
        uncalibrated,
        calibration,
        distribution="probit",
    )
    predictive_only = apply_beta_predictive_calibration(
        uncalibrated,
        calibration,
        distribution="probit",
    )
    common_metadata = {
        "real_data": {
            "dataset": "Whittaker plants",
            "training_sites": len(train_Y),
            "heldout_sites": len(test_Y),
            "limitations": "fixed-effect environment-only model; traits, phylogeny, and latent effects excluded",
        }
    }
    uncalibrated_path = write_beta_posterior_hdf5(
        uncalibrated,
        args.output / "neural_posterior_uncalibrated.h5",
        covariate_names=["Intercept", "TMG"],
        species_names=species_names,
        distribution="probit",
        formula="~ TMG",
        chains=args.neural_chains,
        draws=args.neural_draws,
        seed=args.seed + 1,
        metadata=common_metadata
        | {"artifact_role": "coefficient_posterior_uncalibrated"},
    )
    coefficient_path = write_beta_posterior_hdf5(
        coefficient_calibrated,
        args.output / "neural_posterior.h5",
        covariate_names=["Intercept", "TMG"],
        species_names=species_names,
        distribution="probit",
        formula="~ TMG",
        chains=args.neural_chains,
        draws=args.neural_draws,
        seed=args.seed + 2,
        metadata=common_metadata | {"artifact_role": "coefficient_posterior"},
        calibration=calibration,
    )
    predictive_path = write_beta_posterior_hdf5(
        predictive_only,
        args.output / "neural_predictive_distribution.h5",
        covariate_names=["Intercept", "TMG"],
        species_names=species_names,
        distribution="probit",
        formula="~ TMG",
        chains=args.neural_chains,
        draws=args.neural_draws,
        seed=args.seed + 3,
        metadata=common_metadata | {"artifact_role": "predictive_only"},
        calibration=calibration,
    )
    neural_seconds = time.perf_counter() - neural_start

    sbc_rows = _sbc_rows(
        engine=engine,
        calibration=calibration,
        datasets=sbc_datasets,
        draws=args.sbc_draws,
        seed=args.seed + 300_000,
    )
    sbc_paths = write_sbc_report(
        sbc_rows,
        args.output,
        stem="whittaker_neural_sbc_diagnostics",
    )

    mcmc_model = HmscModel(
        Y=train_Y,
        X=train_X,
        x_formula="~ TMG",
        distr="probit",
    )
    mcmc_start = time.perf_counter()
    mcmc_fit = mcmc_model.sample(
        samples=args.mcmc_samples,
        transient=args.mcmc_transient,
        thin=args.mcmc_thin,
        chains=args.mcmc_chains,
        init="python-native",
        workdir=args.output / "mcmc_work",
        verbose=args.mcmc_verbose,
        output_file=args.output / "mcmc_posterior.h5",
    )
    mcmc_seconds = time.perf_counter() - mcmc_start

    posterior_rows = []
    heldout_rows = []
    for variant, coefficient_variant_path, predictive_variant_path in [
        ("uncalibrated", uncalibrated_path, uncalibrated_path),
        ("coefficient_calibrated", coefficient_path, coefficient_path),
        ("predictive_only_calibrated", coefficient_path, predictive_path),
    ]:
        neural_fit = HmscFit.from_file(coefficient_variant_path)
        neural_predictive_fit = HmscFit.from_file(predictive_variant_path)
        row = compare_beta_posteriors(
            neural_fit,
            mcmc_fit,
            dataset="whittaker_plants",
            distribution="probit",
            neural_seconds=neural_seconds,
            mcmc_seconds=mcmc_seconds,
            X=test_X,
            Y=test_Y,
            formula="~ TMG",
            neural_predictive_fit=neural_predictive_fit,
        )
        row["posterior_variant"] = variant
        posterior_rows.append(row)
        heldout_rows.append(
            _heldout_metrics(
                model=f"neural_{variant}",
                fit=neural_predictive_fit,
                X=test_X,
                Y=test_Y,
                training_prevalence=train_Y.mean(axis=0),
            )
        )
    heldout_rows.append(
        _heldout_metrics(
            model="mcmc_fixed",
            fit=mcmc_fit,
            X=test_X,
            Y=test_Y,
            training_prevalence=train_Y.mean(axis=0),
        )
    )

    comparison_paths = write_benchmark_report(
        posterior_rows,
        args.output,
        stem="whittaker_neural_mcmc_reference",
        title="Whittaker Neural-HMSC MCMC Reference",
    )
    heldout = pd.DataFrame(heldout_rows)
    predictive_acceptance = occurrence_predictive_acceptance(
        _metric_row(heldout, "neural_uncalibrated"),
        _metric_row(heldout, "neural_predictive_only_calibrated"),
        _metric_row(heldout, "mcmc_fixed"),
    )
    coefficient_sbc = next(
        row for row in sbc_rows if row["posterior_variant"] == "coefficient_calibrated"
    )
    acceptance = {
        **predictive_acceptance,
        **{
            key: value
            for key, value in coefficient_sbc.items()
            if key.startswith("sbc_")
        },
    }
    acceptance["qualification_acceptance_passed"] = bool(
        predictive_acceptance["predictive_acceptance_passed"]
        and coefficient_sbc["sbc_acceptance_passed"]
    )
    heldout.to_csv(args.output / "whittaker_heldout_metrics.csv", index=False)
    (args.output / "whittaker_acceptance.json").write_text(
        json.dumps(acceptance, indent=2) + "\n",
        encoding="utf-8",
    )
    report = _render_report(
        heldout=heldout,
        calibration=calibration.to_metadata(),
        sbc_rows=sbc_rows,
        train_Y=train_Y,
        train_X=train_X,
        test_X=test_X,
        training_seconds=training_seconds,
        neural_seconds=neural_seconds,
        mcmc_seconds=mcmc_seconds,
        acceptance=acceptance,
    )
    (args.output / "whittaker_neural_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "status": "completed",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args) | {"source": str(args.source), "output": str(args.output)},
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
        "tensorflow": tf.__version__,
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "simulation_seconds": simulation_seconds,
        "training_seconds": training_seconds,
        "neural_inference_seconds": neural_seconds,
        "mcmc_seconds": mcmc_seconds,
        "checkpoint": str(checkpoint),
        "reports": {
            "comparison_csv": str(comparison_paths.csv),
            "sbc_csv": str(sbc_paths.csv),
            "heldout_csv": str(args.output / "whittaker_heldout_metrics.csv"),
            "acceptance_json": str(args.output / "whittaker_acceptance.json"),
        },
        "acceptance": acceptance,
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(report)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ["train_datasets", "calibration_datasets", "sbc_datasets"]:
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} must be at least two")
    for name in [
        "sbc_draws",
        "epochs",
        "batch_size",
        "neural_chains",
        "neural_draws",
        "mcmc_chains",
        "mcmc_samples",
        "mcmc_thin",
        "mcmc_verbose",
    ]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.mcmc_transient < 0:
        parser.error("--mcmc-transient must be non-negative")
    if args.sbc_draws < 9:
        parser.error("--sbc-draws must be at least nine for the ten-bin SBC diagnostic")


def _simulate_corpus(
    *,
    count: int,
    n_sites: int,
    n_species: int,
    tmg: np.ndarray,
    species_names: list[str],
    seed: int,
) -> list[FixedEffectDataset]:
    return [
        _simulate_dataset(
            n_sites=n_sites,
            n_species=n_species,
            tmg=tmg,
            species_names=species_names,
            seed=seed + idx,
        )
        for idx in range(count)
    ]


def _simulate_dataset(
    *,
    n_sites: int,
    n_species: int,
    tmg: np.ndarray,
    species_names: list[str],
    seed: int,
) -> FixedEffectDataset:
    rng = np.random.default_rng(seed)
    tmg_values = np.asarray(tmg, dtype=float).copy()
    if tmg_values.shape != (n_sites,):
        raise ValueError(
            f"TMG design shape {tmg_values.shape} does not match n_sites={n_sites}"
        )
    tmg_values += rng.normal(scale=0.03, size=n_sites)
    mixture = rng.uniform(size=n_species)
    intercept = np.where(
        mixture < 0.80,
        rng.normal(-1.75, 0.75, size=n_species),
        np.where(
            mixture < 0.95,
            rng.normal(-0.5, 0.6, size=n_species),
            rng.normal(1.5, 0.5, size=n_species),
        ),
    )
    slope = rng.normal(0.0, 0.8, size=n_species)
    beta = np.vstack([intercept, slope])
    design = np.column_stack([np.ones(n_sites), tmg_values])
    linear = design @ beta
    response = rng.binomial(1, ndtr(linear))
    sites = [f"site_{idx:03d}" for idx in range(n_sites)]
    return FixedEffectDataset(
        Y=pd.DataFrame(response, index=sites, columns=species_names),
        X=pd.DataFrame({"TMG": tmg_values}, index=sites),
        truth_beta=pd.DataFrame(
            beta, index=["Intercept", "TMG"], columns=species_names
        ),
        linear_predictor=pd.DataFrame(linear, index=sites, columns=species_names),
        metadata={
            "distribution": "probit",
            "simulation_domain": "whittaker_shape_matched",
            "formula": "~ TMG",
            "intercept_prior": "rare_species_mixture",
            "slope_mean": 0.0,
            "slope_sd": 0.8,
        },
    )


def _sbc_rows(
    *,
    engine: NeuralHmscInference,
    calibration,
    datasets: list[FixedEffectDataset],
    draws: int,
    seed: int,
) -> list[dict[str, object]]:
    data = fixed_shape_training_data(datasets)
    uncalibrated = engine.predict_beta_posterior(data)
    calibrated = apply_beta_scale_calibration(
        uncalibrated, calibration, distribution="probit"
    )
    rows = []
    for idx, (variant, posterior) in enumerate(
        [("uncalibrated", uncalibrated), ("coefficient_calibrated", calibrated)]
    ):
        samples = sample_beta_posterior(posterior, draws=draws, seed=seed + idx).numpy()
        samples = np.transpose(samples, (1, 0, 2, 3))
        diagnostics = beta_sbc_rank_diagnostics(
            samples, data.Beta, n_bins=10, seed=seed + 10 + idx
        )
        row: dict[str, object] = {
            "distribution": "probit",
            "simulation_domain": "whittaker_shape_matched",
            "ood_regime": None,
            "posterior_variant": variant,
        }
        row.update(diagnostics.report_fields())
        rows.append(row)
    rows[1].update(sbc_calibration_acceptance(rows[0], rows[1]))
    return rows


def _design(X: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(X), dtype=np.float32),
            X["TMG"].to_numpy(dtype=np.float32),
        ]
    )


def _heldout_metrics(
    *,
    model: str,
    fit: HmscFit,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    training_prevalence: pd.Series,
) -> dict[str, object]:
    prediction = fit.predict_mean(X).loc[Y.index, Y.columns].clip(1e-9, 1.0 - 1e-9)
    probability = prediction.to_numpy(dtype=float)
    observed = Y.to_numpy(dtype=float)
    observed_richness = observed.sum(axis=1)
    predicted_richness = probability.sum(axis=1)
    row: dict[str, object] = {
        "model": model,
        "brier_score": float(np.mean(np.square(probability - observed))),
        "log_loss": float(
            -np.mean(
                observed * np.log(probability)
                + (1.0 - observed) * np.log(1.0 - probability)
            )
        ),
        "macro_auc": _macro_auc(Y, prediction),
        "auc_species": _auc_species_count(Y),
        "prevalence_mae": float(
            np.mean(np.abs(probability.mean(axis=0) - observed.mean(axis=0)))
        ),
        "richness_mae": float(np.mean(np.abs(predicted_richness - observed_richness))),
        "observed_richness_slope": float(np.polyfit(X["TMG"], observed_richness, 1)[0]),
        "predicted_richness_slope": float(
            np.polyfit(X["TMG"], predicted_richness, 1)[0]
        ),
    }
    aligned_prevalence = training_prevalence.loc[Y.columns].to_numpy(dtype=float)
    for name, mask in [
        ("rare", aligned_prevalence <= 0.10),
        ("intermediate", (aligned_prevalence > 0.10) & (aligned_prevalence <= 0.30)),
        ("common", aligned_prevalence > 0.30),
    ]:
        row[f"{name}_species"] = int(mask.sum())
        row[f"{name}_brier_score"] = (
            float(np.mean(np.square(probability[:, mask] - observed[:, mask])))
            if np.any(mask)
            else float("nan")
        )
        row[f"{name}_prevalence_mae"] = (
            float(
                np.mean(
                    np.abs(
                        probability[:, mask].mean(axis=0)
                        - observed[:, mask].mean(axis=0)
                    )
                )
            )
            if np.any(mask)
            else float("nan")
        )
    return row


def _metric_row(metrics: pd.DataFrame, model: str) -> dict[str, object]:
    matches = metrics.loc[metrics["model"] == model]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one held-out metric row for {model!r}")
    return matches.iloc[0].to_dict()


def _macro_auc(Y: pd.DataFrame, prediction: pd.DataFrame) -> float:
    values = []
    for species in Y.columns:
        observed = Y[species].to_numpy(dtype=int)
        positive = int(observed.sum())
        negative = len(observed) - positive
        if positive == 0 or negative == 0:
            continue
        ranks = (
            pd.Series(prediction[species].to_numpy(dtype=float))
            .rank(method="average")
            .to_numpy()
        )
        rank_sum = float(ranks[observed == 1].sum())
        values.append(
            (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
        )
    return float(np.mean(values)) if values else float("nan")


def _auc_species_count(Y: pd.DataFrame) -> int:
    present = Y.sum(axis=0)
    return int(((present > 0) & (present < len(Y))).sum())


def _render_report(
    *,
    heldout: pd.DataFrame,
    calibration: dict[str, object],
    sbc_rows: list[dict[str, object]],
    train_Y: pd.DataFrame,
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    training_seconds: float,
    neural_seconds: float,
    mcmc_seconds: float,
    acceptance: dict[str, object],
) -> str:
    sbc = pd.DataFrame(sbc_rows)
    sbc_summary = sbc[
        [
            "posterior_variant",
            "sbc_rank_mean",
            "sbc_rank_variance",
            "sbc_chi_square_pvalue",
            "sbc_beta_mean_rmse",
            "sbc_beta_interval_coverage_95",
        ]
    ]
    return "\n".join(
        [
            "# Whittaker Neural-HMSC Real-Data Validation",
            "",
            "Model: fixed-effect probit `presence ~ TMG`.",
            "Traits, phylogeny, and latent site effects are intentionally excluded.",
            "",
            f"Training/held-out sites: {len(train_Y)} / {len(test_X)}",
            f"Species: {train_Y.shape[1]}",
            f"Training species prevalence median: {float(train_Y.mean().median()):.6f}",
            f"Training TMG range: {float(train_X.TMG.min()):.6f} to {float(train_X.TMG.max()):.6f}",
            f"Held-out TMG range: {float(test_X.TMG.min()):.6f} to {float(test_X.TMG.max()):.6f}",
            "",
            "## Held-Out Metrics",
            "",
            heldout.to_string(index=False),
            "",
            "## Shape-Matched Simulation Calibration",
            "",
            sbc_summary.to_string(index=False),
            "",
            f"Coefficient-posterior scale multiplier: {float(calibration['scale_multiplier']):.6f}",
            "Predictive-only scale multiplier: "
            f"{float(calibration['predictive_scale_multiplier']):.6f}",
            f"Predictive method: {calibration['predictive_method']}",
            "",
            "## Acceptance Gates",
            "",
            f"Coefficient SBC acceptance: {bool(acceptance['sbc_acceptance_passed'])}",
            f"Held-out predictive acceptance: {bool(acceptance['predictive_acceptance_passed'])}",
            f"Combined qualification: {bool(acceptance['qualification_acceptance_passed'])}",
            "",
            "The coefficient gate is based only on independent simulated truth. The predictive gate is",
            "based on untouched Whittaker held-out observations. The predictive-only artifact is not",
            "a Beta posterior and is excluded from coefficient-posterior diagnostics.",
            "",
            "## Runtime",
            "",
            f"Neural training: {training_seconds:.3f} seconds",
            f"Neural real-data inference: {neural_seconds:.3f} seconds",
            f"MCMC sampling: {mcmc_seconds:.3f} seconds",
            f"Amortized inference speedup: {mcmc_seconds / max(neural_seconds, np.finfo(float).eps):.3f}x",
            "",
        ]
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    main()
