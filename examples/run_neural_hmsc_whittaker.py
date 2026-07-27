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
from pyhmsc.neural.conditional_calibration import (
    ConditionalBetaOODCalibrationBatch,
    ConditionalBetaScaleCalibration,
    apply_conditional_beta_scale_calibration,
    fit_conditional_beta_scale_calibration,
    fit_external_context_monotone_calibration,
)
from pyhmsc.neural.diagnostics import beta_sbc_stratified_diagnostics
from pyhmsc.neural.inference import NeuralHmscInference
from pyhmsc.neural.mean_calibration import (
    BetaResponseCalibrationBatch,
    apply_beta_predictive_mean_calibration,
    domain_conditional_predictive_mean_selector_metadata,
    fit_beta_response_mean_calibration,
    fit_beta_transfer_response_branch_calibration,
    fit_beta_transfer_response_mean_calibration,
    independent_source_transfer_predictive_mean_selector_metadata,
    select_predictive_mean_calibration_for_context,
)
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
    parser.add_argument(
        "--coefficient-calibration",
        choices=["scalar", "conditional", "external_monotone"],
        default="external_monotone",
        help="coefficient-posterior calibration; predictive calibration remains scalar",
    )
    parser.add_argument("--conditional-calibration-epochs", type=int, default=400)
    parser.add_argument(
        "--conditional-calibration-learning-rate", type=float, default=0.03
    )
    parser.add_argument(
        "--conditional-calibration-regularization", type=float, default=1e-3
    )
    parser.add_argument(
        "--conditional-calibration-rank-penalty-weight", type=float, default=0.02
    )
    parser.add_argument("--external-monotone-datasets", type=int, default=4)
    parser.add_argument("--external-monotone-max-multiplier", type=float, default=2.0)
    parser.add_argument("--external-monotone-min-ood-gain", type=float, default=0.005)
    parser.add_argument(
        "--external-monotone-min-combined-gain", type=float, default=0.005
    )
    parser.add_argument(
        "--predictive-mean-calibration",
        choices=[
            "none",
            "probit_response_affine",
            "probit_transfer_response_affine",
            "probit_source_transfer_response_affine",
        ],
        default="none",
        help=(
            "predictive-only posterior mean calibration applied only to "
            "neural_predictive_distribution.h5"
        ),
    )
    parser.add_argument(
        "--predictive-mean-calibration-validation-datasets",
        type=int,
        default=0,
        help="independent simulated validation datasets for predictive mean calibration",
    )
    parser.add_argument(
        "--predictive-mean-calibration-max-brier-ratio",
        type=float,
        default=1.0,
        help="maximum accepted validation Brier ratio for predictive mean calibration",
    )
    parser.add_argument(
        "--predictive-mean-calibration-max-log-loss-ratio",
        type=float,
        default=1.0,
        help="maximum accepted validation log-loss ratio for predictive mean calibration",
    )
    parser.add_argument(
        "--predictive-mean-calibration-min-improvement",
        type=float,
        default=0.0,
        help="minimum validation Brier+log-loss improvement required to select the correction",
    )
    parser.add_argument(
        "--predictive-mean-source-min-improvement",
        type=float,
        default=5.0e-4,
        help=(
            "minimum source-validation Brier+log-loss improvement required "
            "to select a nonidentity source branch"
        ),
    )
    parser.add_argument(
        "--predictive-mean-transfer-branch-min-improvement",
        type=float,
        default=1.0e-4,
        help=(
            "minimum independent OOD-validation Brier+log-loss improvement "
            "required to select the transfer branch"
        ),
    )
    parser.add_argument(
        "--predictive-mean-selection-policy",
        choices=["apply_selected", "domain_conditional"],
        default="apply_selected",
        help=(
            "how to deploy a selected predictive mean candidate; "
            "domain_conditional keeps source-like contexts scale-only and "
            "stores the candidate for transfer-like contexts"
        ),
    )
    parser.add_argument(
        "--predictive-mean-transfer-min-brier-gain",
        type=float,
        default=1.0e-4,
        help=(
            "minimum validation Brier improvement required before the "
            "domain-conditional selector can deploy the candidate on transfer contexts"
        ),
    )
    parser.add_argument(
        "--predictive-mean-transfer-min-log-loss-gain",
        type=float,
        default=5.0e-4,
        help=(
            "minimum validation log-loss improvement required before the "
            "domain-conditional selector can deploy the candidate on transfer contexts"
        ),
    )
    parser.add_argument(
        "--predictive-mean-transfer-max-slope-delta",
        type=float,
        default=0.05,
        help="maximum absolute slope movement from 1.0 allowed for transfer deployment",
    )
    parser.add_argument(
        "--predictive-mean-transfer-max-abs-intercept",
        type=float,
        default=0.025,
        help="maximum absolute intercept movement allowed for transfer deployment",
    )
    parser.add_argument(
        "--reference-parity-metrics",
        type=Path,
        help=(
            "direct R/Python parity metrics JSON qualifying the Python-native "
            "MCMC comparator against the original R+Python HMSC-HPC boundary"
        ),
    )
    parser.add_argument(
        "--qualified-reference-label",
        default="qualified_python_mcmc_fixed",
        help="held-out metric label to use when --reference-parity-metrics passes",
    )
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()
    _validate_args(parser, args)
    reference_qualification = _load_reference_qualification(
        args.reference_parity_metrics
    )
    reference_model = (
        args.qualified_reference_label
        if reference_qualification is not None
        else "mcmc_fixed"
    )

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
    predictive_mean_validation_datasets = []
    if args.predictive_mean_calibration != "none":
        predictive_mean_validation_datasets = _simulate_corpus(
            count=args.predictive_mean_calibration_validation_datasets,
            seed=args.seed + 150_000,
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
    scalar_calibration = fit_beta_scale_calibration(
        calibration_posterior,
        calibration_data.Beta,
        distribution="probit",
        predictive_X=calibration_data.X,
        predictive_Y=calibration_data.Y,
        predictive_seed=args.seed + 250_000,
    )
    coefficient_calibration = scalar_calibration
    if args.coefficient_calibration in {"conditional", "external_monotone"}:
        coefficient_calibration = fit_conditional_beta_scale_calibration(
            calibration_posterior,
            calibration_data.Beta,
            X=calibration_data.X,
            Y=calibration_data.Y,
            distribution="probit",
            coefficient_names=("Intercept", "TMG"),
            baseline_calibration=scalar_calibration,
            regularization=args.conditional_calibration_regularization,
            epochs=args.conditional_calibration_epochs,
            learning_rate=args.conditional_calibration_learning_rate,
            rank_penalty_weight=args.conditional_calibration_rank_penalty_weight,
        )
        if args.coefficient_calibration == "external_monotone":
            external_batches = _external_monotone_batches(
                engine=engine,
                count=args.external_monotone_datasets,
                n_sites=len(train_Y),
                n_species=train_Y.shape[1],
                tmg=train_X["TMG"].to_numpy(dtype=float),
                species_names=species_names,
                seed=args.seed + 350_000,
            )
            coefficient_calibration = fit_external_context_monotone_calibration(
                coefficient_calibration,
                calibration_posterior,
                calibration_data.Beta,
                X=calibration_data.X,
                Y=calibration_data.Y,
                distribution="probit",
                coefficient_names=("Intercept", "TMG"),
                ood_validation_batches=external_batches,
                max_external_multiplier=args.external_monotone_max_multiplier,
                min_mean_ood_gain=args.external_monotone_min_ood_gain,
                min_combined_shift_gain=args.external_monotone_min_combined_gain,
            )

    predictive_mean_calibration = None
    predictive_mean_selector = None
    predictive_mean_selector_decision = None
    active_predictive_mean_calibration = None
    if args.predictive_mean_calibration != "none":
        mean_validation_data = fixed_shape_training_data(
            predictive_mean_validation_datasets
        )
        mean_validation_posterior = engine.predict_beta_posterior(
            mean_validation_data
        )
        if (
            args.predictive_mean_calibration
            == "probit_source_transfer_response_affine"
        ):
            source_calibration = fit_beta_response_mean_calibration(
                calibration_posterior,
                calibration_X=calibration_data.X,
                calibration_Y=calibration_data.Y,
                validation_posterior=mean_validation_posterior,
                validation_X=mean_validation_data.X,
                validation_Y=mean_validation_data.Y,
                distribution="probit",
                method="probit_response_affine",
                max_validation_brier_ratio=(
                    args.predictive_mean_calibration_max_brier_ratio
                ),
                max_validation_log_loss_ratio=(
                    args.predictive_mean_calibration_max_log_loss_ratio
                ),
                min_validation_score_improvement=(
                    args.predictive_mean_source_min_improvement
                ),
            )
            transfer_calibration_batches = _predictive_mean_transfer_batches(
                engine=engine,
                count=args.predictive_mean_calibration_validation_datasets,
                n_sites=len(train_Y),
                n_species=train_Y.shape[1],
                tmg=train_X["TMG"].to_numpy(dtype=float),
                species_names=species_names,
                seed=args.seed + 450_000,
            )
            transfer_validation_batches = _predictive_mean_transfer_batches(
                engine=engine,
                count=args.predictive_mean_calibration_validation_datasets,
                n_sites=len(train_Y),
                n_species=train_Y.shape[1],
                tmg=train_X["TMG"].to_numpy(dtype=float),
                species_names=species_names,
                seed=args.seed + 550_000,
            )
            transfer_calibration = fit_beta_transfer_response_branch_calibration(
                transfer_calibration_batches,
                validation_batches=transfer_validation_batches,
                distribution="probit",
                max_validation_brier_ratio=(
                    args.predictive_mean_calibration_max_brier_ratio
                ),
                max_validation_log_loss_ratio=(
                    args.predictive_mean_calibration_max_log_loss_ratio
                ),
                min_validation_score_improvement=(
                    args.predictive_mean_transfer_branch_min_improvement
                ),
            )
            predictive_mean_calibration = source_calibration
            predictive_mean_selector = (
                independent_source_transfer_predictive_mean_selector_metadata(
                    source_calibration,
                    transfer_calibration,
                )
            )
            (
                active_predictive_mean_calibration,
                predictive_mean_selector_decision,
            ) = select_predictive_mean_calibration_for_context(
                predictive_mean_selector,
                context="whittaker",
                distribution="probit",
                n_covariates=2,
                n_species=train_Y.shape[1],
            )
        elif args.predictive_mean_calibration == "probit_transfer_response_affine":
            transfer_validation_batches = _predictive_mean_transfer_batches(
                engine=engine,
                count=args.predictive_mean_calibration_validation_datasets,
                n_sites=len(train_Y),
                n_species=train_Y.shape[1],
                tmg=train_X["TMG"].to_numpy(dtype=float),
                species_names=species_names,
                seed=args.seed + 450_000,
            )
            predictive_mean_calibration = (
                fit_beta_transfer_response_mean_calibration(
                    calibration_posterior,
                    calibration_X=calibration_data.X,
                    calibration_Y=calibration_data.Y,
                    source_validation_posterior=mean_validation_posterior,
                    source_validation_X=mean_validation_data.X,
                    source_validation_Y=mean_validation_data.Y,
                    transfer_validation_batches=transfer_validation_batches,
                    distribution="probit",
                    method=args.predictive_mean_calibration,
                    max_source_validation_brier_ratio=(
                        args.predictive_mean_calibration_max_brier_ratio
                    ),
                    max_source_validation_log_loss_ratio=(
                        args.predictive_mean_calibration_max_log_loss_ratio
                    ),
                    max_transfer_validation_brier_ratio=(
                        args.predictive_mean_calibration_max_brier_ratio
                    ),
                    max_transfer_validation_log_loss_ratio=(
                        args.predictive_mean_calibration_max_log_loss_ratio
                    ),
                    min_transfer_validation_score_improvement=(
                        args.predictive_mean_calibration_min_improvement
                    ),
                )
            )
        else:
            predictive_mean_calibration = fit_beta_response_mean_calibration(
                calibration_posterior,
                calibration_X=calibration_data.X,
                calibration_Y=calibration_data.Y,
                validation_posterior=mean_validation_posterior,
                validation_X=mean_validation_data.X,
                validation_Y=mean_validation_data.Y,
                distribution="probit",
                method=args.predictive_mean_calibration,
                max_validation_brier_ratio=(
                    args.predictive_mean_calibration_max_brier_ratio
                ),
                max_validation_log_loss_ratio=(
                    args.predictive_mean_calibration_max_log_loss_ratio
                ),
                min_validation_score_improvement=(
                    args.predictive_mean_calibration_min_improvement
                ),
            )
        if (
            args.predictive_mean_calibration
            != "probit_source_transfer_response_affine"
        ):
            active_predictive_mean_calibration = predictive_mean_calibration
        if (
            args.predictive_mean_calibration
            != "probit_source_transfer_response_affine"
            and args.predictive_mean_selection_policy == "domain_conditional"
        ):
            predictive_mean_selector = (
                domain_conditional_predictive_mean_selector_metadata(
                    predictive_mean_calibration,
                    min_transfer_validation_brier_gain=(
                        args.predictive_mean_transfer_min_brier_gain
                    ),
                    min_transfer_validation_log_loss_gain=(
                        args.predictive_mean_transfer_min_log_loss_gain
                    ),
                    max_transfer_slope_delta=(
                        args.predictive_mean_transfer_max_slope_delta
                    ),
                    max_transfer_abs_intercept=(
                        args.predictive_mean_transfer_max_abs_intercept
                    ),
                )
            )
            (
                active_predictive_mean_calibration,
                predictive_mean_selector_decision,
            ) = select_predictive_mean_calibration_for_context(
                predictive_mean_selector,
                context="whittaker",
                distribution="probit",
                n_covariates=2,
                n_species=train_Y.shape[1],
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
    coefficient_calibrated = _apply_coefficient_calibration(
        uncalibrated,
        coefficient_calibration,
        X=np.asarray(real_input["X"], dtype=np.float32)[None, :, :],
        Y=np.asarray(real_input["Y"], dtype=np.float32)[None, :, :],
        distribution="probit",
        coefficient_names=("Intercept", "TMG"),
    )
    predictive_only = apply_beta_predictive_calibration(
        uncalibrated,
        scalar_calibration,
        distribution="probit",
    )
    predictive_final = predictive_only
    if active_predictive_mean_calibration is not None:
        predictive_final = apply_beta_predictive_mean_calibration(
            predictive_only,
            active_predictive_mean_calibration,
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
        calibration=coefficient_calibration,
    )
    predictive_scale_path = args.output / "neural_predictive_distribution.h5"
    if predictive_mean_calibration is not None:
        predictive_scale_path = args.output / "neural_predictive_distribution_scale_only.h5"
    predictive_scale_path = write_beta_posterior_hdf5(
        predictive_only,
        predictive_scale_path,
        covariate_names=["Intercept", "TMG"],
        species_names=species_names,
        distribution="probit",
        formula="~ TMG",
        chains=args.neural_chains,
        draws=args.neural_draws,
        seed=args.seed + 3,
        metadata=common_metadata | {"artifact_role": "predictive_only"},
        calibration=scalar_calibration,
    )
    predictive_path = predictive_scale_path
    if predictive_mean_calibration is not None:
        predictive_path = write_beta_posterior_hdf5(
            predictive_final,
            args.output / "neural_predictive_distribution.h5",
            covariate_names=["Intercept", "TMG"],
            species_names=species_names,
            distribution="probit",
            formula="~ TMG",
            chains=args.neural_chains,
            draws=args.neural_draws,
            seed=(
                args.seed + 3
                if active_predictive_mean_calibration is None
                else args.seed + 4
            ),
            metadata=common_metadata
            | {
                "artifact_role": "predictive_only",
                "predictive_mean_calibration": (
                    predictive_mean_calibration.to_metadata()
                ),
                "active_predictive_mean_calibration": (
                    None
                    if active_predictive_mean_calibration is None
                    else active_predictive_mean_calibration.to_metadata()
                ),
                "predictive_mean_selector": predictive_mean_selector,
                "predictive_mean_selector_decision": (
                    predictive_mean_selector_decision
                ),
            },
            calibration=scalar_calibration,
        )
    neural_seconds = time.perf_counter() - neural_start

    sbc_rows = _sbc_rows(
        engine=engine,
        calibration=coefficient_calibration,
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
    variants = [
        ("uncalibrated", uncalibrated_path, uncalibrated_path),
        ("coefficient_calibrated", coefficient_path, coefficient_path),
        ("predictive_only_calibrated", coefficient_path, predictive_scale_path),
    ]
    final_predictive_model = "neural_predictive_only_calibrated"
    if predictive_mean_calibration is not None:
        variants.append(
            ("predictive_mean_calibrated", coefficient_path, predictive_path)
        )
        final_predictive_model = "neural_predictive_mean_calibrated"
    for variant, coefficient_variant_path, predictive_variant_path in variants:
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
            model=reference_model,
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
        _metric_row(heldout, final_predictive_model),
        _metric_row(heldout, reference_model),
    )
    coefficient_sbc = next(
        row
        for row in sbc_rows
        if row["posterior_variant"] == "coefficient_calibrated"
        and row["sbc_stratum_kind"] == "overall"
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
    acceptance["final_predictive_model"] = final_predictive_model
    acceptance["predictive_mean_calibration_selected"] = (
        None
        if predictive_mean_calibration is None
        else active_predictive_mean_calibration is not None
    )
    acceptance["predictive_mean_selection_policy"] = (
        args.predictive_mean_selection_policy
    )
    acceptance["predictive_mean_selector_decision"] = (
        predictive_mean_selector_decision
    )
    acceptance["reference_model"] = reference_model
    acceptance["reference_parity_qualified"] = reference_qualification is not None
    heldout.to_csv(args.output / "whittaker_heldout_metrics.csv", index=False)
    (args.output / "whittaker_acceptance.json").write_text(
        json.dumps(acceptance, indent=2) + "\n",
        encoding="utf-8",
    )
    report = _render_report(
        heldout=heldout,
        calibration=coefficient_calibration.to_metadata(),
        predictive_calibration=scalar_calibration.to_metadata(),
        predictive_mean_calibration=(
            None
            if active_predictive_mean_calibration is None
            else active_predictive_mean_calibration.to_metadata()
        ),
        predictive_mean_selector=predictive_mean_selector,
        predictive_mean_selector_decision=predictive_mean_selector_decision,
        sbc_rows=sbc_rows,
        train_Y=train_Y,
        train_X=train_X,
        test_X=test_X,
        training_seconds=training_seconds,
        neural_seconds=neural_seconds,
        mcmc_seconds=mcmc_seconds,
        acceptance=acceptance,
        reference_qualification=reference_qualification,
    )
    (args.output / "whittaker_neural_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "status": "completed",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args)
        | {
            "source": str(args.source),
            "output": str(args.output),
            "reference_parity_metrics": (
                str(args.reference_parity_metrics)
                if args.reference_parity_metrics is not None
                else None
            ),
        },
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
        "coefficient_calibration": args.coefficient_calibration,
        "coefficient_calibration_metadata": coefficient_calibration.to_metadata(),
        "predictive_calibration_metadata": scalar_calibration.to_metadata(),
        "predictive_mean_calibration_metadata": (
            None
            if predictive_mean_calibration is None
            else predictive_mean_calibration.to_metadata()
        ),
        "active_predictive_mean_calibration_metadata": (
            None
            if active_predictive_mean_calibration is None
            else active_predictive_mean_calibration.to_metadata()
        ),
        "predictive_mean_selector_metadata": predictive_mean_selector,
        "predictive_mean_selector_decision": predictive_mean_selector_decision,
        "reference_model": reference_model,
        "reference_qualification": reference_qualification,
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
    if args.conditional_calibration_epochs <= 0:
        parser.error("--conditional-calibration-epochs must be positive")
    if args.conditional_calibration_learning_rate <= 0.0:
        parser.error("--conditional-calibration-learning-rate must be positive")
    if args.conditional_calibration_regularization < 0.0:
        parser.error("--conditional-calibration-regularization must be non-negative")
    if args.conditional_calibration_rank_penalty_weight < 0.0:
        parser.error("--conditional-calibration-rank-penalty-weight must be non-negative")
    if args.external_monotone_datasets < 0:
        parser.error("--external-monotone-datasets must be non-negative")
    if (
        args.coefficient_calibration == "external_monotone"
        and args.external_monotone_datasets <= 0
    ):
        parser.error(
            "--coefficient-calibration external_monotone requires "
            "--external-monotone-datasets"
        )
    if args.external_monotone_max_multiplier < 1.0:
        parser.error("--external-monotone-max-multiplier must be at least one")
    if args.external_monotone_min_ood_gain < 0.0:
        parser.error("--external-monotone-min-ood-gain must be non-negative")
    if args.external_monotone_min_combined_gain < 0.0:
        parser.error("--external-monotone-min-combined-gain must be non-negative")
    if args.predictive_mean_calibration_validation_datasets < 0:
        parser.error(
            "--predictive-mean-calibration-validation-datasets must be non-negative"
        )
    if (
        args.predictive_mean_calibration != "none"
        and args.predictive_mean_calibration_validation_datasets <= 0
    ):
        parser.error(
            "--predictive-mean-calibration requires "
            "--predictive-mean-calibration-validation-datasets"
        )
    if args.predictive_mean_calibration_max_brier_ratio < 1.0:
        parser.error(
            "--predictive-mean-calibration-max-brier-ratio must be at least one"
        )
    if args.predictive_mean_calibration_max_log_loss_ratio < 1.0:
        parser.error(
            "--predictive-mean-calibration-max-log-loss-ratio must be at least one"
        )
    if args.predictive_mean_calibration_min_improvement < 0.0:
        parser.error(
            "--predictive-mean-calibration-min-improvement must be non-negative"
        )
    if (
        args.predictive_mean_calibration != "none"
        and args.predictive_mean_calibration_min_improvement == 0.0
    ):
        parser.error(
            "--predictive-mean-calibration requires a positive "
            "--predictive-mean-calibration-min-improvement"
        )
    if args.predictive_mean_source_min_improvement < 0.0:
        parser.error("--predictive-mean-source-min-improvement must be non-negative")
    if args.predictive_mean_transfer_branch_min_improvement < 0.0:
        parser.error(
            "--predictive-mean-transfer-branch-min-improvement must be non-negative"
        )
    for name in [
        "predictive_mean_transfer_min_brier_gain",
        "predictive_mean_transfer_min_log_loss_gain",
        "predictive_mean_transfer_max_slope_delta",
        "predictive_mean_transfer_max_abs_intercept",
    ]:
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if (
        args.reference_parity_metrics is not None
        and not args.reference_parity_metrics.exists()
    ):
        parser.error(
            f"--reference-parity-metrics does not exist: {args.reference_parity_metrics}"
        )


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
    tmg_shift: float = 0.0,
    tmg_noise: float = 0.03,
    intercept_shift: float = 0.0,
    slope_scale: float = 1.0,
    domain: str = "whittaker_shape_matched",
) -> FixedEffectDataset:
    rng = np.random.default_rng(seed)
    tmg_values = np.asarray(tmg, dtype=float).copy()
    if tmg_values.shape != (n_sites,):
        raise ValueError(
            f"TMG design shape {tmg_values.shape} does not match n_sites={n_sites}"
        )
    tmg_values = tmg_values + float(tmg_shift)
    tmg_values += rng.normal(scale=float(tmg_noise), size=n_sites)
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
    intercept = intercept + float(intercept_shift)
    slope = rng.normal(0.0, 0.8 * float(slope_scale), size=n_species)
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
            "simulation_domain": domain,
            "formula": "~ TMG",
            "intercept_prior": "rare_species_mixture",
            "slope_mean": 0.0,
            "slope_sd": 0.8 * float(slope_scale),
            "tmg_shift": float(tmg_shift),
            "intercept_shift": float(intercept_shift),
        },
    )


def _external_monotone_batches(
    *,
    engine: NeuralHmscInference,
    count: int,
    n_sites: int,
    n_species: int,
    tmg: np.ndarray,
    species_names: list[str],
    seed: int,
) -> list[ConditionalBetaOODCalibrationBatch]:
    batches = []
    regimes = {
        "covariate_shift": {
            "tmg_shift": 0.65,
            "tmg_noise": 0.08,
            "intercept_shift": 0.0,
            "slope_scale": 1.0,
        },
        "effect_size_shift": {
            "tmg_shift": 0.0,
            "tmg_noise": 0.03,
            "intercept_shift": 0.0,
            "slope_scale": 1.75,
        },
        "combined_shift": {
            "tmg_shift": 0.65,
            "tmg_noise": 0.08,
            "intercept_shift": -0.35,
            "slope_scale": 1.75,
        },
    }
    for regime_index, (regime, kwargs) in enumerate(regimes.items()):
        datasets = [
            _simulate_dataset(
                n_sites=n_sites,
                n_species=n_species,
                tmg=tmg,
                species_names=species_names,
                seed=seed + 10_000 * regime_index + idx,
                domain=f"whittaker_{regime}",
                **kwargs,
            )
            for idx in range(count)
        ]
        data = fixed_shape_training_data(datasets)
        batches.append(
            ConditionalBetaOODCalibrationBatch(
                posterior=engine.predict_beta_posterior(data),
                beta_true=data.Beta,
                X=data.X,
                Y=data.Y,
                label=regime,
            )
        )
    return batches


def _predictive_mean_transfer_batches(
    *,
    engine: NeuralHmscInference,
    count: int,
    n_sites: int,
    n_species: int,
    tmg: np.ndarray,
    species_names: list[str],
    seed: int,
) -> list[BetaResponseCalibrationBatch]:
    regimes = {
        "covariate_shift": {
            "tmg_shift": 0.65,
            "tmg_noise": 0.08,
            "intercept_shift": 0.0,
            "slope_scale": 1.0,
        },
        "effect_size_shift": {
            "tmg_shift": 0.0,
            "tmg_noise": 0.03,
            "intercept_shift": 0.0,
            "slope_scale": 1.75,
        },
        "combined_shift": {
            "tmg_shift": 0.65,
            "tmg_noise": 0.08,
            "intercept_shift": -0.35,
            "slope_scale": 1.75,
        },
    }
    batches = []
    base_count, remainder = divmod(count, len(regimes))
    for regime_index, (regime, kwargs) in enumerate(regimes.items()):
        regime_count = base_count + int(regime_index < remainder)
        if regime_count <= 0:
            continue
        datasets = [
            _simulate_dataset(
                n_sites=n_sites,
                n_species=n_species,
                tmg=tmg,
                species_names=species_names,
                seed=seed + 10_000 * regime_index + idx,
                domain=f"whittaker_predictive_mean_{regime}",
                **kwargs,
            )
            for idx in range(regime_count)
        ]
        data = fixed_shape_training_data(datasets)
        batches.append(
            BetaResponseCalibrationBatch(
                posterior=engine.predict_beta_posterior(data),
                X=data.X,
                Y=data.Y,
                label=f"transfer_validation:{regime}",
            )
        )
    return batches


def _apply_coefficient_calibration(
    posterior,
    calibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str,
    coefficient_names: tuple[str, ...],
):
    if isinstance(calibration, ConditionalBetaScaleCalibration):
        return apply_conditional_beta_scale_calibration(
            posterior,
            calibration,
            X=X,
            Y=Y,
            distribution=distribution,
            coefficient_names=coefficient_names,
        )
    return apply_beta_scale_calibration(
        posterior,
        calibration,
        distribution=distribution,
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
    calibrated = _apply_coefficient_calibration(
        uncalibrated,
        calibration,
        X=data.X,
        Y=data.Y,
        distribution="probit",
        coefficient_names=("Intercept", "TMG"),
    )
    rows = []
    for idx, (variant, posterior) in enumerate(
        [("uncalibrated", uncalibrated), ("coefficient_calibrated", calibrated)]
    ):
        samples = sample_beta_posterior(posterior, draws=draws, seed=seed + idx).numpy()
        samples = np.transpose(samples, (1, 0, 2, 3))
        diagnostics_by_stratum = beta_sbc_stratified_diagnostics(
            samples,
            data.Beta,
            X=data.X,
            Y=data.Y,
            distribution="probit",
            covariate_names=datasets[0].truth_beta.index,
            n_bins=10,
            seed=seed + 10 + idx,
        )
        for stratum in diagnostics_by_stratum:
            row: dict[str, object] = {
                "distribution": "probit",
                "simulation_domain": "whittaker_shape_matched",
                "ood_regime": None,
                "posterior_variant": variant,
            }
            row.update(stratum.report_fields())
            rows.append(row)
    overall = {
        str(row["posterior_variant"]): row
        for row in rows
        if row["sbc_stratum_kind"] == "overall"
    }
    overall["coefficient_calibrated"].update(
        sbc_calibration_acceptance(
            overall["uncalibrated"],
            overall["coefficient_calibrated"],
        )
    )
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
        "predictive_rmse": float(np.sqrt(np.mean(np.square(probability - observed)))),
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


def _load_reference_qualification(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("parity_passed", False)):
        raise ValueError(f"reference parity metrics did not pass: {path}")
    boundary = payload.get("boundary_checks", {})
    gates = payload.get("acceptance_gates", {})
    beta = payload.get("beta_compare", {})
    gamma = payload.get("gamma_compare", {})
    deltas = payload.get("metric_deltas_python_native_minus_r_bridge", {})
    return {
        "metrics_path": str(path),
        "parity_passed": True,
        "source": payload.get("source") or payload.get("config") or payload.get("project"),
        "settings": payload.get("settings", {}),
        "posterior_gates": payload.get("posterior_gates", "strict"),
        "boundary_arrays_passed": bool(
            boundary and all(bool(check.get("passed", False)) for check in boundary.values())
        ),
        "acceptance_gates_passed": bool(
            gates and all(bool(gate.get("passed", False)) for gate in gates.values())
        ),
        "beta_mean_correlation": _optional_float(beta.get("mean_correlation")),
        "gamma_mean_correlation": _optional_float(gamma.get("mean_correlation")),
        "random_level_association_correlation": _random_level_association_correlation(payload),
        "metric_deltas_python_native_minus_r_bridge": {
            str(key): _optional_float(value) for key, value in deltas.items()
        },
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _random_level_association_correlation(payload: dict[str, object]) -> float | None:
    random_compare = payload.get("random_level_compare", {})
    if not isinstance(random_compare, dict):
        return None
    levels = random_compare.get("levels", [])
    if not isinstance(levels, list) or not levels:
        return None
    values = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        assoc = level.get("association_compare", {})
        if isinstance(assoc, dict):
            value = _optional_float(assoc.get("mean_correlation"))
            if value is not None:
                values.append(value)
    return min(values) if values else None


def _reference_qualification_lines(
    reference_qualification: dict[str, object] | None,
) -> list[str]:
    if reference_qualification is None:
        return [
            "## Reference Qualification",
            "",
            "No direct R/Python parity metrics were attached for this run. The reference row is a Python-native MCMC comparator only.",
            "",
        ]
    lines = [
        "## Reference Qualification",
        "",
        "The reference row is a Python-native MCMC comparator qualified by a passed direct R/Python HMSC-HPC parity workflow.",
        "",
        f"Parity metrics: `{reference_qualification['metrics_path']}`",
        f"Parity source/config: `{reference_qualification.get('source')}`",
        f"Boundary arrays passed: {bool(reference_qualification['boundary_arrays_passed'])}",
        f"Acceptance gates passed: {bool(reference_qualification['acceptance_gates_passed'])}",
        f"Posterior gate mode: `{reference_qualification.get('posterior_gates')}`",
    ]
    if reference_qualification.get("beta_mean_correlation") is not None:
        lines.append(
            "Beta diagnostic correlation: "
            f"{float(reference_qualification['beta_mean_correlation']):.6f}"
        )
    if reference_qualification.get("gamma_mean_correlation") is not None:
        lines.append(
            "Gamma diagnostic correlation: "
            f"{float(reference_qualification['gamma_mean_correlation']):.6f}"
        )
    if reference_qualification.get("random_level_association_correlation") is not None:
        lines.append(
            "Random-level association diagnostic correlation: "
            f"{float(reference_qualification['random_level_association_correlation']):.6f}"
        )
    return lines + [
        "",
        "This qualifies the Python-only HMSC reference path against the original R+Python boundary; it does not turn neural predictive transfer into exact HMSC posterior equivalence.",
        "",
    ]


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
    predictive_calibration: dict[str, object],
    predictive_mean_calibration: dict[str, object] | None,
    predictive_mean_selector: dict[str, object] | None,
    predictive_mean_selector_decision: dict[str, object] | None,
    sbc_rows: list[dict[str, object]],
    train_Y: pd.DataFrame,
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    training_seconds: float,
    neural_seconds: float,
    mcmc_seconds: float,
    acceptance: dict[str, object],
    reference_qualification: dict[str, object] | None,
) -> str:
    sbc = pd.DataFrame(sbc_rows)
    sbc_summary = sbc[
        [
            "posterior_variant",
            "sbc_stratum_kind",
            "sbc_stratum_label",
            "sbc_n_ranks",
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
            *_reference_qualification_lines(reference_qualification),
            "## Shape-Matched Simulation Calibration",
            "",
            sbc_summary.to_string(index=False),
            "",
            f"Coefficient calibration method: {calibration['method']}",
            "Coefficient-posterior normalization multiplier: "
            f"{float(calibration['scale_multiplier']):.6f}",
            "Coefficient-posterior global scalar baseline: "
            f"{float(calibration.get('global_scale_multiplier', calibration['scale_multiplier'])):.6f}",
            "Predictive-only scale multiplier: "
            f"{float(predictive_calibration['predictive_scale_multiplier']):.6f}",
            f"Predictive method: {predictive_calibration['predictive_method']}",
            "Predictive-only mean calibration: "
            f"{_predictive_mean_calibration_line(predictive_mean_calibration)}",
            "Predictive mean selector: "
            f"{_predictive_mean_selector_line(predictive_mean_selector_decision)}",
            "External monotone selected shrinkage: "
            f"{calibration.get('external_context_monotone', {}).get('selected_shrinkage', 'n/a')}",
            "",
            "## Acceptance Gates",
            "",
            f"Coefficient SBC acceptance: {bool(acceptance['sbc_acceptance_passed'])}",
            f"Held-out predictive acceptance: {bool(acceptance['predictive_acceptance_passed'])}",
            f"Final predictive model: `{acceptance.get('final_predictive_model', 'n/a')}`",
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


def _predictive_mean_calibration_line(
    calibration: dict[str, object] | None,
) -> str:
    if calibration is None:
        return "`none`"
    selected = bool(calibration.get("selected", False))
    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    response = calibration.get("response_validation", {})
    if not isinstance(response, dict):
        response = {}
    brier_ratio = response.get("brier_ratio")
    log_loss_ratio = response.get("log_loss_ratio")
    return (
        f"`{calibration.get('method', 'unknown')}` selected={selected}, "
        f"slope={slope:.4f}, intercept={intercept:.4f}, "
        f"validation_brier_ratio={_format_optional_float(brier_ratio)}, "
        f"validation_log_loss_ratio={_format_optional_float(log_loss_ratio)}"
    )


def _format_optional_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _predictive_mean_selector_line(decision: dict[str, object] | None) -> str:
    if decision is None:
        return "`none`"
    return (
        f"`{decision.get('method', 'unknown')}` context={decision.get('context')}, "
        f"action={decision.get('action')}, reason={decision.get('reason')}"
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
