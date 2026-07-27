"""Validate a frozen Whittaker Neural-HMSC artifact on independent plant data."""

from __future__ import annotations

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.generate_neural_hmsc_big_spatial_transfer import generate_project
from examples.run_neural_hmsc_whittaker import (
    _apply_coefficient_calibration,
    _heldout_metrics,
    _load_reference_qualification,
    _metric_row,
    _reference_qualification_lines,
    _simulate_dataset,
)
from pyhmsc.model import HmscModel
from pyhmsc.neural.benchmark import (
    compare_beta_posteriors,
    occurrence_predictive_acceptance,
    write_benchmark_report,
)
from pyhmsc.neural.calibration import (
    BetaScaleCalibration,
    apply_beta_predictive_calibration,
)
from pyhmsc.neural.conditional_calibration import ConditionalBetaScaleCalibration
from pyhmsc.neural.inference import NeuralHmscInference
from pyhmsc.neural.mean_calibration import (
    BetaPredictiveMeanCalibration,
    BetaResponseCalibrationBatch,
    apply_beta_predictive_mean_calibration,
    evaluate_beta_target_context_gate,
    select_predictive_mean_calibration_for_context,
    target_context_conditioned_source_transfer_selector_metadata,
)
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import fixed_shape_training_data
from pyhmsc.posterior import HmscFit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument(
        "--source-matrix",
        type=Path,
        default=Path("examples/big_spatial/data"),
    )
    parser.add_argument(
        "--source-project",
        type=Path,
        default=Path("examples/projects/big_spatial_plants_validation"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neural-chains", type=int, default=4)
    parser.add_argument("--neural-draws", type=int, default=1000)
    parser.add_argument("--mcmc-chains", type=int, default=2)
    parser.add_argument("--mcmc-samples", type=int, default=1000)
    parser.add_argument("--mcmc-transient", type=int, default=500)
    parser.add_argument("--mcmc-thin", type=int, default=5)
    parser.add_argument("--mcmc-verbose", type=int, default=500)
    parser.add_argument(
        "--reference-parity-metrics",
        type=Path,
        help=(
            "direct R/Python parity metrics JSON qualifying the Python-native "
            "HMSC comparator path against the original R+Python HMSC-HPC boundary"
        ),
    )
    parser.add_argument(
        "--qualified-reference-label",
        default="qualified_python_mcmc_fixed",
        help="held-out metric label to use when --reference-parity-metrics passes",
    )
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--target-context-gate",
        choices=("none", "independent_simulation"),
        default="none",
        help="optional unlabeled-target simulation gate for a transfer branch",
    )
    parser.add_argument("--target-context-gate-datasets", type=int, default=12)
    parser.add_argument(
        "--target-context-gate-max-brier-ratio", type=float, default=1.0
    )
    parser.add_argument(
        "--target-context-gate-max-log-loss-ratio", type=float, default=1.0
    )
    parser.add_argument(
        "--target-context-gate-min-improvement", type=float, default=0.0001
    )
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
    project = args.output / "big_spatial_transfer_project"
    generate_project(args.source_matrix, args.source_project, project)
    train_Y = pd.read_csv(project / "data/train/Y.csv", index_col=0)
    train_X = pd.read_csv(project / "data/train/X.csv", index_col=0)
    test_X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    test_Y_path = project / "data/test/Y.csv"
    species_names = [str(name) for name in train_Y.columns]

    checkpoint = args.frozen_run / "neural_checkpoint"
    coefficient_source = args.frozen_run / "neural_posterior.h5"
    predictive_source = args.frozen_run / "neural_predictive_distribution.h5"
    source_acceptance_path = args.frozen_run / "whittaker_acceptance.json"
    engine = NeuralHmscInference.load(checkpoint)
    expected_dimensions = {"n_sites": 40, "n_covariates": 2, "n_species": 75}
    if engine.dimensions != expected_dimensions:
        raise ValueError(
            f"frozen checkpoint dimensions {engine.dimensions} do not match transfer projection "
            f"{expected_dimensions}"
        )
    if engine.distribution != "probit":
        raise ValueError("frozen transfer checkpoint must use the probit distribution")
    if train_Y.shape != (40, 75) or train_X.shape != (40, 1):
        raise ValueError(
            "generated transfer training data do not match the frozen checkpoint"
        )

    source_fit = HmscFit.from_file(coefficient_source)
    calibration_metadata = source_fit.metadata.get("calibration")
    if not isinstance(calibration_metadata, dict):
        raise ValueError("frozen coefficient posterior has no calibration metadata")
    coefficient_calibration = _source_coefficient_calibration(calibration_metadata)
    predictive_source_fit = HmscFit.from_file(predictive_source)
    predictive_calibration_metadata = predictive_source_fit.metadata.get("calibration")
    if not isinstance(predictive_calibration_metadata, dict):
        raise ValueError("frozen predictive posterior has no calibration metadata")
    predictive_calibration = BetaScaleCalibration.from_metadata(
        predictive_calibration_metadata
    )
    if predictive_calibration.predictive_scale_multiplier is None:
        raise ValueError("frozen predictive calibration has no predictive-only multiplier")
    predictive_mean_calibration_metadata = predictive_source_fit.metadata.get(
        "predictive_mean_calibration"
    )
    predictive_mean_calibration = None
    if isinstance(predictive_mean_calibration_metadata, dict):
        predictive_mean_calibration = BetaPredictiveMeanCalibration.from_metadata(
            predictive_mean_calibration_metadata
        )
    frozen_predictive_mean_selector = predictive_source_fit.metadata.get(
        "predictive_mean_selector"
    )
    predictive_mean_selector = frozen_predictive_mean_selector
    target_context_gate = None
    if args.target_context_gate == "independent_simulation":
        (
            predictive_mean_selector,
            target_context_gate,
        ) = _target_context_conditioned_selector(
            engine=engine,
            selector_metadata=frozen_predictive_mean_selector,
            train_X=train_X,
            test_X=test_X,
            species_names=species_names,
            seed=args.seed,
            datasets=args.target_context_gate_datasets,
            max_brier_ratio=args.target_context_gate_max_brier_ratio,
            max_log_loss_ratio=args.target_context_gate_max_log_loss_ratio,
            min_score_improvement=args.target_context_gate_min_improvement,
        )
    predictive_mean_selector_decision = None
    active_predictive_mean_calibration = predictive_mean_calibration
    if isinstance(predictive_mean_selector, dict):
        (
            active_predictive_mean_calibration,
            predictive_mean_selector_decision,
        ) = select_predictive_mean_calibration_for_context(
            predictive_mean_selector,
            context="big_spatial_transfer",
            distribution="probit",
            n_covariates=2,
            n_species=train_Y.shape[1],
        )
    source_acceptance = json.loads(source_acceptance_path.read_text(encoding="utf-8"))
    if not bool(source_acceptance.get("qualification_acceptance_passed", False)):
        raise ValueError(
            "frozen source run did not pass its combined qualification gate"
        )
    # Target outcomes become available only after all predictive-mean selection.
    test_Y = pd.read_csv(test_Y_path, index_col=0)

    frozen_artifacts = {
        "checkpoint_sha256": _directory_sha256(checkpoint),
        "coefficient_source_sha256": _file_sha256(coefficient_source),
        "predictive_source_sha256": _file_sha256(predictive_source),
        "calibration_sha256": _json_sha256(predictive_calibration.to_metadata()),
        "coefficient_calibration_sha256": _json_sha256(
            coefficient_calibration.to_metadata()
        ),
        "predictive_calibration_sha256": _json_sha256(
            predictive_calibration.to_metadata()
        ),
        "predictive_mean_calibration_sha256": (
            None
            if predictive_mean_calibration is None
            else _json_sha256(predictive_mean_calibration.to_metadata())
        ),
        "predictive_mean_selector_sha256": (
            None
            if not isinstance(frozen_predictive_mean_selector, dict)
            else _json_sha256(frozen_predictive_mean_selector)
        ),
        "source_acceptance_sha256": _file_sha256(source_acceptance_path),
        "weights_updated": False,
        "calibration_updated": False,
        "target_context_selection_gate_evaluated": target_context_gate is not None,
    }
    (args.output / "frozen_artifact_manifest.json").write_text(
        json.dumps(frozen_artifacts, indent=2) + "\n",
        encoding="utf-8",
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
        predictive_calibration,
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
        "transfer_validation": {
            "dataset": "Big Spatial Plant community",
            "source_run": str(args.frozen_run),
            "weights_updated": False,
            "calibration_updated": False,
            "source_covariate": "Max_temp_smooth",
            "projected_covariate": "TMG",
            "target_context_gate": target_context_gate,
        }
    }
    uncalibrated_path = _write_posterior(
        uncalibrated,
        args.output / "neural_posterior_uncalibrated.h5",
        species_names,
        args,
        seed_offset=1,
        metadata=common_metadata
        | {"artifact_role": "coefficient_posterior_uncalibrated"},
    )
    coefficient_path = _write_posterior(
        coefficient_calibrated,
        args.output / "neural_posterior.h5",
        species_names,
        args,
        seed_offset=2,
        metadata=common_metadata | {"artifact_role": "coefficient_posterior"},
        calibration=coefficient_calibration,
    )
    predictive_scale_path = args.output / "neural_predictive_distribution.h5"
    if predictive_mean_calibration is not None:
        predictive_scale_path = args.output / "neural_predictive_distribution_scale_only.h5"
    predictive_scale_path = _write_posterior(
        predictive_only,
        predictive_scale_path,
        species_names,
        args,
        seed_offset=3,
        metadata=common_metadata | {"artifact_role": "predictive_only"},
        calibration=predictive_calibration,
    )
    predictive_path = predictive_scale_path
    if predictive_mean_calibration is not None:
        predictive_path = _write_posterior(
            predictive_final,
            args.output / "neural_predictive_distribution.h5",
            species_names,
            args,
            seed_offset=3 if active_predictive_mean_calibration is None else 4,
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
            calibration=predictive_calibration,
        )
    neural_seconds = time.perf_counter() - neural_start

    mcmc_model = HmscModel(Y=train_Y, X=train_X, x_formula="~ TMG", distr="probit")
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
        variants.append(("predictive_mean_calibrated", coefficient_path, predictive_path))
        final_predictive_model = "neural_predictive_mean_calibrated"
    for variant, coefficient_variant, predictive_variant in variants:
        coefficient_fit = HmscFit.from_file(coefficient_variant)
        predictive_fit = HmscFit.from_file(predictive_variant)
        row = compare_beta_posteriors(
            coefficient_fit,
            mcmc_fit,
            dataset="big_spatial_plants_transfer",
            distribution="probit",
            neural_seconds=neural_seconds,
            mcmc_seconds=mcmc_seconds,
            X=test_X,
            Y=test_Y,
            formula="~ TMG",
            neural_predictive_fit=predictive_fit,
        )
        row["posterior_variant"] = variant
        posterior_rows.append(row)
        heldout_rows.append(
            _heldout_metrics(
                model=f"neural_{variant}",
                fit=predictive_fit,
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
        stem="big_spatial_transfer_neural_mcmc_reference",
        title="Frozen Neural-HMSC Big Spatial Plant Transfer",
    )
    heldout = pd.DataFrame(heldout_rows)
    heldout_path = args.output / "big_spatial_transfer_heldout_metrics.csv"
    heldout.to_csv(heldout_path, index=False)
    predictive_acceptance = occurrence_predictive_acceptance(
        _metric_row(heldout, "neural_uncalibrated"),
        _metric_row(heldout, final_predictive_model),
        _metric_row(heldout, reference_model),
    )
    acceptance = {
        "source_sbc_acceptance_passed": bool(
            source_acceptance["sbc_acceptance_passed"]
        ),
        "source_qualification_acceptance_passed": bool(
            source_acceptance["qualification_acceptance_passed"]
        ),
        **predictive_acceptance,
    }
    acceptance["predictive_transfer_acceptance_passed"] = bool(
        acceptance["source_qualification_acceptance_passed"]
        and acceptance["predictive_acceptance_passed"]
    )
    acceptance["target_coefficient_calibration_assessable"] = False
    acceptance["target_coefficient_calibration_status"] = (
        "not_assessable_without_target_coefficient_truth"
    )
    acceptance["final_predictive_model"] = final_predictive_model
    acceptance["predictive_mean_calibration_selected"] = (
        None
        if predictive_mean_calibration is None
        else active_predictive_mean_calibration is not None
    )
    acceptance["predictive_mean_selector_decision"] = (
        predictive_mean_selector_decision
    )
    acceptance["target_context_gate"] = target_context_gate
    acceptance["reference_model"] = reference_model
    acceptance["reference_parity_qualified"] = reference_qualification is not None
    acceptance_path = args.output / "big_spatial_transfer_acceptance.json"
    acceptance_path.write_text(
        json.dumps(acceptance, indent=2) + "\n", encoding="utf-8"
    )

    report = _render_report(
        train_Y=train_Y,
        train_X=train_X,
        test_Y=test_Y,
        test_X=test_X,
        heldout=heldout,
        calibration=coefficient_calibration.to_metadata(),
        predictive_calibration=predictive_calibration.to_metadata(),
        predictive_mean_calibration=(
            None
            if active_predictive_mean_calibration is None
            else active_predictive_mean_calibration.to_metadata()
        ),
        predictive_mean_selector=predictive_mean_selector,
        predictive_mean_selector_decision=predictive_mean_selector_decision,
        acceptance=acceptance,
        frozen_artifacts=frozen_artifacts,
        neural_seconds=neural_seconds,
        mcmc_seconds=mcmc_seconds,
        reference_qualification=reference_qualification,
    )
    report_path = args.output / "big_spatial_transfer_report.md"
    report_path.write_text(report, encoding="utf-8")
    metadata = {
        "status": "completed",
        "args": vars(args)
        | {
            "frozen_run": str(args.frozen_run),
            "source_matrix": str(args.source_matrix),
            "source_project": str(args.source_project),
            "output": str(args.output),
            "reference_parity_metrics": (
                str(args.reference_parity_metrics)
                if args.reference_parity_metrics is not None
                else None
            ),
        },
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "frozen_artifacts": frozen_artifacts,
        "coefficient_calibration_metadata": coefficient_calibration.to_metadata(),
        "predictive_calibration_metadata": predictive_calibration.to_metadata(),
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
        "target_context_gate": target_context_gate,
        "acceptance": acceptance,
        "reference_model": reference_model,
        "reference_qualification": reference_qualification,
        "neural_inference_seconds": neural_seconds,
        "mcmc_seconds": mcmc_seconds,
        "reports": {
            "comparison_csv": str(comparison_paths.csv),
            "heldout_csv": str(heldout_path),
            "acceptance_json": str(acceptance_path),
            "report": str(report_path),
        },
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in [
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
    if args.target_context_gate_datasets <= 0:
        parser.error("--target-context-gate-datasets must be positive")
    if args.target_context_gate_max_brier_ratio < 1.0:
        parser.error("--target-context-gate-max-brier-ratio must be at least one")
    if args.target_context_gate_max_log_loss_ratio < 1.0:
        parser.error(
            "--target-context-gate-max-log-loss-ratio must be at least one"
        )
    if args.target_context_gate_min_improvement < 0.0:
        parser.error("--target-context-gate-min-improvement must be non-negative")
    if (
        args.reference_parity_metrics is not None
        and not args.reference_parity_metrics.exists()
    ):
        parser.error(
            f"--reference-parity-metrics does not exist: {args.reference_parity_metrics}"
        )


def _target_context_conditioned_selector(
    *,
    engine: NeuralHmscInference,
    selector_metadata: object,
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    species_names: list[str],
    seed: int,
    datasets: int,
    max_brier_ratio: float,
    max_log_loss_ratio: float,
    min_score_improvement: float,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(selector_metadata, dict):
        raise ValueError("target context gating requires frozen selector metadata")
    if selector_metadata.get("method") != "independent_source_transfer_affine_selector":
        raise ValueError(
            "target context gating requires the independent source/transfer method"
        )
    transfer_metadata = selector_metadata.get("transfer_branch")
    if not isinstance(transfer_metadata, dict):
        raise ValueError("frozen selector has no transfer branch metadata")
    transfer_calibration = BetaPredictiveMeanCalibration.from_metadata(
        transfer_metadata
    )
    n_sites = int(engine.dimensions["n_sites"])
    target_tmg = _target_context_covariate_support(
        train_X,
        test_X,
        n_sites=n_sites,
    )
    calibration_batches = _target_context_response_batches(
        engine=engine,
        count=datasets,
        tmg=target_tmg,
        species_names=species_names,
        seed=seed + 650_000,
        pool="calibration",
    )
    validation_batches = _target_context_response_batches(
        engine=engine,
        count=datasets,
        tmg=target_tmg,
        species_names=species_names,
        seed=seed + 750_000,
        pool="validation",
    )
    all_target_tmg = np.concatenate(
        [
            train_X["TMG"].to_numpy(dtype=float),
            test_X["TMG"].to_numpy(dtype=float),
        ]
    )
    context_metadata = {
        "dataset": "big_spatial_transfer",
        "conditioning": "unlabeled_target_covariate_support",
        "target_response_used": False,
        "target_heldout_response_used": False,
        "available_target_sites": int(all_target_tmg.size),
        "simulation_sites": n_sites,
        "community_species": len(species_names),
        "covariate": "TMG",
        "covariate_support": {
            "minimum": float(np.min(all_target_tmg)),
            "maximum": float(np.max(all_target_tmg)),
            "mean": float(np.mean(all_target_tmg)),
            "standard_deviation": float(np.std(all_target_tmg)),
        },
        "support_projection": "deterministic_midpoint_quantiles",
        "prevalence_prior": "rare_species_mixture_without_target_Y",
        "calibration_seed": int(seed + 650_000),
        "validation_seed": int(seed + 750_000),
        "datasets_per_pool": int(datasets),
    }
    gate = evaluate_beta_target_context_gate(
        transfer_calibration,
        calibration_batches,
        validation_batches=validation_batches,
        max_brier_ratio=max_brier_ratio,
        max_log_loss_ratio=max_log_loss_ratio,
        min_score_improvement=min_score_improvement,
        context_metadata=context_metadata,
    )
    selector = target_context_conditioned_source_transfer_selector_metadata(
        selector_metadata,
        gate,
        target_contexts=("big_spatial", "big_spatial_transfer"),
    )
    return selector, gate


def _target_context_covariate_support(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    *,
    n_sites: int,
) -> np.ndarray:
    if "TMG" not in train_X or "TMG" not in test_X:
        raise ValueError("target context requires TMG in train and test covariates")
    values = np.concatenate(
        [
            train_X["TMG"].to_numpy(dtype=float),
            test_X["TMG"].to_numpy(dtype=float),
        ]
    )
    if values.size < n_sites or not np.isfinite(values).all():
        raise ValueError("target context covariates are incomplete or non-finite")
    quantiles = (np.arange(n_sites, dtype=float) + 0.5) / float(n_sites)
    return np.quantile(values, quantiles)


def _target_context_response_batches(
    *,
    engine: NeuralHmscInference,
    count: int,
    tmg: np.ndarray,
    species_names: list[str],
    seed: int,
    pool: str,
) -> list[BetaResponseCalibrationBatch]:
    regimes = {
        "target_support": {
            "tmg_shift": 0.0,
            "tmg_noise": 0.02,
            "intercept_shift": 0.0,
            "slope_scale": 1.0,
        },
        "target_effect_size": {
            "tmg_shift": 0.0,
            "tmg_noise": 0.02,
            "intercept_shift": 0.0,
            "slope_scale": 1.75,
        },
        "target_combined": {
            "tmg_shift": 0.0,
            "tmg_noise": 0.02,
            "intercept_shift": -0.35,
            "slope_scale": 1.75,
        },
    }
    batches = []
    base_count, remainder = divmod(int(count), len(regimes))
    for regime_index, (regime, kwargs) in enumerate(regimes.items()):
        regime_count = base_count + int(regime_index < remainder)
        if regime_count <= 0:
            continue
        simulated = [
            _simulate_dataset(
                n_sites=len(tmg),
                n_species=len(species_names),
                tmg=tmg,
                species_names=species_names,
                seed=seed + 10_000 * regime_index + index,
                domain=f"big_spatial_target_context_{pool}_{regime}",
                **kwargs,
            )
            for index in range(regime_count)
        ]
        data = fixed_shape_training_data(simulated)
        batches.append(
            BetaResponseCalibrationBatch(
                posterior=engine.predict_beta_posterior(data),
                X=data.X,
                Y=data.Y,
                label=f"target_context_{pool}:{regime}",
            )
        )
    return batches


def _write_posterior(
    posterior,
    path: Path,
    species_names: list[str],
    args: argparse.Namespace,
    *,
    seed_offset: int,
    metadata: dict[str, object],
    calibration: BetaScaleCalibration | None = None,
) -> Path:
    return write_beta_posterior_hdf5(
        posterior,
        path,
        covariate_names=["Intercept", "TMG"],
        species_names=species_names,
        distribution="probit",
        formula="~ TMG",
        chains=args.neural_chains,
        draws=args.neural_draws,
        seed=args.seed + seed_offset,
        metadata=metadata,
        calibration=calibration,
    )


def _design(X: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(X), dtype=np.float32),
            X["TMG"].to_numpy(dtype=np.float32),
        ]
    )


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"frozen checkpoint directory has no files: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_coefficient_calibration(
    metadata: dict[str, object],
) -> BetaScaleCalibration | ConditionalBetaScaleCalibration:
    method = str(metadata.get("method", "temperature_scale"))
    if method.startswith("conditional_") or method == "external_context_monotone_scale":
        return ConditionalBetaScaleCalibration.from_metadata(metadata)
    return BetaScaleCalibration.from_metadata(metadata)


def _render_report(
    *,
    train_Y: pd.DataFrame,
    train_X: pd.DataFrame,
    test_Y: pd.DataFrame,
    test_X: pd.DataFrame,
    heldout: pd.DataFrame,
    calibration: dict[str, object],
    predictive_calibration: dict[str, object],
    predictive_mean_calibration: dict[str, object] | None,
    predictive_mean_selector: dict[str, object] | None,
    predictive_mean_selector_decision: dict[str, object] | None,
    acceptance: dict[str, object],
    frozen_artifacts: dict[str, object],
    neural_seconds: float,
    mcmc_seconds: float,
    reference_qualification: dict[str, object] | None,
) -> str:
    return "\n".join(
        [
            "# Frozen Neural-HMSC Big Spatial Plant Transfer",
            "",
            "The Whittaker checkpoint and both calibration scales were reused without updating",
            "weights or calibration parameters. Target holdout observations were used only for",
            "final evaluation.",
            "",
            f"Training/held-out sites: {len(train_Y)} / {len(test_Y)}",
            f"Species: {train_Y.shape[1]}",
            "Environmental projection: standardized `Max_temp_smooth` -> `TMG`",
            f"Training gradient range: {float(train_X.TMG.min()):.6f} to {float(train_X.TMG.max()):.6f}",
            f"Held-out gradient range: {float(test_X.TMG.min()):.6f} to {float(test_X.TMG.max()):.6f}",
            "",
            "## Frozen Artifacts",
            "",
            f"Checkpoint SHA-256: `{frozen_artifacts['checkpoint_sha256']}`",
            f"Coefficient calibration SHA-256: `{frozen_artifacts['coefficient_calibration_sha256']}`",
            f"Predictive calibration SHA-256: `{frozen_artifacts['predictive_calibration_sha256']}`",
            f"Coefficient calibration method: `{calibration['method']}`",
            "Coefficient scale: "
            f"{float(calibration.get('scale_multiplier', calibration.get('normalization_multiplier', 1.0))):.6f}",
            f"Predictive-only scale: {float(predictive_calibration['predictive_scale_multiplier']):.6f}",
            "Predictive-only mean calibration: "
            f"{_predictive_mean_calibration_line(predictive_mean_calibration)}",
            "Predictive mean selector: "
            f"{_predictive_mean_selector_line(predictive_mean_selector_decision)}",
            "Weights updated: False",
            "Calibration updated: False",
            "",
            "## Held-Out Metrics",
            "",
            heldout.to_string(index=False),
            "",
            *_reference_qualification_lines(reference_qualification),
            "## Acceptance",
            "",
            f"Inherited source SBC acceptance: {bool(acceptance['source_sbc_acceptance_passed'])}",
            f"Target predictive acceptance: {bool(acceptance['predictive_acceptance_passed'])}",
            f"Final predictive model: `{acceptance.get('final_predictive_model', 'n/a')}`",
            "Frozen predictive transfer acceptance: "
            f"{bool(acceptance['predictive_transfer_acceptance_passed'])}",
            "Target coefficient calibration assessable: False",
            "",
            "Passing this gate establishes predictive transfer under the fixed-shape projection.",
            "It does not establish target-domain coefficient calibration because real target data",
            "do not provide coefficient truth.",
            "",
            "## Runtime",
            "",
            f"Frozen neural inference: {neural_seconds:.3f} seconds",
            f"MCMC sampling: {mcmc_seconds:.3f} seconds",
            f"Inference-only speedup: {mcmc_seconds / max(neural_seconds, np.finfo(float).eps):.3f}x",
            "",
        ]
    )


def _predictive_mean_calibration_line(
    calibration: dict[str, object] | None,
) -> str:
    if calibration is None:
        return "`none`"
    response = calibration.get("response_validation", {})
    if not isinstance(response, dict):
        response = {}
    return (
        f"`{calibration.get('method', 'unknown')}` "
        f"selected={bool(calibration.get('selected', False))}, "
        f"slope={float(calibration.get('slope', 1.0)):.4f}, "
        f"intercept={float(calibration.get('intercept', 0.0)):.4f}, "
        f"validation_brier_ratio={_format_optional_float(response.get('brier_ratio'))}, "
        f"validation_log_loss_ratio={_format_optional_float(response.get('log_loss_ratio'))}"
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
