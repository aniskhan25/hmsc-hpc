"""Run a small Neural-HMSC fixed-effect benchmark suite.

By default this script trains neural posterior prototypes and writes neural
posterior files plus a manifest. Pass ``--run-mcmc-reference`` to also run the
Python-native Hmsc-HPC sampler and emit comparison reports.
"""

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

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache")
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
loaded_pyhmsc = sys.modules.get("pyhmsc")
if loaded_pyhmsc is not None:
    loaded_file = Path(getattr(loaded_pyhmsc, "__file__", "")).resolve()
    if ROOT not in loaded_file.parents:
        for module_name in list(sys.modules):
            if module_name == "pyhmsc" or module_name.startswith("pyhmsc."):
                del sys.modules[module_name]

import tensorflow as tf

from pyhmsc.neural.benchmark import (
    compare_beta_posteriors,
    poisson_predictive_acceptance,
    sbc_calibration_acceptance,
    write_benchmark_report,
    write_sbc_report,
)
from pyhmsc.neural.calibration import (
    BetaScaleCalibration,
    apply_beta_predictive_calibration,
    apply_beta_scale_calibration,
    fit_beta_scale_calibration,
)
from pyhmsc.neural.conditional_calibration import (
    ConditionalBetaScaleCalibration,
    apply_conditional_beta_scale_calibration,
    conditional_beta_support_trust,
    fit_conditional_beta_scale_calibration,
)
from pyhmsc.neural.diagnostics import beta_sbc_stratified_diagnostics
from pyhmsc.neural.inference import NeuralHmscInference
from pyhmsc.neural.posterior_heads import sample_beta_posterior
from pyhmsc.neural.simulator import (
    FIXED_EFFECT_OOD_REGIMES,
    FixedEffectDataset,
    simulate_fixed_effect_dataset,
    simulate_fixed_effect_ood_dataset,
)
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import fixed_shape_training_data
from pyhmsc.posterior import HmscFit


DISTRIBUTION_SEED_OFFSETS = {"normal": 0, "probit": 1, "poisson": 2}


def distribution_seed(base_seed: int, distribution: str, *, delta: int = 0) -> int:
    """Return a suite-order-independent simulation seed for a distribution."""
    return int(base_seed) + 1000 * DISTRIBUTION_SEED_OFFSETS[distribution] + int(delta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="benchmark output directory")
    parser.add_argument(
        "--suite",
        nargs="+",
        default=["normal", "probit", "poisson"],
        choices=["normal", "probit", "poisson"],
        help="fixed-effect distributions to benchmark",
    )
    parser.add_argument("--n-sites", type=int, default=32)
    parser.add_argument("--n-species", type=int, default=3)
    parser.add_argument("--train-datasets", type=int, default=32)
    parser.add_argument("--calibration-datasets", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mse-weight", type=float)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument(
        "--checkpoint",
        help="reuse a frozen Neural-HMSC checkpoint instead of training a new amortizer",
    )
    parser.add_argument("--neural-chains", type=int, default=2)
    parser.add_argument("--neural-draws", type=int, default=200)
    parser.add_argument(
        "--posterior-family",
        choices=["auto", "diagonal_normal", "full_covariance_normal"],
        default="auto",
    )
    parser.add_argument("--disable-calibration", action="store_true")
    parser.add_argument(
        "--coefficient-calibration",
        choices=["scalar", "conditional"],
        default="scalar",
        help="coefficient-posterior calibration method; predictive calibration remains scalar",
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
    parser.add_argument("--conditional-calibration-rare-weight", type=float, default=4.0)
    parser.add_argument(
        "--conditional-calibration-intermediate-weight", type=float, default=2.0
    )
    parser.add_argument("--conditional-calibration-common-weight", type=float, default=1.0)
    parser.add_argument(
        "--conditional-calibration-support-quantile", type=float, default=0.99
    )
    parser.add_argument(
        "--conditional-calibration-fallback-strength", type=float, default=2.0
    )
    parser.add_argument("--sbc-datasets", type=int, default=32)
    parser.add_argument("--sbc-draws", type=int, default=256)
    parser.add_argument("--sbc-bins", type=int, default=10)
    parser.add_argument(
        "--ood-regimes",
        nargs="*",
        choices=sorted(FIXED_EFFECT_OOD_REGIMES),
        default=sorted(FIXED_EFFECT_OOD_REGIMES),
        help="named OOD simulation regimes included in SBC diagnostics",
    )
    parser.add_argument("--run-mcmc-reference", action="store_true")
    parser.add_argument("--mcmc-chains", type=int, default=2)
    parser.add_argument("--mcmc-samples", type=int, default=200)
    parser.add_argument("--mcmc-transient", type=int, default=100)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--mcmc-verbose", type=int, default=100)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse existing per-distribution artifacts when possible",
    )
    args = parser.parse_args()
    if args.model_seed is None:
        args.model_seed = args.seed
    if args.sbc_datasets != 0 and args.sbc_datasets < 2:
        parser.error("--sbc-datasets must be zero or at least two")
    if args.sbc_draws <= 0:
        parser.error("--sbc-draws must be positive")
    if args.sbc_bins < 2 or args.sbc_bins > args.sbc_draws + 1:
        parser.error("--sbc-bins must be between 2 and --sbc-draws + 1")
    if args.conditional_calibration_epochs <= 0:
        parser.error("--conditional-calibration-epochs must be positive")
    if args.conditional_calibration_learning_rate <= 0.0:
        parser.error("--conditional-calibration-learning-rate must be positive")
    if args.conditional_calibration_regularization < 0.0:
        parser.error("--conditional-calibration-regularization must be non-negative")
    if args.conditional_calibration_rank_penalty_weight < 0.0:
        parser.error("--conditional-calibration-rank-penalty-weight must be non-negative")
    if min(
        args.conditional_calibration_rare_weight,
        args.conditional_calibration_intermediate_weight,
        args.conditional_calibration_common_weight,
    ) <= 0.0:
        parser.error("conditional prevalence weights must be positive")
    if not 0.5 < args.conditional_calibration_support_quantile < 1.0:
        parser.error("--conditional-calibration-support-quantile must be between 0.5 and 1")
    if args.conditional_calibration_fallback_strength < 0.0:
        parser.error("--conditional-calibration-fallback-strength must be non-negative")
    if args.checkpoint and len(args.suite) != 1:
        parser.error("--checkpoint requires exactly one distribution in --suite")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.model_seed)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_run_metadata(
        output / "run_metadata.json", args=args, started_at=started_at, status="running"
    )

    rows = []
    sbc_rows = []
    manifest: dict[str, object] = {
        "started_at": started_at,
        "suite": args.suite,
        "n_sites": args.n_sites,
        "n_species": args.n_species,
        "train_datasets": args.train_datasets,
        "calibration_datasets": args.calibration_datasets,
        "epochs": args.epochs,
        "posterior_family_policy": args.posterior_family,
        "mse_weight": args.mse_weight,
        "calibration_enabled": not bool(args.disable_calibration),
        "coefficient_calibration": args.coefficient_calibration,
        "sbc_datasets": args.sbc_datasets,
        "sbc_draws": args.sbc_draws,
        "sbc_bins": args.sbc_bins,
        "ood_regimes": args.ood_regimes,
        "model_seed": args.model_seed,
        "run_mcmc_reference": bool(args.run_mcmc_reference),
        "datasets": [],
    }
    for distribution in args.suite:
        suite_idx = DISTRIBUTION_SEED_OFFSETS[distribution]
        dataset_dir = output / distribution
        dataset_dir.mkdir(parents=True, exist_ok=True)
        record_path = dataset_dir / "benchmark_record.json"
        neural_path = dataset_dir / "neural_posterior.h5"
        predictive_path = dataset_dir / "neural_predictive_distribution.h5"
        uncalibrated_path = dataset_dir / "neural_posterior_uncalibrated.h5"
        mcmc_path = dataset_dir / "mcmc_reference.h5"
        checkpoint_dir = dataset_dir / "neural_checkpoint"
        sbc_path = dataset_dir / "sbc_diagnostics.json"
        if (
            args.skip_existing
            and neural_path.exists()
            and predictive_path.exists()
            and uncalibrated_path.exists()
        ):
            record = _load_record(record_path)
            if record is None:
                record = {
                    "distribution": distribution,
                    "neural_posterior": str(neural_path),
                    "neural_predictive_distribution": str(predictive_path),
                    "neural_posterior_uncalibrated": str(uncalibrated_path),
                    "neural_checkpoint": str(checkpoint_dir),
                    "data_dir": str(dataset_dir),
                    "neural_inference_wall_time_seconds": None,
                    "reused_existing": True,
                }
            if args.run_mcmc_reference and not mcmc_path.exists():
                test = _load_dataset(dataset_dir, distribution=distribution)
                mcmc_seconds = _run_mcmc_reference(
                    test,
                    output=mcmc_path,
                    workdir=dataset_dir / "mcmc_work",
                    samples=args.mcmc_samples,
                    transient=args.mcmc_transient,
                    thin=args.mcmc_thin,
                    chains=args.mcmc_chains,
                    verbose=args.mcmc_verbose,
                )
                record["mcmc_posterior"] = str(mcmc_path)
                record["mcmc_wall_time_seconds"] = mcmc_seconds
            distribution_sbc_rows = []
            if args.sbc_datasets > 0:
                if sbc_path.exists():
                    distribution_sbc_rows = json.loads(
                        sbc_path.read_text(encoding="utf-8")
                    )
                elif checkpoint_dir.exists():
                    engine = NeuralHmscInference.load(checkpoint_dir)
                    calibration_result = None
                    if not args.disable_calibration:
                        metadata = HmscFit.from_file(neural_path).metadata.get(
                            "calibration"
                        )
                        if isinstance(metadata, dict):
                            if metadata.get("method") in {
                                "conditional_structured_scale",
                                "conditional_rank_aware_scale",
                            }:
                                calibration_result = (
                                    ConditionalBetaScaleCalibration.from_metadata(metadata)
                                )
                            else:
                                calibration_result = BetaScaleCalibration.from_metadata(
                                    metadata
                                )
                    distribution_sbc_rows = _sbc_rows(
                        engine=engine,
                        calibration=calibration_result,
                        distribution=distribution,
                        n_sites=args.n_sites,
                        n_species=args.n_species,
                        n_datasets=args.sbc_datasets,
                        draws=args.sbc_draws,
                        n_bins=args.sbc_bins,
                        ood_regimes=args.ood_regimes,
                        seed=distribution_seed(args.seed, distribution, delta=2000),
                    )
                    sbc_path.write_text(
                        json.dumps(distribution_sbc_rows, indent=2) + "\n",
                        encoding="utf-8",
                    )
                else:
                    raise RuntimeError(
                        f"cannot compute SBC diagnostics without checkpoint {checkpoint_dir}"
                    )
                sbc_rows.extend(distribution_sbc_rows)
                record["sbc_diagnostics"] = str(sbc_path)
            if args.run_mcmc_reference and mcmc_path.exists():
                test = _load_dataset(dataset_dir, distribution=distribution)
                rows.extend(
                    _comparison_rows(
                        distribution=distribution,
                        test=test,
                        neural_path=neural_path,
                        predictive_path=predictive_path,
                        uncalibrated_path=uncalibrated_path,
                        mcmc_path=mcmc_path,
                        neural_seconds=record.get("neural_inference_wall_time_seconds"),
                        mcmc_seconds=record.get("mcmc_wall_time_seconds"),
                        sbc_rows=distribution_sbc_rows,
                    )
                )
            manifest["datasets"].append(record)  # type: ignore[index]
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            continue

        train = None
        if not args.checkpoint:
            train = _datasets(
                count=args.train_datasets,
                n_sites=args.n_sites,
                n_species=args.n_species,
                distribution=distribution,
                seed=distribution_seed(args.seed, distribution),
            )
        calibration = _datasets(
            count=args.calibration_datasets,
            n_sites=args.n_sites,
            n_species=args.n_species,
            distribution=distribution,
            seed=distribution_seed(args.seed, distribution, delta=500),
        )
        test = simulate_fixed_effect_dataset(
            n_sites=args.n_sites,
            n_species=args.n_species,
            distribution=distribution,
            seed=distribution_seed(args.seed, distribution, delta=999),
        )
        _write_dataset(test, dataset_dir)

        if args.checkpoint:
            checkpoint_source = Path(args.checkpoint)
            engine = NeuralHmscInference.load(checkpoint_source)
            if engine.distribution != distribution:
                raise ValueError(
                    "checkpoint distribution mismatch: "
                    f"expected {distribution!r}, got {engine.distribution!r}"
                )
            if engine.dimensions != {
                "n_sites": args.n_sites,
                "n_covariates": 3,
                "n_species": args.n_species,
            }:
                raise ValueError(
                    "checkpoint dimensions do not match the requested benchmark shape"
                )
            training_history = None
        else:
            if train is None:
                raise RuntimeError("training datasets are unavailable")
            tf.keras.utils.set_random_seed(args.model_seed + suite_idx)
            engine = NeuralHmscInference.for_fixed_effects(
                n_sites=args.n_sites,
                n_species=args.n_species,
                n_covariates=3,
                distribution=distribution,
                formula="~ x1 + x2",
                covariate_names=list(test.truth_beta.index),
                species_names=list(test.truth_beta.columns),
                hidden_units=(96, 96),
                posterior_family=args.posterior_family,
            )
            training_history = engine.fit(
                train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.model_seed + suite_idx,
                mse_weight=args.mse_weight,
            )
            engine.save(checkpoint_dir)
            checkpoint_source = checkpoint_dir
            engine = NeuralHmscInference.load(checkpoint_source)

        start = time.perf_counter()
        uncalibrated_fit = engine.infer(
            test,
            output=uncalibrated_path,
            chains=args.neural_chains,
            draws=args.neural_draws,
            seed=args.model_seed + suite_idx,
            metadata={
                "benchmark": {
                    "script": Path(__file__).name,
                    "distribution": distribution,
                }
            },
        )
        uncalibrated_path = uncalibrated_fit.output_file or uncalibrated_path
        uncalibrated_posterior = engine.predict_beta_posterior(test)
        if args.disable_calibration:
            calibration_result = None
            posterior = uncalibrated_posterior
            predictive_posterior = uncalibrated_posterior
            predictive_calibration_result = None
        else:
            calibration_data = fixed_shape_training_data(calibration)
            calibration_posterior = engine.predict_beta_posterior(calibration_data)
            predictive_calibration_result = fit_beta_scale_calibration(
                calibration_posterior,
                calibration_data.Beta,
                nominal_level=0.95,
                distribution=distribution,
                predictive_X=calibration_data.X if distribution == "poisson" else None,
                poisson_eta_clip=(-6.0, 6.0) if distribution == "poisson" else None,
                predictive_seed=args.model_seed + suite_idx + 50,
            )
            if args.coefficient_calibration == "conditional":
                calibration_result = fit_conditional_beta_scale_calibration(
                    calibration_posterior,
                    calibration_data.Beta,
                    X=calibration_data.X,
                    Y=calibration_data.Y,
                    distribution=distribution,
                    coefficient_names=engine.covariate_names,
                    baseline_calibration=predictive_calibration_result,
                    nominal_level=0.95,
                    regularization=args.conditional_calibration_regularization,
                    epochs=args.conditional_calibration_epochs,
                    learning_rate=args.conditional_calibration_learning_rate,
                    prevalence_weights=(
                        args.conditional_calibration_rare_weight,
                        args.conditional_calibration_intermediate_weight,
                        args.conditional_calibration_common_weight,
                    ),
                    rank_penalty_weight=(
                        args.conditional_calibration_rank_penalty_weight
                    ),
                    support_quantile=args.conditional_calibration_support_quantile,
                    fallback_strength=args.conditional_calibration_fallback_strength,
                )
                posterior = apply_conditional_beta_scale_calibration(
                    uncalibrated_posterior,
                    calibration_result,
                    X=fixed_shape_training_data([test]).X,
                    Y=fixed_shape_training_data([test]).Y,
                    distribution=distribution,
                    coefficient_names=engine.covariate_names,
                )
            else:
                calibration_result = predictive_calibration_result
                posterior = apply_beta_scale_calibration(
                    uncalibrated_posterior,
                    calibration_result,
                    distribution=distribution,
                )
            predictive_posterior = apply_beta_predictive_calibration(
                uncalibrated_posterior,
                predictive_calibration_result,
                distribution=distribution,
            )
        neural_path = write_beta_posterior_hdf5(
            posterior,
            neural_path,
            covariate_names=list(test.truth_beta.index),
            species_names=list(test.truth_beta.columns),
            distribution=distribution,
            formula="~ x1 + x2",
            chains=args.neural_chains,
            draws=args.neural_draws,
            seed=args.model_seed + suite_idx + 100,
            metadata={
                "benchmark": {
                    "script": Path(__file__).name,
                    "distribution": distribution,
                },
                "artifact_role": "coefficient_posterior",
            },
            calibration=calibration_result,
        )
        predictive_path = write_beta_posterior_hdf5(
            predictive_posterior,
            predictive_path,
            covariate_names=list(test.truth_beta.index),
            species_names=list(test.truth_beta.columns),
            distribution=distribution,
            formula="~ x1 + x2",
            chains=args.neural_chains,
            draws=args.neural_draws,
            seed=args.model_seed + suite_idx + 200,
            metadata={
                "benchmark": {
                    "script": Path(__file__).name,
                    "distribution": distribution,
                },
                "artifact_role": "predictive_only",
            },
            calibration=predictive_calibration_result,
        )
        neural_seconds = time.perf_counter() - start

        distribution_sbc_rows = []
        if args.sbc_datasets > 0:
            distribution_sbc_rows = _sbc_rows(
                engine=engine,
                calibration=calibration_result,
                distribution=distribution,
                n_sites=args.n_sites,
                n_species=args.n_species,
                n_datasets=args.sbc_datasets,
                draws=args.sbc_draws,
                n_bins=args.sbc_bins,
                ood_regimes=args.ood_regimes,
                seed=distribution_seed(args.seed, distribution, delta=2000),
            )
            sbc_rows.extend(distribution_sbc_rows)
            sbc_path.write_text(
                json.dumps(distribution_sbc_rows, indent=2) + "\n",
                encoding="utf-8",
            )

        record: dict[str, object] = {
            "distribution": distribution,
            "posterior_family": engine.model.posterior_family,
            "model_seed": args.model_seed + suite_idx,
            "neural_posterior": str(neural_path),
            "neural_predictive_distribution": str(predictive_path),
            "neural_posterior_uncalibrated": str(uncalibrated_path),
            "neural_checkpoint": str(checkpoint_source),
            "data_dir": str(dataset_dir),
            "neural_inference_wall_time_seconds": neural_seconds,
        }
        if training_history is None:
            record["reused_frozen_checkpoint"] = True
        else:
            record["training_history"] = {
                "loss": training_history.loss,
                "beta_rmse": training_history.beta_rmse,
                "scale_mean": training_history.scale_mean,
            }
        if distribution_sbc_rows:
            record["sbc_diagnostics"] = str(sbc_path)
        if calibration_result is not None:
            record["calibration"] = calibration_result.to_metadata()
        if predictive_calibration_result is not None:
            record["predictive_calibration"] = (
                predictive_calibration_result.to_metadata()
            )
        if args.run_mcmc_reference:
            mcmc_seconds = _run_mcmc_reference(
                test,
                output=mcmc_path,
                workdir=dataset_dir / "mcmc_work",
                samples=args.mcmc_samples,
                transient=args.mcmc_transient,
                thin=args.mcmc_thin,
                chains=args.mcmc_chains,
                verbose=args.mcmc_verbose,
            )
            rows.extend(
                _comparison_rows(
                    distribution=distribution,
                    test=test,
                    neural_path=neural_path,
                    predictive_path=predictive_path,
                    uncalibrated_path=uncalibrated_path,
                    mcmc_path=mcmc_path,
                    neural_seconds=neural_seconds,
                    mcmc_seconds=mcmc_seconds,
                    sbc_rows=distribution_sbc_rows,
                )
            )
            record["mcmc_posterior"] = str(mcmc_path)
            record["mcmc_wall_time_seconds"] = mcmc_seconds
        manifest["datasets"].append(record)  # type: ignore[index]
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["finished_at"] = finished_at
    (output / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if rows:
        paths = write_benchmark_report(rows, output)
        print(f"Wrote {paths.csv}")
        print(f"Wrote {paths.markdown}")
    else:
        print(f"Wrote neural benchmark artifacts in {output}")
        print(
            "No MCMC comparison report was written because --run-mcmc-reference was not set."
        )
    if sbc_rows:
        sbc_paths = write_sbc_report(sbc_rows, output)
        print(f"Wrote {sbc_paths.csv}")
        print(f"Wrote {sbc_paths.markdown}")
        print(f"Wrote {sbc_paths.json}")
    _write_run_metadata(
        output / "run_metadata.json",
        args=args,
        started_at=started_at,
        finished_at=finished_at,
        status="completed",
    )


def _datasets(
    *,
    count: int,
    n_sites: int,
    n_species: int,
    distribution: str,
    seed: int,
) -> list[FixedEffectDataset]:
    return [
        simulate_fixed_effect_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution=distribution,
            seed=seed + idx,
        )
        for idx in range(count)
    ]


def _sbc_rows(
    *,
    engine: NeuralHmscInference,
    calibration: BetaScaleCalibration | ConditionalBetaScaleCalibration | None,
    distribution: str,
    n_sites: int,
    n_species: int,
    n_datasets: int,
    draws: int,
    n_bins: int,
    ood_regimes: list[str],
    seed: int,
) -> list[dict[str, object]]:
    domains: list[tuple[str, str | None, list[FixedEffectDataset]]] = [
        (
            "in_distribution",
            None,
            _datasets(
                count=n_datasets,
                n_sites=n_sites,
                n_species=n_species,
                distribution=distribution,
                seed=seed,
            ),
        )
    ]
    for regime_idx, regime in enumerate(ood_regimes):
        domains.append(
            (
                "ood",
                regime,
                [
                    simulate_fixed_effect_ood_dataset(
                        n_sites=n_sites,
                        n_species=n_species,
                        distribution=distribution,
                        regime=regime,
                        seed=seed + 10_000 * (regime_idx + 1) + dataset_idx,
                    )
                    for dataset_idx in range(n_datasets)
                ],
            )
        )

    rows: list[dict[str, object]] = []
    for domain_idx, (simulation_domain, ood_regime, datasets) in enumerate(domains):
        data = fixed_shape_training_data(datasets)
        uncalibrated = engine.predict_beta_posterior(data)
        variants = [("uncalibrated", uncalibrated)]
        conditional_trust = None
        if calibration is not None:
            if isinstance(calibration, ConditionalBetaScaleCalibration):
                conditional_trust = conditional_beta_support_trust(
                    uncalibrated,
                    calibration,
                    X=data.X,
                    Y=data.Y,
                    distribution=distribution,
                    coefficient_names=engine.covariate_names,
                )
                calibrated = apply_conditional_beta_scale_calibration(
                    uncalibrated,
                    calibration,
                    X=data.X,
                    Y=data.Y,
                    distribution=distribution,
                    coefficient_names=engine.covariate_names,
                )
            else:
                calibrated = apply_beta_scale_calibration(
                    uncalibrated,
                    calibration,
                    distribution=distribution,
                )
            variants.append(
                (
                    "calibrated",
                    calibrated,
                )
            )
        for variant_idx, (posterior_variant, posterior) in enumerate(variants):
            samples = sample_beta_posterior(
                posterior,
                draws=draws,
                seed=seed + 1000 * domain_idx + variant_idx,
            ).numpy()
            samples = np.transpose(samples, (1, 0, 2, 3))
            diagnostics_by_stratum = beta_sbc_stratified_diagnostics(
                samples,
                data.Beta,
                X=data.X,
                Y=data.Y,
                distribution=distribution,
                covariate_names=datasets[0].truth_beta.index,
                n_bins=n_bins,
                seed=seed + 2000 * domain_idx + variant_idx,
            )
            metadata = datasets[0].metadata
            for stratum in diagnostics_by_stratum:
                row: dict[str, object] = {
                    "distribution": distribution,
                    "simulation_domain": simulation_domain,
                    "ood_regime": ood_regime,
                    "posterior_variant": posterior_variant,
                    "simulation_covariate_mean": metadata.get("covariate_mean"),
                    "simulation_covariate_scale": metadata.get("covariate_scale"),
                    "simulation_beta_scale": metadata.get("beta_scale"),
                }
                row.update(stratum.report_fields())
                if posterior_variant == "calibrated" and conditional_trust is not None:
                    row.update(
                        {
                            "conditional_support_trust_mean": float(
                                np.mean(conditional_trust)
                            ),
                            "conditional_support_trust_min": float(
                                np.min(conditional_trust)
                            ),
                            "conditional_support_fallback_fraction": float(
                                np.mean(conditional_trust < 0.5)
                            ),
                        }
                    )
                rows.append(row)

    in_distribution_rmse = {
        (
            str(row["posterior_variant"]),
            str(row["sbc_stratum_kind"]),
            str(row["sbc_stratum_label"]),
        ): float(row["sbc_beta_mean_rmse"])
        for row in rows
        if row["simulation_domain"] == "in_distribution"
    }
    in_distribution = {
        str(row["posterior_variant"]): row
        for row in rows
        if row["simulation_domain"] == "in_distribution"
        and row["sbc_stratum_kind"] == "overall"
    }
    if "uncalibrated" in in_distribution and "calibrated" in in_distribution:
        in_distribution["calibrated"].update(
            sbc_calibration_acceptance(
                in_distribution["uncalibrated"],
                in_distribution["calibrated"],
            )
        )
    for row in rows:
        if row["simulation_domain"] != "ood":
            continue
        baseline_key = (
            str(row["posterior_variant"]),
            str(row["sbc_stratum_kind"]),
            str(row["sbc_stratum_label"]),
        )
        baseline = in_distribution_rmse.get(baseline_key)
        if baseline is None:
            continue
        row["ood_rmse_ratio_vs_in_distribution"] = float(
            row["sbc_beta_mean_rmse"]
        ) / max(
            baseline,
            np.finfo(float).eps,
        )
    return rows


def _write_dataset(dataset: FixedEffectDataset, output: Path) -> None:
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.Y.to_csv(data_dir / "Y.csv")
    dataset.X.to_csv(data_dir / "X.csv")
    dataset.truth_beta.to_csv(data_dir / "truth_beta.csv")
    dataset.linear_predictor.to_csv(data_dir / "truth_linear_predictor.csv")
    (output / "dataset_metadata.json").write_text(
        json.dumps(dataset.metadata, indent=2), encoding="utf-8"
    )


def _load_dataset(dataset_dir: Path, *, distribution: str) -> FixedEffectDataset:
    Y = _read_csv(dataset_dir / "data" / "Y.csv")
    X = _read_csv(dataset_dir / "data" / "X.csv")
    truth_beta = _read_csv(dataset_dir / "data" / "truth_beta.csv")
    linear = _read_csv(dataset_dir / "data" / "truth_linear_predictor.csv")
    metadata_path = dataset_dir / "dataset_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {"distribution": distribution}
    )
    return FixedEffectDataset(
        Y=Y,
        X=X,
        truth_beta=truth_beta,
        linear_predictor=linear,
        metadata=metadata,
    )


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path, index_col=0)


def _comparison_rows(
    *,
    distribution: str,
    test: FixedEffectDataset,
    neural_path: Path,
    predictive_path: Path,
    uncalibrated_path: Path,
    mcmc_path: Path,
    neural_seconds: float | None,
    mcmc_seconds: float | None,
    sbc_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    poisson_eta_clip = None
    if distribution == "poisson":
        bounds = test.metadata.get("poisson_eta_clip")
        if isinstance(bounds, list) and len(bounds) == 2:
            poisson_eta_clip = (float(bounds[0]), float(bounds[1]))
    uncalibrated_row = compare_beta_posteriors(
        HmscFit.from_file(uncalibrated_path),
        HmscFit.from_file(mcmc_path),
        truth_beta=test.truth_beta,
        dataset=distribution,
        distribution=distribution,
        neural_seconds=neural_seconds,
        mcmc_seconds=mcmc_seconds,
        X=test.X,
        Y=test.Y,
        formula="~ x1 + x2",
        poisson_eta_clip=poisson_eta_clip,
    )
    uncalibrated_row["posterior_variant"] = "uncalibrated"
    calibrated_row = compare_beta_posteriors(
        HmscFit.from_file(neural_path),
        HmscFit.from_file(mcmc_path),
        truth_beta=test.truth_beta,
        dataset=distribution,
        distribution=distribution,
        neural_seconds=neural_seconds,
        mcmc_seconds=mcmc_seconds,
        X=test.X,
        Y=test.Y,
        formula="~ x1 + x2",
        poisson_eta_clip=poisson_eta_clip,
        neural_predictive_fit=HmscFit.from_file(predictive_path),
    )
    calibrated_row["posterior_variant"] = "calibrated"
    if distribution == "poisson":
        predictive_acceptance = poisson_predictive_acceptance(
            uncalibrated_row, calibrated_row
        )
        calibrated_row.update(predictive_acceptance)
        calibrated_sbc = next(
            (
                row
                for row in sbc_rows
                if row.get("simulation_domain") == "in_distribution"
                and row.get("posterior_variant") == "calibrated"
                and row.get("sbc_stratum_kind") == "overall"
            ),
            None,
        )
        sbc_passed = False
        if calibrated_sbc is not None:
            for key in [
                "sbc_beta_interval_coverage_95",
                "sbc_rank_mean",
                "sbc_rank_variance",
                "sbc_expected_rank_mean",
                "sbc_expected_rank_variance",
                "sbc_coverage_error_uncalibrated",
                "sbc_coverage_error_calibrated",
                "sbc_rank_mean_error_uncalibrated",
                "sbc_rank_mean_error_calibrated",
                "sbc_rank_variance_error_uncalibrated",
                "sbc_rank_variance_error_calibrated",
                "sbc_acceptance_passed",
            ]:
                if key in calibrated_sbc:
                    calibrated_row[key] = calibrated_sbc[key]
            sbc_passed = bool(calibrated_sbc.get("sbc_acceptance_passed", False))
        calibrated_row["qualification_acceptance_passed"] = bool(
            predictive_acceptance["predictive_acceptance_passed"] and sbc_passed
        )
    return [uncalibrated_row, calibrated_row]


def _load_record(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_run_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    started_at: str,
    status: str,
    finished_at: str | None = None,
) -> None:
    payload = {
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "argv": sys.argv,
        "args": vars(args),
        "platform": platform.platform(),
        "python": sys.version,
        "tensorflow": tf.__version__,
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "slurm": {
            key: os.environ[key]
            for key in [
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_SUBMIT_DIR",
                "SLURM_CPUS_PER_TASK",
                "SLURM_GPUS",
            ]
            if key in os.environ
        },
        "git_commit": _git_commit(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _run_mcmc_reference(
    dataset: FixedEffectDataset,
    *,
    output: Path,
    workdir: Path,
    samples: int,
    transient: int,
    thin: int,
    chains: int,
    verbose: int,
) -> float:
    from pyhmsc.model import HmscModel

    model = HmscModel(
        Y=dataset.Y,
        X=dataset.X,
        x_formula="~ x1 + x2",
        distr=str(dataset.metadata["distribution"]),
    )
    start = time.perf_counter()
    fit = model.sample(
        samples=samples,
        transient=transient,
        thin=thin,
        chains=chains,
        init="python-native",
        workdir=workdir,
        verbose=verbose,
        output_file=output,
    )
    elapsed = time.perf_counter() - start
    if fit.output_file != output:
        raise RuntimeError(f"Expected MCMC output at {output}, got {fit.output_file}")
    return elapsed


if __name__ == "__main__":
    main()
