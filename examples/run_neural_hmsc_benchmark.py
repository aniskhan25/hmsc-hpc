"""Run a small Neural-HMSC fixed-effect benchmark suite.

By default this script trains neural posterior prototypes and writes neural
posterior files plus a manifest. Pass ``--run-mcmc-reference`` to also run the
Python-native Hmsc-HPC sampler and emit comparison reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache"))

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.benchmark import compare_beta_posteriors, write_benchmark_report
from pyhmsc.neural.calibration import apply_beta_scale_calibration, fit_beta_scale_calibration
from pyhmsc.neural.evaluation import predict_beta_posterior
from pyhmsc.neural.models import FixedShapeBetaPosteriorModel
from pyhmsc.neural.simulator import FixedEffectDataset, simulate_fixed_effect_dataset
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import fixed_shape_training_data, train_fixed_shape_beta_model
from pyhmsc.posterior import HmscFit


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
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--neural-chains", type=int, default=2)
    parser.add_argument("--neural-draws", type=int, default=200)
    parser.add_argument("--disable-calibration", action="store_true")
    parser.add_argument("--run-mcmc-reference", action="store_true")
    parser.add_argument("--mcmc-chains", type=int, default=2)
    parser.add_argument("--mcmc-samples", type=int, default=200)
    parser.add_argument("--mcmc-transient", type=int, default=100)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--mcmc-verbose", type=int, default=100)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    rows = []
    manifest: dict[str, object] = {
        "suite": args.suite,
        "n_sites": args.n_sites,
        "n_species": args.n_species,
        "train_datasets": args.train_datasets,
        "calibration_datasets": args.calibration_datasets,
        "epochs": args.epochs,
        "calibration_enabled": not bool(args.disable_calibration),
        "run_mcmc_reference": bool(args.run_mcmc_reference),
        "datasets": [],
    }
    for suite_idx, distribution in enumerate(args.suite):
        dataset_dir = output / distribution
        dataset_dir.mkdir(parents=True, exist_ok=True)
        train = _datasets(
            count=args.train_datasets,
            n_sites=args.n_sites,
            n_species=args.n_species,
            distribution=distribution,
            seed=args.seed + 1000 * suite_idx,
        )
        calibration = _datasets(
            count=args.calibration_datasets,
            n_sites=args.n_sites,
            n_species=args.n_species,
            distribution=distribution,
            seed=args.seed + 1000 * suite_idx + 500,
        )
        test = simulate_fixed_effect_dataset(
            n_sites=args.n_sites,
            n_species=args.n_species,
            distribution=distribution,
            seed=args.seed + 1000 * suite_idx + 999,
        )
        _write_dataset(test, dataset_dir)

        model = FixedShapeBetaPosteriorModel(
            n_sites=args.n_sites,
            n_covariates=3,
            n_species=args.n_species,
            hidden_units=(96, 96),
        )
        train_fixed_shape_beta_model(
            model,
            fixed_shape_training_data(train),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        test_data = fixed_shape_training_data([test])
        start = time.perf_counter()
        uncalibrated_posterior = predict_beta_posterior(model, test_data)
        uncalibrated_path = write_beta_posterior_hdf5(
            uncalibrated_posterior,
            dataset_dir / "neural_posterior_uncalibrated.h5",
            covariate_names=list(test.truth_beta.index),
            species_names=list(test.truth_beta.columns),
            distribution=distribution,
            formula="~ x1 + x2",
            chains=args.neural_chains,
            draws=args.neural_draws,
            seed=args.seed + suite_idx,
            metadata={"benchmark": {"script": Path(__file__).name, "distribution": distribution}},
        )
        if args.disable_calibration:
            calibration_result = None
            posterior = uncalibrated_posterior
        else:
            calibration_data = fixed_shape_training_data(calibration)
            calibration_posterior = predict_beta_posterior(model, calibration_data)
            calibration_result = fit_beta_scale_calibration(
                calibration_posterior,
                calibration_data.Beta,
                nominal_level=0.95,
                distribution=distribution,
            )
            posterior = apply_beta_scale_calibration(
                uncalibrated_posterior,
                calibration_result,
                distribution=distribution,
            )
        neural_path = write_beta_posterior_hdf5(
            posterior,
            dataset_dir / "neural_posterior.h5",
            covariate_names=list(test.truth_beta.index),
            species_names=list(test.truth_beta.columns),
            distribution=distribution,
            formula="~ x1 + x2",
            chains=args.neural_chains,
            draws=args.neural_draws,
            seed=args.seed + suite_idx + 100,
            metadata={"benchmark": {"script": Path(__file__).name, "distribution": distribution}},
            calibration=calibration_result,
        )
        neural_seconds = time.perf_counter() - start

        record: dict[str, object] = {
            "distribution": distribution,
            "neural_posterior": str(neural_path),
            "neural_posterior_uncalibrated": str(uncalibrated_path),
            "data_dir": str(dataset_dir),
            "neural_inference_wall_time_seconds": neural_seconds,
        }
        if calibration_result is not None:
            record["calibration"] = calibration_result.to_metadata()
        if args.run_mcmc_reference:
            mcmc_path = dataset_dir / "mcmc_reference.h5"
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
            )
            calibrated_row["posterior_variant"] = "calibrated"
            rows.extend([uncalibrated_row, calibrated_row])
            record["mcmc_posterior"] = str(mcmc_path)
            record["mcmc_wall_time_seconds"] = mcmc_seconds
        manifest["datasets"].append(record)  # type: ignore[index]

    (output / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if rows:
        paths = write_benchmark_report(rows, output)
        print(f"Wrote {paths.csv}")
        print(f"Wrote {paths.markdown}")
    else:
        print(f"Wrote neural benchmark artifacts in {output}")
        print("No MCMC comparison report was written because --run-mcmc-reference was not set.")


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


def _write_dataset(dataset: FixedEffectDataset, output: Path) -> None:
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.Y.to_csv(data_dir / "Y.csv")
    dataset.X.to_csv(data_dir / "X.csv")
    dataset.truth_beta.to_csv(data_dir / "truth_beta.csv")
    dataset.linear_predictor.to_csv(data_dir / "truth_linear_predictor.csv")
    (output / "dataset_metadata.json").write_text(json.dumps(dataset.metadata, indent=2), encoding="utf-8")


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
