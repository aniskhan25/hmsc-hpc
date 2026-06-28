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
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache"))

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

from pyhmsc.neural.benchmark import compare_beta_posteriors, write_benchmark_report
from pyhmsc.neural.calibration import apply_beta_scale_calibration, fit_beta_scale_calibration
from pyhmsc.neural.inference import NeuralHmscInference
from pyhmsc.neural.simulator import FixedEffectDataset, simulate_fixed_effect_dataset
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import fixed_shape_training_data
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
    parser.add_argument("--mse-weight", type=float)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--neural-chains", type=int, default=2)
    parser.add_argument("--neural-draws", type=int, default=200)
    parser.add_argument(
        "--posterior-family",
        choices=["diagonal_normal", "full_covariance_normal"],
        default="full_covariance_normal",
    )
    parser.add_argument("--disable-calibration", action="store_true")
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

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_run_metadata(output / "run_metadata.json", args=args, started_at=started_at, status="running")

    rows = []
    manifest: dict[str, object] = {
        "started_at": started_at,
        "suite": args.suite,
        "n_sites": args.n_sites,
        "n_species": args.n_species,
        "train_datasets": args.train_datasets,
        "calibration_datasets": args.calibration_datasets,
        "epochs": args.epochs,
        "posterior_family": args.posterior_family,
        "mse_weight": args.mse_weight,
        "calibration_enabled": not bool(args.disable_calibration),
        "run_mcmc_reference": bool(args.run_mcmc_reference),
        "datasets": [],
    }
    for suite_idx, distribution in enumerate(args.suite):
        dataset_dir = output / distribution
        dataset_dir.mkdir(parents=True, exist_ok=True)
        record_path = dataset_dir / "benchmark_record.json"
        neural_path = dataset_dir / "neural_posterior.h5"
        uncalibrated_path = dataset_dir / "neural_posterior_uncalibrated.h5"
        mcmc_path = dataset_dir / "mcmc_reference.h5"
        checkpoint_dir = dataset_dir / "neural_checkpoint"
        if args.skip_existing and neural_path.exists() and uncalibrated_path.exists():
            record = _load_record(record_path)
            if record is None:
                record = {
                    "distribution": distribution,
                    "neural_posterior": str(neural_path),
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
            if args.run_mcmc_reference and mcmc_path.exists():
                test = _load_dataset(dataset_dir, distribution=distribution)
                rows.extend(
                    _comparison_rows(
                        distribution=distribution,
                        test=test,
                        neural_path=neural_path,
                        uncalibrated_path=uncalibrated_path,
                        mcmc_path=mcmc_path,
                        neural_seconds=record.get("neural_inference_wall_time_seconds"),
                        mcmc_seconds=record.get("mcmc_wall_time_seconds"),
                    )
                )
            manifest["datasets"].append(record)  # type: ignore[index]
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            continue

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
            seed=args.seed + suite_idx,
            mse_weight=args.mse_weight,
        )
        engine.save(checkpoint_dir)
        engine = NeuralHmscInference.load(checkpoint_dir)

        start = time.perf_counter()
        uncalibrated_fit = engine.infer(
            test,
            output=uncalibrated_path,
            chains=args.neural_chains,
            draws=args.neural_draws,
            seed=args.seed + suite_idx,
            metadata={"benchmark": {"script": Path(__file__).name, "distribution": distribution}},
        )
        uncalibrated_path = uncalibrated_fit.output_file or uncalibrated_path
        uncalibrated_posterior = engine.predict_beta_posterior(test)
        if args.disable_calibration:
            calibration_result = None
            posterior = uncalibrated_posterior
        else:
            calibration_data = fixed_shape_training_data(calibration)
            calibration_posterior = engine.predict_beta_posterior(calibration_data)
            calibration_result = fit_beta_scale_calibration(
                calibration_posterior,
                calibration_data.Beta,
                nominal_level=0.95,
                distribution=distribution,
                predictive_X=calibration_data.X if distribution == "poisson" else None,
                poisson_eta_clip=(-6.0, 6.0) if distribution == "poisson" else None,
                predictive_seed=args.seed + suite_idx + 50,
            )
            posterior = apply_beta_scale_calibration(
                uncalibrated_posterior,
                calibration_result,
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
            seed=args.seed + suite_idx + 100,
            metadata={"benchmark": {"script": Path(__file__).name, "distribution": distribution}},
            calibration=calibration_result,
        )
        neural_seconds = time.perf_counter() - start

        record: dict[str, object] = {
            "distribution": distribution,
            "posterior_family": args.posterior_family,
            "neural_posterior": str(neural_path),
            "neural_posterior_uncalibrated": str(uncalibrated_path),
            "neural_checkpoint": str(checkpoint_dir),
            "data_dir": str(dataset_dir),
            "neural_inference_wall_time_seconds": neural_seconds,
            "training_history": {
                "loss": training_history.loss,
                "beta_rmse": training_history.beta_rmse,
                "scale_mean": training_history.scale_mean,
            },
        }
        if calibration_result is not None:
            record["calibration"] = calibration_result.to_metadata()
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
                    uncalibrated_path=uncalibrated_path,
                    mcmc_path=mcmc_path,
                    neural_seconds=neural_seconds,
                    mcmc_seconds=mcmc_seconds,
                )
            )
            record["mcmc_posterior"] = str(mcmc_path)
            record["mcmc_wall_time_seconds"] = mcmc_seconds
        manifest["datasets"].append(record)  # type: ignore[index]
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["finished_at"] = finished_at
    (output / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if rows:
        paths = write_benchmark_report(rows, output)
        print(f"Wrote {paths.csv}")
        print(f"Wrote {paths.markdown}")
    else:
        print(f"Wrote neural benchmark artifacts in {output}")
        print("No MCMC comparison report was written because --run-mcmc-reference was not set.")
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


def _write_dataset(dataset: FixedEffectDataset, output: Path) -> None:
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.Y.to_csv(data_dir / "Y.csv")
    dataset.X.to_csv(data_dir / "X.csv")
    dataset.truth_beta.to_csv(data_dir / "truth_beta.csv")
    dataset.linear_predictor.to_csv(data_dir / "truth_linear_predictor.csv")
    (output / "dataset_metadata.json").write_text(json.dumps(dataset.metadata, indent=2), encoding="utf-8")


def _load_dataset(dataset_dir: Path, *, distribution: str) -> FixedEffectDataset:
    Y = _read_csv(dataset_dir / "data" / "Y.csv")
    X = _read_csv(dataset_dir / "data" / "X.csv")
    truth_beta = _read_csv(dataset_dir / "data" / "truth_beta.csv")
    linear = _read_csv(dataset_dir / "data" / "truth_linear_predictor.csv")
    metadata_path = dataset_dir / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"distribution": distribution}
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
    uncalibrated_path: Path,
    mcmc_path: Path,
    neural_seconds: float | None,
    mcmc_seconds: float | None,
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
    )
    calibrated_row["posterior_variant"] = "calibrated"
    if distribution == "poisson":
        uncalibrated_rmse = float(uncalibrated_row["neural_posterior_predictive_mean_rmse"])
        calibrated_rmse = float(calibrated_row["neural_posterior_predictive_mean_rmse"])
        ratio = calibrated_rmse / max(uncalibrated_rmse, np.finfo(float).eps)
        calibrated_row["predictive_rmse_ratio_vs_uncalibrated"] = ratio
        calibrated_row["predictive_acceptance_passed"] = bool(
            ratio <= 1.25
            and float(calibrated_row["neural_poisson_eta_clipped_fraction"]) <= 0.01
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
