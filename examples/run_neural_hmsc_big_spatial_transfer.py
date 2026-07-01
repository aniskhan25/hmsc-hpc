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
from examples.run_neural_hmsc_whittaker import _heldout_metrics, _metric_row
from pyhmsc.model import HmscModel
from pyhmsc.neural.benchmark import (
    compare_beta_posteriors,
    occurrence_predictive_acceptance,
    write_benchmark_report,
)
from pyhmsc.neural.calibration import (
    BetaScaleCalibration,
    apply_beta_predictive_calibration,
    apply_beta_scale_calibration,
)
from pyhmsc.neural.inference import NeuralHmscInference
from pyhmsc.neural.storage import write_beta_posterior_hdf5
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
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()
    _validate_args(parser, args)

    args.output.mkdir(parents=True, exist_ok=True)
    project = args.output / "big_spatial_transfer_project"
    generate_project(args.source_matrix, args.source_project, project)
    train_Y = pd.read_csv(project / "data/train/Y.csv", index_col=0)
    train_X = pd.read_csv(project / "data/train/X.csv", index_col=0)
    test_Y = pd.read_csv(project / "data/test/Y.csv", index_col=0)
    test_X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    species_names = [str(name) for name in train_Y.columns]

    checkpoint = args.frozen_run / "neural_checkpoint"
    coefficient_source = args.frozen_run / "neural_posterior.h5"
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
    calibration = BetaScaleCalibration.from_metadata(calibration_metadata)
    if calibration.predictive_scale_multiplier is None:
        raise ValueError("frozen calibration has no predictive-only multiplier")
    source_acceptance = json.loads(source_acceptance_path.read_text(encoding="utf-8"))
    if not bool(source_acceptance.get("qualification_acceptance_passed", False)):
        raise ValueError(
            "frozen source run did not pass its combined qualification gate"
        )

    frozen_artifacts = {
        "checkpoint_sha256": _directory_sha256(checkpoint),
        "coefficient_source_sha256": _file_sha256(coefficient_source),
        "calibration_sha256": _json_sha256(calibration.to_metadata()),
        "source_acceptance_sha256": _file_sha256(source_acceptance_path),
        "weights_updated": False,
        "calibration_updated": False,
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
        "transfer_validation": {
            "dataset": "Big Spatial Plant community",
            "source_run": str(args.frozen_run),
            "weights_updated": False,
            "calibration_updated": False,
            "source_covariate": "Max_temp_smooth",
            "projected_covariate": "TMG",
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
        calibration=calibration,
    )
    predictive_path = _write_posterior(
        predictive_only,
        args.output / "neural_predictive_distribution.h5",
        species_names,
        args,
        seed_offset=3,
        metadata=common_metadata | {"artifact_role": "predictive_only"},
        calibration=calibration,
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
    for variant, coefficient_variant, predictive_variant in [
        ("uncalibrated", uncalibrated_path, uncalibrated_path),
        ("coefficient_calibrated", coefficient_path, coefficient_path),
        ("predictive_only_calibrated", coefficient_path, predictive_path),
    ]:
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
        stem="big_spatial_transfer_neural_mcmc_reference",
        title="Frozen Neural-HMSC Big Spatial Plant Transfer",
    )
    heldout = pd.DataFrame(heldout_rows)
    heldout_path = args.output / "big_spatial_transfer_heldout_metrics.csv"
    heldout.to_csv(heldout_path, index=False)
    predictive_acceptance = occurrence_predictive_acceptance(
        _metric_row(heldout, "neural_uncalibrated"),
        _metric_row(heldout, "neural_predictive_only_calibrated"),
        _metric_row(heldout, "mcmc_fixed"),
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
        calibration=calibration.to_metadata(),
        acceptance=acceptance,
        frozen_artifacts=frozen_artifacts,
        neural_seconds=neural_seconds,
        mcmc_seconds=mcmc_seconds,
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
        },
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "frozen_artifacts": frozen_artifacts,
        "acceptance": acceptance,
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


def _render_report(
    *,
    train_Y: pd.DataFrame,
    train_X: pd.DataFrame,
    test_Y: pd.DataFrame,
    test_X: pd.DataFrame,
    heldout: pd.DataFrame,
    calibration: dict[str, object],
    acceptance: dict[str, object],
    frozen_artifacts: dict[str, object],
    neural_seconds: float,
    mcmc_seconds: float,
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
            f"Calibration SHA-256: `{frozen_artifacts['calibration_sha256']}`",
            f"Coefficient scale: {float(calibration['scale_multiplier']):.6f}",
            f"Predictive-only scale: {float(calibration['predictive_scale_multiplier']):.6f}",
            "Weights updated: False",
            "Calibration updated: False",
            "",
            "## Held-Out Metrics",
            "",
            heldout.to_string(index=False),
            "",
            "## Acceptance",
            "",
            f"Inherited source SBC acceptance: {bool(acceptance['source_sbc_acceptance_passed'])}",
            f"Target predictive acceptance: {bool(acceptance['predictive_acceptance_passed'])}",
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
