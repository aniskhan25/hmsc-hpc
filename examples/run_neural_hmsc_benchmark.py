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
    ConditionalBetaOODCalibrationBatch,
    ConditionalBetaScaleCalibration,
    apply_conditional_beta_scale_calibration,
    conditional_beta_effect_size_signal,
    conditional_beta_mean_support_diagnostics,
    conditional_beta_ood_uncertainty_inflation,
    conditional_beta_support_trust,
    fit_conditional_beta_scale_calibration,
    fit_external_context_monotone_calibration,
)
from pyhmsc.neural.diagnostics import beta_sbc_stratified_diagnostics
from pyhmsc.neural.inference import NeuralHmscInference
from pyhmsc.neural.mean_calibration import (
    BetaResponseCalibrationBatch,
    apply_beta_predictive_mean_calibration,
    fit_beta_predictive_mean_calibration,
    fit_beta_response_mean_calibration,
    fit_beta_transfer_response_mean_calibration,
)
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
    parser.add_argument(
        "--rare-calibration-datasets",
        type=int,
        default=0,
        help="additional in-domain simulations for rare-species calibration head",
    )
    parser.add_argument(
        "--rare-validation-datasets",
        type=int,
        default=0,
        help="independent rare simulations used to gate rare-head offsets",
    )
    parser.add_argument(
        "--rare-calibration-intercept-mean",
        type=float,
        default=-1.75,
        help="intercept mean used when generating rare-species calibration simulations",
    )
    parser.add_argument(
        "--rare-calibration-regimes",
        nargs="+",
        default=["intercept_shift", "low_detection", "small_sample"],
        choices=["intercept_shift", "low_detection", "small_sample"],
        help="rare-calibration simulation regimes to balance across",
    )
    parser.add_argument(
        "--rare-calibration-detection-probability",
        type=float,
        default=0.35,
        help="detection thinning probability for the low-detection rare regime",
    )
    parser.add_argument(
        "--rare-calibration-sample-fraction",
        type=float,
        default=0.35,
        help="effective observation fraction for the small-sample rare regime",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mse-weight", type=float)
    parser.add_argument(
        "--rank-mean-penalty-weight",
        type=float,
        default=0.0,
        help="training-time holdout rank-mean penalty weight for rare-prevalence species",
    )
    parser.add_argument(
        "--rank-mean-penalty-holdout-fraction",
        type=float,
        default=0.25,
        help="fraction of training simulations reserved for rank-mean penalty batches",
    )
    parser.add_argument(
        "--rank-mean-penalty-holdout-folds",
        type=int,
        default=1,
        help="number of holdout folds for cross-fit rank-penalty gating",
    )
    parser.add_argument(
        "--rank-mean-penalty-crossfit-min-agreement",
        type=float,
        default=0.75,
        help="minimum rare-rank sign agreement required for signed mean correction",
    )
    parser.add_argument(
        "--rank-mean-penalty-start-fraction",
        type=float,
        default=0.0,
        help="fraction of training epochs to finish before activating the rank penalty",
    )
    parser.add_argument(
        "--rank-mean-penalty-design-guard-weight",
        type=float,
        default=0.0,
        help="weight for the design-information coverage guard inside the rank penalty",
    )
    parser.add_argument(
        "--rank-mean-penalty-design-guard-floor",
        type=float,
        default=0.925,
        help="minimum smooth coverage target for design-information strata",
    )
    parser.add_argument(
        "--rank-mean-penalty-signed-mean-weight",
        type=float,
        default=0.0,
        help="relative weight for signed rare-prevalence posterior-mean correction",
    )
    parser.add_argument(
        "--rank-mean-penalty-design-mean-guard-weight",
        type=float,
        default=0.0,
        help="relative weight for medium/high design-information mean-rank guards",
    )
    parser.add_argument(
        "--rank-mean-penalty-design-mean-guard-tolerance",
        type=float,
        default=0.025,
        help="rank-mean tolerance for design-information mean guards",
    )
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
    parser.add_argument(
        "--probit-anchor",
        choices=["auto", "ridge", "irls_laplace"],
        default="auto",
        help="deterministic coefficient anchor used by newly trained probit amortizers",
    )
    parser.add_argument("--probit-anchor-iterations", type=int, default=8)
    parser.add_argument("--probit-anchor-prior-precision", type=float, default=1.0)
    parser.add_argument("--probit-anchor-eta-clip", type=float, default=6.0)
    parser.add_argument("--disable-calibration", action="store_true")
    parser.add_argument(
        "--coefficient-calibration",
        choices=["scalar", "conditional", "external_monotone"],
        default="scalar",
        help="coefficient-posterior calibration method; predictive calibration remains scalar",
    )
    parser.add_argument(
        "--predictive-mean-calibration",
        choices=[
            "none",
            "affine_shrinkage",
            "probit_response_affine",
            "probit_transfer_response_affine",
        ],
        default="none",
        help=(
            "optional predictive-only Beta-mean calibration competitor; does not "
            "change coefficient posterior or SBC calibration"
        ),
    )
    parser.add_argument(
        "--predictive-mean-calibration-validation-datasets",
        type=int,
        default=0,
        help="independent simulations used to gate predictive mean calibration",
    )
    parser.add_argument(
        "--predictive-mean-calibration-max-rmse-ratio",
        type=float,
        default=1.0,
        help="maximum validation Beta-mean RMSE ratio allowed before fallback",
    )
    parser.add_argument(
        "--predictive-mean-calibration-max-brier-ratio",
        type=float,
        default=1.0,
        help="maximum validation Brier ratio allowed for response-scale mean calibration",
    )
    parser.add_argument(
        "--predictive-mean-calibration-max-log-loss-ratio",
        type=float,
        default=1.0,
        help="maximum validation log-loss ratio allowed for response-scale mean calibration",
    )
    parser.add_argument(
        "--predictive-mean-calibration-min-improvement",
        type=float,
        default=0.0,
        help="minimum validation score improvement required before selection",
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
    parser.add_argument(
        "--conditional-calibration-rare-weight", type=float, default=4.0
    )
    parser.add_argument(
        "--conditional-calibration-intermediate-weight", type=float, default=2.0
    )
    parser.add_argument(
        "--conditional-calibration-common-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--conditional-calibration-support-quantile", type=float, default=0.99
    )
    parser.add_argument(
        "--conditional-calibration-fallback-strength", type=float, default=2.0
    )
    parser.add_argument(
        "--conditional-calibration-ood-uncertainty-strength",
        type=float,
        default=0.75,
        help="support-excess coefficient for bounded OOD posterior-scale inflation",
    )
    parser.add_argument(
        "--conditional-calibration-ood-uncertainty-max-multiplier",
        type=float,
        default=4.0,
        help="maximum extra posterior-scale multiplier from OOD uncertainty inflation",
    )
    parser.add_argument(
        "--conditional-calibration-ood-objective",
        choices=[
            "none",
            "support_excess_rank_coverage",
            "support_effect_gated_rank_coverage",
        ],
        default="none",
        help="fit a learned OOD uncertainty curve from held-out OOD simulations",
    )
    parser.add_argument(
        "--conditional-calibration-ood-datasets",
        type=int,
        default=0,
        help="held-out OOD datasets per regime used by the learned OOD objective",
    )
    parser.add_argument(
        "--conditional-calibration-ood-hard-target-multiplier",
        type=int,
        default=1,
        help=(
            "for effect-size and combined-shift OOD calibration, keep this many "
            "times more target-domain datasets after hard near-boundary selection"
        ),
    )
    parser.add_argument(
        "--conditional-calibration-ood-hard-target-candidate-multiplier",
        type=int,
        default=2,
        help=(
            "candidate-pool multiplier used before selecting hard target-domain "
            "OOD calibration datasets"
        ),
    )
    parser.add_argument(
        "--conditional-calibration-ood-objective-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--conditional-calibration-ood-in-domain-gate-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--conditional-calibration-ood-objective-epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--external-monotone-datasets",
        type=int,
        default=0,
        help=(
            "held-out OOD datasets per regime for the external context-stratified "
            "monotone scale competitor"
        ),
    )
    parser.add_argument(
        "--external-monotone-max-multiplier",
        type=float,
        default=2.0,
        help="maximum extra multiplier applied by the external monotone competitor",
    )
    parser.add_argument(
        "--external-monotone-min-ood-gain",
        type=float,
        default=0.005,
        help="minimum held-out mean OOD coverage gain required to select the wrapper",
    )
    parser.add_argument(
        "--external-monotone-min-combined-gain",
        type=float,
        default=0.005,
        help="minimum held-out combined-shift coverage gain required to select the wrapper",
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
    if args.rare_calibration_datasets < 0:
        parser.error("--rare-calibration-datasets must be non-negative")
    if args.rare_validation_datasets < 0:
        parser.error("--rare-validation-datasets must be non-negative")
    if not 0.0 < args.rare_calibration_detection_probability <= 1.0:
        parser.error("--rare-calibration-detection-probability must be in (0, 1]")
    if not 0.0 < args.rare_calibration_sample_fraction <= 1.0:
        parser.error("--rare-calibration-sample-fraction must be in (0, 1]")
    if args.probit_anchor_iterations <= 0:
        parser.error("--probit-anchor-iterations must be positive")
    if args.probit_anchor_prior_precision <= 0.0:
        parser.error("--probit-anchor-prior-precision must be positive")
    if args.probit_anchor_eta_clip <= 0.0:
        parser.error("--probit-anchor-eta-clip must be positive")
    if args.rank_mean_penalty_weight < 0.0:
        parser.error("--rank-mean-penalty-weight must be non-negative")
    if not 0.0 < args.rank_mean_penalty_holdout_fraction < 1.0:
        parser.error(
            "--rank-mean-penalty-holdout-fraction must be between zero and one"
        )
    if args.rank_mean_penalty_holdout_folds < 1:
        parser.error("--rank-mean-penalty-holdout-folds must be at least one")
    if not 0.0 < args.rank_mean_penalty_crossfit_min_agreement <= 1.0:
        parser.error("--rank-mean-penalty-crossfit-min-agreement must be in (0, 1]")
    if not 0.0 <= args.rank_mean_penalty_start_fraction < 1.0:
        parser.error("--rank-mean-penalty-start-fraction must be in [0, 1)")
    if args.rank_mean_penalty_design_guard_weight < 0.0:
        parser.error("--rank-mean-penalty-design-guard-weight must be non-negative")
    if not 0.0 < args.rank_mean_penalty_design_guard_floor < 1.0:
        parser.error(
            "--rank-mean-penalty-design-guard-floor must be between zero and one"
        )
    if args.rank_mean_penalty_signed_mean_weight < 0.0:
        parser.error("--rank-mean-penalty-signed-mean-weight must be non-negative")
    if args.rank_mean_penalty_design_mean_guard_weight < 0.0:
        parser.error(
            "--rank-mean-penalty-design-mean-guard-weight must be non-negative"
        )
    if args.rank_mean_penalty_design_mean_guard_tolerance <= 0.0:
        parser.error("--rank-mean-penalty-design-mean-guard-tolerance must be positive")
    if args.conditional_calibration_epochs <= 0:
        parser.error("--conditional-calibration-epochs must be positive")
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
    if args.predictive_mean_calibration_max_rmse_ratio < 1.0:
        parser.error(
            "--predictive-mean-calibration-max-rmse-ratio must be at least one"
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
    if (
        args.predictive_mean_calibration
        in {"probit_response_affine", "probit_transfer_response_affine"}
        and args.suite != ["probit"]
    ):
        parser.error(
            "--predictive-mean-calibration probit response methods currently "
            "requires exactly --suite probit"
        )
    if args.conditional_calibration_learning_rate <= 0.0:
        parser.error("--conditional-calibration-learning-rate must be positive")
    if args.conditional_calibration_regularization < 0.0:
        parser.error("--conditional-calibration-regularization must be non-negative")
    if args.conditional_calibration_rank_penalty_weight < 0.0:
        parser.error(
            "--conditional-calibration-rank-penalty-weight must be non-negative"
        )
    if (
        min(
            args.conditional_calibration_rare_weight,
            args.conditional_calibration_intermediate_weight,
            args.conditional_calibration_common_weight,
        )
        <= 0.0
    ):
        parser.error("conditional prevalence weights must be positive")
    if not 0.5 < args.conditional_calibration_support_quantile < 1.0:
        parser.error(
            "--conditional-calibration-support-quantile must be between 0.5 and 1"
        )
    if args.conditional_calibration_fallback_strength < 0.0:
        parser.error("--conditional-calibration-fallback-strength must be non-negative")
    if args.conditional_calibration_ood_uncertainty_strength < 0.0:
        parser.error(
            "--conditional-calibration-ood-uncertainty-strength must be non-negative"
        )
    if args.conditional_calibration_ood_uncertainty_max_multiplier < 1.0:
        parser.error(
            "--conditional-calibration-ood-uncertainty-max-multiplier must be at least one"
        )
    if args.conditional_calibration_ood_datasets < 0:
        parser.error("--conditional-calibration-ood-datasets must be non-negative")
    if args.conditional_calibration_ood_hard_target_multiplier < 1:
        parser.error(
            "--conditional-calibration-ood-hard-target-multiplier must be at least one"
        )
    if args.conditional_calibration_ood_hard_target_candidate_multiplier < 1:
        parser.error(
            "--conditional-calibration-ood-hard-target-candidate-multiplier "
            "must be at least one"
        )
    if args.conditional_calibration_ood_objective != "none":
        if args.conditional_calibration_ood_datasets <= 0:
            parser.error(
                "--conditional-calibration-ood-objective requires "
                "--conditional-calibration-ood-datasets"
            )
        if args.conditional_calibration_ood_objective_epochs <= 0:
            parser.error(
                "--conditional-calibration-ood-objective-epochs must be positive"
            )
    if args.conditional_calibration_ood_objective_weight < 0.0:
        parser.error(
            "--conditional-calibration-ood-objective-weight must be non-negative"
        )
    if args.conditional_calibration_ood_in_domain_gate_weight < 0.0:
        parser.error(
            "--conditional-calibration-ood-in-domain-gate-weight must be non-negative"
        )
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
        "rare_calibration_datasets": args.rare_calibration_datasets,
        "rare_validation_datasets": args.rare_validation_datasets,
        "rare_calibration_intercept_mean": args.rare_calibration_intercept_mean,
        "rare_calibration_regimes": args.rare_calibration_regimes,
        "rare_calibration_detection_probability": (
            args.rare_calibration_detection_probability
        ),
        "rare_calibration_sample_fraction": args.rare_calibration_sample_fraction,
        "epochs": args.epochs,
        "posterior_family_policy": args.posterior_family,
        "probit_anchor_policy": args.probit_anchor,
        "mse_weight": args.mse_weight,
        "calibration_enabled": not bool(args.disable_calibration),
        "coefficient_calibration": args.coefficient_calibration,
        "predictive_mean_calibration": args.predictive_mean_calibration,
        "predictive_mean_calibration_validation_datasets": (
            args.predictive_mean_calibration_validation_datasets
        ),
        "predictive_mean_calibration_max_brier_ratio": (
            args.predictive_mean_calibration_max_brier_ratio
        ),
        "predictive_mean_calibration_max_log_loss_ratio": (
            args.predictive_mean_calibration_max_log_loss_ratio
        ),
        "predictive_mean_calibration_min_improvement": (
            args.predictive_mean_calibration_min_improvement
        ),
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
                                "conditional_rank_aware_anchor_scale",
                                "external_context_monotone_scale",
                            }:
                                calibration_result = (
                                    ConditionalBetaScaleCalibration.from_metadata(
                                        metadata
                                    )
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
                probit_anchor=args.probit_anchor,
                probit_anchor_iterations=args.probit_anchor_iterations,
                probit_anchor_prior_precision=args.probit_anchor_prior_precision,
                probit_anchor_eta_clip=args.probit_anchor_eta_clip,
            )
            training_history = engine.fit(
                train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.model_seed + suite_idx,
                mse_weight=args.mse_weight,
                rank_mean_penalty_weight=args.rank_mean_penalty_weight,
                rank_mean_penalty_holdout_fraction=args.rank_mean_penalty_holdout_fraction,
                rank_mean_penalty_holdout_folds=args.rank_mean_penalty_holdout_folds,
                rank_mean_penalty_crossfit_min_agreement=(
                    args.rank_mean_penalty_crossfit_min_agreement
                ),
                rank_mean_penalty_start_fraction=args.rank_mean_penalty_start_fraction,
                rank_mean_penalty_design_guard_weight=(
                    args.rank_mean_penalty_design_guard_weight
                ),
                rank_mean_penalty_design_guard_floor=(
                    args.rank_mean_penalty_design_guard_floor
                ),
                rank_mean_penalty_signed_mean_weight=(
                    args.rank_mean_penalty_signed_mean_weight
                ),
                rank_mean_penalty_design_mean_guard_weight=(
                    args.rank_mean_penalty_design_mean_guard_weight
                ),
                rank_mean_penalty_design_mean_guard_tolerance=(
                    args.rank_mean_penalty_design_mean_guard_tolerance
                ),
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
        predictive_mean_calibration_result = None
        if args.disable_calibration:
            calibration_result = None
            posterior = uncalibrated_posterior
            predictive_posterior = uncalibrated_posterior
            predictive_calibration_result = None
            ood_hard_pool_diagnostics = []
        else:
            calibration_data = fixed_shape_training_data(calibration)
            calibration_posterior = engine.predict_beta_posterior(calibration_data)
            mean_validation_posterior = None
            mean_validation_beta = None
            mean_transfer_validation_batches = None
            if args.predictive_mean_calibration != "none":
                mean_validation = _datasets(
                    count=args.predictive_mean_calibration_validation_datasets,
                    n_sites=args.n_sites,
                    n_species=args.n_species,
                    distribution=distribution,
                    seed=distribution_seed(args.seed, distribution, delta=750),
                )
                mean_validation_data = fixed_shape_training_data(mean_validation)
                mean_validation_posterior = engine.predict_beta_posterior(
                    mean_validation_data
                )
                mean_validation_beta = mean_validation_data.Beta
                if (
                    args.predictive_mean_calibration
                    == "probit_transfer_response_affine"
                ):
                    mean_transfer_validation_batches = []
                    for regime_idx, regime in enumerate(args.ood_regimes):
                        regime_count = _balanced_regime_count(
                            args.predictive_mean_calibration_validation_datasets,
                            len(args.ood_regimes),
                            regime_idx,
                        )
                        if regime_count <= 0:
                            continue
                        transfer_validation = [
                            simulate_fixed_effect_ood_dataset(
                                n_sites=args.n_sites,
                                n_species=args.n_species,
                                distribution=distribution,
                                regime=regime,
                                seed=distribution_seed(
                                    args.seed,
                                    distribution,
                                    delta=9750 + 10_000 * regime_idx + idx,
                                ),
                            )
                            for idx in range(regime_count)
                        ]
                        transfer_data = fixed_shape_training_data(
                            transfer_validation
                        )
                        mean_transfer_validation_batches.append(
                            BetaResponseCalibrationBatch(
                                posterior=engine.predict_beta_posterior(
                                    transfer_data
                                ),
                                X=transfer_data.X,
                                Y=transfer_data.Y,
                                label=f"transfer_validation:{regime}",
                            )
                        )
            predictive_calibration_result = fit_beta_scale_calibration(
                calibration_posterior,
                calibration_data.Beta,
                nominal_level=0.95,
                distribution=distribution,
                predictive_X=calibration_data.X if distribution == "poisson" else None,
                poisson_eta_clip=(-6.0, 6.0) if distribution == "poisson" else None,
                predictive_seed=args.model_seed + suite_idx + 50,
            )
            ood_hard_pool_diagnostics = []
            if args.coefficient_calibration in {"conditional", "external_monotone"}:
                rare_calibration_batches = None
                rare_validation_batches = None
                if args.rare_calibration_datasets > 0:
                    rare_calibration_batches = []
                    for regime_idx, regime in enumerate(args.rare_calibration_regimes):
                        regime_count = _balanced_regime_count(
                            args.rare_calibration_datasets,
                            len(args.rare_calibration_regimes),
                            regime_idx,
                        )
                        if regime_count <= 0:
                            continue
                        rare_calibration = [
                            _simulate_rare_calibration_dataset(
                                n_sites=args.n_sites,
                                n_species=args.n_species,
                                distribution=distribution,
                                regime=regime,
                                seed=distribution_seed(
                                    args.seed,
                                    distribution,
                                    delta=4500 + 10_000 * regime_idx + idx,
                                ),
                                intercept_mean=args.rare_calibration_intercept_mean,
                                detection_probability=(
                                    args.rare_calibration_detection_probability
                                ),
                                sample_fraction=args.rare_calibration_sample_fraction,
                            )
                            for idx in range(regime_count)
                        ]
                        rare_data = fixed_shape_training_data(rare_calibration)
                        rare_calibration_batches.append(
                            ConditionalBetaOODCalibrationBatch(
                                posterior=engine.predict_beta_posterior(rare_data),
                                beta_true=rare_data.Beta,
                                X=rare_data.X,
                                Y=rare_data.Y,
                                label=f"rare_balanced:{regime}",
                            )
                        )
                if args.rare_validation_datasets > 0:
                    rare_validation_batches = []
                    for regime_idx, regime in enumerate(args.rare_calibration_regimes):
                        regime_count = _balanced_regime_count(
                            args.rare_validation_datasets,
                            len(args.rare_calibration_regimes),
                            regime_idx,
                        )
                        if regime_count <= 0:
                            continue
                        rare_validation = [
                            _simulate_rare_calibration_dataset(
                                n_sites=args.n_sites,
                                n_species=args.n_species,
                                distribution=distribution,
                                regime=regime,
                                seed=distribution_seed(
                                    args.seed,
                                    distribution,
                                    delta=8500 + 10_000 * regime_idx + idx,
                                ),
                                intercept_mean=args.rare_calibration_intercept_mean,
                                detection_probability=(
                                    args.rare_calibration_detection_probability
                                ),
                                sample_fraction=args.rare_calibration_sample_fraction,
                            )
                            for idx in range(regime_count)
                        ]
                        rare_validation_data = fixed_shape_training_data(
                            rare_validation
                        )
                        rare_validation_batches.append(
                            ConditionalBetaOODCalibrationBatch(
                                posterior=engine.predict_beta_posterior(
                                    rare_validation_data
                                ),
                                beta_true=rare_validation_data.Beta,
                                X=rare_validation_data.X,
                                Y=rare_validation_data.Y,
                                label=f"rare_validation:{regime}",
                            )
                        )
                ood_calibration_batches = None
                if args.conditional_calibration_ood_objective != "none":
                    ood_calibration_batches = []
                    for regime_idx, regime in enumerate(args.ood_regimes):
                        hard_pool_diagnostic: dict[str, object] = {}
                        ood_calibration = _ood_datasets(
                            engine=engine,
                            count=args.conditional_calibration_ood_datasets,
                            n_sites=args.n_sites,
                            n_species=args.n_species,
                            distribution=distribution,
                            regime=regime,
                            seed=distribution_seed(
                                args.seed,
                                distribution,
                                delta=1500 + 10_000 * (regime_idx + 1),
                            ),
                            hard_target_multiplier=(
                                args.conditional_calibration_ood_hard_target_multiplier
                            ),
                            hard_target_candidate_multiplier=(
                                args.conditional_calibration_ood_hard_target_candidate_multiplier
                            ),
                            hard_pool_diagnostic=hard_pool_diagnostic,
                        )
                        split_ood = (
                            regime in {"effect_size_shift", "combined_shift"}
                            and args.conditional_calibration_ood_hard_target_multiplier
                            > 1
                        )
                        hard_pool_score_arrays = (
                            _hard_pool_score_arrays(
                                engine=engine,
                                datasets=ood_calibration,
                                regime=regime,
                            )
                            if split_ood
                            else None
                        )
                        if split_ood and hard_pool_score_arrays is not None:
                            ood_groups = _matched_hard_pool_dataset_groups(
                                ood_calibration,
                                score_arrays=hard_pool_score_arrays,
                            )
                        else:
                            ood_groups = _ood_dataset_groups(
                                ood_calibration,
                                split=split_ood,
                            )
                        if hard_pool_diagnostic:
                            hard_pool_diagnostic["matched_train_validation"] = (
                                _hard_pool_group_diagnostics(
                                    ood_calibration,
                                    ood_groups,
                                    score_arrays=hard_pool_score_arrays,
                                )
                            )
                            ood_hard_pool_diagnostics.append(hard_pool_diagnostic)
                        for ood_group in ood_groups:
                            ood_data = fixed_shape_training_data(ood_group)
                            ood_calibration_batches.append(
                                ConditionalBetaOODCalibrationBatch(
                                    posterior=engine.predict_beta_posterior(ood_data),
                                    beta_true=ood_data.Beta,
                                    X=ood_data.X,
                                    Y=ood_data.Y,
                                    label=regime,
                                )
                            )
                external_monotone_batches = None
                if args.coefficient_calibration == "external_monotone":
                    external_monotone_batches = []
                    for regime_idx, regime in enumerate(args.ood_regimes):
                        external_ood = [
                            simulate_fixed_effect_ood_dataset(
                                n_sites=args.n_sites,
                                n_species=args.n_species,
                                distribution=distribution,
                                regime=regime,
                                seed=distribution_seed(
                                    args.seed,
                                    distribution,
                                    delta=6500 + 10_000 * (regime_idx + 1) + idx,
                                ),
                            )
                            for idx in range(args.external_monotone_datasets)
                        ]
                        external_data = fixed_shape_training_data(external_ood)
                        external_monotone_batches.append(
                            ConditionalBetaOODCalibrationBatch(
                                posterior=engine.predict_beta_posterior(external_data),
                                beta_true=external_data.Beta,
                                X=external_data.X,
                                Y=external_data.Y,
                                label=regime,
                            )
                        )
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
                    ood_uncertainty_strength=(
                        args.conditional_calibration_ood_uncertainty_strength
                    ),
                    ood_uncertainty_max_multiplier=(
                        args.conditional_calibration_ood_uncertainty_max_multiplier
                    ),
                    rare_calibration_batches=rare_calibration_batches,
                    rare_validation_batches=rare_validation_batches,
                    ood_calibration_batches=ood_calibration_batches,
                    ood_objective=args.conditional_calibration_ood_objective,
                    ood_objective_weight=(
                        args.conditional_calibration_ood_objective_weight
                    ),
                    ood_in_domain_gate_weight=(
                        args.conditional_calibration_ood_in_domain_gate_weight
                    ),
                    ood_objective_epochs=(
                        args.conditional_calibration_ood_objective_epochs
                    ),
                )
                if args.coefficient_calibration == "external_monotone":
                    calibration_result = fit_external_context_monotone_calibration(
                        calibration_result,
                        calibration_posterior,
                        calibration_data.Beta,
                        X=calibration_data.X,
                        Y=calibration_data.Y,
                        distribution=distribution,
                        coefficient_names=engine.covariate_names,
                        ood_validation_batches=external_monotone_batches,
                        nominal_level=0.95,
                        max_external_multiplier=(
                            args.external_monotone_max_multiplier
                        ),
                        min_mean_ood_gain=args.external_monotone_min_ood_gain,
                        min_combined_shift_gain=(
                            args.external_monotone_min_combined_gain
                        ),
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
            if args.predictive_mean_calibration != "none":
                if args.predictive_mean_calibration == "probit_response_affine":
                    if mean_validation_posterior is None or mean_validation_beta is None:
                        raise RuntimeError(
                            "predictive mean validation data was not prepared"
                        )
                    predictive_mean_calibration_result = (
                        fit_beta_response_mean_calibration(
                            calibration_posterior,
                            calibration_X=calibration_data.X,
                            calibration_Y=calibration_data.Y,
                            validation_posterior=mean_validation_posterior,
                            validation_X=mean_validation_data.X,
                            validation_Y=mean_validation_data.Y,
                            distribution=distribution,
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
                    )
                elif (
                    args.predictive_mean_calibration
                    == "probit_transfer_response_affine"
                ):
                    if (
                        mean_validation_posterior is None
                        or mean_transfer_validation_batches is None
                    ):
                        raise RuntimeError(
                            "predictive mean transfer validation data was not prepared"
                        )
                    predictive_mean_calibration_result = (
                        fit_beta_transfer_response_mean_calibration(
                            calibration_posterior,
                            calibration_X=calibration_data.X,
                            calibration_Y=calibration_data.Y,
                            source_validation_posterior=mean_validation_posterior,
                            source_validation_X=mean_validation_data.X,
                            source_validation_Y=mean_validation_data.Y,
                            transfer_validation_batches=(
                                mean_transfer_validation_batches
                            ),
                            distribution=distribution,
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
                    predictive_mean_calibration_result = (
                        fit_beta_predictive_mean_calibration(
                            calibration_posterior,
                            calibration_data.Beta,
                            validation_posterior=mean_validation_posterior,
                            validation_beta_true=mean_validation_beta,
                            distribution=distribution,
                            method=args.predictive_mean_calibration,
                            max_validation_rmse_ratio=(
                                args.predictive_mean_calibration_max_rmse_ratio
                            ),
                            min_validation_rmse_improvement=(
                                args.predictive_mean_calibration_min_improvement
                            ),
                        )
                    )
                predictive_posterior = apply_beta_predictive_mean_calibration(
                    predictive_posterior,
                    predictive_mean_calibration_result,
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
                "predictive_mean_calibration": (
                    None
                    if predictive_mean_calibration_result is None
                    else predictive_mean_calibration_result.to_metadata()
                ),
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
            "probit_anchor": engine.model.probit_anchor,
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
                "rank_mean_penalty": training_history.rank_mean_penalty,
                "rank_mean_penalty_weight": args.rank_mean_penalty_weight,
                "rank_mean_penalty_holdout_fraction": args.rank_mean_penalty_holdout_fraction,
                "rank_mean_penalty_holdout_folds": (
                    args.rank_mean_penalty_holdout_folds
                ),
                "rank_mean_penalty_crossfit_min_agreement": (
                    args.rank_mean_penalty_crossfit_min_agreement
                ),
                "rank_mean_penalty_start_fraction": args.rank_mean_penalty_start_fraction,
                "rank_mean_penalty_design_guard_weight": (
                    args.rank_mean_penalty_design_guard_weight
                ),
                "rank_mean_penalty_design_guard_floor": (
                    args.rank_mean_penalty_design_guard_floor
                ),
                "rank_mean_penalty_signed_mean_weight": (
                    args.rank_mean_penalty_signed_mean_weight
                ),
                "rank_mean_penalty_design_mean_guard_weight": (
                    args.rank_mean_penalty_design_mean_guard_weight
                ),
                "rank_mean_penalty_design_mean_guard_tolerance": (
                    args.rank_mean_penalty_design_mean_guard_tolerance
                ),
            }
        if args.rare_calibration_datasets > 0:
            record["rare_calibration"] = {
                "datasets": args.rare_calibration_datasets,
                "validation_datasets": args.rare_validation_datasets,
                "intercept_mean": args.rare_calibration_intercept_mean,
                "regimes": args.rare_calibration_regimes,
                "detection_probability": (args.rare_calibration_detection_probability),
                "sample_fraction": args.rare_calibration_sample_fraction,
            }
        if distribution_sbc_rows:
            record["sbc_diagnostics"] = str(sbc_path)
        if calibration_result is not None:
            record["calibration"] = calibration_result.to_metadata()
        if ood_hard_pool_diagnostics:
            record["ood_hard_pool_diagnostics"] = ood_hard_pool_diagnostics
        if predictive_calibration_result is not None:
            record["predictive_calibration"] = (
                predictive_calibration_result.to_metadata()
            )
        if predictive_mean_calibration_result is not None:
            record["predictive_mean_calibration"] = (
                predictive_mean_calibration_result.to_metadata()
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


def _ood_datasets(
    *,
    engine: NeuralHmscInference,
    count: int,
    n_sites: int,
    n_species: int,
    distribution: str,
    regime: str,
    seed: int,
    hard_target_multiplier: int = 1,
    hard_target_candidate_multiplier: int = 1,
    hard_pool_diagnostic: dict[str, object] | None = None,
) -> list[FixedEffectDataset]:
    """Return OOD calibration datasets, optionally enriched for near misses."""
    if count <= 0:
        return []
    target_regime = regime in {"effect_size_shift", "combined_shift"}
    keep_count = int(count)
    candidate_count = int(count)
    generation_count = int(count)
    if target_regime and hard_target_multiplier > 1:
        keep_count = int(count) * int(hard_target_multiplier)
        candidate_count = keep_count * max(1, int(hard_target_candidate_multiplier))
        generation_count = candidate_count * 3
    candidate_context = (
        "low_overlap" if target_regime and hard_target_multiplier > 1 else "default"
    )
    candidates = [
        simulate_fixed_effect_ood_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution=distribution,
            regime=regime,
            seed=seed + idx,
            candidate_context=candidate_context,
        )
        for idx in range(generation_count)
    ]
    if not target_regime or keep_count >= len(candidates):
        if target_regime and hard_pool_diagnostic is not None:
            score_arrays = _hard_pool_score_arrays(
                engine=engine,
                datasets=candidates,
                regime=regime,
            )
            hard_pool_diagnostic.update(
                _hard_pool_selection_diagnostics(
                    regime=regime,
                    requested_count=count,
                    keep_count=len(candidates),
                    candidate_count=len(candidates),
                    hard_target_multiplier=hard_target_multiplier,
                    hard_target_candidate_multiplier=hard_target_candidate_multiplier,
                    score_arrays=score_arrays,
                    selected_indices=np.arange(len(candidates)),
                    selection_applied=False,
                )
            )
        return candidates
    generated_score_arrays = _hard_pool_score_arrays(
        engine=engine,
        datasets=candidates,
        regime=regime,
    )
    candidate_pool_indices = np.arange(len(candidates), dtype=int)
    candidate_generation_diagnostics: dict[str, object] = {}
    if generation_count > candidate_count:
        candidate_pool = _select_low_overlap_candidate_pool_indices(
            generated_score_arrays,
            keep_count=candidate_count,
        )
        candidate_pool_indices = candidate_pool["selected_indices"]
        candidate_generation_diagnostics = candidate_pool["diagnostics"]
        candidate_generation_diagnostics["candidate_context"] = candidate_context
        candidates = [candidates[int(index)] for index in candidate_pool_indices]
        score_arrays = _subset_score_arrays(
            generated_score_arrays,
            candidate_pool_indices,
        )
    else:
        score_arrays = generated_score_arrays
    selection = _select_constrained_hard_pool_indices(
        score_arrays,
        keep_count=keep_count,
    )
    selected_indices = selection["selected_indices"]
    if hard_pool_diagnostic is not None:
        hard_pool_diagnostic.update(
            _hard_pool_selection_diagnostics(
                regime=regime,
                requested_count=count,
                keep_count=keep_count,
                candidate_count=len(candidates),
                hard_target_multiplier=hard_target_multiplier,
                hard_target_candidate_multiplier=hard_target_candidate_multiplier,
                score_arrays=score_arrays,
                selected_indices=selected_indices,
                selection_applied=True,
                constraint_diagnostics=selection["diagnostics"],
                candidate_generation_diagnostics=candidate_generation_diagnostics,
            )
        )
    return [candidates[int(index)] for index in selected_indices]


def _ood_dataset_groups(
    datasets: list[FixedEffectDataset],
    *,
    split: bool,
    scores: np.ndarray | None = None,
) -> list[list[FixedEffectDataset]]:
    """Return one or two OOD dataset groups for calibration batch construction."""
    if not split or len(datasets) < 2:
        return [datasets]
    if scores is not None:
        return _matched_score_dataset_groups(datasets, scores=np.asarray(scores))
    groups = [datasets[::2], datasets[1::2]]
    return [group for group in groups if group]


def _matched_score_dataset_groups(
    datasets: list[FixedEffectDataset],
    *,
    scores: np.ndarray,
) -> list[list[FixedEffectDataset]]:
    """Split datasets into two groups with similar hard-pool score totals."""
    if len(datasets) < 2:
        return [datasets]
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.shape[0] != len(datasets):
        raise ValueError("scores must have one value per OOD dataset")
    order = np.argsort(-values)
    groups: list[list[FixedEffectDataset]] = [[], []]
    totals = [0.0, 0.0]
    max_size = int(np.ceil(len(datasets) / 2))
    for index in order:
        choices = [idx for idx, group in enumerate(groups) if len(group) < max_size]
        target = min(choices, key=lambda idx: (totals[idx], len(groups[idx]), idx))
        groups[target].append(datasets[int(index)])
        totals[target] += float(values[int(index)])
    return [group for group in groups if group]


def _matched_hard_pool_dataset_groups(
    datasets: list[FixedEffectDataset],
    *,
    score_arrays: dict[str, np.ndarray],
) -> list[list[FixedEffectDataset]]:
    """Split target hard pools while balancing raw difficulty and overlap."""
    if len(datasets) < 2:
        return [datasets]
    raw = np.asarray(score_arrays["raw_near_boundary_score"], dtype=float).reshape(-1)
    overlap = np.asarray(score_arrays["overlap_proxy"], dtype=float).reshape(-1)
    score = np.asarray(score_arrays["score"], dtype=float).reshape(-1)
    if raw.shape[0] != len(datasets) or overlap.shape[0] != len(datasets):
        raise ValueError("score_arrays must have one value per OOD dataset")
    balance = np.column_stack(
        [
            _standardize(raw),
            _standardize(overlap),
            0.5 * _standardize(score),
        ]
    )
    priority = np.abs(balance[:, 0]) + np.abs(balance[:, 1]) + np.abs(balance[:, 2])
    order = np.argsort(-priority)
    groups: list[list[FixedEffectDataset]] = [[], []]
    totals = [
        np.zeros(balance.shape[1], dtype=float),
        np.zeros(balance.shape[1], dtype=float),
    ]
    max_size = int(np.ceil(len(datasets) / 2))
    for index in order:
        choices = [idx for idx, group in enumerate(groups) if len(group) < max_size]
        target = min(
            choices,
            key=lambda idx: (
                float(np.sum(np.square(totals[idx] + balance[int(index)]))),
                len(groups[idx]),
                idx,
            ),
        )
        groups[target].append(datasets[int(index)])
        totals[target] = totals[target] + balance[int(index)]
    return [group for group in groups if group]


def _select_constrained_hard_pool_indices(
    score_arrays: dict[str, np.ndarray],
    *,
    keep_count: int,
) -> dict[str, object]:
    """Select hard-pool candidates with raw misses first and low overlap second."""
    raw = np.asarray(score_arrays["raw_near_boundary_score"], dtype=float).reshape(-1)
    overlap = np.asarray(score_arrays["overlap_proxy"], dtype=float).reshape(-1)
    score = np.asarray(score_arrays["score"], dtype=float).reshape(-1)
    n_candidates = int(raw.shape[0])
    keep = min(max(0, int(keep_count)), n_candidates)
    if keep == 0:
        return {
            "selected_indices": np.zeros(0, dtype=int),
            "diagnostics": {
                "raw_threshold": None,
                "overlap_threshold": None,
                "eligible_count": 0,
                "relaxation_steps": 0,
                "selection_priority_summary": _numeric_summary(np.zeros(0)),
            },
        }
    if overlap.shape[0] != n_candidates or score.shape[0] != n_candidates:
        raise ValueError("score_arrays must have consistent candidate counts")
    raw_quantile = 0.50
    overlap_quantile = 0.50
    relaxation_steps = 0
    eligible = np.zeros(n_candidates, dtype=bool)
    while True:
        raw_threshold = float(np.quantile(raw, raw_quantile))
        overlap_threshold = float(np.quantile(overlap, overlap_quantile))
        eligible = (raw >= raw_threshold) & (overlap <= overlap_threshold)
        if int(np.sum(eligible)) >= keep:
            break
        if overlap_quantile < 0.95:
            overlap_quantile = min(0.95, overlap_quantile + 0.05)
        elif raw_quantile > 0.05:
            raw_quantile = max(0.05, raw_quantile - 0.05)
        else:
            break
        relaxation_steps += 1
    raw_rank = _percentile_rank(raw)
    low_overlap_rank = 1.0 - _percentile_rank(overlap)
    score_rank = _percentile_rank(score)
    priority = 0.45 * raw_rank + 0.45 * low_overlap_rank + 0.10 * score_rank
    candidate_indices = np.flatnonzero(eligible)
    if candidate_indices.size < keep:
        candidate_indices = np.arange(n_candidates)
    ordered = candidate_indices[np.argsort(-priority[candidate_indices])]
    selected = ordered[:keep].astype(int)
    diagnostics = {
        "raw_threshold": raw_threshold,
        "overlap_threshold": overlap_threshold,
        "raw_threshold_quantile": float(raw_quantile),
        "overlap_threshold_quantile": float(overlap_quantile),
        "eligible_count": int(np.sum(eligible)),
        "relaxation_steps": int(relaxation_steps),
        "used_fallback_pool": bool(np.sum(eligible) < keep),
        "eligible_summaries": _hard_pool_subset_summary(
            score_arrays, np.flatnonzero(eligible)
        ),
        "selection_priority_summary": _numeric_summary(priority[selected]),
    }
    return {"selected_indices": selected, "diagnostics": diagnostics}


def _select_low_overlap_candidate_pool_indices(
    score_arrays: dict[str, np.ndarray],
    *,
    keep_count: int,
) -> dict[str, object]:
    """Build a candidate pool from low-overlap contexts before hard selection."""
    raw = np.asarray(score_arrays["raw_near_boundary_score"], dtype=float).reshape(-1)
    overlap = np.asarray(score_arrays["overlap_proxy"], dtype=float).reshape(-1)
    score = np.asarray(score_arrays["score"], dtype=float).reshape(-1)
    n_candidates = int(raw.shape[0])
    keep = min(max(0, int(keep_count)), n_candidates)
    if keep == 0:
        return {
            "selected_indices": np.zeros(0, dtype=int),
            "diagnostics": {
                "generated_count": n_candidates,
                "candidate_pool_count": 0,
                "eligible_count": 0,
                "used_fallback_pool": False,
            },
        }
    if overlap.shape[0] != n_candidates or score.shape[0] != n_candidates:
        raise ValueError("score_arrays must have consistent candidate counts")
    raw_quantile = 0.40
    overlap_quantile = min(0.50, max(keep / max(n_candidates, 1), 0.15))
    relaxation_steps = 0
    eligible = np.zeros(n_candidates, dtype=bool)
    while True:
        raw_threshold = float(np.quantile(raw, raw_quantile))
        overlap_threshold = float(np.quantile(overlap, overlap_quantile))
        eligible = (overlap <= overlap_threshold) & (raw >= raw_threshold)
        if int(np.sum(eligible)) >= keep:
            break
        if overlap_quantile < 0.80:
            overlap_quantile = min(0.80, overlap_quantile + 0.05)
        elif raw_quantile > 0.05:
            raw_quantile = max(0.05, raw_quantile - 0.05)
        else:
            break
        relaxation_steps += 1
    raw_rank = _percentile_rank(raw)
    low_overlap_rank = 1.0 - _percentile_rank(overlap)
    score_rank = _percentile_rank(score)
    priority = 0.65 * low_overlap_rank + 0.30 * raw_rank + 0.05 * score_rank
    candidate_indices = np.flatnonzero(eligible)
    used_fallback = candidate_indices.size < keep
    if used_fallback:
        candidate_indices = np.arange(n_candidates)
    ordered = candidate_indices[np.argsort(-priority[candidate_indices])]
    selected = ordered[:keep].astype(int)
    diagnostics = {
        "generated_count": int(n_candidates),
        "candidate_pool_count": int(selected.size),
        "raw_threshold": raw_threshold,
        "overlap_threshold": overlap_threshold,
        "raw_threshold_quantile": float(raw_quantile),
        "overlap_threshold_quantile": float(overlap_quantile),
        "eligible_count": int(np.sum(eligible)),
        "relaxation_steps": int(relaxation_steps),
        "used_fallback_pool": bool(used_fallback),
        "generated_summaries": _hard_pool_subset_summary(score_arrays),
        "eligible_summaries": _hard_pool_subset_summary(
            score_arrays, np.flatnonzero(eligible)
        ),
        "candidate_pool_summaries": _hard_pool_subset_summary(score_arrays, selected),
        "candidate_pool_priority_summary": _numeric_summary(priority[selected]),
    }
    return {"selected_indices": selected, "diagnostics": diagnostics}


def _subset_score_arrays(
    score_arrays: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    selected = np.asarray(indices, dtype=int)
    return {
        key: np.asarray(values, dtype=float).reshape(-1)[selected]
        for key, values in score_arrays.items()
    }


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return stable 0-1 ordinal ranks, where larger values get larger ranks."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return np.zeros(0, dtype=float)
    if array.size == 1:
        return np.ones(1, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, array.size)
    return ranks


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    scale = float(np.std(array))
    if scale <= 1e-12:
        return np.zeros_like(array, dtype=float)
    return (array - float(np.mean(array))) / scale


def _near_boundary_miss_scores(
    *,
    engine: NeuralHmscInference,
    datasets: list[FixedEffectDataset],
    regime: str | None = None,
) -> np.ndarray:
    """Score datasets by near misses, penalizing high-overlap target contexts."""
    return _hard_pool_score_arrays(
        engine=engine,
        datasets=datasets,
        regime=regime,
    )["score"]


def _hard_pool_score_arrays(
    *,
    engine: NeuralHmscInference,
    datasets: list[FixedEffectDataset],
    regime: str | None = None,
) -> dict[str, np.ndarray]:
    """Return per-dataset hard-pool score components for diagnostics."""
    if not datasets:
        empty = np.zeros(0, dtype=float)
        return {
            "score": empty,
            "raw_near_boundary_score": empty,
            "overlap_proxy": empty,
            "miss_rate": empty,
            "near_boundary_miss_rate": empty,
            "miss_excess_mean": empty,
            "absolute_z_mean": empty,
        }
    data = fixed_shape_training_data(datasets)
    posterior = engine.predict_beta_posterior(data)
    mean = np.asarray(posterior.mean, dtype=float)
    scale = np.maximum(np.asarray(posterior.scale, dtype=float), 1e-8)
    truth = np.asarray(data.Beta, dtype=float)
    absolute_z = np.abs(truth - mean) / scale
    miss = absolute_z - 1.959963984540054
    near_boundary = np.exp(-np.square(np.maximum(miss, 0.0) / 0.35))
    outside_gate = 1.0 / (1.0 + np.exp(-(miss / 0.03)))
    outside_near = near_boundary * outside_gate
    raw_score = np.mean(outside_near, axis=(1, 2))
    overlap = np.zeros(raw_score.shape[0], dtype=float)
    score = raw_score.copy()
    if regime in {"effect_size_shift", "combined_shift"}:
        overlap = _target_overlap_proxy(
            posterior_mean=mean,
            X=np.asarray(data.X, dtype=float),
            Y=np.asarray(data.Y, dtype=float),
            regime=str(regime),
        )
        score = score - 0.35 * overlap
    return {
        "score": score,
        "raw_near_boundary_score": raw_score,
        "overlap_proxy": overlap,
        "miss_rate": np.mean(miss > 0.0, axis=(1, 2)),
        "near_boundary_miss_rate": np.mean((miss > 0.0) & (miss <= 0.35), axis=(1, 2)),
        "miss_excess_mean": np.mean(np.maximum(miss, 0.0), axis=(1, 2)),
        "absolute_z_mean": np.mean(absolute_z, axis=(1, 2)),
    }


def _hard_pool_selection_diagnostics(
    *,
    regime: str,
    requested_count: int,
    keep_count: int,
    candidate_count: int,
    hard_target_multiplier: int,
    hard_target_candidate_multiplier: int,
    score_arrays: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    selection_applied: bool,
    constraint_diagnostics: dict[str, object] | None = None,
    candidate_generation_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return JSON-safe diagnostics for hard target-pool selection."""
    selected = np.asarray(selected_indices, dtype=int)
    return {
        "regime": str(regime),
        "requested_count": int(requested_count),
        "keep_count": int(keep_count),
        "candidate_count": int(candidate_count),
        "hard_target_multiplier": int(hard_target_multiplier),
        "hard_target_candidate_multiplier": int(hard_target_candidate_multiplier),
        "selection_applied": bool(selection_applied),
        "selected_indices": [int(index) for index in selected],
        "candidate_summaries": _hard_pool_subset_summary(score_arrays),
        "selected_summaries": _hard_pool_subset_summary(score_arrays, selected),
        "candidate_generation": candidate_generation_diagnostics or {},
        "selection_constraints": constraint_diagnostics or {},
    }


def _hard_pool_group_diagnostics(
    selected_datasets: list[FixedEffectDataset],
    groups: list[list[FixedEffectDataset]],
    *,
    score_arrays: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    """Return train/evaluation hard-pool balance summaries."""
    if score_arrays is None:
        return {"split": False, "groups": []}
    dataset_index = {id(dataset): idx for idx, dataset in enumerate(selected_datasets)}
    group_summaries = []
    score_totals = []
    for group_idx, group in enumerate(groups):
        indices = np.asarray(
            [dataset_index[id(dataset)] for dataset in group], dtype=int
        )
        scores = np.asarray(score_arrays["score"], dtype=float)[indices]
        score_totals.append(float(np.sum(scores)))
        group_summaries.append(
            {
                "group": int(group_idx),
                "dataset_indices": [int(index) for index in indices],
                "score_total": float(np.sum(scores)),
                "summaries": _hard_pool_subset_summary(score_arrays, indices),
            }
        )
    return {
        "split": len(groups) > 1,
        "score_total_difference": (
            float(abs(score_totals[0] - score_totals[1]))
            if len(score_totals) == 2
            else 0.0
        ),
        "groups": group_summaries,
    }


def _hard_pool_subset_summary(
    score_arrays: dict[str, np.ndarray],
    indices: np.ndarray | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize hard-pool score components for all or selected datasets."""
    summary: dict[str, dict[str, float | int | None]] = {}
    for key, values in score_arrays.items():
        array = np.asarray(values, dtype=float).reshape(-1)
        if indices is not None:
            array = array[np.asarray(indices, dtype=int)]
        summary[key] = _numeric_summary(array)
    return summary


def _numeric_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """Return compact JSON-safe distribution statistics."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "max": None,
            "std": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _target_overlap_proxy(
    *,
    posterior_mean: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    regime: str,
) -> np.ndarray:
    """Approximate target-domain contexts likely to overlap in-domain gates."""
    effect_signal = np.log1p(np.abs(np.asarray(posterior_mean, dtype=float)))
    covariates = np.asarray(X, dtype=float)
    if covariates.ndim == 3 and covariates.shape[-1] > 1:
        shifted_covariates = covariates[..., 1:]
    else:
        shifted_covariates = covariates
    support_proxy = np.maximum(
        np.mean(
            np.abs(shifted_covariates), axis=tuple(range(1, shifted_covariates.ndim))
        )
        - 1.0,
        0.0,
    )
    support_proxy = support_proxy.reshape((-1,) + (1,) * (effect_signal.ndim - 1))
    if regime == "effect_size_shift":
        high_effect = 1.0 / (1.0 + np.exp(-((effect_signal - 0.75) / 0.35)))
        support_close = 1.0 / (1.0 + np.exp(-((0.25 - support_proxy) / 0.25)))
        context = high_effect * support_close
    elif regime == "combined_shift":
        support_gate = 1.0 / (1.0 + np.exp(-((support_proxy - 0.20) / 0.35)))
        effect_gate = 1.0 / (1.0 + np.exp(-((effect_signal - 0.25) / 0.50)))
        response = np.asarray(Y, dtype=float)
        if response.ndim == 3:
            community = np.mean(response > 0, axis=1)
            community = np.expand_dims(community, axis=1)
        else:
            community = np.zeros_like(effect_signal, dtype=float)
        low_community = 1.0 / (1.0 + np.exp(-((0.45 - community) / 0.06)))
        context = support_gate * effect_gate * low_community
    else:
        return np.zeros(effect_signal.shape[0], dtype=float)
    return np.mean(context, axis=(1, 2))


def _balanced_regime_count(total: int, n_regimes: int, regime_index: int) -> int:
    base = int(total) // int(n_regimes)
    remainder = int(total) % int(n_regimes)
    return base + (1 if int(regime_index) < remainder else 0)


def _simulate_rare_calibration_dataset(
    *,
    n_sites: int,
    n_species: int,
    distribution: str,
    regime: str,
    seed: int,
    intercept_mean: float,
    detection_probability: float,
    sample_fraction: float,
) -> FixedEffectDataset:
    if regime == "intercept_shift":
        return simulate_fixed_effect_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution=distribution,
            seed=seed,
            intercept_mean=intercept_mean,
        )
    if regime == "low_detection":
        return simulate_fixed_effect_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution=distribution,
            seed=seed,
            detection_probability=detection_probability,
        )
    if regime == "small_sample":
        return simulate_fixed_effect_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution=distribution,
            seed=seed,
            sample_fraction=sample_fraction,
        )
    raise ValueError(f"unsupported rare calibration regime: {regime!r}")


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
        conditional_ood_inflation = None
        conditional_effect_signal = None
        conditional_mean_support = None
        if calibration is not None:
            if isinstance(calibration, ConditionalBetaScaleCalibration):
                conditional_mean_support = conditional_beta_mean_support_diagnostics(
                    uncalibrated, calibration
                )
                conditional_trust = conditional_beta_support_trust(
                    uncalibrated,
                    calibration,
                    X=data.X,
                    Y=data.Y,
                    distribution=distribution,
                    coefficient_names=engine.covariate_names,
                )
                conditional_effect_signal = conditional_beta_effect_size_signal(
                    uncalibrated,
                    calibration,
                )
                conditional_ood_inflation = conditional_beta_ood_uncertainty_inflation(
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
                    if conditional_ood_inflation is not None:
                        row.update(
                            {
                                "conditional_ood_uncertainty_inflation_mean": float(
                                    np.mean(conditional_ood_inflation)
                                ),
                                "conditional_ood_uncertainty_inflation_max": float(
                                    np.max(conditional_ood_inflation)
                                ),
                                "conditional_ood_uncertainty_inflated_fraction": float(
                                    np.mean(conditional_ood_inflation > 1.000001)
                                ),
                            }
                        )
                    if conditional_mean_support is not None:
                        row.update(conditional_mean_support)
                    if conditional_effect_signal is not None:
                        row.update(
                            {
                                "conditional_effect_size_signal_mean": float(
                                    np.mean(conditional_effect_signal)
                                ),
                                "conditional_effect_size_signal_max": float(
                                    np.max(conditional_effect_signal)
                                ),
                                "conditional_effect_size_signal_positive_fraction": float(
                                    np.mean(conditional_effect_signal > 0.0)
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
