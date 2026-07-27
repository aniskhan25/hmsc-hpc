"""Compact fixed evaluation of a simulation-trained MCMC-teacher residual head."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.model import HmscModel
from pyhmsc.neural.calibration import (
    BetaScaleCalibration,
    apply_beta_predictive_calibration,
)
from pyhmsc.neural.deployment import (
    FROZEN_COMPETITOR_GATES,
    PROMOTED_PREDICTIVE_BASELINE_ID,
    validate_predictive_deployment_baseline,
)
from pyhmsc.neural.inference import NeuralHmscInference
from pyhmsc.neural.mean_calibration import (
    BetaPredictiveMeanCalibration,
    apply_beta_predictive_mean_calibration,
)
from pyhmsc.neural.simulator import (
    FixedEffectDataset,
    simulate_fixed_effect_dataset,
    simulate_fixed_effect_ood_dataset,
)
from pyhmsc.neural.teacher_residual import (
    McmcTeacherResponseBatch,
    evaluate_mcmc_teacher_residual_head,
    fit_cross_fitted_mcmc_teacher_residual_head,
)
from pyhmsc.neural.train import fixed_shape_training_data
from pyhmsc.posterior import HmscFit
from pyhmsc.neural.ensemble import file_sha256


REGIMES = (
    "in_distribution",
    "covariate_shift",
    "effect_size_shift",
    "big_spatial_shape",
    "rare_validation",
)
NO_DEGRADATION_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class _AffineEnsembleMember:
    seed: int
    engine: NeuralHmscInference
    scale_calibration: BetaScaleCalibration
    mean_calibration: BetaPredictiveMeanCalibration
    provenance: dict[str, object]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--member-checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--member-predictive-artifacts", nargs="+", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-seeds",
        nargs="+",
        type=int,
        default=[20260731, 20260732, 20260733, 20260734],
    )
    parser.add_argument(
        "--evaluation-seeds",
        nargs="+",
        type=int,
        default=[20260741, 20260742, 20260743],
    )
    parser.add_argument("--teacher-cache-root", type=Path)
    parser.add_argument("--n-sites", type=int, default=40)
    parser.add_argument("--holdout-sites", type=int, default=20)
    parser.add_argument("--small-holdout-sites", type=int, default=12)
    parser.add_argument("--large-holdout-sites", type=int, default=360)
    parser.add_argument("--n-species", type=int, default=75)
    parser.add_argument("--mcmc-samples", type=int, default=60)
    parser.add_argument("--mcmc-transient", type=int, default=40)
    parser.add_argument("--mcmc-chains", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--identity-penalty", type=float, default=0.05)
    parser.add_argument("--max-abs-logit-residual", type=float, default=0.5)
    parser.add_argument(
        "--context-margin-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--model-seed", type=int, default=20260736)
    args = parser.parse_args()
    _validate_args(parser, args)

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = validate_predictive_deployment_baseline(
        args.baseline_bundle,
        expected_baseline_id=PROMOTED_PREDICTIVE_BASELINE_ID,
    )
    baseline_bundle_path = (
        args.baseline_bundle / "baseline.json"
        if args.baseline_bundle.is_dir()
        else args.baseline_bundle
    )
    members, ensemble_provenance = _load_affine_ensemble_members(
        args=args,
        baseline=baseline,
        baseline_bundle_path=baseline_bundle_path,
    )

    corpus_records: list[dict[str, object]] = []
    calibration = _teacher_batches(
        partition="crossfit_calibration",
        seeds=args.calibration_seeds,
        members=members,
        output=args.output / "teacher_corpus",
        args=args,
        records=corpus_records,
    )
    evaluation = _teacher_batches(
        partition="evaluation",
        seeds=args.evaluation_seeds,
        members=members,
        output=args.output / "teacher_corpus",
        args=args,
        records=corpus_records,
    )
    head = fit_cross_fitted_mcmc_teacher_residual_head(
        calibration,
        baseline_id=PROMOTED_PREDICTIVE_BASELINE_ID,
        max_abs_logit_residual=args.max_abs_logit_residual,
        identity_penalty=args.identity_penalty,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        margin_grid=args.context_margin_grid,
        seed=args.model_seed,
    )
    artifact_path = head.save(args.output / "teacher_residual_head")
    evaluation_scores = evaluate_mcmc_teacher_residual_head(head, evaluation)
    raw_evaluation_scores = evaluate_mcmc_teacher_residual_head(
        head,
        evaluation,
        use_selected_shrinkage=False,
        use_context_gate=False,
    )
    gated_raw_evaluation_scores = evaluate_mcmc_teacher_residual_head(
        head,
        evaluation,
        use_selected_shrinkage=False,
    )
    seed_scores = {
        str(seed): evaluate_mcmc_teacher_residual_head(
            head, [batch for batch in evaluation if batch.seed // 100 == seed]
        )
        for seed in args.evaluation_seeds
    }
    raw_seed_scores = {
        str(seed): evaluate_mcmc_teacher_residual_head(
            head,
            [batch for batch in evaluation if batch.seed // 100 == seed],
            use_selected_shrinkage=False,
            use_context_gate=False,
        )
        for seed in args.evaluation_seeds
    }
    gate = _compact_gate(head.selected, evaluation_scores, seed_scores)
    context_decisions = [
        {
            "seed": int(batch.seed),
            "label": str(batch.label),
            "profile": str(batch.profile),
            **head.context_gate.decision(batch.baseline_probability, batch.X),
        }
        for batch in evaluation
    ]
    result = {
        "kind": "cross_fitted_mcmc_teacher_residual_compact_fixed_evaluation",
        "decision": gate["decision"],
        "baseline_id": PROMOTED_PREDICTIVE_BASELINE_ID,
        "baseline_bundle": str(args.baseline_bundle.resolve()),
        "baseline_bundle_sha256": file_sha256(baseline_bundle_path),
        "baseline_default_policy": baseline["default_policy"],
        "simulation_baseline": ensemble_provenance,
        "teacher": {
            "kind": "python_native_hmsc_fixed_effect_mcmc",
            "training_target": "response_probability_only",
            "samples": args.mcmc_samples,
            "transient": args.mcmc_transient,
            "chains": args.mcmc_chains,
            "real_outcomes_used": False,
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "metadata": head.to_metadata(),
        },
        "frozen_competitor_gates": list(FROZEN_COMPETITOR_GATES),
        "coefficient_and_uncertainty_gates": "inherited_unchanged_response_only",
        "evaluation": evaluation_scores,
        "raw_head_evaluation": raw_evaluation_scores,
        "gated_raw_head_evaluation": gated_raw_evaluation_scores,
        "evaluation_by_seed": seed_scores,
        "raw_head_evaluation_by_seed": raw_seed_scores,
        "evaluation_context_decisions": context_decisions,
        "compact_gate": gate,
        "teacher_corpus": corpus_records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (args.output / "mcmc_teacher_residual_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = _score_rows(
        evaluation_scores,
        seed_scores,
        raw_aggregate=raw_evaluation_scores,
        raw_by_seed=raw_seed_scores,
        gated_raw_aggregate=gated_raw_evaluation_scores,
    )
    pd.DataFrame(rows).to_csv(
        args.output / "mcmc_teacher_residual_comparison.csv", index=False
    )
    report = _render_report(result, pd.DataFrame(rows))
    (args.output / "mcmc_teacher_residual_comparison.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


def _teacher_batches(
    *,
    partition: str,
    seeds: list[int],
    members: list[_AffineEnsembleMember],
    output: Path,
    args: argparse.Namespace,
    records: list[dict[str, object]],
) -> list[McmcTeacherResponseBatch]:
    batches = []
    for seed in seeds:
        for regime_index, regime in enumerate(REGIMES):
            dataset_seed = int(seed) * 100 + regime_index
            holdout_profiles = _regime_holdout_profiles(regime, args=args)
            maximum_holdout_sites = max(holdout_profiles.values())
            full_dataset = _simulate_regime(
                regime,
                seed=dataset_seed,
                n_sites=args.n_sites + maximum_holdout_sites,
                n_species=args.n_species,
            )
            dataset, maximum_heldout = _split_simulated_sites(
                full_dataset,
                n_train=args.n_sites,
            )
            data = fixed_shape_training_data([dataset])
            maximum_heldout_data = fixed_shape_training_data([maximum_heldout])
            member_probabilities = []
            for member in members:
                posterior = member.engine.predict_beta_posterior(data)
                posterior = apply_beta_predictive_calibration(
                    posterior, member.scale_calibration, distribution="probit"
                )
                posterior = apply_beta_predictive_mean_calibration(
                    posterior, member.mean_calibration, distribution="probit"
                )
                member_probabilities.append(
                    _probit_probability(posterior, maximum_heldout_data.X)[0]
                )
            maximum_baseline_probability = np.mean(member_probabilities, axis=0)
            teacher_dir = (
                output
                / "sample_size_stable_v3"
                / partition
                / f"seed_{seed}"
                / regime
            )
            teacher_dir.mkdir(parents=True, exist_ok=True)
            teacher_path = teacher_dir / "mcmc_teacher.h5"
            cached_path = _find_cached_teacher(
                args.teacher_cache_root,
                seed=seed,
                regime=regime,
            )
            reused_path = teacher_path if teacher_path.exists() else cached_path
            if reused_path is not None:
                fit = HmscFit.from_file(reused_path)
            else:
                model = HmscModel(
                    Y=dataset.Y,
                    X=dataset.X,
                    x_formula="~ x1",
                    distr="probit",
                )
                fit = model.sample(
                    samples=args.mcmc_samples,
                    transient=args.mcmc_transient,
                    thin=1,
                    chains=args.mcmc_chains,
                    init="python-native",
                    workdir=teacher_dir / "mcmc_work",
                    verbose=max(args.mcmc_samples + args.mcmc_transient + 1, 1000),
                    output_file=teacher_path,
                )
            maximum_teacher_probability = fit.predict_mean(
                maximum_heldout.X
            ).to_numpy(dtype=float)
            for profile, n_holdout_sites in holdout_profiles.items():
                heldout = _prefix_simulated_sites(
                    maximum_heldout,
                    n_sites=n_holdout_sites,
                    profile=profile,
                )
                heldout_data = fixed_shape_training_data([heldout])
                baseline_probability = maximum_baseline_probability[
                    :n_holdout_sites
                ]
                teacher_probability = maximum_teacher_probability[:n_holdout_sites]
                batch = McmcTeacherResponseBatch(
                    baseline_probability=baseline_probability,
                    teacher_probability=teacher_probability,
                    X=heldout_data.X[0],
                    Y=heldout_data.Y[0],
                    label=regime,
                    seed=dataset_seed,
                    profile=profile,
                )
                batches.append(batch)
                records.append({
                    "partition": partition,
                    "evaluation_seed": int(seed),
                    "dataset_seed": dataset_seed,
                    "regime": regime,
                    "holdout_profile": profile,
                    "teacher_posterior": str(
                        (
                            reused_path if reused_path is not None else teacher_path
                        ).resolve()
                    ),
                    "teacher_posterior_sha256": file_sha256(
                        reused_path if reused_path is not None else teacher_path
                    ),
                    "reused_teacher_posterior": reused_path is not None,
                    "teacher_probability_mean": float(np.mean(teacher_probability)),
                    "baseline_probability_mean": float(np.mean(baseline_probability)),
                    "n_training_sites": int(args.n_sites),
                    "n_holdout_sites": int(n_holdout_sites),
                    "maximum_shared_holdout_sites": int(maximum_holdout_sites),
                    "ensemble_member_seeds": [member.seed for member in members],
                })
    return batches


def _find_cached_teacher(
    cache_root: Path | None,
    *,
    seed: int,
    regime: str,
) -> Path | None:
    if cache_root is None:
        return None
    matches = sorted(
        cache_root.glob(
            "*/sample_size_stable_v3/"
            f"*/seed_{int(seed)}/{regime}/mcmc_teacher.h5"
        )
    )
    if len(matches) > 1:
        raise ValueError(
            f"multiple cached MCMC teachers found for seed {seed}, regime {regime}"
        )
    return matches[0] if matches else None


def _load_affine_ensemble_members(
    *,
    args: argparse.Namespace,
    baseline: dict[str, object],
    baseline_bundle_path: Path,
) -> tuple[list[_AffineEnsembleMember], dict[str, object]]:
    bundle_root = baseline_bundle_path.parent
    manifest_record = baseline["datasets"]["big_spatial"]["affine_branch"]
    manifest_path = bundle_root / manifest_record["path"]
    if file_sha256(manifest_path) != manifest_record["sha256"]:
        raise ValueError("frozen Big Spatial affine manifest hash differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_members = manifest["members"]
    if len(args.member_checkpoints) != len(expected_members):
        raise ValueError("member checkpoint count differs from frozen affine ensemble")
    if len(args.member_predictive_artifacts) != len(expected_members):
        raise ValueError("member artifact count differs from frozen affine ensemble")

    members = []
    records = []
    expected_dimensions = {
        "n_sites": int(args.n_sites),
        "n_covariates": 2,
        "n_species": int(args.n_species),
    }
    for expected, checkpoint, artifact in zip(
        expected_members,
        args.member_checkpoints,
        args.member_predictive_artifacts,
    ):
        seed = int(expected["seed"])
        if file_sha256(artifact) != expected["sha256"]:
            raise ValueError(
                f"predictive artifact hash differs from frozen member {seed}"
            )
        checkpoint_metadata_path = checkpoint / "neural_checkpoint.json"
        checkpoint_metadata = json.loads(
            checkpoint_metadata_path.read_text(encoding="utf-8")
        )
        if checkpoint_metadata.get("dimensions") != expected_dimensions:
            raise ValueError(
                f"checkpoint {seed} dimensions differ from compact ensemble shape"
            )
        if checkpoint_metadata.get("distribution") != "probit":
            raise ValueError(f"checkpoint {seed} is not probit")
        if checkpoint_metadata.get("formula", {}).get("X") != "~ TMG":
            raise ValueError(f"checkpoint {seed} has an incompatible formula")

        fit = HmscFit.from_file(artifact)
        active_calibration = fit.metadata.get("active_predictive_mean_calibration")
        if not isinstance(active_calibration, dict) or not active_calibration.get(
            "selected", False
        ):
            raise ValueError(f"frozen member {seed} lacks an active affine branch")
        if active_calibration.get("method") != "probit_transfer_response_branch_affine":
            raise ValueError(f"frozen member {seed} uses an unexpected affine method")
        member = _AffineEnsembleMember(
            seed=seed,
            engine=NeuralHmscInference.load(checkpoint),
            scale_calibration=BetaScaleCalibration.from_metadata(
                fit.metadata["calibration"]
            ),
            mean_calibration=BetaPredictiveMeanCalibration.from_metadata(
                active_calibration
            ),
            provenance={},
        )
        record = {
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_metadata_sha256": file_sha256(checkpoint_metadata_path),
            "checkpoint_weights_sha256": file_sha256(checkpoint / "weights.weights.h5"),
            "predictive_artifact": str(artifact.resolve()),
            "predictive_artifact_sha256": file_sha256(artifact),
            "expected_predictive_artifact_sha256": expected["sha256"],
            "active_affine_method": active_calibration["method"],
            "active_affine_slope": float(active_calibration["slope"]),
            "active_affine_intercept": float(active_calibration["intercept"]),
        }
        members.append(
            _AffineEnsembleMember(
                seed=member.seed,
                engine=member.engine,
                scale_calibration=member.scale_calibration,
                mean_calibration=member.mean_calibration,
                provenance=record,
            )
        )
        records.append(record)
    return members, {
        "procedure": "ordered_three_member_affine_probability_ensemble",
        "dataset_context": "big_spatial_transfer",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "aggregation": manifest["aggregation"],
        "members": records,
    }


def _simulate_regime(
    regime: str,
    *,
    seed: int,
    n_sites: int,
    n_species: int,
) -> FixedEffectDataset:
    if regime == "in_distribution":
        dataset = simulate_fixed_effect_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution="probit",
            seed=seed,
        )
    elif regime in {"covariate_shift", "effect_size_shift"}:
        dataset = simulate_fixed_effect_ood_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution="probit",
            regime=regime,
            seed=seed,
        )
    elif regime == "big_spatial_shape":
        dataset = simulate_fixed_effect_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution="probit",
            beta_scale=1.5,
            intercept_mean=-1.75,
            simulation_domain="ood",
            ood_regime="big_spatial_low_prevalence_effect_shift",
            seed=seed,
        )
    elif regime == "rare_validation":
        dataset = simulate_fixed_effect_dataset(
            n_sites=n_sites,
            n_species=n_species,
            distribution="probit",
            intercept_mean=-1.75,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported teacher regime: {regime!r}")
    return _one_covariate_dataset(dataset, seed=seed)


def _regime_holdout_profiles(
    regime: str,
    *,
    args: argparse.Namespace,
) -> dict[str, int]:
    if regime == "big_spatial_shape":
        return {
            "compact": int(args.holdout_sites),
            "big_spatial": int(args.large_holdout_sites),
        }
    if regime == "rare_validation":
        return {
            "whittaker": int(args.small_holdout_sites),
            "compact": int(args.holdout_sites),
        }
    return {"compact": int(args.holdout_sites)}


def _one_covariate_dataset(
    dataset: FixedEffectDataset,
    *,
    seed: int,
) -> FixedEffectDataset:
    """Project the standard simulator onto the frozen ensemble's one-covariate shape."""
    truth_beta = dataset.truth_beta.loc[["Intercept", "x1"]].copy()
    design = np.column_stack(
        [np.ones(len(dataset.X), dtype=float), dataset.X["x1"].to_numpy(dtype=float)]
    )
    linear = design @ truth_beta.to_numpy(dtype=float)
    outcome = np.random.default_rng(int(seed) + 7919).binomial(1, ndtr(linear))
    return FixedEffectDataset(
        Y=pd.DataFrame(outcome, index=dataset.Y.index, columns=dataset.Y.columns),
        X=dataset.X.loc[:, ["x1"]].copy(),
        truth_beta=truth_beta,
        linear_predictor=pd.DataFrame(
            linear, index=dataset.Y.index, columns=dataset.Y.columns
        ),
        metadata={
            **dataset.metadata,
            "n_covariates": 2,
            "formula": "~ x1",
            "one_covariate_projection_seed": int(seed) + 7919,
        },
    )


def _probit_probability(posterior, design: np.ndarray) -> np.ndarray:
    mean = np.asarray(posterior.mean.numpy(), dtype=float)
    scale = np.asarray(posterior.scale.numpy(), dtype=float)
    design = np.asarray(design, dtype=float)
    linear_mean = np.einsum("bnk,bks->bns", design, mean)
    variance = np.einsum("bnk,bks->bns", np.square(design), np.square(scale))
    return ndtr(linear_mean / np.sqrt(1.0 + np.maximum(variance, 0.0)))


def _split_simulated_sites(
    dataset: FixedEffectDataset,
    *,
    n_train: int,
) -> tuple[FixedEffectDataset, FixedEffectDataset]:
    if n_train <= 0 or n_train >= len(dataset.X):
        raise ValueError("n_train must leave at least one simulated holdout site")
    train_index = dataset.X.index[:n_train]
    holdout_index = dataset.X.index[n_train:]

    def subset(index, role):
        return FixedEffectDataset(
            Y=dataset.Y.loc[index].copy(),
            X=dataset.X.loc[index].copy(),
            truth_beta=dataset.truth_beta.copy(),
            linear_predictor=dataset.linear_predictor.loc[index].copy(),
            metadata={
                **dataset.metadata,
                "site_partition": role,
                "parent_n_sites": len(dataset.X),
            },
        )

    return subset(train_index, "training"), subset(holdout_index, "holdout")


def _prefix_simulated_sites(
    dataset: FixedEffectDataset,
    *,
    n_sites: int,
    profile: str,
) -> FixedEffectDataset:
    if n_sites <= 0 or n_sites > len(dataset.X):
        raise ValueError("profile n_sites must be within the heldout dataset")
    index = dataset.X.index[:n_sites]
    return FixedEffectDataset(
        Y=dataset.Y.loc[index].copy(),
        X=dataset.X.loc[index].copy(),
        truth_beta=dataset.truth_beta.copy(),
        linear_predictor=dataset.linear_predictor.loc[index].copy(),
        metadata={
            **dataset.metadata,
            "holdout_profile": str(profile),
            "profile_n_sites": int(n_sites),
        },
    )


def _compact_gate(
    selected: bool,
    scores: dict[str, object],
    seed_scores: dict[str, dict[str, object]],
) -> dict[str, object]:
    by_label = scores["by_label"]
    no_degradation = bool(
        all(
            row["outcome_brier_ratio"] <= 1.0 + NO_DEGRADATION_TOLERANCE
            and row["outcome_log_loss_ratio"] <= 1.0 + NO_DEGRADATION_TOLERANCE
            for row in by_label.values()
        )
    )
    approved_improved = bool(
        all(
            by_label[label]["teacher_brier_ratio"] < 1.0
            and by_label[label]["teacher_cross_entropy_ratio"] < 1.0
            and by_label[label]["outcome_brier_ratio"] < 1.0
            and by_label[label]["outcome_log_loss_ratio"] < 1.0
            for label in ("effect_size_shift", "big_spatial_shape")
        )
    )
    all_seeds_no_degradation = bool(
        all(
            all(
                row["outcome_brier_ratio"] <= 1.0 + NO_DEGRADATION_TOLERANCE
                and row["outcome_log_loss_ratio"] <= 1.0 + NO_DEGRADATION_TOLERANCE
                for row in seed_result["by_label"].values()
            )
            for seed_result in seed_scores.values()
        )
    )
    passed = bool(
        selected and no_degradation and approved_improved and all_seeds_no_degradation
    )
    return {
        "passed": passed,
        "decision": (
            "mcmc_teacher_residual_compact_gate_passed"
            if passed
            else "mcmc_teacher_residual_compact_gate_failed"
        ),
        "head_selected": bool(selected),
        "all_regime_no_degradation": no_degradation,
        "all_seed_regime_no_degradation": all_seeds_no_degradation,
        "target_and_effect_improved": approved_improved,
        "real_data_evaluation_allowed": False,
        "five_seed_lumi_allowed": False,
    }


def _score_rows(
    aggregate: dict[str, object],
    by_seed: dict[str, dict[str, object]],
    *,
    raw_aggregate: dict[str, object],
    raw_by_seed: dict[str, dict[str, object]],
    gated_raw_aggregate: dict[str, object],
) -> list[dict[str, object]]:
    rows = [
        {
            "candidate_variant": "selected",
            "evaluation_seed": "aggregate",
            "regime": label,
            **metrics,
        }
        for label, metrics in aggregate["by_label"].items()
    ]
    rows.extend(
        {
            "candidate_variant": "gated_raw_head",
            "evaluation_seed": "aggregate",
            "regime": label,
            **metrics,
        }
        for label, metrics in gated_raw_aggregate["by_label"].items()
    )
    rows.extend(
        {
            "candidate_variant": "raw_head",
            "evaluation_seed": "aggregate",
            "regime": label,
            **metrics,
        }
        for label, metrics in raw_aggregate["by_label"].items()
    )
    for seed, result in by_seed.items():
        rows.extend(
            {
                "candidate_variant": "selected",
                "evaluation_seed": seed,
                "regime": label,
                **metrics,
            }
            for label, metrics in result["by_label"].items()
        )
    for seed, result in raw_by_seed.items():
        rows.extend(
            {
                "candidate_variant": "raw_head",
                "evaluation_seed": seed,
                "regime": label,
                **metrics,
            }
            for label, metrics in result["by_label"].items()
        )
    return rows


def _render_report(result: dict[str, object], rows: pd.DataFrame) -> str:
    columns = [
        "candidate_variant",
        "evaluation_seed",
        "regime",
        "teacher_brier_ratio",
        "teacher_cross_entropy_ratio",
        "outcome_brier_ratio",
        "outcome_log_loss_ratio",
        "mean_abs_logit_residual",
    ]
    return "\n".join(
        [
            "# MCMC-Teacher Residual Compact Evaluation",
            "",
            f"Decision: `{result['decision']}`",
            f"Baseline: `{result['baseline_id']}`",
            "Training target: simulated Python-MCMC response probabilities only",
            "Real-data evaluation allowed: False",
            "Five-seed LUMI allowed: False",
            "",
            "```text",
            rows.loc[:, columns].to_string(index=False),
            "```",
            "",
        ]
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for path_name in ("baseline_bundle",):
        path = getattr(args, path_name)
        if not path.exists():
            parser.error(f"--{path_name.replace('_', '-')} does not exist: {path}")
    for option, paths in (
        ("member-checkpoints", args.member_checkpoints),
        ("member-predictive-artifacts", args.member_predictive_artifacts),
    ):
        missing = [path for path in paths if not path.exists()]
        if missing:
            parser.error(f"--{option} does not exist: {missing[0]}")
    if len(args.member_checkpoints) != len(args.member_predictive_artifacts):
        parser.error("member checkpoint and predictive artifact counts must match")
    if args.teacher_cache_root is not None and not args.teacher_cache_root.exists():
        parser.error(f"--teacher-cache-root does not exist: {args.teacher_cache_root}")
    if len(args.calibration_seeds) < 3:
        parser.error("--calibration-seeds requires at least three communities")
    if len(set(args.calibration_seeds)) != len(args.calibration_seeds):
        parser.error("--calibration-seeds must be unique")
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        parser.error("--evaluation-seeds must be unique")
    if set(args.calibration_seeds) & set(args.evaluation_seeds):
        parser.error("calibration and evaluation seeds must be independent")
    if any(value < 0.0 for value in args.context_margin_grid):
        parser.error("--context-margin-grid must be non-negative")
    for name in (
        "n_sites",
        "holdout_sites",
        "small_holdout_sites",
        "large_holdout_sites",
        "n_species",
        "mcmc_samples",
        "mcmc_chains",
        "epochs",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.mcmc_transient < 0:
        parser.error("--mcmc-transient must be non-negative")


if __name__ == "__main__":
    main()
