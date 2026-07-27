"""Requalify frozen predictive ensembles against scale-only and Python MCMC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.ensemble import (
    PredictiveProbabilityEnsemble,
    file_sha256,
)
from pyhmsc.posterior import HmscFit


GATE_METRICS = ("brier_score", "log_loss", "predictive_rmse", "richness_mae")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _validate_args(parser, args)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    provenance = {}
    for dataset in ("whittaker", "big_spatial"):
        predictions, observed, dataset_provenance = prepare_frozen_requalification(
            args.run_root,
            args.seeds,
            dataset=dataset,
            manifest_dir=manifest_dir,
        )
        rows.extend(
            evaluate_precomputed_ensembles(
                predictions,
                observed,
                dataset=dataset,
            )
        )
        provenance[dataset] = dataset_provenance

    frame = pd.DataFrame(rows)
    summary = summarize_probability_ensembles(frame, provenance=provenance)
    frame.to_csv(args.output / "probability_ensemble_comparison.csv", index=False)
    (args.output / "probability_ensemble_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_probability_ensemble_report(frame, summary)
    (args.output / "probability_ensemble_comparison.md").write_text(
        report,
        encoding="utf-8",
    )
    print(report)


def prepare_frozen_requalification(
    run_root: Path,
    seeds: list[int],
    *,
    dataset: str,
    manifest_dir: Path,
) -> tuple[dict[str, dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Build/reload artifacts and freeze neural/MCMC predictions before Y."""
    scale_paths = []
    affine_paths = []
    mcmc_paths = []
    response_paths = []
    member_provenance = []
    reference_X: pd.DataFrame | None = None

    for seed in seeds:
        root = run_root / f"seed_{seed}" / dataset
        paths = _dataset_paths(root, dataset)
        X = pd.read_csv(paths["X"], index_col=0)
        if reference_X is None:
            reference_X = X
        elif not X.equals(reference_X):
            raise ValueError(f"{dataset} covariates differ across frozen seeds")
        scale_paths.append(paths["scale"])
        affine_paths.append(paths["affine"])
        mcmc_paths.append(paths["mcmc"])
        response_paths.append(paths["Y"])
        member_provenance.append(
            _qualified_member_provenance(paths, seed=seed, dataset=dataset)
        )

    if reference_X is None:
        raise ValueError("at least one frozen member is required")
    reference_payload = {
        "kind": "qualified_python_mcmc_probability_reference",
        "aggregation": "arithmetic_mean_response_probability",
        "ordered_members": [
            {
                "seed": int(seed),
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for seed, path in zip(seeds, mcmc_paths)
        ],
    }
    common_provenance = {
        "dataset": dataset,
        "frozen_run_root": str(run_root.resolve()),
        "ordered_seeds": [int(seed) for seed in seeds],
        "response_semantics": "predictive_only",
        "selection_outcomes_used": False,
        "qualified_python_mcmc_reference": reference_payload,
    }
    scale_ensemble = PredictiveProbabilityEnsemble.create(
        scale_paths,
        seeds=seeds,
        calibration_role="scale_only",
        member_provenance=member_provenance,
        provenance=common_provenance,
    )
    affine_ensemble = PredictiveProbabilityEnsemble.create(
        affine_paths,
        seeds=seeds,
        calibration_role="affine_branch",
        member_provenance=member_provenance,
        provenance=common_provenance,
    )
    scale_manifest = scale_ensemble.save(
        manifest_dir / f"{dataset}_scale_only_ensemble.json"
    )
    affine_manifest = affine_ensemble.save(
        manifest_dir / f"{dataset}_affine_branch_ensemble.json"
    )

    # Requalification uses fresh manifest loads, including hash validation.
    scale_ensemble = PredictiveProbabilityEnsemble.from_manifest(scale_manifest)
    affine_ensemble = PredictiveProbabilityEnsemble.from_manifest(affine_manifest)
    if scale_ensemble.seeds != affine_ensemble.seeds:
        raise ValueError("scale and affine ensemble seed order differs")
    if scale_ensemble.compatibility != affine_ensemble.compatibility:
        raise ValueError("scale and affine ensemble compatibility differs")

    mcmc_fits = [HmscFit.from_file(path) for path in mcmc_paths]
    subsets = [("full", tuple(seeds))]
    subsets.extend(
        (f"leave_out_{seed}", tuple(value for value in seeds if value != seed))
        for seed in seeds
    )
    predictions: dict[str, dict[str, Any]] = {}
    reference_prediction: pd.DataFrame | None = None
    positions = {int(seed): index for index, seed in enumerate(seeds)}
    for label, members in subsets:
        scale_subset = (
            scale_ensemble
            if tuple(members) == scale_ensemble.seeds
            else scale_ensemble.subset(members)
        )
        affine_subset = (
            affine_ensemble
            if tuple(members) == affine_ensemble.seeds
            else affine_ensemble.subset(members)
        )
        scale_prediction = scale_subset.predict_mean(reference_X).clip(
            1.0e-9, 1.0 - 1.0e-9
        )
        affine_prediction = affine_subset.predict_mean(reference_X).clip(
            1.0e-9, 1.0 - 1.0e-9
        )
        mcmc_frames = [
            mcmc_fits[positions[seed]].predict_mean(reference_X).clip(
                1.0e-9, 1.0 - 1.0e-9
            )
            for seed in members
        ]
        mcmc_prediction = _average_prediction_frames(mcmc_frames)
        if reference_prediction is None:
            reference_prediction = scale_prediction
        for name, prediction in (
            ("scale", scale_prediction),
            ("affine", affine_prediction),
            ("mcmc", mcmc_prediction),
        ):
            _validate_prediction_frame(
                prediction,
                reference_prediction,
                label=f"{dataset} {label} {members}",
            )
        predictions[label] = {
            "members": tuple(int(seed) for seed in members),
            "scale": scale_prediction.to_numpy(dtype=float),
            "affine": affine_prediction.to_numpy(dtype=float),
            "mcmc": mcmc_prediction.to_numpy(dtype=float),
        }

    # Outcomes remain unopened until every neural and MCMC prediction is frozen.
    if reference_prediction is None:
        raise ValueError("at least one frozen prediction is required")
    observed_frame = pd.read_csv(response_paths[0], index_col=0).loc[
        reference_prediction.index,
        reference_prediction.columns,
    ]
    for path in response_paths[1:]:
        candidate = pd.read_csv(path, index_col=0).loc[
            reference_prediction.index,
            reference_prediction.columns,
        ]
        if not candidate.equals(observed_frame):
            raise ValueError(f"{dataset} outcomes differ across frozen seeds")

    all_parity = all(
        row["reference_parity_qualified"] for row in member_provenance
    )
    all_acceptance = all(
        row["dataset_acceptance_passed"] for row in member_provenance
    )
    return (
        predictions,
        observed_frame.to_numpy(dtype=float),
        {
            "ordered_seeds": [int(seed) for seed in seeds],
            "members": member_provenance,
            "scale_manifest": str(scale_manifest.resolve()),
            "scale_manifest_sha256": file_sha256(scale_manifest),
            "affine_manifest": str(affine_manifest.resolve()),
            "affine_manifest_sha256": file_sha256(affine_manifest),
            "manifest_hash_validation_passed": True,
            "manifest_compatibility_validation_passed": True,
            "scale_parity_provenance_qualified": (
                scale_ensemble.parity_provenance_qualified
            ),
            "affine_parity_provenance_qualified": (
                affine_ensemble.parity_provenance_qualified
            ),
            "all_reference_parity_qualified": all_parity,
            "all_dataset_acceptance_passed": all_acceptance,
            "qualified_python_mcmc_reference": reference_payload,
        },
    )


def evaluate_precomputed_ensembles(
    predictions: dict[str, dict[str, Any]],
    observed: np.ndarray,
    *,
    dataset: str,
) -> list[dict[str, Any]]:
    rows = []
    for label, values in predictions.items():
        rows.append(
            _score_ensemble_row(
                scale_probability=values["scale"],
                affine_probability=values["affine"],
                mcmc_probability=values.get("mcmc"),
                observed=observed,
                dataset=dataset,
                label=label,
                members=values["members"],
            )
        )
    return rows


def evaluate_probability_ensembles(
    scale_predictions: dict[int, np.ndarray],
    affine_predictions: dict[int, np.ndarray],
    observed: np.ndarray,
    *,
    dataset: str,
    mcmc_predictions: dict[int, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Pure-array evaluator retained for compact tests and diagnostics."""
    seeds = sorted(scale_predictions)
    if len(seeds) < 3:
        raise ValueError("probability ensemble evaluation requires at least three seeds")
    if set(affine_predictions) != set(seeds):
        raise ValueError("scale and affine prediction seeds must match")
    if mcmc_predictions is not None and set(mcmc_predictions) != set(seeds):
        raise ValueError("MCMC and neural prediction seeds must match")
    subsets = [("full", tuple(seeds))]
    subsets.extend(
        (f"leave_out_{seed}", tuple(value for value in seeds if value != seed))
        for seed in seeds
    )
    rows = []
    for label, members in subsets:
        scale_probability = np.mean(
            np.stack([scale_predictions[seed] for seed in members]), axis=0
        )
        affine_probability = np.mean(
            np.stack([affine_predictions[seed] for seed in members]), axis=0
        )
        mcmc_probability = None
        if mcmc_predictions is not None:
            mcmc_probability = np.mean(
                np.stack([mcmc_predictions[seed] for seed in members]), axis=0
            )
        rows.append(
            _score_ensemble_row(
                scale_probability=scale_probability,
                affine_probability=affine_probability,
                mcmc_probability=mcmc_probability,
                observed=observed,
                dataset=dataset,
                label=label,
                members=members,
            )
        )
    return rows


def _score_ensemble_row(
    *,
    scale_probability: np.ndarray,
    affine_probability: np.ndarray,
    mcmc_probability: np.ndarray | None,
    observed: np.ndarray,
    dataset: str,
    label: str,
    members: tuple[int, ...],
) -> dict[str, Any]:
    scale_metrics = score_probability(scale_probability, observed)
    affine_metrics = score_probability(affine_probability, observed)
    ratios = {
        metric: affine_metrics[metric]
        / max(scale_metrics[metric], np.finfo(float).eps)
        for metric in GATE_METRICS
    }
    no_degradation = bool(all(value <= 1.0 + 1.0e-12 for value in ratios.values()))
    row: dict[str, Any] = {
        "dataset": str(dataset),
        "ensemble": label,
        "members": ",".join(str(value) for value in members),
        "n_members": len(members),
        **{f"scale_{metric}": value for metric, value in scale_metrics.items()},
        **{f"affine_{metric}": value for metric, value in affine_metrics.items()},
        **{f"{metric}_ratio": value for metric, value in ratios.items()},
        **{f"affine_vs_scale_{metric}_ratio": value for metric, value in ratios.items()},
        "no_degradation_passed": no_degradation,
        "genuine_proper_score_improvement": bool(
            no_degradation
            and ratios["brier_score"] < 1.0
            and ratios["log_loss"] < 1.0
        ),
        "target_response_used_for_selection": False,
    }
    if mcmc_probability is not None:
        mcmc_metrics = score_probability(mcmc_probability, observed)
        row.update({f"mcmc_{key}": value for key, value in mcmc_metrics.items()})
        for candidate, metrics in (("scale", scale_metrics), ("affine", affine_metrics)):
            row.update(
                {
                    f"{candidate}_vs_mcmc_{metric}_ratio": metrics[metric]
                    / max(mcmc_metrics[metric], np.finfo(float).eps)
                    for metric in GATE_METRICS
                }
            )
    return row


def score_probability(probability: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1.0e-9, 1.0 - 1.0e-9)
    observed = np.asarray(observed, dtype=float)
    if probability.shape != observed.shape:
        raise ValueError("probability and observed arrays must have matching shapes")
    squared_error = np.square(probability - observed)
    return {
        "brier_score": float(np.mean(squared_error)),
        "log_loss": float(
            -np.mean(
                observed * np.log(probability)
                + (1.0 - observed) * np.log1p(-probability)
            )
        ),
        "predictive_rmse": float(np.sqrt(np.mean(squared_error))),
        "prevalence_mae": float(
            np.mean(np.abs(probability.mean(axis=0) - observed.mean(axis=0)))
        ),
        "richness_mae": float(
            np.mean(np.abs(probability.sum(axis=1) - observed.sum(axis=1)))
        ),
    }


def summarize_probability_ensembles(
    frame: pd.DataFrame,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    full_big = frame.loc[
        (frame["dataset"] == "big_spatial") & (frame["ensemble"] == "full")
    ]
    if len(full_big) != 1:
        raise ValueError("expected one full Big Spatial ensemble row")
    all_no_degradation = bool(frame["no_degradation_passed"].all())
    full_big_improvement = bool(
        full_big.iloc[0]["genuine_proper_score_improvement"]
    )
    provenance_passed = True
    manifest_validation_passed = provenance is not None
    if provenance is not None:
        provenance_passed = bool(
            all(
                record.get("all_reference_parity_qualified", False)
                and record.get("all_dataset_acceptance_passed", False)
                and record.get("scale_parity_provenance_qualified", False)
                and record.get("affine_parity_provenance_qualified", False)
                for record in provenance.values()
            )
        )
        manifest_validation_passed = bool(
            all(
                record.get("manifest_hash_validation_passed", False)
                and record.get("manifest_compatibility_validation_passed", False)
                for record in provenance.values()
            )
        )
    api_requalification_passed = bool(
        all_no_degradation
        and full_big_improvement
        and provenance_passed
        and manifest_validation_passed
    )
    if api_requalification_passed:
        decision = "predictive_ensemble_api_requalification_passed"
    elif provenance is None and all_no_degradation and full_big_improvement:
        decision = "probability_ensemble_promotion_candidate"
    elif not provenance_passed:
        decision = "probability_ensemble_failed_provenance"
    elif not manifest_validation_passed:
        decision = "probability_ensemble_failed_manifest_validation"
    elif not all_no_degradation:
        decision = "probability_ensemble_failed_leave_one_out_stability"
    else:
        decision = "probability_ensemble_no_genuine_big_spatial_gain"

    mcmc_comparison = {}
    if "mcmc_brier_score" in frame.columns:
        full = frame.loc[frame["ensemble"] == "full"]
        mcmc_comparison = {
            "reference_role": "qualified_python_mcmc_diagnostic_comparator",
            "neural_equivalence_claimed": False,
            "full_rows": full.to_dict(orient="records"),
            "affine_beats_mcmc_brier_all_full_datasets": bool(
                (full["affine_vs_mcmc_brier_score_ratio"] < 1.0).all()
            ),
            "affine_beats_mcmc_log_loss_all_full_datasets": bool(
                (full["affine_vs_mcmc_log_loss_ratio"] < 1.0).all()
            ),
        }
    return {
        "kind": "manifest_backed_probability_ensemble_requalification",
        "decision": decision,
        "api_requalification_passed": api_requalification_passed,
        "all_full_and_leave_one_out_no_degradation": all_no_degradation,
        "full_big_spatial_genuine_proper_score_improvement": full_big_improvement,
        "manifest_validation_passed": manifest_validation_passed,
        "provenance_passed": provenance_passed,
        "target_response_used_for_selection": False,
        "promotion_rule": (
            "the full ensemble and every leave-one-out ensemble must preserve "
            "Brier, log-loss, RMSE, and richness-MAE no degradation on both "
            "datasets, the full Big Spatial ensemble must improve Brier and "
            "log loss, and all manifests/parity provenance must validate"
        ),
        "mcmc_comparison": mcmc_comparison,
        "provenance": {} if provenance is None else provenance,
        "rows": frame.to_dict(orient="records"),
    }


def render_probability_ensemble_report(
    frame: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    columns = [
        "dataset",
        "ensemble",
        "members",
        "brier_score_ratio",
        "log_loss_ratio",
        "predictive_rmse_ratio",
        "richness_mae_ratio",
        "no_degradation_passed",
        "genuine_proper_score_improvement",
    ]
    mcmc_columns = [
        "dataset",
        "ensemble",
        "scale_brier_score",
        "affine_brier_score",
        "mcmc_brier_score",
        "scale_log_loss",
        "affine_log_loss",
        "mcmc_log_loss",
        "affine_vs_mcmc_brier_score_ratio",
        "affine_vs_mcmc_log_loss_ratio",
    ]
    lines = [
        "# Predictive Ensemble API Requalification",
        "",
        "Scale-only and affine response probabilities use ordered, hash-validated ",
        "ensemble manifests over identical members. Qualified Python MCMC is a ",
        "separate statistical reference, not a predictive-only artifact member.",
        "Target outcomes are unavailable until all neural and MCMC predictions are frozen.",
        "",
        f"Decision: `{summary['decision']}`",
        f"API requalification passed: {summary['api_requalification_passed']}",
        "All full/leave-one-out no-degradation gates: "
        f"{summary['all_full_and_leave_one_out_no_degradation']}",
        "Full Big Spatial proper-score improvement: "
        f"{summary['full_big_spatial_genuine_proper_score_improvement']}",
        f"Manifest validation: {summary['manifest_validation_passed']}",
        f"Qualified provenance: {summary['provenance_passed']}",
        "",
        "## Matched Neural Comparison",
        "",
        "```text",
        frame.loc[:, columns].to_string(index=False),
        "```",
    ]
    if set(mcmc_columns).issubset(frame.columns):
        lines.extend(
            [
                "",
                "## Qualified Python MCMC Reference",
                "",
                "Lower scores are better. These scores diagnose the remaining proper-score ",
                "gap; they do not change the neural affine-versus-scale promotion gate.",
                "",
                "```text",
                frame.loc[:, mcmc_columns].to_string(index=False),
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _qualified_member_provenance(
    paths: dict[str, Path],
    *,
    seed: int,
    dataset: str,
) -> dict[str, Any]:
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    qualification = metadata.get("reference_qualification")
    if not isinstance(qualification, dict):
        raise ValueError(f"{dataset} seed {seed} lacks reference qualification")
    parity_path_value = qualification.get("metrics_path") or metadata.get(
        "args", {}
    ).get("reference_parity_metrics")
    if not parity_path_value:
        raise ValueError(f"{dataset} seed {seed} lacks parity metrics path")
    parity_path = Path(str(parity_path_value))
    parity_qualified = bool(
        acceptance.get("reference_parity_qualified", False)
        and qualification.get("parity_passed", False)
        and qualification.get("boundary_arrays_passed", False)
        and qualification.get("acceptance_gates_passed", False)
    )
    acceptance_key = (
        "qualification_acceptance_passed"
        if dataset == "whittaker"
        else "predictive_transfer_acceptance_passed"
    )
    dataset_accepted = bool(acceptance.get(acceptance_key, False))
    if not parity_qualified:
        raise ValueError(f"{dataset} seed {seed} reference parity is not qualified")
    if not dataset_accepted:
        raise ValueError(f"{dataset} seed {seed} dataset acceptance failed")
    if not parity_path.is_file():
        raise FileNotFoundError(f"parity metrics do not exist: {parity_path}")
    return {
        "seed": int(seed),
        "reference_model": metadata.get("reference_model"),
        "reference_parity_qualified": parity_qualified,
        "dataset_acceptance_passed": dataset_accepted,
        "acceptance_path": str(paths["acceptance"].resolve()),
        "acceptance_sha256": file_sha256(paths["acceptance"]),
        "run_metadata_path": str(paths["metadata"].resolve()),
        "run_metadata_sha256": file_sha256(paths["metadata"]),
        "parity_metrics_path": str(parity_path.resolve()),
        "parity_metrics_sha256": file_sha256(parity_path),
        "mcmc_reference_path": str(paths["mcmc"].resolve()),
        "mcmc_reference_sha256": file_sha256(paths["mcmc"]),
        "reference_qualification": qualification,
    }


def _average_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one prediction frame is required")
    reference = frames[0]
    for frame in frames[1:]:
        _validate_prediction_frame(frame, reference, label="MCMC ensemble member")
    values = np.mean(
        np.stack([frame.to_numpy(dtype=float) for frame in frames]), axis=0
    )
    return pd.DataFrame(values, index=reference.index, columns=reference.columns)


def _validate_prediction_frame(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    label: str,
) -> None:
    if not candidate.index.equals(reference.index):
        raise ValueError(f"{label} prediction index mismatch")
    if not candidate.columns.equals(reference.columns):
        raise ValueError(f"{label} prediction species order mismatch")


def _dataset_paths(root: Path, dataset: str) -> dict[str, Path]:
    common = {
        "scale": root / "neural_predictive_distribution_scale_only.h5",
        "affine": root / "neural_predictive_distribution.h5",
        "mcmc": root / "mcmc_posterior.h5",
        "metadata": root / "run_metadata.json",
    }
    if dataset == "whittaker":
        return common | {
            "X": root / "whittaker_holdout/data/test/X.csv",
            "Y": root / "whittaker_holdout/data/test/Y.csv",
            "acceptance": root / "whittaker_acceptance.json",
        }
    if dataset == "big_spatial":
        return common | {
            "X": root / "big_spatial_transfer_project/data/test/X.csv",
            "Y": root / "big_spatial_transfer_project/data/test/Y.csv",
            "acceptance": root / "big_spatial_transfer_acceptance.json",
        }
    raise ValueError(f"unsupported dataset: {dataset!r}")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if len(args.seeds) < 3:
        parser.error("--seeds requires at least three frozen seeds")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if not args.run_root.exists():
        parser.error(f"--run-root does not exist: {args.run_root}")


if __name__ == "__main__":
    main()
