#!/usr/bin/env python3
"""Run the preregistered Milestone 53A trait-Gamma calibration protocol."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache")
)

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.qualify_neural_hmsc_trait_gamma import (  # noqa: E402
    _simulated_mcmc_gate,
    _whittaker_gate,
)
from pyhmsc.neural import simulate_trait_gamma_boundary_dataset  # noqa: E402
from pyhmsc.neural.trait_inference import (  # noqa: E402
    TraitGammaNeuralHmscInference,
    _sha256,
    package_trait_gamma_calibration,
)


PROTOCOL_ID = "neural_hmsc_trait_gamma_m53a_v1"
FROZEN_WEIGHTS_SHA256 = (
    "bc869b8a92e7d9ea0bf11acb565e571816a68dcff220f0a003f22d2d753cdcac"
)
CALIBRATION_BLOCK_STARTS = (
    31_000_001,
    32_000_001,
    33_000_001,
    34_000_001,
    35_000_001,
    36_000_001,
)
CALIBRATION_BLOCK_SIZE = 64
EVALUATION_BLOCK_STARTS = (41_000_001, 42_000_001, 43_000_001)
EVALUATION_BLOCK_SIZE = 258
WHITTAKER_MCMC_SEEDS = (44_000_001, 44_000_002, 44_000_003)
SMOKE_CALIBRATION_START = 51_000_001
SMOKE_EVALUATION_START = 52_000_001
GAMMA_SCALES = (0.55, 0.80, 1.05)
RESIDUAL_SCALES = (0.10, 0.20, 0.35)
RESERVED_SIMULATION_SEEDS = frozenset(
    seed
    for start, size in (
        *((value, CALIBRATION_BLOCK_SIZE) for value in CALIBRATION_BLOCK_STARTS),
        *((value, EVALUATION_BLOCK_SIZE) for value in EVALUATION_BLOCK_STARTS),
    )
    for seed in range(start, start + size)
)
OPEN_RESERVED_CONFIRMATION = "OPEN_M53A_RESERVED_EVALUATION"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run disposable-seed protocol smoke")
    smoke.add_argument("--candidate-checkpoint", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--mcmc-samples", type=int, default=20)
    smoke.add_argument("--mcmc-transient", type=int, default=20)
    smoke.add_argument(
        "--whittaker-source",
        type=Path,
        default=Path("examples/projects/whittaker_plants_hmsc_book"),
    )

    calibrate = subparsers.add_parser(
        "calibrate", help="freeze the preregistered production calibration"
    )
    calibrate.add_argument("--candidate-checkpoint", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="open all reserved evaluation blocks after calibration freeze"
    )
    evaluate.add_argument("--calibration-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--confirmation", required=True)
    evaluate.add_argument("--sbc-draws", type=int, default=256)
    evaluate.add_argument("--mcmc-samples", type=int, default=80)
    evaluate.add_argument("--mcmc-transient", type=int, default=80)
    evaluate.add_argument(
        "--whittaker-source",
        type=Path,
        default=Path("examples/projects/whittaker_plants_hmsc_book"),
    )

    args = parser.parse_args()
    if args.command == "smoke":
        payload = run_smoke(args)
    elif args.command == "calibrate":
        payload = freeze_production_calibration(args)
    else:
        payload = run_reserved_evaluation(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output = _empty_output(args.output)
    candidate = _load_frozen_candidate(args.candidate_checkpoint)
    calibration_datasets, calibration_rows = factorial_corpus(
        (SMOKE_CALIBRATION_START,), 36
    )
    evaluation_datasets, evaluation_rows = factorial_corpus(
        (SMOKE_EVALUATION_START,), 45
    )
    _assert_disposable(calibration_rows + evaluation_rows)
    candidate.calibration = None
    calibration = candidate.fit_calibration(
        calibration_datasets,
        method="split_conformal_scalar_gamma_scale",
        provenance={
            "protocol_id": f"{PROTOCOL_ID}_smoke",
            "corpus_id": "m53a_disposable_factorial_calibration",
            "seeds": [row["seed"] for row in calibration_rows],
            "independent_from_training": True,
            "disposable_smoke": True,
        },
    )
    checkpoint = package_trait_gamma_calibration(
        args.candidate_checkpoint,
        output / "checkpoint",
        calibration=calibration,
        expected_weights_sha256=FROZEN_WEIGHTS_SHA256,
    )
    engine = TraitGammaNeuralHmscInference.load(checkpoint)
    block = evaluate_simulation_block(
        engine,
        evaluation_datasets,
        evaluation_rows,
        draws=64,
        rank_seed=53_000_001,
    )
    simulated_mcmc = _simulated_mcmc_gate(
        engine=engine,
        dataset=evaluation_datasets[0],
        output=output / "simulated_mcmc",
        seed=53_000_002,
        samples=args.mcmc_samples,
        transient=args.mcmc_transient,
        thin=1,
    )
    whittaker = _whittaker_gate(
        engine=engine,
        source=args.whittaker_source,
        output=output / "whittaker",
        seed=53_000_003,
        samples=args.mcmc_samples,
        transient=args.mcmc_transient,
        thin=1,
    )
    finite = (
        _all_finite(block) and _all_finite(simulated_mcmc) and _all_finite(whittaker)
    )
    payload = {
        "kind": "neural_hmsc_trait_gamma_m53a_smoke",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "decision": "smoke_passed" if finite else "smoke_failed",
        "smoke_passed": finite,
        "reserved_seed_opened": False,
        "candidate_weights_sha256": _sha256(checkpoint / "weights.weights.h5"),
        "weights_unchanged": _sha256(checkpoint / "weights.weights.h5")
        == FROZEN_WEIGHTS_SHA256,
        "calibration": calibration.to_metadata(),
        "calibration_cell_counts": _cell_counts(calibration_rows),
        "evaluation_cell_counts": _cell_counts(evaluation_rows),
        "simulation_block": block,
        "simulated_mcmc": simulated_mcmc,
        "whittaker": whittaker,
        "note": "smoke observations are not promotion evidence",
    }
    payload["smoke_passed"] = bool(
        payload["smoke_passed"] and payload["weights_unchanged"]
    )
    payload["decision"] = "smoke_passed" if payload["smoke_passed"] else "smoke_failed"
    _write_json(output / "m53a_smoke.json", payload)
    return payload


def freeze_production_calibration(args: argparse.Namespace) -> dict[str, Any]:
    output = _empty_output(args.output)
    candidate = _load_frozen_candidate(args.candidate_checkpoint)
    datasets, rows = factorial_corpus(CALIBRATION_BLOCK_STARTS, CALIBRATION_BLOCK_SIZE)
    _assert_exact_seed_blocks(rows, CALIBRATION_BLOCK_STARTS, CALIBRATION_BLOCK_SIZE)
    candidate.calibration = None
    calibration = candidate.fit_calibration(
        datasets,
        method="split_conformal_scalar_gamma_scale",
        provenance={
            "protocol_id": PROTOCOL_ID,
            "corpus_id": "m53a_six_block_factorial_calibration",
            "seeds": [row["seed"] for row in rows],
            "seed_blocks": list(CALIBRATION_BLOCK_STARTS),
            "block_size": CALIBRATION_BLOCK_SIZE,
            "factorial": {
                "gamma_scales": list(GAMMA_SCALES),
                "residual_scales": list(RESIDUAL_SCALES),
                "max_cell_count_difference": 1,
            },
            "independent_from_training": True,
            "candidate_outcomes_used_for_parameters": False,
        },
    )
    checkpoint = package_trait_gamma_calibration(
        args.candidate_checkpoint,
        output / "checkpoint",
        calibration=calibration,
        expected_weights_sha256=FROZEN_WEIGHTS_SHA256,
    )
    payload = {
        "kind": "neural_hmsc_trait_gamma_m53a_calibration_freeze",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_reserved_evaluation",
        "candidate_weights_sha256": FROZEN_WEIGHTS_SHA256,
        "packaged_weights_sha256": _sha256(checkpoint / "weights.weights.h5"),
        "weights_unchanged": _sha256(checkpoint / "weights.weights.h5")
        == FROZEN_WEIGHTS_SHA256,
        "checkpoint_manifest_sha256": _sha256(checkpoint / "neural_checkpoint.json"),
        "calibration_artifact_sha256": _sha256(checkpoint / "gamma_calibration.json"),
        "calibration": calibration.to_metadata(),
        "calibration_seed_blocks": list(CALIBRATION_BLOCK_STARTS),
        "calibration_block_size": CALIBRATION_BLOCK_SIZE,
        "calibration_communities": len(rows),
        "calibration_cell_counts": _cell_counts(rows),
        "reserved_evaluation_seed_blocks": list(EVALUATION_BLOCK_STARTS),
        "reserved_evaluation_opened": False,
    }
    if not payload["weights_unchanged"]:
        raise ValueError("Milestone 53A packaging changed frozen weights")
    _write_json(output / "m53a_calibration_freeze.json", payload)
    validate_calibration_freeze(output)
    return payload


def run_reserved_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirmation != OPEN_RESERVED_CONFIRMATION:
        raise ValueError(
            "reserved evaluation confirmation differs; calibration remains sealed"
        )
    freeze = validate_calibration_freeze(args.calibration_root)
    output = _empty_output(args.output)
    engine = TraitGammaNeuralHmscInference.load(
        Path(args.calibration_root) / "checkpoint"
    )
    block_reports = []
    for index, block_start in enumerate(EVALUATION_BLOCK_STARTS):
        datasets, rows = factorial_corpus((block_start,), EVALUATION_BLOCK_SIZE)
        _assert_exact_seed_blocks(rows, (block_start,), EVALUATION_BLOCK_SIZE)
        metrics = evaluate_simulation_block(
            engine,
            datasets,
            rows,
            draws=args.sbc_draws,
            rank_seed=block_start + 700_000,
        )
        mcmc = _simulated_mcmc_gate(
            engine=engine,
            dataset=datasets[0],
            output=output / f"simulation_block_{block_start}" / "mcmc",
            seed=45_000_001 + index,
            samples=args.mcmc_samples,
            transient=args.mcmc_transient,
            thin=1,
        )
        gates = simulation_gates(metrics, mcmc)
        block_reports.append(
            {
                "block_start": block_start,
                "metrics": metrics,
                "mcmc": mcmc,
                "gates": gates,
                "all_gates_passed": all(gates.values()),
            }
        )
    whittaker_reports = []
    for seed in WHITTAKER_MCMC_SEEDS:
        metrics = _whittaker_gate(
            engine=engine,
            source=args.whittaker_source,
            output=output / f"whittaker_{seed}",
            seed=seed,
            samples=args.mcmc_samples,
            transient=args.mcmc_transient,
            thin=1,
        )
        gates = {
            "brier_ratio": metrics["brier_ratio"] <= 1.05,
            "log_loss_ratio": metrics["log_loss_ratio"] <= 1.05,
            "gamma_mean_mae_mcmc": metrics["gamma_mean_mae_mcmc"] <= 0.35,
            "outcomes_evaluation_only": metrics[
                "outcomes_used_for_training_or_selection"
            ]
            is False,
        }
        whittaker_reports.append(
            {
                "seed": seed,
                "metrics": metrics,
                "gates": gates,
                "all_gates_passed": all(gates.values()),
            }
        )
    passed = all(row["all_gates_passed"] for row in block_reports) and all(
        row["all_gates_passed"] for row in whittaker_reports
    )
    payload = {
        "kind": "neural_hmsc_trait_gamma_m53a_evaluation",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "decision": (
            "trait_gamma_probit_promoted"
            if passed
            else "trait_gamma_probit_terminal_failure"
        ),
        "all_gates_passed": passed,
        "terminal_rule_applies": not passed,
        "calibration_freeze_sha256": _sha256(
            Path(args.calibration_root) / "m53a_calibration_freeze.json"
        ),
        "calibration_freeze": freeze,
        "simulation_blocks": block_reports,
        "whittaker_replays": whittaker_reports,
        "reserved_evaluation_opened": True,
        "existing_baselines_modified": False,
    }
    _write_json(output / "m53a_evaluation.json", payload)
    return payload


def factorial_corpus(
    block_starts: Sequence[int], block_size: int
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Build a deterministic near-balanced 3 by 3 simulation factorial."""
    if block_size <= 0 or not block_starts:
        raise ValueError("factorial corpus requires positive non-empty blocks")
    combinations = [
        (gamma_scale, residual_scale)
        for gamma_scale in GAMMA_SCALES
        for residual_scale in RESIDUAL_SCALES
    ]
    datasets = []
    rows = []
    global_index = 0
    for block_start in block_starts:
        for offset in range(int(block_size)):
            gamma_scale, residual_scale = combinations[global_index % len(combinations)]
            seed = int(block_start) + offset
            datasets.append(
                simulate_trait_gamma_boundary_dataset(
                    n_sites=40,
                    n_species=75,
                    seed=seed,
                    gamma_scale=gamma_scale,
                    beta_residual_scale=residual_scale,
                )
            )
            rows.append(
                {
                    "seed": seed,
                    "block_start": int(block_start),
                    "gamma_scale": gamma_scale,
                    "residual_scale": residual_scale,
                    "cell": _cell_name(gamma_scale, residual_scale),
                }
            )
            global_index += 1
    counts = list(_cell_counts(rows).values())
    if max(counts) - min(counts) > 1:
        raise ValueError("factorial cell counts differ by more than one")
    return datasets, rows


def evaluate_simulation_block(
    engine: TraitGammaNeuralHmscInference,
    datasets: Sequence[Any],
    rows: Sequence[dict[str, Any]],
    *,
    draws: int,
    rank_seed: int,
) -> dict[str, Any]:
    if len(datasets) != len(rows) or not datasets:
        raise ValueError("simulation datasets and rows must be non-empty and aligned")
    means = []
    scales = []
    truths = []
    for dataset in datasets:
        posterior = engine.predict_gamma_posterior(dataset)
        means.append(posterior.mean.numpy()[0])
        scales.append(posterior.scale.numpy()[0])
        truths.append(dataset.truth_gamma.to_numpy(dtype=float))
    mean = np.stack(means)
    scale = np.stack(scales)
    truth = np.stack(truths)
    anchor = TraitGammaNeuralHmscInference.for_trait_gamma(
        n_sites=40,
        n_species=75,
        n_covariates=2,
        n_traits=1,
        hidden_units=(64, 64),
    )
    anchor_means = np.stack(
        [
            anchor.predict_gamma_posterior(dataset, calibrated=False).mean.numpy()[0]
            for dataset in datasets
        ]
    )
    covered = np.abs(mean - truth) <= 1.959963984540054 * scale
    rng = np.random.default_rng(rank_seed)
    samples = rng.normal(mean, scale, size=(int(draws),) + mean.shape)
    ranks = np.mean(samples < truth[None, ...], axis=0)
    cell_coverage = {}
    for cell in sorted({row["cell"] for row in rows}):
        mask = np.asarray([row["cell"] == cell for row in rows], dtype=bool)
        cell_coverage[cell] = float(np.mean(covered[mask]))
    rmse = float(np.sqrt(np.mean(np.square(mean - truth))))
    anchor_rmse = float(np.sqrt(np.mean(np.square(anchor_means - truth))))
    return {
        "n_communities": len(datasets),
        "n_coefficients": int(truth.size),
        "coverage_95": float(np.mean(covered)),
        "coverage_by_coefficient": {
            "Intercept": float(np.mean(covered[:, 0, :])),
            "TMG": float(np.mean(covered[:, 1, :])),
        },
        "coverage_by_cell": cell_coverage,
        "rank_mean": float(np.mean(ranks)),
        "rank_variance": float(np.var(ranks)),
        "gamma_rmse": rmse,
        "anchor_gamma_rmse": anchor_rmse,
        "gamma_anchor_rmse_ratio": rmse / max(anchor_rmse, 1e-12),
        "cell_counts": _cell_counts(rows),
    }


def simulation_gates(metrics: dict[str, Any], mcmc: dict[str, Any]) -> dict[str, bool]:
    return {
        "overall_coverage": 0.90 <= metrics["coverage_95"] <= 0.99,
        "intercept_coverage": metrics["coverage_by_coefficient"]["Intercept"] >= 0.90,
        "tmg_coverage": metrics["coverage_by_coefficient"]["TMG"] >= 0.90,
        "cell_coverage": min(metrics["coverage_by_cell"].values()) >= 0.85,
        "rank_mean": 0.40 <= metrics["rank_mean"] <= 0.60,
        "rank_variance": 0.06 <= metrics["rank_variance"] <= 0.11,
        "anchor_no_degradation": metrics["gamma_anchor_rmse_ratio"] <= 1.05,
        "mcmc_gamma_rmse": mcmc["gamma_rmse_ratio"] <= 1.25,
        "mcmc_brier": mcmc["brier_ratio"] <= 1.05,
        "mcmc_log_loss": mcmc["log_loss_ratio"] <= 1.05,
    }


def validate_calibration_freeze(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    path = root / "m53a_calibration_freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "neural_hmsc_trait_gamma_m53a_calibration_freeze":
        raise ValueError("unsupported Milestone 53A calibration freeze")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Milestone 53A protocol identifier differs")
    if payload.get("status") != "frozen_before_reserved_evaluation":
        raise ValueError("Milestone 53A calibration is not sealed")
    if payload.get("candidate_weights_sha256") != FROZEN_WEIGHTS_SHA256:
        raise ValueError("Milestone 53A frozen candidate hash differs")
    if payload.get("packaged_weights_sha256") != FROZEN_WEIGHTS_SHA256:
        raise ValueError("Milestone 53A packaged weights hash differs")
    if tuple(payload.get("calibration_seed_blocks", ())) != CALIBRATION_BLOCK_STARTS:
        raise ValueError("Milestone 53A calibration seed blocks differ")
    if payload.get("calibration_block_size") != CALIBRATION_BLOCK_SIZE:
        raise ValueError("Milestone 53A calibration block size differs")
    if (
        tuple(payload.get("reserved_evaluation_seed_blocks", ()))
        != EVALUATION_BLOCK_STARTS
    ):
        raise ValueError("Milestone 53A reserved evaluation blocks differ")
    checkpoint = root / "checkpoint"
    if _sha256(checkpoint / "weights.weights.h5") != FROZEN_WEIGHTS_SHA256:
        raise ValueError("Milestone 53A checkpoint weights were modified")
    if _sha256(checkpoint / "neural_checkpoint.json") != payload.get(
        "checkpoint_manifest_sha256"
    ):
        raise ValueError("Milestone 53A checkpoint manifest hash differs")
    if _sha256(checkpoint / "gamma_calibration.json") != payload.get(
        "calibration_artifact_sha256"
    ):
        raise ValueError("Milestone 53A calibration artifact hash differs")
    engine = TraitGammaNeuralHmscInference.load(checkpoint)
    if engine.calibration is None or engine.calibration.method != (
        "split_conformal_scalar_gamma_scale"
    ):
        raise ValueError("Milestone 53A conformal calibration is missing")
    return payload


def _load_frozen_candidate(path: Path) -> TraitGammaNeuralHmscInference:
    path = path.expanduser().resolve()
    observed = _sha256(path / "weights.weights.h5")
    if observed != FROZEN_WEIGHTS_SHA256:
        raise ValueError(
            f"candidate weights {observed} differ from frozen {FROZEN_WEIGHTS_SHA256}"
        )
    return TraitGammaNeuralHmscInference.load(path)


def _assert_disposable(rows: Sequence[dict[str, Any]]) -> None:
    overlap = sorted({int(row["seed"]) for row in rows} & RESERVED_SIMULATION_SEEDS)
    if overlap:
        raise ValueError(f"smoke attempted to open reserved seeds: {overlap[:3]}")


def _assert_exact_seed_blocks(
    rows: Sequence[dict[str, Any]], block_starts: Sequence[int], block_size: int
) -> None:
    observed = {int(row["seed"]) for row in rows}
    expected = {
        seed
        for start in block_starts
        for seed in range(int(start), int(start) + int(block_size))
    }
    if observed != expected:
        raise ValueError("Milestone 53A seed block content differs")


def _cell_name(gamma_scale: float, residual_scale: float) -> str:
    return f"gamma_{gamma_scale:.2f}__residual_{residual_scale:.2f}"


def _cell_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["cell"]) for row in rows).items()))


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _empty_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Milestone 53A output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
