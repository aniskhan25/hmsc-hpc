"""Qualification metrics for generative Neural-HMSC iid probit v1.

The functions in this module are model-agnostic once posterior state draws are
available. Production orchestration remains in the sealed example harness.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import ndtr

from pyhmsc.neural.generative_iid import (
    GENERATIVE_IID_PROTOCOL,
    GenerativeIidDataset,
    JointStateLayout,
    association_to_correlation,
)
from pyhmsc.neural.generative_iid_mcmc import fixed_rademacher_projections


INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)
POSTERIOR_DRAWS = 256
EPSILON = 1e-6


def evaluate_state_draws(
    dataset: GenerativeIidDataset,
    state_draws: np.ndarray,
    *,
    layout: JointStateLayout,
    method: str,
    inference_seconds: float | None = None,
    peak_device_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Evaluate one context from draws x padded joint state values."""
    draws = np.asarray(state_draws, dtype=np.float64)
    if draws.ndim == 3:
        draws = draws.reshape(-1, draws.shape[-1])
    if draws.ndim != 2 or draws.shape[1] != layout.size:
        raise ValueError("state_draws must have shape draws x layout.size")
    if not np.isfinite(draws).all():
        raise ValueError("state_draws contain non-finite values")
    n_sites, n_species = dataset.Y.shape
    if n_sites > layout.max_sites or n_species > layout.max_species:
        raise ValueError("dataset exceeds posterior state layout")

    state = _unpack_numpy(draws, layout)
    beta = state["Beta"][:, :, :n_species]
    eta = state["Eta"][:, :n_sites, :]
    loadings = state["Lambda"][:, :, :n_species]
    random_effect = np.einsum("dnh,dhs->dns", eta, loadings)
    association = np.einsum("dhs,dht->dst", loadings, loadings)
    correlation = _association_draws_to_correlation(association)
    probabilities = ndtr(
        np.einsum("ni,dis->dns", dataset.X, beta) + random_effect
    )

    hidden = ~np.asarray(dataset.response_mask, dtype=bool)
    if not hidden.any():
        raise ValueError("qualification contexts require masked response cells")
    masked_probability = np.mean(probabilities, axis=0)[hidden]
    masked_y = np.asarray(dataset.Y, dtype=float)[hidden]
    masked_brier, masked_log_loss = _proper_scores(masked_y, masked_probability)

    new_site_mask = deterministic_new_site_mask(
        n_sites, seed=int(dataset.metadata["seed"])
    )
    new_site_probability = _new_site_probability(
        dataset,
        beta,
        loadings,
        site_mask=new_site_mask,
    )
    new_site_y = np.asarray(dataset.Y, dtype=float)[new_site_mask]
    new_site_brier, new_site_log_loss = _proper_scores(
        new_site_y.ravel(),
        new_site_probability.ravel(),
    )

    truth = {
        "Beta": np.asarray(dataset.truth_beta, dtype=float),
        "R": np.asarray(dataset.truth_random_effect, dtype=float),
        "alpha": np.asarray(dataset.truth_alpha, dtype=float),
        "log_tau": np.asarray(dataset.truth_log_tau, dtype=float),
    }
    values = {
        "Beta": beta,
        "R": random_effect,
        "alpha": state["alpha"],
        "log_tau": state["log_tau"],
    }
    marginal = {
        name: _marginal_diagnostics(values[name], truth[name])
        for name in values
    }
    projections = {
        family: _projection_diagnostics(
            values[family],
            truth[family],
            family=family,
        )
        for family in ("Beta", "R")
    }
    projections["C"] = _projection_diagnostics(
        _off_diagonal_draws(correlation),
        _off_diagonal_values(
            np.asarray(
                dataset.truth_association_correlation,
                dtype=float,
            )
        ),
        family="C",
    )

    richness_coverage, prevalence_coverage = _posterior_predictive_coverage(
        dataset,
        probabilities,
    )
    off_diagonal = np.triu_indices(n_species, k=1)
    mean_c = np.mean(correlation, axis=0)
    truth_c = np.asarray(dataset.truth_association_correlation, dtype=float)
    association_truth_correlation = _safe_correlation(
        mean_c[off_diagonal],
        truth_c[off_diagonal],
    )
    association_rmse = float(
        np.sqrt(np.mean(np.square(mean_c[off_diagonal] - truth_c[off_diagonal])))
    )
    random_effect_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.mean(random_effect, axis=0)
                    - np.asarray(dataset.truth_random_effect, dtype=float)
                )
            )
        )
    )

    metadata = {
        key: dataset.metadata[key]
        for key in (
            "seed",
            "n_sites",
            "n_species",
            "covariate_shape",
            "loading_stratum",
            "prevalence_stratum",
        )
    }
    return {
        "protocol": GENERATIVE_IID_PROTOCOL,
        "method": str(method),
        **metadata,
        "draw_count": int(draws.shape[0]),
        "all_finite": True,
        "marginal": marginal,
        "projections": projections,
        "association_truth_correlation": association_truth_correlation,
        "association_rmse": association_rmse,
        "association_vector_mean": mean_c[off_diagonal].tolist(),
        "random_effect_rmse": random_effect_rmse,
        "mean_absolute_off_diagonal_c": float(
            np.mean(np.abs(mean_c[off_diagonal]))
        ),
        "masked_cell_brier": masked_brier,
        "masked_cell_log_loss": masked_log_loss,
        "new_site_brier": new_site_brier,
        "new_site_log_loss": new_site_log_loss,
        "site_richness_90_coverage": richness_coverage,
        "species_prevalence_90_coverage": prevalence_coverage,
        "invariant_vector_draws": _invariant_vector_draws(
            beta,
            random_effect,
            correlation,
            truth_beta=np.asarray(dataset.truth_beta, dtype=float),
            truth_random_effect=np.asarray(
                dataset.truth_random_effect, dtype=float
            ),
            truth_correlation=np.asarray(
                dataset.truth_association_correlation, dtype=float
            ),
        ),
        "inference_seconds": (
            None if inference_seconds is None else float(inference_seconds)
        ),
        "peak_device_memory_bytes": (
            None
            if peak_device_memory_bytes is None
            else int(peak_device_memory_bytes)
        ),
    }


def deterministic_new_site_mask(n_sites: int, *, seed: int) -> np.ndarray:
    """Return the protocol-owned deterministic 20% site split."""
    if n_sites < 2:
        raise ValueError("new-site split requires at least two sites")
    count = max(1, int(round(0.20 * n_sites)))
    count = min(count, n_sites - 1)
    key = f"{GENERATIVE_IID_PROTOCOL}:new_site:{int(seed)}".encode()
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    )
    selected = np.sort(rng.choice(n_sites, size=count, replace=False))
    mask = np.zeros(n_sites, dtype=bool)
    mask[selected] = True
    return mask


def attach_comparator_metrics(
    candidate: Mapping[str, Any],
    comparator: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Attach preregistered pairwise ratios and invariant comparisons."""
    if int(candidate["seed"]) != int(comparator["seed"]):
        raise ValueError("candidate and comparator owning seeds differ")
    output = dict(candidate)
    output[f"{prefix}_beta_width_ratio"] = _ratio(
        candidate["marginal"]["Beta"]["interval_width_95_median"],
        comparator["marginal"]["Beta"]["interval_width_95_median"],
    )
    output[f"{prefix}_r_width_ratio"] = _ratio(
        candidate["marginal"]["R"]["interval_width_95_median"],
        comparator["marginal"]["R"]["interval_width_95_median"],
    )
    output[f"{prefix}_energy_score_ratio"] = _ratio(
        invariant_energy_score(candidate),
        invariant_energy_score(comparator),
    )
    for metric in (
        "association_rmse",
        "masked_cell_brier",
        "masked_cell_log_loss",
        "new_site_brier",
        "new_site_log_loss",
    ):
        output[f"{prefix}_{metric}_ratio"] = _ratio(
            candidate[metric],
            comparator[metric],
        )
    output[f"{prefix}_mean_absolute_off_diagonal_c"] = comparator[
        "mean_absolute_off_diagonal_c"
    ]
    return output


def invariant_energy_score(row: Mapping[str, Any]) -> float:
    """Energy score of registered invariant draws against the truth vector."""
    payload = row["invariant_vector_draws"]
    draws = np.asarray(payload["draws"], dtype=np.float64)
    truth = np.asarray(payload["truth"], dtype=np.float64)
    if draws.ndim != 2 or truth.shape != (draws.shape[1],):
        raise ValueError("invariant draw payload shape differs")
    first = np.mean(np.linalg.norm(draws - truth[None, :], axis=1))
    shifted = np.roll(draws, shift=max(1, draws.shape[0] // 2), axis=0)
    second = 0.5 * np.mean(np.linalg.norm(draws - shifted, axis=1))
    return float(first - second)


def fixed_validation_gates(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    ablation_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    python_rows: Sequence[Mapping[str, Any]],
    v0_rows: Sequence[Mapping[str, Any]],
    operational: Mapping[str, bool],
    mcmc_diagnostics: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, float],
) -> dict[str, bool]:
    """Evaluate every frozen 502M gate as an explicit conjunct."""
    candidate = _indexed(candidate_rows, "candidate")
    ablation = _indexed(ablation_rows, "ablation")
    exact = _indexed(exact_rows, "exact MCMC")
    python = _indexed(python_rows, "Python HMSC")
    v0 = _indexed(v0_rows, "v0.1")
    if len(candidate) != 324:
        raise ValueError("fixed validation requires exactly 324 candidate rows")
    if set(ablation) != set(candidate):
        raise ValueError("fixed ablation seed ownership differs")
    expected_exact = set(fixed_mcmc_subset_seeds(candidate_rows))
    if set(exact) != expected_exact or set(python) != expected_exact:
        raise ValueError("fixed 36-context comparator ownership differs")
    expected_v0 = {
        seed
        for seed, row in candidate.items()
        if int(row["n_sites"]) == 40 and int(row["n_species"]) == 75
    }
    if set(v0) != expected_v0:
        raise ValueError("matched v0.1 comparator ownership differs")

    rows = list(candidate.values())
    exact_pairs = [
        attach_comparator_metrics(candidate[seed], exact[seed], prefix="exact")
        for seed in sorted(exact)
    ]
    gates: dict[str, bool] = {
        f"operational_{name}": value
        for name, value in sorted(operational.items())
        if isinstance(value, bool)
    }
    gates.update(
        {
            "operational_324_factorial_cells": len(candidate) == 324,
            "operational_all_inference_finite": all(
                row.get("all_finite") is True for row in rows
            ),
            "operational_ablation_all_finite": all(
                row.get("all_finite") is True for row in ablation.values()
            ),
            "operational_exact_all_finite": all(
                row.get("all_finite") is True for row in exact.values()
            ),
            "operational_python_hmsc_all_finite": all(
                row.get("all_finite") is True for row in python.values()
            ),
            "operational_v0_1_all_finite": all(
                row.get("all_finite") is True for row in v0.values()
            ),
            "operational_exact_mcmc_diagnostics": bool(mcmc_diagnostics)
            and all(
                float(item["split_rhat_max"]) <= 1.05
                and float(item["bulk_ess_min"]) >= 200.0
                for item in mcmc_diagnostics
            ),
        }
    )

    gates["beta_95_aggregate"] = _between(
        _pooled(rows, "Beta", "coverage_95"), 0.925, 0.975
    )
    gates["r_95_aggregate"] = _between(
        _pooled(rows, "R", "coverage_95"), 0.90, 0.98
    )
    gates["alpha_95_aggregate"] = _between(
        _pooled(rows, "alpha", "coverage_95"), 0.90, 0.99
    )
    gates["log_tau_95_aggregate"] = _between(
        _pooled(rows, "log_tau", "coverage_95"), 0.90, 0.99
    )
    gates["beta_registered_strata_coverage"] = _all_strata_between(
        rows, family="Beta", metric="coverage_95", lower=0.89, upper=0.99
    )
    gates["r_registered_strata_coverage"] = _all_strata_between(
        rows, family="R", metric="coverage_95", lower=0.87, upper=0.995
    )
    gates["beta_rank_mean_aggregate"] = abs(
        _pooled(rows, "Beta", "rank_mean") - 0.5
    ) <= 0.04
    gates["r_rank_mean_aggregate"] = abs(
        _pooled(rows, "R", "rank_mean") - 0.5
    ) <= 0.04
    gates["registered_strata_rank_means"] = all(
        _all_strata_centered(rows, family=family, tolerance=0.07)
        for family in ("Beta", "R")
    )
    gates["beta_rank_variance_aggregate"] = _between(
        _pooled(rows, "Beta", "rank_variance"), 0.060, 0.108
    )
    gates["r_rank_variance_aggregate"] = _between(
        _pooled(rows, "R", "rank_variance"), 0.060, 0.108
    )
    gates["beta_exact_width_ratio"] = _between(
        _median(exact_pairs, "exact_beta_width_ratio"), 0.75, 1.35
    )
    gates["r_exact_width_ratio"] = _between(
        _median(exact_pairs, "exact_r_width_ratio"), 0.75, 1.35
    )

    for family in ("Beta", "R", "C"):
        prefix = family.lower()
        gates[f"{prefix}_projection_95_coverage"] = _between(
            _pooled_projection(rows, family, "coverage_95"), 0.91, 0.985
        )
        gates[f"{prefix}_projection_rank_mean"] = abs(
            _pooled_projection(rows, family, "rank_mean") - 0.5
        ) <= 0.05
        gates[f"{prefix}_projection_rank_variance"] = _between(
            _pooled_projection(rows, family, "rank_variance"),
            0.055,
            0.115,
        )
    gates["exact_energy_score_ratio_aggregate"] = (
        _median(exact_pairs, "exact_energy_score_ratio") <= 1.10
    )
    gates["exact_energy_score_ratio_regimes"] = _all_regimes_at_most(
        exact_pairs, "exact_energy_score_ratio", 1.20
    )
    gates["covariance_jitter_fraction"] = float(
        operational.get("covariance_jitter_fraction", 1.0)
    ) <= 0.01
    gates["covariance_condition_max"] = float(
        operational.get("covariance_condition_max", math.inf)
    ) <= 1e8

    medium_strong = [
        row for row in rows if row["loading_stratum"] in {"medium", "strong"}
    ]
    association_values = [
        float(row["association_truth_correlation"]) for row in medium_strong
    ]
    gates["association_truth_correlation_median"] = (
        float(np.median(association_values)) >= 0.65
    )
    gates["association_truth_correlation_p10"] = (
        float(np.quantile(association_values, 0.10)) >= 0.25
    )
    gates["random_effect_rmse_vs_ablation"] = (
        np.median([row["random_effect_rmse"] for row in medium_strong])
        <= 0.85
        * np.median(
            [
                ablation[int(row["seed"])]["random_effect_rmse"]
                for row in medium_strong
            ]
        )
    )
    gates["association_rmse_vs_exact"] = (
        _median(exact_pairs, "exact_association_rmse_ratio") <= 1.15
    )
    gates["association_correlation_vs_python"] = (
        _candidate_python_association_correlation(candidate, python) >= 0.70
    )

    weak = [row for row in rows if row["loading_stratum"] == "weak"]
    gates["weak_masked_brier_vs_ablation"] = _score_ratio(
        weak, ablation, "masked_cell_brier"
    ) <= 1.01
    gates["weak_masked_log_loss_vs_ablation"] = _score_ratio(
        weak, ablation, "masked_cell_log_loss"
    ) <= 1.01
    gates["weak_association_magnitude_vs_exact"] = all(
        candidate[seed]["mean_absolute_off_diagonal_c"]
        <= exact[seed]["mean_absolute_off_diagonal_c"] + 0.05
        for seed in expected_exact
        if candidate[seed]["loading_stratum"] == "weak"
    )

    gates["medium_strong_masked_brier_vs_ablation"] = _score_ratio(
        medium_strong, ablation, "masked_cell_brier"
    ) <= 0.98
    gates["medium_strong_masked_log_loss_vs_ablation"] = _score_ratio(
        medium_strong, ablation, "masked_cell_log_loss"
    ) <= 0.99
    for metric in ("masked_cell_brier", "masked_cell_log_loss"):
        gates[f"exact_{metric}_aggregate"] = (
            _median(exact_pairs, f"exact_{metric}_ratio") <= 1.10
        )
        gates[f"exact_{metric}_strata"] = _all_strata_ratio_at_most(
            exact_pairs, f"exact_{metric}_ratio", 1.20
        )
        gates[f"python_{metric}_aggregate"] = _comparator_score_ratio(
            candidate, python, metric
        ) <= 1.10
    gates["exact_new_site_brier"] = (
        _median(exact_pairs, "exact_new_site_brier_ratio") <= 1.03
    )
    gates["exact_new_site_log_loss"] = (
        _median(exact_pairs, "exact_new_site_log_loss_ratio") <= 1.03
    )
    for metric in (
        "site_richness_90_coverage",
        "species_prevalence_90_coverage",
    ):
        gates[f"{metric}_aggregate"] = _between(
            float(np.mean([row[metric] for row in rows])), 0.84, 0.96
        )
        gates[f"{metric}_strata"] = _all_scalar_strata_between(
            rows, metric, 0.78, 0.99
        )
    gates["matched_v0_beta_coverage"] = all(
        candidate[seed]["marginal"]["Beta"]["coverage_95"]
        >= v0[seed]["marginal"]["Beta"]["coverage_95"] - 0.02
        for seed in expected_v0
    )
    gates["matched_v0_masked_scores"] = all(
        _ratio(candidate[seed][metric], v0[seed][metric]) <= 1.03
        for seed in expected_v0
        for metric in ("masked_cell_brier", "masked_cell_log_loss")
    )

    gates["runtime_training_dev_gpu_hours"] = float(
        runtime.get("training_dev_gpu_hours", math.inf)
    ) <= 24.0
    gates["runtime_max_shape_inference_seconds"] = float(
        runtime.get("max_shape_inference_seconds", math.inf)
    ) <= 5.0
    gates["runtime_peak_device_memory"] = float(
        runtime.get("peak_device_memory_bytes", math.inf)
    ) <= 32 * 1024**3
    gates["runtime_speedup_vs_exact_mcmc"] = float(
        runtime.get("speedup_vs_exact_mcmc", 0.0)
    ) >= 20.0
    return gates


def fixed_mcmc_subset_seeds(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    """Select the frozen 36-context MCMC subset from factorial metadata."""
    by_shape: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_shape.setdefault(
            (int(row["n_sites"]), int(row["n_species"])), []
        ).append(row)
    selected = []
    regimes = {
        ("weak", "rare"),
        ("weak", "common"),
        ("strong", "rare"),
        ("strong", "common"),
    }
    for shape_index, shape in enumerate(sorted(by_shape)):
        covariate = "normal" if shape_index % 2 == 0 else "right_skewed"
        for loading, prevalence in sorted(regimes):
            matches = [
                row
                for row in by_shape[shape]
                if row["covariate_shape"] == covariate
                and row["loading_stratum"] == loading
                and row["prevalence_stratum"] == prevalence
            ]
            if not matches:
                raise ValueError("fixed MCMC subset cell is missing")
            selected.append(min(int(row["seed"]) for row in matches))
    if len(selected) != 36:
        raise AssertionError("fixed MCMC subset must contain 36 contexts")
    return tuple(sorted(selected))


def qualification_report(
    *,
    gates: Mapping[str, bool],
    freeze_binding: Mapping[str, Any],
    seed_roles: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable fixed-validation decision record."""
    failed = sorted(name for name, value in gates.items() if not bool(value))
    return {
        "schema_version": 1,
        "kind": "generative_iid_v1_502m_fixed_validation",
        "protocol": GENERATIVE_IID_PROTOCOL,
        "freeze_binding": dict(freeze_binding),
        "seed_roles": dict(seed_roles),
        "artifacts": dict(artifacts),
        "gates": {name: bool(value) for name, value in sorted(gates.items())},
        "all_gates_passed": not failed,
        "failed_gates": failed,
        "reserved_seed_ranges_opened": False,
        "redesign_seed_ranges_opened": False,
        "decision": (
            "eligible_to_authorize_503m_505m"
            if not failed
            else "stop_before_reserved_evaluation"
        ),
    }


def _unpack_numpy(
    draws: np.ndarray, layout: JointStateLayout
) -> dict[str, np.ndarray]:
    return {
        "alpha": draws[:, layout.alpha_slice][:, 0],
        "Beta": draws[:, layout.beta_slice].reshape(
            draws.shape[0], layout.n_covariates, layout.max_species
        ),
        "Eta": draws[:, layout.eta_slice].reshape(
            draws.shape[0], layout.max_sites, layout.n_factors
        ),
        "Lambda": draws[:, layout.lambda_slice].reshape(
            draws.shape[0], layout.n_factors, layout.max_species
        ),
        "log_tau": draws[:, layout.log_tau_slice][:, 0],
    }


def _marginal_diagnostics(
    draws: np.ndarray, truth: np.ndarray
) -> dict[str, float | int]:
    values = np.asarray(draws, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if values.shape[1:] != target.shape:
        raise ValueError("marginal draws and truth shape differ")
    flattened = values.reshape(values.shape[0], -1)
    truth_flat = target.reshape(-1)
    output: dict[str, float | int] = {"element_count": int(truth_flat.size)}
    for level in INTERVAL_LEVELS:
        tail = (1.0 - level) / 2.0
        lower = np.quantile(flattened, tail, axis=0)
        upper = np.quantile(flattened, 1.0 - tail, axis=0)
        suffix = int(round(level * 100))
        output[f"coverage_{suffix}"] = float(
            np.mean((lower <= truth_flat) & (truth_flat <= upper))
        )
        output[f"interval_width_{suffix}_median"] = float(
            np.median(upper - lower)
        )
    normalized_ranks = (
        np.sum(flattened < truth_flat[None, :], axis=0) + 0.5
    ) / (flattened.shape[0] + 1.0)
    output["rank_mean"] = float(np.mean(normalized_ranks))
    output["rank_variance"] = float(np.var(normalized_ranks))
    output["rank_count"] = int(normalized_ranks.size)
    return output


def _projection_diagnostics(
    draws: np.ndarray,
    truth: np.ndarray,
    *,
    family: str,
) -> dict[str, float | int]:
    values = np.asarray(draws, dtype=np.float64).reshape(len(draws), -1)
    target = np.asarray(truth, dtype=np.float64).reshape(-1)
    projection = fixed_rademacher_projections(
        family, values.shape[1], count=16
    )
    projected = values @ projection.T
    truth_projected = target @ projection.T
    result = _marginal_diagnostics(projected, truth_projected)
    result["projection_count"] = 16
    return result


def _invariant_vector_draws(
    beta: np.ndarray,
    random_effect: np.ndarray,
    correlation: np.ndarray,
    *,
    truth_beta: np.ndarray,
    truth_random_effect: np.ndarray,
    truth_correlation: np.ndarray,
) -> dict[str, Any]:
    vectors = []
    truths = []
    for family, values, truth in (
        ("Beta", beta, truth_beta),
        ("R", random_effect, truth_random_effect),
        (
            "C",
            _off_diagonal_draws(correlation),
            _off_diagonal_values(truth_correlation),
        ),
    ):
        flattened = values.reshape(values.shape[0], -1)
        projection = fixed_rademacher_projections(
            family, flattened.shape[1], count=16
        )
        vectors.append(flattened @ projection.T)
        truths.append(np.asarray(truth, dtype=float).reshape(-1) @ projection.T)
    draws = np.concatenate(vectors, axis=1)
    return {
        "draws": draws.tolist(),
        "truth": np.concatenate(truths).tolist(),
    }


def _association_draws_to_correlation(
    association: np.ndarray,
) -> np.ndarray:
    diagonal = np.maximum(
        np.diagonal(association, axis1=-2, axis2=-1), 1e-8
    )
    correlation = association / np.sqrt(
        diagonal[..., :, None] * diagonal[..., None, :]
    )
    indices = np.arange(correlation.shape[-1])
    correlation[..., indices, indices] = 1.0
    return correlation


def _off_diagonal_draws(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    indices = np.triu_indices(matrix.shape[-1], k=1)
    return matrix[..., indices[0], indices[1]]


def _off_diagonal_values(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    indices = np.triu_indices(matrix.shape[-1], k=1)
    return matrix[indices]


def _new_site_probability(
    dataset: GenerativeIidDataset,
    beta: np.ndarray,
    loadings: np.ndarray,
    *,
    site_mask: np.ndarray,
) -> np.ndarray:
    count = int(np.sum(site_mask))
    seed = int(dataset.metadata["seed"])
    eta_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 505])
    )
    eta = eta_rng.normal(size=(beta.shape[0], count, loadings.shape[1]))
    fixed = np.einsum("ni,dis->dns", dataset.X[site_mask], beta)
    random = np.einsum("dnh,dhs->dns", eta, loadings)
    return np.mean(ndtr(fixed + random), axis=0)


def _posterior_predictive_coverage(
    dataset: GenerativeIidDataset,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(dataset.metadata["seed"]), 606])
    )
    replicated = rng.binomial(1, np.clip(probabilities, EPSILON, 1 - EPSILON))
    richness = np.sum(replicated, axis=2)
    observed_richness = np.sum(dataset.Y, axis=1)
    richness_lo, richness_hi = np.quantile(richness, [0.05, 0.95], axis=0)
    prevalence = np.mean(replicated, axis=1)
    observed_prevalence = np.mean(dataset.Y, axis=0)
    prevalence_lo, prevalence_hi = np.quantile(
        prevalence, [0.05, 0.95], axis=0
    )
    return (
        float(
            np.mean(
                (richness_lo <= observed_richness)
                & (observed_richness <= richness_hi)
            )
        ),
        float(
            np.mean(
                (prevalence_lo <= observed_prevalence)
                & (observed_prevalence <= prevalence_hi)
            )
        ),
    )


def _proper_scores(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    target = np.asarray(y, dtype=float)
    prediction = np.clip(np.asarray(probability, dtype=float), EPSILON, 1 - EPSILON)
    if target.shape != prediction.shape:
        raise ValueError("proper-score target and prediction shape differ")
    brier = float(np.mean(np.square(prediction - target)))
    log_loss = float(
        -np.mean(target * np.log(prediction) + (1.0 - target) * np.log(1.0 - prediction))
    )
    return brier, log_loss


def _indexed(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[int, Mapping[str, Any]]:
    output = {int(row["seed"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label} rows contain duplicate owning seeds")
    return output


def _pooled(
    rows: Sequence[Mapping[str, Any]], family: str, metric: str
) -> float:
    weights = np.asarray(
        [row["marginal"][family]["element_count"] for row in rows], dtype=float
    )
    values = np.asarray(
        [row["marginal"][family][metric] for row in rows], dtype=float
    )
    if metric == "rank_variance":
        means = np.asarray(
            [row["marginal"][family]["rank_mean"] for row in rows],
            dtype=float,
        )
        pooled_mean = float(np.average(means, weights=weights))
        second = np.average(values + np.square(means), weights=weights)
        return float(second - pooled_mean**2)
    return float(np.average(values, weights=weights))


def _pooled_projection(
    rows: Sequence[Mapping[str, Any]], family: str, metric: str
) -> float:
    values = np.asarray(
        [row["projections"][family][metric] for row in rows],
        dtype=float,
    )
    if metric == "rank_variance":
        means = np.asarray(
            [row["projections"][family]["rank_mean"] for row in rows],
            dtype=float,
        )
        return float(np.mean(values + np.square(means)) - np.mean(means) ** 2)
    return float(np.mean(values))


_STRATUM_KEYS = (
    "n_sites",
    "n_species",
    "covariate_shape",
    "loading_stratum",
    "prevalence_stratum",
)


def _all_strata_between(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    metric: str,
    lower: float,
    upper: float,
) -> bool:
    return all(
        _between(_pooled(group, family, metric), lower, upper)
        for group in _stratum_groups(rows)
    )


def _all_strata_centered(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    tolerance: float,
) -> bool:
    return all(
        abs(_pooled(group, family, "rank_mean") - 0.5) <= tolerance
        for group in _stratum_groups(rows)
    )


def _all_scalar_strata_between(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    lower: float,
    upper: float,
) -> bool:
    return all(
        _between(float(np.mean([row[metric] for row in group])), lower, upper)
        for group in _stratum_groups(rows)
    )


def _stratum_groups(
    rows: Sequence[Mapping[str, Any]],
) -> Iterable[list[Mapping[str, Any]]]:
    for key in _STRATUM_KEYS:
        values = sorted({row[key] for row in rows}, key=str)
        for value in values:
            yield [row for row in rows if row[key] == value]


def _all_regimes_at_most(
    rows: Sequence[Mapping[str, Any]], metric: str, threshold: float
) -> bool:
    for loading, prevalence in (
        ("weak", "rare"),
        ("weak", "common"),
        ("strong", "rare"),
        ("strong", "common"),
    ):
        group = [
            row
            for row in rows
            if row["loading_stratum"] == loading
            and row["prevalence_stratum"] == prevalence
        ]
        if not group or _median(group, metric) > threshold:
            return False
    return True


def _all_strata_ratio_at_most(
    rows: Sequence[Mapping[str, Any]], metric: str, threshold: float
) -> bool:
    for key in ("covariate_shape", "loading_stratum", "prevalence_stratum"):
        for value in {row[key] for row in rows}:
            group = [row for row in rows if row[key] == value]
            if _median(group, metric) > threshold:
                return False
    return True


def _score_ratio(
    candidate_rows: Sequence[Mapping[str, Any]],
    comparator: Mapping[int, Mapping[str, Any]],
    metric: str,
) -> float:
    numerator = np.mean([row[metric] for row in candidate_rows])
    denominator = np.mean(
        [comparator[int(row["seed"])][metric] for row in candidate_rows]
    )
    return _ratio(numerator, denominator)


def _comparator_score_ratio(
    candidate: Mapping[int, Mapping[str, Any]],
    comparator: Mapping[int, Mapping[str, Any]],
    metric: str,
) -> float:
    seeds = sorted(comparator)
    return _ratio(
        np.mean([candidate[seed][metric] for seed in seeds]),
        np.mean([comparator[seed][metric] for seed in seeds]),
    )


def _candidate_python_association_correlation(
    candidate: Mapping[int, Mapping[str, Any]],
    python: Mapping[int, Mapping[str, Any]],
) -> float:
    candidate_values = []
    python_values = []
    for seed in sorted(python):
        candidate_values.append(candidate[seed]["association_vector_mean"])
        python_values.append(python[seed]["association_vector_mean"])
    return _safe_correlation(
        np.concatenate(candidate_values),
        np.concatenate(python_values),
    )


def _median(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return float(np.median([row[metric] for row in rows]))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), np.finfo(float).eps)


def _between(value: float, lower: float, upper: float) -> bool:
    return bool(lower <= float(value) <= upper)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        raise ValueError("correlation vectors have incompatible shapes")
    if np.std(a) <= np.finfo(float).eps or np.std(b) <= np.finfo(float).eps:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
