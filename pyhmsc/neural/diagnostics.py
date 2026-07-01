"""Simulation-based calibration diagnostics for Neural-HMSC posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtr
from scipy.stats import chisquare


@dataclass(frozen=True)
class BetaSbcDiagnostics:
    """Rank and posterior-accuracy summaries across simulated datasets."""

    n_replicates: int
    n_draws: int
    n_coefficients: int
    n_ranks: int
    n_bins: int
    histogram_counts: tuple[int, ...]
    histogram_expected_counts: tuple[float, ...]
    histogram_edges: tuple[float, ...]
    rank_mean: float
    rank_variance: float
    expected_rank_mean: float
    expected_rank_variance: float
    lower_tail_fraction: float
    upper_tail_fraction: float
    max_abs_coefficient_rank_mean_deviation: float
    chi_square_statistic: float
    chi_square_pvalue: float
    beta_mean_rmse: float
    beta_interval_coverage_80: float
    beta_interval_coverage_90: float
    beta_interval_coverage_95: float

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "parameter": "Beta",
            "n_replicates": self.n_replicates,
            "n_draws": self.n_draws,
            "n_coefficients": self.n_coefficients,
            "n_ranks": self.n_ranks,
            "rank_histogram": {
                "n_bins": self.n_bins,
                "counts": list(self.histogram_counts),
                "expected_counts": list(self.histogram_expected_counts),
                "edges": list(self.histogram_edges),
            },
            "rank_mean": self.rank_mean,
            "rank_variance": self.rank_variance,
            "expected_rank_mean": self.expected_rank_mean,
            "expected_rank_variance": self.expected_rank_variance,
            "lower_tail_fraction": self.lower_tail_fraction,
            "upper_tail_fraction": self.upper_tail_fraction,
            "max_abs_coefficient_rank_mean_deviation": self.max_abs_coefficient_rank_mean_deviation,
            "chi_square_statistic": self.chi_square_statistic,
            "chi_square_pvalue": self.chi_square_pvalue,
            "beta_mean_rmse": self.beta_mean_rmse,
            "beta_interval_coverage_80": self.beta_interval_coverage_80,
            "beta_interval_coverage_90": self.beta_interval_coverage_90,
            "beta_interval_coverage_95": self.beta_interval_coverage_95,
        }

    def report_fields(self, *, prefix: str = "sbc") -> dict[str, Any]:
        """Return flat fields suitable for CSV benchmark reports."""
        return {
            f"{prefix}_n_replicates": self.n_replicates,
            f"{prefix}_n_draws": self.n_draws,
            f"{prefix}_n_coefficients": self.n_coefficients,
            f"{prefix}_n_ranks": self.n_ranks,
            f"{prefix}_n_bins": self.n_bins,
            f"{prefix}_histogram_counts": list(self.histogram_counts),
            f"{prefix}_histogram_expected_counts": list(self.histogram_expected_counts),
            f"{prefix}_rank_mean": self.rank_mean,
            f"{prefix}_rank_variance": self.rank_variance,
            f"{prefix}_expected_rank_mean": self.expected_rank_mean,
            f"{prefix}_expected_rank_variance": self.expected_rank_variance,
            f"{prefix}_lower_tail_fraction": self.lower_tail_fraction,
            f"{prefix}_upper_tail_fraction": self.upper_tail_fraction,
            f"{prefix}_max_abs_coefficient_rank_mean_deviation": (
                self.max_abs_coefficient_rank_mean_deviation
            ),
            f"{prefix}_chi_square_statistic": self.chi_square_statistic,
            f"{prefix}_chi_square_pvalue": self.chi_square_pvalue,
            f"{prefix}_beta_mean_rmse": self.beta_mean_rmse,
            f"{prefix}_beta_interval_coverage_80": self.beta_interval_coverage_80,
            f"{prefix}_beta_interval_coverage_90": self.beta_interval_coverage_90,
            f"{prefix}_beta_interval_coverage_95": self.beta_interval_coverage_95,
        }


@dataclass(frozen=True)
class BetaSbcStratumDiagnostics:
    """SBC diagnostics for one interpretable coefficient subset."""

    kind: str
    label: str
    diagnostics: BetaSbcDiagnostics
    lower_bound: float | None = None
    upper_bound: float | None = None
    value_mean: float | None = None

    def report_fields(self, *, prefix: str = "sbc") -> dict[str, Any]:
        """Return flat stratum metadata plus the standard SBC fields."""
        fields = {
            f"{prefix}_stratum_kind": self.kind,
            f"{prefix}_stratum_label": self.label,
            f"{prefix}_stratum_lower_bound": self.lower_bound,
            f"{prefix}_stratum_upper_bound": self.upper_bound,
            f"{prefix}_stratum_value_mean": self.value_mean,
        }
        fields.update(self.diagnostics.report_fields(prefix=prefix))
        return fields


def beta_sbc_rank_diagnostics(
    posterior_samples: np.ndarray,
    beta_true: np.ndarray,
    *,
    n_bins: int = 10,
    seed: int = 123,
    credible_levels: Sequence[float] = (0.8, 0.9, 0.95),
    coefficient_mask: np.ndarray | None = None,
) -> BetaSbcDiagnostics:
    """Compute randomized SBC ranks for fixed-effect ``Beta`` posteriors.

    ``posterior_samples`` must have shape ``replicates x draws x covariates x
    species``. Ties are broken by uniformly selecting one of their valid rank
    positions, making discrete and degenerate posterior approximations explicit.
    """
    samples = np.asarray(posterior_samples, dtype=float)
    truth = np.asarray(beta_true, dtype=float)
    if samples.ndim != 4:
        raise ValueError(
            "posterior_samples must have shape replicates x draws x covariates x species"
        )
    expected_truth_shape = (samples.shape[0], samples.shape[2], samples.shape[3])
    if truth.shape != expected_truth_shape:
        raise ValueError(
            f"beta_true shape {truth.shape} does not match expected {expected_truth_shape}"
        )
    if samples.shape[0] < 2:
        raise ValueError("SBC diagnostics require at least two simulation replicates")
    if samples.shape[1] < 1:
        raise ValueError("SBC diagnostics require at least one posterior draw")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(truth)):
        raise ValueError("posterior_samples and beta_true must be finite")
    if n_bins < 2 or n_bins > samples.shape[1] + 1:
        raise ValueError("n_bins must be between 2 and n_draws + 1")
    levels = tuple(float(level) for level in credible_levels)
    if levels != (0.8, 0.9, 0.95):
        raise ValueError("credible_levels must be exactly (0.8, 0.9, 0.95)")
    if coefficient_mask is None:
        mask = np.ones(truth.shape, dtype=bool)
    else:
        mask = np.asarray(coefficient_mask, dtype=bool)
        if mask.shape != truth.shape:
            raise ValueError(
                f"coefficient_mask shape {mask.shape} does not match beta_true shape {truth.shape}"
            )
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            "coefficient_mask must select at least two coefficient instances"
        )

    expanded_truth = truth[:, None, :, :]
    less = np.sum(samples < expanded_truth, axis=1)
    ties = np.sum(samples == expanded_truth, axis=1)
    rng = np.random.default_rng(seed)
    tie_offsets = np.floor(rng.random(size=ties.shape) * (ties + 1)).astype(int)
    ranks = less + tie_offsets

    n_draws = int(samples.shape[1])
    rank_fraction = (ranks.astype(float) + 0.5) / float(n_draws + 1)
    selected_ranks = rank_fraction[mask]
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    counts, _ = np.histogram(selected_ranks, bins=edges)
    possible_ranks = (np.arange(n_draws + 1, dtype=float) + 0.5) / float(n_draws + 1)
    possible_counts, _ = np.histogram(possible_ranks, bins=edges)
    expected_counts = selected_ranks.size * possible_counts / float(n_draws + 1)
    nonempty = expected_counts > 0.0
    chi_square = chisquare(counts[nonempty], f_exp=expected_counts[nonempty])

    coefficient_counts = np.sum(mask, axis=0)
    coefficient_sums = np.sum(np.where(mask, rank_fraction, 0.0), axis=0)
    coefficient_means = np.divide(
        coefficient_sums,
        coefficient_counts,
        out=np.full(coefficient_sums.shape, np.nan),
        where=coefficient_counts > 0,
    )
    expected_variance = n_draws * (n_draws + 2) / (12.0 * (n_draws + 1) ** 2)
    posterior_mean = np.mean(samples, axis=1)
    coverage = {}
    for level in levels:
        tail = (1.0 - level) / 2.0
        lower = np.quantile(samples, tail, axis=1)
        upper = np.quantile(samples, 1.0 - tail, axis=1)
        covered = (truth >= lower) & (truth <= upper)
        coverage[level] = float(np.mean(covered[mask]))

    return BetaSbcDiagnostics(
        n_replicates=int(samples.shape[0]),
        n_draws=n_draws,
        n_coefficients=int(np.count_nonzero(coefficient_counts)),
        n_ranks=int(selected_ranks.size),
        n_bins=int(n_bins),
        histogram_counts=tuple(int(value) for value in counts),
        histogram_expected_counts=tuple(float(value) for value in expected_counts),
        histogram_edges=tuple(float(value) for value in edges),
        rank_mean=float(np.mean(selected_ranks)),
        rank_variance=float(np.var(selected_ranks)),
        expected_rank_mean=0.5,
        expected_rank_variance=float(expected_variance),
        lower_tail_fraction=float(np.mean(selected_ranks <= 0.1)),
        upper_tail_fraction=float(np.mean(selected_ranks >= 0.9)),
        max_abs_coefficient_rank_mean_deviation=float(
            np.nanmax(np.abs(coefficient_means - 0.5))
        ),
        chi_square_statistic=float(chi_square.statistic),
        chi_square_pvalue=float(chi_square.pvalue),
        beta_mean_rmse=float(
            np.sqrt(np.mean(np.square((posterior_mean - truth)[mask])))
        ),
        beta_interval_coverage_80=coverage[0.8],
        beta_interval_coverage_90=coverage[0.9],
        beta_interval_coverage_95=coverage[0.95],
    )


def beta_sbc_stratified_diagnostics(
    posterior_samples: np.ndarray,
    beta_true: np.ndarray,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str,
    covariate_names: Sequence[str] | None = None,
    n_bins: int = 10,
    seed: int = 123,
    prevalence_edges: tuple[float, float] = (0.1, 0.3),
    information_quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    min_ranks: int = 2,
) -> list[BetaSbcStratumDiagnostics]:
    """Summarize SBC overall and by prevalence, coefficient, and information.

    Expected information is evaluated at the raw posterior mean, so the same
    feature can later be computed for real datasets without coefficient truth.
    Information quantiles are calculated separately for each coefficient type
    before their masks are combined.
    """
    samples = np.asarray(posterior_samples, dtype=float)
    truth = np.asarray(beta_true, dtype=float)
    design = np.asarray(X, dtype=float)
    response = np.asarray(Y, dtype=float)
    if samples.ndim != 4:
        raise ValueError(
            "posterior_samples must have shape replicates x draws x covariates x species"
        )
    if design.ndim != 3:
        raise ValueError("X must have shape replicates x sites x covariates")
    if response.ndim != 3:
        raise ValueError("Y must have shape replicates x sites x species")
    expected_truth_shape = (samples.shape[0], samples.shape[2], samples.shape[3])
    if truth.shape != expected_truth_shape:
        raise ValueError(
            f"beta_true shape {truth.shape} does not match expected {expected_truth_shape}"
        )
    expected_design_shape = (samples.shape[0], response.shape[1], samples.shape[2])
    if design.shape != expected_design_shape:
        raise ValueError(
            f"X shape {design.shape} does not match expected {expected_design_shape}"
        )
    expected_response_shape = (samples.shape[0], design.shape[1], samples.shape[3])
    if response.shape != expected_response_shape:
        raise ValueError(
            f"Y shape {response.shape} does not match expected {expected_response_shape}"
        )
    if min_ranks < 2:
        raise ValueError("min_ranks must be at least two")
    if not all(
        np.all(np.isfinite(value)) for value in (samples, truth, design, response)
    ):
        raise ValueError("posterior_samples, beta_true, X, and Y must be finite")
    low_prevalence, high_prevalence = (float(value) for value in prevalence_edges)
    if not 0.0 < low_prevalence < high_prevalence < 1.0:
        raise ValueError("prevalence_edges must be ordered values between zero and one")
    low_information, high_information = (
        float(value) for value in information_quantiles
    )
    if not 0.0 < low_information < high_information < 1.0:
        raise ValueError(
            "information_quantiles must be ordered values between zero and one"
        )

    n_covariates = samples.shape[2]
    names = (
        tuple(str(name) for name in covariate_names)
        if covariate_names is not None
        else tuple(f"coefficient_{index}" for index in range(n_covariates))
    )
    if len(names) != n_covariates:
        raise ValueError(
            "covariate_names length must match the posterior covariate dimension"
        )

    specifications: list[
        tuple[str, str, np.ndarray, float | None, float | None, float | None]
    ] = [("overall", "overall", np.ones(truth.shape, dtype=bool), None, None, None)]
    distribution_key = str(distribution).lower()
    if distribution_key in {"probit", "bernoulli", "binomial", "poisson"}:
        prevalence = np.mean(response > 0.0, axis=1)
        prevalence_masks = [
            ("rare", prevalence <= low_prevalence, 0.0, low_prevalence),
            (
                "intermediate",
                (prevalence > low_prevalence) & (prevalence <= high_prevalence),
                low_prevalence,
                high_prevalence,
            ),
            ("common", prevalence > high_prevalence, high_prevalence, 1.0),
        ]
        for label, species_mask, lower, upper in prevalence_masks:
            mask = np.broadcast_to(species_mask[:, None, :], truth.shape)
            specifications.append(
                (
                    "prevalence",
                    label,
                    mask,
                    lower,
                    upper,
                    _masked_mean(prevalence[:, None, :], mask),
                )
            )

    for coefficient_index, name in enumerate(names):
        mask = np.zeros(truth.shape, dtype=bool)
        mask[:, coefficient_index, :] = True
        specifications.append(("coefficient", name, mask, None, None, None))

    information = beta_expected_design_information(
        samples, design, distribution=distribution
    )
    information_masks = {
        "low": np.zeros(truth.shape, dtype=bool),
        "intermediate": np.zeros(truth.shape, dtype=bool),
        "high": np.zeros(truth.shape, dtype=bool),
    }
    for coefficient_index in range(n_covariates):
        values = information[:, coefficient_index, :]
        lower_cut, upper_cut = np.quantile(values, [low_information, high_information])
        information_masks["low"][:, coefficient_index, :] = values <= lower_cut
        information_masks["intermediate"][:, coefficient_index, :] = (
            values > lower_cut
        ) & (values <= upper_cut)
        information_masks["high"][:, coefficient_index, :] = values > upper_cut
    for label, lower, upper in [
        ("low", 0.0, low_information),
        ("intermediate", low_information, high_information),
        ("high", high_information, 1.0),
    ]:
        mask = information_masks[label]
        specifications.append(
            (
                "design_information",
                label,
                mask,
                lower,
                upper,
                _masked_mean(information, mask),
            )
        )

    rows = []
    for kind, label, mask, lower, upper, value_mean in specifications:
        if np.count_nonzero(mask) < min_ranks:
            continue
        diagnostics = beta_sbc_rank_diagnostics(
            samples,
            truth,
            n_bins=n_bins,
            seed=seed,
            coefficient_mask=mask,
        )
        rows.append(
            BetaSbcStratumDiagnostics(
                kind=kind,
                label=label,
                diagnostics=diagnostics,
                lower_bound=lower,
                upper_bound=upper,
                value_mean=value_mean,
            )
        )
    return rows


def beta_expected_design_information(
    posterior_samples: np.ndarray,
    X: np.ndarray,
    *,
    distribution: str,
) -> np.ndarray:
    """Return diagonal expected information at the raw posterior mean."""
    samples = np.asarray(posterior_samples, dtype=float)
    design = np.asarray(X, dtype=float)
    if samples.ndim != 4:
        raise ValueError(
            "posterior_samples must have shape replicates x draws x covariates x species"
        )
    if design.ndim != 3:
        raise ValueError("X must have shape replicates x sites x covariates")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(design)):
        raise ValueError("posterior_samples and X must be finite")
    expected_design_shape = (samples.shape[0], design.shape[1], samples.shape[2])
    if design.shape != expected_design_shape:
        raise ValueError(
            f"X shape {design.shape} does not match posterior dimensions {expected_design_shape}"
        )
    posterior_mean = np.mean(samples, axis=1)
    linear = np.einsum("rnk,rks->rns", design, posterior_mean)
    key = str(distribution).lower()
    if key in {"normal", "gaussian"}:
        weight = np.ones(linear.shape, dtype=float)
    elif key in {"probit", "bernoulli", "binomial"}:
        probability = np.clip(ndtr(linear), 1e-9, 1.0 - 1e-9)
        density = np.exp(-0.5 * np.square(linear)) / np.sqrt(2.0 * np.pi)
        weight = np.square(density) / (probability * (1.0 - probability))
    elif key == "poisson":
        weight = np.exp(np.clip(linear, -20.0, 20.0))
    else:
        raise ValueError(
            f"unsupported distribution for design information: {distribution!r}"
        )
    return np.einsum("rnk,rns->rks", np.square(design), weight)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    broadcast = np.broadcast_to(values, mask.shape)
    return float(np.mean(broadcast[mask]))
