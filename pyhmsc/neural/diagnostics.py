"""Simulation-based calibration diagnostics for Neural-HMSC posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
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


def beta_sbc_rank_diagnostics(
    posterior_samples: np.ndarray,
    beta_true: np.ndarray,
    *,
    n_bins: int = 10,
    seed: int = 123,
    credible_levels: Sequence[float] = (0.8, 0.9, 0.95),
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

    expanded_truth = truth[:, None, :, :]
    less = np.sum(samples < expanded_truth, axis=1)
    ties = np.sum(samples == expanded_truth, axis=1)
    rng = np.random.default_rng(seed)
    tie_offsets = np.floor(rng.random(size=ties.shape) * (ties + 1)).astype(int)
    ranks = less + tie_offsets

    n_draws = int(samples.shape[1])
    rank_fraction = (ranks.astype(float) + 0.5) / float(n_draws + 1)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    counts, _ = np.histogram(rank_fraction, bins=edges)
    possible_ranks = (np.arange(n_draws + 1, dtype=float) + 0.5) / float(n_draws + 1)
    possible_counts, _ = np.histogram(possible_ranks, bins=edges)
    expected_counts = rank_fraction.size * possible_counts / float(n_draws + 1)
    nonempty = expected_counts > 0.0
    chi_square = chisquare(counts[nonempty], f_exp=expected_counts[nonempty])

    coefficient_means = np.mean(rank_fraction, axis=0)
    expected_variance = n_draws * (n_draws + 2) / (12.0 * (n_draws + 1) ** 2)
    posterior_mean = np.mean(samples, axis=1)
    coverage = {}
    for level in levels:
        tail = (1.0 - level) / 2.0
        lower = np.quantile(samples, tail, axis=1)
        upper = np.quantile(samples, 1.0 - tail, axis=1)
        coverage[level] = float(np.mean((truth >= lower) & (truth <= upper)))

    return BetaSbcDiagnostics(
        n_replicates=int(samples.shape[0]),
        n_draws=n_draws,
        n_coefficients=int(samples.shape[2] * samples.shape[3]),
        n_ranks=int(rank_fraction.size),
        n_bins=int(n_bins),
        histogram_counts=tuple(int(value) for value in counts),
        histogram_expected_counts=tuple(float(value) for value in expected_counts),
        histogram_edges=tuple(float(value) for value in edges),
        rank_mean=float(np.mean(rank_fraction)),
        rank_variance=float(np.var(rank_fraction)),
        expected_rank_mean=0.5,
        expected_rank_variance=float(expected_variance),
        lower_tail_fraction=float(np.mean(rank_fraction <= 0.1)),
        upper_tail_fraction=float(np.mean(rank_fraction >= 0.9)),
        max_abs_coefficient_rank_mean_deviation=float(
            np.max(np.abs(coefficient_means - 0.5))
        ),
        chi_square_statistic=float(chi_square.statistic),
        chi_square_pvalue=float(chi_square.pvalue),
        beta_mean_rmse=float(np.sqrt(np.mean(np.square(posterior_mean - truth)))),
        beta_interval_coverage_80=coverage[0.8],
        beta_interval_coverage_90=coverage[0.9],
        beta_interval_coverage_95=coverage[0.95],
    )
