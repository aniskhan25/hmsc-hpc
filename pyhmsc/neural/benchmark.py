"""Reference benchmark helpers for Neural-HMSC posterior approximations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr

from pyhmsc.formulas import build_design_matrix
from pyhmsc.posterior import HmscFit


DEFAULT_CREDIBLE_LEVELS = (0.8, 0.9, 0.95)


@dataclass(frozen=True)
class BenchmarkReportPaths:
    """Paths written by a Neural-HMSC benchmark report."""

    csv: Path
    markdown: Path


@dataclass(frozen=True)
class SbcReportPaths:
    """Paths written by an SBC and OOD diagnostic report."""

    csv: Path
    markdown: Path
    json: Path


def poisson_predictive_acceptance(
    uncalibrated_row: dict[str, Any],
    calibrated_row: dict[str, Any],
    *,
    max_calibrated_rmse_ratio: float = 1.25,
    max_mcmc_rmse_ratio: float = 2.0,
    max_clipped_fraction: float = 0.01,
) -> dict[str, float | bool]:
    """Evaluate Poisson calibration against neural and MCMC baselines."""
    uncalibrated_rmse = float(uncalibrated_row["neural_posterior_predictive_mean_rmse"])
    calibrated_rmse = float(calibrated_row["neural_posterior_predictive_mean_rmse"])
    mcmc_rmse = float(calibrated_row["mcmc_posterior_predictive_mean_rmse"])
    clipped_fraction = float(calibrated_row["neural_poisson_eta_clipped_fraction"])
    epsilon = np.finfo(float).eps
    calibration_ratio = calibrated_rmse / max(uncalibrated_rmse, epsilon)
    mcmc_ratio = calibrated_rmse / max(mcmc_rmse, epsilon)
    return {
        "predictive_rmse_ratio_vs_uncalibrated": calibration_ratio,
        "predictive_rmse_ratio_vs_mcmc": mcmc_ratio,
        "predictive_acceptance_passed": bool(
            calibration_ratio <= max_calibrated_rmse_ratio
            and mcmc_ratio <= max_mcmc_rmse_ratio
            and clipped_fraction <= max_clipped_fraction
        ),
    }


def load_truth_beta(path: str | Path) -> pd.DataFrame:
    """Load a benchmark ``truth_beta.csv`` file."""
    return pd.read_csv(path, index_col=0)


def compare_beta_posteriors(
    neural_fit: HmscFit,
    mcmc_fit: HmscFit,
    *,
    truth_beta: pd.DataFrame | np.ndarray | None = None,
    dataset: str = "dataset",
    distribution: str | None = None,
    credible_levels: Sequence[float] = DEFAULT_CREDIBLE_LEVELS,
    neural_seconds: float | None = None,
    mcmc_seconds: float | None = None,
    X: pd.DataFrame | np.ndarray | None = None,
    Y: pd.DataFrame | np.ndarray | None = None,
    formula: str | None = None,
    poisson_eta_clip: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compare a neural ``Beta`` posterior against an MCMC reference posterior.

    The comparison intentionally uses posterior sample files through
    :class:`pyhmsc.posterior.HmscFit` so neural and MCMC outputs share the same
    storage contract.
    """
    _validate_levels(credible_levels)
    neural_samples = np.asarray(neural_fit.beta_samples(), dtype=float)
    mcmc_samples = np.asarray(mcmc_fit.beta_samples(), dtype=float)
    if neural_samples.shape[2:] != mcmc_samples.shape[2:]:
        raise ValueError(
            "neural and MCMC Beta samples must share covariate/species shape; "
            f"got {neural_samples.shape[2:]} and {mcmc_samples.shape[2:]}"
        )

    distribution = distribution or _fit_distribution(neural_fit) or _fit_distribution(mcmc_fit)
    row: dict[str, Any] = {
        "dataset": dataset,
        "distribution": distribution,
        "n_covariates": int(neural_samples.shape[2]),
        "n_species": int(neural_samples.shape[3]),
        "neural_chains": int(neural_samples.shape[0]),
        "neural_draws": int(neural_samples.shape[1]),
        "mcmc_chains": int(mcmc_samples.shape[0]),
        "mcmc_draws": int(mcmc_samples.shape[1]),
    }
    row.update(_calibration_report_fields(neural_fit, prefix="neural"))

    neural_mean = neural_samples.mean(axis=(0, 1))
    mcmc_mean = mcmc_samples.mean(axis=(0, 1))
    neural_sd = _posterior_sd(neural_samples)
    mcmc_sd = _posterior_sd(mcmc_samples)
    row.update(
        {
            "beta_mean_rmse_mcmc": _rmse(neural_mean, mcmc_mean),
            "beta_mean_mae_mcmc": _mae(neural_mean, mcmc_mean),
            "beta_sd_rmse_mcmc": _rmse(neural_sd, mcmc_sd),
            "posterior_mean_correlation": _correlation(neural_mean, mcmc_mean),
        }
    )

    for level in credible_levels:
        suffix = _level_suffix(level)
        n_lo, n_hi = _interval_arrays(neural_samples, level)
        m_lo, m_hi = _interval_arrays(mcmc_samples, level)
        row[f"beta_ci_overlap_{suffix}"] = _mean_interval_overlap(n_lo, n_hi, m_lo, m_hi)
        row[f"neural_beta_interval_width_mean_{suffix}"] = float(np.mean(n_hi - n_lo))
        row[f"mcmc_beta_interval_width_mean_{suffix}"] = float(np.mean(m_hi - m_lo))

    if truth_beta is not None:
        truth = _align_truth_beta(truth_beta, neural_fit, neural_samples.shape[2:])
        row.update(_truth_metrics("neural", neural_samples, truth, credible_levels))
        row.update(_truth_metrics("mcmc", mcmc_samples, truth, credible_levels))

    if neural_seconds is not None:
        row["neural_inference_wall_time_seconds"] = float(neural_seconds)
    if mcmc_seconds is not None:
        row["mcmc_wall_time_seconds"] = float(mcmc_seconds)
    if neural_seconds is not None and mcmc_seconds is not None and neural_seconds > 0:
        row["speedup_factor"] = float(mcmc_seconds / neural_seconds)

    if X is not None and Y is not None:
        predictive_formula = formula or _fit_formula(neural_fit) or _fit_formula(mcmc_fit)
        if predictive_formula is None:
            raise ValueError("formula is required for predictive benchmark metrics")
        predictive_distribution = distribution or "normal"
        predictive_eta_clip = _validate_poisson_eta_clip(
            poisson_eta_clip,
            distribution=predictive_distribution,
        )
        if predictive_eta_clip is not None:
            row["predictive_poisson_eta_clip_lower"] = predictive_eta_clip[0]
            row["predictive_poisson_eta_clip_upper"] = predictive_eta_clip[1]
        row.update(
            _predictive_metrics(
                "neural",
                neural_samples,
                X=X,
                Y=Y,
                formula=predictive_formula,
                distribution=predictive_distribution,
                poisson_eta_clip=predictive_eta_clip,
            )
        )
        row.update(
            _predictive_metrics(
                "mcmc",
                mcmc_samples,
                X=X,
                Y=Y,
                formula=predictive_formula,
                distribution=predictive_distribution,
                poisson_eta_clip=predictive_eta_clip,
            )
        )

    return row


def compare_beta_posterior_files(
    *,
    neural_posterior: str | Path,
    mcmc_posterior: str | Path,
    truth_beta: str | Path | pd.DataFrame | np.ndarray | None = None,
    dataset: str = "dataset",
    distribution: str | None = None,
    credible_levels: Sequence[float] = DEFAULT_CREDIBLE_LEVELS,
    neural_seconds: float | None = None,
    mcmc_seconds: float | None = None,
    X: str | Path | pd.DataFrame | np.ndarray | None = None,
    Y: str | Path | pd.DataFrame | np.ndarray | None = None,
    formula: str | None = None,
    poisson_eta_clip: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Load posterior files and return one benchmark metric row."""
    truth = _read_optional_frame(truth_beta)
    x_frame = _read_optional_frame(X)
    y_frame = _read_optional_frame(Y)
    return compare_beta_posteriors(
        HmscFit.from_file(neural_posterior),
        HmscFit.from_file(mcmc_posterior),
        truth_beta=truth,
        dataset=dataset,
        distribution=distribution,
        credible_levels=credible_levels,
        neural_seconds=neural_seconds,
        mcmc_seconds=mcmc_seconds,
        X=x_frame,
        Y=y_frame,
        formula=formula,
        poisson_eta_clip=poisson_eta_clip,
    )


def compare_gamma_posteriors(
    neural_fit: HmscFit,
    mcmc_fit: HmscFit,
    *,
    truth_gamma: pd.DataFrame | np.ndarray | None = None,
    dataset: str = "dataset",
    distribution: str | None = None,
    credible_levels: Sequence[float] = DEFAULT_CREDIBLE_LEVELS,
) -> dict[str, Any]:
    """Compare a neural ``Gamma`` posterior against an MCMC reference."""
    _validate_levels(credible_levels)
    neural_samples = np.asarray(neural_fit.gamma_samples(), dtype=float)
    mcmc_samples = np.asarray(mcmc_fit.gamma_samples(), dtype=float)
    if neural_samples.shape[2:] != mcmc_samples.shape[2:]:
        raise ValueError(
            "neural and MCMC Gamma samples must share covariate/trait shape; "
            f"got {neural_samples.shape[2:]} and {mcmc_samples.shape[2:]}"
        )
    distribution = distribution or _fit_distribution(neural_fit) or _fit_distribution(mcmc_fit)
    neural_mean = neural_samples.mean(axis=(0, 1))
    mcmc_mean = mcmc_samples.mean(axis=(0, 1))
    neural_sd = _posterior_sd(neural_samples)
    mcmc_sd = _posterior_sd(mcmc_samples)
    row: dict[str, Any] = {
        "dataset": dataset,
        "parameter": "Gamma",
        "distribution": distribution,
        "n_covariates": int(neural_samples.shape[2]),
        "n_traits": int(neural_samples.shape[3]),
        "neural_chains": int(neural_samples.shape[0]),
        "neural_draws": int(neural_samples.shape[1]),
        "mcmc_chains": int(mcmc_samples.shape[0]),
        "mcmc_draws": int(mcmc_samples.shape[1]),
        "gamma_mean_rmse_mcmc": _rmse(neural_mean, mcmc_mean),
        "gamma_mean_mae_mcmc": _mae(neural_mean, mcmc_mean),
        "gamma_sd_rmse_mcmc": _rmse(neural_sd, mcmc_sd),
        "gamma_posterior_mean_correlation": _correlation(neural_mean, mcmc_mean),
    }
    for level in credible_levels:
        suffix = _level_suffix(level)
        n_lo, n_hi = _interval_arrays(neural_samples, level)
        m_lo, m_hi = _interval_arrays(mcmc_samples, level)
        row[f"gamma_ci_overlap_{suffix}"] = _mean_interval_overlap(n_lo, n_hi, m_lo, m_hi)
        row[f"neural_gamma_interval_width_mean_{suffix}"] = float(np.mean(n_hi - n_lo))
        row[f"mcmc_gamma_interval_width_mean_{suffix}"] = float(np.mean(m_hi - m_lo))
    if truth_gamma is not None:
        truth = _align_truth_gamma(truth_gamma, neural_fit, neural_samples.shape[2:])
        row.update(_gamma_truth_metrics("neural", neural_samples, truth, credible_levels))
        row.update(_gamma_truth_metrics("mcmc", mcmc_samples, truth, credible_levels))
    return row


def compare_gamma_posterior_files(
    *,
    neural_posterior: str | Path,
    mcmc_posterior: str | Path,
    truth_gamma: str | Path | pd.DataFrame | np.ndarray | None = None,
    dataset: str = "dataset",
    distribution: str | None = None,
    credible_levels: Sequence[float] = DEFAULT_CREDIBLE_LEVELS,
) -> dict[str, Any]:
    """Load posterior files and return one Gamma benchmark metric row."""
    truth = _read_optional_frame(truth_gamma)
    return compare_gamma_posteriors(
        HmscFit.from_file(neural_posterior),
        HmscFit.from_file(mcmc_posterior),
        truth_gamma=truth,
        dataset=dataset,
        distribution=distribution,
        credible_levels=credible_levels,
    )


def compare_iid_association_posteriors(
    neural_fit: HmscFit,
    mcmc_fit: HmscFit,
    *,
    truth_lambda: pd.DataFrame | np.ndarray | None = None,
    dataset: str = "dataset",
    level: int = 0,
) -> dict[str, Any]:
    """Compare identifiable iid species associations from Lambda samples."""
    neural_samples = neural_fit.species_association_samples(level=level, correlation=False)
    mcmc_samples = mcmc_fit.species_association_samples(level=level, correlation=False)
    if neural_samples.shape[2:] != mcmc_samples.shape[2:]:
        raise ValueError(
            "neural and MCMC association samples must share species shape; "
            f"got {neural_samples.shape[2:]} and {mcmc_samples.shape[2:]}"
        )
    neural_mean = neural_samples.mean(axis=(0, 1))
    mcmc_mean = mcmc_samples.mean(axis=(0, 1))
    row: dict[str, Any] = {
        "dataset": dataset,
        "parameter": "Associations",
        "random_level": int(level),
        "n_species": int(neural_samples.shape[-1]),
        "neural_chains": int(neural_samples.shape[0]),
        "neural_draws": int(neural_samples.shape[1]),
        "mcmc_chains": int(mcmc_samples.shape[0]),
        "mcmc_draws": int(mcmc_samples.shape[1]),
        "association_rmse_mcmc": _rmse(neural_mean, mcmc_mean),
        "association_mae_mcmc": _mae(neural_mean, mcmc_mean),
        "association_correlation_mcmc": _correlation(neural_mean, mcmc_mean),
    }
    if truth_lambda is not None:
        truth_loadings = truth_lambda.to_numpy(dtype=float) if isinstance(truth_lambda, pd.DataFrame) else np.asarray(truth_lambda, dtype=float)
        truth_association = truth_loadings.T @ truth_loadings
        if truth_association.shape != neural_mean.shape:
            raise ValueError(
                f"truth association shape {truth_association.shape} does not match posterior shape {neural_mean.shape}"
            )
        row["neural_association_rmse_truth"] = _rmse(neural_mean, truth_association)
        row["mcmc_association_rmse_truth"] = _rmse(mcmc_mean, truth_association)
        row["neural_association_correlation_truth"] = _correlation(neural_mean, truth_association)
        row["mcmc_association_correlation_truth"] = _correlation(mcmc_mean, truth_association)
    return row


def compare_iid_association_posterior_files(
    *,
    neural_posterior: str | Path,
    mcmc_posterior: str | Path,
    truth_lambda: str | Path | pd.DataFrame | np.ndarray | None = None,
    dataset: str = "dataset",
    level: int = 0,
) -> dict[str, Any]:
    """Load posterior files and compare iid association summaries."""
    truth = _read_optional_frame(truth_lambda)
    return compare_iid_association_posteriors(
        HmscFit.from_file(neural_posterior),
        HmscFit.from_file(mcmc_posterior),
        truth_lambda=truth,
        dataset=dataset,
        level=level,
    )


def write_benchmark_report(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    stem: str = "neural_hmsc_mcmc_reference",
    title: str = "Neural-HMSC MCMC Reference Benchmark",
) -> BenchmarkReportPaths:
    """Write benchmark rows as CSV and Markdown summary files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        raise ValueError("rows must contain at least one benchmark result")
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(render_benchmark_markdown(frame, title=title), encoding="utf-8")
    return BenchmarkReportPaths(csv=csv_path, markdown=markdown_path)


def write_sbc_report(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    stem: str = "neural_hmsc_sbc_diagnostics",
) -> SbcReportPaths:
    """Write simulation-based calibration and OOD diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    if not records:
        raise ValueError("rows must contain at least one SBC result")
    frame = pd.DataFrame(records)
    csv_frame = frame.copy()
    for column in csv_frame.columns:
        if csv_frame[column].map(lambda value: isinstance(value, (list, tuple, dict))).any():
            csv_frame[column] = csv_frame[column].map(
                lambda value: json.dumps(value) if isinstance(value, (list, tuple, dict)) else value
            )
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    csv_frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sbc_markdown(frame), encoding="utf-8")
    return SbcReportPaths(csv=csv_path, markdown=markdown_path, json=json_path)


def render_sbc_markdown(rows: pd.DataFrame | Sequence[dict[str, Any]]) -> str:
    """Render a compact SBC and OOD summary."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty:
        raise ValueError("rows must contain at least one SBC result")
    preferred = [
        "distribution",
        "simulation_domain",
        "ood_regime",
        "posterior_variant",
        "sbc_n_replicates",
        "sbc_n_draws",
        "sbc_rank_mean",
        "sbc_rank_variance",
        "sbc_expected_rank_variance",
        "sbc_lower_tail_fraction",
        "sbc_upper_tail_fraction",
        "sbc_max_abs_coefficient_rank_mean_deviation",
        "sbc_chi_square_pvalue",
        "sbc_beta_mean_rmse",
        "sbc_beta_interval_coverage_95",
        "ood_rmse_ratio_vs_in_distribution",
    ]
    columns = [column for column in preferred if column in frame.columns]
    summary = frame.loc[:, columns].copy()
    for column in summary.columns:
        if pd.api.types.is_float_dtype(summary[column]):
            summary[column] = summary[column].map(
                lambda value: f"{value:.4g}" if pd.notna(value) else ""
            )
    lines = [
        "# Neural-HMSC Simulation-Based Calibration Diagnostics",
        "",
        _markdown_table(summary),
        "",
        "## Metric Notes",
        "",
        "- Uniform SBC ranks have mean 0.5 and variance close to the reported discrete-rank expectation.",
        "- Tail imbalance and low chi-square p-values flag posterior bias or dispersion mismatch; they are diagnostics, not independent hypothesis tests.",
        "- OOD rows use the same fitted amortizer and calibration as the in-distribution row.",
        "- `ood_rmse_ratio_vs_in_distribution` measures posterior-mean degradation relative to the matching posterior variant.",
        "",
    ]
    return "\n".join(lines)


def render_benchmark_markdown(
    rows: pd.DataFrame | Sequence[dict[str, Any]],
    *,
    title: str = "Neural-HMSC MCMC Reference Benchmark",
) -> str:
    """Render a compact Markdown benchmark report."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty:
        raise ValueError("rows must contain at least one benchmark result")
    preferred = [
        "dataset",
        "posterior_variant",
        "distribution",
        "beta_mean_rmse_mcmc",
        "beta_sd_rmse_mcmc",
        "beta_ci_overlap_95",
        "gamma_mean_rmse_mcmc",
        "gamma_sd_rmse_mcmc",
        "gamma_ci_overlap_95",
        "association_rmse_mcmc",
        "association_correlation_mcmc",
        "neural_association_rmse_truth",
        "neural_association_correlation_truth",
        "neural_calibration_method",
        "neural_calibration_scale_multiplier",
        "neural_calibration_coverage_scale_multiplier",
        "neural_calibration_uncalibrated_coverage",
        "neural_calibration_calibrated_coverage",
        "neural_beta_mean_rmse_truth",
        "mcmc_beta_mean_rmse_truth",
        "predictive_poisson_eta_clip_lower",
        "predictive_poisson_eta_clip_upper",
        "neural_posterior_predictive_mean_rmse",
        "mcmc_posterior_predictive_mean_rmse",
        "predictive_rmse_ratio_vs_uncalibrated",
        "predictive_rmse_ratio_vs_mcmc",
        "predictive_acceptance_passed",
        "speedup_factor",
    ]
    columns = [column for column in preferred if column in frame.columns]
    summary = frame.loc[:, columns].copy()
    for column in summary.columns:
        if pd.api.types.is_float_dtype(summary[column]):
            summary[column] = summary[column].map(lambda value: f"{value:.4g}" if pd.notna(value) else "")

    lines = [
        f"# {title}",
        "",
        "This report compares Neural-HMSC `Beta` posterior samples against an MCMC reference posterior.",
        "",
        _markdown_table(summary),
        "",
        "## Metric Notes",
        "",
        "- `beta_mean_rmse_mcmc`: RMSE between neural and MCMC posterior means.",
        "- `beta_sd_rmse_mcmc`: RMSE between neural and MCMC posterior standard deviations.",
        "- `beta_ci_overlap_*`: mean interval overlap coefficient between neural and MCMC credible intervals.",
        "- `*_truth` metrics compare posterior summaries to simulated `truth_beta` when available.",
        "- predictive RMSE uses posterior mean response predictions from fixed-effect `Beta` samples.",
        "- Poisson predictive metrics report explicit eta bounds when the declared benchmark model uses them.",
        "- `predictive_acceptance_passed` requires calibrated RMSE <= 1.25x uncalibrated RMSE, <= 2x MCMC RMSE, and <1% clipped eta draws.",
        "",
    ]
    return "\n".join(lines)


def _truth_metrics(
    prefix: str,
    samples: np.ndarray,
    truth: np.ndarray,
    credible_levels: Sequence[float],
) -> dict[str, float]:
    mean = samples.mean(axis=(0, 1))
    metrics = {
        f"{prefix}_beta_mean_rmse_truth": _rmse(mean, truth),
        f"{prefix}_beta_mean_mae_truth": _mae(mean, truth),
    }
    for level in credible_levels:
        suffix = _level_suffix(level)
        lo, hi = _interval_arrays(samples, level)
        metrics[f"{prefix}_beta_interval_coverage_truth_{suffix}"] = float(np.mean((truth >= lo) & (truth <= hi)))
    return metrics


def _gamma_truth_metrics(
    prefix: str,
    samples: np.ndarray,
    truth: np.ndarray,
    credible_levels: Sequence[float],
) -> dict[str, float]:
    mean = samples.mean(axis=(0, 1))
    metrics = {
        f"{prefix}_gamma_mean_rmse_truth": _rmse(mean, truth),
        f"{prefix}_gamma_mean_mae_truth": _mae(mean, truth),
    }
    for level in credible_levels:
        suffix = _level_suffix(level)
        lo, hi = _interval_arrays(samples, level)
        metrics[f"{prefix}_gamma_interval_coverage_truth_{suffix}"] = float(np.mean((truth >= lo) & (truth <= hi)))
    return metrics


def _predictive_metrics(
    prefix: str,
    beta_samples: np.ndarray,
    *,
    X: pd.DataFrame | np.ndarray,
    Y: pd.DataFrame | np.ndarray,
    formula: str,
    distribution: str,
    poisson_eta_clip: tuple[float, float] | None = None,
) -> dict[str, float]:
    x_frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_frame = Y if isinstance(Y, pd.DataFrame) else pd.DataFrame(Y)
    design = build_design_matrix(formula, x_frame)
    if design.shape[1] != beta_samples.shape[2]:
        raise ValueError(
            "prediction design does not match Beta covariate dimension; "
            f"got {design.shape[1]} and {beta_samples.shape[2]}"
        )
    linear = np.einsum("nk,cdks->cdns", design.to_numpy(dtype=float), beta_samples)
    response = _response_scale(linear, distribution, poisson_eta_clip=poisson_eta_clip)
    prediction = response.mean(axis=(0, 1))
    observed = y_frame.to_numpy(dtype=float)
    if prediction.shape != observed.shape:
        raise ValueError(f"prediction and observed Y shapes differ: {prediction.shape} vs {observed.shape}")
    species_prediction = response.mean(axis=2).reshape(-1, response.shape[-1])
    observed_species = observed.mean(axis=0)
    lo = np.quantile(species_prediction, 0.025, axis=0)
    hi = np.quantile(species_prediction, 0.975, axis=0)
    metrics = {
        f"{prefix}_posterior_predictive_mean_rmse": _rmse(prediction, observed),
        f"{prefix}_species_mean_coverage_95": float(np.mean((observed_species >= lo) & (observed_species <= hi))),
    }
    if str(distribution).lower() == "poisson" and poisson_eta_clip is not None:
        lower, upper = poisson_eta_clip
        metrics.update(
            {
                f"{prefix}_poisson_eta_below_clip_fraction": float(np.mean(linear < lower)),
                f"{prefix}_poisson_eta_above_clip_fraction": float(np.mean(linear > upper)),
                f"{prefix}_poisson_eta_clipped_fraction": float(np.mean((linear < lower) | (linear > upper))),
            }
        )
    return metrics


def _read_optional_frame(value: str | Path | pd.DataFrame | np.ndarray | None) -> pd.DataFrame | np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (pd.DataFrame, np.ndarray)):
        return value
    return pd.read_csv(value, index_col=0)


def _align_truth_beta(
    truth_beta: pd.DataFrame | np.ndarray,
    fit: HmscFit,
    shape: tuple[int, int],
) -> np.ndarray:
    if isinstance(truth_beta, pd.DataFrame):
        beta_mean = fit.beta_mean()
        if set(beta_mean.index).issubset(set(truth_beta.index)) and set(beta_mean.columns).issubset(set(truth_beta.columns)):
            truth = truth_beta.loc[beta_mean.index, beta_mean.columns].to_numpy(dtype=float)
        else:
            truth = truth_beta.to_numpy(dtype=float)
    else:
        truth = np.asarray(truth_beta, dtype=float)
    if truth.shape != shape:
        raise ValueError(f"truth_beta shape {truth.shape} does not match Beta shape {shape}")
    return truth


def _align_truth_gamma(
    truth_gamma: pd.DataFrame | np.ndarray,
    fit: HmscFit,
    shape: tuple[int, int],
) -> np.ndarray:
    if isinstance(truth_gamma, pd.DataFrame):
        gamma_mean = fit.gamma_mean()
        if set(gamma_mean.index).issubset(set(truth_gamma.index)) and set(gamma_mean.columns).issubset(set(truth_gamma.columns)):
            truth = truth_gamma.loc[gamma_mean.index, gamma_mean.columns].to_numpy(dtype=float)
        else:
            truth = truth_gamma.to_numpy(dtype=float)
    else:
        truth = np.asarray(truth_gamma, dtype=float)
    if truth.shape != shape:
        raise ValueError(f"truth_gamma shape {truth.shape} does not match Gamma shape {shape}")
    return truth


def _posterior_sd(samples: np.ndarray) -> np.ndarray:
    return samples.reshape((-1,) + samples.shape[2:]).std(axis=0, ddof=1)


def _interval_arrays(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - level) / 2.0
    return (
        np.quantile(samples, alpha, axis=(0, 1)),
        np.quantile(samples, 1.0 - alpha, axis=(0, 1)),
    )


def _mean_interval_overlap(
    left_lo: np.ndarray,
    left_hi: np.ndarray,
    right_lo: np.ndarray,
    right_hi: np.ndarray,
) -> float:
    intersection = np.maximum(0.0, np.minimum(left_hi, right_hi) - np.maximum(left_lo, right_lo))
    union = np.maximum(left_hi, right_hi) - np.minimum(left_lo, right_lo)
    exact = (union == 0.0) & (left_lo == right_lo) & (left_hi == right_hi)
    overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0.0)
    overlap = np.where(exact, 1.0, overlap)
    return float(np.mean(overlap))


def _response_scale(
    linear: np.ndarray,
    distribution: str,
    *,
    poisson_eta_clip: tuple[float, float] | None = None,
) -> np.ndarray:
    key = str(distribution).lower()
    if key in {"normal", "gaussian"}:
        return linear
    if key == "poisson":
        if poisson_eta_clip is None:
            with np.errstate(over="raise", invalid="raise"):
                try:
                    return np.exp(linear)
                except FloatingPointError as exc:
                    raise ValueError(
                        "Poisson response predictions overflowed; pass the eta bounds "
                        "declared by the benchmark model via poisson_eta_clip"
                    ) from exc
        return np.exp(np.clip(linear, poisson_eta_clip[0], poisson_eta_clip[1]))
    if key in {"probit", "bernoulli", "binomial"}:
        return ndtr(linear)
    raise ValueError(f"Unsupported predictive distribution {distribution!r}")


def _validate_poisson_eta_clip(
    value: tuple[float, float] | None,
    *,
    distribution: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if str(distribution).lower() != "poisson":
        raise ValueError("poisson_eta_clip is only valid for Poisson predictive metrics")
    if len(value) != 2:
        raise ValueError("poisson_eta_clip must contain exactly two bounds")
    lower, upper = float(value[0]), float(value[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("poisson_eta_clip must contain finite, ordered bounds")
    return lower, upper


def _fit_distribution(fit: HmscFit) -> str | None:
    metadata = fit.metadata if isinstance(fit.metadata, dict) else {}
    distribution = metadata.get("distribution")
    return str(distribution) if distribution is not None else None


def _fit_formula(fit: HmscFit) -> str | None:
    metadata = fit.metadata if isinstance(fit.metadata, dict) else {}
    formula = metadata.get("formula")
    if isinstance(formula, dict) and formula.get("X"):
        return str(formula["X"])
    return None


def _calibration_report_fields(fit: HmscFit, *, prefix: str) -> dict[str, Any]:
    metadata = fit.metadata if isinstance(fit.metadata, dict) else {}
    calibration = metadata.get("calibration")
    if not isinstance(calibration, dict):
        return {}
    fields: dict[str, Any] = {}
    for source, target in [
        ("method", "method"),
        ("scale_multiplier", "scale_multiplier"),
        ("coverage_scale_multiplier", "coverage_scale_multiplier"),
        ("nominal_level", "nominal_level"),
        ("uncalibrated_coverage", "uncalibrated_coverage"),
        ("calibrated_coverage", "calibrated_coverage"),
        ("n_observations", "n_observations"),
        ("predictive_score_uncalibrated", "predictive_score_uncalibrated"),
        ("predictive_score_calibrated", "predictive_score_calibrated"),
        ("predictive_rate_rmse_uncalibrated", "predictive_rate_rmse_uncalibrated"),
        ("predictive_rate_rmse_calibrated", "predictive_rate_rmse_calibrated"),
    ]:
        if source in calibration:
            fields[f"{prefix}_calibration_{target}"] = calibration[source]
    domain = calibration.get("domain")
    if isinstance(domain, dict):
        for source in ["distribution", "n_covariates", "n_species"]:
            if source in domain:
                fields[f"{prefix}_calibration_domain_{source}"] = domain[source]
    return fields


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(left, dtype=float) - np.asarray(right, dtype=float)) ** 2)))


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=float).ravel()
    right_flat = np.asarray(right, dtype=float).ravel()
    if left_flat.size < 2 or np.std(left_flat) == 0.0 or np.std(right_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def _validate_levels(levels: Sequence[float]) -> None:
    if not levels:
        raise ValueError("credible_levels must not be empty")
    for level in levels:
        if not 0.0 < float(level) < 1.0:
            raise ValueError("credible levels must be between 0 and 1")


def _level_suffix(level: float) -> str:
    percent = level * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return str(int(round(percent)))
    return f"{percent:g}".replace(".", "_")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])
