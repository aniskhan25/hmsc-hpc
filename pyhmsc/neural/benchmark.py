"""Reference benchmark helpers for Neural-HMSC posterior approximations."""

from __future__ import annotations

from dataclasses import dataclass
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
        row.update(
            _predictive_metrics(
                "neural",
                neural_samples,
                X=X,
                Y=Y,
                formula=predictive_formula,
                distribution=predictive_distribution,
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
        "distribution",
        "beta_mean_rmse_mcmc",
        "beta_sd_rmse_mcmc",
        "beta_ci_overlap_95",
        "neural_beta_mean_rmse_truth",
        "mcmc_beta_mean_rmse_truth",
        "neural_posterior_predictive_mean_rmse",
        "mcmc_posterior_predictive_mean_rmse",
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


def _predictive_metrics(
    prefix: str,
    beta_samples: np.ndarray,
    *,
    X: pd.DataFrame | np.ndarray,
    Y: pd.DataFrame | np.ndarray,
    formula: str,
    distribution: str,
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
    response = _response_scale(linear, distribution)
    prediction = response.mean(axis=(0, 1))
    observed = y_frame.to_numpy(dtype=float)
    if prediction.shape != observed.shape:
        raise ValueError(f"prediction and observed Y shapes differ: {prediction.shape} vs {observed.shape}")
    species_prediction = response.mean(axis=2).reshape(-1, response.shape[-1])
    observed_species = observed.mean(axis=0)
    lo = np.quantile(species_prediction, 0.025, axis=0)
    hi = np.quantile(species_prediction, 0.975, axis=0)
    return {
        f"{prefix}_posterior_predictive_mean_rmse": _rmse(prediction, observed),
        f"{prefix}_species_mean_coverage_95": float(np.mean((observed_species >= lo) & (observed_species <= hi))),
    }


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


def _response_scale(linear: np.ndarray, distribution: str) -> np.ndarray:
    key = str(distribution).lower()
    if key in {"normal", "gaussian"}:
        return linear
    if key == "poisson":
        return np.exp(np.clip(linear, -20.0, 20.0))
    if key in {"probit", "bernoulli", "binomial"}:
        return ndtr(linear)
    raise ValueError(f"Unsupported predictive distribution {distribution!r}")


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
