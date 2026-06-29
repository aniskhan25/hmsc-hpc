"""Calibration helpers for experimental Neural-HMSC posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
from scipy.special import gammaln, logsumexp
from scipy.stats import norm

from pyhmsc.neural.posterior_heads import BetaPosterior


@dataclass(frozen=True)
class BetaScaleCalibration:
    """Post-hoc scale calibration for diagonal Normal ``Beta`` posteriors."""

    scale_multiplier: float
    nominal_level: float
    uncalibrated_coverage: float
    calibrated_coverage: float
    n_observations: int
    distribution: str | None = None
    n_covariates: int | None = None
    n_species: int | None = None
    method: str = "temperature_scale"
    coverage_scale_multiplier: float | None = None
    predictive_score_uncalibrated: float | None = None
    predictive_score_calibrated: float | None = None
    predictive_rate_rmse_uncalibrated: float | None = None
    predictive_rate_rmse_calibrated: float | None = None

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "BetaScaleCalibration":
        """Reconstruct a calibrator from stored posterior metadata."""
        domain = metadata.get("domain", {})
        return cls(
            scale_multiplier=float(metadata["scale_multiplier"]),
            nominal_level=float(metadata["nominal_level"]),
            uncalibrated_coverage=float(metadata["uncalibrated_coverage"]),
            calibrated_coverage=float(metadata["calibrated_coverage"]),
            n_observations=int(metadata["n_observations"]),
            distribution=domain.get("distribution"),
            n_covariates=None if domain.get("n_covariates") is None else int(domain["n_covariates"]),
            n_species=None if domain.get("n_species") is None else int(domain["n_species"]),
            method=str(metadata.get("method", "temperature_scale")),
            coverage_scale_multiplier=_optional_float(metadata, "coverage_scale_multiplier"),
            predictive_score_uncalibrated=_optional_float(metadata, "predictive_score_uncalibrated"),
            predictive_score_calibrated=_optional_float(metadata, "predictive_score_calibrated"),
            predictive_rate_rmse_uncalibrated=_optional_float(metadata, "predictive_rate_rmse_uncalibrated"),
            predictive_rate_rmse_calibrated=_optional_float(metadata, "predictive_rate_rmse_calibrated"),
        )

    def validate_domain(
        self,
        *,
        distribution: str | None = None,
        n_covariates: int | None = None,
        n_species: int | None = None,
    ) -> None:
        """Raise if this calibrator is applied outside its fitted domain."""
        if distribution is not None and self.distribution is not None:
            if str(distribution).lower() != str(self.distribution).lower():
                raise ValueError(
                    "calibration distribution mismatch: "
                    f"expected {self.distribution!r}, got {distribution!r}"
                )
        if n_covariates is not None and self.n_covariates is not None and int(n_covariates) != self.n_covariates:
            raise ValueError(
                "calibration covariate dimension mismatch: "
                f"expected {self.n_covariates}, got {n_covariates}"
            )
        if n_species is not None and self.n_species is not None and int(n_species) != self.n_species:
            raise ValueError(
                "calibration species dimension mismatch: "
                f"expected {self.n_species}, got {n_species}"
            )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable calibration metadata."""
        metadata = {
            "method": self.method,
            "parameter": "Beta",
            "scale_multiplier": float(self.scale_multiplier),
            "nominal_level": float(self.nominal_level),
            "uncalibrated_coverage": float(self.uncalibrated_coverage),
            "calibrated_coverage": float(self.calibrated_coverage),
            "n_observations": int(self.n_observations),
            "domain": {
                "distribution": self.distribution,
                "n_covariates": self.n_covariates,
                "n_species": self.n_species,
            },
        }
        if self.coverage_scale_multiplier is not None:
            metadata["coverage_scale_multiplier"] = float(self.coverage_scale_multiplier)
        if self.predictive_score_uncalibrated is not None:
            metadata["predictive_score_uncalibrated"] = float(self.predictive_score_uncalibrated)
        if self.predictive_score_calibrated is not None:
            metadata["predictive_score_calibrated"] = float(self.predictive_score_calibrated)
        if self.predictive_rate_rmse_uncalibrated is not None:
            metadata["predictive_rate_rmse_uncalibrated"] = float(self.predictive_rate_rmse_uncalibrated)
        if self.predictive_rate_rmse_calibrated is not None:
            metadata["predictive_rate_rmse_calibrated"] = float(self.predictive_rate_rmse_calibrated)
        return metadata


def fit_beta_scale_calibration(
    posterior: BetaPosterior,
    beta_true: np.ndarray,
    *,
    nominal_level: float = 0.95,
    distribution: str | None = None,
    species_mask: np.ndarray | None = None,
    min_multiplier: float = 1e-6,
    predictive_X: np.ndarray | None = None,
    predictive_Y: np.ndarray | None = None,
    poisson_eta_clip: tuple[float, float] | None = None,
    predictive_draws: int = 128,
    predictive_seed: int = 123,
    predictive_min_multiplier: float = 0.25,
    max_predictive_rate_rmse_ratio: float = 1.25,
    max_predictive_log_score_ratio: float = 1.10,
) -> BetaScaleCalibration:
    """Fit a scalar posterior-scale multiplier from a calibration split.

    The multiplier is the empirical quantile of standardized absolute ``Beta``
    errors divided by the Normal critical value for ``nominal_level``.
    """
    if not 0.0 < nominal_level < 1.0:
        raise ValueError("nominal_level must be between 0 and 1")
    if min_multiplier <= 0.0:
        raise ValueError("min_multiplier must be positive")
    if predictive_min_multiplier <= 0.0:
        raise ValueError("predictive_min_multiplier must be positive")
    if predictive_draws <= 0:
        raise ValueError("predictive_draws must be positive")
    if max_predictive_rate_rmse_ratio < 1.0:
        raise ValueError("max_predictive_rate_rmse_ratio must be at least 1")
    if max_predictive_log_score_ratio < 1.0:
        raise ValueError("max_predictive_log_score_ratio must be at least 1")
    mean = _as_numpy(posterior.mean)
    scale = _as_numpy(posterior.scale)
    truth = np.asarray(beta_true, dtype=float)
    if mean.shape != scale.shape or mean.shape != truth.shape:
        raise ValueError("posterior mean, scale, and beta_true must have the same shape")
    if np.any(scale <= 0.0):
        raise ValueError("posterior scales must be positive before calibration")

    mask = _calibration_mask(mean.shape, species_mask)
    if not np.any(mask):
        raise ValueError("calibration mask selects no coefficients")
    z_value = float(norm.ppf(0.5 + nominal_level / 2.0))
    standardized = np.abs(mean - truth) / scale
    selected = standardized[mask]
    coverage_multiplier = float(np.quantile(selected, nominal_level) / z_value)
    coverage_multiplier = max(coverage_multiplier, float(min_multiplier))
    multiplier = coverage_multiplier
    method = "temperature_scale"
    predictive_score_uncalibrated = None
    predictive_score_calibrated = None
    predictive_rate_rmse_uncalibrated = None
    predictive_rate_rmse_calibrated = None
    predictive_inputs = predictive_X is not None or predictive_Y is not None
    if predictive_inputs:
        if predictive_X is None:
            raise ValueError("predictive_X is required for predictive scale selection")
        if str(distribution).lower() != "poisson":
            raise ValueError("predictive scale selection currently supports Poisson calibration only")
        predictive_response = (
            _simulate_poisson_replicate(
                X=np.asarray(predictive_X, dtype=float),
                beta_true=truth,
                eta_clip=poisson_eta_clip,
                seed=predictive_seed + 1,
            )
            if predictive_Y is None
            else np.asarray(predictive_Y, dtype=float)
        )
        (
            multiplier,
            predictive_score_uncalibrated,
            predictive_score_calibrated,
            predictive_rate_rmse_uncalibrated,
            predictive_rate_rmse_calibrated,
        ) = _select_poisson_scale(
            posterior,
            X=np.asarray(predictive_X, dtype=float),
            Y=predictive_response,
            beta_true=truth,
            coverage_multiplier=coverage_multiplier,
            eta_clip=poisson_eta_clip,
            draws=predictive_draws,
            seed=predictive_seed,
            min_multiplier=predictive_min_multiplier,
            max_rate_rmse_ratio=max_predictive_rate_rmse_ratio,
            max_log_score_ratio=max_predictive_log_score_ratio,
        )
        method = "poisson_balanced_score_scale"
    uncalibrated_coverage = _coverage(mean, scale, truth, nominal_level, mask)
    calibrated_coverage = _coverage(mean, scale * multiplier, truth, nominal_level, mask)
    return BetaScaleCalibration(
        scale_multiplier=multiplier,
        nominal_level=float(nominal_level),
        uncalibrated_coverage=uncalibrated_coverage,
        calibrated_coverage=calibrated_coverage,
        n_observations=int(np.sum(mask)),
        distribution=None if distribution is None else str(distribution),
        n_covariates=int(mean.shape[1]),
        n_species=int(mean.shape[2]),
        method=method,
        coverage_scale_multiplier=coverage_multiplier,
        predictive_score_uncalibrated=predictive_score_uncalibrated,
        predictive_score_calibrated=predictive_score_calibrated,
        predictive_rate_rmse_uncalibrated=predictive_rate_rmse_uncalibrated,
        predictive_rate_rmse_calibrated=predictive_rate_rmse_calibrated,
    )


def apply_beta_scale_calibration(
    posterior: BetaPosterior,
    calibration: BetaScaleCalibration,
    *,
    distribution: str | None = None,
) -> BetaPosterior:
    """Return a calibrated posterior with unchanged means and rescaled scales."""
    mean = tf.convert_to_tensor(posterior.mean)
    scale = tf.convert_to_tensor(posterior.scale)
    if mean.shape.rank != 3:
        raise ValueError("Beta calibration expects posterior mean with shape batch x covariates x species")
    calibration.validate_domain(
        distribution=distribution,
        n_covariates=int(mean.shape[1]),
        n_species=int(mean.shape[2]),
    )
    multiplier = tf.cast(calibration.scale_multiplier, scale.dtype)
    scale_tril = None
    if posterior.scale_tril is not None:
        scale_tril = tf.convert_to_tensor(posterior.scale_tril) * multiplier
    return BetaPosterior(mean=mean, scale=scale * multiplier, scale_tril=scale_tril)


def calibration_metadata(calibration: BetaScaleCalibration | dict[str, Any]) -> dict[str, Any]:
    """Return JSON-serializable calibration metadata."""
    if isinstance(calibration, BetaScaleCalibration):
        return calibration.to_metadata()
    if isinstance(calibration, dict):
        return dict(calibration)
    raise TypeError("calibration must be a BetaScaleCalibration or metadata dict")


def _coverage(
    mean: np.ndarray,
    scale: np.ndarray,
    truth: np.ndarray,
    level: float,
    mask: np.ndarray,
) -> float:
    z_value = float(norm.ppf(0.5 + level / 2.0))
    covered = (truth >= mean - z_value * scale) & (truth <= mean + z_value * scale)
    return float(np.mean(covered[mask]))


def _calibration_mask(shape: tuple[int, ...], species_mask: np.ndarray | None) -> np.ndarray:
    if species_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(species_mask, dtype=bool)
    if mask.shape != (shape[0], shape[2]):
        raise ValueError(
            "species_mask must have shape batch x species for Beta calibration; "
            f"got {mask.shape}, expected {(shape[0], shape[2])}"
        )
    return np.broadcast_to(mask[:, None, :], shape)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _optional_float(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    return None if value is None else float(value)


def _select_poisson_scale(
    posterior: BetaPosterior,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    beta_true: np.ndarray,
    coverage_multiplier: float,
    eta_clip: tuple[float, float] | None,
    draws: int,
    seed: int,
    min_multiplier: float,
    max_rate_rmse_ratio: float,
    max_log_score_ratio: float,
) -> tuple[float, float, float, float, float]:
    mean = _as_numpy(posterior.mean)
    if X.ndim != 3 or X.shape[0] != mean.shape[0] or X.shape[2] != mean.shape[1]:
        raise ValueError("predictive_X must have shape batch x sites x covariates")
    if Y.shape != (mean.shape[0], X.shape[1], mean.shape[2]):
        raise ValueError("predictive_Y must have shape batch x sites x species")
    if np.any(Y < 0.0) or np.any(Y != np.floor(Y)):
        raise ValueError("predictive_Y must contain non-negative integer Poisson counts")
    if eta_clip is not None:
        if len(eta_clip) != 2 or not np.all(np.isfinite(eta_clip)) or eta_clip[0] >= eta_clip[1]:
            raise ValueError("poisson_eta_clip must contain finite, ordered bounds")

    rng = np.random.default_rng(seed)
    if posterior.scale_tril is None:
        noise = rng.normal(size=(draws,) + mean.shape)
        deviations = noise * _as_numpy(posterior.scale)[None, ...]
    else:
        scale_tril = _as_numpy(posterior.scale_tril)
        expected = (mean.shape[0], mean.shape[2], mean.shape[1], mean.shape[1])
        if scale_tril.shape != expected:
            raise ValueError(f"posterior scale_tril has shape {scale_tril.shape}, expected {expected}")
        noise = rng.normal(size=(draws, mean.shape[0], mean.shape[2], mean.shape[1]))
        deviations = np.transpose(np.einsum("bsij,dbsj->dbsi", scale_tril, noise), (0, 1, 3, 2))

    upper = max(float(coverage_multiplier), 1.0)
    bounded_coverage_multiplier = max(float(coverage_multiplier), float(min_multiplier))
    candidates = np.unique(
        np.concatenate(
            [
                np.geomspace(min_multiplier, upper, 31),
                np.asarray([1.0, bounded_coverage_multiplier], dtype=float),
            ]
        )
    )
    truth_linear = np.einsum("bnk,bks->bns", X, beta_true)
    if eta_clip is not None:
        truth_linear = np.clip(truth_linear, eta_clip[0], eta_clip[1])
    truth_rate = np.exp(truth_linear)
    scores = []
    rate_rmses = []
    for multiplier in candidates:
        score, predictive_rate = _poisson_predictive_score(
            mean,
            deviations,
            X,
            Y,
            multiplier,
            eta_clip,
        )
        scores.append(score)
        rate_rmses.append(float(np.sqrt(np.mean((predictive_rate - truth_rate) ** 2))))
    scores = np.asarray(scores)
    rate_rmses = np.asarray(rate_rmses)
    baseline_index = int(np.argmin(np.abs(candidates - 1.0)))
    eligible = (
        (rate_rmses <= rate_rmses[baseline_index] * float(max_rate_rmse_ratio))
        & (scores <= scores[baseline_index] * float(max_log_score_ratio))
    )
    eligible_indices = np.flatnonzero(eligible)
    best_index = int(eligible_indices[np.argmin(rate_rmses[eligible_indices])])
    return (
        float(candidates[best_index]),
        float(scores[baseline_index]),
        float(scores[best_index]),
        float(rate_rmses[baseline_index]),
        float(rate_rmses[best_index]),
    )


def _simulate_poisson_replicate(
    *,
    X: np.ndarray,
    beta_true: np.ndarray,
    eta_clip: tuple[float, float] | None,
    seed: int,
) -> np.ndarray:
    if X.ndim != 3 or beta_true.ndim != 3:
        raise ValueError("predictive_X and beta_true must be batched arrays")
    linear = np.einsum("bnk,bks->bns", X, beta_true)
    if eta_clip is not None:
        if len(eta_clip) != 2 or not np.all(np.isfinite(eta_clip)) or eta_clip[0] >= eta_clip[1]:
            raise ValueError("poisson_eta_clip must contain finite, ordered bounds")
        linear = np.clip(linear, eta_clip[0], eta_clip[1])
    return np.random.default_rng(seed).poisson(np.exp(linear)).astype(float)


def _poisson_predictive_score(
    mean: np.ndarray,
    deviations: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    multiplier: float,
    eta_clip: tuple[float, float] | None,
) -> tuple[float, np.ndarray]:
    beta = mean[None, ...] + float(multiplier) * deviations
    linear = np.einsum("bnk,dbks->dbns", X, beta)
    if eta_clip is not None:
        linear = np.clip(linear, eta_clip[0], eta_clip[1])
    with np.errstate(over="raise", invalid="raise"):
        try:
            rate = np.exp(linear)
        except FloatingPointError:
            return float("inf"), np.full(Y.shape, np.inf)
    log_probability = Y[None, ...] * linear - rate - gammaln(Y[None, ...] + 1.0)
    log_predictive = logsumexp(log_probability, axis=0) - np.log(deviations.shape[0])
    return float(-np.mean(log_predictive)), rate.mean(axis=0)
