"""Calibration helpers for experimental Neural-HMSC posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
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
        return {
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


def fit_beta_scale_calibration(
    posterior: BetaPosterior,
    beta_true: np.ndarray,
    *,
    nominal_level: float = 0.95,
    distribution: str | None = None,
    species_mask: np.ndarray | None = None,
    min_multiplier: float = 1e-6,
) -> BetaScaleCalibration:
    """Fit a scalar posterior-scale multiplier from a calibration split.

    The multiplier is the empirical quantile of standardized absolute ``Beta``
    errors divided by the Normal critical value for ``nominal_level``.
    """
    if not 0.0 < nominal_level < 1.0:
        raise ValueError("nominal_level must be between 0 and 1")
    if min_multiplier <= 0.0:
        raise ValueError("min_multiplier must be positive")
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
    multiplier = float(np.quantile(selected, nominal_level) / z_value)
    multiplier = max(multiplier, float(min_multiplier))
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
    return BetaPosterior(mean=mean, scale=scale * multiplier)


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
