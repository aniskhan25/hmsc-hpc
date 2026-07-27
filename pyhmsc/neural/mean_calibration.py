"""Predictive-only posterior-mean calibration for Neural-HMSC competitors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from scipy.special import ndtr

from pyhmsc.neural.posterior_heads import BetaPosterior


DEFAULT_SOURCE_LIKE_CONTEXTS = ("whittaker", "whittaker_source", "source_like")
DEFAULT_TRANSFER_LIKE_CONTEXTS = (
    "big_spatial",
    "big_spatial_transfer",
    "transfer_like",
)


@dataclass(frozen=True)
class BetaResponseCalibrationBatch:
    """Response-scale validation batch for predictive-mean calibration."""

    posterior: BetaPosterior
    X: np.ndarray
    Y: np.ndarray
    label: str = "transfer"


@dataclass(frozen=True)
class BetaPredictiveMeanCalibration:
    """Conservative affine shrinkage for predictive-only Beta means.

    This calibrator is intentionally predictive-only. It may be applied to
    `neural_predictive_distribution.h5`, but it must not be interpreted as a
    coefficient-posterior calibration or used for SBC rank diagnostics.
    """

    slope: float
    intercept: float
    method: str
    distribution: str | None
    n_covariates: int | None
    n_species: int | None
    calibration_rmse_uncalibrated: float
    calibration_rmse_calibrated: float
    validation_rmse_uncalibrated: float
    validation_rmse_calibrated: float
    validation_rmse_ratio: float
    selected: bool
    max_validation_rmse_ratio: float
    min_validation_rmse_improvement: float
    n_calibration_observations: int
    n_validation_observations: int
    calibration_brier_uncalibrated: float | None = None
    calibration_brier_calibrated: float | None = None
    calibration_log_loss_uncalibrated: float | None = None
    calibration_log_loss_calibrated: float | None = None
    validation_brier_uncalibrated: float | None = None
    validation_brier_calibrated: float | None = None
    validation_brier_ratio: float | None = None
    validation_log_loss_uncalibrated: float | None = None
    validation_log_loss_calibrated: float | None = None
    validation_log_loss_ratio: float | None = None
    max_validation_brier_ratio: float | None = None
    max_validation_log_loss_ratio: float | None = None
    min_validation_score_improvement: float | None = None
    transfer_validation_brier_uncalibrated: float | None = None
    transfer_validation_brier_calibrated: float | None = None
    transfer_validation_brier_ratio: float | None = None
    transfer_validation_log_loss_uncalibrated: float | None = None
    transfer_validation_log_loss_calibrated: float | None = None
    transfer_validation_log_loss_ratio: float | None = None
    max_transfer_validation_brier_ratio: float | None = None
    max_transfer_validation_log_loss_ratio: float | None = None
    min_transfer_validation_score_improvement: float | None = None
    n_transfer_validation_observations: int = 0
    transfer_validation_labels: tuple[str, ...] = ()

    @classmethod
    def identity(
        cls,
        *,
        distribution: str | None = None,
        n_covariates: int | None = None,
        n_species: int | None = None,
        method: str = "affine_shrinkage",
    ) -> "BetaPredictiveMeanCalibration":
        """Return a no-op predictive-mean calibration."""
        return cls(
            slope=1.0,
            intercept=0.0,
            method=method,
            distribution=distribution,
            n_covariates=n_covariates,
            n_species=n_species,
            calibration_rmse_uncalibrated=0.0,
            calibration_rmse_calibrated=0.0,
            validation_rmse_uncalibrated=0.0,
            validation_rmse_calibrated=0.0,
            validation_rmse_ratio=1.0,
            selected=False,
            max_validation_rmse_ratio=1.0,
            min_validation_rmse_improvement=0.0,
            n_calibration_observations=0,
            n_validation_observations=0,
        )

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "BetaPredictiveMeanCalibration":
        """Reconstruct the calibrator from metadata."""
        domain = metadata.get("domain", {})
        validation = metadata.get("validation", {})
        calibration = metadata.get("calibration_fit", {})
        response_calibration = metadata.get("response_calibration_fit", {})
        response_validation = metadata.get("response_validation", {})
        transfer_validation = metadata.get("transfer_response_validation", {})
        return cls(
            slope=float(metadata["slope"]),
            intercept=float(metadata["intercept"]),
            method=str(metadata.get("method", "affine_shrinkage")),
            distribution=domain.get("distribution"),
            n_covariates=(
                None
                if domain.get("n_covariates") is None
                else int(domain["n_covariates"])
            ),
            n_species=(
                None if domain.get("n_species") is None else int(domain["n_species"])
            ),
            calibration_rmse_uncalibrated=float(
                calibration.get("rmse_uncalibrated", 0.0)
            ),
            calibration_rmse_calibrated=float(calibration.get("rmse_calibrated", 0.0)),
            validation_rmse_uncalibrated=float(
                validation.get("rmse_uncalibrated", 0.0)
            ),
            validation_rmse_calibrated=float(validation.get("rmse_calibrated", 0.0)),
            validation_rmse_ratio=float(validation.get("rmse_ratio", 1.0)),
            selected=bool(metadata.get("selected", False)),
            max_validation_rmse_ratio=float(
                metadata.get("max_validation_rmse_ratio", 1.0)
            ),
            min_validation_rmse_improvement=float(
                metadata.get("min_validation_rmse_improvement", 0.0)
            ),
            n_calibration_observations=int(calibration.get("n_observations", 0)),
            n_validation_observations=int(validation.get("n_observations", 0)),
            calibration_brier_uncalibrated=_optional_float(
                response_calibration, "brier_uncalibrated"
            ),
            calibration_brier_calibrated=_optional_float(
                response_calibration, "brier_calibrated"
            ),
            calibration_log_loss_uncalibrated=_optional_float(
                response_calibration, "log_loss_uncalibrated"
            ),
            calibration_log_loss_calibrated=_optional_float(
                response_calibration, "log_loss_calibrated"
            ),
            validation_brier_uncalibrated=_optional_float(
                response_validation, "brier_uncalibrated"
            ),
            validation_brier_calibrated=_optional_float(
                response_validation, "brier_calibrated"
            ),
            validation_brier_ratio=_optional_float(response_validation, "brier_ratio"),
            validation_log_loss_uncalibrated=_optional_float(
                response_validation, "log_loss_uncalibrated"
            ),
            validation_log_loss_calibrated=_optional_float(
                response_validation, "log_loss_calibrated"
            ),
            validation_log_loss_ratio=_optional_float(
                response_validation, "log_loss_ratio"
            ),
            max_validation_brier_ratio=_optional_float(
                metadata, "max_validation_brier_ratio"
            ),
            max_validation_log_loss_ratio=_optional_float(
                metadata, "max_validation_log_loss_ratio"
            ),
            min_validation_score_improvement=_optional_float(
                metadata, "min_validation_score_improvement"
            ),
            transfer_validation_brier_uncalibrated=_optional_float(
                transfer_validation, "brier_uncalibrated"
            ),
            transfer_validation_brier_calibrated=_optional_float(
                transfer_validation, "brier_calibrated"
            ),
            transfer_validation_brier_ratio=_optional_float(
                transfer_validation, "brier_ratio"
            ),
            transfer_validation_log_loss_uncalibrated=_optional_float(
                transfer_validation, "log_loss_uncalibrated"
            ),
            transfer_validation_log_loss_calibrated=_optional_float(
                transfer_validation, "log_loss_calibrated"
            ),
            transfer_validation_log_loss_ratio=_optional_float(
                transfer_validation, "log_loss_ratio"
            ),
            max_transfer_validation_brier_ratio=_optional_float(
                metadata, "max_transfer_validation_brier_ratio"
            ),
            max_transfer_validation_log_loss_ratio=_optional_float(
                metadata, "max_transfer_validation_log_loss_ratio"
            ),
            min_transfer_validation_score_improvement=_optional_float(
                metadata, "min_transfer_validation_score_improvement"
            ),
            n_transfer_validation_observations=int(
                transfer_validation.get("n_observations", 0)
            ),
            transfer_validation_labels=tuple(
                str(value) for value in transfer_validation.get("labels", ())
            ),
        )

    def validate_domain(
        self,
        *,
        distribution: str | None = None,
        n_covariates: int | None = None,
        n_species: int | None = None,
    ) -> None:
        """Raise if the calibrator is applied outside its fitted domain."""
        if distribution is not None and self.distribution is not None:
            if str(distribution).lower() != str(self.distribution).lower():
                raise ValueError(
                    "predictive mean calibration distribution mismatch: "
                    f"expected {self.distribution!r}, got {distribution!r}"
                )
        if (
            n_covariates is not None
            and self.n_covariates is not None
            and int(n_covariates) != self.n_covariates
        ):
            raise ValueError(
                "predictive mean calibration covariate dimension mismatch: "
                f"expected {self.n_covariates}, got {n_covariates}"
            )
        if (
            n_species is not None
            and self.n_species is not None
            and int(n_species) != self.n_species
        ):
            raise ValueError(
                "predictive mean calibration species dimension mismatch: "
                f"expected {self.n_species}, got {n_species}"
            )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""
        metadata = {
            "semantics_version": 1,
            "method": self.method,
            "parameter": "Beta",
            "artifact_role": "predictive_only_mean",
            "selected": bool(self.selected),
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "max_validation_rmse_ratio": float(self.max_validation_rmse_ratio),
            "min_validation_rmse_improvement": float(
                self.min_validation_rmse_improvement
            ),
            "domain": {
                "distribution": self.distribution,
                "n_covariates": self.n_covariates,
                "n_species": self.n_species,
            },
            "calibration_fit": {
                "rmse_uncalibrated": float(self.calibration_rmse_uncalibrated),
                "rmse_calibrated": float(self.calibration_rmse_calibrated),
                "n_observations": int(self.n_calibration_observations),
            },
            "validation": {
                "rmse_uncalibrated": float(self.validation_rmse_uncalibrated),
                "rmse_calibrated": float(self.validation_rmse_calibrated),
                "rmse_ratio": float(self.validation_rmse_ratio),
                "n_observations": int(self.n_validation_observations),
            },
        }
        if self.max_validation_brier_ratio is not None:
            metadata["max_validation_brier_ratio"] = float(
                self.max_validation_brier_ratio
            )
        if self.max_validation_log_loss_ratio is not None:
            metadata["max_validation_log_loss_ratio"] = float(
                self.max_validation_log_loss_ratio
            )
        if self.min_validation_score_improvement is not None:
            metadata["min_validation_score_improvement"] = float(
                self.min_validation_score_improvement
            )
        if self.max_transfer_validation_brier_ratio is not None:
            metadata["max_transfer_validation_brier_ratio"] = float(
                self.max_transfer_validation_brier_ratio
            )
        if self.max_transfer_validation_log_loss_ratio is not None:
            metadata["max_transfer_validation_log_loss_ratio"] = float(
                self.max_transfer_validation_log_loss_ratio
            )
        if self.min_transfer_validation_score_improvement is not None:
            metadata["min_transfer_validation_score_improvement"] = float(
                self.min_transfer_validation_score_improvement
            )
        response_calibration = {
            "brier_uncalibrated": self.calibration_brier_uncalibrated,
            "brier_calibrated": self.calibration_brier_calibrated,
            "log_loss_uncalibrated": self.calibration_log_loss_uncalibrated,
            "log_loss_calibrated": self.calibration_log_loss_calibrated,
        }
        response_validation = {
            "brier_uncalibrated": self.validation_brier_uncalibrated,
            "brier_calibrated": self.validation_brier_calibrated,
            "brier_ratio": self.validation_brier_ratio,
            "log_loss_uncalibrated": self.validation_log_loss_uncalibrated,
            "log_loss_calibrated": self.validation_log_loss_calibrated,
            "log_loss_ratio": self.validation_log_loss_ratio,
        }
        if any(value is not None for value in response_calibration.values()):
            metadata["response_calibration_fit"] = {
                key: None if value is None else float(value)
                for key, value in response_calibration.items()
            }
        if any(value is not None for value in response_validation.values()):
            metadata["response_validation"] = {
                key: None if value is None else float(value)
                for key, value in response_validation.items()
            }
        transfer_validation = {
            "brier_uncalibrated": self.transfer_validation_brier_uncalibrated,
            "brier_calibrated": self.transfer_validation_brier_calibrated,
            "brier_ratio": self.transfer_validation_brier_ratio,
            "log_loss_uncalibrated": self.transfer_validation_log_loss_uncalibrated,
            "log_loss_calibrated": self.transfer_validation_log_loss_calibrated,
            "log_loss_ratio": self.transfer_validation_log_loss_ratio,
        }
        if any(value is not None for value in transfer_validation.values()):
            metadata["transfer_response_validation"] = {
                key: None if value is None else float(value)
                for key, value in transfer_validation.items()
            }
            metadata["transfer_response_validation"]["n_observations"] = int(
                self.n_transfer_validation_observations
            )
            metadata["transfer_response_validation"]["labels"] = [
                str(value) for value in self.transfer_validation_labels
            ]
        return metadata


def fit_beta_predictive_mean_calibration(
    calibration_posterior: BetaPosterior,
    calibration_beta_true: np.ndarray,
    *,
    validation_posterior: BetaPosterior | None = None,
    validation_beta_true: np.ndarray | None = None,
    distribution: str | None = None,
    method: str = "affine_shrinkage",
    max_validation_rmse_ratio: float = 1.0,
    min_validation_rmse_improvement: float = 0.0,
) -> BetaPredictiveMeanCalibration:
    """Fit a conservative affine correction for predictive-only Beta means."""
    if method != "affine_shrinkage":
        raise ValueError("method must be 'affine_shrinkage'")
    if max_validation_rmse_ratio < 1.0:
        raise ValueError("max_validation_rmse_ratio must be at least one")
    if min_validation_rmse_improvement < 0.0:
        raise ValueError("min_validation_rmse_improvement must be non-negative")

    calibration_mean = _posterior_mean_array(calibration_posterior)
    calibration_truth = np.asarray(calibration_beta_true, dtype=float)
    if calibration_mean.shape != calibration_truth.shape:
        raise ValueError(
            "calibration posterior mean and beta_true must have the same shape"
        )
    if validation_posterior is None:
        validation_mean = calibration_mean
        validation_truth = calibration_truth
    else:
        if validation_beta_true is None:
            raise ValueError("validation_beta_true is required with validation_posterior")
        validation_mean = _posterior_mean_array(validation_posterior)
        validation_truth = np.asarray(validation_beta_true, dtype=float)
        if validation_mean.shape != validation_truth.shape:
            raise ValueError(
                "validation posterior mean and beta_true must have the same shape"
            )

    slope, intercept = _fit_affine(calibration_mean, calibration_truth)
    calibration_calibrated = _apply_affine(calibration_mean, slope, intercept)
    validation_calibrated = _apply_affine(validation_mean, slope, intercept)
    calibration_rmse_uncalibrated = _rmse(calibration_mean, calibration_truth)
    calibration_rmse_calibrated = _rmse(calibration_calibrated, calibration_truth)
    validation_rmse_uncalibrated = _rmse(validation_mean, validation_truth)
    validation_rmse_calibrated = _rmse(validation_calibrated, validation_truth)
    ratio = validation_rmse_calibrated / max(
        validation_rmse_uncalibrated, np.finfo(float).eps
    )
    improvement = validation_rmse_uncalibrated - validation_rmse_calibrated
    selected = bool(
        ratio <= float(max_validation_rmse_ratio)
        and improvement >= float(min_validation_rmse_improvement)
    )
    if not selected:
        slope = 1.0
        intercept = 0.0
        validation_rmse_calibrated = validation_rmse_uncalibrated
        ratio = 1.0

    return BetaPredictiveMeanCalibration(
        slope=float(slope),
        intercept=float(intercept),
        method=method,
        distribution=None if distribution is None else str(distribution),
        n_covariates=int(calibration_mean.shape[1]),
        n_species=int(calibration_mean.shape[2]),
        calibration_rmse_uncalibrated=calibration_rmse_uncalibrated,
        calibration_rmse_calibrated=calibration_rmse_calibrated,
        validation_rmse_uncalibrated=validation_rmse_uncalibrated,
        validation_rmse_calibrated=validation_rmse_calibrated,
        validation_rmse_ratio=float(ratio),
        selected=selected,
        max_validation_rmse_ratio=float(max_validation_rmse_ratio),
        min_validation_rmse_improvement=float(min_validation_rmse_improvement),
        n_calibration_observations=int(calibration_mean.size),
        n_validation_observations=int(validation_mean.size),
    )


def fit_beta_response_mean_calibration(
    calibration_posterior: BetaPosterior,
    *,
    calibration_X: np.ndarray,
    calibration_Y: np.ndarray,
    validation_posterior: BetaPosterior,
    validation_X: np.ndarray,
    validation_Y: np.ndarray,
    distribution: str,
    method: str = "probit_response_affine",
    slope_grid: np.ndarray | None = None,
    intercept_grid: np.ndarray | None = None,
    max_validation_brier_ratio: float = 1.0,
    max_validation_log_loss_ratio: float = 1.0,
    min_validation_score_improvement: float = 0.0,
) -> BetaPredictiveMeanCalibration:
    """Fit predictive-only mean movement against response-scale scores."""
    if method != "probit_response_affine":
        raise ValueError("method must be 'probit_response_affine'")
    if str(distribution).lower() != "probit":
        raise ValueError("probit_response_affine requires distribution='probit'")
    if max_validation_brier_ratio < 1.0:
        raise ValueError("max_validation_brier_ratio must be at least one")
    if max_validation_log_loss_ratio < 1.0:
        raise ValueError("max_validation_log_loss_ratio must be at least one")
    if min_validation_score_improvement < 0.0:
        raise ValueError("min_validation_score_improvement must be non-negative")

    calibration_mean = _posterior_mean_array(calibration_posterior)
    validation_mean = _posterior_mean_array(validation_posterior)
    calibration_X = np.asarray(calibration_X, dtype=float)
    calibration_Y = np.asarray(calibration_Y, dtype=float)
    validation_X = np.asarray(validation_X, dtype=float)
    validation_Y = np.asarray(validation_Y, dtype=float)
    _validate_response_inputs(calibration_mean, calibration_X, calibration_Y)
    _validate_response_inputs(validation_mean, validation_X, validation_Y)
    if np.any((calibration_Y != 0.0) & (calibration_Y != 1.0)) or np.any(
        (validation_Y != 0.0) & (validation_Y != 1.0)
    ):
        raise ValueError("probit response mean calibration requires binary Y")

    if slope_grid is None:
        slope_grid = np.linspace(0.75, 1.25, 21)
    if intercept_grid is None:
        intercept_grid = np.linspace(-0.4, 0.4, 33)
    candidates = _candidate_pairs(slope_grid, intercept_grid)
    calibration_scores = []
    for slope, intercept in candidates:
        probability = _probit_predictive_probability(
            calibration_posterior,
            calibration_X,
            slope=slope,
            intercept=intercept,
        )
        calibration_scores.append(_response_score_record(probability, calibration_Y))
    calibration_objective = np.asarray(
        [record["brier"] + record["log_loss"] for record in calibration_scores],
        dtype=float,
    )
    best_index = int(np.argmin(calibration_objective))
    best_slope, best_intercept = candidates[best_index]

    calibration_identity = _response_score_record(
        _probit_predictive_probability(
            calibration_posterior,
            calibration_X,
            slope=1.0,
            intercept=0.0,
        ),
        calibration_Y,
    )
    calibration_selected = calibration_scores[best_index]
    validation_identity = _response_score_record(
        _probit_predictive_probability(
            validation_posterior,
            validation_X,
            slope=1.0,
            intercept=0.0,
        ),
        validation_Y,
    )
    validation_selected = _response_score_record(
        _probit_predictive_probability(
            validation_posterior,
            validation_X,
            slope=best_slope,
            intercept=best_intercept,
        ),
        validation_Y,
    )
    brier_ratio = validation_selected["brier"] / max(
        validation_identity["brier"], np.finfo(float).eps
    )
    log_loss_ratio = validation_selected["log_loss"] / max(
        validation_identity["log_loss"], np.finfo(float).eps
    )
    identity_score = validation_identity["brier"] + validation_identity["log_loss"]
    selected_score = validation_selected["brier"] + validation_selected["log_loss"]
    score_improvement = identity_score - selected_score
    selected = bool(
        brier_ratio <= float(max_validation_brier_ratio)
        and log_loss_ratio <= float(max_validation_log_loss_ratio)
        and score_improvement >= float(min_validation_score_improvement)
    )
    if not selected:
        best_slope = 1.0
        best_intercept = 0.0
        validation_selected = validation_identity
        brier_ratio = 1.0
        log_loss_ratio = 1.0

    return BetaPredictiveMeanCalibration(
        slope=float(best_slope),
        intercept=float(best_intercept),
        method=method,
        distribution=str(distribution),
        n_covariates=int(calibration_mean.shape[1]),
        n_species=int(calibration_mean.shape[2]),
        calibration_rmse_uncalibrated=0.0,
        calibration_rmse_calibrated=0.0,
        validation_rmse_uncalibrated=0.0,
        validation_rmse_calibrated=0.0,
        validation_rmse_ratio=1.0,
        selected=selected,
        max_validation_rmse_ratio=1.0,
        min_validation_rmse_improvement=0.0,
        n_calibration_observations=int(calibration_Y.size),
        n_validation_observations=int(validation_Y.size),
        calibration_brier_uncalibrated=float(calibration_identity["brier"]),
        calibration_brier_calibrated=float(calibration_selected["brier"]),
        calibration_log_loss_uncalibrated=float(
            calibration_identity["log_loss"]
        ),
        calibration_log_loss_calibrated=float(calibration_selected["log_loss"]),
        validation_brier_uncalibrated=float(validation_identity["brier"]),
        validation_brier_calibrated=float(validation_selected["brier"]),
        validation_brier_ratio=float(brier_ratio),
        validation_log_loss_uncalibrated=float(validation_identity["log_loss"]),
        validation_log_loss_calibrated=float(validation_selected["log_loss"]),
        validation_log_loss_ratio=float(log_loss_ratio),
        max_validation_brier_ratio=float(max_validation_brier_ratio),
        max_validation_log_loss_ratio=float(max_validation_log_loss_ratio),
        min_validation_score_improvement=float(min_validation_score_improvement),
    )


def fit_beta_transfer_response_mean_calibration(
    calibration_posterior: BetaPosterior,
    *,
    calibration_X: np.ndarray,
    calibration_Y: np.ndarray,
    source_validation_posterior: BetaPosterior,
    source_validation_X: np.ndarray,
    source_validation_Y: np.ndarray,
    transfer_validation_batches: list[BetaResponseCalibrationBatch],
    distribution: str,
    method: str = "probit_transfer_response_affine",
    slope_grid: np.ndarray | None = None,
    intercept_grid: np.ndarray | None = None,
    max_source_validation_brier_ratio: float = 1.0,
    max_source_validation_log_loss_ratio: float = 1.0,
    max_transfer_validation_brier_ratio: float = 1.0,
    max_transfer_validation_log_loss_ratio: float = 1.0,
    min_transfer_validation_score_improvement: float = 0.0,
) -> BetaPredictiveMeanCalibration:
    """Fit predictive-only response-mean movement with transfer validation.

    The candidate is predictive-only: it may change posterior means in
    predictive artifacts, but it must not be used as coefficient-posterior
    calibration or SBC uncertainty calibration.
    """
    if method != "probit_transfer_response_affine":
        raise ValueError("method must be 'probit_transfer_response_affine'")
    if str(distribution).lower() != "probit":
        raise ValueError(
            "probit_transfer_response_affine requires distribution='probit'"
        )
    if not transfer_validation_batches:
        raise ValueError("transfer_validation_batches must not be empty")
    for name, value in {
        "max_source_validation_brier_ratio": max_source_validation_brier_ratio,
        "max_source_validation_log_loss_ratio": max_source_validation_log_loss_ratio,
        "max_transfer_validation_brier_ratio": max_transfer_validation_brier_ratio,
        "max_transfer_validation_log_loss_ratio": max_transfer_validation_log_loss_ratio,
    }.items():
        if float(value) < 1.0:
            raise ValueError(f"{name} must be at least one")
    if min_transfer_validation_score_improvement < 0.0:
        raise ValueError(
            "min_transfer_validation_score_improvement must be non-negative"
        )

    calibration_mean = _posterior_mean_array(calibration_posterior)
    calibration_X = np.asarray(calibration_X, dtype=float)
    calibration_Y = np.asarray(calibration_Y, dtype=float)
    source_validation_mean = _posterior_mean_array(source_validation_posterior)
    source_validation_X = np.asarray(source_validation_X, dtype=float)
    source_validation_Y = np.asarray(source_validation_Y, dtype=float)
    _validate_response_inputs(calibration_mean, calibration_X, calibration_Y)
    _validate_response_inputs(
        source_validation_mean,
        source_validation_X,
        source_validation_Y,
    )
    if np.any((calibration_Y != 0.0) & (calibration_Y != 1.0)) or np.any(
        (source_validation_Y != 0.0) & (source_validation_Y != 1.0)
    ):
        raise ValueError("probit transfer response calibration requires binary Y")
    transfer_batches = [
        _validated_response_batch(batch) for batch in transfer_validation_batches
    ]

    if slope_grid is None:
        slope_grid = np.linspace(0.85, 1.15, 13)
    if intercept_grid is None:
        intercept_grid = np.linspace(-0.2, 0.2, 17)
    candidates = _candidate_pairs(slope_grid, intercept_grid)

    calibration_scores = []
    transfer_scores = []
    for slope, intercept in candidates:
        calibration_scores.append(
            _score_response_batch(
                calibration_posterior,
                calibration_X,
                calibration_Y,
                slope=slope,
                intercept=intercept,
            )
        )
        transfer_scores.append(
            _combined_response_score(
                transfer_batches,
                slope=slope,
                intercept=intercept,
            )
        )
    objective = np.asarray(
        [
            source["brier"]
            + source["log_loss"]
            + transfer["brier"]
            + transfer["log_loss"]
            for source, transfer in zip(calibration_scores, transfer_scores)
        ],
        dtype=float,
    )
    best_index = int(np.argmin(objective))
    best_slope, best_intercept = candidates[best_index]
    calibration_identity = _score_response_batch(
        calibration_posterior,
        calibration_X,
        calibration_Y,
        slope=1.0,
        intercept=0.0,
    )
    calibration_selected = calibration_scores[best_index]
    source_identity = _score_response_batch(
        source_validation_posterior,
        source_validation_X,
        source_validation_Y,
        slope=1.0,
        intercept=0.0,
    )
    source_selected = _score_response_batch(
        source_validation_posterior,
        source_validation_X,
        source_validation_Y,
        slope=best_slope,
        intercept=best_intercept,
    )
    transfer_identity = _combined_response_score(
        transfer_batches,
        slope=1.0,
        intercept=0.0,
    )
    transfer_selected = transfer_scores[best_index]

    source_brier_ratio = source_selected["brier"] / max(
        source_identity["brier"], np.finfo(float).eps
    )
    source_log_loss_ratio = source_selected["log_loss"] / max(
        source_identity["log_loss"], np.finfo(float).eps
    )
    transfer_brier_ratio = transfer_selected["brier"] / max(
        transfer_identity["brier"], np.finfo(float).eps
    )
    transfer_log_loss_ratio = transfer_selected["log_loss"] / max(
        transfer_identity["log_loss"], np.finfo(float).eps
    )
    transfer_improvement = (
        transfer_identity["brier"]
        + transfer_identity["log_loss"]
        - transfer_selected["brier"]
        - transfer_selected["log_loss"]
    )
    selected = bool(
        source_brier_ratio <= float(max_source_validation_brier_ratio)
        and source_log_loss_ratio <= float(max_source_validation_log_loss_ratio)
        and transfer_brier_ratio <= float(max_transfer_validation_brier_ratio)
        and transfer_log_loss_ratio <= float(max_transfer_validation_log_loss_ratio)
        and transfer_improvement
        >= float(min_transfer_validation_score_improvement)
    )
    if not selected:
        best_slope = 1.0
        best_intercept = 0.0
        source_selected = source_identity
        transfer_selected = transfer_identity
        source_brier_ratio = 1.0
        source_log_loss_ratio = 1.0
        transfer_brier_ratio = 1.0
        transfer_log_loss_ratio = 1.0

    return BetaPredictiveMeanCalibration(
        slope=float(best_slope),
        intercept=float(best_intercept),
        method=method,
        distribution=str(distribution),
        n_covariates=int(calibration_mean.shape[1]),
        n_species=int(calibration_mean.shape[2]),
        calibration_rmse_uncalibrated=0.0,
        calibration_rmse_calibrated=0.0,
        validation_rmse_uncalibrated=0.0,
        validation_rmse_calibrated=0.0,
        validation_rmse_ratio=1.0,
        selected=selected,
        max_validation_rmse_ratio=1.0,
        min_validation_rmse_improvement=0.0,
        n_calibration_observations=int(calibration_Y.size),
        n_validation_observations=int(source_validation_Y.size),
        calibration_brier_uncalibrated=float(calibration_identity["brier"]),
        calibration_brier_calibrated=float(calibration_selected["brier"]),
        calibration_log_loss_uncalibrated=float(calibration_identity["log_loss"]),
        calibration_log_loss_calibrated=float(calibration_selected["log_loss"]),
        validation_brier_uncalibrated=float(source_identity["brier"]),
        validation_brier_calibrated=float(source_selected["brier"]),
        validation_brier_ratio=float(source_brier_ratio),
        validation_log_loss_uncalibrated=float(source_identity["log_loss"]),
        validation_log_loss_calibrated=float(source_selected["log_loss"]),
        validation_log_loss_ratio=float(source_log_loss_ratio),
        max_validation_brier_ratio=float(max_source_validation_brier_ratio),
        max_validation_log_loss_ratio=float(max_source_validation_log_loss_ratio),
        min_validation_score_improvement=0.0,
        transfer_validation_brier_uncalibrated=float(transfer_identity["brier"]),
        transfer_validation_brier_calibrated=float(transfer_selected["brier"]),
        transfer_validation_brier_ratio=float(transfer_brier_ratio),
        transfer_validation_log_loss_uncalibrated=float(
            transfer_identity["log_loss"]
        ),
        transfer_validation_log_loss_calibrated=float(
            transfer_selected["log_loss"]
        ),
        transfer_validation_log_loss_ratio=float(transfer_log_loss_ratio),
        max_transfer_validation_brier_ratio=float(
            max_transfer_validation_brier_ratio
        ),
        max_transfer_validation_log_loss_ratio=float(
            max_transfer_validation_log_loss_ratio
        ),
        min_transfer_validation_score_improvement=float(
            min_transfer_validation_score_improvement
        ),
        n_transfer_validation_observations=int(
            sum(batch.Y.size for batch in transfer_batches)
        ),
        transfer_validation_labels=tuple(batch.label for batch in transfer_batches),
    )


def fit_beta_transfer_response_branch_calibration(
    calibration_batches: list[BetaResponseCalibrationBatch],
    *,
    validation_batches: list[BetaResponseCalibrationBatch],
    distribution: str,
    method: str = "probit_transfer_response_branch_affine",
    slope_grid: np.ndarray | None = None,
    intercept_grid: np.ndarray | None = None,
    max_validation_brier_ratio: float = 1.0,
    max_validation_log_loss_ratio: float = 1.0,
    min_validation_score_improvement: float = 0.0,
) -> BetaPredictiveMeanCalibration:
    """Fit a transfer branch on OOD simulations and gate it on a separate pool."""
    if method != "probit_transfer_response_branch_affine":
        raise ValueError("method must be 'probit_transfer_response_branch_affine'")
    if str(distribution).lower() != "probit":
        raise ValueError(
            "probit_transfer_response_branch_affine requires distribution='probit'"
        )
    if not calibration_batches:
        raise ValueError("calibration_batches must not be empty")
    if not validation_batches:
        raise ValueError("validation_batches must not be empty")
    for name, value in {
        "max_validation_brier_ratio": max_validation_brier_ratio,
        "max_validation_log_loss_ratio": max_validation_log_loss_ratio,
    }.items():
        if float(value) < 1.0:
            raise ValueError(f"{name} must be at least one")
    if min_validation_score_improvement < 0.0:
        raise ValueError("min_validation_score_improvement must be non-negative")

    fitted_batches = [
        _validated_response_batch(batch) for batch in calibration_batches
    ]
    heldout_batches = [
        _validated_response_batch(batch) for batch in validation_batches
    ]
    if slope_grid is None:
        slope_grid = np.linspace(0.85, 1.15, 13)
    if intercept_grid is None:
        intercept_grid = np.linspace(-0.2, 0.2, 17)
    candidates = _candidate_pairs(slope_grid, intercept_grid)
    calibration_scores = [
        _combined_response_score(
            fitted_batches,
            slope=slope,
            intercept=intercept,
        )
        for slope, intercept in candidates
    ]
    objective = np.asarray(
        [score["brier"] + score["log_loss"] for score in calibration_scores],
        dtype=float,
    )
    best_index = int(np.argmin(objective))
    best_slope, best_intercept = candidates[best_index]
    calibration_identity = _combined_response_score(
        fitted_batches,
        slope=1.0,
        intercept=0.0,
    )
    calibration_selected = calibration_scores[best_index]
    validation_identity = _combined_response_score(
        heldout_batches,
        slope=1.0,
        intercept=0.0,
    )
    validation_selected = _combined_response_score(
        heldout_batches,
        slope=best_slope,
        intercept=best_intercept,
    )
    brier_ratio = validation_selected["brier"] / max(
        validation_identity["brier"], np.finfo(float).eps
    )
    log_loss_ratio = validation_selected["log_loss"] / max(
        validation_identity["log_loss"], np.finfo(float).eps
    )
    validation_improvement = (
        validation_identity["brier"]
        + validation_identity["log_loss"]
        - validation_selected["brier"]
        - validation_selected["log_loss"]
    )
    selected = bool(
        brier_ratio <= float(max_validation_brier_ratio)
        and log_loss_ratio <= float(max_validation_log_loss_ratio)
        and validation_improvement >= float(min_validation_score_improvement)
    )
    if not selected:
        best_slope = 1.0
        best_intercept = 0.0
        validation_selected = validation_identity
        brier_ratio = 1.0
        log_loss_ratio = 1.0

    first_mean = _posterior_mean_array(fitted_batches[0].posterior)
    calibration_observations = sum(batch.Y.size for batch in fitted_batches)
    validation_observations = sum(batch.Y.size for batch in heldout_batches)
    validation_labels = tuple(batch.label for batch in heldout_batches)
    return BetaPredictiveMeanCalibration(
        slope=float(best_slope),
        intercept=float(best_intercept),
        method=method,
        distribution=str(distribution),
        n_covariates=int(first_mean.shape[1]),
        n_species=int(first_mean.shape[2]),
        calibration_rmse_uncalibrated=0.0,
        calibration_rmse_calibrated=0.0,
        validation_rmse_uncalibrated=0.0,
        validation_rmse_calibrated=0.0,
        validation_rmse_ratio=1.0,
        selected=selected,
        max_validation_rmse_ratio=1.0,
        min_validation_rmse_improvement=0.0,
        n_calibration_observations=int(calibration_observations),
        n_validation_observations=int(validation_observations),
        calibration_brier_uncalibrated=float(calibration_identity["brier"]),
        calibration_brier_calibrated=float(calibration_selected["brier"]),
        calibration_log_loss_uncalibrated=float(
            calibration_identity["log_loss"]
        ),
        calibration_log_loss_calibrated=float(
            calibration_selected["log_loss"]
        ),
        validation_brier_uncalibrated=float(validation_identity["brier"]),
        validation_brier_calibrated=float(validation_selected["brier"]),
        validation_brier_ratio=float(brier_ratio),
        validation_log_loss_uncalibrated=float(validation_identity["log_loss"]),
        validation_log_loss_calibrated=float(validation_selected["log_loss"]),
        validation_log_loss_ratio=float(log_loss_ratio),
        max_validation_brier_ratio=float(max_validation_brier_ratio),
        max_validation_log_loss_ratio=float(max_validation_log_loss_ratio),
        min_validation_score_improvement=float(min_validation_score_improvement),
        transfer_validation_brier_uncalibrated=float(
            validation_identity["brier"]
        ),
        transfer_validation_brier_calibrated=float(validation_selected["brier"]),
        transfer_validation_brier_ratio=float(brier_ratio),
        transfer_validation_log_loss_uncalibrated=float(
            validation_identity["log_loss"]
        ),
        transfer_validation_log_loss_calibrated=float(
            validation_selected["log_loss"]
        ),
        transfer_validation_log_loss_ratio=float(log_loss_ratio),
        max_transfer_validation_brier_ratio=float(max_validation_brier_ratio),
        max_transfer_validation_log_loss_ratio=float(
            max_validation_log_loss_ratio
        ),
        min_transfer_validation_score_improvement=float(
            min_validation_score_improvement
        ),
        n_transfer_validation_observations=int(validation_observations),
        transfer_validation_labels=validation_labels,
    )


def apply_beta_predictive_mean_calibration(
    posterior: BetaPosterior,
    calibration: BetaPredictiveMeanCalibration,
    *,
    distribution: str | None = None,
) -> BetaPosterior:
    """Apply predictive-only mean calibration without changing uncertainty."""
    mean = tf.convert_to_tensor(posterior.mean)
    scale = tf.convert_to_tensor(posterior.scale)
    if mean.shape.rank != 3:
        raise ValueError(
            "predictive mean calibration expects mean with shape batch x covariates x species"
        )
    calibration.validate_domain(
        distribution=distribution,
        n_covariates=int(mean.shape[1]),
        n_species=int(mean.shape[2]),
    )
    slope = tf.cast(calibration.slope, mean.dtype)
    intercept = tf.cast(calibration.intercept, mean.dtype)
    return BetaPosterior(
        mean=mean * slope + intercept,
        scale=scale,
        scale_tril=posterior.scale_tril,
    )


def domain_conditional_predictive_mean_selector_metadata(
    calibration: BetaPredictiveMeanCalibration,
    *,
    source_like_contexts: tuple[str, ...] = DEFAULT_SOURCE_LIKE_CONTEXTS,
    transfer_like_contexts: tuple[str, ...] = DEFAULT_TRANSFER_LIKE_CONTEXTS,
    min_transfer_validation_brier_gain: float = 1.0e-4,
    min_transfer_validation_log_loss_gain: float = 5.0e-4,
    max_transfer_slope_delta: float = 0.05,
    max_transfer_abs_intercept: float = 0.025,
) -> dict[str, Any]:
    """Return metadata for a conservative context-conditioned selector."""
    guard = _transfer_stability_guard(
        calibration,
        min_validation_brier_gain=min_transfer_validation_brier_gain,
        min_validation_log_loss_gain=min_transfer_validation_log_loss_gain,
        max_slope_delta=max_transfer_slope_delta,
        max_abs_intercept=max_transfer_abs_intercept,
    )
    if not calibration.selected or not guard["passed"]:
        active_contexts: tuple[str, ...] = ()
    else:
        active_contexts = tuple(str(value) for value in transfer_like_contexts)
    return {
        "semantics_version": 1,
        "method": "domain_conditional_context_selector",
        "artifact_role": "predictive_only_mean_selector",
        "parameter": "Beta",
        "candidate": calibration.to_metadata(),
        "source_like_contexts": [str(value) for value in source_like_contexts],
        "transfer_like_contexts": [str(value) for value in transfer_like_contexts],
        "active_contexts": list(active_contexts),
        "default_action": "identity",
        "transfer_stability_guard": guard,
        "selection_rule": (
            "apply candidate only for transfer-like contexts when the candidate "
            "passed independent response-scale validation and transfer-stability "
            "gain/amplitude checks; otherwise use identity"
        ),
    }


def independent_source_transfer_predictive_mean_selector_metadata(
    source_calibration: BetaPredictiveMeanCalibration,
    transfer_calibration: BetaPredictiveMeanCalibration,
    *,
    source_like_contexts: tuple[str, ...] = DEFAULT_SOURCE_LIKE_CONTEXTS,
    transfer_like_contexts: tuple[str, ...] = DEFAULT_TRANSFER_LIKE_CONTEXTS,
) -> dict[str, Any]:
    """Serialize independently selected source and transfer affine branches."""
    source_calibration.validate_domain(
        distribution=transfer_calibration.distribution,
        n_covariates=transfer_calibration.n_covariates,
        n_species=transfer_calibration.n_species,
    )
    return {
        "semantics_version": 1,
        "method": "independent_source_transfer_affine_selector",
        "artifact_role": "predictive_only_mean_selector",
        "parameter": "Beta",
        "source_branch": source_calibration.to_metadata(),
        "transfer_branch": transfer_calibration.to_metadata(),
        "source_like_contexts": [str(value) for value in source_like_contexts],
        "transfer_like_contexts": [str(value) for value in transfer_like_contexts],
        "default_action": "identity",
        "selection_rule": (
            "choose the independently simulation-gated source or transfer branch "
            "from a predeclared deployment context; never use real held-out outcomes"
        ),
    }


def evaluate_beta_target_context_gate(
    calibration: BetaPredictiveMeanCalibration,
    calibration_batches: list[BetaResponseCalibrationBatch],
    *,
    validation_batches: list[BetaResponseCalibrationBatch],
    max_brier_ratio: float = 1.0,
    max_log_loss_ratio: float = 1.0,
    min_score_improvement: float = 0.0,
    context_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen mean calibrator on two target-shaped simulation pools."""
    if not calibration_batches:
        raise ValueError("calibration_batches must not be empty")
    if not validation_batches:
        raise ValueError("validation_batches must not be empty")
    if float(max_brier_ratio) < 1.0:
        raise ValueError("max_brier_ratio must be at least one")
    if float(max_log_loss_ratio) < 1.0:
        raise ValueError("max_log_loss_ratio must be at least one")
    if float(min_score_improvement) < 0.0:
        raise ValueError("min_score_improvement must be non-negative")

    fitted_batches = [
        _validated_response_batch(batch) for batch in calibration_batches
    ]
    heldout_batches = [
        _validated_response_batch(batch) for batch in validation_batches
    ]
    first_mean = _posterior_mean_array(fitted_batches[0].posterior)
    calibration.validate_domain(
        distribution="probit",
        n_covariates=int(first_mean.shape[1]),
        n_species=int(first_mean.shape[2]),
    )
    thresholds = {
        "max_brier_ratio": float(max_brier_ratio),
        "max_log_loss_ratio": float(max_log_loss_ratio),
        "min_score_improvement": float(min_score_improvement),
    }
    calibration_record = _predictive_mean_gate_record(
        calibration,
        fitted_batches,
        thresholds=thresholds,
    )
    validation_record = _predictive_mean_gate_record(
        calibration,
        heldout_batches,
        thresholds=thresholds,
    )
    failure_reasons: list[str] = []
    if not calibration.selected:
        failure_reasons.append("generic_transfer_branch_not_selected")
    for pool_name, record in (
        ("target_calibration", calibration_record),
        ("target_validation", validation_record),
    ):
        failure_reasons.extend(
            f"{pool_name}:{reason}" for reason in record["failure_reasons"]
        )
    return {
        "semantics_version": 1,
        "kind": "target_context_independent_simulation_gate",
        "artifact_role": "predictive_only_mean_selection_gate",
        "parameter": "Beta",
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "thresholds": thresholds,
        "candidate": {
            "method": calibration.method,
            "selected_by_generic_ood_gate": bool(calibration.selected),
            "slope": float(calibration.slope),
            "intercept": float(calibration.intercept),
        },
        "target_calibration": calibration_record,
        "target_validation": validation_record,
        "context": {} if context_metadata is None else dict(context_metadata),
        "target_responses_used": False,
        "selection_rule": (
            "the frozen generic-OOD branch must improve Brier plus log loss "
            "without degrading either metric on independent target-shaped "
            "calibration and validation simulations"
        ),
    }


def target_context_conditioned_source_transfer_selector_metadata(
    selector_metadata: dict[str, Any],
    target_context_gate: dict[str, Any],
    *,
    target_contexts: tuple[str, ...] = DEFAULT_TRANSFER_LIKE_CONTEXTS,
) -> dict[str, Any]:
    """Attach a target-shaped simulation gate to a source/transfer selector."""
    if selector_metadata.get("method") != "independent_source_transfer_affine_selector":
        raise ValueError(
            "target context gating requires an independent source/transfer selector"
        )
    if target_context_gate.get("kind") != "target_context_independent_simulation_gate":
        raise ValueError("invalid target context gate metadata")
    return {
        **selector_metadata,
        "semantics_version": 2,
        "method": "target_context_conditioned_source_transfer_affine_selector",
        "generic_selector_method": selector_metadata["method"],
        "target_contexts": [str(value) for value in target_contexts],
        "target_context_gate": dict(target_context_gate),
        "selection_rule": (
            "choose the predeclared source branch directly; apply the transfer "
            "branch only when both its frozen generic OOD gate and an independent "
            "target-shaped simulation gate pass; never use target outcomes"
        ),
    }


def select_predictive_mean_calibration_for_context(
    selector_metadata: dict[str, Any],
    *,
    context: str,
    distribution: str | None = None,
    n_covariates: int | None = None,
    n_species: int | None = None,
) -> tuple[BetaPredictiveMeanCalibration | None, dict[str, Any]]:
    """Select a predictive-mean calibration for a named deployment context."""
    method = str(selector_metadata.get("method", ""))
    if method == "target_context_conditioned_source_transfer_affine_selector":
        return _select_target_context_conditioned_transfer_branch(
            selector_metadata,
            context=context,
            distribution=distribution,
            n_covariates=n_covariates,
            n_species=n_species,
        )
    if method == "independent_source_transfer_affine_selector":
        return _select_independent_source_transfer_branch(
            selector_metadata,
            context=context,
            distribution=distribution,
            n_covariates=n_covariates,
            n_species=n_species,
        )
    if method != "domain_conditional_context_selector":
        raise ValueError(f"unsupported predictive mean selector method: {method!r}")
    context_label = str(context)
    active_contexts = {str(value) for value in selector_metadata.get("active_contexts", [])}
    source_like = {
        str(value) for value in selector_metadata.get("source_like_contexts", [])
    }
    transfer_like = {
        str(value) for value in selector_metadata.get("transfer_like_contexts", [])
    }
    decision = {
        "context": context_label,
        "method": method,
        "action": "identity",
        "selected": False,
        "reason": "context_not_active",
        "context_family": "unknown",
    }
    if context_label in source_like:
        decision["context_family"] = "source_like"
        decision["reason"] = "source_like_context_uses_identity"
        return None, decision
    if context_label in transfer_like:
        decision["context_family"] = "transfer_like"
    if context_label not in active_contexts:
        guard = selector_metadata.get("transfer_stability_guard")
        if (
            context_label in transfer_like
            and isinstance(guard, dict)
            and not bool(guard.get("passed", False))
        ):
            decision["reason"] = "candidate_failed_transfer_stability_guard"
            decision["transfer_stability_guard"] = guard
        return None, decision

    candidate_metadata = selector_metadata.get("candidate")
    if not isinstance(candidate_metadata, dict):
        decision["reason"] = "missing_candidate_metadata"
        return None, decision
    candidate = BetaPredictiveMeanCalibration.from_metadata(candidate_metadata)
    if not candidate.selected:
        decision["reason"] = "candidate_not_selected"
        return None, decision
    candidate.validate_domain(
        distribution=distribution,
        n_covariates=n_covariates,
        n_species=n_species,
    )
    decision.update(
        {
            "action": "apply_candidate",
            "selected": True,
            "reason": "active_transfer_like_context",
            "candidate_method": candidate.method,
            "candidate_slope": float(candidate.slope),
            "candidate_intercept": float(candidate.intercept),
            "transfer_stability_guard": selector_metadata.get(
                "transfer_stability_guard"
            ),
        }
    )
    return candidate, decision


def _select_independent_source_transfer_branch(
    selector_metadata: dict[str, Any],
    *,
    context: str,
    distribution: str | None,
    n_covariates: int | None,
    n_species: int | None,
) -> tuple[BetaPredictiveMeanCalibration | None, dict[str, Any]]:
    context_label = str(context)
    method = str(
        selector_metadata.get("method", "independent_source_transfer_affine_selector")
    )
    source_like = {
        str(value) for value in selector_metadata.get("source_like_contexts", [])
    }
    transfer_like = {
        str(value) for value in selector_metadata.get("transfer_like_contexts", [])
    }
    if context_label in source_like:
        family = "source_like"
        branch_name = "source_branch"
    elif context_label in transfer_like:
        family = "transfer_like"
        branch_name = "transfer_branch"
    else:
        return None, {
            "context": context_label,
            "method": method,
            "context_family": "unknown",
            "branch": None,
            "action": "identity",
            "selected": False,
            "reason": "unknown_context_uses_identity",
        }
    branch_metadata = selector_metadata.get(branch_name)
    if not isinstance(branch_metadata, dict):
        return None, {
            "context": context_label,
            "method": method,
            "context_family": family,
            "branch": branch_name,
            "action": "identity",
            "selected": False,
            "reason": "missing_branch_metadata",
        }
    branch = BetaPredictiveMeanCalibration.from_metadata(branch_metadata)
    branch.validate_domain(
        distribution=distribution,
        n_covariates=n_covariates,
        n_species=n_species,
    )
    decision = {
        "context": context_label,
        "method": method,
        "context_family": family,
        "branch": branch_name,
        "action": "identity",
        "selected": False,
        "reason": "branch_not_selected_on_independent_simulation",
        "candidate_method": branch.method,
        "candidate_slope": float(branch.slope),
        "candidate_intercept": float(branch.intercept),
    }
    if not branch.selected:
        return None, decision
    decision.update(
        {
            "action": "apply_candidate",
            "selected": True,
            "reason": "independently_selected_context_branch",
        }
    )
    return branch, decision


def _select_target_context_conditioned_transfer_branch(
    selector_metadata: dict[str, Any],
    *,
    context: str,
    distribution: str | None,
    n_covariates: int | None,
    n_species: int | None,
) -> tuple[BetaPredictiveMeanCalibration | None, dict[str, Any]]:
    branch, decision = _select_independent_source_transfer_branch(
        selector_metadata,
        context=context,
        distribution=distribution,
        n_covariates=n_covariates,
        n_species=n_species,
    )
    if decision.get("context_family") != "transfer_like" or branch is None:
        return branch, decision

    target_contexts = {
        str(value) for value in selector_metadata.get("target_contexts", [])
    }
    gate = selector_metadata.get("target_context_gate")
    decision["generic_ood_gate_passed"] = True
    decision["target_context_gate"] = gate
    if str(context) not in target_contexts:
        decision.update(
            {
                "action": "identity",
                "selected": False,
                "reason": "transfer_context_missing_target_gate",
            }
        )
        return None, decision
    if not isinstance(gate, dict) or not bool(gate.get("passed", False)):
        decision.update(
            {
                "action": "identity",
                "selected": False,
                "reason": "target_context_simulation_gate_failed",
            }
        )
        return None, decision
    decision["reason"] = "generic_and_target_context_simulation_gates_passed"
    return branch, decision


def _transfer_stability_guard(
    calibration: BetaPredictiveMeanCalibration,
    *,
    min_validation_brier_gain: float,
    min_validation_log_loss_gain: float,
    max_slope_delta: float,
    max_abs_intercept: float,
) -> dict[str, Any]:
    thresholds = {
        "min_validation_brier_gain": _nonnegative_float(
            min_validation_brier_gain,
            "min_transfer_validation_brier_gain",
        ),
        "min_validation_log_loss_gain": _nonnegative_float(
            min_validation_log_loss_gain,
            "min_transfer_validation_log_loss_gain",
        ),
        "max_slope_delta": _nonnegative_float(
            max_slope_delta,
            "max_transfer_slope_delta",
        ),
        "max_abs_intercept": _nonnegative_float(
            max_abs_intercept,
            "max_transfer_abs_intercept",
        ),
    }
    brier_gain = _metric_gain(
        calibration.transfer_validation_brier_uncalibrated,
        calibration.transfer_validation_brier_calibrated,
    )
    log_loss_gain = _metric_gain(
        calibration.transfer_validation_log_loss_uncalibrated,
        calibration.transfer_validation_log_loss_calibrated,
    )
    if brier_gain is None:
        brier_gain = _metric_gain(
            calibration.validation_brier_uncalibrated,
            calibration.validation_brier_calibrated,
        )
    if log_loss_gain is None:
        log_loss_gain = _metric_gain(
            calibration.validation_log_loss_uncalibrated,
            calibration.validation_log_loss_calibrated,
        )
    slope_delta = abs(float(calibration.slope) - 1.0)
    abs_intercept = abs(float(calibration.intercept))
    failures: list[str] = []
    if not calibration.selected:
        failures.append("candidate_not_selected")
    if brier_gain is None:
        failures.append("missing_validation_brier_gain")
    elif brier_gain + 1.0e-12 < thresholds["min_validation_brier_gain"]:
        failures.append(
            "validation_brier_gain_below_margin:"
            f"{brier_gain:.6g}<"
            f"{thresholds['min_validation_brier_gain']:.6g}"
        )
    if log_loss_gain is None:
        failures.append("missing_validation_log_loss_gain")
    elif log_loss_gain + 1.0e-12 < thresholds["min_validation_log_loss_gain"]:
        failures.append(
            "validation_log_loss_gain_below_margin:"
            f"{log_loss_gain:.6g}<"
            f"{thresholds['min_validation_log_loss_gain']:.6g}"
        )
    if slope_delta > thresholds["max_slope_delta"] + 1.0e-12:
        failures.append(
            "slope_delta_above_cap:"
            f"{slope_delta:.6g}>{thresholds['max_slope_delta']:.6g}"
        )
    if abs_intercept > thresholds["max_abs_intercept"] + 1.0e-12:
        failures.append(
            "abs_intercept_above_cap:"
            f"{abs_intercept:.6g}>{thresholds['max_abs_intercept']:.6g}"
        )
    return {
        "passed": not failures,
        "failure_reasons": failures,
        "thresholds": thresholds,
        "metrics": {
            "validation_brier_gain": brier_gain,
            "validation_log_loss_gain": log_loss_gain,
            "slope_delta": float(slope_delta),
            "abs_intercept": float(abs_intercept),
            "candidate_selected": bool(calibration.selected),
        },
    }


def _validated_response_batch(
    batch: BetaResponseCalibrationBatch,
) -> BetaResponseCalibrationBatch:
    mean = _posterior_mean_array(batch.posterior)
    X = np.asarray(batch.X, dtype=float)
    Y = np.asarray(batch.Y, dtype=float)
    _validate_response_inputs(mean, X, Y)
    if np.any((Y != 0.0) & (Y != 1.0)):
        raise ValueError("probit transfer response calibration requires binary Y")
    return BetaResponseCalibrationBatch(
        posterior=batch.posterior,
        X=X,
        Y=Y,
        label=str(batch.label),
    )


def _score_response_batch(
    posterior: BetaPosterior,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    slope: float,
    intercept: float,
) -> dict[str, float]:
    probability = _probit_predictive_probability(
        posterior,
        X,
        slope=slope,
        intercept=intercept,
    )
    record = _response_score_record(probability, Y)
    record["n_observations"] = float(np.asarray(Y).size)
    return record


def _combined_response_score(
    batches: list[BetaResponseCalibrationBatch],
    *,
    slope: float,
    intercept: float,
) -> dict[str, float]:
    total_n = 0.0
    brier = 0.0
    log_loss = 0.0
    for batch in batches:
        record = _score_response_batch(
            batch.posterior,
            batch.X,
            batch.Y,
            slope=slope,
            intercept=intercept,
        )
        n = float(record["n_observations"])
        total_n += n
        brier += float(record["brier"]) * n
        log_loss += float(record["log_loss"]) * n
    if total_n <= 0.0:
        raise ValueError("response calibration batches must contain observations")
    return {
        "brier": float(brier / total_n),
        "log_loss": float(log_loss / total_n),
        "n_observations": float(total_n),
    }


def _predictive_mean_gate_record(
    calibration: BetaPredictiveMeanCalibration,
    batches: list[BetaResponseCalibrationBatch],
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    identity = _combined_response_score(
        batches,
        slope=1.0,
        intercept=0.0,
    )
    candidate = _combined_response_score(
        batches,
        slope=float(calibration.slope),
        intercept=float(calibration.intercept),
    )
    brier_ratio = candidate["brier"] / max(
        identity["brier"], np.finfo(float).eps
    )
    log_loss_ratio = candidate["log_loss"] / max(
        identity["log_loss"], np.finfo(float).eps
    )
    score_improvement = (
        identity["brier"]
        + identity["log_loss"]
        - candidate["brier"]
        - candidate["log_loss"]
    )
    failure_reasons = []
    if brier_ratio > thresholds["max_brier_ratio"] + 1.0e-12:
        failure_reasons.append(
            f"brier_ratio_above_limit:{brier_ratio:.6g}>"
            f"{thresholds['max_brier_ratio']:.6g}"
        )
    if log_loss_ratio > thresholds["max_log_loss_ratio"] + 1.0e-12:
        failure_reasons.append(
            f"log_loss_ratio_above_limit:{log_loss_ratio:.6g}>"
            f"{thresholds['max_log_loss_ratio']:.6g}"
        )
    if score_improvement + 1.0e-12 < thresholds["min_score_improvement"]:
        failure_reasons.append(
            f"score_improvement_below_margin:{score_improvement:.6g}<"
            f"{thresholds['min_score_improvement']:.6g}"
        )
    return {
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "brier_identity": float(identity["brier"]),
        "brier_candidate": float(candidate["brier"]),
        "brier_ratio": float(brier_ratio),
        "log_loss_identity": float(identity["log_loss"]),
        "log_loss_candidate": float(candidate["log_loss"]),
        "log_loss_ratio": float(log_loss_ratio),
        "score_improvement": float(score_improvement),
        "n_observations": int(identity["n_observations"]),
        "labels": [str(batch.label) for batch in batches],
    }


def _metric_gain(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _posterior_mean_array(posterior: BetaPosterior) -> np.ndarray:
    mean = posterior.mean
    if hasattr(mean, "numpy"):
        mean = mean.numpy()
    array = np.asarray(mean, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            "Beta posterior mean must have shape batch x covariates x species"
        )
    return array


def _validate_response_inputs(
    mean: np.ndarray,
    design: np.ndarray,
    response: np.ndarray,
) -> None:
    if design.ndim != 3:
        raise ValueError("X must have shape batch x sites x covariates")
    if response.ndim != 3:
        raise ValueError("Y must have shape batch x sites x species")
    if design.shape[0] != mean.shape[0] or design.shape[2] != mean.shape[1]:
        raise ValueError("X shape is incompatible with posterior mean")
    if response.shape != (mean.shape[0], design.shape[1], mean.shape[2]):
        raise ValueError("Y shape is incompatible with posterior mean and X")


def _candidate_pairs(
    slope_grid: np.ndarray,
    intercept_grid: np.ndarray,
) -> list[tuple[float, float]]:
    slopes = np.asarray(slope_grid, dtype=float).reshape(-1)
    intercepts = np.asarray(intercept_grid, dtype=float).reshape(-1)
    if slopes.size == 0 or intercepts.size == 0:
        raise ValueError("slope_grid and intercept_grid must be non-empty")
    if np.any(~np.isfinite(slopes)) or np.any(~np.isfinite(intercepts)):
        raise ValueError("slope_grid and intercept_grid must be finite")
    pairs = [
        (float(slope), float(intercept))
        for slope in slopes
        for intercept in intercepts
    ]
    if (1.0, 0.0) not in pairs:
        pairs.append((1.0, 0.0))
    return pairs


def _probit_predictive_probability(
    posterior: BetaPosterior,
    design: np.ndarray,
    *,
    slope: float,
    intercept: float,
) -> np.ndarray:
    mean = _posterior_mean_array(posterior) * float(slope) + float(intercept)
    design = np.asarray(design, dtype=float)
    linear_mean = np.einsum("bnk,bks->bns", design, mean)
    if posterior.scale_tril is None:
        scale = posterior.scale
        if hasattr(scale, "numpy"):
            scale = scale.numpy()
        scale = np.asarray(scale, dtype=float)
        variance = np.einsum("bnk,bks->bns", np.square(design), np.square(scale))
    else:
        scale_tril = posterior.scale_tril
        if hasattr(scale_tril, "numpy"):
            scale_tril = scale_tril.numpy()
        scale_tril = np.asarray(scale_tril, dtype=float)
        projected_scale = np.einsum("bnk,bskj->bnsj", design, scale_tril)
        variance = np.sum(np.square(projected_scale), axis=-1)
    return ndtr(linear_mean / np.sqrt(1.0 + np.maximum(variance, 0.0)))


def _response_score_record(
    probability: np.ndarray,
    response: np.ndarray,
) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    response = np.asarray(response, dtype=float)
    return {
        "brier": float(np.mean(np.square(probability - response))),
        "log_loss": float(
            -np.mean(
                response * np.log(probability)
                + (1.0 - response) * np.log1p(-probability)
            )
        ),
    }


def _fit_affine(mean: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    x = np.asarray(mean, dtype=float).reshape(-1)
    y = np.asarray(truth, dtype=float).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("mean and truth must contain the same non-empty values")
    x_centered = x - float(np.mean(x))
    variance = float(np.mean(np.square(x_centered)))
    if variance <= np.finfo(float).eps:
        return 1.0, float(np.mean(y) - np.mean(x))
    slope = float(np.mean(x_centered * (y - float(np.mean(y)))) / variance)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def _apply_affine(mean: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return np.asarray(mean, dtype=float) * float(slope) + float(intercept)


def _rmse(values: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values) - np.asarray(truth)))))


def _optional_float(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    return None if value is None else float(value)
