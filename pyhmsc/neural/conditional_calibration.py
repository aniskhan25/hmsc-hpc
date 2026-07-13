"""Conditional coefficient-scale calibration for Neural-HMSC posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
from scipy.special import ndtr
from scipy.stats import norm

from pyhmsc.neural.calibration import BetaScaleCalibration, fit_beta_scale_calibration
from pyhmsc.neural.posterior_heads import BetaPosterior


_RAW_FEATURE_NAMES = ("prevalence_logit", "log_design_information", "log_raw_scale")
_CONDITIONAL_METHODS = {
    "conditional_structured_scale",
    "conditional_rank_aware_scale",
    "conditional_rank_aware_anchor_scale",
}
_OOD_OBJECTIVES = {
    "none",
    "support_excess_rank_coverage",
    "support_effect_gated_rank_coverage",
}


@dataclass(frozen=True)
class ConditionalBetaOODCalibrationBatch:
    """Held-out OOD calibration data for coefficient-scale objectives."""

    posterior: BetaPosterior
    beta_true: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    label: str = "ood"
    weight: float = 1.0


@dataclass(frozen=True)
class ConditionalBetaScaleCalibration:
    """Serializable structured scale head fitted from simulation truth."""

    global_scale_multiplier: float
    normalization_multiplier: float
    feature_location: tuple[float, float, float]
    feature_scale: tuple[float, float, float]
    weights: tuple[float, ...]
    feature_names: tuple[str, ...]
    coefficient_names: tuple[str, ...]
    nominal_level: float
    uncalibrated_coverage: float
    calibrated_coverage: float
    n_observations: int
    distribution: str
    n_covariates: int
    n_species: int
    regularization: float
    epochs: int
    learning_rate: float
    scalar_nll: float
    conditional_nll: float
    scalar_rank_loss: float = 0.0
    conditional_rank_loss: float = 0.0
    prevalence_weights: tuple[float, float, float] = (4.0, 2.0, 1.0)
    prevalence_edges: tuple[float, float] = (0.1, 0.3)
    rank_penalty_weight: float = 0.02
    rank_mean_tolerance: float = 0.025
    rank_variance_tolerance: float = 0.015
    support_lower: tuple[float, float, float] = (-1e9, -1e9, -1e9)
    support_upper: tuple[float, float, float] = (1e9, 1e9, 1e9)
    support_precision: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    support_radius: float = 1e9
    support_quantile: float = 0.99
    fallback_strength: float = 2.0
    mean_magnitude_location: float = 0.0
    mean_magnitude_scale: float = 1.0
    mean_magnitude_lower: float = -1e9
    mean_magnitude_upper: float = 1e9
    ood_uncertainty_strength: float = 0.0
    ood_uncertainty_max_multiplier: float = 1.0
    ood_objective: str = "none"
    ood_objective_weight: float = 0.0
    ood_in_domain_gate_weight: float = 0.0
    ood_inflation_parameters: tuple[float, ...] | None = None
    ood_objective_domains: tuple[str, ...] = ()
    ood_objective_n_observations: int = 0
    ood_objective_loss: float = 0.0
    ood_objective_rank_loss: float = 0.0
    ood_in_domain_gate_loss: float = 0.0
    min_multiplier: float = 0.1
    max_multiplier: float = 20.0
    method: str = "conditional_rank_aware_anchor_scale"

    @property
    def scale_multiplier(self) -> float:
        """Return the normalization term used around the conditional head."""
        return self.normalization_multiplier

    def validate_domain(
        self,
        *,
        distribution: str | None = None,
        n_covariates: int | None = None,
        n_species: int | None = None,
        coefficient_names: Sequence[str] | None = None,
    ) -> None:
        """Raise when application data do not match the fitted domain."""
        if distribution is not None and str(distribution).lower() != self.distribution.lower():
            raise ValueError(
                "conditional calibration distribution mismatch: "
                f"expected {self.distribution!r}, got {distribution!r}"
            )
        if n_covariates is not None and int(n_covariates) != self.n_covariates:
            raise ValueError(
                "conditional calibration covariate dimension mismatch: "
                f"expected {self.n_covariates}, got {n_covariates}"
            )
        if n_species is not None and int(n_species) != self.n_species:
            raise ValueError(
                "conditional calibration species dimension mismatch: "
                f"expected {self.n_species}, got {n_species}"
            )
        if coefficient_names is not None:
            names = tuple(str(name) for name in coefficient_names)
            if names != self.coefficient_names:
                raise ValueError(
                    "conditional calibration coefficient names mismatch: "
                    f"expected {self.coefficient_names!r}, got {names!r}"
                )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable conditional-calibration metadata."""
        semantics_version = {
            "conditional_structured_scale": 3,
            "conditional_rank_aware_scale": 4,
            "conditional_rank_aware_anchor_scale": 5,
        }[self.method]
        if (
            self.method == "conditional_rank_aware_anchor_scale"
            and self.ood_uncertainty_strength > 0.0
            and self.ood_uncertainty_max_multiplier > 1.0
        ):
            semantics_version = 6
        if self.ood_inflation_parameters is not None:
            semantics_version = 7
            if len(self.ood_inflation_parameters) >= 7:
                semantics_version = 8
        ood_uncertainty = {
            "transform": "support_excess_exp",
            "strength": float(self.ood_uncertainty_strength),
            "max_multiplier": float(self.ood_uncertainty_max_multiplier),
        }
        if self.ood_inflation_parameters is not None:
            curve: dict[str, float]
            if len(self.ood_inflation_parameters) >= 7:
                if len(self.ood_inflation_parameters) >= 8:
                    (
                        offset,
                        support_linear,
                        support_quadratic,
                        effect_linear,
                        effect_quadratic,
                        effect_gate_intercept,
                        effect_gate_support_linear,
                        effect_gate_effect_linear,
                    ) = self.ood_inflation_parameters[:8]
                else:
                    (
                        offset,
                        support_linear,
                        support_quadratic,
                        effect_linear,
                        effect_quadratic,
                        effect_gate_intercept,
                        effect_gate_support_linear,
                    ) = self.ood_inflation_parameters[:7]
                    effect_gate_effect_linear = 0.0
                transform = "support_effect_gated_learned_softplus"
                curve = {
                    "offset": float(offset),
                    "support_linear": float(support_linear),
                    "support_quadratic": float(support_quadratic),
                    "effect_linear": float(effect_linear),
                    "effect_quadratic": float(effect_quadratic),
                    "effect_gate_intercept": float(effect_gate_intercept),
                    "effect_gate_support_linear": float(effect_gate_support_linear),
                    "effect_gate_effect_linear": float(effect_gate_effect_linear),
                }
            elif len(self.ood_inflation_parameters) >= 5:
                (
                    offset,
                    support_linear,
                    support_quadratic,
                    effect_linear,
                    effect_quadratic,
                ) = self.ood_inflation_parameters[:5]
                transform = "support_effect_learned_softplus"
                curve = {
                    "offset": float(offset),
                    "support_linear": float(support_linear),
                    "support_quadratic": float(support_quadratic),
                    "effect_linear": float(effect_linear),
                    "effect_quadratic": float(effect_quadratic),
                }
            else:
                offset, support_linear, support_quadratic = self.ood_inflation_parameters
                transform = "support_excess_learned_softplus"
                curve = {
                    "offset": float(offset),
                    "linear": float(support_linear),
                    "quadratic": float(support_quadratic),
                }
            ood_uncertainty.update(
                {
                    "transform": transform,
                    "curve": curve,
                }
            )
        return {
            "semantics_version": semantics_version,
            "method": self.method,
            "parameter": "Beta",
            "scale_multiplier": float(self.normalization_multiplier),
            "scale_multiplier_kind": "conditional_normalization",
            "global_scale_multiplier": float(self.global_scale_multiplier),
            "nominal_level": float(self.nominal_level),
            "uncalibrated_coverage": float(self.uncalibrated_coverage),
            "calibrated_coverage": float(self.calibrated_coverage),
            "n_observations": int(self.n_observations),
            "domain": {
                "distribution": self.distribution,
                "n_covariates": int(self.n_covariates),
                "n_species": int(self.n_species),
                "coefficient_names": list(self.coefficient_names),
            },
            "features": {
                "raw_names": list(_RAW_FEATURE_NAMES),
                "design_names": list(self.feature_names),
                "location": list(self.feature_location),
                "scale": list(self.feature_scale),
            },
            "weights": list(self.weights),
            "training": {
                "regularization": float(self.regularization),
                "epochs": int(self.epochs),
                "learning_rate": float(self.learning_rate),
                "scalar_nll": float(self.scalar_nll),
                "conditional_nll": float(self.conditional_nll),
                "scalar_rank_loss": float(self.scalar_rank_loss),
                "conditional_rank_loss": float(self.conditional_rank_loss),
            },
            "ood_objective": {
                "name": self.ood_objective,
                "weight": float(self.ood_objective_weight),
                "in_domain_gate_weight": float(self.ood_in_domain_gate_weight),
                "domains": list(self.ood_objective_domains),
                "n_observations": int(self.ood_objective_n_observations),
                "loss": float(self.ood_objective_loss),
                "rank_loss": float(self.ood_objective_rank_loss),
                "in_domain_gate_loss": float(self.ood_in_domain_gate_loss),
            },
            "rank_aware": {
                "prevalence_weights": list(self.prevalence_weights),
                "prevalence_edges": list(self.prevalence_edges),
                "penalty_weight": float(self.rank_penalty_weight),
                "mean_tolerance": float(self.rank_mean_tolerance),
                "variance_tolerance": float(self.rank_variance_tolerance),
            },
            "support": {
                "lower": list(self.support_lower),
                "upper": list(self.support_upper),
                "precision": [list(row) for row in self.support_precision],
                "radius": float(self.support_radius),
                "quantile": float(self.support_quantile),
                "fallback_strength": float(self.fallback_strength),
                "fallback_multiplier": float(self.global_scale_multiplier),
                "blend_space": "log_scale",
                "mean_magnitude": {
                    "transform": "log1p_abs",
                    "location": float(self.mean_magnitude_location),
                    "scale": float(self.mean_magnitude_scale),
                    "lower": float(self.mean_magnitude_lower),
                    "upper": float(self.mean_magnitude_upper),
                },
                "ood_uncertainty": {
                    **ood_uncertainty,
                },
            },
            "multiplier_bounds": [
                float(self.min_multiplier),
                float(self.max_multiplier),
            ],
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "ConditionalBetaScaleCalibration":
        """Reconstruct a conditional calibrator from stored metadata."""
        method = str(metadata.get("method"))
        if method not in _CONDITIONAL_METHODS:
            raise ValueError("metadata does not describe conditional structured scaling")
        domain = metadata["domain"]
        features = metadata["features"]
        training = metadata["training"]
        bounds = metadata.get("multiplier_bounds", (0.1, 20.0))
        raw_names = tuple(str(value) for value in features["raw_names"])
        if raw_names != _RAW_FEATURE_NAMES:
            raise ValueError("conditional calibration raw feature specification mismatch")
        feature_names = tuple(str(value) for value in features["design_names"])
        weights = tuple(float(value) for value in metadata["weights"])
        if len(weights) != len(feature_names):
            raise ValueError("conditional calibration weights do not match features")
        location = tuple(float(value) for value in features["location"])
        feature_scale = tuple(float(value) for value in features["scale"])
        if len(location) != 3 or len(feature_scale) != 3:
            raise ValueError("conditional calibration requires three raw features")
        rank_aware = metadata.get("rank_aware", {})
        support = metadata.get("support", {})
        support_lower = tuple(float(value) for value in support.get("lower", (-1e9,) * 3))
        support_upper = tuple(float(value) for value in support.get("upper", (1e9,) * 3))
        support_precision = tuple(
            tuple(float(value) for value in row)
            for row in support.get("precision", np.eye(3).tolist())
        )
        if (
            len(support_lower) != 3
            or len(support_upper) != 3
            or len(support_precision) != 3
            or any(len(row) != 3 for row in support_precision)
        ):
            raise ValueError("conditional calibration support must have three dimensions")
        mean_support = support.get("mean_magnitude", {})
        ood_uncertainty = support.get("ood_uncertainty", {})
        ood_objective = metadata.get("ood_objective", {})
        ood_curve = ood_uncertainty.get("curve")
        ood_inflation_parameters = None
        if isinstance(ood_curve, dict):
            if (
                "effect_gate_intercept" in ood_curve
                or "effect_gate_support_linear" in ood_curve
                or "effect_gate_effect_linear" in ood_curve
            ):
                ood_inflation_parameters = (
                    float(ood_curve["offset"]),
                    float(ood_curve["support_linear"]),
                    float(ood_curve["support_quadratic"]),
                    float(ood_curve["effect_linear"]),
                    float(ood_curve["effect_quadratic"]),
                    float(ood_curve["effect_gate_intercept"]),
                    float(ood_curve["effect_gate_support_linear"]),
                    float(ood_curve.get("effect_gate_effect_linear", 0.0)),
                )
            elif "effect_linear" in ood_curve or "effect_quadratic" in ood_curve:
                ood_inflation_parameters = (
                    float(ood_curve["offset"]),
                    float(ood_curve["support_linear"]),
                    float(ood_curve["support_quadratic"]),
                    float(ood_curve["effect_linear"]),
                    float(ood_curve["effect_quadratic"]),
                )
            else:
                ood_inflation_parameters = (
                    float(ood_curve["offset"]),
                    float(ood_curve["linear"]),
                    float(ood_curve["quadratic"]),
                )
        return cls(
            global_scale_multiplier=float(metadata["global_scale_multiplier"]),
            normalization_multiplier=float(metadata["scale_multiplier"]),
            feature_location=location,
            feature_scale=feature_scale,
            weights=weights,
            feature_names=feature_names,
            coefficient_names=tuple(str(value) for value in domain["coefficient_names"]),
            nominal_level=float(metadata["nominal_level"]),
            uncalibrated_coverage=float(metadata["uncalibrated_coverage"]),
            calibrated_coverage=float(metadata["calibrated_coverage"]),
            n_observations=int(metadata["n_observations"]),
            distribution=str(domain["distribution"]),
            n_covariates=int(domain["n_covariates"]),
            n_species=int(domain["n_species"]),
            regularization=float(training["regularization"]),
            epochs=int(training["epochs"]),
            learning_rate=float(training["learning_rate"]),
            scalar_nll=float(training["scalar_nll"]),
            conditional_nll=float(training["conditional_nll"]),
            scalar_rank_loss=float(training.get("scalar_rank_loss", 0.0)),
            conditional_rank_loss=float(training.get("conditional_rank_loss", 0.0)),
            prevalence_weights=tuple(
                float(value)
                for value in rank_aware.get("prevalence_weights", (1.0, 1.0, 1.0))
            ),
            prevalence_edges=tuple(
                float(value)
                for value in rank_aware.get("prevalence_edges", (0.1, 0.3))
            ),
            rank_penalty_weight=float(rank_aware.get("penalty_weight", 0.0)),
            rank_mean_tolerance=float(rank_aware.get("mean_tolerance", 0.025)),
            rank_variance_tolerance=float(
                rank_aware.get("variance_tolerance", 0.015)
            ),
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=float(support.get("radius", 1e9)),
            support_quantile=float(support.get("quantile", 1.0)),
            fallback_strength=float(support.get("fallback_strength", 0.0)),
            mean_magnitude_location=float(mean_support.get("location", 0.0)),
            mean_magnitude_scale=float(mean_support.get("scale", 1.0)),
            mean_magnitude_lower=float(mean_support.get("lower", -1e9)),
            mean_magnitude_upper=float(mean_support.get("upper", 1e9)),
            ood_uncertainty_strength=float(ood_uncertainty.get("strength", 0.0)),
            ood_uncertainty_max_multiplier=float(
                ood_uncertainty.get("max_multiplier", 1.0)
            ),
            ood_objective=str(ood_objective.get("name", "none")),
            ood_objective_weight=float(ood_objective.get("weight", 0.0)),
            ood_in_domain_gate_weight=float(
                ood_objective.get("in_domain_gate_weight", 0.0)
            ),
            ood_inflation_parameters=ood_inflation_parameters,
            ood_objective_domains=tuple(
                str(value) for value in ood_objective.get("domains", ())
            ),
            ood_objective_n_observations=int(
                ood_objective.get("n_observations", 0)
            ),
            ood_objective_loss=float(ood_objective.get("loss", 0.0)),
            ood_objective_rank_loss=float(ood_objective.get("rank_loss", 0.0)),
            ood_in_domain_gate_loss=float(
                ood_objective.get("in_domain_gate_loss", 0.0)
            ),
            min_multiplier=float(bounds[0]),
            max_multiplier=float(bounds[1]),
            method=method,
        )


def fit_conditional_beta_scale_calibration(
    posterior: BetaPosterior,
    beta_true: np.ndarray,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str,
    coefficient_names: Sequence[str] | None = None,
    baseline_calibration: BetaScaleCalibration | None = None,
    nominal_level: float = 0.95,
    species_mask: np.ndarray | None = None,
    regularization: float = 1e-3,
    epochs: int = 400,
    learning_rate: float = 0.03,
    prevalence_weights: tuple[float, float, float] = (4.0, 2.0, 1.0),
    prevalence_edges: tuple[float, float] = (0.1, 0.3),
    rank_penalty_weight: float = 0.02,
    rank_mean_tolerance: float = 0.025,
    rank_variance_tolerance: float = 0.015,
    support_quantile: float = 0.99,
    fallback_strength: float = 2.0,
    ood_uncertainty_strength: float = 0.75,
    ood_uncertainty_max_multiplier: float = 4.0,
    ood_calibration_batches: Sequence[ConditionalBetaOODCalibrationBatch] | None = None,
    ood_objective: str = "none",
    ood_objective_weight: float = 1.0,
    ood_in_domain_gate_weight: float = 10.0,
    ood_objective_epochs: int | None = None,
    support_ridge: float = 1e-4,
    min_multiplier: float = 0.1,
    max_multiplier: float = 20.0,
) -> ConditionalBetaScaleCalibration:
    """Fit a structured conditional scale head on simulated calibration truth.

    The head combines prevalence-weighted Gaussian log score with analytic SBC
    rank-moment penalties. A final scalar normalization restores nominal
    marginal coverage, while a feature-support gate falls back to the frozen
    scalar multiplier under covariate shift.
    """
    if not 0.0 < nominal_level < 1.0:
        raise ValueError("nominal_level must be between zero and one")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if len(prevalence_weights) != 3 or any(value <= 0.0 for value in prevalence_weights):
        raise ValueError("prevalence_weights must contain three positive values")
    low_prevalence, high_prevalence = (float(value) for value in prevalence_edges)
    if not 0.0 < low_prevalence < high_prevalence < 1.0:
        raise ValueError("prevalence_edges must be ordered values between zero and one")
    if rank_penalty_weight < 0.0:
        raise ValueError("rank_penalty_weight must be non-negative")
    if rank_mean_tolerance <= 0.0 or rank_variance_tolerance <= 0.0:
        raise ValueError("rank tolerances must be positive")
    if not 0.5 < support_quantile < 1.0:
        raise ValueError("support_quantile must be between 0.5 and 1")
    if fallback_strength < 0.0 or support_ridge <= 0.0:
        raise ValueError("fallback_strength must be non-negative and support_ridge positive")
    if ood_uncertainty_strength < 0.0 or ood_uncertainty_max_multiplier < 1.0:
        raise ValueError(
            "ood uncertainty strength must be non-negative and max multiplier at least one"
        )
    if ood_objective not in _OOD_OBJECTIVES:
        raise ValueError(f"unsupported OOD objective: {ood_objective!r}")
    if ood_objective_weight < 0.0 or ood_in_domain_gate_weight < 0.0:
        raise ValueError("OOD objective weights must be non-negative")
    if ood_objective_epochs is not None and ood_objective_epochs <= 0:
        raise ValueError("ood_objective_epochs must be positive")
    if not 0.0 < min_multiplier < max_multiplier:
        raise ValueError("multiplier bounds must be positive and ordered")

    mean, scale, design, response = _validated_arrays(posterior, X=X, Y=Y)
    truth = np.asarray(beta_true, dtype=float)
    if truth.shape != mean.shape or not np.all(np.isfinite(truth)):
        raise ValueError("beta_true must be finite and match the posterior shape")
    names = _coefficient_names(coefficient_names, mean.shape[1])
    mask = _coefficient_mask(mean.shape, species_mask)
    if not np.any(mask):
        raise ValueError("calibration mask selects no coefficients")

    baseline = baseline_calibration or fit_beta_scale_calibration(
        posterior,
        truth,
        nominal_level=nominal_level,
        distribution=distribution,
        species_mask=species_mask,
    )
    baseline.validate_domain(
        distribution=distribution,
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
    )
    if not np.isclose(baseline.nominal_level, nominal_level):
        raise ValueError("baseline calibration nominal level does not match")

    raw_features = _raw_features(
        mean=mean,
        scale=scale,
        X=design,
        Y=response,
        distribution=distribution,
    )
    selected_raw = raw_features[mask]
    location = np.mean(selected_raw, axis=0)
    feature_scale = np.std(selected_raw, axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    selected_standardized = (selected_raw - location) / feature_scale
    support_tail = (1.0 - support_quantile) / 2.0
    support_lower = np.quantile(selected_standardized, support_tail, axis=0)
    support_upper = np.quantile(selected_standardized, 1.0 - support_tail, axis=0)
    support_covariance = np.cov(selected_standardized, rowvar=False)
    support_precision = np.linalg.inv(
        support_covariance + float(support_ridge) * np.eye(3)
    )
    support_distance = np.sqrt(
        np.maximum(
            np.einsum(
                "ni,ij,nj->n",
                selected_standardized,
                support_precision,
                selected_standardized,
            ),
            0.0,
        )
    )
    support_radius = float(np.quantile(support_distance, support_quantile))
    selected_mean_magnitude = np.log1p(np.abs(mean))[mask]
    mean_magnitude_location = float(np.mean(selected_mean_magnitude))
    mean_magnitude_scale = float(np.std(selected_mean_magnitude))
    if mean_magnitude_scale <= 1e-8:
        mean_magnitude_scale = 1.0
    standardized_mean_magnitude = (
        selected_mean_magnitude - mean_magnitude_location
    ) / mean_magnitude_scale
    mean_magnitude_lower = float(
        np.quantile(standardized_mean_magnitude, support_tail)
    )
    mean_magnitude_upper = float(
        np.quantile(standardized_mean_magnitude, 1.0 - support_tail)
    )
    feature_design, feature_names = _structured_design(
        raw_features,
        location=location,
        scale=feature_scale,
        n_covariates=mean.shape[1],
    )

    signed_standardized_error = (truth - mean) / scale
    standardized_error = np.abs(signed_standardized_error)
    selected_design = feature_design[mask.reshape(-1)]
    selected_error = standardized_error[mask]
    selected_signed_error = signed_standardized_error[mask]
    prevalence = _prevalence(response)
    prevalence_by_coefficient = np.broadcast_to(prevalence[:, None, :], mean.shape)
    selected_prevalence = prevalence_by_coefficient[mask]
    observation_weights = _prevalence_observation_weights(
        selected_prevalence,
        prevalence_weights=prevalence_weights,
        prevalence_edges=prevalence_edges,
    )
    rank_groups = _prevalence_group_masks(
        selected_prevalence, prevalence_edges=prevalence_edges
    )
    weights = tf.Variable(
        np.zeros(selected_design.shape[1], dtype=np.float64), dtype=tf.float64
    )
    design_tensor = tf.constant(selected_design, dtype=tf.float64)
    error_tensor = tf.constant(selected_error, dtype=tf.float64)
    signed_error_tensor = tf.constant(selected_signed_error, dtype=tf.float64)
    observation_weight_tensor = tf.constant(observation_weights, dtype=tf.float64)
    rank_group_tensors = [tf.constant(group) for group in rank_groups]
    base_log_scale = tf.constant(np.log(baseline.scale_multiplier), dtype=tf.float64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            log_multiplier = tf.clip_by_value(
                base_log_scale + tf.linalg.matvec(design_tensor, weights),
                np.log(min_multiplier),
                np.log(max_multiplier),
            )
            coefficient_nll = (
                log_multiplier
                + 0.5 * tf.square(error_tensor) * tf.exp(-2.0 * log_multiplier)
            )
            nll = tf.reduce_sum(observation_weight_tensor * coefficient_nll) / tf.reduce_sum(
                observation_weight_tensor
            )
            rank_probability = _tf_normal_cdf(
                signed_error_tensor * tf.exp(-log_multiplier)
            )
            rank_loss = _tf_rank_moment_loss(
                rank_probability,
                rank_group_tensors,
                mean_tolerance=rank_mean_tolerance,
                variance_tolerance=rank_variance_tolerance,
            )
            penalty = tf.cast(regularization, tf.float64) * tf.reduce_mean(
                tf.square(weights)
            )
            loss = nll + tf.cast(rank_penalty_weight, tf.float64) * rank_loss + penalty
        gradient = tape.gradient(loss, weights)
        optimizer.apply_gradients([(gradient, weights)])

    fitted_weights = weights.numpy()
    adjustment = np.exp(
        np.clip(feature_design @ fitted_weights, -20.0, 20.0)
    ).reshape(mean.shape)
    support_trust = _support_trust(
        raw_features,
        location=location,
        scale=feature_scale,
        lower=support_lower,
        upper=support_upper,
        precision=support_precision,
        radius=support_radius,
        fallback_strength=fallback_strength,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
        mean_magnitude_lower=mean_magnitude_lower,
        mean_magnitude_upper=mean_magnitude_upper,
    )
    support_excess = _support_excess(
        raw_features,
        location=location,
        scale=feature_scale,
        lower=support_lower,
        upper=support_upper,
        precision=support_precision,
        radius=support_radius,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
        mean_magnitude_lower=mean_magnitude_lower,
        mean_magnitude_upper=mean_magnitude_upper,
    )
    effect_signal = _effect_size_signal(
        np.log1p(np.abs(mean)),
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
    )
    z_value = float(norm.ppf(0.5 + nominal_level / 2.0))
    normalization = _fit_coverage_normalization(
        standardized_error=standardized_error,
        adjustment=adjustment,
        trust=support_trust,
        support_excess=support_excess,
        effect_signal=effect_signal,
        global_multiplier=baseline.scale_multiplier,
        mask=mask,
        nominal_level=nominal_level,
        z_value=z_value,
        ood_uncertainty_strength=ood_uncertainty_strength,
        ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    ood_inflation_parameters = None
    ood_objective_loss = 0.0
    ood_objective_rank_loss = 0.0
    ood_in_domain_gate_loss = 0.0
    ood_objective_n_observations = 0
    ood_objective_domains: tuple[str, ...] = ()
    if ood_objective != "none":
        batches = tuple(ood_calibration_batches or ())
        if not batches:
            raise ValueError("OOD objective requires at least one OOD calibration batch")
        (
            ood_inflation_parameters,
            ood_objective_loss,
            ood_objective_rank_loss,
            ood_in_domain_gate_loss,
            ood_objective_n_observations,
            ood_objective_domains,
        ) = _fit_ood_inflation_parameters(
            batches,
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            in_domain_signed_error=selected_signed_error,
            in_domain_adjustment=adjustment[mask],
            in_domain_trust=support_trust[mask],
            in_domain_support_excess=support_excess[mask],
            in_domain_effect_signal=effect_signal[mask],
            in_domain_rank_groups=rank_groups,
            distribution=distribution,
            n_covariates=mean.shape[1],
            n_species=mean.shape[2],
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=ood_uncertainty_max_multiplier,
            objective_weight=ood_objective_weight,
            in_domain_gate_weight=ood_in_domain_gate_weight,
            epochs=ood_objective_epochs or max(50, epochs // 2),
            learning_rate=learning_rate,
            gate_effect_branch=ood_objective == "support_effect_gated_rank_coverage",
            prevalence_edges=(low_prevalence, high_prevalence),
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
        )
        normalization = _fit_coverage_normalization(
            standardized_error=standardized_error,
            adjustment=adjustment,
            trust=support_trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            global_multiplier=baseline.scale_multiplier,
            mask=mask,
            nominal_level=nominal_level,
            z_value=z_value,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
    multipliers = _blend_with_scalar_fallback(
        adjustment,
        support_trust,
        support_excess=support_excess,
        effect_signal=effect_signal,
        normalization=normalization,
        global_multiplier=baseline.scale_multiplier,
        ood_uncertainty_strength=ood_uncertainty_strength,
        ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
        ood_inflation_parameters=ood_inflation_parameters,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    uncalibrated_coverage = _coverage(mean, scale, truth, mask, z_value)
    calibrated_coverage = _coverage(
        mean, scale * multipliers, truth, mask, z_value
    )
    scalar_multiplier = np.full(selected_error.shape, baseline.scale_multiplier)
    conditional_multiplier = multipliers[mask]

    return ConditionalBetaScaleCalibration(
        global_scale_multiplier=float(baseline.scale_multiplier),
        normalization_multiplier=normalization,
        feature_location=tuple(float(value) for value in location),
        feature_scale=tuple(float(value) for value in feature_scale),
        weights=tuple(float(value) for value in fitted_weights),
        feature_names=feature_names,
        coefficient_names=names,
        nominal_level=float(nominal_level),
        uncalibrated_coverage=uncalibrated_coverage,
        calibrated_coverage=calibrated_coverage,
        n_observations=int(np.count_nonzero(mask)),
        distribution=str(distribution),
        n_covariates=int(mean.shape[1]),
        n_species=int(mean.shape[2]),
        regularization=float(regularization),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        scalar_nll=_scale_nll(selected_error, scalar_multiplier),
        conditional_nll=_scale_nll(selected_error, conditional_multiplier),
        scalar_rank_loss=_rank_moment_loss(
            selected_signed_error / scalar_multiplier,
            rank_groups,
            mean_tolerance=rank_mean_tolerance,
            variance_tolerance=rank_variance_tolerance,
        ),
        conditional_rank_loss=_rank_moment_loss(
            selected_signed_error / conditional_multiplier,
            rank_groups,
            mean_tolerance=rank_mean_tolerance,
            variance_tolerance=rank_variance_tolerance,
        ),
        prevalence_weights=tuple(float(value) for value in prevalence_weights),
        prevalence_edges=(low_prevalence, high_prevalence),
        rank_penalty_weight=float(rank_penalty_weight),
        rank_mean_tolerance=float(rank_mean_tolerance),
        rank_variance_tolerance=float(rank_variance_tolerance),
        support_lower=tuple(float(value) for value in support_lower),
        support_upper=tuple(float(value) for value in support_upper),
        support_precision=tuple(
            tuple(float(value) for value in row) for row in support_precision
        ),
        support_radius=support_radius,
        support_quantile=float(support_quantile),
        fallback_strength=float(fallback_strength),
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
        mean_magnitude_lower=mean_magnitude_lower,
        mean_magnitude_upper=mean_magnitude_upper,
        ood_uncertainty_strength=float(ood_uncertainty_strength),
        ood_uncertainty_max_multiplier=float(ood_uncertainty_max_multiplier),
        ood_objective=ood_objective,
        ood_objective_weight=float(ood_objective_weight if ood_objective != "none" else 0.0),
        ood_in_domain_gate_weight=float(
            ood_in_domain_gate_weight if ood_objective != "none" else 0.0
        ),
        ood_inflation_parameters=ood_inflation_parameters,
        ood_objective_domains=ood_objective_domains,
        ood_objective_n_observations=ood_objective_n_observations,
        ood_objective_loss=ood_objective_loss,
        ood_objective_rank_loss=ood_objective_rank_loss,
        ood_in_domain_gate_loss=ood_in_domain_gate_loss,
        min_multiplier=float(min_multiplier),
        max_multiplier=float(max_multiplier),
    )


def conditional_beta_scale_multipliers(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str | None = None,
    coefficient_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Predict one positive scale multiplier per Beta coefficient."""
    mean, scale, design, response = _validated_arrays(posterior, X=X, Y=Y)
    names = calibration.coefficient_names if coefficient_names is None else coefficient_names
    calibration.validate_domain(
        distribution=distribution or calibration.distribution,
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
        coefficient_names=names,
    )
    raw_features = _raw_features(
        mean=mean,
        scale=scale,
        X=design,
        Y=response,
        distribution=distribution or calibration.distribution,
    )
    feature_design, feature_names = _structured_design(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        n_covariates=mean.shape[1],
    )
    if feature_names != calibration.feature_names:
        raise ValueError("conditional calibration feature specification mismatch")
    adjustment = np.exp(
        np.clip(feature_design @ np.asarray(calibration.weights), -20.0, 20.0)
    ).reshape(mean.shape)
    trust = _support_trust(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        lower=np.asarray(calibration.support_lower),
        upper=np.asarray(calibration.support_upper),
        precision=np.asarray(calibration.support_precision),
        radius=calibration.support_radius,
        fallback_strength=calibration.fallback_strength,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
        mean_magnitude_lower=calibration.mean_magnitude_lower,
        mean_magnitude_upper=calibration.mean_magnitude_upper,
    )
    support_excess = _support_excess(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        lower=np.asarray(calibration.support_lower),
        upper=np.asarray(calibration.support_upper),
        precision=np.asarray(calibration.support_precision),
        radius=calibration.support_radius,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
        mean_magnitude_lower=calibration.mean_magnitude_lower,
        mean_magnitude_upper=calibration.mean_magnitude_upper,
    )
    effect_signal = _effect_size_signal(
        np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
    )
    return _blend_with_scalar_fallback(
        adjustment,
        trust,
        support_excess=support_excess,
        effect_signal=effect_signal,
        normalization=calibration.normalization_multiplier,
        global_multiplier=calibration.global_scale_multiplier,
        ood_uncertainty_strength=calibration.ood_uncertainty_strength,
        ood_uncertainty_max_multiplier=calibration.ood_uncertainty_max_multiplier,
        ood_inflation_parameters=calibration.ood_inflation_parameters,
        min_multiplier=calibration.min_multiplier,
        max_multiplier=calibration.max_multiplier,
    )


def conditional_beta_ood_uncertainty_inflation(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str | None = None,
    coefficient_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Return the bounded OOD uncertainty inflation applied to scale multipliers."""
    mean, _, design, response = _validated_arrays(posterior, X=X, Y=Y)
    names = calibration.coefficient_names if coefficient_names is None else coefficient_names
    calibration.validate_domain(
        distribution=distribution or calibration.distribution,
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
        coefficient_names=names,
    )
    raw_features = _raw_features(
        mean=mean,
        scale=_as_numpy(posterior.scale),
        X=design,
        Y=response,
        distribution=distribution or calibration.distribution,
    )
    support_excess = _support_excess(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        lower=np.asarray(calibration.support_lower),
        upper=np.asarray(calibration.support_upper),
        precision=np.asarray(calibration.support_precision),
        radius=calibration.support_radius,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
        mean_magnitude_lower=calibration.mean_magnitude_lower,
        mean_magnitude_upper=calibration.mean_magnitude_upper,
    )
    effect_signal = _effect_size_signal(
        np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
    )
    return _ood_uncertainty_inflation(
        support_excess,
        effect_signal=effect_signal,
        strength=calibration.ood_uncertainty_strength,
        max_multiplier=calibration.ood_uncertainty_max_multiplier,
        learned_parameters=calibration.ood_inflation_parameters,
    )


def conditional_beta_support_trust(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str | None = None,
    coefficient_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Return coefficient-level trust in the learned conditional adjustment."""
    mean, scale, design, response = _validated_arrays(posterior, X=X, Y=Y)
    names = calibration.coefficient_names if coefficient_names is None else coefficient_names
    calibration.validate_domain(
        distribution=distribution or calibration.distribution,
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
        coefficient_names=names,
    )
    raw_features = _raw_features(
        mean=mean,
        scale=scale,
        X=design,
        Y=response,
        distribution=distribution or calibration.distribution,
    )
    return _support_trust(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        lower=np.asarray(calibration.support_lower),
        upper=np.asarray(calibration.support_upper),
        precision=np.asarray(calibration.support_precision),
        radius=calibration.support_radius,
        fallback_strength=calibration.fallback_strength,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
        mean_magnitude_lower=calibration.mean_magnitude_lower,
        mean_magnitude_upper=calibration.mean_magnitude_upper,
    )


def conditional_beta_mean_support_diagnostics(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
) -> dict[str, float]:
    """Summarize posterior-mean magnitude relative to calibration support."""
    mean = _as_numpy(posterior.mean)
    if mean.ndim != 3:
        raise ValueError(
            "posterior mean must have batch x covariate x species shape"
        )
    calibration.validate_domain(
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
    )
    standardized = (
        np.log1p(np.abs(mean)) - calibration.mean_magnitude_location
    ) / calibration.mean_magnitude_scale
    outside = (standardized < calibration.mean_magnitude_lower) | (
        standardized > calibration.mean_magnitude_upper
    )
    return {
        "conditional_mean_magnitude_support_outside_fraction": float(
            np.mean(outside)
        ),
        "conditional_mean_magnitude_support_max_abs_z": float(
            np.max(np.abs(standardized))
        ),
    }


def conditional_beta_effect_size_signal(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
) -> np.ndarray:
    """Return the coefficient-level positive posterior-mean magnitude signal."""
    mean = _as_numpy(posterior.mean)
    if mean.ndim != 3:
        raise ValueError(
            "posterior mean must have batch x covariate x species shape"
        )
    calibration.validate_domain(
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
    )
    return _effect_size_signal(
        np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
    )


def apply_conditional_beta_scale_calibration(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str | None = None,
    coefficient_names: Sequence[str] | None = None,
) -> BetaPosterior:
    """Apply coefficient-specific scaling while preserving posterior means."""
    multipliers = conditional_beta_scale_multipliers(
        posterior,
        calibration,
        X=X,
        Y=Y,
        distribution=distribution,
        coefficient_names=coefficient_names,
    )
    mean = tf.convert_to_tensor(posterior.mean)
    multiplier_tensor = tf.cast(multipliers, mean.dtype)
    if posterior.scale_tril is None:
        return BetaPosterior(
            mean=mean,
            scale=tf.convert_to_tensor(posterior.scale) * multiplier_tensor,
        )

    scale_tril = tf.convert_to_tensor(posterior.scale_tril)
    per_species = tf.transpose(multiplier_tensor, [0, 2, 1])
    calibrated_tril = scale_tril * per_species[..., :, None]
    marginal = tf.sqrt(tf.reduce_sum(tf.square(calibrated_tril), axis=-1))
    return BetaPosterior(
        mean=mean,
        scale=tf.transpose(marginal, [0, 2, 1]),
        scale_tril=calibrated_tril,
    )


def _validated_arrays(
    posterior: BetaPosterior, *, X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = _as_numpy(posterior.mean)
    scale = _as_numpy(posterior.scale)
    design = np.asarray(X, dtype=float)
    response = np.asarray(Y, dtype=float)
    if mean.ndim != 3 or scale.shape != mean.shape:
        raise ValueError("posterior mean and scale must have batch x covariate x species shape")
    if (
        design.ndim != 3
        or design.shape[0] != mean.shape[0]
        or design.shape[2] != mean.shape[1]
    ):
        raise ValueError("X must have shape batch x sites x covariates")
    if response.ndim != 3 or response.shape != (
        mean.shape[0],
        design.shape[1],
        mean.shape[2],
    ):
        raise ValueError("Y must have shape batch x sites x species")
    if np.any(scale <= 0.0):
        raise ValueError("posterior scales must be positive")
    if not all(np.all(np.isfinite(value)) for value in (mean, scale, design, response)):
        raise ValueError("posterior, X, and Y values must be finite")
    return mean, scale, design, response


def _raw_features(
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str,
) -> np.ndarray:
    prevalence = _prevalence(Y)
    epsilon = 0.5 / float(X.shape[1] + 1)
    prevalence = np.clip(prevalence, epsilon, 1.0 - epsilon)
    prevalence_logit = np.log(prevalence / (1.0 - prevalence))
    prevalence_feature = np.broadcast_to(prevalence_logit[:, None, :], mean.shape)
    information = _expected_design_information(mean, X, distribution=distribution)
    return np.stack(
        [
            prevalence_feature,
            np.log(np.maximum(information, 1e-12)),
            np.log(scale),
        ],
        axis=-1,
    )


def _prevalence(Y: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(Y) > 0.0, axis=1)


def _prevalence_observation_weights(
    prevalence: np.ndarray,
    *,
    prevalence_weights: tuple[float, float, float],
    prevalence_edges: tuple[float, float],
) -> np.ndarray:
    low, high = prevalence_edges
    rare, intermediate, common = prevalence_weights
    return np.where(
        prevalence <= low,
        rare,
        np.where(prevalence <= high, intermediate, common),
    ).astype(float)


def _prevalence_group_masks(
    prevalence: np.ndarray, *, prevalence_edges: tuple[float, float]
) -> list[np.ndarray]:
    low, high = prevalence_edges
    candidates = [
        np.ones(prevalence.shape, dtype=bool),
        prevalence <= low,
        (prevalence > low) & (prevalence <= high),
        prevalence > high,
    ]
    return [mask for mask in candidates if np.count_nonzero(mask) >= 2]


def _expected_design_information(
    mean: np.ndarray, X: np.ndarray, *, distribution: str
) -> np.ndarray:
    linear = np.einsum("bnk,bks->bns", X, mean)
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
            f"unsupported distribution for conditional calibration: {distribution!r}"
        )
    return np.einsum("bnk,bns->bks", np.square(X), weight)


def _structured_design(
    raw_features: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
    n_covariates: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    standardized = (raw_features - location) / scale
    flattened = standardized.reshape(-1, standardized.shape[-1])
    columns = []
    names = []
    for feature_index, feature_name in enumerate(_RAW_FEATURE_NAMES):
        values = flattened[:, feature_index]
        columns.extend([values, np.maximum(values, 0.0)])
        names.extend([feature_name, f"{feature_name}_positive_hinge"])

    shape = raw_features.shape[:3]
    coefficient_index = np.broadcast_to(
        np.arange(n_covariates)[None, :, None], shape
    ).reshape(-1)
    centered_identity = np.eye(n_covariates)[coefficient_index] - 1.0 / n_covariates
    prevalence = flattened[:, 0]
    for index in range(n_covariates):
        columns.append(centered_identity[:, index])
        names.append(f"coefficient_{index}")
    for index in range(n_covariates):
        columns.append(centered_identity[:, index] * prevalence)
        names.append(f"prevalence_by_coefficient_{index}")
    return np.column_stack(columns), tuple(names)


def _support_trust(
    raw_features: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    precision: np.ndarray,
    radius: float,
    fallback_strength: float,
    mean_magnitude: np.ndarray,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
) -> np.ndarray:
    if fallback_strength <= 0.0:
        return np.ones(raw_features.shape[:3], dtype=float)
    total_excess = _support_excess(
        raw_features,
        location=location,
        scale=scale,
        lower=lower,
        upper=upper,
        precision=precision,
        radius=radius,
        mean_magnitude=mean_magnitude,
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
        mean_magnitude_lower=mean_magnitude_lower,
        mean_magnitude_upper=mean_magnitude_upper,
    )
    return np.exp(-float(fallback_strength) * np.square(total_excess))


def _support_excess(
    raw_features: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    precision: np.ndarray,
    radius: float,
    mean_magnitude: np.ndarray,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
) -> np.ndarray:
    standardized = (raw_features - location) / scale
    lower_excess = np.maximum(lower - standardized, 0.0)
    upper_excess = np.maximum(standardized - upper, 0.0)
    box_excess = np.sqrt(np.sum(np.square(lower_excess + upper_excess), axis=-1))
    distance = np.sqrt(
        np.maximum(
            np.einsum(
                "...i,ij,...j->...",
                standardized,
                precision,
                standardized,
            ),
            0.0,
        )
    )
    radial_excess = np.maximum(distance - float(radius), 0.0)
    standardized_mean = (
        mean_magnitude - float(mean_magnitude_location)
    ) / float(mean_magnitude_scale)
    mean_excess = np.maximum(
        float(mean_magnitude_lower) - standardized_mean, 0.0
    ) + np.maximum(standardized_mean - float(mean_magnitude_upper), 0.0)
    total_excess = np.sqrt(
        np.square(box_excess) + np.square(radial_excess) + np.square(mean_excess)
    )
    return total_excess


def _effect_size_signal(
    mean_magnitude: np.ndarray,
    *,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
) -> np.ndarray:
    """Return positive standardized posterior-mean magnitude."""
    standardized = (
        mean_magnitude - float(mean_magnitude_location)
    ) / float(mean_magnitude_scale)
    return np.maximum(standardized, 0.0)


def _blend_with_scalar_fallback(
    adjustment: np.ndarray,
    trust: np.ndarray,
    *,
    support_excess: np.ndarray,
    effect_signal: np.ndarray | None = None,
    normalization: float,
    global_multiplier: float,
    ood_uncertainty_strength: float,
    ood_uncertainty_max_multiplier: float,
    min_multiplier: float,
    max_multiplier: float,
    ood_inflation_parameters: tuple[float, ...] | None = None,
) -> np.ndarray:
    conditional = np.clip(
        float(normalization) * adjustment,
        min_multiplier,
        max_multiplier,
    )
    log_multiplier = (
        trust * np.log(conditional)
        + (1.0 - trust) * np.log(float(global_multiplier))
    )
    multiplier = np.exp(log_multiplier) * _ood_uncertainty_inflation(
        support_excess,
        effect_signal=effect_signal,
        strength=ood_uncertainty_strength,
        max_multiplier=ood_uncertainty_max_multiplier,
        learned_parameters=ood_inflation_parameters,
    )
    return np.clip(multiplier, min_multiplier, max_multiplier)


def _ood_uncertainty_inflation(
    support_excess: np.ndarray,
    *,
    effect_signal: np.ndarray | None = None,
    strength: float,
    max_multiplier: float,
    learned_parameters: tuple[float, ...] | None = None,
) -> np.ndarray:
    if max_multiplier <= 1.0:
        return np.ones_like(support_excess, dtype=float)
    if learned_parameters is not None:
        parameters = tuple(float(value) for value in learned_parameters)
        effect = (
            np.zeros_like(support_excess, dtype=float)
            if effect_signal is None
            else np.asarray(effect_signal, dtype=float)
        )
        log_inflation = _learned_ood_log_inflation_numpy(
            support_excess,
            effect_signal=effect,
            parameters=parameters,
            max_multiplier=max_multiplier,
        )
        return np.exp(log_inflation)
    if strength <= 0.0:
        return np.ones_like(support_excess, dtype=float)
    log_inflation = np.minimum(
        float(strength) * np.square(support_excess),
        np.log(float(max_multiplier)),
    )
    return np.exp(log_inflation)


def _fit_coverage_normalization(
    *,
    standardized_error: np.ndarray,
    adjustment: np.ndarray,
    trust: np.ndarray,
    support_excess: np.ndarray,
    effect_signal: np.ndarray | None = None,
    global_multiplier: float,
    mask: np.ndarray,
    nominal_level: float,
    z_value: float,
    ood_uncertainty_strength: float,
    ood_uncertainty_max_multiplier: float,
    min_multiplier: float,
    max_multiplier: float,
    ood_inflation_parameters: tuple[float, ...] | None = None,
) -> float:
    lower = float(min_multiplier)
    upper = float(max_multiplier)
    for _ in range(64):
        midpoint = float(np.sqrt(lower * upper))
        multiplier = _blend_with_scalar_fallback(
            adjustment,
            trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            normalization=midpoint,
            global_multiplier=global_multiplier,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        coverage = float(
            np.mean(standardized_error[mask] <= z_value * multiplier[mask])
        )
        if coverage < nominal_level:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def _fit_ood_inflation_parameters(
    batches: Sequence[ConditionalBetaOODCalibrationBatch],
    *,
    location: np.ndarray,
    feature_scale: np.ndarray,
    feature_names: tuple[str, ...],
    support_lower: np.ndarray,
    support_upper: np.ndarray,
    support_precision: np.ndarray,
    support_radius: float,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
    normalization: float,
    global_multiplier: float,
    fallback_strength: float,
    fitted_weights: np.ndarray,
    in_domain_signed_error: np.ndarray,
    in_domain_adjustment: np.ndarray,
    in_domain_trust: np.ndarray,
    in_domain_support_excess: np.ndarray,
    in_domain_effect_signal: np.ndarray,
    in_domain_rank_groups: list[np.ndarray],
    distribution: str,
    n_covariates: int,
    n_species: int,
    min_multiplier: float,
    max_multiplier: float,
    max_ood_multiplier: float,
    objective_weight: float,
    in_domain_gate_weight: float,
    epochs: int,
    learning_rate: float,
    gate_effect_branch: bool,
    prevalence_edges: tuple[float, float],
    rank_mean_tolerance: float,
    rank_variance_tolerance: float,
    nominal_level: float,
    z_value: float,
) -> tuple[tuple[float, ...], float, float, float, int, tuple[str, ...]]:
    """Fit a learned support-excess inflation curve from held-out OOD batches."""
    if max_ood_multiplier <= 1.0:
        raise ValueError("learned OOD inflation requires max multiplier greater than one")

    domain_arrays = []
    labels = []
    n_observations = 0
    for batch in batches:
        if batch.weight <= 0.0:
            raise ValueError("OOD calibration batch weights must be positive")
        mean, scale, design, response = _validated_arrays(
            batch.posterior, X=batch.X, Y=batch.Y
        )
        if mean.shape[1] != n_covariates or mean.shape[2] != n_species:
            raise ValueError("OOD calibration batch domain does not match calibration")
        truth = np.asarray(batch.beta_true, dtype=float)
        if truth.shape != mean.shape or not np.all(np.isfinite(truth)):
            raise ValueError("OOD beta_true must be finite and match posterior shape")
        raw_features = _raw_features(
            mean=mean,
            scale=scale,
            X=design,
            Y=response,
            distribution=distribution,
        )
        design_matrix, names = _structured_design(
            raw_features,
            location=location,
            scale=feature_scale,
            n_covariates=n_covariates,
        )
        if names != feature_names:
            raise ValueError("OOD calibration feature specification mismatch")
        adjustment = np.exp(
            np.clip(design_matrix @ fitted_weights, -20.0, 20.0)
        ).reshape(mean.shape)
        trust = _support_trust(
            raw_features,
            location=location,
            scale=feature_scale,
            lower=support_lower,
            upper=support_upper,
            precision=support_precision,
            radius=support_radius,
            fallback_strength=fallback_strength,
            mean_magnitude=np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
        )
        support_excess = _support_excess(
            raw_features,
            location=location,
            scale=feature_scale,
            lower=support_lower,
            upper=support_upper,
            precision=support_precision,
            radius=support_radius,
            mean_magnitude=np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
        )
        effect_signal = _effect_size_signal(
            np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
        )
        base_multiplier = _blend_with_scalar_fallback(
            adjustment,
            trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            normalization=normalization,
            global_multiplier=global_multiplier,
            ood_uncertainty_strength=0.0,
            ood_uncertainty_max_multiplier=1.0,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        signed_error = (truth - mean) / scale
        prevalence = _prevalence(response)
        prevalence_by_coefficient = np.broadcast_to(prevalence[:, None, :], mean.shape)
        rank_groups = _prevalence_group_masks(
            prevalence_by_coefficient.reshape(-1),
            prevalence_edges=prevalence_edges,
        )
        domain_arrays.append(
            {
                "signed_error": signed_error.reshape(-1),
                "base_multiplier": base_multiplier.reshape(-1),
                "support_excess": support_excess.reshape(-1),
                "effect_signal": effect_signal.reshape(-1),
                "rank_groups": rank_groups,
                "weight": float(batch.weight),
            }
        )
        labels.append(str(batch.label))
        n_observations += int(signed_error.size)

    in_domain_base_multiplier = _blend_with_scalar_fallback(
        in_domain_adjustment,
        in_domain_trust,
        support_excess=in_domain_support_excess,
        effect_signal=in_domain_effect_signal,
        normalization=normalization,
        global_multiplier=global_multiplier,
        ood_uncertainty_strength=0.0,
        ood_uncertainty_max_multiplier=1.0,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )

    offset = tf.Variable(-4.0, dtype=tf.float64)
    raw_support_linear = tf.Variable(_softplus_inverse(1e-3), dtype=tf.float64)
    raw_support_quadratic = tf.Variable(_softplus_inverse(0.75), dtype=tf.float64)
    raw_effect_linear = tf.Variable(_softplus_inverse(1e-3), dtype=tf.float64)
    raw_effect_quadratic = tf.Variable(_softplus_inverse(0.1), dtype=tf.float64)
    effect_gate_intercept = tf.Variable(-4.0, dtype=tf.float64)
    raw_effect_gate_support_linear = tf.Variable(
        _softplus_inverse(2.0), dtype=tf.float64
    )
    raw_effect_gate_effect_linear = tf.Variable(
        _softplus_inverse(4.0), dtype=tf.float64
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    expected_rank_variance = tf.constant(1.0 / 12.0, dtype=tf.float64)
    target_coverage = tf.constant(float(nominal_level), dtype=tf.float64)

    tf_domains = []
    for arrays in domain_arrays:
        tf_domains.append(
            {
                "signed_error": tf.constant(arrays["signed_error"], dtype=tf.float64),
                "base_multiplier": tf.constant(
                    arrays["base_multiplier"], dtype=tf.float64
                ),
                "support_excess": tf.constant(
                    arrays["support_excess"], dtype=tf.float64
                ),
                "effect_signal": tf.constant(
                    arrays["effect_signal"], dtype=tf.float64
                ),
                "rank_groups": [
                    tf.constant(group, dtype=tf.bool) for group in arrays["rank_groups"]
                ],
                "weight": tf.constant(float(arrays["weight"]), dtype=tf.float64),
            }
        )
    in_domain = {
        "signed_error": tf.constant(in_domain_signed_error, dtype=tf.float64),
        "base_multiplier": tf.constant(in_domain_base_multiplier, dtype=tf.float64),
        "support_excess": tf.constant(in_domain_support_excess, dtype=tf.float64),
        "effect_signal": tf.constant(in_domain_effect_signal, dtype=tf.float64),
        "rank_groups": [
            tf.constant(group, dtype=tf.bool) for group in in_domain_rank_groups
        ],
    }

    def log_inflation_for(arrays: dict[str, Any]) -> tf.Tensor:
        support_linear = tf.nn.softplus(raw_support_linear)
        support_quadratic = tf.nn.softplus(raw_support_quadratic)
        effect_linear = tf.nn.softplus(raw_effect_linear)
        effect_quadratic = tf.nn.softplus(raw_effect_quadratic)
        effect_gate_support_linear = tf.nn.softplus(
            raw_effect_gate_support_linear
        )
        effect_gate_effect_linear = tf.nn.softplus(raw_effect_gate_effect_linear)
        return _tf_learned_ood_log_inflation(
            arrays["support_excess"],
            effect_signal=arrays["effect_signal"],
            offset=offset,
            support_linear=support_linear,
            support_quadratic=support_quadratic,
            effect_linear=effect_linear,
            effect_quadratic=effect_quadratic,
            effect_gate_intercept=(
                effect_gate_intercept if gate_effect_branch else None
            ),
            effect_gate_support_linear=(
                effect_gate_support_linear if gate_effect_branch else None
            ),
            effect_gate_effect_linear=(
                effect_gate_effect_linear if gate_effect_branch else None
            ),
            max_multiplier=max_ood_multiplier,
        )

    def total_multiplier(arrays: dict[str, Any]) -> tf.Tensor:
        log_inflation = log_inflation_for(arrays)
        return tf.clip_by_value(
            arrays["base_multiplier"] * tf.exp(log_inflation),
            float(min_multiplier),
            float(max_multiplier),
        )

    last_ood_loss = tf.constant(0.0, dtype=tf.float64)
    last_rank_loss = tf.constant(0.0, dtype=tf.float64)
    last_gate_loss = tf.constant(0.0, dtype=tf.float64)
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            ood_losses = []
            rank_losses = []
            for arrays in tf_domains:
                multiplier = total_multiplier(arrays)
                signed_error = arrays["signed_error"]
                nll = tf.reduce_mean(
                    tf.math.log(multiplier)
                    + 0.5 * tf.square(signed_error / multiplier)
                )
                rank_probability = _tf_normal_cdf(signed_error / multiplier)
                rank_loss = _tf_rank_moment_loss(
                    rank_probability,
                    arrays["rank_groups"],
                    mean_tolerance=rank_mean_tolerance,
                    variance_tolerance=rank_variance_tolerance,
                )
                coverage = tf.reduce_mean(
                    tf.cast(
                        tf.abs(signed_error) <= float(z_value) * multiplier,
                        tf.float64,
                    )
                )
                coverage_loss = tf.square(
                    tf.nn.relu((target_coverage - coverage) / 0.05)
                )
                ood_losses.append(arrays["weight"] * (nll + rank_loss + coverage_loss))
                rank_losses.append(rank_loss)
            ood_loss = tf.reduce_mean(tf.stack(ood_losses))
            ood_rank_loss = tf.reduce_mean(tf.stack(rank_losses))

            in_multiplier = total_multiplier(in_domain)
            in_rank_probability = _tf_normal_cdf(
                in_domain["signed_error"] / in_multiplier
            )
            in_rank_losses = []
            for group in in_domain["rank_groups"]:
                selected = tf.boolean_mask(in_rank_probability, group)
                rank_mean = tf.reduce_mean(selected)
                rank_variance = tf.reduce_mean(tf.square(selected - rank_mean))
                in_rank_losses.append(
                    tf.square(
                        tf.nn.relu(
                            tf.abs(rank_mean - 0.5) / rank_mean_tolerance - 1.0
                        )
                    )
                    + tf.square(
                        tf.nn.relu(
                            tf.abs(rank_variance - expected_rank_variance)
                            / rank_variance_tolerance
                            - 1.0
                        )
                    )
                )
            in_coverage = tf.reduce_mean(
                tf.cast(
                    tf.abs(in_domain["signed_error"]) <= float(z_value) * in_multiplier,
                    tf.float64,
                )
            )
            gate_loss = tf.reduce_mean(tf.stack(in_rank_losses)) + tf.square(
                tf.nn.relu((0.90 - in_coverage) / 0.05)
            )
            if gate_effect_branch:
                in_extra_log_inflation = log_inflation_for(in_domain)
                gate_loss = gate_loss + 0.05 * tf.reduce_mean(
                    tf.square(
                        tf.nn.relu(
                            (
                                in_extra_log_inflation
                                - tf.math.log(tf.constant(1.05, dtype=tf.float64))
                            )
                            / tf.math.log(tf.constant(1.25, dtype=tf.float64))
                        )
                    )
                )
            loss = (
                float(objective_weight) * ood_loss
                + float(in_domain_gate_weight) * gate_loss
            )
        variables = [
            offset,
            raw_support_linear,
            raw_support_quadratic,
            raw_effect_linear,
            raw_effect_quadratic,
        ]
        if gate_effect_branch:
            variables.extend(
                [
                    effect_gate_intercept,
                    raw_effect_gate_support_linear,
                    raw_effect_gate_effect_linear,
                ]
            )
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(
            zip(gradients, variables)
        )
        last_ood_loss = ood_loss
        last_rank_loss = ood_rank_loss
        last_gate_loss = gate_loss

    learned_values = [
        float(offset.numpy()),
        float(tf.nn.softplus(raw_support_linear).numpy()),
        float(tf.nn.softplus(raw_support_quadratic).numpy()),
        float(tf.nn.softplus(raw_effect_linear).numpy()),
        float(tf.nn.softplus(raw_effect_quadratic).numpy()),
    ]
    if gate_effect_branch:
        learned_values.extend(
            [
                float(effect_gate_intercept.numpy()),
                float(tf.nn.softplus(raw_effect_gate_support_linear).numpy()),
                float(tf.nn.softplus(raw_effect_gate_effect_linear).numpy()),
            ]
        )
    learned = tuple(learned_values)
    return (
        learned,
        float(last_ood_loss.numpy()),
        float(last_rank_loss.numpy()),
        float(last_gate_loss.numpy()),
        n_observations,
        tuple(labels),
    )


def _softplus_inverse(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus inverse requires a positive value")
    return float(np.log(np.expm1(value)))


def _learned_ood_log_inflation_numpy(
    support_excess: np.ndarray,
    *,
    effect_signal: np.ndarray,
    parameters: tuple[float, ...],
    max_multiplier: float,
) -> np.ndarray:
    if len(parameters) >= 7:
        if len(parameters) >= 8:
            (
                offset,
                support_linear,
                support_quadratic,
                effect_linear,
                effect_quadratic,
                effect_gate_intercept,
                effect_gate_support_linear,
                effect_gate_effect_linear,
            ) = parameters[:8]
        else:
            (
                offset,
                support_linear,
                support_quadratic,
                effect_linear,
                effect_quadratic,
                effect_gate_intercept,
                effect_gate_support_linear,
            ) = parameters[:7]
            effect_gate_effect_linear = 0.0
        effect_gate = _sigmoid_numpy(
            float(effect_gate_intercept)
            + float(effect_gate_support_linear) * support_excess
            + float(effect_gate_effect_linear) * effect_signal
        )
    elif len(parameters) >= 5:
        offset, support_linear, support_quadratic, effect_linear, effect_quadratic = (
            parameters[:5]
        )
        effect_gate = 1.0
    elif len(parameters) == 3:
        offset, support_linear, support_quadratic = parameters
        effect_linear = 0.0
        effect_quadratic = 0.0
        effect_gate = 1.0
    else:
        raise ValueError(
            "learned OOD inflation curve requires three, five, or seven parameters"
        )
    raw = (
        float(offset)
        + float(support_linear) * support_excess
        + float(support_quadratic) * np.square(support_excess)
        + effect_gate
        * (
            float(effect_linear) * effect_signal
            + float(effect_quadratic) * np.square(effect_signal)
        )
    )
    baseline = np.logaddexp(0.0, float(offset))
    log_inflation = np.logaddexp(0.0, raw) - baseline
    return np.clip(log_inflation, 0.0, np.log(float(max_multiplier)))


def _sigmoid_numpy(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def _tf_learned_ood_log_inflation(
    support_excess: tf.Tensor,
    *,
    effect_signal: tf.Tensor,
    offset: tf.Tensor,
    support_linear: tf.Tensor,
    support_quadratic: tf.Tensor,
    effect_linear: tf.Tensor,
    effect_quadratic: tf.Tensor,
    effect_gate_intercept: tf.Tensor | None = None,
    effect_gate_support_linear: tf.Tensor | None = None,
    effect_gate_effect_linear: tf.Tensor | None = None,
    max_multiplier: float,
) -> tf.Tensor:
    if (
        effect_gate_intercept is None
        or effect_gate_support_linear is None
        or effect_gate_effect_linear is None
    ):
        effect_gate = tf.constant(1.0, dtype=tf.float64)
    else:
        effect_gate = tf.sigmoid(
            effect_gate_intercept
            + effect_gate_support_linear * support_excess
            + effect_gate_effect_linear * effect_signal
        )
    raw = (
        offset
        + support_linear * support_excess
        + support_quadratic * tf.square(support_excess)
        + effect_gate
        * (effect_linear * effect_signal + effect_quadratic * tf.square(effect_signal))
    )
    baseline = tf.nn.softplus(offset)
    log_inflation = tf.nn.softplus(raw) - baseline
    return tf.clip_by_value(log_inflation, 0.0, np.log(float(max_multiplier)))


def _tf_normal_cdf(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (1.0 + tf.math.erf(value / tf.sqrt(tf.constant(2.0, tf.float64))))


def _tf_rank_moment_loss(
    rank_probability: tf.Tensor,
    groups: list[tf.Tensor],
    *,
    mean_tolerance: float,
    variance_tolerance: float,
) -> tf.Tensor:
    losses = []
    expected_variance = tf.constant(1.0 / 12.0, dtype=tf.float64)
    for group in groups:
        selected = tf.boolean_mask(rank_probability, group)
        rank_mean = tf.reduce_mean(selected)
        rank_variance = tf.reduce_mean(tf.square(selected - rank_mean))
        losses.append(
            tf.square((rank_mean - 0.5) / mean_tolerance)
            + tf.square((rank_variance - expected_variance) / variance_tolerance)
        )
    return tf.reduce_mean(tf.stack(losses))


def _rank_moment_loss(
    signed_standardized_error: np.ndarray,
    groups: list[np.ndarray],
    *,
    mean_tolerance: float,
    variance_tolerance: float,
) -> float:
    ranks = ndtr(signed_standardized_error)
    losses = []
    for group in groups:
        selected = ranks[group]
        losses.append(
            ((float(np.mean(selected)) - 0.5) / mean_tolerance) ** 2
            + ((float(np.var(selected)) - 1.0 / 12.0) / variance_tolerance) ** 2
        )
    return float(np.mean(losses))


def _coefficient_names(
    names: Sequence[str] | None, n_covariates: int
) -> tuple[str, ...]:
    result = (
        tuple(str(name) for name in names)
        if names is not None
        else tuple(f"coefficient_{index}" for index in range(n_covariates))
    )
    if len(result) != n_covariates:
        raise ValueError("coefficient_names length must match posterior covariates")
    return result


def _coefficient_mask(
    shape: tuple[int, ...], species_mask: np.ndarray | None
) -> np.ndarray:
    if species_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(species_mask, dtype=bool)
    if mask.shape != (shape[0], shape[2]):
        raise ValueError("species_mask must have shape batch x species")
    return np.broadcast_to(mask[:, None, :], shape)


def _coverage(
    mean: np.ndarray,
    scale: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    z_value: float,
) -> float:
    covered = np.abs(mean - truth) <= z_value * scale
    return float(np.mean(covered[mask]))


def _scale_nll(error: np.ndarray, multiplier: np.ndarray) -> float:
    return float(
        np.mean(np.log(multiplier) + 0.5 * np.square(error / multiplier))
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)
