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
    min_multiplier: float = 0.1
    max_multiplier: float = 20.0
    method: str = "conditional_structured_scale"

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
        return {
            "semantics_version": 3,
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
            },
            "multiplier_bounds": [
                float(self.min_multiplier),
                float(self.max_multiplier),
            ],
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "ConditionalBetaScaleCalibration":
        """Reconstruct a conditional calibrator from stored metadata."""
        if metadata.get("method") != "conditional_structured_scale":
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
            min_multiplier=float(bounds[0]),
            max_multiplier=float(bounds[1]),
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
    min_multiplier: float = 0.1,
    max_multiplier: float = 20.0,
) -> ConditionalBetaScaleCalibration:
    """Fit a structured conditional scale head on simulated calibration truth.

    The head minimizes conditional Gaussian log score around the frozen neural
    posterior mean. A final scalar normalization restores nominal marginal
    coverage without discarding the learned coefficient-level scale ratios.
    """
    if not 0.0 < nominal_level < 1.0:
        raise ValueError("nominal_level must be between zero and one")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
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
    feature_design, feature_names = _structured_design(
        raw_features,
        location=location,
        scale=feature_scale,
        n_covariates=mean.shape[1],
    )

    standardized_error = np.abs(mean - truth) / scale
    selected_design = feature_design[mask.reshape(-1)]
    selected_error = standardized_error[mask]
    weights = tf.Variable(
        np.zeros(selected_design.shape[1], dtype=np.float64), dtype=tf.float64
    )
    design_tensor = tf.constant(selected_design, dtype=tf.float64)
    error_tensor = tf.constant(selected_error, dtype=tf.float64)
    base_log_scale = tf.constant(np.log(baseline.scale_multiplier), dtype=tf.float64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            log_multiplier = tf.clip_by_value(
                base_log_scale + tf.linalg.matvec(design_tensor, weights),
                np.log(min_multiplier),
                np.log(max_multiplier),
            )
            nll = tf.reduce_mean(
                log_multiplier
                + 0.5 * tf.square(error_tensor) * tf.exp(-2.0 * log_multiplier)
            )
            penalty = tf.cast(regularization, tf.float64) * tf.reduce_mean(
                tf.square(weights)
            )
            loss = nll + penalty
        gradient = tape.gradient(loss, weights)
        optimizer.apply_gradients([(gradient, weights)])

    fitted_weights = weights.numpy()
    adjustment = np.exp(
        np.clip(feature_design @ fitted_weights, -20.0, 20.0)
    ).reshape(mean.shape)
    z_value = float(norm.ppf(0.5 + nominal_level / 2.0))
    normalization = float(
        np.quantile((standardized_error / adjustment)[mask], nominal_level) / z_value
    )
    normalization = float(np.clip(normalization, min_multiplier, max_multiplier))
    multipliers = np.clip(
        normalization * adjustment, min_multiplier, max_multiplier
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
    return np.clip(
        calibration.normalization_multiplier * adjustment,
        calibration.min_multiplier,
        calibration.max_multiplier,
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
    prevalence = np.mean(Y > 0.0, axis=1)
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
