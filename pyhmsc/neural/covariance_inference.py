"""Frozen within-species covariance overlay for Neural-HMSC v0.1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np
import tensorflow as tf

from pyhmsc.neural.inference import NeuralHmscCompatibilityError, NeuralHmscInference
from pyhmsc.neural.models import probit_irls_laplace_full_anchor
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.release import NeuralHmscRelease, load_neural_hmsc_release
from pyhmsc.neural.simulator import FixedEffectDataset
from pyhmsc.neural.train import FixedShapeTrainingData, fixed_shape_training_data


CORRELATION_OVERLAY_KIND = "pyhmsc_neural_fixed_probit_covariance_overlay"
CORRELATION_OVERLAY_SCHEMA_VERSION = 1
CORRELATION_OVERLAY_ID = "neural_hmsc_fixed_probit_covariance_v1"
CORRELATION_OVERLAY_MANIFEST = "correlation_overlay.json"
CORRELATION_OVERLAY_WEIGHTS = "correlation_head.weights.h5"
CORRELATION_FEATURE_NAMES = (
    "laplace_fisher_z",
    "observed_prevalence_logit",
    "posterior_intercept_mean",
    "posterior_tmg_mean",
    "log_posterior_intercept_sd",
    "log_posterior_tmg_sd",
    "sample_tmg_mean",
    "log_sample_tmg_sd",
    "log_design_condition_number",
)
CORRELATION_RHO_LIMIT = 0.98
CORRELATION_ANCHOR_CLIP = 0.979
CORRELATION_DELTA_LIMIT = 0.75
CORRELATION_NORMALIZER_SD_FLOOR = 1e-6
CORRELATION_DELTA_PENALTY = 0.01

M56_PREREGISTRATION_SHA256 = (
    "d99b63da87103c3d8891cb2fab5bb7ffad30a188ed7be920950345581f8b2d4b"
)
M56_AUDIT_SHA256 = "5bb9236967afb5a2a1adc166781f4a34359a7469150aa2e19117752dd1fce29c"
BOUND_RELEASE_CONTENT_SHA256 = (
    "affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8"
)
BOUND_PACKAGE_MANIFEST_SHA256 = (
    "d2daa81ec841390df59324a208216ffa0032ac514e6c679649d98815490bdbc7"
)
BOUND_CHECKPOINT_MANIFEST_SHA256 = (
    "f62cd2217df6cc71cbe9f915c0cfbd3a3327b6684b3c5452bd9399aa130133a8"
)
BOUND_WEIGHTS_SHA256 = (
    "bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9"
)
BOUND_CALIBRATION_SHA256 = (
    "595fc0796d36802002cee09b270d53162f1fce100b83aecd32476e0958a0fd94"
)
BOUND_MEMBER_SEED = 20260721


@dataclass(frozen=True)
class CorrelationFeatureNormalizer:
    """Frozen training-only standardization for the nine overlay features."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "CorrelationFeatureNormalizer":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != len(CORRELATION_FEATURE_NAMES):
            raise ValueError("correlation features must be batch x species x 9")
        flat = values.reshape(-1, values.shape[-1])
        mean = np.mean(flat, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(flat, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scale, CORRELATION_NORMALIZER_SD_FLOOR)
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray | tf.Tensor) -> tf.Tensor:
        values = tf.cast(features, tf.float32)
        return (values - tf.constant(self.mean)) / tf.constant(self.scale)

    def to_record(self) -> dict[str, Any]:
        return {
            "feature_names": list(CORRELATION_FEATURE_NAMES),
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
            "scale_floor": CORRELATION_NORMALIZER_SD_FLOOR,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "CorrelationFeatureNormalizer":
        if tuple(record.get("feature_names", ())) != CORRELATION_FEATURE_NAMES:
            raise ValueError("correlation feature names or order differ")
        if float(record.get("scale_floor", -1.0)) != CORRELATION_NORMALIZER_SD_FLOOR:
            raise ValueError("correlation feature normalizer floor differs")
        mean = np.asarray(record.get("mean"), dtype=np.float32)
        scale = np.asarray(record.get("scale"), dtype=np.float32)
        expected = (len(CORRELATION_FEATURE_NAMES),)
        if mean.shape != expected or scale.shape != expected:
            raise ValueError("correlation normalizer vectors must contain nine values")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("correlation normalizer contains non-finite values")
        if np.any(scale < CORRELATION_NORMALIZER_SD_FLOOR):
            raise ValueError("correlation normalizer scale is below its frozen floor")
        return cls(mean=mean, scale=scale)


class FixedProbitCorrelationHead(tf.keras.Model):
    """Shared frozen-architecture residual head for intercept/slope correlation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_32 = tf.keras.layers.Dense(32, activation="relu", name="hidden_32")
        self.hidden_16 = tf.keras.layers.Dense(16, activation="relu", name="hidden_16")
        self.raw_delta = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="raw_delta",
        )

    def call(self, features: tf.Tensor) -> tf.Tensor:
        values = self.hidden_32(tf.cast(features, tf.float32))
        values = self.hidden_16(values)
        return self.raw_delta(values)[..., 0]


@dataclass(frozen=True)
class CorrelationOverlayPrediction:
    posterior: BetaPosterior
    correlation: tf.Tensor
    anchor_correlation: tf.Tensor
    delta_z: tf.Tensor
    features: tf.Tensor


@dataclass
class FixedProbitCovarianceInference:
    """Apply a learned covariance overlay to one immutable Neural-HMSC v0.1 member."""

    base: NeuralHmscInference
    head: FixedProbitCorrelationHead
    normalizer: CorrelationFeatureNormalizer
    base_binding: dict[str, Any]
    training_record: dict[str, Any]

    @classmethod
    def from_release(
        cls,
        registry_root: str | Path,
        *,
        normalizer: CorrelationFeatureNormalizer,
        model_seed: int,
    ) -> "FixedProbitCovarianceInference":
        release = load_neural_hmsc_release(registry_root)
        binding = validate_bound_v0_1_release(release)
        tf.keras.utils.set_random_seed(int(model_seed))
        head = FixedProbitCorrelationHead(name="fixed_probit_correlation_head")
        _build_correlation_head(head)
        return cls(
            base=release.load_checkpoint(seed=BOUND_MEMBER_SEED),
            head=head,
            normalizer=normalizer,
            base_binding=binding,
            training_record={},
        )

    @classmethod
    def initialize(
        cls,
        base: NeuralHmscInference,
        *,
        normalizer: CorrelationFeatureNormalizer,
        base_binding: dict[str, Any],
        model_seed: int,
    ) -> "FixedProbitCovarianceInference":
        tf.keras.utils.set_random_seed(int(model_seed))
        head = FixedProbitCorrelationHead(name="fixed_probit_correlation_head")
        _build_correlation_head(head)
        engine = cls(
            base=base,
            head=head,
            normalizer=normalizer,
            base_binding=dict(base_binding),
            training_record={},
        )
        engine._validate_base_scope()
        return engine

    @classmethod
    def load(
        cls,
        overlay: str | Path,
        *,
        registry_root: str | Path,
    ) -> "FixedProbitCovarianceInference":
        overlay = Path(overlay)
        manifest_path = overlay / CORRELATION_OVERLAY_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_overlay_manifest(manifest)
        weights = overlay / CORRELATION_OVERLAY_WEIGHTS
        if _file_sha256(weights) != manifest["weights_sha256"]:
            raise ValueError("correlation overlay weight hash mismatch")
        release = load_neural_hmsc_release(registry_root)
        binding = validate_bound_v0_1_release(release)
        if binding != manifest["base_binding"]:
            raise ValueError("correlation overlay base release binding differs")
        normalizer = CorrelationFeatureNormalizer.from_record(manifest["normalizer"])
        head = FixedProbitCorrelationHead(name="fixed_probit_correlation_head")
        _build_correlation_head(head)
        head.load_weights(weights)
        engine = cls(
            base=release.load_checkpoint(seed=BOUND_MEMBER_SEED),
            head=head,
            normalizer=normalizer,
            base_binding=binding,
            training_record=dict(manifest["training"]),
        )
        engine._validate_base_scope()
        return engine

    def save(self, overlay: str | Path) -> Path:
        """Atomically write a hash-bound overlay without copying the base release."""
        destination = Path(overlay)
        if destination.exists():
            raise FileExistsError(f"correlation overlay already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            weights = temporary / CORRELATION_OVERLAY_WEIGHTS
            _build_correlation_head(self.head)
            self.head.save_weights(weights)
            manifest = self._manifest(weights_sha256=_file_sha256(weights))
            (temporary / CORRELATION_OVERLAY_MANIFEST).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def fit(
        self,
        datasets: Sequence[FixedEffectDataset],
        *,
        epochs: int = 100,
        batch_size: int = 9,
        learning_rate: float = 0.001,
        seed: int,
        verbose: int = 0,
    ) -> dict[str, list[float]]:
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        data = fixed_shape_training_data(datasets)
        base_posterior, laplace, features = self._base_components(data)
        normalized = self.normalizer.transform(features)
        truth = tf.transpose(tf.constant(data.Beta, dtype=tf.float32), [0, 2, 1])
        mean = tf.stop_gradient(tf.transpose(base_posterior.mean, [0, 2, 1]))
        scale = tf.stop_gradient(tf.transpose(base_posterior.scale, [0, 2, 1]))
        anchor = tf.stop_gradient(_correlation_from_scale_tril(laplace.scale_tril))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        rng = np.random.default_rng(int(seed))
        n_communities = int(data.X.shape[0])
        history = {"loss": [], "joint_nll": [], "delta_penalty": []}
        for epoch in range(int(epochs)):
            order = rng.permutation(n_communities)
            losses: list[float] = []
            nlls: list[float] = []
            penalties: list[float] = []
            for start in range(0, n_communities, int(batch_size)):
                index = order[start : start + int(batch_size)]
                with tf.GradientTape() as tape:
                    raw = self.head(tf.gather(normalized, index), training=True)
                    delta_z, rho = correlation_from_raw_delta(
                        tf.gather(anchor, index), raw
                    )
                    nll = bivariate_beta_negative_log_probability(
                        tf.gather(truth, index),
                        tf.gather(mean, index),
                        tf.gather(scale, index),
                        rho,
                    )
                    nll_mean = tf.reduce_mean(nll)
                    penalty = CORRELATION_DELTA_PENALTY * tf.reduce_mean(
                        tf.square(delta_z)
                    )
                    loss = nll_mean + penalty
                gradients = tape.gradient(loss, self.head.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.head.trainable_variables))
                losses.append(float(loss))
                nlls.append(float(nll_mean))
                penalties.append(float(penalty))
            history["loss"].append(float(np.mean(losses)))
            history["joint_nll"].append(float(np.mean(nlls)))
            history["delta_penalty"].append(float(np.mean(penalties)))
            if verbose:
                print(f"epoch {epoch + 1}/{epochs} " f"loss={history['loss'][-1]:.6f}")
        self.training_record = {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "community_count": n_communities,
            "final_loss": history["loss"][-1],
            "final_joint_nll": history["joint_nll"][-1],
            "final_delta_penalty": history["delta_penalty"][-1],
        }
        return history

    def predict_beta_posterior(self, value: Any) -> BetaPosterior:
        return self.predict_details(value).posterior

    def predict_details(self, value: Any) -> CorrelationOverlayPrediction:
        self._reject_structural_inputs(value)
        data, context = self.base._prepare_inference_data(value)
        self.base._check_training_data_shape(data, batch_size=None)
        self._validate_context(context)
        base_posterior, laplace, features = self._base_components(data)
        raw = self.head(self.normalizer.transform(features), training=False)
        anchor = _correlation_from_scale_tril(laplace.scale_tril)
        delta_z, rho = correlation_from_raw_delta(anchor, raw)
        scale_tril = covariance_scale_tril(base_posterior.scale, rho)
        return CorrelationOverlayPrediction(
            posterior=BetaPosterior(
                mean=base_posterior.mean,
                scale=base_posterior.scale,
                scale_tril=scale_tril,
            ),
            correlation=rho,
            anchor_correlation=anchor,
            delta_z=delta_z,
            features=tf.constant(features),
        )

    def _base_components(
        self, data: FixedShapeTrainingData
    ) -> tuple[BetaPosterior, BetaPosterior, np.ndarray]:
        base_posterior = self.base.predict_beta_posterior(data, calibrated=True)
        laplace = probit_irls_laplace_full_anchor(
            tf.constant(data.X),
            tf.constant(data.Y),
            iterations=self.base.model.probit_anchor_iterations,
            prior_precision=self.base.model.probit_anchor_prior_precision,
            eta_clip=self.base.model.probit_anchor_eta_clip,
        )
        features = correlation_features(data.X, data.Y, base_posterior, laplace)
        return base_posterior, laplace, features

    def _validate_base_scope(self) -> None:
        expected = {"n_sites": 40, "n_covariates": 2, "n_species": 75}
        if self.base.dimensions != expected:
            raise NeuralHmscCompatibilityError(
                f"correlation overlay requires dimensions {expected}, got {self.base.dimensions}"
            )
        if self.base.distribution != "probit":
            raise NeuralHmscCompatibilityError(
                "correlation overlay requires distribution='probit'"
            )
        if _normalized_formula(self.base.formula) != "~TMG":
            raise NeuralHmscCompatibilityError(
                "correlation overlay requires formula '~ TMG'"
            )
        if self.base.covariate_names != ("Intercept", "TMG"):
            raise NeuralHmscCompatibilityError(
                "correlation overlay requires ordered coefficients ('Intercept', 'TMG')"
            )
        if self.base.model.probit_anchor != "irls_laplace":
            raise NeuralHmscCompatibilityError(
                "correlation overlay requires the frozen IRLS/Laplace anchor"
            )

    def _validate_context(self, context: Any) -> None:
        self._validate_base_scope()
        if context.distribution != "probit":
            raise NeuralHmscCompatibilityError("correlation input must use probit")
        if _normalized_formula(context.formula) != "~TMG":
            raise NeuralHmscCompatibilityError(
                "correlation input formula must be exactly '~ TMG'"
            )
        if tuple(context.covariate_names) != ("Intercept", "TMG"):
            raise NeuralHmscCompatibilityError(
                "correlation input coefficient order must be ('Intercept', 'TMG')"
            )

    @staticmethod
    def _reject_structural_inputs(value: Any) -> None:
        if isinstance(value, dict):
            unsupported = {
                "T",
                "Tr",
                "traits",
                "phylogeny",
                "study_design",
                "random_effects",
                "coords",
            }.intersection(value)
            if unsupported:
                raise NeuralHmscCompatibilityError(
                    f"correlation overlay does not support structural inputs: {sorted(unsupported)}"
                )

    def _manifest(self, *, weights_sha256: str) -> dict[str, Any]:
        return {
            "kind": CORRELATION_OVERLAY_KIND,
            "schema_version": CORRELATION_OVERLAY_SCHEMA_VERSION,
            "artifact_id": CORRELATION_OVERLAY_ID,
            "claim_scope": "fixed_40x75x2_probit_within_species_intercept_tmg_covariance",
            "preregistration_sha256": M56_PREREGISTRATION_SHA256,
            "artifact_seed_audit_sha256": M56_AUDIT_SHA256,
            "base_binding": dict(self.base_binding),
            "weights_file": CORRELATION_OVERLAY_WEIGHTS,
            "weights_sha256": weights_sha256,
            "normalizer": self.normalizer.to_record(),
            "architecture": {
                "input_features": 9,
                "hidden_units": [32, 16],
                "activation": "relu",
                "output_units": 1,
                "rho_limit": CORRELATION_RHO_LIMIT,
                "anchor_clip": CORRELATION_ANCHOR_CLIP,
                "delta_z_limit": CORRELATION_DELTA_LIMIT,
            },
            "training": dict(self.training_record),
            "base_mean_and_marginal_scale_modified": False,
            "target_outcome_selection_performed": False,
        }


def fit_fixed_probit_covariance_overlay(
    base: NeuralHmscInference,
    datasets: Sequence[FixedEffectDataset],
    *,
    base_binding: dict[str, Any],
    model_seed: int,
    epochs: int = 100,
    batch_size: int = 9,
    learning_rate: float = 0.001,
    verbose: int = 0,
) -> tuple[FixedProbitCovarianceInference, dict[str, list[float]]]:
    """Fit the frozen M56 head and training-only feature normalizer."""
    data = fixed_shape_training_data(datasets)
    base_posterior = base.predict_beta_posterior(data, calibrated=True)
    laplace = probit_irls_laplace_full_anchor(
        tf.constant(data.X),
        tf.constant(data.Y),
        iterations=base.model.probit_anchor_iterations,
        prior_precision=base.model.probit_anchor_prior_precision,
        eta_clip=base.model.probit_anchor_eta_clip,
    )
    normalizer = CorrelationFeatureNormalizer.fit(
        correlation_features(data.X, data.Y, base_posterior, laplace)
    )
    engine = FixedProbitCovarianceInference.initialize(
        base,
        normalizer=normalizer,
        base_binding=base_binding,
        model_seed=model_seed,
    )
    history = engine.fit(
        datasets,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=model_seed,
        verbose=verbose,
    )
    return engine, history


def correlation_features(
    X: np.ndarray,
    Y: np.ndarray,
    base_posterior: BetaPosterior,
    laplace_posterior: BetaPosterior,
) -> np.ndarray:
    """Construct the frozen ordered nine-feature tensor."""
    design = np.asarray(X, dtype=np.float64)
    response = np.asarray(Y, dtype=np.float64)
    mean = np.asarray(base_posterior.mean, dtype=np.float64)
    scale = np.asarray(base_posterior.scale, dtype=np.float64)
    if design.ndim != 3 or design.shape[2] != 2:
        raise ValueError("correlation features require batch x site x 2 design")
    if response.shape[:2] != design.shape[:2]:
        raise ValueError("correlation feature X/Y dimensions differ")
    if mean.shape != scale.shape or mean.shape != (
        design.shape[0],
        2,
        response.shape[2],
    ):
        raise ValueError("base posterior dimensions differ from X/Y")
    anchor = np.asarray(
        _correlation_from_scale_tril(laplace_posterior.scale_tril), dtype=np.float64
    )
    rho_anchor = np.clip(anchor, -CORRELATION_ANCHOR_CLIP, CORRELATION_ANCHOR_CLIP)
    z_anchor = np.arctanh(rho_anchor / CORRELATION_RHO_LIMIT)
    prevalence = np.clip(np.mean(response, axis=1), 1e-4, 1.0 - 1e-4)
    prevalence_logit = np.log(prevalence / (1.0 - prevalence))
    tmg = design[:, :, 1]
    tmg_mean = np.mean(tmg, axis=1)
    tmg_sd = np.std(tmg, axis=1, ddof=1)
    condition = np.asarray(
        [np.linalg.cond(matrix) for matrix in design], dtype=np.float64
    )
    n_species = response.shape[2]
    shared = [
        np.repeat(tmg_mean[:, None], n_species, axis=1),
        np.repeat(np.log(np.maximum(tmg_sd, 1e-12))[:, None], n_species, axis=1),
        np.repeat(np.log(np.maximum(condition, 1.0))[:, None], n_species, axis=1),
    ]
    features = np.stack(
        [
            z_anchor,
            prevalence_logit,
            mean[:, 0, :],
            mean[:, 1, :],
            np.log(np.maximum(scale[:, 0, :], 1e-12)),
            np.log(np.maximum(scale[:, 1, :], 1e-12)),
            *shared,
        ],
        axis=-1,
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("correlation features contain non-finite values")
    return features.astype(np.float32)


def correlation_from_raw_delta(
    anchor_correlation: tf.Tensor, raw_delta: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    anchor = tf.clip_by_value(
        tf.cast(anchor_correlation, tf.float32),
        -CORRELATION_ANCHOR_CLIP,
        CORRELATION_ANCHOR_CLIP,
    )
    z_anchor = tf.atanh(anchor / CORRELATION_RHO_LIMIT)
    delta_z = CORRELATION_DELTA_LIMIT * tf.tanh(tf.cast(raw_delta, tf.float32))
    rho = CORRELATION_RHO_LIMIT * tf.tanh(z_anchor + delta_z)
    return delta_z, rho


def covariance_scale_tril(scale: tf.Tensor, correlation: tf.Tensor) -> tf.Tensor:
    """Reconstruct a two-coefficient Cholesky factor with exact marginals."""
    marginal = tf.transpose(tf.cast(scale, tf.float32), [0, 2, 1])
    if marginal.shape.rank != 3 or marginal.shape[-1] != 2:
        raise ValueError("covariance overlay requires two marginal scales")
    rho = tf.cast(correlation, marginal.dtype)
    sigma0 = marginal[..., 0]
    sigma1 = marginal[..., 1]
    zeros = tf.zeros_like(sigma0)
    row0 = tf.stack([sigma0, zeros], axis=-1)
    row1 = tf.stack(
        [
            rho * sigma1,
            sigma1 * tf.sqrt(tf.maximum(1.0 - tf.square(rho), 1e-6)),
        ],
        axis=-1,
    )
    return tf.stack([row0, row1], axis=-2)


def bivariate_beta_negative_log_probability(
    truth: tf.Tensor,
    mean: tf.Tensor,
    scale: tf.Tensor,
    correlation: tf.Tensor,
) -> tf.Tensor:
    """Per-species bivariate Gaussian NLL under marginal SD plus correlation."""
    truth = tf.cast(truth, tf.float32)
    mean = tf.cast(mean, tf.float32)
    scale = tf.maximum(tf.cast(scale, tf.float32), 1e-12)
    rho = tf.clip_by_value(
        tf.cast(correlation, tf.float32),
        -CORRELATION_RHO_LIMIT,
        CORRELATION_RHO_LIMIT,
    )
    standardized = (truth - mean) / scale
    one_minus_rho2 = tf.maximum(1.0 - tf.square(rho), 1e-6)
    quadratic = (
        tf.square(standardized[..., 0])
        - 2.0 * rho * standardized[..., 0] * standardized[..., 1]
        + tf.square(standardized[..., 1])
    ) / one_minus_rho2
    log_determinant = (
        2.0 * tf.math.log(scale[..., 0])
        + 2.0 * tf.math.log(scale[..., 1])
        + tf.math.log(one_minus_rho2)
    )
    return 0.5 * (
        tf.constant(2.0 * np.log(2.0 * np.pi), dtype=tf.float32)
        + log_determinant
        + quadratic
    )


def validate_bound_v0_1_release(release: NeuralHmscRelease) -> dict[str, Any]:
    """Validate and return the exact immutable v0.1 member binding."""
    if release.release_id != "neural_hmsc_v0_1":
        raise ValueError("covariance overlay requires neural_hmsc_v0_1")
    manifest = release.manifest
    if manifest.get("content_sha256") != BOUND_RELEASE_CONTENT_SHA256:
        raise ValueError("bound Neural-HMSC v0.1 release content hash differs")
    records = {str(row["path"]): row for row in manifest.get("inventory", ())}
    expected = {
        "checkpoints/package_manifest.json": BOUND_PACKAGE_MANIFEST_SHA256,
        (
            f"checkpoints/{BOUND_MEMBER_SEED}/neural_checkpoint/"
            "neural_checkpoint.json"
        ): BOUND_CHECKPOINT_MANIFEST_SHA256,
        (
            f"checkpoints/{BOUND_MEMBER_SEED}/neural_checkpoint/" "weights.weights.h5"
        ): BOUND_WEIGHTS_SHA256,
        (
            f"checkpoints/{BOUND_MEMBER_SEED}/neural_checkpoint/"
            "coefficient_calibration.json"
        ): BOUND_CALIBRATION_SHA256,
    }
    for path, digest in expected.items():
        if records.get(path, {}).get("sha256") != digest:
            raise ValueError(f"bound Neural-HMSC v0.1 artifact hash differs: {path}")
        if _file_sha256(release.path / path) != digest:
            raise ValueError(f"bound Neural-HMSC v0.1 file changed: {path}")
    base = release.load_checkpoint(seed=BOUND_MEMBER_SEED)
    FixedProbitCovarianceInference(
        base=base,
        head=FixedProbitCorrelationHead(),
        normalizer=CorrelationFeatureNormalizer(
            mean=np.zeros(9, dtype=np.float32),
            scale=np.ones(9, dtype=np.float32),
        ),
        base_binding={},
        training_record={},
    )._validate_base_scope()
    return {
        "release_id": release.release_id,
        "release_content_sha256": BOUND_RELEASE_CONTENT_SHA256,
        "package_manifest_sha256": BOUND_PACKAGE_MANIFEST_SHA256,
        "member_seed": BOUND_MEMBER_SEED,
        "checkpoint_manifest_sha256": BOUND_CHECKPOINT_MANIFEST_SHA256,
        "weights_sha256": BOUND_WEIGHTS_SHA256,
        "calibration_sha256": BOUND_CALIBRATION_SHA256,
    }


def _validate_overlay_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("kind") != CORRELATION_OVERLAY_KIND:
        raise ValueError("unsupported correlation overlay kind")
    if manifest.get("schema_version") != CORRELATION_OVERLAY_SCHEMA_VERSION:
        raise ValueError("unsupported correlation overlay schema")
    if manifest.get("artifact_id") != CORRELATION_OVERLAY_ID:
        raise ValueError("correlation overlay artifact identifier differs")
    if manifest.get("preregistration_sha256") != M56_PREREGISTRATION_SHA256:
        raise ValueError("correlation overlay preregistration hash differs")
    if manifest.get("artifact_seed_audit_sha256") != M56_AUDIT_SHA256:
        raise ValueError("correlation overlay seed-audit hash differs")
    if manifest.get("weights_file") != CORRELATION_OVERLAY_WEIGHTS:
        raise ValueError("correlation overlay weight filename differs")
    if not _is_sha256(str(manifest.get("weights_sha256", ""))):
        raise ValueError("correlation overlay weight hash is invalid")
    expected_architecture = {
        "input_features": 9,
        "hidden_units": [32, 16],
        "activation": "relu",
        "output_units": 1,
        "rho_limit": CORRELATION_RHO_LIMIT,
        "anchor_clip": CORRELATION_ANCHOR_CLIP,
        "delta_z_limit": CORRELATION_DELTA_LIMIT,
    }
    if manifest.get("architecture") != expected_architecture:
        raise ValueError("correlation overlay architecture differs")
    if manifest.get("base_mean_and_marginal_scale_modified") is not False:
        raise ValueError("correlation overlay may not modify base marginals")
    if manifest.get("target_outcome_selection_performed") is not False:
        raise ValueError("correlation overlay may not use target outcomes")
    CorrelationFeatureNormalizer.from_record(manifest.get("normalizer", {}))


def _correlation_from_scale_tril(scale_tril: tf.Tensor | None) -> tf.Tensor:
    if scale_tril is None:
        raise ValueError("full Laplace scale_tril is required")
    covariance = tf.matmul(scale_tril, scale_tril, transpose_b=True)
    diagonal = tf.linalg.diag_part(covariance)
    denominator = tf.sqrt(tf.maximum(diagonal[..., 0] * diagonal[..., 1], 1e-12))
    return covariance[..., 0, 1] / denominator


def _build_correlation_head(head: FixedProbitCorrelationHead) -> None:
    head(tf.zeros((1, 1, len(CORRELATION_FEATURE_NAMES)), dtype=tf.float32))


def _normalized_formula(formula: str) -> str:
    return "".join(str(formula).split())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
