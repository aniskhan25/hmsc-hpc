"""Frozen bivariate Student-t posterior overlay for Neural-HMSC v0.1."""

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
from scipy.special import ndtr
import tensorflow as tf

from pyhmsc.neural.covariance_inference import (
    BOUND_MEMBER_SEED,
    validate_bound_v0_1_release,
)
from pyhmsc.neural.inference import NeuralHmscCompatibilityError, NeuralHmscInference
from pyhmsc.neural.models import probit_irls_laplace_full_anchor
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.release import load_neural_hmsc_release
from pyhmsc.neural.simulator import FixedEffectDataset
from pyhmsc.neural.train import FixedShapeTrainingData, fixed_shape_training_data
from pyhmsc.neural.variable_inference import (
    VARIABLE_SHAPE_BASELINE_ID,
    validate_variable_shape_baseline,
)


STUDENT_T_OVERLAY_KIND = "pyhmsc_neural_fixed_probit_student_t_overlay"
STUDENT_T_OVERLAY_SCHEMA_VERSION = 1
STUDENT_T_OVERLAY_ID = "neural_hmsc_fixed_probit_student_t_v1"
STUDENT_T_OVERLAY_MANIFEST = "student_t_overlay.json"
STUDENT_T_OVERLAY_WEIGHTS = "student_t_head.weights.h5"

M57_DECISION_SHA256 = "a1a7bc4a54eca4c78f6b32537f1afff662a524557accbd99d7267a28bc2cb2ba"
M57_AUDIT_SHA256 = "1e1150a04cd17643db37988bfc010b611f8f49d638dbd40ead49cd5329b9b25c"
M57_PREREGISTRATION_SHA256 = (
    "10878c65bb16746a4a9c57fa91d6a4fd3cbcc753739a816f6cc8b9b738f1a388"
)

BOUND_VARIABLE_CONTENT_SHA256 = (
    "badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9"
)
BOUND_VARIABLE_BASELINE_MANIFEST_SHA256 = (
    "b9387efc147ecd9e3978c80cb2cc2a3ebdcddd5c68445c8b63ce8f37af61a2f1"
)
BOUND_VARIABLE_CHECKPOINT_MANIFEST_SHA256 = (
    "cf46ebfdfc457e71a0da28f48f7709613f7e47b101b946553f711d5e1e4f47a5"
)
BOUND_VARIABLE_WEIGHTS_SHA256 = (
    "70ef4548eeb1dc3a0d9367cb8edaedb5a2030370179241f35b372aecd8d5c4cd"
)
BOUND_VARIABLE_CALIBRATION_SHA256 = (
    "c3c8fd4ff50583ced5273c009e501ea0b6f400ff144a74f510513633edd7b771"
)
BOUND_M56_FREEZE_SHA256 = (
    "c4fcb04cf1ebd7123be12144803de319ce1ff16a31e4fc5a1fb3e224f361a526"
)
BOUND_M56_OVERLAY_MANIFEST_SHA256 = (
    "24f7eafa4a886afab94711bab77c56e76aef726fc93c0911c372b639bfa0121d"
)
BOUND_M56_WEIGHTS_SHA256 = (
    "66033d4f84cd443abf94053923e929180c0307fb08ac2a1bb9eaa75fe32ccde5"
)

STUDENT_T_FEATURE_NAMES = (
    "base_intercept_mean",
    "base_tmg_mean",
    "log_base_intercept_sd",
    "log_base_tmg_sd",
    "irls_intercept_mode",
    "irls_tmg_mode",
    "log_irls_intercept_sd",
    "log_irls_tmg_sd",
    "laplace_fisher_z",
    "observed_prevalence_logit",
    "sample_tmg_mean",
    "log_sample_tmg_sd",
    "log_design_condition_number",
    "normalized_probit_score_intercept",
    "normalized_probit_score_tmg",
)
STUDENT_T_NORMALIZER_SD_FLOOR = 1e-6
STUDENT_T_RHO_LIMIT = 0.98
STUDENT_T_LAPLACE_RHO_CLIP = 0.979
STUDENT_T_DF_MIN = 2.1
STUDENT_T_DF_RANGE = 27.9
STUDENT_T_LOCATION_LIMIT = 2.0
STUDENT_T_LOG_SCALE_LIMIT = 1.5
STUDENT_T_INITIAL_DF = 10.0
STUDENT_T_MODEL_SEED = 321_900_001


@dataclass(frozen=True)
class StudentTBetaPosterior:
    """Bivariate per-species Student-t posterior for fixed-effect Beta."""

    mean: tf.Tensor
    marginal_scale: tf.Tensor
    covariance_tril: tf.Tensor
    student_t_scale_tril: tf.Tensor
    degrees_of_freedom: tf.Tensor

    @property
    def scale(self) -> tf.Tensor:
        """Compatibility alias for the reported marginal standard deviation."""
        return self.marginal_scale

    @property
    def scale_tril(self) -> tf.Tensor:
        """Compatibility alias for the covariance Cholesky factor."""
        return self.covariance_tril

    @property
    def posterior_family(self) -> str:
        return "bivariate_student_t"


@dataclass(frozen=True)
class StudentTFeatureNormalizer:
    """Training-only population standardization for the frozen 15 features."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "StudentTFeatureNormalizer":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != len(STUDENT_T_FEATURE_NAMES):
            raise ValueError("Student-t features must be batch x species x 15")
        flat = values.reshape(-1, values.shape[-1])
        mean = np.mean(flat, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(flat, axis=0, ddof=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scale, STUDENT_T_NORMALIZER_SD_FLOOR)
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray | tf.Tensor) -> tf.Tensor:
        values = tf.cast(features, tf.float32)
        return (values - tf.constant(self.mean)) / tf.constant(self.scale)

    def to_record(self) -> dict[str, Any]:
        return {
            "feature_names": list(STUDENT_T_FEATURE_NAMES),
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
            "population_standard_deviation": True,
            "scale_floor": STUDENT_T_NORMALIZER_SD_FLOOR,
            "fit_role": "training_realizations_only",
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "StudentTFeatureNormalizer":
        if tuple(record.get("feature_names", ())) != STUDENT_T_FEATURE_NAMES:
            raise ValueError("Student-t feature names or order differ")
        if record.get("population_standard_deviation") is not True:
            raise ValueError("Student-t normalizer must use population SD")
        if float(record.get("scale_floor", -1.0)) != STUDENT_T_NORMALIZER_SD_FLOOR:
            raise ValueError("Student-t normalizer floor differs")
        if record.get("fit_role") != "training_realizations_only":
            raise ValueError("Student-t normalizer fit role differs")
        mean = np.asarray(record.get("mean"), dtype=np.float32)
        scale = np.asarray(record.get("scale"), dtype=np.float32)
        expected = (len(STUDENT_T_FEATURE_NAMES),)
        if mean.shape != expected or scale.shape != expected:
            raise ValueError("Student-t normalizer vectors must contain 15 values")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("Student-t normalizer contains non-finite values")
        if np.any(scale < STUDENT_T_NORMALIZER_SD_FLOOR):
            raise ValueError("Student-t normalizer scale is below its frozen floor")
        return cls(mean=mean, scale=scale)


class FixedProbitStudentTHead(tf.keras.Model):
    """Frozen 15-64-64-32-6 shared species head."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_64_a = tf.keras.layers.Dense(
            64, activation="relu", name="hidden_64_a"
        )
        self.hidden_64_b = tf.keras.layers.Dense(
            64, activation="relu", name="hidden_64_b"
        )
        self.hidden_32 = tf.keras.layers.Dense(32, activation="relu", name="hidden_32")
        initial_df_probability = (
            STUDENT_T_INITIAL_DF - STUDENT_T_DF_MIN
        ) / STUDENT_T_DF_RANGE
        initial_df_logit = np.log(
            initial_df_probability / (1.0 - initial_df_probability)
        )
        self.raw_posterior = tf.keras.layers.Dense(
            6,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(
                [0.0, 0.0, 0.0, 0.0, 0.0, initial_df_logit]
            ),
            name="raw_posterior",
        )

    def call(self, features: tf.Tensor) -> tf.Tensor:
        values = self.hidden_64_a(tf.cast(features, tf.float32))
        values = self.hidden_64_b(values)
        values = self.hidden_32(values)
        return self.raw_posterior(values)


@dataclass(frozen=True)
class StudentTOverlayPrediction:
    posterior: StudentTBetaPosterior
    raw_outputs: tf.Tensor
    features: tf.Tensor
    base_posterior: BetaPosterior
    laplace_posterior: BetaPosterior


def student_t_posterior_from_raw(
    base_posterior: BetaPosterior,
    raw_outputs: tf.Tensor,
) -> StudentTBetaPosterior:
    """Apply the exact frozen six-output transform."""
    base_mean = tf.stop_gradient(
        tf.transpose(tf.cast(base_posterior.mean, tf.float32), [0, 2, 1])
    )
    base_scale = tf.stop_gradient(
        tf.transpose(tf.cast(base_posterior.scale, tf.float32), [0, 2, 1])
    )
    raw = tf.cast(raw_outputs, tf.float32)
    if raw.shape.rank != 3 or raw.shape[-1] != 6:
        raise ValueError("Student-t raw outputs must be batch x species x 6")
    if base_mean.shape != base_scale.shape or base_mean.shape[-1] != 2:
        raise ValueError("Student-t overlay requires two base coefficients")
    location = base_mean + (
        STUDENT_T_LOCATION_LIMIT * base_scale * tf.tanh(raw[..., 0:2])
    )
    marginal_scale = base_scale * tf.exp(
        STUDENT_T_LOG_SCALE_LIMIT * tf.tanh(raw[..., 2:4])
    )
    correlation = STUDENT_T_RHO_LIMIT * tf.tanh(raw[..., 4])
    degrees_of_freedom = STUDENT_T_DF_MIN + (
        STUDENT_T_DF_RANGE * tf.math.sigmoid(raw[..., 5])
    )
    covariance_tril = _covariance_tril(marginal_scale, correlation)
    t_scale = tf.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
    student_t_scale_tril = covariance_tril * t_scale[..., None, None]
    return StudentTBetaPosterior(
        mean=tf.transpose(location, [0, 2, 1]),
        marginal_scale=tf.transpose(marginal_scale, [0, 2, 1]),
        covariance_tril=covariance_tril,
        student_t_scale_tril=student_t_scale_tril,
        degrees_of_freedom=degrees_of_freedom,
    )


def bivariate_student_t_negative_log_probability(
    posterior: StudentTBetaPosterior,
    beta_true: tf.Tensor,
) -> tf.Tensor:
    """Return exact per-community, per-species bivariate Student-t NLL."""
    truth = tf.cast(beta_true, tf.float32)
    if truth.shape.rank != 3:
        raise ValueError("Student-t truth must be batch x coefficient x species")
    residual = tf.transpose(truth - posterior.mean, [0, 2, 1])
    solved = tf.linalg.triangular_solve(
        posterior.student_t_scale_tril, residual[..., None], lower=True
    )
    delta = tf.reduce_sum(tf.square(solved), axis=(-2, -1))
    nu = tf.cast(posterior.degrees_of_freedom, tf.float32)
    d = tf.constant(2.0, dtype=tf.float32)
    log_determinant = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(posterior.student_t_scale_tril)), axis=-1
    )
    return (
        -tf.math.lgamma((nu + d) / 2.0)
        + tf.math.lgamma(nu / 2.0)
        + (d / 2.0) * tf.math.log(nu * tf.constant(np.pi, tf.float32))
        + 0.5 * log_determinant
        + ((nu + d) / 2.0) * tf.math.log1p(delta / nu)
    )


def sample_student_t_beta_posterior(
    posterior: StudentTBetaPosterior,
    *,
    draws: int,
    seed: int | np.random.SeedSequence,
) -> np.ndarray:
    """Draw deterministic samples with shape draw x batch x coefficient x species."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance_tril = np.asarray(posterior.covariance_tril, dtype=np.float64)
    degrees = np.asarray(posterior.degrees_of_freedom, dtype=np.float64)
    if mean.ndim != 3 or mean.shape[1] != 2:
        raise ValueError("Student-t mean must be batch x 2 x species")
    expected_tril = (mean.shape[0], mean.shape[2], 2, 2)
    if covariance_tril.shape != expected_tril:
        raise ValueError("Student-t covariance Cholesky dimensions differ")
    if degrees.shape != (mean.shape[0], mean.shape[2]):
        raise ValueError("Student-t degrees-of-freedom dimensions differ")
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(draws,) + degrees.shape + (2,))
    chi_square = rng.chisquare(df=degrees, size=(draws,) + degrees.shape)
    correlated = np.einsum("bsij,dbsj->dbsi", covariance_tril, normal)
    factor = np.sqrt((degrees[None, ...] - 2.0) / chi_square)
    values = mean.transpose(0, 2, 1)[None, ...] + factor[..., None] * correlated
    return values.transpose(0, 1, 3, 2)


def write_student_t_beta_posterior_hdf5(
    posterior: StudentTBetaPosterior,
    output: str | Path,
    *,
    covariate_names: Sequence[str],
    species_names: Sequence[str],
    distribution: str = "probit",
    formula: str = "~ TMG",
    chains: int = 1,
    draws: int = 100,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write Student-t Beta draws in the existing pyhmsc posterior shape."""
    if chains <= 0 or draws <= 0:
        raise ValueError("chains and draws must be positive")
    mean = np.asarray(posterior.mean)
    if mean.shape[0] != 1:
        raise ValueError("Student-t HDF5 output supports one dataset at a time")
    if mean.shape[1:] != (len(covariate_names), len(species_names)):
        raise ValueError("covariate/species names do not match Student-t posterior")
    sampled = sample_student_t_beta_posterior(
        posterior, draws=chains * draws, seed=seed
    )
    beta = sampled[:, 0].reshape(chains, draws, *mean.shape[1:])
    record = dict(metadata or {})
    record.update(
        {
            "posterior_family": posterior.posterior_family,
            "posterior_semantics": "multivariate_student_t_scale_and_covariance",
            "student_t_overlay_id": STUDENT_T_OVERLAY_ID,
            "distribution": distribution,
            "formula": {"X": formula},
            "covariate_names": list(covariate_names),
            "species_names": list(species_names),
            "nChains": int(chains),
            "nDraws": int(draws),
            "seed": int(seed),
        }
    )
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to write Neural-HMSC posterior files") from exc
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.create_dataset("Beta", data=beta)
        handle.create_dataset(
            "StudentTDegreesOfFreedom",
            data=np.asarray(posterior.degrees_of_freedom)[0],
        )
        handle.create_dataset(
            "StudentTCovarianceCholesky",
            data=np.asarray(posterior.covariance_tril)[0],
        )
        handle.create_dataset(
            "StudentTScaleCholesky",
            data=np.asarray(posterior.student_t_scale_tril)[0],
        )
        handle.attrs["nChains"] = int(chains)
        handle.attrs["nDraws"] = int(draws)
        handle.attrs["pyhmsc_metadata"] = json.dumps(record, sort_keys=True)
    return output


def student_t_features(
    X: np.ndarray,
    Y: np.ndarray,
    base_posterior: BetaPosterior,
    laplace_posterior: BetaPosterior,
) -> np.ndarray:
    """Construct the exact ordered 15-feature tensor."""
    design = np.asarray(X, dtype=np.float64)
    response = np.asarray(Y, dtype=np.float64)
    base_mean = np.asarray(base_posterior.mean, dtype=np.float64)
    base_scale = np.asarray(base_posterior.scale, dtype=np.float64)
    laplace_mean = np.asarray(laplace_posterior.mean, dtype=np.float64)
    laplace_scale = np.asarray(laplace_posterior.scale, dtype=np.float64)
    if design.ndim != 3 or design.shape[2] != 2:
        raise ValueError("Student-t features require batch x site x 2 design")
    if response.shape[:2] != design.shape[:2]:
        raise ValueError("Student-t feature X/Y dimensions differ")
    expected = (design.shape[0], 2, response.shape[2])
    for name, value in (
        ("base mean", base_mean),
        ("base scale", base_scale),
        ("Laplace mean", laplace_mean),
        ("Laplace scale", laplace_scale),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} dimensions differ from X/Y")
    laplace_rho = np.asarray(
        _correlation_from_scale_tril(laplace_posterior.scale_tril),
        dtype=np.float64,
    )
    fisher_z = np.arctanh(
        np.clip(laplace_rho, -STUDENT_T_LAPLACE_RHO_CLIP, STUDENT_T_LAPLACE_RHO_CLIP)
        / STUDENT_T_RHO_LIMIT
    )
    prevalence = np.clip(np.mean(response, axis=1), 1e-4, 1.0 - 1e-4)
    prevalence_logit = np.log(prevalence / (1.0 - prevalence))
    tmg = design[:, :, 1]
    tmg_mean = np.mean(tmg, axis=1)
    tmg_sd = np.std(tmg, axis=1, ddof=1)
    condition = np.asarray([np.linalg.cond(value) for value in design])
    scores = _normalized_probit_scores(design, response, base_mean)
    species = response.shape[2]

    def shared(values: np.ndarray) -> np.ndarray:
        return np.repeat(values[:, None], species, axis=1)

    features = np.stack(
        [
            base_mean[:, 0, :],
            base_mean[:, 1, :],
            np.log(np.maximum(base_scale[:, 0, :], 1e-12)),
            np.log(np.maximum(base_scale[:, 1, :], 1e-12)),
            laplace_mean[:, 0, :],
            laplace_mean[:, 1, :],
            np.log(np.maximum(laplace_scale[:, 0, :], 1e-12)),
            np.log(np.maximum(laplace_scale[:, 1, :], 1e-12)),
            fisher_z,
            prevalence_logit,
            shared(tmg_mean),
            shared(np.log(np.maximum(tmg_sd, 1e-12))),
            shared(np.log(np.maximum(condition, 1.0))),
            scores[:, 0, :],
            scores[:, 1, :],
        ],
        axis=-1,
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("Student-t features contain non-finite values")
    return features.astype(np.float32)


@dataclass
class FixedProbitStudentTInference:
    """Apply the frozen Student-t head to one immutable v0.1 member."""

    base: NeuralHmscInference
    head: FixedProbitStudentTHead
    normalizer: StudentTFeatureNormalizer
    base_binding: dict[str, Any]
    variable_v1_binding: dict[str, Any]
    m56_negative_binding: dict[str, Any]
    training_record: dict[str, Any]
    model_seed: int

    @classmethod
    def initialize(
        cls,
        base: NeuralHmscInference,
        *,
        normalizer: StudentTFeatureNormalizer,
        base_binding: dict[str, Any],
        variable_v1_binding: dict[str, Any],
        m56_negative_binding: dict[str, Any],
        model_seed: int,
    ) -> "FixedProbitStudentTInference":
        if int(model_seed) != STUDENT_T_MODEL_SEED:
            raise ValueError(
                f"Student-t model seed must be the frozen value {STUDENT_T_MODEL_SEED}"
            )
        tf.keras.utils.set_random_seed(int(model_seed))
        head = FixedProbitStudentTHead(name="fixed_probit_student_t_head")
        _build_head(head)
        base.model.trainable = False
        engine = cls(
            base=base,
            head=head,
            normalizer=normalizer,
            base_binding=dict(base_binding),
            variable_v1_binding=dict(variable_v1_binding),
            m56_negative_binding=dict(m56_negative_binding),
            training_record={},
            model_seed=int(model_seed),
        )
        engine._validate_base_scope()
        return engine

    @classmethod
    def load(
        cls,
        overlay: str | Path,
        *,
        registry_root: str | Path,
        variable_registry_root: str | Path,
        m56_root: str | Path,
    ) -> "FixedProbitStudentTInference":
        overlay = Path(overlay)
        manifest = json.loads(
            (overlay / STUDENT_T_OVERLAY_MANIFEST).read_text(encoding="utf-8")
        )
        _validate_manifest(manifest)
        weights = overlay / STUDENT_T_OVERLAY_WEIGHTS
        if _file_sha256(weights) != manifest["weights_sha256"]:
            raise ValueError("Student-t overlay weight hash mismatch")
        release = load_neural_hmsc_release(registry_root)
        base_binding = validate_bound_v0_1_release(release)
        variable_binding = validate_bound_variable_v1(variable_registry_root)
        m56_binding = validate_bound_m56_negative(m56_root)
        if base_binding != manifest["base_binding"]:
            raise ValueError("Student-t overlay v0.1 binding differs")
        if variable_binding != manifest["variable_v1_binding"]:
            raise ValueError("Student-t overlay variable-v1 binding differs")
        if m56_binding != manifest["m56_negative_binding"]:
            raise ValueError("Student-t overlay M56 binding differs")
        normalizer = StudentTFeatureNormalizer.from_record(manifest["normalizer"])
        model_seed = int(manifest["architecture"]["model_seed"])
        tf.keras.utils.set_random_seed(model_seed)
        head = FixedProbitStudentTHead(name="fixed_probit_student_t_head")
        _build_head(head)
        head.load_weights(weights)
        base = release.load_checkpoint(seed=BOUND_MEMBER_SEED)
        base.model.trainable = False
        engine = cls(
            base=base,
            head=head,
            normalizer=normalizer,
            base_binding=base_binding,
            variable_v1_binding=variable_binding,
            m56_negative_binding=m56_binding,
            training_record=dict(manifest["training"]),
            model_seed=model_seed,
        )
        engine._validate_base_scope()
        return engine

    def save(self, overlay: str | Path) -> Path:
        destination = Path(overlay)
        if destination.exists():
            raise FileExistsError(f"Student-t overlay already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            weights = temporary / STUDENT_T_OVERLAY_WEIGHTS
            _build_head(self.head)
            self.head.save_weights(weights)
            manifest = self._manifest(weights_sha256=_file_sha256(weights))
            (temporary / STUDENT_T_OVERLAY_MANIFEST).write_text(
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
        epochs: int = 150,
        batch_contexts: int = 9,
        learning_rate: float = 0.0005,
        seed: int,
        verbose: int = 0,
    ) -> dict[str, list[float]]:
        if epochs <= 0 or batch_contexts <= 0 or learning_rate <= 0.0:
            raise ValueError("Student-t training controls must be positive")
        groups = _paired_context_groups(datasets)
        data = fixed_shape_training_data(datasets)
        base, laplace, features = self._base_components(data)
        normalized = self.normalizer.transform(features)
        truth = tf.constant(data.Beta, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        rng = np.random.default_rng(int(seed))
        history = {"loss": []}
        for epoch in range(int(epochs)):
            order = rng.permutation(len(groups))
            epoch_losses: list[float] = []
            for start in range(0, len(groups), int(batch_contexts)):
                selected = order[start : start + int(batch_contexts)]
                index = np.concatenate([groups[value] for value in selected])
                with tf.GradientTape() as tape:
                    raw = self.head(tf.gather(normalized, index), training=True)
                    posterior = student_t_posterior_from_raw(
                        _gather_beta_posterior(base, index), raw
                    )
                    nll = bivariate_student_t_negative_log_probability(
                        posterior, tf.gather(truth, index)
                    )
                    loss = tf.reduce_mean(nll)
                gradients = tape.gradient(loss, self.head.trainable_variables)
                if any(value is None for value in gradients):
                    raise RuntimeError("Student-t head produced a missing gradient")
                optimizer.apply_gradients(zip(gradients, self.head.trainable_variables))
                epoch_losses.append(float(loss))
            history["loss"].append(float(np.mean(epoch_losses)))
            if verbose:
                print(f"epoch {epoch + 1}/{epochs} loss={history['loss'][-1]:.6f}")
        self.training_record = {
            "epochs": int(epochs),
            "batch_owning_contexts": int(batch_contexts),
            "responses_per_context": 2,
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "owning_context_count": len(groups),
            "realization_count": len(datasets),
            "final_loss": history["loss"][-1],
            "objective": "unweighted_mean_bivariate_student_t_nll",
            "early_stopping": False,
            "gradient_clipping": False,
            "weight_decay": False,
        }
        return history

    def predict_beta_posterior(self, value: Any) -> StudentTBetaPosterior:
        return self.predict_details(value).posterior

    def predict_details(self, value: Any) -> StudentTOverlayPrediction:
        self._reject_structural_inputs(value)
        data, context = self.base._prepare_inference_data(value)
        self.base._check_training_data_shape(data, batch_size=None)
        self._validate_context(context)
        base, laplace, features = self._base_components(data)
        raw = self.head(self.normalizer.transform(features), training=False)
        return StudentTOverlayPrediction(
            posterior=student_t_posterior_from_raw(base, raw),
            raw_outputs=raw,
            features=tf.constant(features),
            base_posterior=base,
            laplace_posterior=laplace,
        )

    def _base_components(
        self, data: FixedShapeTrainingData
    ) -> tuple[BetaPosterior, BetaPosterior, np.ndarray]:
        base = self.base.predict_beta_posterior(data, calibrated=True)
        laplace = probit_irls_laplace_full_anchor(
            tf.constant(data.X),
            tf.constant(data.Y),
            iterations=self.base.model.probit_anchor_iterations,
            prior_precision=self.base.model.probit_anchor_prior_precision,
            eta_clip=self.base.model.probit_anchor_eta_clip,
        )
        return base, laplace, student_t_features(data.X, data.Y, base, laplace)

    def _validate_base_scope(self) -> None:
        expected = {"n_sites": 40, "n_covariates": 2, "n_species": 75}
        if self.base.dimensions != expected:
            raise NeuralHmscCompatibilityError(
                f"Student-t overlay requires dimensions {expected}, got {self.base.dimensions}"
            )
        if self.base.distribution != "probit":
            raise NeuralHmscCompatibilityError(
                "Student-t overlay requires distribution='probit'"
            )
        if _normalized_formula(self.base.formula) != "~TMG":
            raise NeuralHmscCompatibilityError(
                "Student-t overlay requires formula '~ TMG'"
            )
        if self.base.covariate_names != ("Intercept", "TMG"):
            raise NeuralHmscCompatibilityError(
                "Student-t overlay requires ordered coefficients ('Intercept', 'TMG')"
            )
        if self.base.model.probit_anchor != "irls_laplace":
            raise NeuralHmscCompatibilityError(
                "Student-t overlay requires the frozen IRLS/Laplace anchor"
            )

    def _validate_context(self, context: Any) -> None:
        self._validate_base_scope()
        if context.distribution != "probit":
            raise NeuralHmscCompatibilityError("Student-t input must use probit")
        if _normalized_formula(context.formula) != "~TMG":
            raise NeuralHmscCompatibilityError(
                "Student-t input formula must be exactly '~ TMG'"
            )
        if tuple(context.covariate_names) != ("Intercept", "TMG"):
            raise NeuralHmscCompatibilityError(
                "Student-t coefficient order must be ('Intercept', 'TMG')"
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
                    f"Student-t overlay does not support structural inputs: {sorted(unsupported)}"
                )

    def _manifest(self, *, weights_sha256: str) -> dict[str, Any]:
        return {
            "kind": STUDENT_T_OVERLAY_KIND,
            "schema_version": STUDENT_T_OVERLAY_SCHEMA_VERSION,
            "artifact_id": STUDENT_T_OVERLAY_ID,
            "claim_scope": "fixed_40x75x2_probit_bivariate_student_t_beta",
            "decision_sha256": M57_DECISION_SHA256,
            "artifact_seed_audit_sha256": M57_AUDIT_SHA256,
            "preregistration_sha256": M57_PREREGISTRATION_SHA256,
            "base_binding": dict(self.base_binding),
            "variable_v1_binding": dict(self.variable_v1_binding),
            "m56_negative_binding": dict(self.m56_negative_binding),
            "weights_file": STUDENT_T_OVERLAY_WEIGHTS,
            "weights_sha256": weights_sha256,
            "normalizer": self.normalizer.to_record(),
            "architecture": _architecture_record(),
            "posterior_semantics": {
                "family": "bivariate_student_t",
                "reported_scale": "marginal_standard_deviation",
                "density_scale": "student_t_scale_tril",
                "covariance_exists": True,
            },
            "training": dict(self.training_record),
            "base_checkpoint_trainable": False,
            "target_outcome_selection_performed": False,
        }


def fit_fixed_probit_student_t_overlay(
    base: NeuralHmscInference,
    datasets: Sequence[FixedEffectDataset],
    *,
    base_binding: dict[str, Any],
    variable_v1_binding: dict[str, Any],
    m56_negative_binding: dict[str, Any],
    model_seed: int,
    epochs: int = 150,
    batch_contexts: int = 9,
    learning_rate: float = 0.0005,
    verbose: int = 0,
) -> tuple[FixedProbitStudentTInference, dict[str, list[float]]]:
    """Fit the frozen M57 normalizer and Student-t head."""
    _paired_context_groups(datasets)
    data = fixed_shape_training_data(datasets)
    base_posterior = base.predict_beta_posterior(data, calibrated=True)
    laplace = probit_irls_laplace_full_anchor(
        tf.constant(data.X),
        tf.constant(data.Y),
        iterations=base.model.probit_anchor_iterations,
        prior_precision=base.model.probit_anchor_prior_precision,
        eta_clip=base.model.probit_anchor_eta_clip,
    )
    normalizer = StudentTFeatureNormalizer.fit(
        student_t_features(data.X, data.Y, base_posterior, laplace)
    )
    engine = FixedProbitStudentTInference.initialize(
        base,
        normalizer=normalizer,
        base_binding=base_binding,
        variable_v1_binding=variable_v1_binding,
        m56_negative_binding=m56_negative_binding,
        model_seed=model_seed,
    )
    history = engine.fit(
        datasets,
        epochs=epochs,
        batch_contexts=batch_contexts,
        learning_rate=learning_rate,
        seed=model_seed,
        verbose=verbose,
    )
    return engine, history


def validate_bound_variable_v1(registry_root: str | Path) -> dict[str, Any]:
    """Validate the exact immutable variable-v1 regression baseline."""
    root = Path(registry_root)
    baseline = (
        root
        if root.name == VARIABLE_SHAPE_BASELINE_ID
        else root / VARIABLE_SHAPE_BASELINE_ID
    )
    payload = validate_variable_shape_baseline(
        baseline, expected_baseline_id=VARIABLE_SHAPE_BASELINE_ID
    )
    manifest = baseline / "baseline.json"
    if _file_sha256(manifest) != BOUND_VARIABLE_BASELINE_MANIFEST_SHA256:
        raise ValueError("bound variable-v1 baseline manifest changed")
    if payload.get("content_sha256") != BOUND_VARIABLE_CONTENT_SHA256:
        raise ValueError("bound variable-v1 content hash differs")
    checkpoint = payload.get("checkpoint", {})
    if checkpoint.get("manifest_sha256") != BOUND_VARIABLE_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("bound variable-v1 checkpoint manifest hash differs")
    if checkpoint.get("weights_sha256") != BOUND_VARIABLE_WEIGHTS_SHA256:
        raise ValueError("bound variable-v1 weights hash differs")
    records = {str(row["path"]): row for row in payload.get("inventory", ())}
    calibration_rows = [
        row
        for path, row in records.items()
        if path.endswith("variable_coefficient_calibration.json")
    ]
    if len(calibration_rows) != 1 or calibration_rows[0]["sha256"] != (
        BOUND_VARIABLE_CALIBRATION_SHA256
    ):
        raise ValueError("bound variable-v1 calibration hash differs")
    return {
        "baseline_id": VARIABLE_SHAPE_BASELINE_ID,
        "content_sha256": BOUND_VARIABLE_CONTENT_SHA256,
        "baseline_manifest_sha256": BOUND_VARIABLE_BASELINE_MANIFEST_SHA256,
        "checkpoint_manifest_sha256": BOUND_VARIABLE_CHECKPOINT_MANIFEST_SHA256,
        "weights_sha256": BOUND_VARIABLE_WEIGHTS_SHA256,
        "calibration_sha256": BOUND_VARIABLE_CALIBRATION_SHA256,
    }


def validate_bound_m56_negative(root: str | Path) -> dict[str, Any]:
    """Validate the frozen failed M56 overlay used only as a negative comparator."""
    root = Path(root)
    freeze = root / "freeze.json"
    overlay_manifest = root / "overlay/correlation_overlay.json"
    weights = root / "overlay/correlation_head.weights.h5"
    expected = {
        freeze: BOUND_M56_FREEZE_SHA256,
        overlay_manifest: BOUND_M56_OVERLAY_MANIFEST_SHA256,
        weights: BOUND_M56_WEIGHTS_SHA256,
    }
    for path, digest in expected.items():
        if _file_sha256(path) != digest:
            raise ValueError(f"bound M56 negative artifact changed: {path}")
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    if payload.get("validation_passed") is not False:
        raise ValueError("M56 negative reference no longer records terminal failure")
    return {
        "role": "failed_negative_comparator",
        "freeze_sha256": BOUND_M56_FREEZE_SHA256,
        "overlay_manifest_sha256": BOUND_M56_OVERLAY_MANIFEST_SHA256,
        "weights_sha256": BOUND_M56_WEIGHTS_SHA256,
        "validation_passed": False,
    }


def _normalized_probit_scores(
    design: np.ndarray, response: np.ndarray, base_mean: np.ndarray
) -> np.ndarray:
    eta = np.einsum("bnk,bks->bns", design, base_mean)
    probability = np.clip(ndtr(eta), 1e-6, 1.0 - 1e-6)
    density = np.maximum(np.exp(-0.5 * np.square(eta)) / np.sqrt(2.0 * np.pi), 1e-6)
    denominator = probability * (1.0 - probability)
    score = np.einsum(
        "bnk,bns->bks",
        design,
        density * (response - probability) / denominator,
    )
    information = 1.0 + np.einsum(
        "bnk,bns->bks",
        np.square(design),
        np.square(density) / denominator,
    )
    return score / np.sqrt(information)


def _covariance_tril(marginal_scale: tf.Tensor, correlation: tf.Tensor) -> tf.Tensor:
    sigma0 = marginal_scale[..., 0]
    sigma1 = marginal_scale[..., 1]
    zeros = tf.zeros_like(sigma0)
    return tf.stack(
        [
            tf.stack([sigma0, zeros], axis=-1),
            tf.stack(
                [
                    correlation * sigma1,
                    sigma1 * tf.sqrt(tf.maximum(1.0 - tf.square(correlation), 1e-6)),
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def _correlation_from_scale_tril(scale_tril: tf.Tensor | None) -> tf.Tensor:
    if scale_tril is None:
        raise ValueError("full Laplace scale_tril is required")
    covariance = tf.matmul(scale_tril, scale_tril, transpose_b=True)
    diagonal = tf.linalg.diag_part(covariance)
    denominator = tf.sqrt(tf.maximum(diagonal[..., 0] * diagonal[..., 1], 1e-12))
    return covariance[..., 0, 1] / denominator


def _gather_beta_posterior(
    posterior: BetaPosterior, index: np.ndarray
) -> BetaPosterior:
    return BetaPosterior(
        mean=tf.gather(posterior.mean, index),
        scale=tf.gather(posterior.scale, index),
        scale_tril=(
            None
            if posterior.scale_tril is None
            else tf.gather(posterior.scale_tril, index)
        ),
    )


def _paired_context_groups(
    datasets: Sequence[FixedEffectDataset],
) -> list[np.ndarray]:
    if not datasets:
        raise ValueError("Student-t training requires paired datasets")
    grouped: dict[int, list[tuple[int, int]]] = {}
    for index, dataset in enumerate(datasets):
        metadata = dataset.metadata
        if (
            "owning_context_seed" not in metadata
            or "response_replicate" not in metadata
        ):
            raise ValueError(
                "Student-t training metadata requires owning_context_seed and response_replicate"
            )
        context = int(metadata["owning_context_seed"])
        replicate = int(metadata["response_replicate"])
        grouped.setdefault(context, []).append((replicate, index))
    result: list[np.ndarray] = []
    for context in sorted(grouped):
        rows = sorted(grouped[context])
        if [replicate for replicate, _ in rows] != [0, 1]:
            raise ValueError(
                f"owning context {context} must contain response replicates 0 and 1"
            )
        first = datasets[rows[0][1]]
        second = datasets[rows[1][1]]
        if not np.array_equal(first.X.to_numpy(), second.X.to_numpy()):
            raise ValueError("paired Student-t responses must share identical X")
        if not np.array_equal(
            first.truth_beta.to_numpy(), second.truth_beta.to_numpy()
        ):
            raise ValueError("paired Student-t responses must share identical Beta")
        result.append(np.asarray([rows[0][1], rows[1][1]], dtype=np.int64))
    return result


def _build_head(head: FixedProbitStudentTHead) -> None:
    head(tf.zeros((1, 1, len(STUDENT_T_FEATURE_NAMES)), dtype=tf.float32))


def _architecture_record() -> dict[str, Any]:
    return {
        "input_features": 15,
        "hidden_units": [64, 64, 32],
        "activation": "relu",
        "output_units": 6,
        "hidden_kernel_initializer": "seeded_glorot_uniform",
        "hidden_bias_initializer": "zeros",
        "final_kernel_initializer": "zeros",
        "initial_degrees_of_freedom": STUDENT_T_INITIAL_DF,
        "model_seed": STUDENT_T_MODEL_SEED,
        "location_limit_base_sd": STUDENT_T_LOCATION_LIMIT,
        "log_scale_limit": STUDENT_T_LOG_SCALE_LIMIT,
        "rho_limit": STUDENT_T_RHO_LIMIT,
        "degrees_of_freedom_bounds": [STUDENT_T_DF_MIN, 30.0],
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("kind") != STUDENT_T_OVERLAY_KIND:
        raise ValueError("unsupported Student-t overlay kind")
    if manifest.get("schema_version") != STUDENT_T_OVERLAY_SCHEMA_VERSION:
        raise ValueError("unsupported Student-t overlay schema")
    if manifest.get("artifact_id") != STUDENT_T_OVERLAY_ID:
        raise ValueError("Student-t overlay identifier differs")
    expected_hashes = {
        "decision_sha256": M57_DECISION_SHA256,
        "artifact_seed_audit_sha256": M57_AUDIT_SHA256,
        "preregistration_sha256": M57_PREREGISTRATION_SHA256,
    }
    for key, expected in expected_hashes.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Student-t overlay {key} differs")
    if manifest.get("weights_file") != STUDENT_T_OVERLAY_WEIGHTS:
        raise ValueError("Student-t overlay weight filename differs")
    if not _is_sha256(str(manifest.get("weights_sha256", ""))):
        raise ValueError("Student-t overlay weight hash is invalid")
    if manifest.get("architecture") != _architecture_record():
        raise ValueError("Student-t overlay architecture differs")
    if manifest.get("base_checkpoint_trainable") is not False:
        raise ValueError("Student-t overlay may not train the base checkpoint")
    if manifest.get("target_outcome_selection_performed") is not False:
        raise ValueError("Student-t overlay may not select on target outcomes")
    StudentTFeatureNormalizer.from_record(manifest.get("normalizer", {}))


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
