"""Milestone 54 v2.1 gated variable-design probit inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf

from pyhmsc.neural.inference import NeuralHmscCompatibilityError
from pyhmsc.neural.models import GatedVariableDesignBetaPosteriorModel
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.simulator import FixedEffectDataset
from pyhmsc.neural.train import (
    VariableDesignPredictiveAuxiliaryData,
    VariableDesignTrainingData,
    variable_design_training_data,
)
from pyhmsc.neural.variable_design_inference import (
    VARIABLE_DESIGN_CHECKPOINT_MANIFEST,
    VARIABLE_DESIGN_CHECKPOINT_WEIGHTS,
    VariableDesignNeuralHmscInference,
    _build_model,
    _load_calibration,
    _sha256,
)


GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION = "0.2"
GATED_VARIABLE_DESIGN_MODEL_FAMILY = "gated_variable_design_fixed_effect_beta"
GATED_VARIABLE_DESIGN_TRAINING_CORPUS_VERSION = (
    "neural_hmsc_variable_design_m54_v2_1"
)
GATED_VARIABLE_DESIGN_MEAN_CORRECTION_LIMIT = 0.5
GATED_VARIABLE_DESIGN_MSE_WEIGHT = 0.25
GATED_VARIABLE_DESIGN_PREDICTIVE_WEIGHT = 1.0
GATED_VARIABLE_DESIGN_MIN_SUPPORT_RATIO = 1.5
GATED_VARIABLE_DESIGN_MAX_SUPPORT_RATIO = 64.0
GATED_VARIABLE_DESIGN_LOG_LOSS_EPSILON = 1e-7


class GatedVariableDesignNeuralHmscInference(VariableDesignNeuralHmscInference):
    """Experimental v2.1 facade with support-gated posterior mean movement."""

    @classmethod
    def for_fixed_effects(
        cls,
        *,
        min_sites: int = 12,
        max_sites: int = 128,
        min_species: int = 2,
        max_species: int = 100,
        min_covariates: int = 2,
        max_covariates: int = 8,
        hidden_units: Sequence[int] = (48, 48),
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        probit_anchor_eta_clip: float = 6.0,
        max_design_condition_number: float = 1e6,
    ) -> "GatedVariableDesignNeuralHmscInference":
        if not 0 < min_sites <= max_sites <= 128:
            raise ValueError("gated variable-design site range must be within [1, 128]")
        if not 0 < min_species <= max_species <= 100:
            raise ValueError(
                "gated variable-design species range must be within [1, 100]"
            )
        if not 2 <= min_covariates <= max_covariates <= 8:
            raise ValueError(
                "gated variable-design covariate range must remain within [2, 8]"
            )
        if not np.isfinite(max_design_condition_number) or (
            max_design_condition_number <= 1.0
        ):
            raise ValueError("max_design_condition_number must be finite and above one")
        model = GatedVariableDesignBetaPosteriorModel(
            hidden_units=tuple(int(value) for value in hidden_units),
            mean_correction_limit=GATED_VARIABLE_DESIGN_MEAN_CORRECTION_LIMIT,
            probit_anchor_iterations=probit_anchor_iterations,
            probit_anchor_prior_precision=probit_anchor_prior_precision,
            probit_anchor_eta_clip=probit_anchor_eta_clip,
            min_support_ratio=GATED_VARIABLE_DESIGN_MIN_SUPPORT_RATIO,
            max_support_ratio=GATED_VARIABLE_DESIGN_MAX_SUPPORT_RATIO,
        )
        _build_model(
            model,
            n_sites=min_sites,
            n_species=min_species,
            n_covariates=min_covariates,
        )
        return cls(
            model=model,
            min_sites=int(min_sites),
            max_sites=int(max_sites),
            min_species=int(min_species),
            max_species=int(max_species),
            min_covariates=int(min_covariates),
            max_covariates=int(max_covariates),
            max_design_condition_number=float(max_design_condition_number),
            checkpoint_version=GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION,
            training_corpus_version=GATED_VARIABLE_DESIGN_TRAINING_CORPUS_VERSION,
        )

    @classmethod
    def load(
        cls, checkpoint: str | Path
    ) -> "GatedVariableDesignNeuralHmscInference":
        root = Path(checkpoint).expanduser().resolve()
        manifest_path = root / VARIABLE_DESIGN_CHECKPOINT_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_gated_manifest(manifest)
        ranges = manifest["shape_range"]
        model_config = manifest["model"]
        engine = cls.for_fixed_effects(
            min_sites=int(ranges["n_sites"][0]),
            max_sites=int(ranges["n_sites"][1]),
            min_species=int(ranges["n_species"][0]),
            max_species=int(ranges["n_species"][1]),
            min_covariates=int(ranges["n_covariates"][0]),
            max_covariates=int(ranges["n_covariates"][1]),
            hidden_units=tuple(int(value) for value in model_config["hidden_units"]),
            probit_anchor_iterations=int(model_config["probit_anchor_iterations"]),
            probit_anchor_prior_precision=float(
                model_config["probit_anchor_prior_precision"]
            ),
            probit_anchor_eta_clip=float(model_config["probit_anchor_eta_clip"]),
            max_design_condition_number=float(
                manifest["support"]["max_design_condition_number"]
            ),
        )
        weights = root / VARIABLE_DESIGN_CHECKPOINT_WEIGHTS
        expected_hash = str(manifest["artifacts"]["weights"]["sha256"])
        if _sha256(weights) != expected_hash:
            raise NeuralHmscCompatibilityError(
                "gated variable-design checkpoint weight hash mismatch"
            )
        engine.model.load_weights(weights)
        engine.training_corpus_version = str(manifest["training_corpus_version"])
        engine.calibration = _load_calibration(root, manifest)
        engine.checkpoint_path = root
        return engine

    def fit(
        self,
        datasets: Sequence[FixedEffectDataset],
        *,
        predictive_auxiliary: VariableDesignPredictiveAuxiliaryData,
        epochs: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        mse_weight: float = GATED_VARIABLE_DESIGN_MSE_WEIGHT,
        predictive_weight: float = GATED_VARIABLE_DESIGN_PREDICTIVE_WEIGHT,
        seed: int,
    ) -> dict[str, list[float]]:
        if not datasets:
            raise ValueError("training datasets must not be empty")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not np.isclose(mse_weight, GATED_VARIABLE_DESIGN_MSE_WEIGHT):
            raise ValueError("v2.1 coefficient MSE weight is frozen at 0.25")
        if not np.isclose(
            predictive_weight, GATED_VARIABLE_DESIGN_PREDICTIVE_WEIGHT
        ):
            raise ValueError("v2.1 predictive auxiliary weight is frozen at 1.0")
        if len(datasets) != len(predictive_auxiliary.context_seeds):
            raise ValueError(
                "coefficient and predictive auxiliary corpus counts must match"
            )
        coefficient_seeds = tuple(_dataset_seed(dataset) for dataset in datasets)
        all_seeds = (
            coefficient_seeds
            + predictive_auxiliary.context_seeds
            + predictive_auxiliary.heldout_seeds
        )
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError("v2.1 training and predictive seed roles overlap")

        coefficient_data = variable_design_training_data(datasets)
        self._check_data(coefficient_data)
        self._check_data(predictive_auxiliary.contexts)
        self._check_data(predictive_auxiliary.heldouts)
        tf.keras.utils.set_random_seed(seed)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        rng = np.random.default_rng(seed)
        history: dict[str, list[float]] = {
            "loss": [],
            "beta_nll": [],
            "beta_rmse": [],
            "score_loss": [],
            "score_brier": [],
            "score_log_loss": [],
            "scale_mean": [],
            "support_gate_mean": [],
        }
        for _ in range(int(epochs)):
            coefficient_order = rng.permutation(len(datasets))
            auxiliary_order = rng.permutation(len(datasets))
            epoch_values = {name: [] for name in history}
            for start in range(0, len(datasets), int(batch_size)):
                coefficient_batch = coefficient_order[start : start + int(batch_size)]
                auxiliary_batch = auxiliary_order[start : start + int(batch_size)]
                coefficient_inputs = _model_inputs(
                    coefficient_data, coefficient_batch
                )
                auxiliary_inputs = _model_inputs(
                    predictive_auxiliary.contexts, auxiliary_batch
                )
                truth = tf.convert_to_tensor(
                    coefficient_data.Beta[coefficient_batch], dtype=tf.float32
                )
                active = tf.cast(
                    coefficient_data.covariate_mask[coefficient_batch, :, None]
                    * coefficient_data.species_mask[coefficient_batch, None, :],
                    tf.float32,
                )
                with tf.GradientTape() as tape:
                    coefficient_posterior = self.model(
                        coefficient_inputs, training=True
                    )
                    beta_nll, beta_mse = _coefficient_loss(
                        coefficient_posterior, truth, active
                    )
                    auxiliary_posterior = self.model(auxiliary_inputs, training=True)
                    score_loss, score_brier, score_log_loss = (
                        variable_design_probit_score_loss(
                            auxiliary_posterior,
                            predictive_auxiliary.heldouts,
                            indices=auxiliary_batch,
                        )
                    )
                    loss = (
                        beta_nll
                        + GATED_VARIABLE_DESIGN_MSE_WEIGHT * beta_mse
                        + GATED_VARIABLE_DESIGN_PREDICTIVE_WEIGHT * score_loss
                    )
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    (gradient, variable)
                    for gradient, variable in zip(
                        gradients, self.model.trainable_variables
                    )
                    if gradient is not None
                )
                valid_scale = tf.boolean_mask(
                    coefficient_posterior.scale, active > 0
                )
                auxiliary_gate = self.model.support_gate(auxiliary_inputs)
                auxiliary_active = tf.cast(
                    predictive_auxiliary.contexts.covariate_mask[
                        auxiliary_batch, :, None
                    ]
                    * predictive_auxiliary.contexts.species_mask[
                        auxiliary_batch, None, :
                    ],
                    tf.float32,
                )
                valid_gate = tf.boolean_mask(auxiliary_gate, auxiliary_active > 0)
                epoch_values["loss"].append(float(loss.numpy()))
                epoch_values["beta_nll"].append(float(beta_nll.numpy()))
                epoch_values["beta_rmse"].append(
                    float(tf.sqrt(beta_mse).numpy())
                )
                epoch_values["score_loss"].append(float(score_loss.numpy()))
                epoch_values["score_brier"].append(float(score_brier.numpy()))
                epoch_values["score_log_loss"].append(
                    float(score_log_loss.numpy())
                )
                epoch_values["scale_mean"].append(
                    float(tf.reduce_mean(valid_scale).numpy())
                )
                epoch_values["support_gate_mean"].append(
                    float(tf.reduce_mean(valid_gate).numpy())
                )
            for name in history:
                history[name].append(float(np.mean(epoch_values[name])))
        return history

    def predict_support_gate(self, value: Any) -> tf.Tensor:
        """Return support-gate values after normal compatibility validation."""
        data, context = self._prepare(value)
        dimensions = self._check_data(data, expected_batch=1)
        self._check_context(context, dimensions)
        return self.model.support_gate(_model_inputs(data, np.asarray([0])))

    def check_compatibility(self, value: Any) -> dict[str, Any]:
        report = super().check_compatibility(value)
        return {
            **report,
            "model_family": GATED_VARIABLE_DESIGN_MODEL_FAMILY,
            "checkpoint_version": GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION,
        }

    def _manifest(
        self,
        *,
        weights_sha256: str | None = None,
        calibration_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._manifest(
            weights_sha256=weights_sha256,
            calibration_record=calibration_record,
        )
        payload["checkpoint_version"] = GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION
        payload["model_family"] = GATED_VARIABLE_DESIGN_MODEL_FAMILY
        payload["model"].update(
            {
                "projection_outputs": 3,
                "min_support_ratio": GATED_VARIABLE_DESIGN_MIN_SUPPORT_RATIO,
                "max_support_ratio": GATED_VARIABLE_DESIGN_MAX_SUPPORT_RATIO,
                "mean_representation": "convex_anchor_residual_support_gate",
            }
        )
        payload["training_objective"] = {
            "coefficient_nll_weight": 1.0,
            "coefficient_mse_weight": GATED_VARIABLE_DESIGN_MSE_WEIGHT,
            "predictive_weight": GATED_VARIABLE_DESIGN_PREDICTIVE_WEIGHT,
            "predictive_log_loss_weight": 0.5,
            "predictive_brier_weight": 0.5,
            "log_loss_probability_epsilon": GATED_VARIABLE_DESIGN_LOG_LOSS_EPSILON,
            "predictive_response_role": "independent_simulated_heldout",
        }
        payload["limitations"][0] = (
            "experimental Milestone 54 v2.1 redesign; not release-qualified"
        )
        return payload


def variable_design_probit_score_loss(
    posterior: BetaPosterior,
    heldouts: VariableDesignTrainingData,
    *,
    indices: np.ndarray | Sequence[int],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return the frozen equal-weight heldout probit proper-score objective."""
    selected = np.asarray(indices, dtype=int)
    design = tf.convert_to_tensor(heldouts.X[selected], dtype=tf.float32)
    response = tf.convert_to_tensor(heldouts.Y[selected], dtype=tf.float32)
    site_mask = tf.convert_to_tensor(
        heldouts.site_mask[selected, :, None], dtype=tf.float32
    )
    species_mask = tf.convert_to_tensor(
        heldouts.species_mask[selected, None, :], dtype=tf.float32
    )
    active = site_mask * species_mask
    linear_mean = tf.einsum("bnk,bks->bns", design, posterior.mean)
    linear_variance = tf.einsum(
        "bnk,bks->bns", tf.square(design), tf.square(posterior.scale)
    )
    standardized = linear_mean / tf.sqrt(1.0 + linear_variance)
    probability = 0.5 * (
        1.0 + tf.math.erf(standardized / tf.sqrt(tf.constant(2.0, tf.float32)))
    )
    clipped = tf.clip_by_value(
        probability,
        GATED_VARIABLE_DESIGN_LOG_LOSS_EPSILON,
        1.0 - GATED_VARIABLE_DESIGN_LOG_LOSS_EPSILON,
    )
    denominator = tf.maximum(tf.reduce_sum(active), 1.0)
    brier = tf.reduce_sum(tf.square(response - probability) * active) / denominator
    log_loss = -tf.reduce_sum(
        (
            response * tf.math.log(clipped)
            + (1.0 - response) * tf.math.log(1.0 - clipped)
        )
        * active
    ) / denominator
    return 0.5 * log_loss + 0.5 * brier, brier, log_loss


def _coefficient_loss(
    posterior: BetaPosterior, truth: tf.Tensor, active: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    variance = tf.maximum(tf.square(posterior.scale), 1e-12)
    point_nll = 0.5 * (
        tf.math.log(tf.constant(2.0 * np.pi, dtype=tf.float32))
        + tf.math.log(variance)
        + tf.square(truth - posterior.mean) / variance
    )
    denominator = tf.maximum(tf.reduce_sum(active), 1.0)
    nll = tf.reduce_sum(point_nll * active) / denominator
    mse = tf.reduce_sum(tf.square(truth - posterior.mean) * active) / denominator
    return nll, mse


def _model_inputs(
    data: VariableDesignTrainingData, indices: np.ndarray | Sequence[int]
) -> dict[str, np.ndarray]:
    selected = np.asarray(indices, dtype=int)
    return {
        "X": data.X[selected],
        "Y": data.Y[selected],
        "site_mask": data.site_mask[selected],
        "species_mask": data.species_mask[selected],
        "covariate_mask": data.covariate_mask[selected],
    }


def _dataset_seed(dataset: FixedEffectDataset) -> int:
    if "seed" not in dataset.metadata:
        raise ValueError("v2.1 coefficient training dataset seed is missing")
    return int(dataset.metadata["seed"])


def _validate_gated_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("checkpoint_version") != GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION:
        raise NeuralHmscCompatibilityError(
            "unsupported gated variable-design checkpoint version"
        )
    if manifest.get("model_family") != GATED_VARIABLE_DESIGN_MODEL_FAMILY:
        raise NeuralHmscCompatibilityError(
            "unsupported gated variable-design model family"
        )
    if manifest.get("distribution") != "probit":
        raise NeuralHmscCompatibilityError(
            "gated variable-design checkpoints must use probit"
        )
    model = manifest.get("model", {})
    if (
        model.get("projection_outputs") != 3
        or model.get("mean_representation")
        != "convex_anchor_residual_support_gate"
        or float(model.get("min_support_ratio", np.nan))
        != GATED_VARIABLE_DESIGN_MIN_SUPPORT_RATIO
        or float(model.get("max_support_ratio", np.nan))
        != GATED_VARIABLE_DESIGN_MAX_SUPPORT_RATIO
    ):
        raise NeuralHmscCompatibilityError(
            "gated variable-design representation metadata differs"
        )
    expected_objective = {
        "coefficient_nll_weight": 1.0,
        "coefficient_mse_weight": GATED_VARIABLE_DESIGN_MSE_WEIGHT,
        "predictive_weight": GATED_VARIABLE_DESIGN_PREDICTIVE_WEIGHT,
        "predictive_log_loss_weight": 0.5,
        "predictive_brier_weight": 0.5,
        "log_loss_probability_epsilon": GATED_VARIABLE_DESIGN_LOG_LOSS_EPSILON,
        "predictive_response_role": "independent_simulated_heldout",
    }
    if manifest.get("training_objective") != expected_objective:
        raise NeuralHmscCompatibilityError(
            "gated variable-design training objective metadata differs"
        )
