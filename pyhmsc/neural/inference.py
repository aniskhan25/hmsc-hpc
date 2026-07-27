"""Public Neural-HMSC inference facade.

The first supported API surface is intentionally narrow: fixed-effect Beta
posterior inference for compiled Python-native HMSC artifacts and matching
synthetic fixed-effect datasets. Lower-level neural modules remain available
for experiments, but this module defines the checkpoint and compatibility
boundary users should build against.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache")
)

import tensorflow as tf

from pyhmsc.neural.conditional_calibration import (
    ConditionalBetaScaleCalibration,
    apply_conditional_beta_scale_calibration,
)
from pyhmsc.neural.evaluation import predict_beta_posterior
from pyhmsc.neural.models import FixedShapeBetaPosteriorModel
from pyhmsc.neural.posterior_heads import BetaPosterior, beta_negative_log_probability
from pyhmsc.neural.simulator import FixedEffectDataset
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import (
    FixedShapeTrainingData,
    FixedShapeTrainingHistory,
    fixed_shape_training_data,
)
from pyhmsc.posterior import HmscFit
from pyhmsc.serialization import read_compiled_model


NEURAL_CHECKPOINT_VERSION = "0.4"
LEGACY_NEURAL_CHECKPOINT_VERSIONS = {"0.2", "0.3"}
NEURAL_TRAINING_CORPUS_VERSION = "0.1"
SUPPORTED_MODEL_FAMILY = "fixed_effect_beta"
CHECKPOINT_MANIFEST = "neural_checkpoint.json"
CHECKPOINT_WEIGHTS = "weights.weights.h5"
CHECKPOINT_COEFFICIENT_CALIBRATION = "coefficient_calibration.json"
COEFFICIENT_CALIBRATION_KIND = "pyhmsc_neural_coefficient_calibration"
COEFFICIENT_CALIBRATION_SCHEMA_VERSION = 1


class NeuralHmscCompatibilityError(ValueError):
    """Raised when a model artifact is outside the public Neural-HMSC API."""


@dataclass
class NeuralHmscInference:
    """Stable facade for amortized Neural-HMSC posterior inference.

    The Milestone 11 API supports fixed-effect Beta posteriors with a fixed
    site/covariate/species shape. It returns ordinary :class:`HmscFit` objects,
    so downstream code can use ``beta_mean()``, ``beta_ci()``, ``predict_mean()``,
    and existing HDF5-compatible posterior tooling.
    """

    model: FixedShapeBetaPosteriorModel
    model_family: str = SUPPORTED_MODEL_FAMILY
    distribution: str = "normal"
    formula: str = "~ x1 + x2"
    covariate_names: tuple[str, ...] = ("Intercept", "x1", "x2")
    species_names: tuple[str, ...] = field(default_factory=tuple)
    hidden_units: tuple[int, ...] = (64, 64)
    checkpoint_version: str = NEURAL_CHECKPOINT_VERSION
    training_corpus_version: str = NEURAL_TRAINING_CORPUS_VERSION
    coefficient_calibration: ConditionalBetaScaleCalibration | None = None
    coefficient_calibration_record: dict[str, Any] | None = None
    coefficient_calibration_provenance: dict[str, Any] | None = None

    @classmethod
    def for_fixed_effects(
        cls,
        *,
        n_sites: int,
        n_species: int,
        n_covariates: int = 3,
        distribution: str = "normal",
        formula: str = "~ x1 + x2",
        covariate_names: Sequence[str] | None = None,
        species_names: Sequence[str] | None = None,
        hidden_units: Sequence[int] = (64, 64),
        posterior_family: str = "auto",
        probit_anchor: str = "auto",
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        probit_anchor_eta_clip: float = 6.0,
    ) -> "NeuralHmscInference":
        """Create an untrained fixed-effect Beta inference engine."""
        covariate_names = tuple(
            covariate_names or _default_covariate_names(n_covariates)
        )
        species_names = tuple(
            species_names or [f"sp{idx + 1}" for idx in range(n_species)]
        )
        if len(covariate_names) != n_covariates:
            raise ValueError("covariate_names length must match n_covariates")
        if len(species_names) != n_species:
            raise ValueError("species_names length must match n_species")
        hidden_units = tuple(int(value) for value in hidden_units)
        posterior_family = _resolve_posterior_family(
            posterior_family, distribution=distribution
        )
        model = FixedShapeBetaPosteriorModel(
            n_sites=n_sites,
            n_covariates=n_covariates,
            n_species=n_species,
            hidden_units=hidden_units,
            posterior_family=posterior_family,
            distribution=str(distribution),
            probit_anchor=probit_anchor,
            probit_anchor_iterations=probit_anchor_iterations,
            probit_anchor_prior_precision=probit_anchor_prior_precision,
            probit_anchor_eta_clip=probit_anchor_eta_clip,
        )
        _build_fixed_shape_model(model)
        return cls(
            model=model,
            distribution=str(distribution),
            formula=str(formula),
            covariate_names=tuple(str(name) for name in covariate_names),
            species_names=tuple(str(name) for name in species_names),
            hidden_units=hidden_units,
        )

    @classmethod
    def load(cls, checkpoint: str | Path) -> "NeuralHmscInference":
        """Load a versioned Neural-HMSC checkpoint directory."""
        checkpoint = Path(checkpoint)
        manifest_path = checkpoint / CHECKPOINT_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Neural-HMSC checkpoint manifest not found: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = str(manifest.get("checkpoint_version", ""))
        if version not in {
            NEURAL_CHECKPOINT_VERSION,
            *LEGACY_NEURAL_CHECKPOINT_VERSIONS,
        }:
            raise NeuralHmscCompatibilityError(
                f"unsupported Neural-HMSC checkpoint version {version!r}; "
                f"expected one of {sorted({NEURAL_CHECKPOINT_VERSION, *LEGACY_NEURAL_CHECKPOINT_VERSIONS})!r}"
            )
        model_family = str(manifest.get("model_family", ""))
        if model_family != SUPPORTED_MODEL_FAMILY:
            raise NeuralHmscCompatibilityError(
                f"unsupported Neural-HMSC model_family {model_family!r}; "
                f"expected {SUPPORTED_MODEL_FAMILY!r}"
            )
        dimensions = manifest.get("dimensions", {})
        hidden_units = tuple(
            int(value) for value in manifest.get("hidden_units", (64, 64))
        )
        posterior_family = str(manifest.get("posterior_family", "diagonal_normal"))
        model = FixedShapeBetaPosteriorModel(
            n_sites=int(dimensions["n_sites"]),
            n_covariates=int(dimensions["n_covariates"]),
            n_species=int(dimensions["n_species"]),
            hidden_units=hidden_units,
            posterior_family=posterior_family,
            distribution=str(manifest.get("distribution", "normal")),
            probit_anchor=str(manifest.get("probit_anchor", "ridge")),
            probit_anchor_iterations=int(manifest.get("probit_anchor_iterations", 8)),
            probit_anchor_prior_precision=float(
                manifest.get("probit_anchor_prior_precision", 1.0)
            ),
            probit_anchor_eta_clip=float(manifest.get("probit_anchor_eta_clip", 6.0)),
        )
        _build_fixed_shape_model(model)
        model.load_weights(checkpoint / CHECKPOINT_WEIGHTS)
        names = manifest.get("names", {})
        formula = manifest.get("formula", {})
        covariate_names = tuple(
            str(name)
            for name in names.get(
                "covariates", _default_covariate_names(model.n_covariates)
            )
        )
        species_names = tuple(
            str(name)
            for name in names.get(
                "species", [f"sp{idx + 1}" for idx in range(model.n_species)]
            )
        )
        (
            coefficient_calibration,
            coefficient_calibration_record,
            coefficient_calibration_provenance,
        ) = _load_checkpoint_coefficient_calibration(
            checkpoint,
            manifest,
            distribution=str(manifest.get("distribution", "normal")),
            n_covariates=model.n_covariates,
            n_species=model.n_species,
            covariate_names=covariate_names,
            training_corpus_version=str(
                manifest.get("training_corpus_version", NEURAL_TRAINING_CORPUS_VERSION)
            ),
        )
        return cls(
            model=model,
            model_family=model_family,
            distribution=str(manifest.get("distribution", "normal")),
            formula=str(
                formula.get("X", "~ x1 + x2") if isinstance(formula, dict) else formula
            ),
            covariate_names=covariate_names,
            species_names=species_names,
            hidden_units=hidden_units,
            checkpoint_version=NEURAL_CHECKPOINT_VERSION,
            training_corpus_version=str(
                manifest.get("training_corpus_version", NEURAL_TRAINING_CORPUS_VERSION)
            ),
            coefficient_calibration=coefficient_calibration,
            coefficient_calibration_record=coefficient_calibration_record,
            coefficient_calibration_provenance=coefficient_calibration_provenance,
        )

    @property
    def dimensions(self) -> dict[str, int]:
        """Return the fixed shape supported by this engine."""
        return {
            "n_sites": int(self.model.n_sites),
            "n_covariates": int(self.model.n_covariates),
            "n_species": int(self.model.n_species),
        }

    def save(self, checkpoint: str | Path) -> Path:
        """Write a versioned Neural-HMSC checkpoint directory."""
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)
        _build_fixed_shape_model(self.model)
        self.model.save_weights(checkpoint / CHECKPOINT_WEIGHTS)
        calibration_record = None
        if self.coefficient_calibration is not None:
            calibration_record = _write_checkpoint_coefficient_calibration(
                checkpoint,
                self.coefficient_calibration,
                provenance=self.coefficient_calibration_provenance,
                training_corpus_version=self.training_corpus_version,
            )
        manifest = self._manifest(calibration_record=calibration_record)
        (checkpoint / CHECKPOINT_MANIFEST).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return checkpoint

    def fit(
        self,
        datasets: Sequence[FixedEffectDataset],
        *,
        epochs: int = 40,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        mse_weight: float | None = None,
        rank_mean_penalty_weight: float = 0.0,
        rank_mean_penalty_holdout_fraction: float = 0.25,
        rank_mean_penalty_holdout_folds: int = 1,
        rank_mean_penalty_crossfit_min_agreement: float = 0.75,
        rank_mean_penalty_start_fraction: float = 0.0,
        rank_mean_penalty_prevalence_threshold: float = 0.1,
        rank_mean_penalty_tolerance: float = 0.025,
        rank_mean_penalty_design_guard_weight: float = 0.0,
        rank_mean_penalty_design_guard_floor: float = 0.925,
        rank_mean_penalty_signed_mean_weight: float = 0.0,
        rank_mean_penalty_design_mean_guard_weight: float = 0.0,
        rank_mean_penalty_design_mean_guard_tolerance: float = 0.025,
        seed: int = 123,
        verbose: int = 0,
    ) -> FixedShapeTrainingHistory:
        """Train the fixed-shape Beta amortizer from public dataset objects."""
        if not datasets:
            raise ValueError("datasets must not be empty")
        for dataset in datasets:
            self._check_dataset_compatibility(dataset)
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if mse_weight is None:
            mse_weight = (
                5.0 if self.model.posterior_family == "full_covariance_normal" else 0.25
            )
        if mse_weight < 0.0:
            raise ValueError("mse_weight must be non-negative")
        if rank_mean_penalty_weight < 0.0:
            raise ValueError("rank_mean_penalty_weight must be non-negative")
        if not 0.0 < rank_mean_penalty_holdout_fraction < 1.0:
            raise ValueError(
                "rank_mean_penalty_holdout_fraction must be between zero and one"
            )
        rank_mean_penalty_holdout_folds = int(rank_mean_penalty_holdout_folds)
        if rank_mean_penalty_holdout_folds < 1:
            raise ValueError("rank_mean_penalty_holdout_folds must be at least one")
        if not 0.0 < rank_mean_penalty_crossfit_min_agreement <= 1.0:
            raise ValueError(
                "rank_mean_penalty_crossfit_min_agreement must be in (0, 1]"
            )
        if not 0.0 <= rank_mean_penalty_start_fraction < 1.0:
            raise ValueError("rank_mean_penalty_start_fraction must be in [0, 1)")
        if not 0.0 < rank_mean_penalty_prevalence_threshold < 1.0:
            raise ValueError(
                "rank_mean_penalty_prevalence_threshold must be between zero and one"
            )
        if rank_mean_penalty_tolerance <= 0.0:
            raise ValueError("rank_mean_penalty_tolerance must be positive")
        if rank_mean_penalty_design_guard_weight < 0.0:
            raise ValueError(
                "rank_mean_penalty_design_guard_weight must be non-negative"
            )
        if not 0.0 < rank_mean_penalty_design_guard_floor < 1.0:
            raise ValueError(
                "rank_mean_penalty_design_guard_floor must be between zero and one"
            )
        if rank_mean_penalty_signed_mean_weight < 0.0:
            raise ValueError(
                "rank_mean_penalty_signed_mean_weight must be non-negative"
            )
        if rank_mean_penalty_design_mean_guard_weight < 0.0:
            raise ValueError(
                "rank_mean_penalty_design_mean_guard_weight must be non-negative"
            )
        if rank_mean_penalty_design_mean_guard_tolerance <= 0.0:
            raise ValueError(
                "rank_mean_penalty_design_mean_guard_tolerance must be positive"
            )

        data = fixed_shape_training_data(datasets)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        rng = np.random.default_rng(seed)
        history = {
            "loss": [],
            "beta_rmse": [],
            "scale_mean": [],
            "rank_mean_penalty": [],
        }
        all_indices = np.arange(data.Beta.shape[0])
        if rank_mean_penalty_weight > 0.0:
            n_holdout = max(
                1, int(round(len(all_indices) * rank_mean_penalty_holdout_fraction))
            )
            if n_holdout >= len(all_indices):
                raise ValueError(
                    "rank mean penalty holdout split leaves no training datasets"
                )
            if n_holdout < rank_mean_penalty_holdout_folds:
                raise ValueError(
                    "rank_mean_penalty_holdout_folds cannot exceed holdout datasets"
                )
            split_order = np.array(all_indices, copy=True)
            rng.shuffle(split_order)
            holdout_indices = np.sort(split_order[:n_holdout])
            training_indices = np.sort(split_order[n_holdout:])
            rank_holdout_folds = [
                np.asarray(fold, dtype=int)
                for fold in np.array_split(
                    holdout_indices, rank_mean_penalty_holdout_folds
                )
                if len(fold) > 0
            ]
        else:
            holdout_indices = np.asarray([], dtype=int)
            training_indices = all_indices
            rank_holdout_folds = []
        rank_penalty_start_epoch = int(
            np.floor(float(rank_mean_penalty_start_fraction) * int(epochs))
        )

        for epoch in range(int(epochs)):
            active_rank_weight = (
                float(rank_mean_penalty_weight)
                if epoch >= rank_penalty_start_epoch
                else 0.0
            )
            order = np.array(training_indices, copy=True)
            rng.shuffle(order)
            epoch_loss = []
            epoch_rmse = []
            epoch_scale = []
            epoch_rank_penalty = []
            for start in range(0, len(order), int(batch_size)):
                batch = order[start : start + int(batch_size)]
                x_batch = tf.convert_to_tensor(data.X[batch], dtype=tf.float32)
                y_batch = tf.convert_to_tensor(data.Y[batch], dtype=tf.float32)
                beta_batch = tf.convert_to_tensor(data.Beta[batch], dtype=tf.float32)
                if active_rank_weight > 0.0:
                    holdout_batches = [
                        rng.choice(
                            fold,
                            size=min(int(batch_size), len(fold)),
                            replace=len(fold) < int(batch_size),
                        )
                        for fold in rank_holdout_folds
                    ]
                    holdout_x_folds = [
                        tf.convert_to_tensor(data.X[holdout_batch], dtype=tf.float32)
                        for holdout_batch in holdout_batches
                    ]
                    holdout_y_folds = [
                        tf.convert_to_tensor(data.Y[holdout_batch], dtype=tf.float32)
                        for holdout_batch in holdout_batches
                    ]
                    holdout_beta_folds = [
                        tf.convert_to_tensor(data.Beta[holdout_batch], dtype=tf.float32)
                        for holdout_batch in holdout_batches
                    ]
                with tf.GradientTape() as tape:
                    posterior = self.model({"X": x_batch, "Y": y_batch}, training=True)
                    nll = beta_negative_log_probability(posterior, beta_batch)
                    mse = tf.reduce_mean(tf.square(beta_batch - posterior.mean))
                    rank_penalty = tf.constant(0.0, dtype=tf.float32)
                    if active_rank_weight > 0.0:
                        holdout_posteriors = [
                            self.model({"X": holdout_x, "Y": holdout_y}, training=True)
                            for holdout_x, holdout_y in zip(
                                holdout_x_folds, holdout_y_folds
                            )
                        ]
                        if len(holdout_posteriors) > 1:
                            rank_penalty = _crossfit_holdout_rank_mean_penalty(
                                holdout_posteriors,
                                holdout_beta_folds,
                                holdout_x_folds,
                                holdout_y_folds,
                                distribution=self.distribution,
                                prevalence_threshold=(
                                    rank_mean_penalty_prevalence_threshold
                                ),
                                tolerance=rank_mean_penalty_tolerance,
                                design_guard_weight=(
                                    rank_mean_penalty_design_guard_weight
                                ),
                                design_guard_floor=rank_mean_penalty_design_guard_floor,
                                signed_mean_weight=(
                                    rank_mean_penalty_signed_mean_weight
                                ),
                                design_mean_guard_weight=(
                                    rank_mean_penalty_design_mean_guard_weight
                                ),
                                design_mean_guard_tolerance=(
                                    rank_mean_penalty_design_mean_guard_tolerance
                                ),
                                min_agreement=(
                                    rank_mean_penalty_crossfit_min_agreement
                                ),
                            )
                        else:
                            rank_penalty = _holdout_rank_mean_penalty(
                                holdout_posteriors[0],
                                holdout_beta_folds[0],
                                holdout_x_folds[0],
                                holdout_y_folds[0],
                                distribution=self.distribution,
                                prevalence_threshold=(
                                    rank_mean_penalty_prevalence_threshold
                                ),
                                tolerance=rank_mean_penalty_tolerance,
                                design_guard_weight=(
                                    rank_mean_penalty_design_guard_weight
                                ),
                                design_guard_floor=rank_mean_penalty_design_guard_floor,
                                signed_mean_weight=(
                                    rank_mean_penalty_signed_mean_weight
                                ),
                                design_mean_guard_weight=(
                                    rank_mean_penalty_design_mean_guard_weight
                                ),
                                design_mean_guard_tolerance=(
                                    rank_mean_penalty_design_mean_guard_tolerance
                                ),
                            )
                    loss = (
                        nll
                        + float(mse_weight) * mse
                        + active_rank_weight * rank_penalty
                    )
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    (gradient, variable)
                    for gradient, variable in zip(
                        gradients, self.model.trainable_variables
                    )
                    if gradient is not None
                )
                rmse = tf.sqrt(tf.reduce_mean(tf.square(beta_batch - posterior.mean)))
                epoch_loss.append(float(loss.numpy()))
                epoch_rmse.append(float(rmse.numpy()))
                epoch_scale.append(float(tf.reduce_mean(posterior.scale).numpy()))
                epoch_rank_penalty.append(float(rank_penalty.numpy()))
            history["loss"].append(float(np.mean(epoch_loss)))
            history["beta_rmse"].append(float(np.mean(epoch_rmse)))
            history["scale_mean"].append(float(np.mean(epoch_scale)))
            history["rank_mean_penalty"].append(float(np.mean(epoch_rank_penalty)))
            if verbose:
                print(
                    f"epoch {epoch + 1}/{epochs} "
                    f"loss={history['loss'][-1]:.4f} "
                    f"beta_rmse={history['beta_rmse'][-1]:.4f} "
                    f"scale_mean={history['scale_mean'][-1]:.4f} "
                    f"rank_mean_penalty={history['rank_mean_penalty'][-1]:.4f}"
                )

        return FixedShapeTrainingHistory(
            loss=history["loss"],
            beta_rmse=history["beta_rmse"],
            scale_mean=history["scale_mean"],
            rank_mean_penalty=history["rank_mean_penalty"],
        )

    def check_compatibility(self, model_or_compiled_artifact: Any) -> dict[str, Any]:
        """Return a compatibility summary or raise a clear compatibility error."""
        data, context = self._prepare_inference_data(model_or_compiled_artifact)
        self._check_training_data_shape(data, batch_size=1)
        if self.coefficient_calibration is not None:
            try:
                self.coefficient_calibration.validate_domain(
                    distribution=context.distribution,
                    n_covariates=int(data.X.shape[2]),
                    n_species=int(data.Y.shape[2]),
                    coefficient_names=context.covariate_names,
                )
            except ValueError as exc:
                raise NeuralHmscCompatibilityError(
                    f"compiled artifact calibration domain mismatch: {exc}"
                ) from exc
        return {
            "compatible": True,
            "model_family": self.model_family,
            "posterior_family": self.model.posterior_family,
            "probit_anchor": self.model.probit_anchor,
            "distribution": context.distribution,
            "formula": context.formula,
            "dimensions": {
                "n_sites": int(data.X.shape[1]),
                "n_covariates": int(data.X.shape[2]),
                "n_species": int(data.Y.shape[2]),
            },
        }

    def infer(
        self,
        model_or_compiled_artifact: Any,
        *,
        draws: int = 1000,
        chains: int = 1,
        seed: int | None = None,
        output: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HmscFit:
        """Infer a fixed-effect Beta posterior and return an ``HmscFit``."""
        if draws <= 0:
            raise ValueError("draws must be positive")
        if chains <= 0:
            raise ValueError("chains must be positive")
        data, context = self._prepare_inference_data(model_or_compiled_artifact)
        self._check_training_data_shape(data, batch_size=1)
        posterior = self._calibrated_posterior(
            predict_beta_posterior(self.model, data),
            data=data,
            context=context,
        )
        posterior_metadata = {"neural_api": self._manifest()}
        if metadata:
            posterior_metadata.update(metadata)
        if output is None:
            with tempfile.TemporaryDirectory(prefix="neural-hmsc-fit-") as tmp:
                path = write_beta_posterior_hdf5(
                    posterior,
                    Path(tmp) / "posterior.h5",
                    covariate_names=context.covariate_names,
                    species_names=context.species_names,
                    distribution=context.distribution,
                    formula=context.formula,
                    chains=chains,
                    draws=draws,
                    seed=seed,
                    metadata=posterior_metadata,
                    calibration=self.coefficient_calibration,
                )
                fit = HmscFit.from_file(path)
        else:
            path = write_beta_posterior_hdf5(
                posterior,
                output,
                covariate_names=context.covariate_names,
                species_names=context.species_names,
                distribution=context.distribution,
                formula=context.formula,
                chains=chains,
                draws=draws,
                seed=seed,
                metadata=posterior_metadata,
                calibration=self.coefficient_calibration,
            )
            fit = HmscFit.from_file(path)
            fit.output_file = path
        return fit

    def predict_beta_posterior(
        self,
        model_or_compiled_artifact: Any,
        *,
        calibrated: bool = True,
    ) -> BetaPosterior:
        """Return the fixed-effect ``Beta`` posterior.

        Packaged coefficient calibration is applied by default. Set
        ``calibrated=False`` only for diagnostics against the raw amortizer.
        """
        data, context = self._prepare_inference_data(model_or_compiled_artifact)
        self._check_training_data_shape(data, batch_size=None)
        posterior = predict_beta_posterior(self.model, data)
        if not calibrated:
            return posterior
        return self._calibrated_posterior(posterior, data=data, context=context)

    def _calibrated_posterior(
        self,
        posterior: BetaPosterior,
        *,
        data: FixedShapeTrainingData,
        context: "_InferenceContext",
    ) -> BetaPosterior:
        if self.coefficient_calibration is None:
            return posterior
        return apply_conditional_beta_scale_calibration(
            posterior,
            self.coefficient_calibration,
            X=data.X,
            Y=data.Y,
            distribution=context.distribution,
            coefficient_names=context.covariate_names,
        )

    def _prepare_inference_data(
        self, value: Any
    ) -> tuple[FixedShapeTrainingData, "_InferenceContext"]:
        if isinstance(value, FixedShapeTrainingData):
            return value, _InferenceContext(
                distribution=self.distribution,
                formula=self.formula,
                covariate_names=self.covariate_names,
                species_names=self.species_names,
            )
        if isinstance(value, FixedEffectDataset):
            self._check_dataset_compatibility(value)
            data = fixed_shape_training_data([value])
            return data, _InferenceContext(
                distribution=str(value.metadata.get("distribution", self.distribution)),
                formula=self.formula,
                covariate_names=tuple(str(name) for name in value.truth_beta.index),
                species_names=tuple(str(name) for name in value.truth_beta.columns),
            )
        if isinstance(value, dict):
            return self._prepare_mapping(value)
        if isinstance(value, (str, Path)):
            return self._prepare_compiled_artifact(value)
        raise NeuralHmscCompatibilityError(
            "NeuralHmscInference.infer supports FixedEffectDataset, {'X': ..., 'Y': ...}, "
            "or a Python-native compiled init.json/directory"
        )

    def _prepare_mapping(
        self, value: dict[str, Any]
    ) -> tuple[FixedShapeTrainingData, "_InferenceContext"]:
        if "X" not in value or "Y" not in value:
            raise NeuralHmscCompatibilityError("mapping input must contain 'X' and 'Y'")
        X = _as_design_array(value["X"])
        Y = _as_response_array(value["Y"])
        data = _training_data_from_arrays(X, Y)
        formula = str(value.get("formula", self.formula))
        distribution = str(value.get("distribution", self.distribution))
        covariate_names = tuple(
            str(name) for name in value.get("covariate_names", self.covariate_names)
        )
        species_names = tuple(
            str(name) for name in value.get("species_names", self.species_names)
        )
        return data, _InferenceContext(
            distribution=distribution,
            formula=formula,
            covariate_names=covariate_names,
            species_names=species_names,
        )

    def _prepare_compiled_artifact(
        self, value: str | Path
    ) -> tuple[FixedShapeTrainingData, "_InferenceContext"]:
        path = Path(value)
        if path.is_dir():
            path = path / "init.json"
        metadata, arrays = read_compiled_model(path)
        _check_compiled_artifact_supported(metadata)
        required = {"X", "Y"}
        missing = sorted(required.difference(arrays))
        if missing:
            raise NeuralHmscCompatibilityError(
                f"compiled artifact is missing arrays: {missing}"
            )
        X = np.asarray(arrays["X"], dtype=np.float32)
        Y = np.asarray(arrays["Y"], dtype=np.float32)
        data = _training_data_from_arrays(X, Y)
        names = metadata.get("names", {})
        formula = metadata.get("formula", {})
        return data, _InferenceContext(
            distribution=str(metadata.get("distribution", self.distribution)),
            formula=str(
                formula.get("X", self.formula) if isinstance(formula, dict) else formula
            ),
            covariate_names=tuple(
                str(name) for name in names.get("covariates", self.covariate_names)
            ),
            species_names=tuple(
                str(name) for name in names.get("species", self.species_names)
            ),
        )

    def _check_dataset_compatibility(self, dataset: FixedEffectDataset) -> None:
        if dataset.Y.shape != (self.model.n_sites, self.model.n_species):
            raise NeuralHmscCompatibilityError(
                "dataset response shape does not match checkpoint; "
                f"got {dataset.Y.shape}, expected {(self.model.n_sites, self.model.n_species)}"
            )
        if len(dataset.truth_beta.index) != self.model.n_covariates:
            raise NeuralHmscCompatibilityError(
                "dataset Beta covariate shape does not match checkpoint; "
                f"got {len(dataset.truth_beta.index)}, expected {self.model.n_covariates}"
            )
        distribution = str(dataset.metadata.get("distribution", self.distribution))
        if distribution != self.distribution:
            raise NeuralHmscCompatibilityError(
                f"dataset distribution {distribution!r} does not match checkpoint distribution {self.distribution!r}"
            )

    def _check_training_data_shape(
        self, data: FixedShapeTrainingData, *, batch_size: int | None
    ) -> None:
        if data.X.ndim != 3:
            raise NeuralHmscCompatibilityError(
                f"X must be a 3D batch; got shape {data.X.shape}"
            )
        if data.Y.ndim != 3:
            raise NeuralHmscCompatibilityError(
                f"Y must be a 3D batch; got shape {data.Y.shape}"
            )
        if batch_size is not None and data.X.shape[0] != batch_size:
            raise NeuralHmscCompatibilityError(
                f"X batch size {data.X.shape[0]} does not match expected {batch_size}"
            )
        if batch_size is not None and data.Y.shape[0] != batch_size:
            raise NeuralHmscCompatibilityError(
                f"Y batch size {data.Y.shape[0]} does not match expected {batch_size}"
            )
        expected_tail = (self.model.n_sites, self.model.n_covariates)
        if data.X.shape[1:] != expected_tail:
            raise NeuralHmscCompatibilityError(
                f"X shape {data.X.shape} does not match checkpoint tail {expected_tail}"
            )
        expected_y_tail = (self.model.n_sites, self.model.n_species)
        if data.Y.shape[1:] != expected_y_tail:
            raise NeuralHmscCompatibilityError(
                f"Y shape {data.Y.shape} does not match checkpoint tail {expected_y_tail}"
            )

    def _manifest(
        self, *, calibration_record: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        manifest = {
            "checkpoint_version": self.checkpoint_version,
            "training_corpus_version": self.training_corpus_version,
            "model_family": self.model_family,
            "posterior_family": self.model.posterior_family,
            "distribution": self.distribution,
            "probit_anchor": self.model.probit_anchor,
            "probit_anchor_iterations": self.model.probit_anchor_iterations,
            "probit_anchor_prior_precision": self.model.probit_anchor_prior_precision,
            "probit_anchor_eta_clip": self.model.probit_anchor_eta_clip,
            "formula": {"X": self.formula},
            "dimensions": self.dimensions,
            "names": {
                "covariates": list(self.covariate_names),
                "species": list(self.species_names),
            },
            "hidden_units": list(self.hidden_units),
            "limitations": _PUBLIC_LIMITATIONS,
        }
        record = (
            self.coefficient_calibration_record
            if calibration_record is None
            else calibration_record
        )
        if record is not None:
            manifest["coefficient_calibration"] = dict(record)
        return manifest


def package_neural_hmsc_coefficient_calibration(
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    *,
    calibration_metadata: dict[str, Any],
    provenance: dict[str, Any],
) -> Path:
    """Bind a frozen external-monotone calibration to an existing checkpoint.

    Model weights are copied byte-for-byte. This function validates and packages
    retained metadata; it never fits or selects calibration parameters.
    """
    source = Path(source_checkpoint).expanduser().resolve()
    output = Path(output_checkpoint).expanduser().resolve()
    if source == output:
        raise ValueError("output_checkpoint must differ from source_checkpoint")
    if output.exists():
        raise FileExistsError(f"output checkpoint already exists: {output}")
    engine = NeuralHmscInference.load(source)
    if engine.coefficient_calibration is not None:
        raise ValueError("source checkpoint already contains coefficient calibration")
    calibration = ConditionalBetaScaleCalibration.from_metadata(
        dict(calibration_metadata)
    )
    _validate_packaged_calibration_domain(
        calibration,
        distribution=engine.distribution,
        n_covariates=engine.model.n_covariates,
        n_species=engine.model.n_species,
        covariate_names=engine.covariate_names,
    )
    _validate_calibration_provenance(
        provenance,
        training_corpus_version=engine.training_corpus_version,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    try:
        record = _write_checkpoint_coefficient_calibration(
            output,
            calibration,
            provenance=provenance,
            training_corpus_version=engine.training_corpus_version,
            calibration_metadata=calibration_metadata,
        )
        manifest_path = output / CHECKPOINT_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checkpoint_version"] = NEURAL_CHECKPOINT_VERSION
        manifest["coefficient_calibration"] = record
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        NeuralHmscInference.load(output)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output


@dataclass(frozen=True)
class _InferenceContext:
    distribution: str
    formula: str
    covariate_names: tuple[str, ...]
    species_names: tuple[str, ...]


_PUBLIC_LIMITATIONS = [
    "fixed-effect Beta posterior inference only",
    "fixed site/covariate/species shape per checkpoint",
    "no trait, phylogeny, iid latent, spatial latent, random-effect, or detection submodel inference",
    "uncertainty is an amortized neural approximation, not an MCMC posterior",
]


def _write_checkpoint_coefficient_calibration(
    checkpoint: Path,
    calibration: ConditionalBetaScaleCalibration,
    *,
    provenance: dict[str, Any] | None,
    training_corpus_version: str,
    calibration_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if provenance is None:
        raise ValueError("packaged coefficient calibration requires provenance")
    _validate_packaged_calibration_domain(
        calibration,
        distribution=calibration.distribution,
        n_covariates=calibration.n_covariates,
        n_species=calibration.n_species,
        covariate_names=calibration.coefficient_names,
    )
    _validate_calibration_provenance(
        provenance,
        training_corpus_version=training_corpus_version,
    )
    metadata = (
        calibration.to_metadata()
        if calibration_metadata is None
        else dict(calibration_metadata)
    )
    reconstructed = ConditionalBetaScaleCalibration.from_metadata(metadata)
    _validate_packaged_calibration_domain(
        reconstructed,
        distribution=calibration.distribution,
        n_covariates=calibration.n_covariates,
        n_species=calibration.n_species,
        covariate_names=calibration.coefficient_names,
    )
    calibration_sha256 = hashlib.sha256(_canonical_json_bytes(metadata)).hexdigest()
    payload = {
        "schema_version": COEFFICIENT_CALIBRATION_SCHEMA_VERSION,
        "kind": COEFFICIENT_CALIBRATION_KIND,
        "parameter": "Beta",
        "method": calibration.method,
        "calibration_sha256": calibration_sha256,
        "calibration": metadata,
        "provenance": dict(provenance),
    }
    path = checkpoint / CHECKPOINT_COEFFICIENT_CALIBRATION
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "schema_version": COEFFICIENT_CALIBRATION_SCHEMA_VERSION,
        "kind": COEFFICIENT_CALIBRATION_KIND,
        "path": CHECKPOINT_COEFFICIENT_CALIBRATION,
        "sha256": _file_sha256(path),
        "calibration_sha256": calibration_sha256,
        "parameter": "Beta",
        "method": calibration.method,
        "distribution": calibration.distribution,
        "n_covariates": calibration.n_covariates,
        "n_species": calibration.n_species,
        "coefficient_names": list(calibration.coefficient_names),
    }


def _load_checkpoint_coefficient_calibration(
    checkpoint: Path,
    manifest: dict[str, Any],
    *,
    distribution: str,
    n_covariates: int,
    n_species: int,
    covariate_names: tuple[str, ...],
    training_corpus_version: str,
) -> tuple[
    ConditionalBetaScaleCalibration | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    record = manifest.get("coefficient_calibration")
    if record is None:
        return None, None, None
    if str(manifest.get("checkpoint_version")) != NEURAL_CHECKPOINT_VERSION:
        raise NeuralHmscCompatibilityError(
            "packaged coefficient calibration requires the current checkpoint version"
        )
    if not isinstance(record, dict):
        raise NeuralHmscCompatibilityError(
            "checkpoint coefficient_calibration record must be an object"
        )
    expected_record = {
        "schema_version": COEFFICIENT_CALIBRATION_SCHEMA_VERSION,
        "kind": COEFFICIENT_CALIBRATION_KIND,
        "path": CHECKPOINT_COEFFICIENT_CALIBRATION,
        "parameter": "Beta",
        "method": "external_context_monotone_scale",
        "distribution": distribution,
        "n_covariates": int(n_covariates),
        "n_species": int(n_species),
        "coefficient_names": list(covariate_names),
    }
    for key, expected in expected_record.items():
        if record.get(key) != expected:
            raise NeuralHmscCompatibilityError(
                f"checkpoint coefficient calibration {key} mismatch"
            )
    expected_hash = str(record.get("sha256", ""))
    expected_calibration_hash = str(record.get("calibration_sha256", ""))
    if not _is_sha256(expected_hash) or not _is_sha256(expected_calibration_hash):
        raise NeuralHmscCompatibilityError(
            "checkpoint coefficient calibration hash is invalid"
        )
    path = (checkpoint / str(record["path"])).resolve()
    try:
        path.relative_to(checkpoint.resolve())
    except ValueError as exc:
        raise NeuralHmscCompatibilityError(
            "checkpoint coefficient calibration path escapes checkpoint"
        ) from exc
    if _file_sha256(path) != expected_hash:
        raise NeuralHmscCompatibilityError(
            "checkpoint coefficient calibration artifact hash mismatch"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in {
        "schema_version": COEFFICIENT_CALIBRATION_SCHEMA_VERSION,
        "kind": COEFFICIENT_CALIBRATION_KIND,
        "parameter": "Beta",
        "method": "external_context_monotone_scale",
        "calibration_sha256": expected_calibration_hash,
    }.items():
        if payload.get(key) != expected:
            raise NeuralHmscCompatibilityError(
                f"coefficient calibration artifact {key} mismatch"
            )
    metadata = payload.get("calibration")
    if not isinstance(metadata, dict):
        raise NeuralHmscCompatibilityError(
            "coefficient calibration artifact lacks metadata"
        )
    if (
        hashlib.sha256(_canonical_json_bytes(metadata)).hexdigest()
        != expected_calibration_hash
    ):
        raise NeuralHmscCompatibilityError(
            "coefficient calibration metadata hash mismatch"
        )
    provenance = payload.get("provenance")
    try:
        _validate_calibration_provenance(
            provenance,
            training_corpus_version=training_corpus_version,
        )
        calibration = ConditionalBetaScaleCalibration.from_metadata(metadata)
        _validate_packaged_calibration_domain(
            calibration,
            distribution=distribution,
            n_covariates=n_covariates,
            n_species=n_species,
            covariate_names=covariate_names,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NeuralHmscCompatibilityError(
            f"invalid packaged coefficient calibration: {exc}"
        ) from exc
    return calibration, dict(record), dict(provenance)


def _validate_packaged_calibration_domain(
    calibration: ConditionalBetaScaleCalibration,
    *,
    distribution: str,
    n_covariates: int,
    n_species: int,
    covariate_names: Sequence[str],
) -> None:
    if calibration.method != "external_context_monotone_scale":
        raise ValueError(
            "public checkpoint packaging supports external_context_monotone_scale only"
        )
    calibration.validate_domain(
        distribution=distribution,
        n_covariates=n_covariates,
        n_species=n_species,
        coefficient_names=covariate_names,
    )


def _validate_calibration_provenance(
    provenance: Any,
    *,
    training_corpus_version: str,
) -> None:
    if not isinstance(provenance, dict):
        raise ValueError("coefficient calibration provenance must be an object")
    expected = {
        "kind": "independent_simulation_calibration_provenance",
        "training_corpus_version": str(training_corpus_version),
        "calibration_training_role": "independent_simulation",
        "target_response_used_for_calibration": False,
        "packaging_refit_performed": False,
        "packaging_reselection_performed": False,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"coefficient calibration provenance {key} mismatch")
    if not _is_sha256(str(provenance.get("source_run_metadata_sha256", ""))):
        raise ValueError("coefficient calibration provenance source hash is invalid")
    if not isinstance(provenance.get("source_seed"), int):
        raise ValueError("coefficient calibration provenance source_seed is invalid")
    if not str(provenance.get("source_run_metadata_path", "")):
        raise ValueError("coefficient calibration provenance source path is missing")


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


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


def _check_compiled_artifact_supported(metadata: dict[str, Any]) -> None:
    capabilities = metadata.get("capabilities", {})
    unsupported = []
    for key in ("random_levels", "traits", "phylogeny", "spatial"):
        if capabilities.get(key):
            unsupported.append(key)
    if metadata.get("random_levels"):
        unsupported.append("random_levels")
    if unsupported:
        unique = sorted(set(unsupported))
        raise NeuralHmscCompatibilityError(
            "this Neural-HMSC checkpoint supports fixed-effect Beta artifacts only; "
            f"unsupported compiled features: {unique}"
        )
    distribution = str(metadata.get("distribution", ""))
    if distribution not in {"normal", "probit", "poisson"}:
        raise NeuralHmscCompatibilityError(f"unsupported distribution {distribution!r}")


def _training_data_from_arrays(X: np.ndarray, Y: np.ndarray) -> FixedShapeTrainingData:
    if X.ndim != 2:
        raise NeuralHmscCompatibilityError(
            f"X must be a 2D design matrix; got shape {X.shape}"
        )
    if Y.ndim != 2:
        raise NeuralHmscCompatibilityError(
            f"Y must be a 2D response matrix; got shape {Y.shape}"
        )
    if X.shape[0] != Y.shape[0]:
        raise NeuralHmscCompatibilityError("X and Y must have the same number of sites")
    beta = np.zeros((1, X.shape[1], Y.shape[1]), dtype=np.float32)
    return FixedShapeTrainingData(
        X=X[None, :, :].astype(np.float32),
        Y=Y[None, :, :].astype(np.float32),
        Beta=beta,
    )


def _as_design_array(value: Any) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        return value.to_numpy(dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _as_response_array(value: Any) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        return value.to_numpy(dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _default_covariate_names(n_covariates: int) -> list[str]:
    if n_covariates == 3:
        return ["Intercept", "x1", "x2"]
    return ["Intercept"] + [f"x{idx}" for idx in range(1, n_covariates)]


def _resolve_posterior_family(value: str, *, distribution: str) -> str:
    family = str(value).lower()
    if family == "auto":
        return (
            "full_covariance_normal"
            if str(distribution).lower() == "poisson"
            else "diagonal_normal"
        )
    if family not in {"diagonal_normal", "full_covariance_normal"}:
        raise ValueError(
            "posterior_family must be 'auto', 'diagonal_normal', or 'full_covariance_normal'"
        )
    return family


def _build_fixed_shape_model(model: FixedShapeBetaPosteriorModel) -> None:
    X = np.zeros((1, model.n_sites, model.n_covariates), dtype=np.float32)
    Y = np.zeros((1, model.n_sites, model.n_species), dtype=np.float32)
    model({"X": X, "Y": Y}, training=False)


def _holdout_rank_mean_penalty(
    posterior: BetaPosterior,
    beta_true: tf.Tensor,
    design: tf.Tensor,
    response: tf.Tensor,
    *,
    distribution: str,
    prevalence_threshold: float,
    tolerance: float,
    design_guard_weight: float,
    design_guard_floor: float,
    signed_mean_weight: float,
    design_mean_guard_weight: float,
    design_mean_guard_tolerance: float,
) -> tf.Tensor:
    """Return prevalence-weighted rank and design-coverage penalty."""
    beta_true = tf.cast(beta_true, posterior.mean.dtype)
    design = tf.cast(design, posterior.mean.dtype)
    response = tf.cast(response, posterior.mean.dtype)
    prevalence = tf.reduce_mean(response, axis=1)
    rare_species = prevalence <= tf.cast(prevalence_threshold, posterior.mean.dtype)
    intermediate_species = tf.logical_and(
        prevalence > tf.cast(prevalence_threshold, posterior.mean.dtype),
        prevalence <= tf.cast(0.3, posterior.mean.dtype),
    )
    common_species = prevalence > tf.cast(0.3, posterior.mean.dtype)
    rare_weight = tf.cast(rare_species[:, None, :], posterior.mean.dtype)
    weighted_prevalence = (
        rare_weight
        + tf.cast(intermediate_species[:, None, :], posterior.mean.dtype) * 0.35
        + tf.cast(common_species[:, None, :], posterior.mean.dtype) * 0.10
    )
    rank_probability = _tf_normal_cdf(
        (beta_true - posterior.mean) / tf.maximum(posterior.scale, 1e-6)
    )

    def weighted_rank_loss(weights: tf.Tensor) -> tf.Tensor:
        denominator = tf.reduce_sum(weights)
        rank_mean = tf.reduce_sum(rank_probability * weights) / tf.maximum(
            denominator, tf.cast(1.0, posterior.mean.dtype)
        )
        active = tf.cast(denominator >= 1.0, posterior.mean.dtype)
        return active * tf.square((rank_mean - 0.5) / float(tolerance))

    losses = [
        weighted_rank_loss(rare_weight),
        0.25 * weighted_rank_loss(weighted_prevalence),
    ]
    for coefficient_index in range(int(posterior.mean.shape[1])):
        coefficient_mask = tf.one_hot(
            coefficient_index,
            int(posterior.mean.shape[1]),
            dtype=posterior.mean.dtype,
        )[None, :, None]
        losses.append(weighted_rank_loss(rare_weight * coefficient_mask))
    rank_loss = tf.reduce_mean(tf.stack(losses))
    if signed_mean_weight > 0.0:
        rank_loss = rank_loss + float(
            signed_mean_weight
        ) * _signed_posterior_mean_rank_penalty(
            posterior,
            beta_true,
            rare_weight,
            tolerance=tolerance,
        )
    if design_mean_guard_weight > 0.0:
        rank_loss = rank_loss + float(
            design_mean_guard_weight
        ) * _design_rank_mean_guard_penalty(
            posterior,
            beta_true,
            design,
            distribution=distribution,
            tolerance=design_mean_guard_tolerance,
        )
    if design_guard_weight <= 0.0:
        return rank_loss
    return rank_loss + float(design_guard_weight) * _design_coverage_guard_penalty(
        posterior,
        beta_true,
        design,
        distribution=distribution,
        coverage_floor=design_guard_floor,
    )


def _crossfit_holdout_rank_mean_penalty(
    posteriors: Sequence[BetaPosterior],
    beta_true_folds: Sequence[tf.Tensor],
    design_folds: Sequence[tf.Tensor],
    response_folds: Sequence[tf.Tensor],
    *,
    distribution: str,
    prevalence_threshold: float,
    tolerance: float,
    design_guard_weight: float,
    design_guard_floor: float,
    signed_mean_weight: float,
    design_mean_guard_weight: float,
    design_mean_guard_tolerance: float,
    min_agreement: float,
) -> tf.Tensor:
    """Return multi-holdout rank loss with stable signed-mean gating."""
    base_losses = []
    signed_losses = []
    rare_deltas = []
    rare_active = []
    design_deltas = []
    design_active = []
    for posterior, beta_true, design, response in zip(
        posteriors, beta_true_folds, design_folds, response_folds
    ):
        beta_true = tf.cast(beta_true, posterior.mean.dtype)
        design = tf.cast(design, posterior.mean.dtype)
        response = tf.cast(response, posterior.mean.dtype)
        prevalence = tf.reduce_mean(response, axis=1)
        rare_weight = tf.cast(
            (prevalence <= tf.cast(prevalence_threshold, posterior.mean.dtype))[
                :, None, :
            ],
            posterior.mean.dtype,
        )
        base_losses.append(
            _holdout_rank_mean_penalty(
                posterior,
                beta_true,
                design,
                response,
                distribution=distribution,
                prevalence_threshold=prevalence_threshold,
                tolerance=tolerance,
                design_guard_weight=design_guard_weight,
                design_guard_floor=design_guard_floor,
                signed_mean_weight=0.0,
                design_mean_guard_weight=design_mean_guard_weight,
                design_mean_guard_tolerance=design_mean_guard_tolerance,
            )
        )
        signed_losses.append(
            _signed_posterior_mean_rank_penalty(
                posterior,
                beta_true,
                rare_weight,
                tolerance=tolerance,
            )
        )
        fold_rare_deltas, fold_rare_active = _rare_rank_delta_vector(
            posterior,
            beta_true,
            rare_weight,
        )
        rare_deltas.append(fold_rare_deltas)
        rare_active.append(fold_rare_active)
        fold_design_deltas, fold_design_active = _design_rank_delta_vector(
            posterior,
            beta_true,
            design,
            distribution=distribution,
        )
        design_deltas.append(fold_design_deltas)
        design_active.append(fold_design_active)

    base_loss = tf.reduce_mean(tf.stack(base_losses))
    if signed_mean_weight <= 0.0:
        return base_loss
    stability_gate = _crossfit_signed_rank_stability_gate(
        rare_deltas,
        rare_active,
        tolerance=tolerance,
        min_agreement=min_agreement,
    )
    design_gate = _crossfit_design_rank_gate(
        design_deltas,
        design_active,
        tolerance=design_mean_guard_tolerance,
    )
    signed_gate = tf.stop_gradient(stability_gate * design_gate)
    signed_loss = tf.reduce_mean(tf.stack(signed_losses))
    return base_loss + float(signed_mean_weight) * signed_gate * signed_loss


def _rare_rank_delta_vector(
    posterior: BetaPosterior,
    beta_true: tf.Tensor,
    rare_weight: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return rare rank mean deltas overall and by coefficient."""
    dtype = posterior.mean.dtype
    rank_probability = _tf_normal_cdf(
        (beta_true - posterior.mean) / tf.maximum(posterior.scale, 1e-6)
    )
    weights = [rare_weight]
    for coefficient_index in range(int(posterior.mean.shape[1])):
        coefficient_mask = tf.one_hot(
            coefficient_index,
            int(posterior.mean.shape[1]),
            dtype=dtype,
        )[None, :, None]
        weights.append(rare_weight * coefficient_mask)
    deltas = []
    active = []
    for local_weights in weights:
        denominator = tf.reduce_sum(local_weights)
        rank_mean = tf.reduce_sum(rank_probability * local_weights) / tf.maximum(
            denominator, tf.cast(1.0, dtype)
        )
        deltas.append(rank_mean - tf.cast(0.5, dtype))
        active.append(tf.cast(denominator >= 1.0, dtype))
    return tf.stack(deltas), tf.stack(active)


def _design_rank_delta_vector(
    posterior: BetaPosterior,
    beta_true: tf.Tensor,
    design: tf.Tensor,
    *,
    distribution: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return rank mean deltas for medium/high design-information strata."""
    dtype = posterior.mean.dtype
    design_information = _expected_design_information(
        posterior.mean,
        design,
        distribution=distribution,
    )
    flattened = tf.reshape(design_information, [-1])
    sorted_values = tf.sort(flattened)
    n_values = tf.shape(sorted_values)[0]
    low = sorted_values[tf.maximum(0, n_values // 3)]
    high = sorted_values[tf.maximum(0, (2 * n_values) // 3)]
    masks = (
        tf.logical_and(design_information > low, design_information <= high),
        design_information > high,
    )
    rank_probability = _tf_normal_cdf(
        (beta_true - posterior.mean) / tf.maximum(posterior.scale, 1e-6)
    )
    deltas = []
    active = []
    for mask in masks:
        weights = tf.cast(mask, dtype)
        denominator = tf.reduce_sum(weights)
        rank_mean = tf.reduce_sum(rank_probability * weights) / tf.maximum(
            denominator, tf.cast(1.0, dtype)
        )
        deltas.append(rank_mean - tf.cast(0.5, dtype))
        active.append(tf.cast(denominator >= 1.0, dtype))
    return tf.stack(deltas), tf.stack(active)


def _crossfit_signed_rank_stability_gate(
    deltas: Sequence[tf.Tensor],
    active: Sequence[tf.Tensor],
    *,
    tolerance: float,
    min_agreement: float,
) -> tf.Tensor:
    """Return one when active rare-rank directions are stable across folds."""
    delta_matrix = tf.stack(deltas)
    active_matrix = tf.stack(active)
    dtype = delta_matrix.dtype
    active_counts = tf.reduce_sum(active_matrix, axis=0)
    signed_votes = tf.reduce_sum(tf.sign(delta_matrix) * active_matrix, axis=0)
    agreement = tf.abs(signed_votes) / tf.maximum(active_counts, tf.cast(1.0, dtype))
    mean_delta = tf.reduce_sum(delta_matrix * active_matrix, axis=0) / tf.maximum(
        active_counts, tf.cast(1.0, dtype)
    )
    component_active = active_counts >= 2.0
    component_ok = tf.logical_or(
        tf.logical_not(component_active),
        tf.logical_or(
            agreement >= tf.cast(min_agreement, dtype),
            tf.abs(mean_delta) <= tf.cast(tolerance, dtype),
        ),
    )
    any_active = tf.reduce_any(component_active)
    return tf.cast(tf.logical_and(any_active, tf.reduce_all(component_ok)), dtype)


def _crossfit_design_rank_gate(
    deltas: Sequence[tf.Tensor],
    active: Sequence[tf.Tensor],
    *,
    tolerance: float,
) -> tf.Tensor:
    """Return one when medium/high design rank means remain within tolerance."""
    delta_matrix = tf.stack(deltas)
    active_matrix = tf.stack(active)
    dtype = delta_matrix.dtype
    active_counts = tf.reduce_sum(active_matrix, axis=0)
    mean_delta = tf.reduce_sum(delta_matrix * active_matrix, axis=0) / tf.maximum(
        active_counts, tf.cast(1.0, dtype)
    )
    component_active = active_counts >= 1.0
    component_ok = tf.logical_or(
        tf.logical_not(component_active),
        tf.abs(mean_delta) <= tf.cast(tolerance, dtype),
    )
    return tf.cast(tf.reduce_all(component_ok), dtype)


def _signed_posterior_mean_rank_penalty(
    posterior: BetaPosterior,
    beta_true: tf.Tensor,
    weights: tf.Tensor,
    *,
    tolerance: float,
) -> tf.Tensor:
    """Return a signed mean-shift penalty that targets rank direction."""
    dtype = posterior.mean.dtype
    rank_probability = _tf_normal_cdf(
        (beta_true - posterior.mean) / tf.maximum(posterior.scale, 1e-6)
    )
    normalized_mean_error = (posterior.mean - beta_true) / tf.maximum(
        posterior.scale, 1e-6
    )

    def directional_loss(local_weights: tf.Tensor) -> tf.Tensor:
        denominator = tf.reduce_sum(local_weights)
        active = tf.cast(denominator >= 1.0, dtype)
        rank_mean = tf.reduce_sum(rank_probability * local_weights) / tf.maximum(
            denominator, tf.cast(1.0, dtype)
        )
        rank_delta = rank_mean - tf.cast(0.5, dtype)
        activation = tf.nn.relu(tf.abs(rank_delta) - float(tolerance)) / float(
            tolerance
        )
        clipped_delta = tf.clip_by_value(
            rank_delta,
            tf.cast(-4.0 * tolerance, dtype),
            tf.cast(4.0 * tolerance, dtype),
        )
        target_mean = tf.stop_gradient(
            posterior.mean + 2.0 * clipped_delta * posterior.scale
        )
        shift_loss = tf.reduce_sum(
            tf.square(
                (posterior.mean - target_mean) / tf.maximum(posterior.scale, 1e-6)
            )
            * local_weights
        ) / tf.maximum(denominator, tf.cast(1.0, dtype))
        mean_bias = tf.reduce_sum(normalized_mean_error * local_weights) / tf.maximum(
            denominator, tf.cast(1.0, dtype)
        )
        direction = tf.stop_gradient(tf.sign(rank_delta))
        target_bias = tf.stop_gradient(direction * tf.cast(tolerance, dtype))
        signed_bias_loss = tf.square(
            (mean_bias - target_bias) / tf.cast(tolerance, dtype)
        )
        return active * activation * (0.25 * shift_loss + signed_bias_loss)

    losses = [directional_loss(weights)]
    for coefficient_index in range(int(posterior.mean.shape[1])):
        coefficient_mask = tf.one_hot(
            coefficient_index,
            int(posterior.mean.shape[1]),
            dtype=dtype,
        )[None, :, None]
        losses.append(directional_loss(weights * coefficient_mask))
    return tf.reduce_mean(tf.stack(losses))


def _design_rank_mean_guard_penalty(
    posterior: BetaPosterior,
    beta_true: tf.Tensor,
    design: tf.Tensor,
    *,
    distribution: str,
    tolerance: float,
) -> tf.Tensor:
    """Constrain posterior-mean rank drift in medium/high design strata."""
    dtype = posterior.mean.dtype
    design_information = _expected_design_information(
        posterior.mean,
        design,
        distribution=distribution,
    )
    flattened = tf.reshape(design_information, [-1])
    sorted_values = tf.sort(flattened)
    n_values = tf.shape(sorted_values)[0]
    low = sorted_values[tf.maximum(0, n_values // 3)]
    high = sorted_values[tf.maximum(0, (2 * n_values) // 3)]
    masks = (
        tf.logical_and(design_information > low, design_information <= high),
        design_information > high,
    )
    rank_probability = _tf_normal_cdf(
        (beta_true - posterior.mean) / tf.maximum(posterior.scale, 1e-6)
    )
    normalized_mean_error = (posterior.mean - beta_true) / tf.maximum(
        posterior.scale, 1e-6
    )
    losses = []
    for mask in masks:
        weights = tf.cast(mask, dtype)
        denominator = tf.reduce_sum(weights)
        active = tf.cast(denominator >= 1.0, dtype)
        rank_mean = tf.reduce_sum(rank_probability * weights) / tf.maximum(
            denominator, tf.cast(1.0, dtype)
        )
        mean_bias = tf.reduce_sum(normalized_mean_error * weights) / tf.maximum(
            denominator, tf.cast(1.0, dtype)
        )
        rank_error = tf.nn.relu(tf.abs(rank_mean - 0.5) - float(tolerance)) / float(
            tolerance
        )
        losses.append(active * (tf.square(rank_error) + 0.10 * tf.square(mean_bias)))
    return tf.reduce_mean(tf.stack(losses))


def _design_coverage_guard_penalty(
    posterior: BetaPosterior,
    beta_true: tf.Tensor,
    design: tf.Tensor,
    *,
    distribution: str,
    coverage_floor: float,
) -> tf.Tensor:
    """Return smooth coverage-floor penalty by expected design-information tertile."""
    design_information = _expected_design_information(
        posterior.mean,
        design,
        distribution=distribution,
    )
    flattened = tf.reshape(design_information, [-1])
    sorted_values = tf.sort(flattened)
    n_values = tf.shape(sorted_values)[0]
    low = sorted_values[tf.maximum(0, n_values // 3)]
    high = sorted_values[tf.maximum(0, (2 * n_values) // 3)]
    low_mask = design_information <= low
    mid_mask = tf.logical_and(design_information > low, design_information <= high)
    high_mask = design_information > high
    standardized_abs_error = tf.abs(beta_true - posterior.mean) / tf.maximum(
        posterior.scale, 1e-6
    )
    z_value = tf.constant(1.959963984540054, dtype=posterior.mean.dtype)
    smooth_coverage = tf.sigmoid((z_value - standardized_abs_error) / 0.10)

    losses = []
    for mask in (low_mask, mid_mask, high_mask):
        weights = tf.cast(mask, posterior.mean.dtype)
        denominator = tf.reduce_sum(weights)
        coverage = tf.reduce_sum(smooth_coverage * weights) / tf.maximum(
            denominator, tf.cast(1.0, posterior.mean.dtype)
        )
        active = tf.cast(denominator >= 1.0, posterior.mean.dtype)
        losses.append(
            active
            * tf.square(
                tf.nn.relu(
                    (tf.cast(coverage_floor, posterior.mean.dtype) - coverage) / 0.05
                )
            )
        )
    return tf.reduce_mean(tf.stack(losses))


def _expected_design_information(
    mean: tf.Tensor,
    design: tf.Tensor,
    *,
    distribution: str,
) -> tf.Tensor:
    """Return coefficient-level expected design information."""
    linear = tf.einsum("bnk,bks->bns", design, mean)
    key = str(distribution).lower()
    if key in {"normal", "gaussian"}:
        weight = tf.ones_like(linear)
    elif key in {"probit", "bernoulli", "binomial"}:
        probability = tf.clip_by_value(_tf_normal_cdf(linear), 1e-6, 1.0 - 1e-6)
        density = tf.exp(-0.5 * tf.square(linear)) / tf.sqrt(
            tf.constant(2.0 * np.pi, dtype=mean.dtype)
        )
        weight = tf.square(density) / (probability * (1.0 - probability))
    elif key == "poisson":
        weight = tf.exp(tf.clip_by_value(linear, -20.0, 20.0))
    else:
        weight = tf.ones_like(linear)
    return tf.einsum("bnk,bns->bks", tf.square(design), weight)


def _tf_normal_cdf(value: tf.Tensor) -> tf.Tensor:
    value = tf.cast(value, tf.float32)
    return 0.5 * (1.0 + tf.math.erf(value / tf.sqrt(tf.constant(2.0, tf.float32))))
