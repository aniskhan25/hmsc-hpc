"""Public Neural-HMSC inference facade.

The first supported API surface is intentionally narrow: fixed-effect Beta
posterior inference for compiled Python-native HMSC artifacts and matching
synthetic fixed-effect datasets. Lower-level neural modules remain available
for experiments, but this module defines the checkpoint and compatibility
boundary users should build against.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache"))

import tensorflow as tf

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


NEURAL_CHECKPOINT_VERSION = "0.1"
NEURAL_TRAINING_CORPUS_VERSION = "0.1"
SUPPORTED_MODEL_FAMILY = "fixed_effect_beta"
CHECKPOINT_MANIFEST = "neural_checkpoint.json"
CHECKPOINT_WEIGHTS = "weights.weights.h5"


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
    ) -> "NeuralHmscInference":
        """Create an untrained fixed-effect Beta inference engine."""
        covariate_names = tuple(covariate_names or _default_covariate_names(n_covariates))
        species_names = tuple(species_names or [f"sp{idx + 1}" for idx in range(n_species)])
        if len(covariate_names) != n_covariates:
            raise ValueError("covariate_names length must match n_covariates")
        if len(species_names) != n_species:
            raise ValueError("species_names length must match n_species")
        hidden_units = tuple(int(value) for value in hidden_units)
        model = FixedShapeBetaPosteriorModel(
            n_sites=n_sites,
            n_covariates=n_covariates,
            n_species=n_species,
            hidden_units=hidden_units,
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
            raise FileNotFoundError(f"Neural-HMSC checkpoint manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = str(manifest.get("checkpoint_version", ""))
        if version != NEURAL_CHECKPOINT_VERSION:
            raise NeuralHmscCompatibilityError(
                f"unsupported Neural-HMSC checkpoint version {version!r}; "
                f"expected {NEURAL_CHECKPOINT_VERSION!r}"
            )
        model_family = str(manifest.get("model_family", ""))
        if model_family != SUPPORTED_MODEL_FAMILY:
            raise NeuralHmscCompatibilityError(
                f"unsupported Neural-HMSC model_family {model_family!r}; "
                f"expected {SUPPORTED_MODEL_FAMILY!r}"
            )
        dimensions = manifest.get("dimensions", {})
        hidden_units = tuple(int(value) for value in manifest.get("hidden_units", (64, 64)))
        model = FixedShapeBetaPosteriorModel(
            n_sites=int(dimensions["n_sites"]),
            n_covariates=int(dimensions["n_covariates"]),
            n_species=int(dimensions["n_species"]),
            hidden_units=hidden_units,
        )
        _build_fixed_shape_model(model)
        model.load_weights(checkpoint / CHECKPOINT_WEIGHTS)
        names = manifest.get("names", {})
        formula = manifest.get("formula", {})
        return cls(
            model=model,
            model_family=model_family,
            distribution=str(manifest.get("distribution", "normal")),
            formula=str(formula.get("X", "~ x1 + x2") if isinstance(formula, dict) else formula),
            covariate_names=tuple(str(name) for name in names.get("covariates", _default_covariate_names(model.n_covariates))),
            species_names=tuple(str(name) for name in names.get("species", [f"sp{idx + 1}" for idx in range(model.n_species)])),
            hidden_units=hidden_units,
            checkpoint_version=version,
            training_corpus_version=str(manifest.get("training_corpus_version", NEURAL_TRAINING_CORPUS_VERSION)),
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
        manifest = self._manifest()
        (checkpoint / CHECKPOINT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return checkpoint

    def fit(
        self,
        datasets: Sequence[FixedEffectDataset],
        *,
        epochs: int = 40,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        mse_weight: float = 0.25,
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

        data = fixed_shape_training_data(datasets)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        rng = np.random.default_rng(seed)
        history = {"loss": [], "beta_rmse": [], "scale_mean": []}

        for epoch in range(int(epochs)):
            order = np.arange(data.Beta.shape[0])
            rng.shuffle(order)
            epoch_loss = []
            epoch_rmse = []
            epoch_scale = []
            for start in range(0, len(order), int(batch_size)):
                batch = order[start : start + int(batch_size)]
                x_batch = tf.convert_to_tensor(data.X[batch], dtype=tf.float32)
                y_batch = tf.convert_to_tensor(data.Y[batch], dtype=tf.float32)
                beta_batch = tf.convert_to_tensor(data.Beta[batch], dtype=tf.float32)
                with tf.GradientTape() as tape:
                    posterior = self.model({"X": x_batch, "Y": y_batch}, training=True)
                    nll = beta_negative_log_probability(posterior, beta_batch)
                    mse = tf.reduce_mean(tf.square(beta_batch - posterior.mean))
                    loss = nll + float(mse_weight) * mse
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    (gradient, variable)
                    for gradient, variable in zip(gradients, self.model.trainable_variables)
                    if gradient is not None
                )
                rmse = tf.sqrt(tf.reduce_mean(tf.square(beta_batch - posterior.mean)))
                epoch_loss.append(float(loss.numpy()))
                epoch_rmse.append(float(rmse.numpy()))
                epoch_scale.append(float(tf.reduce_mean(posterior.scale).numpy()))
            history["loss"].append(float(np.mean(epoch_loss)))
            history["beta_rmse"].append(float(np.mean(epoch_rmse)))
            history["scale_mean"].append(float(np.mean(epoch_scale)))
            if verbose:
                print(
                    f"epoch {epoch + 1}/{epochs} "
                    f"loss={history['loss'][-1]:.4f} "
                    f"beta_rmse={history['beta_rmse'][-1]:.4f} "
                    f"scale_mean={history['scale_mean'][-1]:.4f}"
                )

        return FixedShapeTrainingHistory(
            loss=history["loss"],
            beta_rmse=history["beta_rmse"],
            scale_mean=history["scale_mean"],
        )

    def check_compatibility(self, model_or_compiled_artifact: Any) -> dict[str, Any]:
        """Return a compatibility summary or raise a clear compatibility error."""
        data, context = self._prepare_inference_data(model_or_compiled_artifact)
        self._check_training_data_shape(data, batch_size=1)
        return {
            "compatible": True,
            "model_family": self.model_family,
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
        posterior = predict_beta_posterior(self.model, data)
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
            )
            fit = HmscFit.from_file(path)
            fit.output_file = path
        return fit

    def predict_beta_posterior(self, model_or_compiled_artifact: Any) -> BetaPosterior:
        """Return the raw diagonal-normal fixed-effect ``Beta`` posterior."""
        data, _ = self._prepare_inference_data(model_or_compiled_artifact)
        self._check_training_data_shape(data, batch_size=None)
        return predict_beta_posterior(self.model, data)

    def _prepare_inference_data(self, value: Any) -> tuple[FixedShapeTrainingData, "_InferenceContext"]:
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

    def _prepare_mapping(self, value: dict[str, Any]) -> tuple[FixedShapeTrainingData, "_InferenceContext"]:
        if "X" not in value or "Y" not in value:
            raise NeuralHmscCompatibilityError("mapping input must contain 'X' and 'Y'")
        X = _as_design_array(value["X"])
        Y = _as_response_array(value["Y"])
        data = _training_data_from_arrays(X, Y)
        formula = str(value.get("formula", self.formula))
        distribution = str(value.get("distribution", self.distribution))
        covariate_names = tuple(str(name) for name in value.get("covariate_names", self.covariate_names))
        species_names = tuple(str(name) for name in value.get("species_names", self.species_names))
        return data, _InferenceContext(
            distribution=distribution,
            formula=formula,
            covariate_names=covariate_names,
            species_names=species_names,
        )

    def _prepare_compiled_artifact(self, value: str | Path) -> tuple[FixedShapeTrainingData, "_InferenceContext"]:
        path = Path(value)
        if path.is_dir():
            path = path / "init.json"
        metadata, arrays = read_compiled_model(path)
        _check_compiled_artifact_supported(metadata)
        required = {"X", "Y"}
        missing = sorted(required.difference(arrays))
        if missing:
            raise NeuralHmscCompatibilityError(f"compiled artifact is missing arrays: {missing}")
        X = np.asarray(arrays["X"], dtype=np.float32)
        Y = np.asarray(arrays["Y"], dtype=np.float32)
        data = _training_data_from_arrays(X, Y)
        names = metadata.get("names", {})
        formula = metadata.get("formula", {})
        return data, _InferenceContext(
            distribution=str(metadata.get("distribution", self.distribution)),
            formula=str(formula.get("X", self.formula) if isinstance(formula, dict) else formula),
            covariate_names=tuple(str(name) for name in names.get("covariates", self.covariate_names)),
            species_names=tuple(str(name) for name in names.get("species", self.species_names)),
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

    def _check_training_data_shape(self, data: FixedShapeTrainingData, *, batch_size: int | None) -> None:
        if data.X.ndim != 3:
            raise NeuralHmscCompatibilityError(f"X must be a 3D batch; got shape {data.X.shape}")
        if data.Y.ndim != 3:
            raise NeuralHmscCompatibilityError(f"Y must be a 3D batch; got shape {data.Y.shape}")
        if batch_size is not None and data.X.shape[0] != batch_size:
            raise NeuralHmscCompatibilityError(f"X batch size {data.X.shape[0]} does not match expected {batch_size}")
        if batch_size is not None and data.Y.shape[0] != batch_size:
            raise NeuralHmscCompatibilityError(f"Y batch size {data.Y.shape[0]} does not match expected {batch_size}")
        expected_tail = (self.model.n_sites, self.model.n_covariates)
        if data.X.shape[1:] != expected_tail:
            raise NeuralHmscCompatibilityError(f"X shape {data.X.shape} does not match checkpoint tail {expected_tail}")
        expected_y_tail = (self.model.n_sites, self.model.n_species)
        if data.Y.shape[1:] != expected_y_tail:
            raise NeuralHmscCompatibilityError(f"Y shape {data.Y.shape} does not match checkpoint tail {expected_y_tail}")

    def _manifest(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "training_corpus_version": self.training_corpus_version,
            "model_family": self.model_family,
            "distribution": self.distribution,
            "formula": {"X": self.formula},
            "dimensions": self.dimensions,
            "names": {
                "covariates": list(self.covariate_names),
                "species": list(self.species_names),
            },
            "hidden_units": list(self.hidden_units),
            "limitations": _PUBLIC_LIMITATIONS,
        }


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
        raise NeuralHmscCompatibilityError(f"X must be a 2D design matrix; got shape {X.shape}")
    if Y.ndim != 2:
        raise NeuralHmscCompatibilityError(f"Y must be a 2D response matrix; got shape {Y.shape}")
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


def _build_fixed_shape_model(model: FixedShapeBetaPosteriorModel) -> None:
    X = np.zeros((1, model.n_sites, model.n_covariates), dtype=np.float32)
    Y = np.zeros((1, model.n_sites, model.n_species), dtype=np.float32)
    model({"X": X, "Y": Y}, training=False)
