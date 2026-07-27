"""Public variable-shape Neural-HMSC fixed-effect inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np
import tensorflow as tf

from pyhmsc.neural.inference import (
    NeuralHmscCompatibilityError,
    _InferenceContext,
    _as_design_array,
    _as_response_array,
    _check_compiled_artifact_supported,
)
from pyhmsc.neural.models import VariableShapeBetaPosteriorModel
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.simulator import FixedEffectDataset
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import (
    FixedShapeTrainingHistory,
    VariableShapeTrainingData,
    variable_shape_training_data,
)
from pyhmsc.posterior import HmscFit
from pyhmsc.serialization import read_compiled_model


VARIABLE_CHECKPOINT_VERSION = "0.1"
VARIABLE_TRAINING_CORPUS_VERSION = "0.2"
VARIABLE_MODEL_FAMILY = "variable_shape_fixed_effect_beta"
VARIABLE_CHECKPOINT_MANIFEST = "neural_checkpoint.json"
VARIABLE_CHECKPOINT_WEIGHTS = "weights.weights.h5"
VARIABLE_CALIBRATION_ARTIFACT = "variable_coefficient_calibration.json"
VARIABLE_CALIBRATION_KIND = "pyhmsc_variable_shape_beta_calibration"
VARIABLE_CALIBRATION_SCHEMA_VERSION = 1
VARIABLE_SHAPE_BASELINE_ID = "neural_hmsc_variable_probit_v1"
VARIABLE_SHAPE_BASELINE_KIND = "pyhmsc_variable_shape_deployment_baseline"
VARIABLE_SHAPE_BASELINE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VariableShapeBetaCalibration:
    """Species-agnostic scale calibration fitted on independent simulations."""

    scale_multiplier: float
    target_coverage: float = 0.95
    method: str = "independent_simulation_scalar_scale"
    n_coefficients: int = 0
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale_multiplier) or self.scale_multiplier <= 0.0:
            raise ValueError("scale_multiplier must be positive and finite")
        if not 0.0 < self.target_coverage < 1.0:
            raise ValueError("target_coverage must be between zero and one")
        if self.method != "independent_simulation_scalar_scale":
            raise ValueError("unsupported variable-shape calibration method")
        if int(self.n_coefficients) <= 0:
            raise ValueError("n_coefficients must be positive")
        _validate_calibration_provenance(self.provenance)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "parameter": "Beta",
            "semantics": "coefficient_posterior_uncertainty",
            "scale_multiplier": float(self.scale_multiplier),
            "target_coverage": float(self.target_coverage),
            "n_coefficients": int(self.n_coefficients),
            "provenance": dict(self.provenance or {}),
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "VariableShapeBetaCalibration":
        if metadata.get("parameter") != "Beta":
            raise ValueError("variable-shape calibration parameter must be Beta")
        if metadata.get("semantics") != "coefficient_posterior_uncertainty":
            raise ValueError("variable-shape calibration semantics differ")
        return cls(
            scale_multiplier=float(metadata["scale_multiplier"]),
            target_coverage=float(metadata["target_coverage"]),
            method=str(metadata["method"]),
            n_coefficients=int(metadata["n_coefficients"]),
            provenance=dict(metadata["provenance"]),
        )


@dataclass
class VariableShapeNeuralHmscInference:
    """Inference facade for one probit checkpoint over declared shape ranges."""

    model: VariableShapeBetaPosteriorModel
    min_sites: int
    max_sites: int
    min_species: int
    max_species: int
    distribution: str = "probit"
    formula: str = "~ x1 + x2"
    covariate_names: tuple[str, ...] = ("Intercept", "x1", "x2")
    checkpoint_version: str = VARIABLE_CHECKPOINT_VERSION
    training_corpus_version: str = VARIABLE_TRAINING_CORPUS_VERSION
    calibration: VariableShapeBetaCalibration | None = None

    @classmethod
    def for_fixed_effects(
        cls,
        *,
        min_sites: int,
        max_sites: int,
        min_species: int,
        max_species: int,
        n_covariates: int = 3,
        distribution: str = "probit",
        formula: str = "~ x1 + x2",
        covariate_names: Sequence[str] | None = None,
        hidden_units: Sequence[int] = (48, 48),
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        probit_anchor_eta_clip: float = 6.0,
    ) -> "VariableShapeNeuralHmscInference":
        distribution = str(distribution).lower()
        if distribution != "probit":
            raise ValueError(
                "the public variable-shape checkpoint currently supports probit only"
            )
        _validate_shape_range(min_sites, max_sites, "sites")
        _validate_shape_range(min_species, max_species, "species")
        names = tuple(
            str(value)
            for value in (
                covariate_names
                or ["Intercept"] + [f"x{index}" for index in range(1, n_covariates)]
            )
        )
        if len(names) != int(n_covariates) or names[0] != "Intercept":
            raise ValueError(
                "covariate_names must match n_covariates and start with Intercept"
            )
        model = VariableShapeBetaPosteriorModel(
            n_covariates=int(n_covariates),
            hidden_units=tuple(int(value) for value in hidden_units),
            distribution=distribution,
            probit_anchor_iterations=probit_anchor_iterations,
            probit_anchor_prior_precision=probit_anchor_prior_precision,
            probit_anchor_eta_clip=probit_anchor_eta_clip,
        )
        _build_variable_model(model, min_sites=min_sites, min_species=min_species)
        return cls(
            model=model,
            min_sites=int(min_sites),
            max_sites=int(max_sites),
            min_species=int(min_species),
            max_species=int(max_species),
            distribution=distribution,
            formula=str(formula),
            covariate_names=names,
        )

    @property
    def shape_range(self) -> dict[str, list[int]]:
        return {
            "n_sites": [self.min_sites, self.max_sites],
            "n_species": [self.min_species, self.max_species],
            "n_covariates": [self.model.n_covariates, self.model.n_covariates],
        }

    @classmethod
    def load(cls, checkpoint: str | Path) -> "VariableShapeNeuralHmscInference":
        root = Path(checkpoint).expanduser().resolve()
        manifest_path = root / VARIABLE_CHECKPOINT_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint_version") != VARIABLE_CHECKPOINT_VERSION:
            raise NeuralHmscCompatibilityError(
                "unsupported variable-shape checkpoint version"
            )
        if manifest.get("model_family") != VARIABLE_MODEL_FAMILY:
            raise NeuralHmscCompatibilityError(
                "unsupported variable-shape checkpoint model family"
            )
        if manifest.get("distribution") != "probit":
            raise NeuralHmscCompatibilityError(
                "public variable-shape checkpoints must be probit"
            )
        shape_range = manifest.get("shape_range", {})
        names = manifest.get("names", {})
        model = VariableShapeBetaPosteriorModel(
            n_covariates=int(manifest["n_covariates"]),
            hidden_units=tuple(int(value) for value in manifest["hidden_units"]),
            distribution="probit",
            probit_anchor_iterations=int(manifest["probit_anchor_iterations"]),
            probit_anchor_prior_precision=float(
                manifest["probit_anchor_prior_precision"]
            ),
            probit_anchor_eta_clip=float(manifest["probit_anchor_eta_clip"]),
        )
        min_sites, max_sites = (int(value) for value in shape_range["n_sites"])
        min_species, max_species = (int(value) for value in shape_range["n_species"])
        _validate_shape_range(min_sites, max_sites, "sites")
        _validate_shape_range(min_species, max_species, "species")
        _build_variable_model(model, min_sites=min_sites, min_species=min_species)
        model.load_weights(root / VARIABLE_CHECKPOINT_WEIGHTS)
        calibration = _load_calibration(root, manifest)
        return cls(
            model=model,
            min_sites=min_sites,
            max_sites=max_sites,
            min_species=min_species,
            max_species=max_species,
            distribution="probit",
            formula=str(manifest["formula"]["X"]),
            covariate_names=tuple(str(value) for value in names["covariates"]),
            checkpoint_version=VARIABLE_CHECKPOINT_VERSION,
            training_corpus_version=str(manifest["training_corpus_version"]),
            calibration=calibration,
        )

    def save(self, checkpoint: str | Path) -> Path:
        root = Path(checkpoint).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"variable-shape checkpoint is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        _build_variable_model(
            self.model, min_sites=self.min_sites, min_species=self.min_species
        )
        self.model.save_weights(root / VARIABLE_CHECKPOINT_WEIGHTS)
        calibration_record = (
            None
            if self.calibration is None
            else _write_calibration(root, self.calibration)
        )
        manifest = self._manifest(calibration_record=calibration_record)
        (root / VARIABLE_CHECKPOINT_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return root

    def fit(
        self,
        datasets: Sequence[FixedEffectDataset],
        *,
        epochs: int = 40,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        mse_weight: float = 0.25,
        seed: int = 123,
    ) -> FixedShapeTrainingHistory:
        if not datasets:
            raise ValueError("datasets must not be empty")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if learning_rate <= 0.0 or mse_weight < 0.0:
            raise ValueError(
                "learning_rate must be positive and mse_weight non-negative"
            )
        for dataset in datasets:
            self._check_dataset(dataset)
        data = variable_shape_training_data(datasets)
        self._check_data(data)
        tf.keras.utils.set_random_seed(seed)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        rng = np.random.default_rng(seed)
        history = {"loss": [], "beta_rmse": [], "scale_mean": []}
        for _ in range(int(epochs)):
            order = rng.permutation(len(datasets))
            losses = []
            rmses = []
            scales = []
            for start in range(0, len(order), int(batch_size)):
                batch = order[start : start + int(batch_size)]
                inputs = {
                    "X": data.X[batch],
                    "Y": data.Y[batch],
                    "site_mask": data.site_mask[batch],
                    "species_mask": data.species_mask[batch],
                }
                truth = tf.convert_to_tensor(data.Beta[batch], dtype=tf.float32)
                mask = tf.cast(data.species_mask[batch, None, :], tf.float32)
                with tf.GradientTape() as tape:
                    posterior = self.model(inputs, training=True)
                    variance = tf.square(posterior.scale)
                    point_nll = 0.5 * (
                        tf.math.log(2.0 * np.pi)
                        + tf.math.log(tf.maximum(variance, 1e-12))
                        + tf.square(truth - posterior.mean)
                        / tf.maximum(variance, 1e-12)
                    )
                    denominator = tf.maximum(
                        tf.reduce_sum(mask)
                        * tf.cast(self.model.n_covariates, tf.float32),
                        1.0,
                    )
                    nll = tf.reduce_sum(point_nll * mask) / denominator
                    mse = (
                        tf.reduce_sum(tf.square(truth - posterior.mean) * mask)
                        / denominator
                    )
                    loss = nll + float(mse_weight) * mse
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    (gradient, variable)
                    for gradient, variable in zip(
                        gradients, self.model.trainable_variables
                    )
                    if gradient is not None
                )
                losses.append(float(loss.numpy()))
                rmses.append(float(tf.sqrt(mse).numpy()))
                valid_scale = tf.boolean_mask(
                    posterior.scale,
                    tf.broadcast_to(mask > 0, tf.shape(posterior.scale)),
                )
                scales.append(float(tf.reduce_mean(valid_scale).numpy()))
            history["loss"].append(float(np.mean(losses)))
            history["beta_rmse"].append(float(np.mean(rmses)))
            history["scale_mean"].append(float(np.mean(scales)))
        return FixedShapeTrainingHistory(
            loss=history["loss"],
            beta_rmse=history["beta_rmse"],
            scale_mean=history["scale_mean"],
        )

    def fit_calibration(
        self,
        datasets: Sequence[FixedEffectDataset],
        *,
        target_coverage: float = 0.95,
        provenance: dict[str, Any],
    ) -> VariableShapeBetaCalibration:
        if not datasets:
            raise ValueError("calibration datasets must not be empty")
        if not 0.0 < target_coverage < 1.0:
            raise ValueError("target_coverage must be between zero and one")
        standardized = []
        count = 0
        for dataset in datasets:
            self._check_dataset(dataset)
            posterior = self.predict_beta_posterior(dataset, calibrated=False)
            truth = dataset.truth_beta.to_numpy(dtype=np.float32)
            standardized.append(
                np.abs(posterior.mean.numpy()[0] - truth)
                / np.maximum(posterior.scale.numpy()[0], np.finfo(float).eps)
            )
            count += truth.size
        target_z = _normal_quantile(0.5 + target_coverage / 2.0)
        multiplier = float(
            np.quantile(
                np.concatenate([value.ravel() for value in standardized]),
                target_coverage,
            )
            / target_z
        )
        calibration = VariableShapeBetaCalibration(
            scale_multiplier=max(multiplier, 1e-3),
            target_coverage=float(target_coverage),
            n_coefficients=count,
            provenance=dict(provenance),
        )
        self.calibration = calibration
        return calibration

    def check_compatibility(self, value: Any) -> dict[str, Any]:
        data, context = self._prepare(value)
        self._check_data(data, expected_batch=1)
        self._check_context(context)
        return {
            "compatible": True,
            "model_family": VARIABLE_MODEL_FAMILY,
            "distribution": context.distribution,
            "formula": context.formula,
            "shape_range": self.shape_range,
            "dimensions": {
                "n_sites": int(data.site_mask[0].sum()),
                "n_covariates": int(data.X.shape[2]),
                "n_species": int(data.species_mask[0].sum()),
            },
        }

    def predict_beta_posterior(
        self, value: Any, *, calibrated: bool = True
    ) -> BetaPosterior:
        data, context = self._prepare(value)
        self._check_data(data)
        self._check_context(context)
        posterior = self.model(
            {
                "X": data.X,
                "Y": data.Y,
                "site_mask": data.site_mask,
                "species_mask": data.species_mask,
            },
            training=False,
        )
        if not calibrated or self.calibration is None:
            return posterior
        multiplier = tf.cast(self.calibration.scale_multiplier, posterior.scale.dtype)
        return BetaPosterior(
            mean=posterior.mean,
            scale=posterior.scale * multiplier,
        )

    def infer(
        self,
        value: Any,
        *,
        draws: int = 1000,
        chains: int = 1,
        seed: int | None = None,
        output: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HmscFit:
        if draws <= 0 or chains <= 0:
            raise ValueError("draws and chains must be positive")
        data, context = self._prepare(value)
        self._check_data(data, expected_batch=1)
        self._check_context(context)
        posterior = self.predict_beta_posterior(data)
        posterior_metadata = {"neural_api": self._manifest()}
        if metadata:
            posterior_metadata.update(metadata)
        kwargs = {
            "covariate_names": context.covariate_names,
            "species_names": context.species_names,
            "distribution": context.distribution,
            "formula": context.formula,
            "chains": chains,
            "draws": draws,
            "seed": seed,
            "metadata": posterior_metadata,
            "calibration": self.calibration,
        }
        if output is None:
            with tempfile.TemporaryDirectory(prefix="variable-neural-hmsc-") as tmp:
                path = write_beta_posterior_hdf5(
                    posterior, Path(tmp) / "posterior.h5", **kwargs
                )
                return HmscFit.from_file(path)
        path = write_beta_posterior_hdf5(posterior, output, **kwargs)
        fit = HmscFit.from_file(path)
        fit.output_file = path
        return fit

    def _prepare(
        self, value: Any
    ) -> tuple[VariableShapeTrainingData, _InferenceContext]:
        if isinstance(value, VariableShapeTrainingData):
            species = tuple(f"sp{index + 1}" for index in range(value.Y.shape[2]))
            return value, _InferenceContext(
                distribution=self.distribution,
                formula=self.formula,
                covariate_names=self.covariate_names,
                species_names=species,
            )
        if isinstance(value, FixedEffectDataset):
            self._check_dataset(value)
            return variable_shape_training_data([value]), _InferenceContext(
                distribution=str(value.metadata.get("distribution", "")),
                formula=str(value.metadata.get("formula", self.formula)),
                covariate_names=tuple(str(name) for name in value.truth_beta.index),
                species_names=tuple(str(name) for name in value.Y.columns),
            )
        if isinstance(value, dict):
            if "X" not in value or "Y" not in value:
                raise NeuralHmscCompatibilityError(
                    "mapping input must contain 'X' and 'Y'"
                )
            data = _variable_data_from_arrays(
                _as_design_array(value["X"]), _as_response_array(value["Y"])
            )
            return data, _InferenceContext(
                distribution=str(value.get("distribution", self.distribution)),
                formula=str(value.get("formula", self.formula)),
                covariate_names=tuple(
                    str(name)
                    for name in value.get("covariate_names", self.covariate_names)
                ),
                species_names=tuple(
                    str(name)
                    for name in value.get(
                        "species_names",
                        [f"sp{index + 1}" for index in range(data.Y.shape[2])],
                    )
                ),
            )
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_dir():
                path = path / "init.json"
            compiled, arrays = read_compiled_model(path)
            _check_compiled_artifact_supported(compiled)
            if "X" not in arrays or "Y" not in arrays:
                raise NeuralHmscCompatibilityError(
                    "compiled artifact must contain X and Y arrays"
                )
            data = _variable_data_from_arrays(arrays["X"], arrays["Y"])
            names = compiled.get("names", {})
            formula = compiled.get("formula", {})
            return data, _InferenceContext(
                distribution=str(compiled.get("distribution", "")),
                formula=str(
                    formula.get("X", self.formula)
                    if isinstance(formula, dict)
                    else formula
                ),
                covariate_names=tuple(
                    str(name) for name in names.get("covariates", self.covariate_names)
                ),
                species_names=tuple(
                    str(name)
                    for name in names.get(
                        "species",
                        [f"sp{index + 1}" for index in range(data.Y.shape[2])],
                    )
                ),
            )
        raise NeuralHmscCompatibilityError(
            "variable-shape inference supports FixedEffectDataset, mapping, "
            "VariableShapeTrainingData, or compiled init.json/directory"
        )

    def _check_dataset(self, dataset: FixedEffectDataset) -> None:
        distribution = str(dataset.metadata.get("distribution", ""))
        if distribution != self.distribution:
            raise NeuralHmscCompatibilityError(
                f"dataset distribution {distribution!r} does not match "
                f"checkpoint distribution {self.distribution!r}"
            )
        if (
            tuple(str(name) for name in dataset.truth_beta.index)
            != self.covariate_names
        ):
            raise NeuralHmscCompatibilityError(
                "dataset ordered covariates do not match checkpoint"
            )
        _check_count_range(len(dataset.X), self.min_sites, self.max_sites, "site")
        _check_count_range(
            dataset.Y.shape[1], self.min_species, self.max_species, "species"
        )

    def _check_data(
        self, data: VariableShapeTrainingData, *, expected_batch: int | None = None
    ) -> None:
        if data.X.ndim != 3 or data.Y.ndim != 3:
            raise NeuralHmscCompatibilityError("variable X and Y must be rank-3")
        if data.site_mask.shape != data.X.shape[:2]:
            raise NeuralHmscCompatibilityError("site_mask shape differs from X")
        if data.species_mask.shape != (data.Y.shape[0], data.Y.shape[2]):
            raise NeuralHmscCompatibilityError("species_mask shape differs from Y")
        if data.X.shape[:2] != data.Y.shape[:2]:
            raise NeuralHmscCompatibilityError("X and Y batch/site shapes differ")
        if data.X.shape[2] != self.model.n_covariates:
            raise NeuralHmscCompatibilityError(
                "X covariate count does not match checkpoint"
            )
        if expected_batch is not None and data.X.shape[0] != expected_batch:
            raise NeuralHmscCompatibilityError(
                f"expected batch size {expected_batch}, got {data.X.shape[0]}"
            )
        for count in data.site_mask.sum(axis=1):
            _check_count_range(int(count), self.min_sites, self.max_sites, "site")
        for count in data.species_mask.sum(axis=1):
            _check_count_range(
                int(count), self.min_species, self.max_species, "species"
            )

    def _check_context(self, context: _InferenceContext) -> None:
        if context.distribution != self.distribution:
            raise NeuralHmscCompatibilityError(
                f"artifact distribution {context.distribution!r} does not match "
                f"checkpoint distribution {self.distribution!r}"
            )
        if context.covariate_names != self.covariate_names:
            raise NeuralHmscCompatibilityError(
                "artifact ordered covariates do not match checkpoint"
            )
        if context.formula != self.formula:
            raise NeuralHmscCompatibilityError(
                f"artifact formula {context.formula!r} does not match "
                f"checkpoint formula {self.formula!r}"
            )

    def _manifest(
        self, *, calibration_record: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        record = calibration_record
        if record is None and self.calibration is not None:
            record = {
                "path": VARIABLE_CALIBRATION_ARTIFACT,
                "method": self.calibration.method,
            }
        payload = {
            "checkpoint_version": self.checkpoint_version,
            "training_corpus_version": self.training_corpus_version,
            "model_family": VARIABLE_MODEL_FAMILY,
            "posterior_family": "diagonal_normal",
            "distribution": self.distribution,
            "formula": {"X": self.formula},
            "n_covariates": self.model.n_covariates,
            "shape_range": self.shape_range,
            "names": {"covariates": list(self.covariate_names)},
            "hidden_units": list(self.model.hidden_units),
            "probit_anchor_iterations": self.model.probit_anchor_iterations,
            "probit_anchor_prior_precision": (self.model.probit_anchor_prior_precision),
            "probit_anchor_eta_clip": self.model.probit_anchor_eta_clip,
            "limitations": [
                "fixed-effect probit Beta posterior inference only",
                "site and species counts must remain inside the declared ranges",
                "covariate count, order, and formula are fixed per checkpoint",
                "no traits, phylogeny, random effects, spatial effects, or detection submodel",
                "uncertainty is an amortized approximation, not an MCMC posterior",
            ],
        }
        if record is not None:
            payload["coefficient_calibration"] = record
        return payload


def freeze_variable_shape_baseline(
    *,
    registry_root: str | Path,
    candidate_checkpoint: str | Path,
    qualification_root: str | Path,
    baseline_id: str = VARIABLE_SHAPE_BASELINE_ID,
) -> Path:
    """Atomically freeze the predeclared candidate and multi-seed evidence."""
    _validate_baseline_id(baseline_id)
    registry = Path(registry_root).expanduser().resolve()
    destination = registry / baseline_id
    if destination.exists():
        raise FileExistsError(f"variable-shape baseline already exists: {destination}")
    checkpoint = Path(candidate_checkpoint).expanduser().resolve()
    qualification = Path(qualification_root).expanduser().resolve()
    aggregate_path = qualification / "variable_shape_multiseed_qualification.json"
    aggregate_markdown = qualification / "variable_shape_multiseed_qualification.md"
    aggregate = _validate_multiseed_qualification(aggregate_path)
    if not aggregate_markdown.is_file():
        raise FileNotFoundError(
            f"variable-shape qualification Markdown not found: {aggregate_markdown}"
        )
    expected = aggregate["candidate_checkpoint"]
    if (
        _sha256(checkpoint / VARIABLE_CHECKPOINT_MANIFEST)
        != expected["manifest_sha256"]
    ):
        raise ValueError("candidate checkpoint manifest hash differs from evidence")
    if _sha256(checkpoint / VARIABLE_CHECKPOINT_WEIGHTS) != expected["weights_sha256"]:
        raise ValueError("candidate checkpoint weights hash differs from evidence")
    candidate = VariableShapeNeuralHmscInference.load(checkpoint)
    if candidate.calibration is None:
        raise ValueError("candidate checkpoint lacks coefficient calibration")

    registry.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{baseline_id}.", dir=registry))
    try:
        shutil.copytree(checkpoint, staging / "checkpoint")
        evidence = staging / "evidence"
        evidence.mkdir()
        shutil.copy2(aggregate_path, evidence / aggregate_path.name)
        shutil.copy2(aggregate_markdown, evidence / aggregate_markdown.name)
        run_records = []
        for row in aggregate["runs"]:
            seed = int(row["base_seed"])
            source_json = Path(row["report_path"]).expanduser().resolve()
            if _sha256(source_json) != row["report_sha256"]:
                raise ValueError(f"qualification report hash differs for seed {seed}")
            source_markdown = source_json.with_suffix(".md")
            seed_dir = evidence / f"seed_{seed}"
            seed_dir.mkdir()
            shutil.copy2(source_json, seed_dir / source_json.name)
            shutil.copy2(source_markdown, seed_dir / source_markdown.name)
            run_records.append(
                {
                    "base_seed": seed,
                    "json": _file_record(seed_dir / source_json.name, staging),
                    "markdown": _file_record(seed_dir / source_markdown.name, staging),
                }
            )
        inventory = _baseline_inventory(staging)
        payload = {
            "schema_version": VARIABLE_SHAPE_BASELINE_SCHEMA_VERSION,
            "kind": VARIABLE_SHAPE_BASELINE_KIND,
            "baseline_id": baseline_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "qualified",
            "model_family": VARIABLE_MODEL_FAMILY,
            "distribution": "probit",
            "shape_range": candidate.shape_range,
            "checkpoint": {
                "path": "checkpoint",
                "manifest_sha256": expected["manifest_sha256"],
                "weights_sha256": expected["weights_sha256"],
            },
            "qualification": {
                "aggregate_json": _file_record(evidence / aggregate_path.name, staging),
                "aggregate_markdown": _file_record(
                    evidence / aggregate_markdown.name, staging
                ),
                "runs": run_records,
                "required_decision": "variable_shape_probit_promoted",
            },
            "fixed_release_id": "neural_hmsc_v0_1",
            "fixed_release_modified": False,
            "qualified_python_mcmc_role": "statistical_reference_only",
            "inventory": inventory,
            "content_sha256": _inventory_sha256(inventory),
        }
        baseline_path = staging / "baseline.json"
        baseline_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_variable_shape_baseline(
            baseline_path, expected_baseline_id=baseline_id
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "baseline.json"


def validate_variable_shape_baseline(
    baseline_path: str | Path,
    *,
    expected_baseline_id: str | None = None,
) -> dict[str, Any]:
    """Validate the immutable variable-shape checkpoint and evidence bundle."""
    path = Path(baseline_path).expanduser().resolve()
    if path.is_dir():
        path = path / "baseline.json"
    root = path.parent
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != VARIABLE_SHAPE_BASELINE_KIND:
        raise ValueError("unsupported variable-shape baseline kind")
    if int(payload.get("schema_version", -1)) != (
        VARIABLE_SHAPE_BASELINE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported variable-shape baseline schema")
    baseline_id = str(payload.get("baseline_id", ""))
    _validate_baseline_id(baseline_id)
    if expected_baseline_id is not None and baseline_id != expected_baseline_id:
        raise ValueError("variable-shape baseline identifier differs")
    if payload.get("status") != "qualified":
        raise ValueError("variable-shape baseline is not qualified")
    if payload.get("model_family") != VARIABLE_MODEL_FAMILY:
        raise ValueError("variable-shape baseline model family differs")
    if payload.get("distribution") != "probit":
        raise ValueError("variable-shape baseline distribution differs")
    if payload.get("fixed_release_id") != "neural_hmsc_v0_1":
        raise ValueError("variable-shape baseline fixed release reference differs")
    if payload.get("fixed_release_modified") is not False:
        raise ValueError("variable-shape baseline modified the fixed release")
    if payload.get("qualified_python_mcmc_role") != "statistical_reference_only":
        raise ValueError("variable-shape baseline MCMC role differs")

    inventory = payload.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("variable-shape baseline inventory is missing")
    if payload.get("content_sha256") != _inventory_sha256(inventory):
        raise ValueError("variable-shape baseline inventory digest differs")
    expected_files = {str(row["path"]) for row in inventory}
    if len(expected_files) != len(inventory):
        raise ValueError("variable-shape baseline inventory paths are duplicated")
    actual_files = {
        str(file.relative_to(root))
        for file in root.rglob("*")
        if file.is_file() and file != path
    }
    if actual_files != expected_files:
        raise ValueError("variable-shape baseline inventory file set differs")
    for row in inventory:
        file = _contained_file(root, str(row["path"]))
        if _sha256(file) != row["sha256"]:
            raise ValueError(f"variable-shape baseline hash mismatch for {file}")
        if file.stat().st_size != int(row["bytes"]):
            raise ValueError(f"variable-shape baseline size mismatch for {file}")

    qualification = payload.get("qualification", {})
    aggregate_path = _validate_file_record(root, qualification.get("aggregate_json"))
    aggregate = _validate_multiseed_qualification(aggregate_path)
    _validate_file_record(root, qualification.get("aggregate_markdown"))
    if qualification.get("required_decision") != aggregate["decision"]:
        raise ValueError("variable-shape baseline qualification decision differs")
    if len(qualification.get("runs", ())) != len(aggregate["runs"]):
        raise ValueError("variable-shape baseline qualification run count differs")
    for record, source in zip(qualification["runs"], aggregate["runs"]):
        if int(record["base_seed"]) != int(source["base_seed"]):
            raise ValueError("variable-shape baseline qualification seed differs")
        run_json = _validate_file_record(root, record["json"])
        _validate_file_record(root, record["markdown"])
        if _sha256(run_json) != source["report_sha256"]:
            raise ValueError("variable-shape baseline run report hash differs")

    checkpoint = root / str(payload["checkpoint"]["path"])
    if (
        _sha256(checkpoint / VARIABLE_CHECKPOINT_MANIFEST)
        != payload["checkpoint"]["manifest_sha256"]
    ):
        raise ValueError("variable-shape baseline checkpoint manifest hash differs")
    if (
        _sha256(checkpoint / VARIABLE_CHECKPOINT_WEIGHTS)
        != payload["checkpoint"]["weights_sha256"]
    ):
        raise ValueError("variable-shape baseline checkpoint weights hash differs")
    engine = VariableShapeNeuralHmscInference.load(checkpoint)
    if engine.shape_range != payload["shape_range"]:
        raise ValueError("variable-shape baseline checkpoint range differs")
    if engine.calibration is None:
        raise ValueError("variable-shape baseline checkpoint lacks calibration")
    return payload


def load_variable_shape_baseline(
    registry_root: str | Path,
    *,
    baseline_id: str = VARIABLE_SHAPE_BASELINE_ID,
) -> VariableShapeNeuralHmscInference:
    """Resolve a qualified variable-shape checkpoint by immutable ID."""
    root = Path(registry_root).expanduser().resolve() / baseline_id
    payload = validate_variable_shape_baseline(root, expected_baseline_id=baseline_id)
    return VariableShapeNeuralHmscInference.load(
        root / str(payload["checkpoint"]["path"])
    )


def _variable_data_from_arrays(X: Any, Y: Any) -> VariableShapeTrainingData:
    design = np.asarray(X, dtype=np.float32)
    response = np.asarray(Y, dtype=np.float32)
    if design.ndim != 2 or response.ndim != 2:
        raise NeuralHmscCompatibilityError("X and Y must be two-dimensional")
    if design.shape[0] != response.shape[0]:
        raise NeuralHmscCompatibilityError("X and Y site counts differ")
    return VariableShapeTrainingData(
        X=design[None, ...],
        Y=response[None, ...],
        Beta=np.zeros((1, design.shape[1], response.shape[1]), dtype=np.float32),
        site_mask=np.ones((1, design.shape[0]), dtype=bool),
        species_mask=np.ones((1, response.shape[1]), dtype=bool),
    )


def _build_variable_model(
    model: VariableShapeBetaPosteriorModel, *, min_sites: int, min_species: int
) -> None:
    model(
        {
            "X": tf.zeros((1, min_sites, model.n_covariates), dtype=tf.float32),
            "Y": tf.zeros((1, min_sites, min_species), dtype=tf.float32),
            "site_mask": tf.ones((1, min_sites), dtype=tf.bool),
            "species_mask": tf.ones((1, min_species), dtype=tf.bool),
        },
        training=False,
    )


def _write_calibration(
    root: Path, calibration: VariableShapeBetaCalibration
) -> dict[str, Any]:
    payload = {
        "schema_version": VARIABLE_CALIBRATION_SCHEMA_VERSION,
        "kind": VARIABLE_CALIBRATION_KIND,
        "calibration": calibration.to_metadata(),
    }
    path = root / VARIABLE_CALIBRATION_ARTIFACT
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "path": VARIABLE_CALIBRATION_ARTIFACT,
        "sha256": _sha256(path),
        "kind": VARIABLE_CALIBRATION_KIND,
        "schema_version": VARIABLE_CALIBRATION_SCHEMA_VERSION,
        "method": calibration.method,
        "parameter": "Beta",
    }


def _load_calibration(
    root: Path, manifest: dict[str, Any]
) -> VariableShapeBetaCalibration | None:
    record = manifest.get("coefficient_calibration")
    if record is None:
        return None
    if not isinstance(record, dict):
        raise NeuralHmscCompatibilityError(
            "variable-shape calibration record must be an object"
        )
    path = (root / str(record.get("path", ""))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise NeuralHmscCompatibilityError(
            "variable-shape calibration escapes checkpoint"
        ) from exc
    if _sha256(path) != record.get("sha256"):
        raise NeuralHmscCompatibilityError(
            "variable-shape calibration artifact hash mismatch"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != VARIABLE_CALIBRATION_KIND:
        raise NeuralHmscCompatibilityError(
            "unsupported variable-shape calibration kind"
        )
    if int(payload.get("schema_version", -1)) != VARIABLE_CALIBRATION_SCHEMA_VERSION:
        raise NeuralHmscCompatibilityError(
            "unsupported variable-shape calibration schema"
        )
    return VariableShapeBetaCalibration.from_metadata(payload["calibration"])


def _validate_calibration_provenance(provenance: dict[str, Any] | None) -> None:
    expected = {
        "kind": "independent_variable_shape_simulation_calibration",
        "target_ecological_response_used": False,
        "shape_selection_role": "predeclared_range",
    }
    if not isinstance(provenance, dict):
        raise ValueError("variable-shape calibration provenance is required")
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"variable-shape calibration provenance {key} differs")
    if not isinstance(provenance.get("seeds"), list) or not provenance["seeds"]:
        raise ValueError("variable-shape calibration provenance seeds are required")
    if not str(provenance.get("corpus_id", "")):
        raise ValueError("variable-shape calibration provenance corpus_id is required")


def _validate_shape_range(low: int, high: int, label: str) -> None:
    if int(low) <= 0 or int(high) < int(low):
        raise ValueError(f"{label} range must be positive and ordered")


def _check_count_range(count: int, low: int, high: int, label: str) -> None:
    if not low <= count <= high:
        raise NeuralHmscCompatibilityError(
            f"{label} count {count} is outside checkpoint range [{low}, {high}]"
        )


def _normal_quantile(probability: float) -> float:
    return float(
        np.sqrt(2.0)
        * tf.math.erfinv(tf.constant(2.0 * probability - 1.0, dtype=tf.float64)).numpy()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_multiseed_qualification(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "neural_hmsc_variable_shape_multiseed_qualification":
        raise ValueError("unsupported variable-shape multi-seed qualification kind")
    if payload.get("decision") != "variable_shape_probit_promoted":
        raise ValueError("variable-shape multi-seed qualification did not promote")
    if payload.get("all_runs_passed") is not True:
        raise ValueError("variable-shape multi-seed qualification has failed runs")
    if payload.get("candidate_selected_using_sensitivity_outcomes") is not False:
        raise ValueError("variable-shape candidate used sensitivity outcomes")
    if payload.get("fixed_release_modified") is not False:
        raise ValueError("variable-shape qualification modified fixed v0.1")
    rows = payload.get("runs")
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("variable-shape qualification requires at least three runs")
    if any(
        row.get("decision") != "variable_shape_probit_qualified"
        or row.get("all_gates_passed") is not True
        for row in rows
    ):
        raise ValueError("variable-shape qualification run failed")
    return payload


def _baseline_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
        if path.name != "baseline.json"
    ]


def _inventory_sha256(inventory: list[dict[str, Any]]) -> str:
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "sha256": _sha256(path),
    }


def _validate_file_record(root: Path, record: Any) -> Path:
    if (
        not isinstance(record, dict)
        or not record.get("path")
        or not record.get("sha256")
    ):
        raise ValueError("variable-shape baseline file record is incomplete")
    path = _contained_file(root, str(record["path"]))
    if _sha256(path) != record["sha256"]:
        raise ValueError("variable-shape baseline file record hash differs")
    return path


def _contained_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("variable-shape baseline path escapes root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"variable-shape baseline file not found: {path}")
    return path


def _validate_baseline_id(value: str) -> None:
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value
    ):
        raise ValueError(
            "baseline_id must contain only lowercase letters, digits, or '_'"
        )
