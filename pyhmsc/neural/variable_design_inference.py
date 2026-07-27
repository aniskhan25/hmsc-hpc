"""Experimental variable-design fixed-effect probit inference skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
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
from pyhmsc.neural.models import VariableDesignBetaPosteriorModel
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.simulator import FixedEffectDataset
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import (
    VariableDesignTrainingData,
    variable_design_training_data,
)
from pyhmsc.posterior import HmscFit
from pyhmsc.serialization import read_compiled_model


VARIABLE_DESIGN_CHECKPOINT_VERSION = "0.1"
VARIABLE_DESIGN_TRAINING_CORPUS_VERSION = "untrained_skeleton"
VARIABLE_DESIGN_MODEL_FAMILY = "variable_design_fixed_effect_beta"
VARIABLE_DESIGN_CHECKPOINT_MANIFEST = "neural_checkpoint.json"
VARIABLE_DESIGN_CHECKPOINT_WEIGHTS = "weights.weights.h5"
VARIABLE_DESIGN_CALIBRATION_ARTIFACT = "variable_design_calibration.json"
VARIABLE_DESIGN_CALIBRATION_KIND = "pyhmsc_variable_design_beta_calibration"


@dataclass(frozen=True)
class VariableDesignBetaCalibration:
    """Independent split-conformal scale calibration for variable-design Beta."""

    scale_multiplier: float
    n_coefficients: int
    target_coverage: float = 0.95
    method: str = "split_conformal_scalar_beta_scale"
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale_multiplier) or self.scale_multiplier <= 0.0:
            raise ValueError("scale_multiplier must be positive and finite")
        if int(self.n_coefficients) <= 0:
            raise ValueError("n_coefficients must be positive")
        if not 0.0 < self.target_coverage < 1.0:
            raise ValueError("target_coverage must be between zero and one")
        if self.method != "split_conformal_scalar_beta_scale":
            raise ValueError("unsupported variable-design calibration method")
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
    def from_metadata(cls, metadata: dict[str, Any]) -> "VariableDesignBetaCalibration":
        if metadata.get("parameter") != "Beta":
            raise ValueError("variable-design calibration parameter must be Beta")
        if metadata.get("semantics") != "coefficient_posterior_uncertainty":
            raise ValueError("variable-design calibration semantics differ")
        return cls(
            scale_multiplier=float(metadata["scale_multiplier"]),
            n_coefficients=int(metadata["n_coefficients"]),
            target_coverage=float(metadata["target_coverage"]),
            method=str(metadata["method"]),
            provenance=dict(metadata["provenance"]),
        )


@dataclass
class VariableDesignNeuralHmscInference:
    """Experimental facade for bounded variable-design probit inference."""

    model: VariableDesignBetaPosteriorModel
    min_sites: int
    max_sites: int
    min_species: int
    max_species: int
    min_covariates: int
    max_covariates: int
    max_design_condition_number: float = 1e6
    checkpoint_version: str = VARIABLE_DESIGN_CHECKPOINT_VERSION
    training_corpus_version: str = VARIABLE_DESIGN_TRAINING_CORPUS_VERSION
    distribution: str = "probit"
    calibration: VariableDesignBetaCalibration | None = None
    checkpoint_path: Path | None = None

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
        mean_correction_limit: float = 0.5,
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        probit_anchor_eta_clip: float = 6.0,
        max_design_condition_number: float = 1e6,
    ) -> "VariableDesignNeuralHmscInference":
        _validate_range(min_sites, max_sites, "sites")
        _validate_range(min_species, max_species, "species")
        _validate_range(min_covariates, max_covariates, "covariates")
        if min_covariates < 2 or max_covariates > 8:
            raise ValueError(
                "variable-design covariate range must remain within [2, 8]"
            )
        if max_sites > 128 or max_species > 100:
            raise ValueError("variable-design dimensions exceed Milestone 54 scope")
        if not np.isfinite(max_design_condition_number) or (
            max_design_condition_number <= 1.0
        ):
            raise ValueError("max_design_condition_number must be finite and above one")
        model = VariableDesignBetaPosteriorModel(
            hidden_units=tuple(int(value) for value in hidden_units),
            mean_correction_limit=mean_correction_limit,
            probit_anchor_iterations=probit_anchor_iterations,
            probit_anchor_prior_precision=probit_anchor_prior_precision,
            probit_anchor_eta_clip=probit_anchor_eta_clip,
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
        )

    @property
    def shape_range(self) -> dict[str, list[int]]:
        return {
            "n_sites": [self.min_sites, self.max_sites],
            "n_species": [self.min_species, self.max_species],
            "n_covariates": [self.min_covariates, self.max_covariates],
        }

    def save(self, checkpoint: str | Path) -> Path:
        root = Path(checkpoint).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"variable-design checkpoint is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        _build_model(
            self.model,
            n_sites=self.min_sites,
            n_species=self.min_species,
            n_covariates=self.min_covariates,
        )
        weights = root / VARIABLE_DESIGN_CHECKPOINT_WEIGHTS
        self.model.save_weights(weights)
        calibration_record = (
            None
            if self.calibration is None
            else _write_calibration(root, self.calibration)
        )
        manifest = self._manifest(
            weights_sha256=_sha256(weights),
            calibration_record=calibration_record,
        )
        (root / VARIABLE_DESIGN_CHECKPOINT_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return root

    @classmethod
    def load(cls, checkpoint: str | Path) -> "VariableDesignNeuralHmscInference":
        root = Path(checkpoint).expanduser().resolve()
        manifest_path = root / VARIABLE_DESIGN_CHECKPOINT_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint_version") != VARIABLE_DESIGN_CHECKPOINT_VERSION:
            raise NeuralHmscCompatibilityError(
                "unsupported variable-design checkpoint version"
            )
        if manifest.get("model_family") != VARIABLE_DESIGN_MODEL_FAMILY:
            raise NeuralHmscCompatibilityError(
                "unsupported variable-design checkpoint model family"
            )
        if manifest.get("distribution") != "probit":
            raise NeuralHmscCompatibilityError(
                "variable-design checkpoints must use probit"
            )
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
            mean_correction_limit=float(model_config["mean_correction_limit"]),
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
                "variable-design checkpoint weight hash mismatch"
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
        epochs: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        mse_weight: float = 0.25,
        seed: int,
    ) -> dict[str, list[float]]:
        if not datasets:
            raise ValueError("training datasets must not be empty")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if learning_rate <= 0.0 or mse_weight < 0.0:
            raise ValueError(
                "learning_rate must be positive and mse_weight non-negative"
            )
        data = variable_design_training_data(datasets)
        self._check_data(data)
        tf.keras.utils.set_random_seed(seed)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        rng = np.random.default_rng(seed)
        history: dict[str, list[float]] = {
            "loss": [],
            "beta_rmse": [],
            "scale_mean": [],
        }
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
                    "covariate_mask": data.covariate_mask[batch],
                }
                truth = tf.convert_to_tensor(data.Beta[batch], dtype=tf.float32)
                active = tf.cast(
                    data.covariate_mask[batch, :, None]
                    * data.species_mask[batch, None, :],
                    tf.float32,
                )
                with tf.GradientTape() as tape:
                    posterior = self.model(inputs, training=True)
                    variance = tf.maximum(tf.square(posterior.scale), 1e-12)
                    point_nll = 0.5 * (
                        tf.math.log(tf.constant(2.0 * np.pi, dtype=tf.float32))
                        + tf.math.log(variance)
                        + tf.square(truth - posterior.mean) / variance
                    )
                    denominator = tf.maximum(tf.reduce_sum(active), 1.0)
                    nll = tf.reduce_sum(point_nll * active) / denominator
                    mse = (
                        tf.reduce_sum(tf.square(truth - posterior.mean) * active)
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
                valid_scale = tf.boolean_mask(posterior.scale, active > 0)
                losses.append(float(loss.numpy()))
                rmses.append(float(tf.sqrt(mse).numpy()))
                scales.append(float(tf.reduce_mean(valid_scale).numpy()))
            history["loss"].append(float(np.mean(losses)))
            history["beta_rmse"].append(float(np.mean(rmses)))
            history["scale_mean"].append(float(np.mean(scales)))
        return history

    def fit_calibration(
        self,
        datasets: Sequence[FixedEffectDataset],
        *,
        target_coverage: float = 0.95,
        provenance: dict[str, Any],
    ) -> VariableDesignBetaCalibration:
        if not datasets:
            raise ValueError("calibration datasets must not be empty")
        standardized = []
        for dataset in datasets:
            posterior = self.predict_beta_posterior(dataset, calibrated=False)
            truth = dataset.truth_beta.to_numpy(dtype=float)
            standardized.append(
                np.abs(posterior.mean.numpy()[0] - truth)
                / np.maximum(posterior.scale.numpy()[0], np.finfo(float).eps)
            )
        scores = np.concatenate([value.ravel() for value in standardized])
        score_quantile = _finite_sample_quantile(scores, target_coverage)
        target_z = _normal_quantile(0.5 + target_coverage / 2.0)
        calibration = VariableDesignBetaCalibration(
            scale_multiplier=max(float(score_quantile / target_z), 1e-3),
            n_coefficients=int(scores.size),
            target_coverage=float(target_coverage),
            provenance=dict(provenance),
        )
        self.calibration = calibration
        return calibration

    def check_compatibility(self, value: Any) -> dict[str, Any]:
        data, context = self._prepare(value)
        dimensions = self._check_data(data, expected_batch=1)
        self._check_context(context, dimensions)
        return {
            "compatible": True,
            "model_family": VARIABLE_DESIGN_MODEL_FAMILY,
            "distribution": context.distribution,
            "formula": context.formula,
            "covariate_names": list(context.covariate_names),
            "shape_range": self.shape_range,
            "dimensions": dimensions,
        }

    def predict_beta_posterior(
        self, value: Any, *, calibrated: bool = True
    ) -> BetaPosterior:
        data, context = self._prepare(value)
        dimensions = self._check_data(data, expected_batch=1)
        self._check_context(context, dimensions)
        posterior = self.model(
            {
                "X": data.X,
                "Y": data.Y,
                "site_mask": data.site_mask,
                "species_mask": data.species_mask,
                "covariate_mask": data.covariate_mask,
            },
            training=False,
        )
        if not calibrated or self.calibration is None:
            return posterior
        return BetaPosterior(
            mean=posterior.mean,
            scale=posterior.scale * self.calibration.scale_multiplier,
        )

    def infer(
        self,
        value: Any,
        *,
        draws: int = 1000,
        chains: int = 1,
        seed: int | None = None,
        output: str | Path | None = None,
    ) -> HmscFit:
        if draws <= 0 or chains <= 0:
            raise ValueError("draws and chains must be positive")
        data, context = self._prepare(value)
        dimensions = self._check_data(data, expected_batch=1)
        self._check_context(context, dimensions)
        posterior = self.predict_beta_posterior(data)
        kwargs = {
            "covariate_names": context.covariate_names,
            "species_names": context.species_names,
            "distribution": context.distribution,
            "formula": context.formula,
            "chains": chains,
            "draws": draws,
            "seed": seed,
            "metadata": {"neural_api": self._manifest()},
            "calibration": self.calibration,
        }
        if output is None:
            with tempfile.TemporaryDirectory(prefix="variable-design-hmsc-") as tmp:
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
    ) -> tuple[VariableDesignTrainingData, _InferenceContext]:
        if isinstance(value, VariableDesignTrainingData):
            if value.X.shape[0] != 1 or len(value.covariate_names) != 1:
                raise NeuralHmscCompatibilityError(
                    "public variable-design inference requires one dataset"
                )
            names = value.covariate_names[0]
            species_count = int(value.species_mask[0].sum())
            return value, _InferenceContext(
                distribution="probit",
                formula=_formula_for_names(names),
                covariate_names=names,
                species_names=tuple(f"sp{index + 1}" for index in range(species_count)),
            )
        if isinstance(value, FixedEffectDataset):
            names = tuple(str(name) for name in value.truth_beta.index)
            formula = str(value.metadata.get("formula", _formula_for_names(names)))
            return variable_design_training_data([value]), _InferenceContext(
                distribution=str(value.metadata.get("distribution", "")),
                formula=formula,
                covariate_names=names,
                species_names=tuple(str(name) for name in value.Y.columns),
            )
        if isinstance(value, dict):
            if "X" not in value or "Y" not in value:
                raise NeuralHmscCompatibilityError(
                    "mapping input must contain 'X' and 'Y'"
                )
            design = _as_design_array(value["X"])
            response = _as_response_array(value["Y"])
            names = tuple(
                str(name)
                for name in value.get(
                    "covariate_names", _default_covariate_names(design.shape[1])
                )
            )
            species = tuple(
                str(name)
                for name in value.get(
                    "species_names",
                    [f"sp{index + 1}" for index in range(response.shape[1])],
                )
            )
            return _data_from_arrays(design, response, names), _InferenceContext(
                distribution=str(value.get("distribution", "probit")),
                formula=str(value.get("formula", _formula_for_names(names))),
                covariate_names=names,
                species_names=species,
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
            names_record = compiled.get("names", {})
            names = tuple(str(name) for name in names_record.get("covariates", []))
            species = tuple(str(name) for name in names_record.get("species", []))
            formula_record = compiled.get("formula", {})
            formula = (
                formula_record.get("X", "")
                if isinstance(formula_record, dict)
                else formula_record
            )
            return _data_from_arrays(
                arrays["X"], arrays["Y"], names
            ), _InferenceContext(
                distribution=str(compiled.get("distribution", "")),
                formula=str(formula),
                covariate_names=names,
                species_names=species,
            )
        raise NeuralHmscCompatibilityError(
            "variable-design inference supports FixedEffectDataset, mapping, "
            "VariableDesignTrainingData, or compiled init.json/directory"
        )

    def _check_data(
        self, data: VariableDesignTrainingData, *, expected_batch: int | None = None
    ) -> dict[str, int]:
        if data.X.ndim != 3 or data.Y.ndim != 3 or data.Beta.ndim != 3:
            raise NeuralHmscCompatibilityError(
                "variable-design X, Y, and Beta must be rank 3"
            )
        if data.X.shape[:2] != data.Y.shape[:2]:
            raise NeuralHmscCompatibilityError("X and Y batch/site shapes differ")
        if data.Beta.shape != (data.X.shape[0], data.X.shape[2], data.Y.shape[2]):
            raise NeuralHmscCompatibilityError("Beta shape differs from X/Y")
        expected_masks = {
            "site_mask": (data.X.shape[0], data.X.shape[1]),
            "species_mask": (data.Y.shape[0], data.Y.shape[2]),
            "covariate_mask": (data.X.shape[0], data.X.shape[2]),
        }
        for name, shape in expected_masks.items():
            if np.asarray(getattr(data, name)).shape != shape:
                raise NeuralHmscCompatibilityError(f"{name} shape differs")
        if expected_batch is not None and data.X.shape[0] != expected_batch:
            raise NeuralHmscCompatibilityError(
                f"expected batch size {expected_batch}, got {data.X.shape[0]}"
            )

        dimensions = None
        for index in range(data.X.shape[0]):
            site_mask = np.asarray(data.site_mask[index], dtype=bool)
            species_mask = np.asarray(data.species_mask[index], dtype=bool)
            covariate_mask = np.asarray(data.covariate_mask[index], dtype=bool)
            _require_prefix_mask(site_mask, "site_mask")
            _require_prefix_mask(species_mask, "species_mask")
            _require_prefix_mask(covariate_mask, "covariate_mask")
            n_sites = int(site_mask.sum())
            n_species = int(species_mask.sum())
            n_covariates = int(covariate_mask.sum())
            _check_range(n_sites, self.min_sites, self.max_sites, "site")
            _check_range(n_species, self.min_species, self.max_species, "species")
            _check_range(
                n_covariates,
                self.min_covariates,
                self.max_covariates,
                "covariate",
            )
            active_design = np.asarray(
                data.X[index, :n_sites, :n_covariates], dtype=float
            )
            active_response = np.asarray(
                data.Y[index, :n_sites, :n_species], dtype=float
            )
            if (
                not np.isfinite(active_design).all()
                or not np.isfinite(active_response).all()
            ):
                raise NeuralHmscCompatibilityError(
                    "active variable-design values must be finite"
                )
            if not np.allclose(active_design[:, 0], 1.0, rtol=0.0, atol=1e-6):
                raise NeuralHmscCompatibilityError(
                    "variable-design X must contain one leading intercept column"
                )
            if not np.all(np.isin(active_response, (0.0, 1.0))):
                raise NeuralHmscCompatibilityError(
                    "variable-design probit Y must be binary"
                )
            if np.linalg.matrix_rank(active_design) < n_covariates:
                raise NeuralHmscCompatibilityError(
                    "variable-design matrix is rank deficient"
                )
            condition = float(np.linalg.cond(active_design))
            if not np.isfinite(condition) or (
                condition > self.max_design_condition_number
            ):
                raise NeuralHmscCompatibilityError(
                    "variable-design matrix is outside condition-number support"
                )
            dimensions = {
                "n_sites": n_sites,
                "n_covariates": n_covariates,
                "n_species": n_species,
            }
        if dimensions is None:
            raise NeuralHmscCompatibilityError("variable-design batch is empty")
        return dimensions

    def _check_context(
        self, context: _InferenceContext, dimensions: dict[str, int]
    ) -> None:
        if context.distribution != "probit":
            raise NeuralHmscCompatibilityError(
                f"variable-design distribution must be 'probit', got "
                f"{context.distribution!r}"
            )
        if len(context.covariate_names) != dimensions["n_covariates"]:
            raise NeuralHmscCompatibilityError(
                "covariate-name count differs from active design columns"
            )
        if (
            not context.covariate_names
            or context.covariate_names[0] != "Intercept"
            or "Intercept" in context.covariate_names[1:]
            or len(set(context.covariate_names)) != len(context.covariate_names)
        ):
            raise NeuralHmscCompatibilityError(
                "covariate names must be unique with one leading Intercept"
            )
        if len(context.species_names) != dimensions["n_species"]:
            raise NeuralHmscCompatibilityError(
                "species-name count differs from active response columns"
            )
        if not context.formula.strip():
            raise NeuralHmscCompatibilityError(
                "variable-design formula provenance is required"
            )

    def _manifest(
        self,
        *,
        weights_sha256: str | None = None,
        calibration_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checkpoint_version": self.checkpoint_version,
            "training_corpus_version": self.training_corpus_version,
            "model_family": VARIABLE_DESIGN_MODEL_FAMILY,
            "posterior_family": "diagonal_normal",
            "distribution": "probit",
            "shape_range": self.shape_range,
            "model": {
                "hidden_units": list(self.model.hidden_units),
                "mean_correction_limit": self.model.mean_correction_limit,
                "probit_anchor_iterations": self.model.probit_anchor_iterations,
                "probit_anchor_prior_precision": (
                    self.model.probit_anchor_prior_precision
                ),
                "probit_anchor_eta_clip": self.model.probit_anchor_eta_clip,
            },
            "support": {
                "leading_intercept_required": True,
                "full_column_rank_required": True,
                "max_design_condition_number": self.max_design_condition_number,
            },
            "limitations": [
                "experimental Milestone 54 candidate; not release-qualified",
                "fixed-effect probit Beta posterior inference only",
                "coefficient calibration is required for qualification",
                "no traits, phylogeny, random effects, spatial effects, or detection submodel",
                "uncertainty is not release-qualified",
            ],
        }
        if weights_sha256 is not None:
            payload["artifacts"] = {
                "weights": {
                    "path": VARIABLE_DESIGN_CHECKPOINT_WEIGHTS,
                    "sha256": weights_sha256,
                }
            }
        if calibration_record is None and self.calibration is not None:
            calibration_record = {
                "path": VARIABLE_DESIGN_CALIBRATION_ARTIFACT,
                "method": self.calibration.method,
            }
        if calibration_record is not None:
            payload["coefficient_calibration"] = calibration_record
        return payload


def _data_from_arrays(
    X: Any, Y: Any, covariate_names: Sequence[str]
) -> VariableDesignTrainingData:
    design = np.asarray(X, dtype=np.float32)
    response = np.asarray(Y, dtype=np.float32)
    if design.ndim != 2 or response.ndim != 2:
        raise NeuralHmscCompatibilityError("X and Y must be two-dimensional")
    if design.shape[0] != response.shape[0]:
        raise NeuralHmscCompatibilityError("X and Y site counts differ")
    names = tuple(str(name) for name in covariate_names)
    return VariableDesignTrainingData(
        X=design[None, ...],
        Y=response[None, ...],
        Beta=np.zeros((1, design.shape[1], response.shape[1]), dtype=np.float32),
        site_mask=np.ones((1, design.shape[0]), dtype=bool),
        species_mask=np.ones((1, response.shape[1]), dtype=bool),
        covariate_mask=np.ones((1, design.shape[1]), dtype=bool),
        covariate_names=(names,),
    )


def _build_model(
    model: VariableDesignBetaPosteriorModel,
    *,
    n_sites: int,
    n_species: int,
    n_covariates: int,
) -> None:
    design = np.zeros((1, n_sites, n_covariates), dtype=np.float32)
    design[:, :, 0] = 1.0
    model(
        {
            "X": design,
            "Y": np.zeros((1, n_sites, n_species), dtype=np.float32),
            "site_mask": np.ones((1, n_sites), dtype=bool),
            "species_mask": np.ones((1, n_species), dtype=bool),
            "covariate_mask": np.ones((1, n_covariates), dtype=bool),
        },
        training=False,
    )


def _default_covariate_names(n_covariates: int) -> tuple[str, ...]:
    return ("Intercept",) + tuple(f"x{index}" for index in range(1, int(n_covariates)))


def _formula_for_names(names: Sequence[str]) -> str:
    predictors = [str(name) for name in names[1:]]
    return "~ " + " + ".join(predictors) if predictors else "~ 1"


def _require_prefix_mask(mask: np.ndarray, name: str) -> None:
    count = int(mask.sum())
    expected = np.arange(mask.size) < count
    if not np.array_equal(mask, expected):
        raise NeuralHmscCompatibilityError(f"{name} must mark one contiguous prefix")


def _validate_range(low: int, high: int, label: str) -> None:
    if int(low) <= 0 or int(high) < int(low):
        raise ValueError(f"{label} range must be positive and ordered")


def _check_range(count: int, low: int, high: int, label: str) -> None:
    if not low <= count <= high:
        raise NeuralHmscCompatibilityError(
            f"{label} count {count} is outside checkpoint range [{low}, {high}]"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_calibration(
    root: Path, calibration: VariableDesignBetaCalibration
) -> dict[str, Any]:
    path = root / VARIABLE_DESIGN_CALIBRATION_ARTIFACT
    payload = {
        "schema_version": 1,
        "kind": VARIABLE_DESIGN_CALIBRATION_KIND,
        "calibration": calibration.to_metadata(),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "path": VARIABLE_DESIGN_CALIBRATION_ARTIFACT,
        "method": calibration.method,
        "sha256": _sha256(path),
    }


def _load_calibration(
    root: Path, manifest: dict[str, Any]
) -> VariableDesignBetaCalibration | None:
    record = manifest.get("coefficient_calibration")
    if record is None:
        return None
    if record.get("path") != VARIABLE_DESIGN_CALIBRATION_ARTIFACT:
        raise NeuralHmscCompatibilityError(
            "variable-design calibration artifact path differs"
        )
    path = root / VARIABLE_DESIGN_CALIBRATION_ARTIFACT
    if _sha256(path) != record.get("sha256"):
        raise NeuralHmscCompatibilityError(
            "variable-design calibration artifact hash mismatch"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != VARIABLE_DESIGN_CALIBRATION_KIND:
        raise NeuralHmscCompatibilityError(
            "unsupported variable-design calibration artifact"
        )
    return VariableDesignBetaCalibration.from_metadata(payload["calibration"])


def _validate_calibration_provenance(provenance: dict[str, Any] | None) -> None:
    if provenance is None:
        raise ValueError("variable-design calibration provenance is required")
    expected = {
        "independent_from_training": True,
        "target_ecological_response_used": False,
    }
    for key, value in expected.items():
        if provenance.get(key) is not value:
            raise ValueError(f"variable-design calibration provenance {key} differs")
    if not isinstance(provenance.get("seeds"), list) or not provenance["seeds"]:
        raise ValueError("variable-design calibration provenance seeds are required")
    if not str(provenance.get("corpus_id", "")):
        raise ValueError("variable-design calibration provenance corpus_id is required")


def _finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    values = np.asarray(scores, dtype=float).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("finite conformal scores are required")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    order = min(int(np.ceil((values.size + 1) * coverage)), values.size)
    return float(np.partition(values, order - 1)[order - 1])


def _normal_quantile(probability: float) -> float:
    return float(
        np.sqrt(2.0)
        * tf.math.erfinv(tf.constant(2.0 * probability - 1.0, dtype=tf.float64)).numpy()
    )
