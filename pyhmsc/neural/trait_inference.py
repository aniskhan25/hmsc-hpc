"""Public trait-mediated Gamma inference for a qualified Hmsc boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np
import tensorflow as tf

from pyhmsc.neural.inference import NeuralHmscCompatibilityError
from pyhmsc.neural.models import TraitGammaPosteriorModel
from pyhmsc.neural.posterior_heads import BetaPosterior, GammaPosterior
from pyhmsc.neural.simulator import TraitEffectDataset
from pyhmsc.neural.storage import write_trait_gamma_posterior_hdf5
from pyhmsc.neural.train import (
    FixedShapeTrainingHistory,
    TraitEffectTrainingData,
    train_trait_gamma_model,
    trait_effect_training_data,
)
from pyhmsc.posterior import HmscFit
from pyhmsc.serialization import read_compiled_model


TRAIT_GAMMA_CHECKPOINT_VERSION = "0.1"
TRAIT_GAMMA_TRAINING_CORPUS_VERSION = "0.1"
TRAIT_GAMMA_MODEL_FAMILY = "trait_mediated_gamma_beta"
TRAIT_GAMMA_BASELINE_ID = "neural_hmsc_trait_gamma_probit_v1"
TRAIT_GAMMA_CHECKPOINT_MANIFEST = "neural_checkpoint.json"
TRAIT_GAMMA_CHECKPOINT_WEIGHTS = "weights.weights.h5"
TRAIT_GAMMA_CALIBRATION_ARTIFACT = "gamma_calibration.json"
TRAIT_GAMMA_BASELINE_KIND = "pyhmsc_trait_gamma_deployment_baseline"


@dataclass(frozen=True)
class TraitGammaCalibration:
    """Scalar Gamma uncertainty calibration from independent simulations."""

    scale_multiplier: float
    target_coverage: float
    n_coefficients: int
    provenance: dict[str, Any]
    method: str = "independent_simulation_scalar_gamma_scale"

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale_multiplier) or self.scale_multiplier <= 0.0:
            raise ValueError("Gamma scale_multiplier must be positive and finite")
        if not 0.0 < self.target_coverage < 1.0:
            raise ValueError("Gamma target_coverage must be between zero and one")
        if self.n_coefficients <= 0:
            raise ValueError("Gamma calibration coefficient count must be positive")
        if self.method not in {
            "independent_simulation_scalar_gamma_scale",
            "split_conformal_scalar_gamma_scale",
        }:
            raise ValueError("unsupported Gamma calibration method")
        _validate_calibration_provenance(self.provenance)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "parameter": "Gamma",
            "semantics": "coefficient_posterior_uncertainty",
            "scale_multiplier": float(self.scale_multiplier),
            "target_coverage": float(self.target_coverage),
            "n_coefficients": int(self.n_coefficients),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "TraitGammaCalibration":
        if payload.get("parameter") != "Gamma":
            raise ValueError("trait calibration parameter must be Gamma")
        if payload.get("semantics") != "coefficient_posterior_uncertainty":
            raise ValueError("trait calibration uncertainty semantics differ")
        return cls(
            scale_multiplier=float(payload["scale_multiplier"]),
            target_coverage=float(payload["target_coverage"]),
            n_coefficients=int(payload["n_coefficients"]),
            provenance=dict(payload["provenance"]),
            method=str(payload["method"]),
        )


@dataclass(frozen=True)
class _TraitContext:
    distribution: str
    formula: str
    trait_formula: str
    covariate_names: tuple[str, ...]
    species_names: tuple[str, ...]
    trait_names: tuple[str, ...]


@dataclass
class TraitGammaNeuralHmscInference:
    """Inference facade for one fixed-shape trait-mediated Gamma checkpoint."""

    model: TraitGammaPosteriorModel
    distribution: str
    formula: str
    trait_formula: str
    covariate_names: tuple[str, ...]
    trait_names: tuple[str, ...]
    hidden_units: tuple[int, ...]
    checkpoint_version: str = TRAIT_GAMMA_CHECKPOINT_VERSION
    training_corpus_version: str = TRAIT_GAMMA_TRAINING_CORPUS_VERSION
    calibration: TraitGammaCalibration | None = None

    @classmethod
    def for_trait_gamma(
        cls,
        *,
        n_sites: int,
        n_species: int,
        n_covariates: int,
        n_traits: int,
        distribution: str = "probit",
        formula: str = "~ TMG",
        trait_formula: str = "~ CN",
        covariate_names: Sequence[str] = ("Intercept", "TMG"),
        trait_names: Sequence[str] = ("CN",),
        hidden_units: Sequence[int] = (64, 64),
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        gamma_prior_precision: float = 1.0,
    ) -> "TraitGammaNeuralHmscInference":
        distribution = str(distribution).lower()
        if distribution != "probit":
            raise ValueError("the public trait-Gamma checkpoint supports probit only")
        covariates = tuple(str(value) for value in covariate_names)
        traits = tuple(str(value) for value in trait_names)
        if len(covariates) != n_covariates or covariates[0] != "Intercept":
            raise ValueError("ordered covariates must match and start with Intercept")
        if len(traits) != n_traits or n_traits <= 0:
            raise ValueError("ordered trait names must match n_traits")
        hidden = tuple(int(value) for value in hidden_units)
        model = TraitGammaPosteriorModel(
            n_sites=n_sites,
            n_covariates=n_covariates,
            n_species=n_species,
            n_traits=n_traits,
            hidden_units=hidden,
            distribution=distribution,
            probit_anchor_iterations=probit_anchor_iterations,
            probit_anchor_prior_precision=probit_anchor_prior_precision,
            gamma_prior_precision=gamma_prior_precision,
        )
        _build_model(model)
        return cls(
            model=model,
            distribution=distribution,
            formula=str(formula),
            trait_formula=str(trait_formula),
            covariate_names=covariates,
            trait_names=traits,
            hidden_units=hidden,
        )

    @property
    def dimensions(self) -> dict[str, int]:
        return {
            "n_sites": self.model.n_sites,
            "n_species": self.model.n_species,
            "n_covariates": self.model.n_covariates,
            "n_traits": self.model.n_traits,
        }

    def fit(
        self,
        datasets: Sequence[TraitEffectDataset],
        *,
        epochs: int = 40,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        mse_weight: float = 0.25,
        seed: int = 123,
    ) -> FixedShapeTrainingHistory:
        if not datasets:
            raise ValueError("trait-Gamma training datasets must not be empty")
        for dataset in datasets:
            self._check_dataset(dataset)
        tf.keras.utils.set_random_seed(seed)
        return train_trait_gamma_model(
            self.model,
            trait_effect_training_data(datasets),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mse_weight=mse_weight,
        )

    def fit_calibration(
        self,
        datasets: Sequence[TraitEffectDataset],
        *,
        target_coverage: float = 0.95,
        provenance: dict[str, Any],
        method: str = "independent_simulation_scalar_gamma_scale",
    ) -> TraitGammaCalibration:
        if not datasets:
            raise ValueError("Gamma calibration datasets must not be empty")
        standardized: list[np.ndarray] = []
        count = 0
        for dataset in datasets:
            posterior = self.predict_gamma_posterior(dataset, calibrated=False)
            truth = dataset.truth_gamma.to_numpy(dtype=float)
            standardized.append(
                np.abs(posterior.mean.numpy()[0] - truth)
                / np.maximum(posterior.scale.numpy()[0], np.finfo(float).eps)
            )
            count += truth.size
        target_z = _normal_quantile(0.5 + target_coverage / 2.0)
        scores = np.concatenate([value.ravel() for value in standardized])
        if method == "split_conformal_scalar_gamma_scale":
            score_quantile = finite_sample_conformal_quantile(
                scores, target_coverage=target_coverage
            )
        elif method == "independent_simulation_scalar_gamma_scale":
            score_quantile = float(np.quantile(scores, target_coverage))
        else:
            raise ValueError(f"unsupported Gamma calibration method {method!r}")
        multiplier = float(score_quantile / target_z)
        self.calibration = TraitGammaCalibration(
            scale_multiplier=max(multiplier, 1e-3),
            target_coverage=float(target_coverage),
            n_coefficients=count,
            provenance=dict(provenance),
            method=method,
        )
        return self.calibration

    def predict_gamma_posterior(
        self, value: Any, *, calibrated: bool = True
    ) -> GammaPosterior:
        data, context = self._prepare(value)
        self._check_data(data, expected_batch=None)
        self._check_context(context)
        posterior = self.model({"X": data.X, "Y": data.Y, "T": data.T}, training=False)
        if not calibrated or self.calibration is None:
            return posterior
        return GammaPosterior(
            mean=posterior.mean,
            scale=posterior.scale * float(self.calibration.scale_multiplier),
        )

    def predict_beta_posterior(self, value: Any) -> BetaPosterior:
        data, context = self._prepare(value)
        self._check_data(data, expected_batch=None)
        self._check_context(context)
        return self.model.beta_anchor({"X": data.X, "Y": data.Y})

    def check_compatibility(self, value: Any) -> dict[str, Any]:
        data, context = self._prepare(value)
        self._check_data(data, expected_batch=1)
        self._check_context(context)
        return {
            "compatible": True,
            "model_family": TRAIT_GAMMA_MODEL_FAMILY,
            "distribution": context.distribution,
            "formula": {"X": context.formula, "T": context.trait_formula},
            "dimensions": self.dimensions,
            "posterior_parameters": ["Beta", "Gamma"],
            "joint_posterior_coupling": False,
        }

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
        gamma = self.predict_gamma_posterior(data)
        beta = self.predict_beta_posterior(data)
        artifact_metadata = {"neural_api": self._manifest()}
        if metadata:
            artifact_metadata.update(metadata)
        kwargs = {
            "covariate_names": context.covariate_names,
            "species_names": context.species_names,
            "trait_names": context.trait_names,
            "distribution": context.distribution,
            "formula": context.formula,
            "trait_formula": context.trait_formula,
            "chains": chains,
            "draws": draws,
            "seed": seed,
            "metadata": artifact_metadata,
            "gamma_calibration": self.calibration,
        }
        if output is None:
            with tempfile.TemporaryDirectory(prefix="trait-gamma-neural-hmsc-") as tmp:
                path = write_trait_gamma_posterior_hdf5(
                    gamma, beta, Path(tmp) / "posterior.h5", **kwargs
                )
                return HmscFit.from_file(path)
        path = write_trait_gamma_posterior_hdf5(gamma, beta, output, **kwargs)
        fit = HmscFit.from_file(path)
        fit.output_file = path
        return fit

    def save(self, checkpoint: str | Path) -> Path:
        root = Path(checkpoint).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"trait-Gamma checkpoint is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        _build_model(self.model)
        weights = root / TRAIT_GAMMA_CHECKPOINT_WEIGHTS
        self.model.save_weights(weights)
        calibration_record = None
        if self.calibration is not None:
            payload = self.calibration.to_metadata() | {
                "kind": "pyhmsc_trait_gamma_calibration",
                "schema_version": 1,
                "weights_sha256": _sha256(weights),
            }
            calibration_path = root / TRAIT_GAMMA_CALIBRATION_ARTIFACT
            calibration_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            calibration_record = {
                "path": TRAIT_GAMMA_CALIBRATION_ARTIFACT,
                "sha256": _sha256(calibration_path),
                "method": self.calibration.method,
            }
        manifest = self._manifest(calibration_record=calibration_record)
        (root / TRAIT_GAMMA_CHECKPOINT_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return root

    @classmethod
    def load(cls, checkpoint: str | Path) -> "TraitGammaNeuralHmscInference":
        root = Path(checkpoint).expanduser().resolve()
        manifest = json.loads(
            (root / TRAIT_GAMMA_CHECKPOINT_MANIFEST).read_text(encoding="utf-8")
        )
        if manifest.get("checkpoint_version") != TRAIT_GAMMA_CHECKPOINT_VERSION:
            raise NeuralHmscCompatibilityError(
                "unsupported trait-Gamma checkpoint version"
            )
        if manifest.get("model_family") != TRAIT_GAMMA_MODEL_FAMILY:
            raise NeuralHmscCompatibilityError("unsupported trait-Gamma model family")
        dimensions = manifest["dimensions"]
        names = manifest["names"]
        formula = manifest["formula"]
        anchor = manifest["anchor"]
        engine = cls.for_trait_gamma(
            n_sites=int(dimensions["n_sites"]),
            n_species=int(dimensions["n_species"]),
            n_covariates=int(dimensions["n_covariates"]),
            n_traits=int(dimensions["n_traits"]),
            distribution=str(manifest["distribution"]),
            formula=str(formula["X"]),
            trait_formula=str(formula["T"]),
            covariate_names=names["covariates"],
            trait_names=names["traits"],
            hidden_units=manifest["hidden_units"],
            probit_anchor_iterations=int(anchor["probit_iterations"]),
            probit_anchor_prior_precision=float(anchor["beta_prior_precision"]),
            gamma_prior_precision=float(anchor["gamma_prior_precision"]),
        )
        weights = root / TRAIT_GAMMA_CHECKPOINT_WEIGHTS
        engine.model.load_weights(weights)
        record = manifest.get("gamma_calibration")
        if record is not None:
            calibration_path = root / str(record["path"])
            if _sha256(calibration_path) != record["sha256"]:
                raise ValueError("trait-Gamma calibration artifact hash differs")
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            if payload.get("weights_sha256") != _sha256(weights):
                raise ValueError("trait-Gamma calibration weights hash differs")
            engine.calibration = TraitGammaCalibration.from_metadata(payload)
        return engine

    def _prepare(self, value: Any) -> tuple[TraitEffectTrainingData, _TraitContext]:
        if isinstance(value, TraitEffectTrainingData):
            return value, _TraitContext(
                self.distribution,
                self.formula,
                self.trait_formula,
                self.covariate_names,
                tuple(f"sp{idx + 1}" for idx in range(value.Y.shape[2])),
                self.trait_names,
            )
        if isinstance(value, TraitEffectDataset):
            self._check_dataset(value)
            return trait_effect_training_data([value]), _TraitContext(
                str(value.metadata["distribution"]),
                str(value.metadata["formula"]),
                str(value.metadata["trait_formula"]),
                tuple(str(name) for name in value.truth_gamma.index),
                tuple(str(name) for name in value.Y.columns),
                tuple(str(name) for name in value.truth_gamma.columns),
            )
        if isinstance(value, dict):
            for key in ("X", "Y", "T"):
                if key not in value:
                    raise NeuralHmscCompatibilityError(
                        f"mapping input must contain {key!r}"
                    )
            data = _training_data_from_arrays(value["X"], value["Y"], value["T"])
            return data, _TraitContext(
                str(value.get("distribution", self.distribution)),
                str(value.get("formula", self.formula)),
                str(value.get("trait_formula", self.trait_formula)),
                tuple(value.get("covariate_names", self.covariate_names)),
                tuple(
                    value.get(
                        "species_names",
                        [f"sp{idx + 1}" for idx in range(data.Y.shape[2])],
                    )
                ),
                tuple(value.get("trait_names", self.trait_names)),
            )
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_dir():
                path = path / "init.json"
            metadata, arrays = read_compiled_model(path)
            _check_compiled_structure(metadata, arrays)
            data = _training_data_from_arrays(arrays["X"], arrays["Y"], arrays["T"])
            names = metadata.get("names", {})
            formula = metadata.get("formula", {})
            return data, _TraitContext(
                str(metadata.get("distribution", "")),
                str(formula.get("X", "")),
                str(formula.get("T", self.trait_formula)),
                tuple(str(value) for value in names.get("covariates", ())),
                tuple(str(value) for value in names.get("species", ())),
                tuple(str(value) for value in names.get("traits", ())),
            )
        raise NeuralHmscCompatibilityError(
            "trait-Gamma inference supports TraitEffectDataset, mapping, "
            "TraitEffectTrainingData, or compiled init.json/directory"
        )

    def _check_dataset(self, dataset: TraitEffectDataset) -> None:
        data = trait_effect_training_data([dataset])
        self._check_data(data, expected_batch=1)
        context = _TraitContext(
            str(dataset.metadata.get("distribution", "")),
            str(dataset.metadata.get("formula", "")),
            str(dataset.metadata.get("trait_formula", "")),
            tuple(str(name) for name in dataset.truth_gamma.index),
            tuple(str(name) for name in dataset.Y.columns),
            tuple(str(name) for name in dataset.truth_gamma.columns),
        )
        self._check_context(context)

    def _check_data(
        self, data: TraitEffectTrainingData, *, expected_batch: int | None
    ) -> None:
        expected = (
            (self.model.n_sites, self.model.n_covariates),
            (self.model.n_sites, self.model.n_species),
            (self.model.n_species, self.model.n_traits),
        )
        if data.X.ndim != 3 or data.Y.ndim != 3 or data.T.ndim != 3:
            raise NeuralHmscCompatibilityError("trait-Gamma X, Y, and T must be rank-3")
        if (
            data.X.shape[1:] != expected[0]
            or data.Y.shape[1:] != expected[1]
            or data.T.shape[1:] != expected[2]
        ):
            raise NeuralHmscCompatibilityError(
                f"trait-Gamma data shape differs from checkpoint: "
                f"{data.X.shape[1:]}, {data.Y.shape[1:]}, {data.T.shape[1:]} vs {expected}"
            )
        if not (data.X.shape[0] == data.Y.shape[0] == data.T.shape[0]):
            raise NeuralHmscCompatibilityError("trait-Gamma batch sizes differ")
        if expected_batch is not None and data.X.shape[0] != expected_batch:
            raise NeuralHmscCompatibilityError(
                f"expected trait-Gamma batch {expected_batch}, got {data.X.shape[0]}"
            )

    def _check_context(self, context: _TraitContext) -> None:
        checks = {
            "distribution": (context.distribution, self.distribution),
            "X formula": (context.formula, self.formula),
            "T formula": (context.trait_formula, self.trait_formula),
            "ordered covariates": (context.covariate_names, self.covariate_names),
            "ordered traits": (context.trait_names, self.trait_names),
        }
        for label, (observed, expected) in checks.items():
            if observed != expected:
                raise NeuralHmscCompatibilityError(
                    f"artifact {label} {observed!r} does not match checkpoint {expected!r}"
                )

    def _manifest(
        self, *, calibration_record: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        record = calibration_record
        if record is None and self.calibration is not None:
            record = {
                "path": TRAIT_GAMMA_CALIBRATION_ARTIFACT,
                "method": self.calibration.method,
            }
        payload = {
            "checkpoint_version": self.checkpoint_version,
            "training_corpus_version": self.training_corpus_version,
            "model_family": TRAIT_GAMMA_MODEL_FAMILY,
            "posterior_family": "separate_diagonal_normal_marginals",
            "distribution": self.distribution,
            "formula": {"X": self.formula, "T": self.trait_formula},
            "dimensions": self.dimensions,
            "names": {
                "covariates": list(self.covariate_names),
                "traits": list(self.trait_names),
            },
            "hidden_units": list(self.hidden_units),
            "anchor": {
                "Beta": "probit_irls_laplace",
                "Gamma": "joint_site_species_probit_irls_laplace",
                "probit_iterations": self.model.probit_anchor_iterations,
                "beta_prior_precision": self.model.probit_anchor_prior_precision,
                "gamma_prior_precision": self.model.gamma_prior_precision,
            },
            "limitations": [
                "fixed-shape probit Beta and Gamma marginal inference only",
                "exact X/T formulas, preprocessing, dimensions, and ordered columns are required",
                "no phylogeny, random effects, spatial effects, or detection submodel",
                "Beta and Gamma draws are separate marginals, not a coupled joint posterior",
                "qualified Python MCMC remains the statistical reference",
            ],
        }
        if record is not None:
            payload["gamma_calibration"] = record
        return payload


def _training_data_from_arrays(X: Any, Y: Any, T: Any) -> TraitEffectTrainingData:
    design = np.asarray(X, dtype=np.float32)
    response = np.asarray(Y, dtype=np.float32)
    traits = np.asarray(T, dtype=np.float32)
    if design.ndim != 2 or response.ndim != 2 or traits.ndim != 2:
        raise NeuralHmscCompatibilityError(
            "mapping/compiled X, Y, and T must be rank-2"
        )
    if design.shape[0] != response.shape[0] or response.shape[1] != traits.shape[0]:
        raise NeuralHmscCompatibilityError(
            "mapping/compiled X, Y, and T dimensions differ"
        )
    return TraitEffectTrainingData(
        X=design[None, ...],
        Y=response[None, ...],
        T=traits[None, ...],
        Beta=np.zeros((1, design.shape[1], response.shape[1]), dtype=np.float32),
        Gamma=np.zeros((1, design.shape[1], traits.shape[1]), dtype=np.float32),
    )


def _check_compiled_structure(metadata: dict[str, Any], arrays: dict[str, Any]) -> None:
    capabilities = metadata.get("capabilities", {})
    if not capabilities.get("traits", False):
        raise NeuralHmscCompatibilityError("compiled artifact has no trait structure")
    unsupported = [
        name
        for name in ("random_levels", "phylogeny", "spatial")
        if capabilities.get(name, False)
    ]
    if unsupported:
        raise NeuralHmscCompatibilityError(
            f"compiled trait-Gamma artifact has unsupported structure: {unsupported}"
        )
    missing = sorted({"X", "Y", "T"}.difference(arrays))
    if missing:
        raise NeuralHmscCompatibilityError(
            f"compiled trait-Gamma arrays missing: {missing}"
        )


def _build_model(model: TraitGammaPosteriorModel) -> None:
    model(
        {
            "X": tf.zeros((1, model.n_sites, model.n_covariates), tf.float32),
            "Y": tf.zeros((1, model.n_sites, model.n_species), tf.float32),
            "T": tf.zeros((1, model.n_species, model.n_traits), tf.float32),
        },
        training=False,
    )


def _validate_calibration_provenance(provenance: dict[str, Any]) -> None:
    if provenance.get("independent_from_training") is not True:
        raise ValueError("Gamma calibration must be independent from training")
    if not provenance.get("corpus_id"):
        raise ValueError("Gamma calibration corpus_id is required")
    seeds = provenance.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Gamma calibration seeds are required")


def _normal_quantile(probability: float) -> float:
    return float(
        np.sqrt(2.0)
        * tf.math.erfinv(tf.constant(2.0 * probability - 1.0, tf.float64)).numpy()
    )


def finite_sample_conformal_quantile(scores: Any, *, target_coverage: float) -> float:
    """Return the split-conformal finite-sample upper order statistic."""
    values = np.asarray(scores, dtype=float).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("conformal scores must be non-empty and finite")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between zero and one")
    order = min(
        int(values.size),
        int(math.ceil((int(values.size) + 1) * float(target_coverage))),
    )
    return float(np.partition(values, order - 1)[order - 1])


def package_trait_gamma_calibration(
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    *,
    calibration: TraitGammaCalibration,
    expected_weights_sha256: str | None = None,
) -> Path:
    """Bind a frozen Gamma calibration without rewriting model weights."""
    source = Path(source_checkpoint).expanduser().resolve()
    output = Path(output_checkpoint).expanduser().resolve()
    if source == output:
        raise ValueError("output checkpoint must differ from source checkpoint")
    if output.exists():
        raise FileExistsError(f"output checkpoint already exists: {output}")
    engine = TraitGammaNeuralHmscInference.load(source)
    weights = source / TRAIT_GAMMA_CHECKPOINT_WEIGHTS
    weights_sha256 = _sha256(weights)
    if (
        expected_weights_sha256 is not None
        and weights_sha256 != expected_weights_sha256
    ):
        raise ValueError("source trait-Gamma weights differ from frozen candidate")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    try:
        payload = calibration.to_metadata() | {
            "kind": "pyhmsc_trait_gamma_calibration",
            "schema_version": 1,
            "weights_sha256": weights_sha256,
        }
        calibration_path = output / TRAIT_GAMMA_CALIBRATION_ARTIFACT
        calibration_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_path = output / TRAIT_GAMMA_CHECKPOINT_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["gamma_calibration"] = {
            "path": TRAIT_GAMMA_CALIBRATION_ARTIFACT,
            "sha256": _sha256(calibration_path),
            "method": calibration.method,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        loaded = TraitGammaNeuralHmscInference.load(output)
        if loaded.calibration != calibration:
            raise ValueError("packaged trait-Gamma calibration round-trip differs")
        if _sha256(output / TRAIT_GAMMA_CHECKPOINT_WEIGHTS) != weights_sha256:
            raise ValueError("packaging changed frozen trait-Gamma weights")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_trait_gamma_baseline(
    *,
    registry_root: str | Path,
    candidate_checkpoint: str | Path,
    qualification_report: str | Path,
    fixed_release_digest: str,
    variable_release_digest: str,
    baseline_id: str = TRAIT_GAMMA_BASELINE_ID,
) -> Path:
    """Atomically freeze a qualified trait-Gamma checkpoint and its evidence."""
    registry = Path(registry_root).expanduser().resolve()
    destination = registry / baseline_id
    if destination.exists():
        raise FileExistsError(f"trait-Gamma baseline already exists: {destination}")
    report_path = Path(qualification_report).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("decision") != "trait_gamma_probit_promoted"
        or report.get("all_gates_passed") is not True
    ):
        raise ValueError("trait-Gamma qualification report did not promote")
    if report.get("fixed_release_content_sha256") != fixed_release_digest:
        raise ValueError("fixed release digest differs from qualification")
    if report.get("variable_release_content_sha256") != variable_release_digest:
        raise ValueError("variable release digest differs from qualification")
    checkpoint = Path(candidate_checkpoint).expanduser().resolve()
    TraitGammaNeuralHmscInference.load(checkpoint)
    registry.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{baseline_id}.", dir=registry))
    try:
        shutil.copytree(checkpoint, staging / "checkpoint")
        shutil.copy2(report_path, staging / "qualification.json")
        inventory = _inventory(staging)
        manifest = {
            "kind": TRAIT_GAMMA_BASELINE_KIND,
            "schema_version": 1,
            "baseline_id": baseline_id,
            "status": "qualified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content_sha256": _inventory_sha256(inventory),
            "inventory": inventory,
            "checkpoint": {"path": "checkpoint"},
            "qualification": {"path": "qualification.json"},
            "fixed_release_id": "neural_hmsc_v0_1",
            "fixed_release_content_sha256": fixed_release_digest,
            "variable_release_id": "neural_hmsc_variable_probit_v1",
            "variable_release_content_sha256": variable_release_digest,
            "existing_releases_modified": False,
        }
        (staging / "baseline.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_trait_gamma_baseline(staging, expected_baseline_id=baseline_id)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def validate_trait_gamma_baseline(
    path: str | Path, *, expected_baseline_id: str | None = None
) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    payload = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
    if (
        payload.get("kind") != TRAIT_GAMMA_BASELINE_KIND
        or payload.get("schema_version") != 1
    ):
        raise ValueError("unsupported trait-Gamma baseline manifest")
    if (
        expected_baseline_id is not None
        and payload.get("baseline_id") != expected_baseline_id
    ):
        raise ValueError("trait-Gamma baseline identifier differs")
    if payload.get("existing_releases_modified") is not False:
        raise ValueError("trait-Gamma baseline modified an existing release")
    actual = _inventory(root)
    if actual != payload.get("inventory") or _inventory_sha256(actual) != payload.get(
        "content_sha256"
    ):
        raise ValueError("trait-Gamma baseline inventory differs")
    report = json.loads(
        (root / payload["qualification"]["path"]).read_text(encoding="utf-8")
    )
    if (
        report.get("decision") != "trait_gamma_probit_promoted"
        or report.get("all_gates_passed") is not True
    ):
        raise ValueError("trait-Gamma frozen qualification is not promoted")
    TraitGammaNeuralHmscInference.load(root / payload["checkpoint"]["path"])
    return payload


def load_trait_gamma_baseline(
    registry_root: str | Path, *, baseline_id: str = TRAIT_GAMMA_BASELINE_ID
) -> TraitGammaNeuralHmscInference:
    root = Path(registry_root).expanduser().resolve() / baseline_id
    payload = validate_trait_gamma_baseline(root, expected_baseline_id=baseline_id)
    return TraitGammaNeuralHmscInference.load(root / payload["checkpoint"]["path"])


def _inventory(root: Path) -> list[dict[str, Any]]:
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
