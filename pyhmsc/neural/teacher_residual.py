"""Predictive-only residual calibration distilled from simulated MCMC teachers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf


TEACHER_RESIDUAL_KIND = "pyhmsc_mcmc_teacher_logit_residual"
TEACHER_RESIDUAL_SCHEMA_VERSION = 3
NO_DEGRADATION_TOLERANCE = 1.0e-10
TEACHER_REPRESENTATION_VERSION = 3
TEACHER_RESIDUAL_FEATURES_V2 = (
    "baseline_logit",
    "baseline_probability",
    "species_prevalence_logit",
    "site_richness_fraction_logit",
    "log_design_information",
    "covariate_mean_abs",
    "covariate_root_mean_square",
    "baseline_logit_by_covariate_support",
    "log_n_sites",
    "log_n_species",
)
TEACHER_CONTEXT_FEATURES_V2 = (
    "probability_mean",
    "probability_std",
    "probability_q10",
    "probability_q90",
    "mean_abs_logit",
    "species_prevalence_q10",
    "species_prevalence_q50",
    "species_prevalence_q90",
    "site_richness_fraction_std",
    "mean_log_design_information",
    "covariate_mean",
    "covariate_std",
    "covariate_mean_abs",
    "covariate_root_mean_square",
)
TEACHER_RESIDUAL_FEATURES = (
    "baseline_logit",
    "baseline_probability",
    "species_prevalence_logit",
    "site_richness_fraction_logit",
    "log_mean_bernoulli_information",
    "covariate_mean_abs",
    "covariate_root_mean_square",
    "baseline_logit_by_covariate_support",
    "bounded_log_site_count",
    "log_n_species",
)
TEACHER_CONTEXT_FEATURES = (
    "probability_mean",
    "probability_std",
    "probability_q10",
    "probability_q90",
    "mean_abs_logit",
    "species_prevalence_q10",
    "species_prevalence_q50",
    "species_prevalence_q90",
    "site_richness_fraction_std",
    "mean_log_bernoulli_information",
    "bounded_log_site_count",
    "covariate_mean",
    "covariate_std",
    "covariate_mean_abs",
    "covariate_root_mean_square",
)


@dataclass(frozen=True)
class McmcTeacherResponseBatch:
    """One simulation-only response-probability distillation batch."""

    baseline_probability: np.ndarray
    teacher_probability: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    label: str
    seed: int
    profile: str = "compact"


@dataclass(frozen=True)
class ContextIdentityGate:
    """Outcome-independent prototype gate with an exact identity fallback."""

    feature_location: np.ndarray
    feature_scale: np.ndarray
    prototypes: dict[str, np.ndarray]
    approved_labels: tuple[str, ...]
    margin: float
    approved_distance_caps: dict[str, float]
    representation_version: int = TEACHER_REPRESENTATION_VERSION

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _context_feature_names(self.representation_version)

    def decision(
        self,
        baseline_probability: np.ndarray,
        X: np.ndarray,
    ) -> dict[str, Any]:
        summary = response_context_summary(
            baseline_probability,
            X,
            representation_version=self.representation_version,
        )
        normalized = (summary - self.feature_location) / self.feature_scale
        distances = {
            label: float(np.linalg.norm(normalized - prototype))
            for label, prototype in self.prototypes.items()
        }
        approved = {
            label: value
            for label, value in distances.items()
            if label in self.approved_labels
        }
        fallback = {
            label: value
            for label, value in distances.items()
            if label not in self.approved_labels
        }
        if not approved or not fallback:
            raise ValueError("context gate requires approved and fallback prototypes")
        approved_label = min(approved, key=approved.get)
        fallback_label = min(fallback, key=fallback.get)
        distance_cap = self.approved_distance_caps.get(approved_label, np.inf)
        active = bool(
            approved[approved_label] + self.margin < fallback[fallback_label]
            and approved[approved_label] <= distance_cap
        )
        return {
            "active": bool(active),
            "selected_label": approved_label if active else fallback_label,
            "approved_label": approved_label,
            "fallback_label": fallback_label,
            "approved_distance": approved[approved_label],
            "fallback_distance": fallback[fallback_label],
            "distance_margin": fallback[fallback_label] - approved[approved_label],
            "required_margin": float(self.margin),
            "approved_distance_cap": float(distance_cap),
            "within_approved_distance_cap": bool(
                approved[approved_label] <= distance_cap
            ),
            "distances": distances,
        }

    def multiplier(
        self,
        baseline_probability: np.ndarray,
        X: np.ndarray,
    ) -> float:
        return 1.0 if self.decision(baseline_probability, X)["active"] else 0.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "method": "nearest_prototype_identity_expert",
            "feature_names": list(self.feature_names),
            "representation_version": int(self.representation_version),
            "feature_location": self.feature_location.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "prototypes": {
                label: value.tolist() for label, value in self.prototypes.items()
            },
            "approved_labels": list(self.approved_labels),
            "margin": float(self.margin),
            "approved_distance_caps": {
                label: float(value)
                for label, value in self.approved_distance_caps.items()
            },
            "uses_outcomes": False,
            "fallback": "identity",
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "ContextIdentityGate":
        if payload.get("method") != "nearest_prototype_identity_expert":
            raise ValueError("unsupported teacher context gate")
        representation_version = int(payload.get("representation_version", 2))
        if tuple(payload.get("feature_names", ())) != _context_feature_names(
            representation_version
        ):
            raise ValueError("teacher context gate features differ")
        if payload.get("uses_outcomes") is not False:
            raise ValueError("teacher context gate used outcomes")
        if payload.get("fallback") != "identity":
            raise ValueError("teacher context gate lacks identity fallback")
        return cls(
            feature_location=np.asarray(payload["feature_location"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            prototypes={
                str(label): np.asarray(value, dtype=float)
                for label, value in payload["prototypes"].items()
            },
            approved_labels=tuple(str(value) for value in payload["approved_labels"]),
            margin=float(payload["margin"]),
            approved_distance_caps={
                str(label): float(value)
                for label, value in payload.get("approved_distance_caps", {}).items()
            },
            representation_version=representation_version,
        )


class McmcTeacherResidualHead:
    """Bounded logit-residual head with an independently selected shrinkage."""

    def __init__(
        self,
        model: tf.keras.Model,
        *,
        feature_location: np.ndarray,
        feature_scale: np.ndarray,
        max_abs_logit_residual: float,
        selected_shrinkage: float,
        baseline_id: str,
        metadata: dict[str, Any],
        context_gate: ContextIdentityGate | None = None,
        representation_version: int = TEACHER_REPRESENTATION_VERSION,
    ) -> None:
        self.model = model
        self.feature_location = np.asarray(feature_location, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.max_abs_logit_residual = float(max_abs_logit_residual)
        self.selected_shrinkage = float(selected_shrinkage)
        self.baseline_id = str(baseline_id)
        self.metadata = dict(metadata)
        self.context_gate = context_gate
        self.representation_version = int(representation_version)
        if context_gate is not None and (
            context_gate.representation_version != self.representation_version
        ):
            raise ValueError("teacher head and context gate representations differ")

    @property
    def selected(self) -> bool:
        return self.selected_shrinkage > 0.0

    def predict_residual(
        self,
        baseline_probability: np.ndarray,
        X: np.ndarray,
        *,
        use_selected_shrinkage: bool = True,
        use_context_gate: bool = True,
    ) -> np.ndarray:
        probability = _probability_array(baseline_probability)
        features = response_context_features(
            probability,
            X,
            representation_version=self.representation_version,
        )
        normalized = (features - self.feature_location) / self.feature_scale
        raw = self.model(normalized, training=False).numpy().reshape(probability.shape)
        shrinkage = self.selected_shrinkage if use_selected_shrinkage else 1.0
        context_multiplier = (
            self.context_gate.multiplier(probability, X)
            if use_context_gate and self.context_gate is not None
            else 1.0
        )
        return (
            float(shrinkage)
            * float(context_multiplier)
            * self.max_abs_logit_residual
            * np.tanh(raw)
        )

    def predict_mean(
        self,
        baseline_probability: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        probability = _probability_array(baseline_probability)
        residual = self.predict_residual(probability, X)
        return _sigmoid(_logit(probability) + residual)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": TEACHER_RESIDUAL_SCHEMA_VERSION,
            "kind": TEACHER_RESIDUAL_KIND,
            "artifact_role": "predictive_only_mean_residual",
            "baseline_id": self.baseline_id,
            "teacher": "qualified_python_mcmc_simulation_probability",
            "training_target": "mcmc_response_probability_only",
            "real_outcomes_used_for_training_or_selection": False,
            "coefficient_posterior_modified": False,
            "feature_names": list(
                _residual_feature_names(self.representation_version)
            ),
            "representation_version": int(self.representation_version),
            "feature_location": self.feature_location.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "max_abs_logit_residual": self.max_abs_logit_residual,
            "selected_shrinkage": self.selected_shrinkage,
            "selected": self.selected,
            "context_gate": (
                None if self.context_gate is None else self.context_gate.to_metadata()
            ),
            "model": {
                "input_dim": int(self.feature_location.size),
                "hidden_units": [
                    int(layer.units)
                    for layer in self.model.layers
                    if isinstance(layer, tf.keras.layers.Dense)
                ],
            },
            "diagnostics": self.metadata,
        }

    def save(self, output: str | Path) -> Path:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(directory / "weights.weights.h5")
        path = directory / "teacher_residual.json"
        path.write_text(
            json.dumps(self.to_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "McmcTeacherResidualHead":
        source = Path(path)
        metadata_path = source / "teacher_residual.json" if source.is_dir() else source
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("kind") != TEACHER_RESIDUAL_KIND:
            raise ValueError("unsupported MCMC-teacher residual artifact kind")
        schema_version = int(payload.get("schema_version", -1))
        if schema_version not in {1, 2, 3}:
            raise ValueError("unsupported MCMC-teacher residual schema")
        if payload.get("artifact_role") != "predictive_only_mean_residual":
            raise ValueError("MCMC-teacher residual is not predictive-only")
        if payload.get("training_target") != "mcmc_response_probability_only":
            raise ValueError("MCMC-teacher residual has unsupported training target")
        if payload.get("real_outcomes_used_for_training_or_selection") is not False:
            raise ValueError("MCMC-teacher residual used real outcomes")
        if payload.get("coefficient_posterior_modified") is not False:
            raise ValueError("MCMC-teacher residual changed coefficient semantics")
        representation_version = int(
            payload.get("representation_version", 2 if schema_version < 3 else 3)
        )
        feature_names = _residual_feature_names(representation_version)
        if tuple(payload.get("feature_names", ())) != feature_names:
            raise ValueError("MCMC-teacher residual features differ")
        hidden_units = tuple(int(value) for value in payload["model"]["hidden_units"])
        model = _residual_model(len(feature_names), hidden_units)
        model.load_weights(metadata_path.parent / "weights.weights.h5")
        return cls(
            model,
            feature_location=np.asarray(payload["feature_location"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            max_abs_logit_residual=float(payload["max_abs_logit_residual"]),
            selected_shrinkage=float(payload["selected_shrinkage"]),
            baseline_id=str(payload["baseline_id"]),
            metadata=dict(payload.get("diagnostics", {})),
            context_gate=(
                None
                if payload.get("context_gate") is None
                else ContextIdentityGate.from_metadata(payload["context_gate"])
            ),
            representation_version=representation_version,
        )


def fit_mcmc_teacher_residual_head(
    training_batches: Sequence[McmcTeacherResponseBatch],
    validation_batches: Sequence[McmcTeacherResponseBatch],
    *,
    baseline_id: str,
    hidden_units: tuple[int, ...] = (32, 16, 1),
    max_abs_logit_residual: float = 0.5,
    identity_penalty: float = 0.05,
    brier_weight: float = 1.0,
    epochs: int = 300,
    learning_rate: float = 0.01,
    shrinkage_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    target_label: str = "big_spatial_shape",
    seed: int = 0,
) -> McmcTeacherResidualHead:
    """Fit on teacher probabilities and select shrinkage on simulations only."""
    if not training_batches or not validation_batches:
        raise ValueError("training_batches and validation_batches must not be empty")
    if max_abs_logit_residual <= 0.0:
        raise ValueError("max_abs_logit_residual must be positive")
    if identity_penalty < 0.0 or brier_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if epochs <= 0 or learning_rate <= 0.0:
        raise ValueError("epochs and learning_rate must be positive")
    shrinkages = sorted({float(value) for value in shrinkage_grid})
    if (
        not shrinkages
        or shrinkages[0] != 0.0
        or any(value < 0.0 or value > 1.0 for value in shrinkages)
    ):
        raise ValueError("shrinkage_grid must include 0 and stay within [0, 1]")

    training = _stack_batches(training_batches)
    features = training["features"]
    location = np.mean(features, axis=0).astype(np.float32)
    scale = np.std(features, axis=0).astype(np.float32)
    scale = np.where(scale < 1.0e-5, 1.0, scale).astype(np.float32)
    normalized = ((features - location) / scale).astype(np.float32)
    baseline_logit = _logit(training["baseline"]).reshape(-1, 1).astype(np.float32)
    teacher = training["teacher"].reshape(-1, 1).astype(np.float32)

    tf.keras.utils.set_random_seed(int(seed))
    model = _residual_model(len(TEACHER_RESIDUAL_FEATURES), hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
    feature_tensor = tf.constant(normalized)
    baseline_tensor = tf.constant(baseline_logit)
    teacher_tensor = tf.constant(teacher)
    losses = []
    for _ in range(int(epochs)):
        with tf.GradientTape() as tape:
            raw = model(feature_tensor, training=True)
            residual = tf.cast(max_abs_logit_residual, raw.dtype) * tf.math.tanh(raw)
            candidate = tf.math.sigmoid(baseline_tensor + residual)
            clipped = tf.clip_by_value(candidate, 1.0e-6, 1.0 - 1.0e-6)
            cross_entropy = -tf.reduce_mean(
                teacher_tensor * tf.math.log(clipped)
                + (1.0 - teacher_tensor) * tf.math.log1p(-clipped)
            )
            brier = tf.reduce_mean(tf.square(candidate - teacher_tensor))
            identity = tf.reduce_mean(tf.square(residual))
            loss = (
                cross_entropy
                + float(brier_weight) * brier
                + float(identity_penalty) * identity
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(float(loss.numpy()))

    provisional = McmcTeacherResidualHead(
        model,
        feature_location=location,
        feature_scale=scale,
        max_abs_logit_residual=max_abs_logit_residual,
        selected_shrinkage=1.0,
        baseline_id=baseline_id,
        metadata={},
    )
    candidates = []
    for shrinkage in shrinkages:
        scores = _score_batches(
            provisional,
            validation_batches,
            shrinkage=shrinkage,
        )
        target = scores["by_label"].get(target_label)
        no_degradation = all(
            row["outcome_brier_ratio"] <= 1.0 and row["outcome_log_loss_ratio"] <= 1.0
            for row in scores["by_label"].values()
        )
        target_improved = bool(
            target is not None
            and target["teacher_brier_ratio"] < 1.0
            and target["teacher_cross_entropy_ratio"] < 1.0
            and target["outcome_brier_ratio"] < 1.0
            and target["outcome_log_loss_ratio"] < 1.0
        )
        accepted = bool(shrinkage > 0.0 and no_degradation and target_improved)
        candidates.append(
            {
                "shrinkage": shrinkage,
                "accepted": accepted,
                "objective": scores["overall"]["teacher_brier_candidate"]
                + scores["overall"]["teacher_cross_entropy_candidate"],
                "scores": scores,
            }
        )
    accepted = [row for row in candidates if row["accepted"]]
    selected = min(accepted, key=lambda row: row["objective"]) if accepted else None
    selected_shrinkage = 0.0 if selected is None else float(selected["shrinkage"])
    metadata = {
        "training": {
            "n_batches": len(training_batches),
            "n_probability_targets": int(teacher.size),
            "balanced_targets_per_batch": int(training["targets_per_batch"]),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "identity_penalty": float(identity_penalty),
            "brier_weight": float(brier_weight),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "outcomes_used_in_gradient": False,
            "batch_seeds": [int(batch.seed) for batch in training_batches],
        },
        "validation": {
            "n_batches": len(validation_batches),
            "target_label": target_label,
            "selection_rule": (
                "strict outcome no degradation in every simulated validation "
                "regime plus target teacher and outcome proper-score improvement"
            ),
            "candidates": candidates,
            "selected_shrinkage": selected_shrinkage,
            "batch_seeds": [int(batch.seed) for batch in validation_batches],
        },
    }
    return McmcTeacherResidualHead(
        model,
        feature_location=location,
        feature_scale=scale,
        max_abs_logit_residual=max_abs_logit_residual,
        selected_shrinkage=selected_shrinkage,
        baseline_id=baseline_id,
        metadata=metadata,
    )


def evaluate_mcmc_teacher_residual_head(
    head: McmcTeacherResidualHead,
    batches: Sequence[McmcTeacherResponseBatch],
    *,
    use_selected_shrinkage: bool = True,
    use_context_gate: bool = True,
) -> dict[str, Any]:
    """Score a frozen head against baseline, MCMC teacher, and simulated Y."""
    shrinkage = head.selected_shrinkage if use_selected_shrinkage else 1.0
    return _score_batches(
        head,
        batches,
        shrinkage=shrinkage,
        use_context_gate=use_context_gate,
    )


def response_context_features(
    baseline_probability: np.ndarray,
    X: np.ndarray,
    *,
    representation_version: int = TEACHER_REPRESENTATION_VERSION,
) -> np.ndarray:
    """Build site-species context features without target responses."""
    probability = _probability_array(baseline_probability)
    design = np.asarray(X, dtype=float)
    if design.ndim != 2 or design.shape[0] != probability.shape[0]:
        raise ValueError("X must be a site-by-covariate matrix aligned to probability")
    if design.shape[1] > 1 and np.allclose(design[:, 0], 1.0):
        covariates = design[:, 1:]
    else:
        covariates = design
    if covariates.shape[1] == 0:
        mean_abs = np.zeros(probability.shape[0], dtype=float)
        root_mean_square = np.zeros(probability.shape[0], dtype=float)
    else:
        mean_abs = np.mean(np.abs(covariates), axis=1)
        root_mean_square = np.sqrt(np.mean(np.square(covariates), axis=1))
    baseline_logit = _logit(probability)
    species_prevalence = np.mean(probability, axis=0)
    site_richness_fraction = np.mean(probability, axis=1)
    mean_information = np.mean(probability * (1.0 - probability), axis=0)
    support = mean_abs[:, None]
    common = [
        baseline_logit,
        probability,
        np.broadcast_to(_logit(species_prevalence)[None, :], probability.shape),
        np.broadcast_to(_logit(site_richness_fraction)[:, None], probability.shape),
    ]
    if representation_version == 2:
        information = probability.shape[0] * mean_information
        size = np.log1p(probability.shape[0])
        common.append(
            np.broadcast_to(np.log1p(information)[None, :], probability.shape)
        )
    elif representation_version == 3:
        size = _bounded_log_site_count(probability.shape[0])
        common.append(
            np.broadcast_to(
                np.log(np.maximum(mean_information, 1.0e-6))[None, :],
                probability.shape,
            )
        )
    else:
        raise ValueError("unsupported teacher representation version")
    features = np.stack(
        common
        + [
            np.broadcast_to(mean_abs[:, None], probability.shape),
            np.broadcast_to(root_mean_square[:, None], probability.shape),
            baseline_logit * support,
            np.full(probability.shape, size),
            np.full(probability.shape, np.log1p(probability.shape[1])),
        ],
        axis=-1,
    )
    return features.reshape(-1, features.shape[-1]).astype(np.float32)


def response_context_summary(
    baseline_probability: np.ndarray,
    X: np.ndarray,
    *,
    representation_version: int = TEACHER_REPRESENTATION_VERSION,
) -> np.ndarray:
    """Build one outcome-independent summary for context expert selection."""
    probability = _probability_array(baseline_probability)
    design = np.asarray(X, dtype=float)
    if design.ndim != 2 or design.shape[0] != probability.shape[0]:
        raise ValueError("X must be a site-by-covariate matrix aligned to probability")
    if design.shape[1] > 1 and np.allclose(design[:, 0], 1.0):
        covariates = design[:, 1:]
    else:
        covariates = design
    values = covariates.reshape(-1)
    if values.size == 0:
        values = np.zeros(1, dtype=float)
    prevalence = np.mean(probability, axis=0)
    richness = np.mean(probability, axis=1)
    mean_information = np.mean(probability * (1.0 - probability), axis=0)
    common = [
        np.mean(probability),
        np.std(probability),
        np.quantile(probability, 0.10),
        np.quantile(probability, 0.90),
        np.mean(np.abs(_logit(probability))),
        np.quantile(prevalence, 0.10),
        np.quantile(prevalence, 0.50),
        np.quantile(prevalence, 0.90),
        np.std(richness),
    ]
    if representation_version == 2:
        information = probability.shape[0] * mean_information
        common.append(np.mean(np.log1p(information)))
    elif representation_version == 3:
        common.extend(
            [
                np.mean(np.log(np.maximum(mean_information, 1.0e-6))),
                _bounded_log_site_count(probability.shape[0]),
            ]
        )
    else:
        raise ValueError("unsupported teacher representation version")
    summary = np.asarray(
        common
        + [
            np.mean(values),
            np.std(values),
            np.mean(np.abs(values)),
            np.sqrt(np.mean(np.square(values))),
        ],
        dtype=np.float32,
    )
    if np.any(~np.isfinite(summary)):
        raise ValueError("context summary contains non-finite values")
    return summary


def _bounded_log_site_count(n_sites: int) -> float:
    if int(n_sites) <= 0:
        raise ValueError("n_sites must be positive")
    return float(np.tanh(np.log(float(n_sites) / 20.0) / np.log(18.0)))


def _residual_feature_names(representation_version: int) -> tuple[str, ...]:
    if representation_version == 2:
        return TEACHER_RESIDUAL_FEATURES_V2
    if representation_version == 3:
        return TEACHER_RESIDUAL_FEATURES
    raise ValueError("unsupported teacher representation version")


def _context_feature_names(representation_version: int) -> tuple[str, ...]:
    if representation_version == 2:
        return TEACHER_CONTEXT_FEATURES_V2
    if representation_version == 3:
        return TEACHER_CONTEXT_FEATURES
    raise ValueError("unsupported teacher representation version")


def fit_context_identity_gate(
    batches: Sequence[McmcTeacherResponseBatch],
    *,
    approved_labels: Sequence[str],
    margin: float,
    approved_distance_caps: dict[str, float] | None = None,
) -> ContextIdentityGate:
    """Fit outcome-independent regime prototypes from simulated contexts."""
    if not batches:
        raise ValueError("context gate batches must not be empty")
    if margin < 0.0:
        raise ValueError("context gate margin must be non-negative")
    approved = tuple(dict.fromkeys(str(value) for value in approved_labels))
    if not approved:
        raise ValueError("approved_labels must not be empty")
    caps = approved_distance_caps or {}
    if any(label not in approved for label in caps):
        raise ValueError("approved distance caps must target approved labels")
    if any(not np.isfinite(value) or value <= 0.0 for value in caps.values()):
        raise ValueError("approved distance caps must be positive and finite")
    summaries = np.stack(
        [
            response_context_summary(batch.baseline_probability, batch.X)
            for batch in batches
        ]
    )
    location = np.mean(summaries, axis=0).astype(np.float32)
    scale = np.std(summaries, axis=0).astype(np.float32)
    scale = np.where(scale < 1.0e-5, 1.0, scale).astype(np.float32)
    normalized = (summaries - location) / scale
    labels = np.asarray([str(batch.label) for batch in batches], dtype=object)
    prototypes = {
        label: np.mean(normalized[labels == label], axis=0).astype(np.float32)
        for label in sorted(set(labels.tolist()))
    }
    missing = sorted(set(approved) - set(prototypes))
    if missing:
        raise ValueError(f"approved context labels are missing: {missing}")
    if set(prototypes).issubset(set(approved)):
        raise ValueError("context gate requires at least one fallback label")
    return ContextIdentityGate(
        feature_location=location,
        feature_scale=scale,
        prototypes=prototypes,
        approved_labels=approved,
        margin=float(margin),
        approved_distance_caps={label: float(value) for label, value in caps.items()},
    )


def fit_cross_fitted_mcmc_teacher_residual_head(
    calibration_batches: Sequence[McmcTeacherResponseBatch],
    *,
    baseline_id: str,
    approved_labels: tuple[str, ...] = (
        "effect_size_shift",
        "big_spatial_shape",
    ),
    fallback_labels: tuple[str, ...] = (
        "covariate_shift",
        "rare_validation",
    ),
    hidden_units: tuple[int, ...] = (32, 16, 1),
    max_abs_logit_residual: float = 0.5,
    identity_penalty: float = 0.05,
    brier_weight: float = 1.0,
    epochs: int = 300,
    learning_rate: float = 0.01,
    shrinkage_grid: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    margin_grid: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    approved_distance_cap_grid: Sequence[Sequence[float]] = (
        (3.0, 2.5),
        (3.5, 2.5),
        (4.0, 3.0),
    ),
    seed: int = 0,
) -> McmcTeacherResidualHead:
    """Select a gated residual through leave-one-community-out simulation folds."""
    if not calibration_batches:
        raise ValueError("calibration_batches must not be empty")
    communities = sorted({_community_seed(batch.seed) for batch in calibration_batches})
    if len(communities) < 3:
        raise ValueError(
            "cross-fitted teacher selection requires at least three communities"
        )
    shrinkages = sorted({float(value) for value in shrinkage_grid})
    margins = sorted({float(value) for value in margin_grid})
    if not shrinkages or any(value <= 0.0 or value > 1.0 for value in shrinkages):
        raise ValueError("cross-fit shrinkage_grid must stay within (0, 1]")
    if not margins or any(value < 0.0 for value in margins):
        raise ValueError("context margin_grid must be non-negative")
    cap_profiles = [
        tuple(float(value) for value in profile)
        for profile in approved_distance_cap_grid
    ]
    if not cap_profiles or any(
        len(profile) != len(approved_labels) for profile in cap_profiles
    ):
        raise ValueError(
            "each approved distance-cap profile must match approved_labels"
        )
    if any(
        not np.isfinite(value) or value <= 0.0
        for profile in cap_profiles
        for value in profile
    ):
        raise ValueError("approved distance caps must be positive and finite")

    fold_models = []
    for fold_index, community in enumerate(communities):
        training = [
            batch
            for batch in calibration_batches
            if _community_seed(batch.seed) != community
        ]
        validation = [
            batch
            for batch in calibration_batches
            if _community_seed(batch.seed) == community
        ]
        raw_head = fit_mcmc_teacher_residual_head(
            training,
            validation,
            baseline_id=baseline_id,
            hidden_units=hidden_units,
            max_abs_logit_residual=max_abs_logit_residual,
            identity_penalty=identity_penalty,
            brier_weight=brier_weight,
            epochs=epochs,
            learning_rate=learning_rate,
            shrinkage_grid=(0.0,),
            seed=int(seed) + fold_index,
        )
        fold_models.append((community, training, validation, raw_head))

    candidates = []
    for margin in margins:
        for cap_profile in cap_profiles:
            distance_caps = dict(zip(approved_labels, cap_profile))
            for shrinkage in shrinkages:
                folds = []
                for community, training, validation, raw_head in fold_models:
                    gate = fit_context_identity_gate(
                        training,
                        approved_labels=approved_labels,
                        margin=margin,
                        approved_distance_caps=distance_caps,
                    )
                    gated_head = McmcTeacherResidualHead(
                        raw_head.model,
                        feature_location=raw_head.feature_location,
                        feature_scale=raw_head.feature_scale,
                        max_abs_logit_residual=raw_head.max_abs_logit_residual,
                        selected_shrinkage=shrinkage,
                        baseline_id=baseline_id,
                        metadata={},
                        context_gate=gate,
                    )
                    scores = evaluate_mcmc_teacher_residual_head(gated_head, validation)
                    decisions = _context_decisions(gate, validation)
                    folds.append(
                        {
                            "community": int(community),
                            "scores": scores,
                            "context_decisions": decisions,
                        }
                    )
                candidate = _cross_fit_candidate(
                    shrinkage=shrinkage,
                    margin=margin,
                    approved_distance_caps=distance_caps,
                    folds=folds,
                    approved_labels=approved_labels,
                    fallback_labels=fallback_labels,
                )
                candidates.append(candidate)

    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    selected = min(accepted, key=lambda value: value["objective"]) if accepted else None
    selected_shrinkage = 0.0 if selected is None else float(selected["shrinkage"])
    selected_margin = max(margins) if selected is None else float(selected["margin"])
    selected_caps = (
        dict(zip(approved_labels, cap_profiles[0]))
        if selected is None
        else dict(selected["approved_distance_caps"])
    )

    final_raw = fit_mcmc_teacher_residual_head(
        calibration_batches,
        calibration_batches,
        baseline_id=baseline_id,
        hidden_units=hidden_units,
        max_abs_logit_residual=max_abs_logit_residual,
        identity_penalty=identity_penalty,
        brier_weight=brier_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        shrinkage_grid=(0.0,),
        seed=int(seed) + len(communities),
    )
    final_gate = fit_context_identity_gate(
        calibration_batches,
        approved_labels=approved_labels,
        margin=selected_margin,
        approved_distance_caps=selected_caps,
    )
    metadata = {
        "training": final_raw.metadata["training"],
        "cross_fit": {
            "method": "leave_one_simulated_community_out",
            "communities": communities,
            "n_folds": len(communities),
            "approved_labels": list(approved_labels),
            "fallback_labels": list(fallback_labels),
            "selection_rule": (
                "every fold and regime preserves outcome proper scores; approved "
                "contexts improve in every fold; unstable fallback contexts are identity"
            ),
            "candidates": candidates,
            "selected_shrinkage": selected_shrinkage,
            "selected_margin": selected_margin,
            "selected_approved_distance_caps": selected_caps,
            "outcomes_used_in_gradient": False,
            "outcomes_used_for_cross_fit_selection": True,
        },
    }
    return McmcTeacherResidualHead(
        final_raw.model,
        feature_location=final_raw.feature_location,
        feature_scale=final_raw.feature_scale,
        max_abs_logit_residual=final_raw.max_abs_logit_residual,
        selected_shrinkage=selected_shrinkage,
        baseline_id=baseline_id,
        metadata=metadata,
        context_gate=final_gate,
    )


def _community_seed(seed: int) -> int:
    value = int(seed)
    return value // 100 if abs(value) >= 100 else value


def _context_decisions(
    gate: ContextIdentityGate,
    batches: Sequence[McmcTeacherResponseBatch],
) -> list[dict[str, Any]]:
    rows = []
    for batch in batches:
        decision = gate.decision(batch.baseline_probability, batch.X)
        rows.append(
            {
                "seed": int(batch.seed),
                "label": str(batch.label),
                "profile": str(batch.profile),
                **decision,
            }
        )
    return rows


def _cross_fit_candidate(
    *,
    shrinkage: float,
    margin: float,
    approved_distance_caps: dict[str, float],
    folds: Sequence[dict[str, Any]],
    approved_labels: Sequence[str],
    fallback_labels: Sequence[str],
) -> dict[str, Any]:
    no_degradation = all(
        row["outcome_brier_ratio"] <= 1.0 + NO_DEGRADATION_TOLERANCE
        and row["outcome_log_loss_ratio"] <= 1.0 + NO_DEGRADATION_TOLERANCE
        for fold in folds
        for row in fold["scores"]["by_label"].values()
    )
    approved_direction_stable = all(
        fold["scores"]["by_label"][label]["outcome_brier_ratio"] < 1.0
        and fold["scores"]["by_label"][label]["outcome_log_loss_ratio"] < 1.0
        for fold in folds
        for label in approved_labels
    )
    approved_teacher_ratios = {
        label: {
            "teacher_brier_ratio": float(
                np.mean(
                    [
                        fold["scores"]["by_label"][label]["teacher_brier_ratio"]
                        for fold in folds
                    ]
                )
            ),
            "teacher_cross_entropy_ratio": float(
                np.mean(
                    [
                        fold["scores"]["by_label"][label]["teacher_cross_entropy_ratio"]
                        for fold in folds
                    ]
                )
            ),
        }
        for label in approved_labels
    }
    approved_teacher_improved = all(
        row["teacher_brier_ratio"] < 1.0 and row["teacher_cross_entropy_ratio"] < 1.0
        for row in approved_teacher_ratios.values()
    )
    approved_activated = all(
        decision["active"]
        for fold in folds
        for decision in fold["context_decisions"]
        if decision["label"] in approved_labels
    )
    fallback_identity = all(
        not decision["active"]
        for fold in folds
        for decision in fold["context_decisions"]
        if decision["label"] in fallback_labels
    )
    objective = float(
        np.mean(
            [
                fold["scores"]["by_label"][label][metric]
                for fold in folds
                for label in approved_labels
                for metric in ("outcome_brier_ratio", "outcome_log_loss_ratio")
            ]
        )
    )
    accepted = bool(
        no_degradation
        and approved_direction_stable
        and approved_teacher_improved
        and approved_activated
        and fallback_identity
    )
    return {
        "shrinkage": float(shrinkage),
        "margin": float(margin),
        "approved_distance_caps": {
            label: float(value) for label, value in approved_distance_caps.items()
        },
        "accepted": accepted,
        "objective": objective,
        "all_fold_regime_no_degradation": bool(no_degradation),
        "approved_direction_stable": bool(approved_direction_stable),
        "approved_teacher_improved": bool(approved_teacher_improved),
        "approved_activated": bool(approved_activated),
        "fallback_identity": bool(fallback_identity),
        "approved_teacher_ratios": approved_teacher_ratios,
        "folds": list(folds),
    }


def _stack_batches(
    batches: Sequence[McmcTeacherResponseBatch],
) -> dict[str, np.ndarray]:
    features = []
    baseline = []
    teacher = []
    outcome = []
    target_count = min(
        _probability_array(batch.baseline_probability).size for batch in batches
    )
    for batch in batches:
        base = _probability_array(batch.baseline_probability)
        target = _probability_array(batch.teacher_probability)
        Y = np.asarray(batch.Y, dtype=float)
        if base.shape != target.shape or base.shape != Y.shape:
            raise ValueError("baseline, teacher, and Y shapes must match")
        batch_features = response_context_features(base, batch.X)
        indices = np.linspace(
            0,
            base.size - 1,
            num=target_count,
            dtype=int,
        )
        features.append(batch_features[indices])
        baseline.append(base.reshape(-1)[indices])
        teacher.append(target.reshape(-1)[indices])
        outcome.append(Y.reshape(-1)[indices])
    return {
        "features": np.concatenate(features, axis=0),
        "baseline": np.concatenate(baseline),
        "teacher": np.concatenate(teacher),
        "outcome": np.concatenate(outcome),
        "targets_per_batch": np.asarray(target_count),
    }


def _score_batches(
    head: McmcTeacherResidualHead,
    batches: Sequence[McmcTeacherResponseBatch],
    *,
    shrinkage: float,
    use_context_gate: bool = True,
) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, float]]] = {}
    by_context: dict[str, list[dict[str, float]]] = {}
    overall_rows = []
    for batch in batches:
        baseline = _probability_array(batch.baseline_probability)
        residual = head.predict_residual(
            baseline,
            batch.X,
            use_selected_shrinkage=False,
            use_context_gate=use_context_gate,
        ) * float(shrinkage)
        candidate = _sigmoid(_logit(baseline) + residual)
        row = _score_pair(baseline, candidate, batch.teacher_probability, batch.Y)
        by_label.setdefault(str(batch.label), []).append(row)
        context = f"{batch.label}:{batch.profile}"
        by_context.setdefault(context, []).append(row)
        overall_rows.append(row)
    return {
        "overall": _average_score_rows(overall_rows),
        "by_label": {
            label: _average_score_rows(rows) for label, rows in by_label.items()
        },
        "by_context": {
            context: _average_score_rows(rows)
            for context, rows in by_context.items()
        },
    }


def _score_pair(
    baseline: np.ndarray,
    candidate: np.ndarray,
    teacher: np.ndarray,
    outcome: np.ndarray,
) -> dict[str, float]:
    baseline = _probability_array(baseline)
    candidate = _probability_array(candidate)
    teacher = _probability_array(teacher)
    outcome = np.asarray(outcome, dtype=float)
    teacher_brier_baseline = float(np.mean(np.square(baseline - teacher)))
    teacher_brier_candidate = float(np.mean(np.square(candidate - teacher)))
    teacher_ce_baseline = _soft_log_loss(baseline, teacher)
    teacher_ce_candidate = _soft_log_loss(candidate, teacher)
    outcome_brier_baseline = float(np.mean(np.square(baseline - outcome)))
    outcome_brier_candidate = float(np.mean(np.square(candidate - outcome)))
    outcome_log_baseline = _soft_log_loss(baseline, outcome)
    outcome_log_candidate = _soft_log_loss(candidate, outcome)
    eps = np.finfo(float).eps
    return {
        "teacher_brier_baseline": teacher_brier_baseline,
        "teacher_brier_candidate": teacher_brier_candidate,
        "teacher_brier_ratio": teacher_brier_candidate
        / max(teacher_brier_baseline, eps),
        "teacher_cross_entropy_baseline": teacher_ce_baseline,
        "teacher_cross_entropy_candidate": teacher_ce_candidate,
        "teacher_cross_entropy_ratio": teacher_ce_candidate
        / max(teacher_ce_baseline, eps),
        "outcome_brier_baseline": outcome_brier_baseline,
        "outcome_brier_candidate": outcome_brier_candidate,
        "outcome_brier_ratio": outcome_brier_candidate
        / max(outcome_brier_baseline, eps),
        "outcome_log_loss_baseline": outcome_log_baseline,
        "outcome_log_loss_candidate": outcome_log_candidate,
        "outcome_log_loss_ratio": outcome_log_candidate
        / max(outcome_log_baseline, eps),
        "mean_abs_logit_residual": float(
            np.mean(np.abs(_logit(candidate) - _logit(baseline)))
        ),
    }


def _average_score_rows(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("score rows must not be empty")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _residual_model(input_dim: int, hidden_units: Sequence[int]) -> tf.keras.Model:
    if not hidden_units or int(hidden_units[-1]) != 1:
        raise ValueError("hidden_units must end in one output unit")
    inputs = tf.keras.Input(shape=(int(input_dim),), name="response_context")
    value = inputs
    for index, units in enumerate(hidden_units[:-1]):
        value = tf.keras.layers.Dense(
            int(units), activation="relu", name=f"hidden_{index}"
        )(value)
    outputs = tf.keras.layers.Dense(
        1,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="logit_residual",
    )(value)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mcmc_teacher_residual")


def _probability_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("probability must be a non-empty site-by-species matrix")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("probabilities must be finite and within [0, 1]")
    return np.clip(array, 1.0e-6, 1.0 - 1.0e-6)


def _logit(value: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(value, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(probability) - np.log1p(-probability)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return np.where(
        value >= 0.0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )


def _soft_log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.clip(np.asarray(probability, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    target = np.asarray(target, dtype=float)
    return float(
        -np.mean(target * np.log(probability) + (1.0 - target) * np.log1p(-probability))
    )
