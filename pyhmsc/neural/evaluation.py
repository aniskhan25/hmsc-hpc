"""Evaluation helpers for experimental Neural-HMSC prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from pyhmsc.neural.posterior_heads import BetaPosterior, GammaPosterior, IidLatentPosterior
from pyhmsc.neural.train import (
    FixedShapeTrainingData,
    IidLatentTrainingData,
    SpatialLatentTrainingData,
    TraitEffectTrainingData,
    VariableShapeTrainingData,
)


@dataclass(frozen=True)
class BetaPosteriorMetrics:
    """Basic fixed-shape Beta posterior metrics."""

    beta_mean_rmse_truth: float
    beta_mean_mae_truth: float
    beta_interval_coverage_truth_95: float
    beta_interval_width_mean_95: float
    beta_scale_min: float
    beta_scale_mean: float
    zero_baseline_rmse_truth: float


@dataclass(frozen=True)
class GammaPosteriorMetrics:
    """Basic trait-effect Gamma posterior metrics."""

    gamma_mean_rmse_truth: float
    gamma_mean_mae_truth: float
    gamma_interval_coverage_truth_95: float
    gamma_interval_width_mean_95: float
    gamma_scale_min: float
    gamma_scale_mean: float
    zero_baseline_rmse_truth: float


@dataclass(frozen=True)
class IidLatentMetrics:
    """Invariant iid latent-factor recovery metrics."""

    random_effect_rmse_truth: float
    association_rmse_truth: float
    association_correlation_truth: float
    eta_scale_mean: float
    lambda_scale_mean: float


@dataclass(frozen=True)
class SpatialLatentMetrics:
    """Spatial latent-factor recovery and holdout metrics."""

    random_effect_rmse_truth: float
    association_rmse_truth: float
    association_correlation_truth: float
    holdout_nearest_rmse_truth: float
    holdout_conditional_rmse_truth: float
    residual_nearest_correlation: float


def predict_beta_posterior(model: tf.keras.Model, data: FixedShapeTrainingData) -> BetaPosterior:
    """Run a fixed-shape Beta model on prepared arrays."""
    return model({"X": data.X, "Y": data.Y}, training=False)


def predict_iid_latent_posterior(model: tf.keras.Model, data: IidLatentTrainingData) -> IidLatentPosterior:
    """Run an iid latent-factor model on prepared arrays."""
    return model({"X": data.X, "Y": data.Y, "group_codes": data.group_codes}, training=False)


def predict_spatial_latent_posterior(model: tf.keras.Model, data: SpatialLatentTrainingData) -> IidLatentPosterior:
    """Run a full-spatial latent-factor model on prepared arrays."""
    return model({"X": data.X, "Y": data.Y, "coords": data.coords}, training=False)


def predict_gamma_posterior(model: tf.keras.Model, data: TraitEffectTrainingData) -> GammaPosterior:
    """Run a trait-mediated Gamma model on prepared arrays."""
    return model({"X": data.X, "Y": data.Y, "T": data.T}, training=False)


def predict_variable_beta_posterior(model: tf.keras.Model, data: VariableShapeTrainingData) -> BetaPosterior:
    """Run a variable-shape Beta model on padded arrays."""
    return model(
        {
            "X": data.X,
            "Y": data.Y,
            "site_mask": data.site_mask,
            "species_mask": data.species_mask,
        },
        training=False,
    )


def evaluate_iid_latent_posterior(
    posterior: IidLatentPosterior,
    data: IidLatentTrainingData,
) -> IidLatentMetrics:
    """Evaluate iid latent factors through identifiable invariant summaries."""
    eta = posterior.eta_mean.numpy()
    loadings = posterior.lambda_mean.numpy()
    predicted_group_effect = np.einsum("bgf,bfs->bgs", eta, loadings)
    predicted_site_effect = np.stack(
        [
            predicted_group_effect[batch_idx, data.group_codes[batch_idx]]
            for batch_idx in range(predicted_group_effect.shape[0])
        ],
        axis=0,
    )
    truth_association = np.einsum("bfs,bft->bst", data.Lambda, data.Lambda)
    predicted_association = np.einsum("bfs,bft->bst", loadings, loadings)
    return IidLatentMetrics(
        random_effect_rmse_truth=float(np.sqrt(np.mean((predicted_site_effect - data.random_effect) ** 2))),
        association_rmse_truth=float(np.sqrt(np.mean((predicted_association - truth_association) ** 2))),
        association_correlation_truth=_flat_correlation(predicted_association, truth_association),
        eta_scale_mean=float(np.mean(posterior.eta_scale.numpy())),
        lambda_scale_mean=float(np.mean(posterior.lambda_scale.numpy())),
    )


def evaluate_spatial_latent_posterior(
    posterior: IidLatentPosterior,
    data: SpatialLatentTrainingData,
    *,
    spatial_range: float = 0.25,
) -> SpatialLatentMetrics:
    """Evaluate spatial latent factors through invariant and holdout metrics."""
    base = evaluate_iid_latent_posterior(posterior, data)
    nearest = spatial_holdout_random_effect_rmse(
        posterior,
        data,
        mode="nearest",
        spatial_range=spatial_range,
    )
    conditional = spatial_holdout_random_effect_rmse(
        posterior,
        data,
        mode="conditional",
        spatial_range=spatial_range,
    )
    predicted = _site_random_effect(posterior.eta_mean.numpy(), posterior.lambda_mean.numpy(), data.group_codes)
    residual = data.random_effect - predicted
    return SpatialLatentMetrics(
        random_effect_rmse_truth=base.random_effect_rmse_truth,
        association_rmse_truth=base.association_rmse_truth,
        association_correlation_truth=base.association_correlation_truth,
        holdout_nearest_rmse_truth=nearest,
        holdout_conditional_rmse_truth=conditional,
        residual_nearest_correlation=_nearest_residual_correlation(residual, data.coords),
    )


def spatial_holdout_random_effect_rmse(
    posterior: IidLatentPosterior,
    data: SpatialLatentTrainingData,
    *,
    mode: str = "conditional",
    spatial_range: float = 0.25,
) -> float:
    """Evaluate held-out spatial random-effect interpolation."""
    if mode not in {"nearest", "conditional"}:
        raise ValueError("mode must be 'nearest' or 'conditional'")
    eta = posterior.eta_mean.numpy()
    loadings = posterior.lambda_mean.numpy()
    errors = []
    for batch_idx in range(data.coords.shape[0]):
        train = data.train_mask[batch_idx]
        test = data.test_mask[batch_idx]
        train_coords = data.coords[batch_idx, train]
        test_coords = data.coords[batch_idx, test]
        train_eta = eta[batch_idx, train]
        if mode == "nearest":
            distances = _cross_distances(test_coords, train_coords)
            predicted_eta = train_eta[np.argmin(distances, axis=1)]
        else:
            distances = _cross_distances(test_coords, train_coords)
            weights = np.exp(-distances / float(spatial_range))
            weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), np.finfo(float).eps)
            predicted_eta = weights @ train_eta
        predicted_effect = predicted_eta @ loadings[batch_idx]
        truth = data.random_effect[batch_idx, test]
        errors.append((predicted_effect - truth).ravel())
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def evaluate_gamma_posterior(
    posterior: GammaPosterior,
    gamma_true: np.ndarray,
    *,
    z_value: float = 1.959963984540054,
) -> GammaPosteriorMetrics:
    """Evaluate Gamma posterior mean and interval behavior against truth."""
    mean = posterior.mean.numpy()
    scale = posterior.scale.numpy()
    gamma_true = np.asarray(gamma_true, dtype=np.float32)
    error = mean - gamma_true
    lower = mean - z_value * scale
    upper = mean + z_value * scale
    covered = (gamma_true >= lower) & (gamma_true <= upper)
    return GammaPosteriorMetrics(
        gamma_mean_rmse_truth=float(np.sqrt(np.mean(error**2))),
        gamma_mean_mae_truth=float(np.mean(np.abs(error))),
        gamma_interval_coverage_truth_95=float(np.mean(covered)),
        gamma_interval_width_mean_95=float(np.mean(upper - lower)),
        gamma_scale_min=float(np.min(scale)),
        gamma_scale_mean=float(np.mean(scale)),
        zero_baseline_rmse_truth=float(np.sqrt(np.mean(gamma_true**2))),
    )


def _flat_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=float).ravel()
    right_flat = np.asarray(right, dtype=float).ravel()
    if left_flat.size < 2 or np.std(left_flat) == 0.0 or np.std(right_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def _site_random_effect(eta: np.ndarray, loadings: np.ndarray, group_codes: np.ndarray) -> np.ndarray:
    group_effect = np.einsum("bgf,bfs->bgs", eta, loadings)
    return np.stack(
        [group_effect[batch_idx, group_codes[batch_idx]] for batch_idx in range(group_effect.shape[0])],
        axis=0,
    )


def _nearest_residual_correlation(residual: np.ndarray, coords: np.ndarray) -> float:
    values = residual.mean(axis=-1)
    paired = []
    for batch_idx in range(coords.shape[0]):
        distances = _cross_distances(coords[batch_idx], coords[batch_idx])
        np.fill_diagonal(distances, np.inf)
        nearest = np.argmin(distances, axis=1)
        paired.append(np.column_stack([values[batch_idx], values[batch_idx, nearest]]))
    pairs = np.vstack(paired)
    return _flat_correlation(pairs[:, 0], pairs[:, 1])


def _cross_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def evaluate_beta_posterior(
    posterior: BetaPosterior,
    beta_true: np.ndarray,
    *,
    z_value: float = 1.959963984540054,
) -> BetaPosteriorMetrics:
    """Evaluate posterior mean and interval behavior against simulated truth."""
    mean = posterior.mean.numpy()
    scale = posterior.scale.numpy()
    beta_true = np.asarray(beta_true, dtype=np.float32)
    error = mean - beta_true
    lower = mean - z_value * scale
    upper = mean + z_value * scale
    covered = (beta_true >= lower) & (beta_true <= upper)
    return BetaPosteriorMetrics(
        beta_mean_rmse_truth=float(np.sqrt(np.mean(error**2))),
        beta_mean_mae_truth=float(np.mean(np.abs(error))),
        beta_interval_coverage_truth_95=float(np.mean(covered)),
        beta_interval_width_mean_95=float(np.mean(upper - lower)),
        beta_scale_min=float(np.min(scale)),
        beta_scale_mean=float(np.mean(scale)),
        zero_baseline_rmse_truth=float(np.sqrt(np.mean(beta_true**2))),
    )


def evaluate_masked_beta_posterior(
    posterior: BetaPosterior,
    beta_true: np.ndarray,
    species_mask: np.ndarray,
    *,
    z_value: float = 1.959963984540054,
) -> BetaPosteriorMetrics:
    """Evaluate variable-shape Beta posterior metrics over unpadded species."""
    mean = posterior.mean.numpy()
    scale = posterior.scale.numpy()
    beta_true = np.asarray(beta_true, dtype=np.float32)
    mask = np.asarray(species_mask, dtype=bool)[:, None, :]
    mask = np.broadcast_to(mask, beta_true.shape)
    error = mean - beta_true
    lower = mean - z_value * scale
    upper = mean + z_value * scale
    covered = (beta_true >= lower) & (beta_true <= upper)
    return BetaPosteriorMetrics(
        beta_mean_rmse_truth=float(np.sqrt(np.mean(error[mask] ** 2))),
        beta_mean_mae_truth=float(np.mean(np.abs(error[mask]))),
        beta_interval_coverage_truth_95=float(np.mean(covered[mask])),
        beta_interval_width_mean_95=float(np.mean((upper - lower)[mask])),
        beta_scale_min=float(np.min(scale[mask])),
        beta_scale_mean=float(np.mean(scale[mask])),
        zero_baseline_rmse_truth=float(np.sqrt(np.mean(beta_true[mask] ** 2))),
    )
