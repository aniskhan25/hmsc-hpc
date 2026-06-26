"""Evaluation helpers for experimental Neural-HMSC prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from pyhmsc.neural.posterior_heads import BetaPosterior, GammaPosterior, IidLatentPosterior
from pyhmsc.neural.train import FixedShapeTrainingData, IidLatentTrainingData, TraitEffectTrainingData, VariableShapeTrainingData


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


def predict_beta_posterior(model: tf.keras.Model, data: FixedShapeTrainingData) -> BetaPosterior:
    """Run a fixed-shape Beta model on prepared arrays."""
    return model({"X": data.X, "Y": data.Y}, training=False)


def predict_iid_latent_posterior(model: tf.keras.Model, data: IidLatentTrainingData) -> IidLatentPosterior:
    """Run an iid latent-factor model on prepared arrays."""
    return model({"X": data.X, "Y": data.Y, "group_codes": data.group_codes}, training=False)


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
