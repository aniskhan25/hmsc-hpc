"""Evaluation helpers for experimental Neural-HMSC prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.train import FixedShapeTrainingData


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


def predict_beta_posterior(model: tf.keras.Model, data: FixedShapeTrainingData) -> BetaPosterior:
    """Run a fixed-shape Beta model on prepared arrays."""
    return model({"X": data.X, "Y": data.Y}, training=False)


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

