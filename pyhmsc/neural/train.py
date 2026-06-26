"""Training helpers for experimental Neural-HMSC prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import tensorflow as tf

from pyhmsc.neural.posterior_heads import beta_negative_log_probability
from pyhmsc.neural.simulator import FixedEffectDataset


@dataclass(frozen=True)
class FixedShapeTrainingData:
    """Tensor arrays for fixed-shape Beta posterior training."""

    X: np.ndarray
    Y: np.ndarray
    Beta: np.ndarray


@dataclass(frozen=True)
class FixedShapeTrainingHistory:
    """Compact training history for fixed-shape prototype loops."""

    loss: list[float]
    beta_rmse: list[float]
    scale_mean: list[float]


def fixed_shape_training_data(datasets: Sequence[FixedEffectDataset]) -> FixedShapeTrainingData:
    """Convert same-shape fixed-effect datasets to model-ready arrays."""
    if not datasets:
        raise ValueError("datasets must not be empty")
    X_arrays = []
    Y_arrays = []
    beta_arrays = []
    expected_shape = None
    for dataset in datasets:
        design = np.column_stack(
            [
                np.ones(len(dataset.X), dtype=np.float32),
                dataset.X[["x1", "x2"]].to_numpy(dtype=np.float32),
            ]
        )
        Y = dataset.Y.to_numpy(dtype=np.float32)
        beta = dataset.truth_beta.loc[["Intercept", "x1", "x2"]].to_numpy(dtype=np.float32)
        shape = (design.shape, Y.shape, beta.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all datasets must have the same fixed shapes")
        X_arrays.append(design)
        Y_arrays.append(Y)
        beta_arrays.append(beta)
    return FixedShapeTrainingData(
        X=np.stack(X_arrays).astype(np.float32),
        Y=np.stack(Y_arrays).astype(np.float32),
        Beta=np.stack(beta_arrays).astype(np.float32),
    )


def train_fixed_shape_beta_model(
    model: tf.keras.Model,
    data: FixedShapeTrainingData,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    mse_weight: float = 0.25,
    verbose: int = 0,
) -> FixedShapeTrainingHistory:
    """Train a fixed-shape Beta posterior model against simulated truth."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dataset = (
        tf.data.Dataset.from_tensor_slices((data.X, data.Y, data.Beta))
        .shuffle(buffer_size=len(data.Beta), seed=123, reshuffle_each_iteration=True)
        .batch(batch_size)
    )
    history = {"loss": [], "beta_rmse": [], "scale_mean": []}

    def train_step(
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        beta_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            posterior = model({"X": x_batch, "Y": y_batch}, training=True)
            nll = beta_negative_log_probability(posterior, beta_batch)
            mse = tf.reduce_mean(tf.square(beta_batch - posterior.mean))
            loss = nll + float(mse_weight) * mse
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(beta_batch - posterior.mean)))
        scale_mean = tf.reduce_mean(posterior.scale)
        return loss, rmse, scale_mean

    for epoch in range(int(epochs)):
        epoch_loss = []
        epoch_rmse = []
        epoch_scale = []
        for x_batch, y_batch, beta_batch in dataset:
            loss, rmse, scale_mean = train_step(x_batch, y_batch, beta_batch)
            epoch_loss.append(float(loss.numpy()))
            epoch_rmse.append(float(rmse.numpy()))
            epoch_scale.append(float(scale_mean.numpy()))
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
