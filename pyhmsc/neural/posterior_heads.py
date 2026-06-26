"""Posterior heads for experimental Neural-HMSC models."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class BetaPosterior:
    """Diagonal Normal posterior approximation for fixed-effect Beta."""

    mean: tf.Tensor
    scale: tf.Tensor


class DiagonalNormalBetaHead(tf.keras.layers.Layer):
    """Map encoder features to a diagonal Normal posterior over Beta."""

    def __init__(
        self,
        n_covariates: int,
        n_species: int,
        min_scale: float = 1e-3,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_covariates = int(n_covariates)
        self.n_species = int(n_species)
        self.min_scale = float(min_scale)
        self.projection = tf.keras.layers.Dense(
            2 * self.n_covariates * self.n_species,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(self, features: tf.Tensor) -> BetaPosterior:
        raw = self.projection(features)
        mean_raw, scale_raw = tf.split(raw, 2, axis=-1)
        mean = tf.reshape(mean_raw, (-1, self.n_covariates, self.n_species))
        scale = tf.reshape(tf.nn.softplus(scale_raw) + self.min_scale, (-1, self.n_covariates, self.n_species))
        return BetaPosterior(mean=mean, scale=scale)


def beta_negative_log_probability(posterior: BetaPosterior, beta_true: tf.Tensor) -> tf.Tensor:
    """Mean negative log probability of true Beta under a diagonal Normal."""
    beta_true = tf.cast(beta_true, posterior.mean.dtype)
    variance = tf.square(posterior.scale)
    log_prob = -0.5 * (
        tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=posterior.mean.dtype))
        + tf.math.log(variance)
        + tf.square(beta_true - posterior.mean) / variance
    )
    return -tf.reduce_mean(tf.reduce_sum(log_prob, axis=(1, 2)))


def sample_beta_posterior(
    posterior: BetaPosterior,
    draws: int,
    seed: int | None = None,
) -> tf.Tensor:
    """Draw Beta samples with shape draws x batch x covariates x species."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    if seed is not None:
        generator = tf.random.Generator.from_seed(seed)
        noise = generator.normal((draws,) + tuple(posterior.mean.shape), dtype=posterior.mean.dtype)
    else:
        noise = tf.random.normal((draws,) + tuple(posterior.mean.shape), dtype=posterior.mean.dtype)
    return posterior.mean[None, ...] + noise * posterior.scale[None, ...]
