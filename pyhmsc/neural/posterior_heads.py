"""Posterior heads for experimental Neural-HMSC models."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class BetaPosterior:
    """Normal posterior approximation for fixed-effect Beta.

    ``scale`` always contains marginal standard deviations with shape
    ``batch x covariates x species``. Full-covariance posteriors additionally
    carry a per-species Cholesky factor with shape
    ``batch x species x covariates x covariates``.
    """

    mean: tf.Tensor
    scale: tf.Tensor
    scale_tril: tf.Tensor | None = None

    @property
    def posterior_family(self) -> str:
        return "full_covariance_normal" if self.scale_tril is not None else "diagonal_normal"


@dataclass(frozen=True)
class GammaPosterior:
    """Diagonal Normal posterior approximation for trait-effect Gamma."""

    mean: tf.Tensor
    scale: tf.Tensor


@dataclass(frozen=True)
class IidLatentPosterior:
    """Diagonal Normal posterior approximation for iid latent factors."""

    beta_mean: tf.Tensor
    beta_scale: tf.Tensor
    eta_mean: tf.Tensor
    eta_scale: tf.Tensor
    lambda_mean: tf.Tensor
    lambda_scale: tf.Tensor


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


class FullCovarianceNormalBetaHead(tf.keras.layers.Layer):
    """Map encoder features to per-species full-covariance Beta posteriors."""

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
        self.n_tril = self.n_covariates * (self.n_covariates + 1) // 2
        self.projection = tf.keras.layers.Dense(
            self.n_species * (self.n_covariates + self.n_tril),
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )
        basis = []
        for row in range(self.n_covariates):
            for column in range(row + 1):
                matrix = [[0.0] * self.n_covariates for _ in range(self.n_covariates)]
                matrix[row][column] = 1.0
                basis.append(matrix)
        self._tril_basis = tf.constant(basis, dtype=tf.float32)

    def call(self, features: tf.Tensor) -> BetaPosterior:
        raw = self.projection(features)
        mean_size = self.n_covariates * self.n_species
        mean_raw, tril_raw = tf.split(raw, [mean_size, self.n_species * self.n_tril], axis=-1)
        mean = tf.reshape(mean_raw, (-1, self.n_covariates, self.n_species))
        tril_values = tf.reshape(tril_raw, (-1, self.n_species, self.n_tril))
        scale_tril = tf.einsum(
            "bsn,nij->bsij",
            tril_values,
            tf.cast(self._tril_basis, tril_values.dtype),
        )
        diagonal = tf.nn.softplus(tf.linalg.diag_part(scale_tril)) + self.min_scale
        scale_tril = tf.linalg.set_diag(scale_tril, diagonal)
        marginal_scale = tf.sqrt(tf.reduce_sum(tf.square(scale_tril), axis=-1))
        scale = tf.transpose(marginal_scale, [0, 2, 1])
        return BetaPosterior(mean=mean, scale=scale, scale_tril=scale_tril)


class DiagonalNormalGammaHead(tf.keras.layers.Layer):
    """Map encoder features to a diagonal Normal posterior over Gamma."""

    def __init__(
        self,
        n_covariates: int,
        n_traits: int,
        min_scale: float = 1e-3,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_covariates = int(n_covariates)
        self.n_traits = int(n_traits)
        self.min_scale = float(min_scale)
        self.projection = tf.keras.layers.Dense(
            2 * self.n_covariates * self.n_traits,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(self, features: tf.Tensor) -> GammaPosterior:
        raw = self.projection(features)
        mean_raw, scale_raw = tf.split(raw, 2, axis=-1)
        mean = tf.reshape(mean_raw, (-1, self.n_covariates, self.n_traits))
        scale = tf.reshape(tf.nn.softplus(scale_raw) + self.min_scale, (-1, self.n_covariates, self.n_traits))
        return GammaPosterior(mean=mean, scale=scale)


def beta_negative_log_probability(posterior: BetaPosterior, beta_true: tf.Tensor) -> tf.Tensor:
    """Mean negative log probability of true Beta under its Normal family."""
    beta_true = tf.cast(beta_true, posterior.mean.dtype)
    if posterior.scale_tril is not None:
        residual = tf.transpose(beta_true - posterior.mean, [0, 2, 1])
        solved = tf.linalg.triangular_solve(
            tf.cast(posterior.scale_tril, posterior.mean.dtype),
            residual[..., None],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solved), axis=(-2, -1))
        log_determinant = tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(posterior.scale_tril)),
            axis=-1,
        )
        normalizer = tf.cast(tf.shape(posterior.mean)[1], posterior.mean.dtype) * tf.math.log(
            2.0 * tf.constant(3.141592653589793, dtype=posterior.mean.dtype)
        )
        nll = 0.5 * (normalizer + 2.0 * log_determinant + mahalanobis)
        return tf.reduce_mean(tf.reduce_sum(nll, axis=1))
    variance = tf.square(posterior.scale)
    log_prob = -0.5 * (
        tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=posterior.mean.dtype))
        + tf.math.log(variance)
        + tf.square(beta_true - posterior.mean) / variance
    )
    return -tf.reduce_mean(tf.reduce_sum(log_prob, axis=(1, 2)))


def gamma_negative_log_probability(posterior: GammaPosterior, gamma_true: tf.Tensor) -> tf.Tensor:
    """Mean negative log probability of true Gamma under a diagonal Normal."""
    gamma_true = tf.cast(gamma_true, posterior.mean.dtype)
    variance = tf.square(posterior.scale)
    log_prob = -0.5 * (
        tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=posterior.mean.dtype))
        + tf.math.log(variance)
        + tf.square(gamma_true - posterior.mean) / variance
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
    generator = tf.random.Generator.from_seed(seed) if seed is not None else None
    if posterior.scale_tril is not None:
        noise_shape = (
            draws,
            int(posterior.mean.shape[0]),
            int(posterior.mean.shape[2]),
            int(posterior.mean.shape[1]),
        )
        noise = (
            generator.normal(noise_shape, dtype=posterior.mean.dtype)
            if generator is not None
            else tf.random.normal(noise_shape, dtype=posterior.mean.dtype)
        )
        correlated = tf.einsum("bsij,dbsj->dbsi", posterior.scale_tril, noise)
        return posterior.mean[None, ...] + tf.transpose(correlated, [0, 1, 3, 2])
    noise_shape = (draws,) + tuple(posterior.mean.shape)
    noise = (
        generator.normal(noise_shape, dtype=posterior.mean.dtype)
        if generator is not None
        else tf.random.normal(noise_shape, dtype=posterior.mean.dtype)
    )
    return posterior.mean[None, ...] + noise * posterior.scale[None, ...]


def sample_gamma_posterior(
    posterior: GammaPosterior,
    draws: int,
    seed: int | None = None,
) -> tf.Tensor:
    """Draw Gamma samples with shape draws x batch x covariates x traits."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    if seed is not None:
        generator = tf.random.Generator.from_seed(seed)
        noise = generator.normal((draws,) + tuple(posterior.mean.shape), dtype=posterior.mean.dtype)
    else:
        noise = tf.random.normal((draws,) + tuple(posterior.mean.shape), dtype=posterior.mean.dtype)
    return posterior.mean[None, ...] + noise * posterior.scale[None, ...]
