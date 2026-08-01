"""Orbit-symmetrized generative Neural-HMSC iid probit posterior.

This module implements the preregistered
``generative_neural_hmsc_iid_probit_v2_orbit`` representation.  It reuses the
v1 generative state layout, prior, and likelihood, but does not load or anchor
to v1 posterior outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import numpy as np
import tensorflow as tf

from pyhmsc.neural.generative_iid import (
    GENERATIVE_IID_MAX_SITES,
    GENERATIVE_IID_MAX_SPECIES,
    GENERATIVE_IID_MIN_SITES,
    GENERATIVE_IID_MIN_SPECIES,
    GENERATIVE_IID_N_COVARIATES,
    GENERATIVE_IID_N_FACTORS,
    GenerativeIidBatch,
    GenerativeTrainingHistory,
    JointStateLayout,
    generative_log_prior,
    probit_log_likelihood,
)


GENERATIVE_IID_V2_PROTOCOL = "generative_neural_hmsc_iid_probit_v2_orbit"
GENERATIVE_IID_V2_SCHEMA_VERSION = 2
GENERATIVE_IID_V2_POSTERIOR_RANK = 16
GENERATIVE_IID_V2_HIDDEN_WIDTH = 96
GENERATIVE_IID_V2_ATTENTION_HEADS = 4
GENERATIVE_IID_V2_ATTENTION_BLOCKS = 4
GENERATIVE_IID_V2_FEEDFORWARD_WIDTH = 192
GENERATIVE_IID_V2_REFINEMENT_STEPS = (0.05, 0.025, 0.0125, 0.00625)
GENERATIVE_IID_V2_REFINEMENT_DRAWS = 8
GENERATIVE_IID_V2_PREREGISTRATION_SHA256 = (
    "a2eaee0441833167f707f7cb9ae6b1162ba4e118ee3dfc1a245983cc9ada24c2"
)
GENERATIVE_IID_V2_SEED_AUDIT_SHA256 = (
    "9a463943508651e74855701cdbd9870961efd3fd3c07a444674da36a67d49344"
)
GENERATIVE_IID_V1_SOURCE_SHA256 = (
    "a7885c9123ac4e52beb1ed366fd5c09857f132789e21cac540be6c96663b8d52"
)
GENERATIVE_IID_V1_ARTIFACT_SHA256 = (
    "fb6429a5a58eee2caffcd1f33118847db269b53cfdcd4fc3556d9ae1ed523cac"
)


def _seed_pair(seed: int | None, stream: int) -> tf.Tensor | None:
    if seed is None:
        return None
    return tf.constant([int(seed), int(stream)], dtype=tf.int32)


def _normal(
    shape: tf.Tensor | list[int],
    *,
    dtype: tf.dtypes.DType,
    seed: int | None,
    stream: int,
) -> tf.Tensor:
    pair = _seed_pair(seed, stream)
    if pair is None:
        return tf.random.normal(shape, dtype=dtype)
    return tf.random.stateless_normal(shape, seed=pair, dtype=dtype)


def _uniform(
    shape: tf.Tensor | list[int],
    *,
    dtype: tf.dtypes.DType,
    seed: int | None,
    stream: int,
) -> tf.Tensor:
    pair = _seed_pair(seed, stream)
    if pair is None:
        return tf.random.uniform(shape, dtype=dtype)
    return tf.random.stateless_uniform(shape, seed=pair, dtype=dtype)


def _gamma(
    draws: int,
    *,
    alpha: tf.Tensor,
    dtype: tf.dtypes.DType,
    seed: int | None,
    stream: int,
) -> tf.Tensor:
    pair = _seed_pair(seed, stream)
    if pair is None:
        return tf.random.gamma([draws], alpha=alpha, beta=0.5, dtype=dtype)
    return tf.random.stateless_gamma(
        tf.stack([draws, tf.shape(alpha)[0]]),
        seed=pair,
        alpha=alpha,
        beta=0.5,
        dtype=dtype,
    )


def _log_i0(value: tf.Tensor) -> tf.Tensor:
    value = tf.convert_to_tensor(value)
    absolute = tf.abs(value)
    return tf.math.log(tf.math.bessel_i0e(absolute)) + absolute


def _masked_low_rank_terms(
    log_diagonal_scale: tf.Tensor,
    low_rank_factor: tf.Tensor,
    mask: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return inverse diagonal, weighted factor, Cholesky, and log determinant."""
    dtype = log_diagonal_scale.dtype
    # ROCm's batched float32 Cholesky can reject these analytically positive
    # rank-16 Woodbury systems during refinement. Evaluate only this small
    # factorization in symmetric float64 CPU arithmetic, then return tensors in
    # the model dtype. This preserves the represented covariance and density.
    work_dtype = tf.float64
    log_scale = tf.cast(log_diagonal_scale, work_dtype)
    mask_float = tf.cast(mask, work_dtype)
    variance = tf.exp(2.0 * log_scale) * mask_float + (1.0 - mask_float)
    inverse_variance = tf.math.reciprocal(variance)
    factor = tf.cast(low_rank_factor, work_dtype) * mask_float[..., None]
    weighted_factor = factor * inverse_variance[..., None]
    rank = tf.shape(factor)[-1]
    small = tf.eye(
        rank,
        batch_shape=[tf.shape(factor)[0]],
        dtype=work_dtype,
    ) + tf.einsum("bdr,bds->brs", factor, weighted_factor)
    small = 0.5 * (small + tf.linalg.matrix_transpose(small))
    with tf.device("/CPU:0"):
        chol = tf.linalg.cholesky(small)
    logdet = tf.reduce_sum(tf.math.log(variance) * mask_float, axis=-1)
    logdet += 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    return tuple(
        tf.cast(value, dtype)
        for value in (inverse_variance, weighted_factor, chol, logdet)
    )


def _masked_low_rank_quadratic(
    residual: tf.Tensor,
    *,
    inverse_variance: tf.Tensor,
    weighted_factor: tf.Tensor,
    chol: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """Evaluate r' Sigma^-1 r for residual shape (draw, batch, dimension)."""
    dtype = residual.dtype
    residual = residual * tf.cast(mask[None, ...], dtype)
    weighted = residual * inverse_variance[None, ...]
    projected = tf.einsum("bdr,kbd->kbr", weighted_factor, residual)
    solved = tf.linalg.cholesky_solve(chol[None, ...], projected[..., None])
    correction = tf.reduce_sum(projected * solved[..., 0], axis=-1)
    return tf.reduce_sum(residual * weighted, axis=-1) - correction


def _masked_row_solve(
    value: tf.Tensor,
    *,
    inverse_variance: tf.Tensor,
    weighted_factor: tf.Tensor,
    chol: tf.Tensor,
    row_mask: tf.Tensor,
) -> tf.Tensor:
    """Apply a masked row covariance inverse without materializing it."""
    value = value * tf.cast(row_mask[None, ..., None], value.dtype)
    weighted = value * inverse_variance[None, ..., None]
    projected = tf.einsum("brk,dbri->dbki", weighted_factor, value)
    solved = tf.linalg.cholesky_solve(chol[None, ...], projected)
    correction = tf.einsum("brk,dbki->dbri", weighted_factor, solved)
    return weighted - correction


@dataclass(frozen=True)
class MaskedLowRankStudentT:
    """Masked multivariate Student-t with diagonal plus low-rank scale."""

    mean: tf.Tensor
    log_diagonal_scale: tf.Tensor
    low_rank_factor: tf.Tensor
    degrees_of_freedom: tf.Tensor
    mask: tf.Tensor

    @property
    def diagonal_scale(self) -> tf.Tensor:
        return tf.exp(self.log_diagonal_scale)

    def sample(self, draws: int, seed: int | None = None) -> tf.Tensor:
        if draws <= 0:
            raise ValueError("draws must be positive")
        dtype = self.mean.dtype
        batch = tf.shape(self.mean)[0]
        dimension = tf.shape(self.mean)[1]
        rank = tf.shape(self.low_rank_factor)[2]
        diagonal_noise = _normal(
            [draws, batch, dimension],
            dtype=dtype,
            seed=seed,
            stream=11,
        )
        rank_noise = _normal(
            [draws, batch, rank],
            dtype=dtype,
            seed=seed,
            stream=12,
        )
        chi_square = _gamma(
            draws,
            alpha=self.degrees_of_freedom / 2.0,
            dtype=dtype,
            seed=seed,
            stream=13,
        )
        radial = tf.sqrt(
            self.degrees_of_freedom[None, :] / tf.maximum(chi_square, 1e-12)
        )
        gaussian = self.diagonal_scale[None, ...] * diagonal_noise + tf.einsum(
            "bdr,kbr->kbd", self.low_rank_factor, rank_noise
        )
        mask = tf.cast(self.mask[None, ...], dtype)
        return self.mean[None, ...] + gaussian * radial[..., None] * mask

    def log_prob(self, value: tf.Tensor) -> tf.Tensor:
        value = tf.cast(value, self.mean.dtype)
        if value.shape.rank == 2:
            value = value[None, ...]
        inverse, weighted_factor, chol, logdet = _masked_low_rank_terms(
            self.log_diagonal_scale,
            self.low_rank_factor,
            self.mask,
        )
        quadratic = _masked_low_rank_quadratic(
            value - self.mean[None, ...],
            inverse_variance=inverse,
            weighted_factor=weighted_factor,
            chol=chol,
            mask=self.mask,
        )
        dtype = self.mean.dtype
        dimension = tf.reduce_sum(tf.cast(self.mask, dtype), axis=-1)
        df = self.degrees_of_freedom
        normalizer = (
            tf.math.lgamma((df + dimension) / 2.0)
            - tf.math.lgamma(df / 2.0)
            - 0.5 * (dimension * tf.math.log(df * tf.cast(math.pi, dtype)) + logdet)
        )
        return normalizer[None, ...] - 0.5 * (
            df[None, ...] + dimension[None, ...]
        ) * tf.math.log1p(quadratic / df[None, ...])


@dataclass(frozen=True)
class OrbitMatrixNormal:
    """Matrix-Normal latent posterior averaged exactly over O(2)."""

    mean: tf.Tensor
    log_diagonal_scale: tf.Tensor
    low_rank_factor: tf.Tensor
    row_mask: tf.Tensor

    @property
    def diagonal_scale(self) -> tf.Tensor:
        return tf.exp(self.log_diagonal_scale)

    def sample(self, draws: int, seed: int | None = None) -> tf.Tensor:
        if draws <= 0:
            raise ValueError("draws must be positive")
        mean = self.mean
        if mean.shape.rank == 3:
            mean = mean[None, ...]
        dtype = mean.dtype
        batch = tf.shape(mean)[1]
        rows = tf.shape(mean)[2]
        rank = tf.shape(self.low_rank_factor)[2]
        diagonal_noise = _normal(
            [draws, batch, rows, 2],
            dtype=dtype,
            seed=seed,
            stream=21,
        )
        rank_noise = _normal(
            [draws, batch, rank, 2],
            dtype=dtype,
            seed=seed,
            stream=22,
        )
        base = mean + (
            self.diagonal_scale[None, ..., None] * diagonal_noise
            + tf.einsum("brk,dbki->dbri", self.low_rank_factor, rank_noise)
        )
        angle = (
            2.0
            * tf.cast(math.pi, dtype)
            * _uniform(
                [draws, batch],
                dtype=dtype,
                seed=seed,
                stream=23,
            )
        )
        reflection_draw = _uniform(
            [draws, batch],
            dtype=dtype,
            seed=seed,
            stream=24,
        )
        reflection = tf.where(
            reflection_draw < 0.5,
            -tf.ones_like(reflection_draw),
            tf.ones_like(reflection_draw),
        )
        cosine = tf.cos(angle)
        sine = tf.sin(angle)
        first = tf.stack([cosine, sine], axis=-1)
        second = tf.stack([-sine * reflection, cosine * reflection], axis=-1)
        orthogonal = tf.stack([first, second], axis=-1)
        rotated = tf.einsum("dbri,dbij->dbrj", base, orthogonal)
        return rotated * tf.cast(self.row_mask[None, ..., None], dtype)

    def base_log_prob(self, value: tf.Tensor) -> tf.Tensor:
        """Evaluate the unsymmetrized matrix-Normal base density."""
        value = tf.cast(value, self.mean.dtype)
        if value.shape.rank == 3:
            value = value[None, ...]
        mean = self.mean
        if mean.shape.rank == 3:
            mean = mean[None, ...]
        inverse, weighted_factor, chol, logdet = _masked_low_rank_terms(
            self.log_diagonal_scale,
            self.low_rank_factor,
            self.row_mask,
        )
        residual = value - mean
        solved = _masked_row_solve(
            residual,
            inverse_variance=inverse,
            weighted_factor=weighted_factor,
            chol=chol,
            row_mask=self.row_mask,
        )
        quadratic = tf.reduce_sum(residual * solved, axis=(-2, -1))
        active_rows = tf.reduce_sum(tf.cast(self.row_mask, value.dtype), axis=-1)
        return (
            -active_rows[None, ...] * tf.math.log(tf.cast(2.0 * math.pi, value.dtype))
            - logdet[None, ...]
            - 0.5 * quadratic
        )

    def log_prob(self, value: tf.Tensor) -> tf.Tensor:
        value = tf.cast(value, self.mean.dtype)
        if value.shape.rank == 3:
            value = value[None, ...]
        mean = self.mean
        if mean.shape.rank == 3:
            mean = mean[None, ...]
        inverse, weighted_factor, chol, logdet = _masked_low_rank_terms(
            self.log_diagonal_scale,
            self.low_rank_factor,
            self.row_mask,
        )
        solved_value = _masked_row_solve(
            value,
            inverse_variance=inverse,
            weighted_factor=weighted_factor,
            chol=chol,
            row_mask=self.row_mask,
        )
        solved_mean = _masked_row_solve(
            mean,
            inverse_variance=inverse,
            weighted_factor=weighted_factor,
            chol=chol,
            row_mask=self.row_mask,
        )
        value_quadratic = tf.reduce_sum(value * solved_value, axis=(-2, -1))
        mean_quadratic = tf.reduce_sum(mean * solved_mean, axis=(-2, -1))
        cross = tf.einsum("dbri,dbrj->dbij", value, solved_mean)
        r_plus = tf.sqrt(
            tf.maximum(
                tf.square(cross[..., 0, 0] + cross[..., 1, 1])
                + tf.square(cross[..., 1, 0] - cross[..., 0, 1]),
                tf.cast(0.0, value.dtype),
            )
        )
        r_minus = tf.sqrt(
            tf.maximum(
                tf.square(cross[..., 0, 0] - cross[..., 1, 1])
                + tf.square(cross[..., 1, 0] + cross[..., 0, 1]),
                tf.cast(0.0, value.dtype),
            )
        )
        orbit_log_integral = tf.reduce_logsumexp(
            tf.stack([_log_i0(r_plus), _log_i0(r_minus)], axis=-1),
            axis=-1,
        ) - tf.math.log(tf.cast(2.0, value.dtype))
        active_rows = tf.reduce_sum(tf.cast(self.row_mask, value.dtype), axis=-1)
        normalizer = (
            -active_rows * tf.math.log(tf.cast(2.0 * math.pi, value.dtype)) - logdet
        )
        return (
            normalizer[None, ...]
            - 0.5 * (value_quadratic + mean_quadratic)
            + orbit_log_integral
        )


class ThetaConditionedLatentMean(tf.keras.layers.Layer):
    """Row-local FiLM mean conditioned on invariant global summaries."""

    def __init__(self, width: int = GENERATIVE_IID_V2_HIDDEN_WIDTH, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)
        self.summary_projection = tf.keras.layers.Dense(2 * self.width)
        self.normalization = tf.keras.layers.LayerNormalization()
        self.output_projection = tf.keras.layers.Dense(2)

    def call(self, row_features: tf.Tensor, summary: tf.Tensor) -> tf.Tensor:
        summary = tf.convert_to_tensor(summary)
        row_features = tf.convert_to_tensor(row_features)
        if summary.shape.rank == 2:
            summary = summary[None, ...]
        film = self.summary_projection(summary)
        gamma, shift = tf.split(film, 2, axis=-1)
        rows = row_features[None, ...]
        conditioned = (
            rows * (1.0 + 0.1 * tf.tanh(gamma[..., None, :])) + shift[..., None, :]
        )
        return self.output_projection(self.normalization(conditioned))

    def get_config(self) -> dict[str, int]:
        return {"width": self.width}


@dataclass(frozen=True)
class JointOrbitPosterior:
    """Conditional Student-t/orbit posterior represented in the v1 raw state."""

    global_posterior: MaskedLowRankStudentT
    latent_row_features: tf.Tensor
    latent_mean_offset: tf.Tensor
    latent_log_diagonal_scale: tf.Tensor
    latent_low_rank_factor: tf.Tensor
    latent_mean_conditioner: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    row_mask: tf.Tensor
    layout: JointStateLayout
    site_mask: tf.Tensor
    species_mask: tf.Tensor
    refinement_trace: tuple[dict[str, tf.Tensor], ...] = ()

    @property
    def state_mask(self) -> tf.Tensor:
        return _raw_state_mask(self.layout, self.site_mask, self.species_mask)

    def _global_summary(self, theta: tf.Tensor) -> tf.Tensor:
        if theta.shape.rank == 2:
            theta = theta[None, ...]
        species = tf.cast(self.species_mask[None, ...], theta.dtype)
        beta_start = 1
        beta_stop = beta_start + 2 * self.layout.max_species
        beta = tf.reshape(
            theta[..., beta_start:beta_stop],
            [
                tf.shape(theta)[0],
                tf.shape(theta)[1],
                2,
                self.layout.max_species,
            ],
        )
        count = tf.reduce_sum(species, axis=-1)
        beta_mean = tf.math.divide_no_nan(
            tf.reduce_sum(beta * species[..., None, :], axis=-1),
            count[..., None],
        )
        beta_second = tf.math.divide_no_nan(
            tf.reduce_sum(tf.square(beta) * species[..., None, :], axis=-1),
            count[..., None],
        )
        return tf.concat(
            [
                theta[..., 0:1],
                theta[..., -1:],
                beta_mean,
                beta_second,
            ],
            axis=-1,
        )

    def conditional_latent_mean(self, theta: tf.Tensor) -> tf.Tensor:
        mean = self.latent_mean_conditioner(
            self.latent_row_features, self._global_summary(theta)
        )
        return (mean + self.latent_mean_offset[None, ...]) * tf.cast(
            self.row_mask[None, ..., None], mean.dtype
        )

    def sample(self, draws: int, seed: int | None = None) -> tf.Tensor:
        theta = self.global_posterior.sample(draws, seed=seed)
        latent = OrbitMatrixNormal(
            mean=self.conditional_latent_mean(theta),
            log_diagonal_scale=self.latent_log_diagonal_scale,
            low_rank_factor=self.latent_low_rank_factor,
            row_mask=self.row_mask,
        ).sample(draws, seed=None if seed is None else seed + 1000)
        return _assemble_raw_state(theta, latent, self.layout)

    def log_prob(self, value: tf.Tensor) -> tf.Tensor:
        value = tf.convert_to_tensor(value)
        if value.shape.rank == 2:
            value = value[None, ...]
        theta, latent = _split_raw_state(value, self.layout)
        global_log_prob = self.global_posterior.log_prob(theta)
        latent_log_prob = OrbitMatrixNormal(
            mean=self.conditional_latent_mean(theta),
            log_diagonal_scale=self.latent_log_diagonal_scale,
            low_rank_factor=self.latent_low_rank_factor,
            row_mask=self.row_mask,
        ).log_prob(latent)
        return global_log_prob + latent_log_prob

    def with_parameters(
        self,
        *,
        global_mean: tf.Tensor | None = None,
        global_log_diagonal_scale: tf.Tensor | None = None,
        latent_mean_offset: tf.Tensor | None = None,
        latent_log_diagonal_scale: tf.Tensor | None = None,
        refinement_trace: tuple[dict[str, tf.Tensor], ...] | None = None,
    ) -> "JointOrbitPosterior":
        global_posterior = replace(
            self.global_posterior,
            mean=(self.global_posterior.mean if global_mean is None else global_mean),
            log_diagonal_scale=(
                self.global_posterior.log_diagonal_scale
                if global_log_diagonal_scale is None
                else global_log_diagonal_scale
            ),
        )
        return replace(
            self,
            global_posterior=global_posterior,
            latent_mean_offset=(
                self.latent_mean_offset
                if latent_mean_offset is None
                else latent_mean_offset
            ),
            latent_log_diagonal_scale=(
                self.latent_log_diagonal_scale
                if latent_log_diagonal_scale is None
                else latent_log_diagonal_scale
            ),
            refinement_trace=(
                self.refinement_trace if refinement_trace is None else refinement_trace
            ),
        )

    def invariant_moments(self) -> dict[str, tf.Tensor]:
        """Analytic invariant moments at the global posterior location."""
        theta = self.global_posterior.mean
        mean = self.conditional_latent_mean(theta)[0]
        site_mean = mean[:, : self.layout.max_sites]
        species_mean = mean[:, self.layout.max_sites :]
        factor = self.latent_low_rank_factor
        site_factor = factor[:, : self.layout.max_sites]
        species_factor = factor[:, self.layout.max_sites :]
        random_effect = tf.einsum(
            "bni,bsi->bns", site_mean, species_mean
        ) + 2.0 * tf.einsum("bnr,bsr->bns", site_factor, species_factor)
        association = tf.einsum(
            "bsi,bti->bst", species_mean, species_mean
        ) + 2.0 * tf.einsum("bsr,btr->bst", species_factor, species_factor)
        association += tf.linalg.diag(
            2.0
            * tf.exp(2.0 * self.latent_log_diagonal_scale[:, self.layout.max_sites :])
        )
        beta = tf.reshape(
            theta[:, 1 : 1 + 2 * self.layout.max_species],
            [-1, 2, self.layout.max_species],
        )
        site = tf.cast(self.site_mask[..., None], random_effect.dtype)
        species = tf.cast(self.species_mask, random_effect.dtype)
        return {
            "alpha": theta[:, 0],
            "Beta": beta * species[:, None, :],
            "log_tau": theta[:, -1],
            "R": random_effect * site * species[:, None, :],
            "C": association * species[:, :, None] * species[:, None, :],
        }


class MaskedBipartiteAttentionBlock(tf.keras.layers.Layer):
    """Edge-aware site/species cross-attention without entity identifiers."""

    def __init__(
        self,
        *,
        width: int = GENERATIVE_IID_V2_HIDDEN_WIDTH,
        heads: int = GENERATIVE_IID_V2_ATTENTION_HEADS,
        feedforward_width: int = GENERATIVE_IID_V2_FEEDFORWARD_WIDTH,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if width % heads:
            raise ValueError("attention width must be divisible by heads")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.feedforward_width = int(feedforward_width)
        self.site_norm = tf.keras.layers.LayerNormalization()
        self.species_norm = tf.keras.layers.LayerNormalization()
        self.site_query = tf.keras.layers.Dense(width, use_bias=False)
        self.site_key = tf.keras.layers.Dense(width, use_bias=False)
        self.site_value = tf.keras.layers.Dense(width, use_bias=False)
        self.species_query = tf.keras.layers.Dense(width, use_bias=False)
        self.species_key = tf.keras.layers.Dense(width, use_bias=False)
        self.species_value = tf.keras.layers.Dense(width, use_bias=False)
        self.site_edge_bias = tf.keras.layers.Dense(heads, use_bias=False)
        self.species_edge_bias = tf.keras.layers.Dense(heads, use_bias=False)
        self.site_edge_value = tf.keras.layers.Dense(width, use_bias=False)
        self.species_edge_value = tf.keras.layers.Dense(width, use_bias=False)
        self.site_output = tf.keras.layers.Dense(width)
        self.species_output = tf.keras.layers.Dense(width)
        self.site_ff_norm = tf.keras.layers.LayerNormalization()
        self.species_ff_norm = tf.keras.layers.LayerNormalization()
        self.site_ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(feedforward_width, activation="gelu"),
                tf.keras.layers.Dense(width),
            ]
        )
        self.species_ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(feedforward_width, activation="gelu"),
                tf.keras.layers.Dense(width),
            ]
        )

    def _heads(self, value: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(value)
        return tf.reshape(
            value,
            tf.concat([shape[:-1], [self.heads, self.head_width]], axis=0),
        )

    def call(
        self,
        site_state: tf.Tensor,
        species_state: tf.Tensor,
        edge_features: tf.Tensor,
        pair_mask: tf.Tensor,
        site_mask: tf.Tensor,
        species_mask: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        site_normalized = self.site_norm(site_state)
        species_normalized = self.species_norm(species_state)
        site_query = self._heads(self.site_query(site_normalized))
        species_key = self._heads(self.site_key(species_normalized))
        species_value = self._heads(self.site_value(species_normalized))
        site_logits = tf.einsum("bnhd,bshd->bnhs", site_query, species_key) / math.sqrt(
            self.head_width
        )
        site_logits += tf.transpose(self.site_edge_bias(edge_features), [0, 1, 3, 2])
        site_attention = _masked_softmax64(
            site_logits, pair_mask[:, :, None, :], axis=-1
        )
        site_edge_value = self._heads(self.site_edge_value(edge_features))
        site_values = species_value[:, None, ...] + site_edge_value
        site_message = tf.einsum("bnhs,bnshd->bnhd", site_attention, site_values)
        site_message = tf.reshape(
            site_message,
            [tf.shape(site_state)[0], tf.shape(site_state)[1], self.width],
        )

        species_query = self._heads(self.species_query(species_normalized))
        site_key = self._heads(self.species_key(site_normalized))
        site_value = self._heads(self.species_value(site_normalized))
        species_logits = tf.einsum(
            "bshd,bnhd->bshn", species_query, site_key
        ) / math.sqrt(self.head_width)
        species_logits += tf.transpose(
            self.species_edge_bias(edge_features), [0, 2, 3, 1]
        )
        species_attention = _masked_softmax64(
            species_logits,
            tf.transpose(pair_mask, [0, 2, 1])[:, :, None, :],
            axis=-1,
        )
        species_edge_value = self._heads(self.species_edge_value(edge_features))
        species_values = site_value[:, :, None, ...] + species_edge_value
        species_message = tf.einsum(
            "bshn,bnshd->bshd", species_attention, species_values
        )
        species_message = tf.reshape(
            species_message,
            [
                tf.shape(species_state)[0],
                tf.shape(species_state)[1],
                self.width,
            ],
        )

        site_state = site_state + self.site_output(site_message, training=training)
        species_state = species_state + self.species_output(
            species_message, training=training
        )
        site_state = site_state + self.site_ff(
            self.site_ff_norm(site_state), training=training
        )
        species_state = species_state + self.species_ff(
            self.species_ff_norm(species_state), training=training
        )
        site_state *= tf.cast(site_mask[..., None], site_state.dtype)
        species_state *= tf.cast(species_mask[..., None], species_state.dtype)
        return site_state, species_state

    def get_config(self) -> dict[str, int]:
        return {
            "width": self.width,
            "heads": self.heads,
            "feedforward_width": self.feedforward_width,
        }


class GenerativeIidOrbitPosteriorModel(tf.keras.Model):
    """Frozen four-block encoder for the v2 joint orbit posterior."""

    def __init__(
        self,
        *,
        max_sites: int = GENERATIVE_IID_MAX_SITES,
        max_species: int = GENERATIVE_IID_MAX_SPECIES,
        hidden_width: int = GENERATIVE_IID_V2_HIDDEN_WIDTH,
        attention_heads: int = GENERATIVE_IID_V2_ATTENTION_HEADS,
        attention_blocks: int = GENERATIVE_IID_V2_ATTENTION_BLOCKS,
        feedforward_width: int = GENERATIVE_IID_V2_FEEDFORWARD_WIDTH,
        posterior_rank: int = GENERATIVE_IID_V2_POSTERIOR_RANK,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        frozen = (96, 4, 4, 192, 16)
        observed = (
            hidden_width,
            attention_heads,
            attention_blocks,
            feedforward_width,
            posterior_rank,
        )
        if observed != frozen:
            raise ValueError(
                "v2 requires width 96, four heads, four blocks, "
                "feed-forward width 192, and posterior rank 16"
            )
        _validate_model_bounds(max_sites, max_species)
        self.max_sites = int(max_sites)
        self.max_species = int(max_species)
        self.hidden_width = int(hidden_width)
        self.attention_heads = int(attention_heads)
        self.attention_blocks = int(attention_blocks)
        self.feedforward_width = int(feedforward_width)
        self.posterior_rank = int(posterior_rank)
        self.layout = JointStateLayout(self.max_sites, self.max_species)
        self.site_initial = tf.keras.layers.Dense(hidden_width, activation="gelu")
        self.species_initial = tf.keras.layers.Dense(hidden_width, activation="gelu")
        self.blocks = [
            MaskedBipartiteAttentionBlock(
                width=hidden_width,
                heads=attention_heads,
                feedforward_width=feedforward_width,
                name=f"bipartite_attention_{index}",
            )
            for index in range(attention_blocks)
        ]
        self.beta_head = _head(2 + posterior_rank, hidden_width, "beta_head")
        self.global_head = _head(
            2 * (2 + posterior_rank) + 1,
            hidden_width,
            "global_head",
        )
        self.site_covariance_head = _head(
            1 + posterior_rank, hidden_width, "site_covariance_head"
        )
        self.species_covariance_head = _head(
            1 + posterior_rank, hidden_width, "species_covariance_head"
        )
        self.latent_mean_conditioner = ThetaConditionedLatentMean(
            hidden_width, name="theta_conditioned_latent_mean"
        )

    def build(self, input_shape) -> None:
        if self.built:
            return
        X = tf.zeros([1, self.max_sites, 2], dtype=tf.float32)
        X = tf.tensor_scatter_nd_update(
            X,
            [[0, index, 0] for index in range(GENERATIVE_IID_MIN_SITES)],
            tf.ones([GENERATIVE_IID_MIN_SITES], dtype=tf.float32),
        )
        site_mask = tf.sequence_mask([GENERATIVE_IID_MIN_SITES], self.max_sites)
        species_mask = tf.sequence_mask([GENERATIVE_IID_MIN_SPECIES], self.max_species)
        self.call(
            {
                "X": X,
                "Y": tf.zeros([1, self.max_sites, self.max_species], tf.float32),
                "response_mask": (site_mask[:, :, None] & species_mask[:, None, :]),
                "site_mask": site_mask,
                "species_mask": species_mask,
            },
            training=False,
            refine=False,
        )
        super().build(input_shape)

    def get_config(self) -> dict[str, object]:
        return {
            "max_sites": self.max_sites,
            "max_species": self.max_species,
            "hidden_width": self.hidden_width,
            "attention_heads": self.attention_heads,
            "attention_blocks": self.attention_blocks,
            "feedforward_width": self.feedforward_width,
            "posterior_rank": self.posterior_rank,
        }

    def call(
        self,
        inputs: dict[str, tf.Tensor],
        training: bool = False,
        *,
        refine: bool = True,
        refinement_seed: int | None = None,
    ) -> JointOrbitPosterior:
        X, Y, response_mask, site_mask, species_mask = _validated_inputs(
            inputs, self.max_sites, self.max_species
        )
        dtype = X.dtype
        pair_mask = response_mask & site_mask[:, :, None] & species_mask[:, None, :]
        pair_float = tf.cast(pair_mask, dtype)
        site_count = tf.reduce_sum(pair_float, axis=2, keepdims=True)
        species_count = tf.reduce_sum(pair_float, axis=1)
        site_prevalence = tf.math.divide_no_nan(
            tf.reduce_sum(Y * pair_float, axis=2, keepdims=True),
            site_count,
        )
        species_prevalence = tf.math.divide_no_nan(
            tf.reduce_sum(Y * pair_float, axis=1), species_count
        )
        site_fraction = tf.math.divide_no_nan(
            site_count,
            tf.reduce_sum(tf.cast(species_mask, dtype), axis=1, keepdims=True)[
                :, None, :
            ],
        )
        species_fraction = tf.math.divide_no_nan(
            species_count,
            tf.reduce_sum(tf.cast(site_mask, dtype), axis=1, keepdims=True),
        )
        site_state = self.site_initial(
            tf.concat([X, site_fraction, site_prevalence], axis=-1),
            training=training,
        )
        species_state = self.species_initial(
            tf.stack([species_prevalence, species_fraction], axis=-1),
            training=training,
        )
        edge_features = tf.concat(
            [
                Y[..., None] * pair_float[..., None],
                pair_float[..., None],
                tf.broadcast_to(
                    X[:, :, None, :],
                    [
                        tf.shape(X)[0],
                        self.max_sites,
                        self.max_species,
                        GENERATIVE_IID_N_COVARIATES,
                    ],
                ),
                tf.broadcast_to(
                    site_fraction[:, :, None, :],
                    [tf.shape(X)[0], self.max_sites, self.max_species, 1],
                ),
                tf.broadcast_to(
                    species_fraction[:, None, :, None],
                    [tf.shape(X)[0], self.max_sites, self.max_species, 1],
                ),
            ],
            axis=-1,
        )
        for block in self.blocks:
            site_state, species_state = block(
                site_state,
                species_state,
                edge_features,
                pair_mask,
                site_mask,
                species_mask,
                training=training,
            )

        community = tf.concat(
            [
                _masked_mean64(site_state, site_mask, axis=1),
                _masked_max(site_state, site_mask, axis=1),
                _masked_mean64(species_state, species_mask, axis=1),
                _masked_max(species_state, species_mask, axis=1),
            ],
            axis=-1,
        )
        design_mean = _masked_mean64(X, site_mask, axis=1)
        design_variance = _masked_mean64(
            tf.square(X - design_mean[:, None, :]), site_mask, axis=1
        )
        design_scale = tf.sqrt(tf.maximum(design_variance, 1e-8))
        coefficient_type = tf.constant([1.0, 0.0], dtype=dtype)
        design_local = tf.stack(
            [
                design_mean,
                design_scale,
                tf.broadcast_to(coefficient_type[None, :], tf.shape(design_mean)),
            ],
            axis=-1,
        )
        beta_tokens = tf.concat(
            [
                tf.broadcast_to(
                    species_state[:, None, :, :],
                    [
                        tf.shape(X)[0],
                        2,
                        self.max_species,
                        self.hidden_width,
                    ],
                ),
                tf.broadcast_to(
                    design_local[:, :, None, :],
                    [tf.shape(X)[0], 2, self.max_species, 3],
                ),
                tf.broadcast_to(
                    community[:, None, None, :],
                    [
                        tf.shape(X)[0],
                        2,
                        self.max_species,
                        4 * self.hidden_width,
                    ],
                ),
            ],
            axis=-1,
        )
        beta_raw = self.beta_head(beta_tokens, training=training)
        global_raw = self.global_head(community, training=training)
        global_parameter_raw = tf.reshape(
            global_raw[:, :-1],
            [tf.shape(X)[0], 2, 2 + self.posterior_rank],
        )
        beta_flat = tf.reshape(
            beta_raw,
            [
                tf.shape(X)[0],
                2 * self.max_species,
                2 + self.posterior_rank,
            ],
        )
        ordered = tf.concat(
            [
                global_parameter_raw[:, 0:1],
                beta_flat,
                global_parameter_raw[:, 1:2],
            ],
            axis=1,
        )
        global_mask = _global_state_mask(self.layout, species_mask)
        global_float = tf.cast(global_mask, dtype)
        global_mean = ordered[..., 0] * global_float
        global_log_scale = (
            tf.math.log(tf.nn.softplus(ordered[..., 1]) + 1e-4) * global_float
        )
        global_factor = ordered[..., 2:] * global_float[..., None]
        degrees_of_freedom = 4.0 + 26.0 * tf.sigmoid(global_raw[:, -1])
        global_posterior = MaskedLowRankStudentT(
            mean=global_mean,
            log_diagonal_scale=global_log_scale,
            low_rank_factor=global_factor,
            degrees_of_freedom=degrees_of_freedom,
            mask=global_mask,
        )

        site_covariance = self.site_covariance_head(site_state, training=training)
        species_covariance = self.species_covariance_head(
            species_state, training=training
        )
        covariance = tf.concat([site_covariance, species_covariance], axis=1)
        row_mask = tf.concat([site_mask, species_mask], axis=1)
        row_float = tf.cast(row_mask, dtype)
        latent_log_scale = (
            tf.math.log(tf.nn.softplus(covariance[..., 0]) + 1e-4) * row_float
        )
        latent_factor = covariance[..., 1:] * row_float[..., None]
        row_features = tf.concat([site_state, species_state], axis=1)
        posterior = JointOrbitPosterior(
            global_posterior=global_posterior,
            latent_row_features=row_features,
            latent_mean_offset=tf.zeros(
                [tf.shape(X)[0], self.max_sites + self.max_species, 2],
                dtype=dtype,
            ),
            latent_log_diagonal_scale=latent_log_scale,
            latent_low_rank_factor=latent_factor,
            latent_mean_conditioner=self.latent_mean_conditioner,
            row_mask=row_mask,
            layout=self.layout,
            site_mask=site_mask,
            species_mask=species_mask,
        )
        # The conditional block is part of every checkpoint, even when the
        # caller requests encoder-only output without refinement.
        posterior.conditional_latent_mean(global_posterior.mean)
        if not refine:
            return posterior
        return refine_joint_orbit_posterior(
            posterior,
            inputs,
            seed=refinement_seed,
        )


def joint_orbit_iwelbo_by_batch(
    posterior: JointOrbitPosterior,
    inputs: dict[str, tf.Tensor | np.ndarray],
    *,
    draws: int = GENERATIVE_IID_V2_REFINEMENT_DRAWS,
    kl_weight: float = 1.0,
    seed: int | None = None,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Evaluate the unchanged generative IWAE objective."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    if not 0.0 < kl_weight <= 1.0:
        raise ValueError("kl_weight must be in (0, 1]")
    samples = posterior.sample(draws, seed=seed)
    log_likelihood = probit_log_likelihood(
        samples,
        layout=posterior.layout,
        X=tf.cast(inputs["X"], tf.float32),
        Y=tf.cast(inputs["Y"], tf.float32),
        response_mask=tf.cast(inputs["response_mask"], tf.bool),
        site_mask=posterior.site_mask,
        species_mask=posterior.species_mask,
    )
    log_prior = generative_log_prior(
        samples,
        layout=posterior.layout,
        site_mask=posterior.site_mask,
        species_mask=posterior.species_mask,
    )
    log_q = posterior.log_prob(samples)
    weights = log_likelihood + float(kl_weight) * (log_prior - log_q)
    iwelbo = tf.reduce_logsumexp(weights, axis=0) - tf.math.log(
        tf.cast(draws, weights.dtype)
    )
    return iwelbo, {
        "iwelbo": tf.reduce_mean(iwelbo),
        "log_likelihood": tf.reduce_mean(log_likelihood),
        "log_prior": tf.reduce_mean(log_prior),
        "log_q": tf.reduce_mean(log_q),
    }


def generative_iid_v2_log_joint(
    state: tf.Tensor,
    inputs: dict[str, tf.Tensor | np.ndarray],
    *,
    layout: JointStateLayout,
    site_mask: tf.Tensor,
    species_mask: tf.Tensor,
) -> tf.Tensor:
    """Expose the unchanged v1 target used by the v2 variational family."""
    return generative_log_prior(
        state,
        layout=layout,
        site_mask=site_mask,
        species_mask=species_mask,
    ) + probit_log_likelihood(
        state,
        layout=layout,
        X=tf.cast(inputs["X"], state.dtype),
        Y=tf.cast(inputs["Y"], state.dtype),
        response_mask=tf.cast(inputs["response_mask"], tf.bool),
        site_mask=site_mask,
        species_mask=species_mask,
    )


def importance_weighted_orbit_loss(
    posterior: JointOrbitPosterior,
    inputs: dict[str, tf.Tensor | np.ndarray],
    *,
    draws: int = GENERATIVE_IID_V2_REFINEMENT_DRAWS,
    kl_weight: float = 1.0,
    seed: int | None = None,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    iwelbo, diagnostics = joint_orbit_iwelbo_by_batch(
        posterior,
        inputs,
        draws=draws,
        kl_weight=kl_weight,
        seed=seed,
    )
    return -tf.reduce_mean(iwelbo), diagnostics


def refine_joint_orbit_posterior(
    posterior: JointOrbitPosterior,
    inputs: dict[str, tf.Tensor | np.ndarray],
    *,
    seed: int | None = None,
    kl_weight: float = 1.0,
) -> JointOrbitPosterior:
    """Apply the four frozen first-order common-random IWAE refinement steps."""
    current = posterior
    trace: list[dict[str, tf.Tensor]] = []
    names = (
        "global_mean",
        "global_log_diagonal_scale",
        "latent_mean_offset",
        "latent_log_diagonal_scale",
    )
    for step_index, step_size in enumerate(GENERATIVE_IID_V2_REFINEMENT_STEPS):
        parameters = [
            current.global_posterior.mean,
            current.global_posterior.log_diagonal_scale,
            current.latent_mean_offset,
            current.latent_log_diagonal_scale,
        ]
        objective_seed = None if seed is None else int(seed) + step_index * 100
        with tf.GradientTape() as tape:
            for parameter in parameters:
                tape.watch(parameter)
            current_iwelbo, _ = joint_orbit_iwelbo_by_batch(
                current,
                inputs,
                draws=GENERATIVE_IID_V2_REFINEMENT_DRAWS,
                kl_weight=kl_weight,
                seed=objective_seed,
            )
            objective = tf.reduce_mean(current_iwelbo)
        gradients = tape.gradient(objective, parameters)
        baseline_scores: list[tf.Tensor] = []
        accepted_scores: list[tf.Tensor] = []
        accepted_flags: list[tf.Tensor] = []
        for block_index, (name, parameter, gradient) in enumerate(
            zip(names, parameters, gradients)
        ):
            if gradient is None:
                raise RuntimeError(f"missing refinement gradient for {name}")
            gradient = tf.stop_gradient(gradient)
            rms = tf.sqrt(tf.reduce_mean(tf.square(gradient)) + 1e-12)
            gradient = tf.clip_by_norm(gradient / rms, 1.0)
            baseline_score, _ = joint_orbit_iwelbo_by_batch(
                current,
                inputs,
                draws=GENERATIVE_IID_V2_REFINEMENT_DRAWS,
                kl_weight=kl_weight,
                seed=objective_seed,
            )
            baseline = tf.reduce_mean(baseline_score)
            accepted = tf.constant(False)
            selected_score = baseline
            selected_parameter = parameter
            for backtrack in range(4):
                proposal = (
                    parameter + (float(step_size) / float(2**backtrack)) * gradient
                )
                keyword = {name: proposal}
                proposal_posterior = current.with_parameters(**keyword)
                proposal_score, _ = joint_orbit_iwelbo_by_batch(
                    proposal_posterior,
                    inputs,
                    draws=GENERATIVE_IID_V2_REFINEMENT_DRAWS,
                    kl_weight=kl_weight,
                    seed=objective_seed,
                )
                proposal_mean = tf.reduce_mean(proposal_score)
                qualifies = tf.logical_and(
                    tf.logical_not(accepted),
                    proposal_mean >= baseline,
                )
                selected_parameter = tf.where(qualifies, proposal, selected_parameter)
                selected_score = tf.where(qualifies, proposal_mean, selected_score)
                accepted = tf.logical_or(accepted, qualifies)
            current = current.with_parameters(**{name: selected_parameter})
            parameters[block_index] = (
                current.global_posterior.mean
                if name == "global_mean"
                else (
                    current.global_posterior.log_diagonal_scale
                    if name == "global_log_diagonal_scale"
                    else (
                        current.latent_mean_offset
                        if name == "latent_mean_offset"
                        else current.latent_log_diagonal_scale
                    )
                )
            )
            baseline_scores.append(baseline)
            accepted_scores.append(selected_score)
            accepted_flags.append(accepted)
        trace.append(
            {
                "step": tf.cast(step_index, tf.int32),
                "step_size": tf.cast(step_size, tf.float32),
                "accepted": tf.stack(accepted_flags),
                "gradient_iwelbo": objective,
                "baseline_iwelbo": tf.stack(baseline_scores),
                "accepted_iwelbo": tf.stack(accepted_scores),
            }
        )
    return current.with_parameters(refinement_trace=tuple(trace))


def train_generative_iid_orbit_model(
    model: GenerativeIidOrbitPosteriorModel,
    batch: GenerativeIidBatch,
    *,
    epochs: int = 200,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    final_learning_rate: float = 3e-5,
    weight_decay: float = 1e-5,
    gradient_clip_norm: float = 5.0,
    model_seed: int = 511900001,
    importance_draws: int = GENERATIVE_IID_V2_REFINEMENT_DRAWS,
) -> GenerativeTrainingHistory:
    """Train the v2 candidate with the inherited outer schedule."""
    if importance_draws != GENERATIVE_IID_V2_REFINEMENT_DRAWS:
        raise ValueError("v2 production objective requires eight IWAE draws")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    tf.keras.utils.set_random_seed(int(model_seed))
    steps_per_epoch = math.ceil(len(batch.metadata) / batch_size)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=max(epochs * steps_per_epoch, 1),
        alpha=final_learning_rate / learning_rate,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=schedule, weight_decay=weight_decay
    )
    rng = np.random.default_rng(model_seed)
    losses: list[float] = []
    iwelbos: list[float] = []
    gradient_norms: list[float] = []
    for epoch in range(epochs):
        order = rng.permutation(len(batch.metadata))
        epoch_loss: list[float] = []
        epoch_iwelbo: list[float] = []
        epoch_norm: list[float] = []
        kl_weight = min(1.0, 0.25 + 0.75 * epoch / 19.0)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            inputs = {
                name: value[indices] for name, value in batch.model_inputs().items()
            }
            with tf.GradientTape() as tape:
                posterior = model(
                    inputs,
                    training=True,
                    refine=True,
                )
                loss, diagnostics = importance_weighted_orbit_loss(
                    posterior,
                    inputs,
                    draws=importance_draws,
                    kl_weight=kl_weight,
                )
            gradients = tape.gradient(loss, model.trainable_variables)
            pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, model.trainable_variables)
                if gradient is not None
            ]
            if not pairs or not all(
                bool(tf.reduce_all(tf.math.is_finite(gradient)))
                for gradient, _ in pairs
            ):
                raise FloatingPointError("non-finite v2 gradient")
            clipped, norm = tf.clip_by_global_norm(
                [gradient for gradient, _ in pairs], gradient_clip_norm
            )
            optimizer.apply_gradients(zip(clipped, [variable for _, variable in pairs]))
            if not bool(tf.math.is_finite(loss)):
                raise FloatingPointError("non-finite v2 loss")
            epoch_loss.append(float(loss))
            epoch_iwelbo.append(float(diagnostics["iwelbo"]))
            epoch_norm.append(float(norm))
        losses.append(float(np.mean(epoch_loss)))
        iwelbos.append(float(np.mean(epoch_iwelbo)))
        gradient_norms.append(float(np.mean(epoch_norm)))
    return GenerativeTrainingHistory(losses, iwelbos, gradient_norms)


def _global_state_mask(layout: JointStateLayout, species_mask: tf.Tensor) -> tf.Tensor:
    species_mask = tf.cast(species_mask, tf.bool)
    batch = tf.shape(species_mask)[0]
    return tf.concat(
        [
            tf.ones([batch, 1], dtype=tf.bool),
            tf.reshape(
                tf.broadcast_to(
                    species_mask[:, None, :],
                    [batch, 2, layout.max_species],
                ),
                [batch, 2 * layout.max_species],
            ),
            tf.ones([batch, 1], dtype=tf.bool),
        ],
        axis=1,
    )


def _raw_state_mask(
    layout: JointStateLayout,
    site_mask: tf.Tensor,
    species_mask: tf.Tensor,
) -> tf.Tensor:
    batch = tf.shape(site_mask)[0]
    return tf.concat(
        [
            tf.ones([batch, 1], dtype=tf.bool),
            tf.reshape(
                tf.broadcast_to(
                    species_mask[:, None, :],
                    [batch, 2, layout.max_species],
                ),
                [batch, 2 * layout.max_species],
            ),
            tf.reshape(
                tf.broadcast_to(
                    site_mask[:, :, None],
                    [batch, layout.max_sites, 2],
                ),
                [batch, 2 * layout.max_sites],
            ),
            tf.reshape(
                tf.broadcast_to(
                    species_mask[:, None, :],
                    [batch, 2, layout.max_species],
                ),
                [batch, 2 * layout.max_species],
            ),
            tf.ones([batch, 1], dtype=tf.bool),
        ],
        axis=1,
    )


def _assemble_raw_state(
    theta: tf.Tensor,
    latent: tf.Tensor,
    layout: JointStateLayout,
) -> tf.Tensor:
    beta_stop = 1 + 2 * layout.max_species
    eta = latent[..., : layout.max_sites, :]
    species_rows = latent[..., layout.max_sites :, :]
    loadings = tf.linalg.matrix_transpose(species_rows)
    return tf.concat(
        [
            theta[..., 0:1],
            theta[..., 1:beta_stop],
            tf.reshape(
                eta,
                [
                    tf.shape(theta)[0],
                    tf.shape(theta)[1],
                    2 * layout.max_sites,
                ],
            ),
            tf.reshape(
                loadings,
                [
                    tf.shape(theta)[0],
                    tf.shape(theta)[1],
                    2 * layout.max_species,
                ],
            ),
            theta[..., -1:],
        ],
        axis=-1,
    )


def _split_raw_state(
    state: tf.Tensor, layout: JointStateLayout
) -> tuple[tf.Tensor, tf.Tensor]:
    parameters = layout.unpack(state)
    theta = tf.concat(
        [
            parameters["alpha"][..., None],
            tf.reshape(
                parameters["Beta"],
                [
                    tf.shape(state)[0],
                    tf.shape(state)[1],
                    2 * layout.max_species,
                ],
            ),
            parameters["log_tau"][..., None],
        ],
        axis=-1,
    )
    latent = tf.concat(
        [
            parameters["Eta"],
            tf.linalg.matrix_transpose(parameters["Lambda"]),
        ],
        axis=-2,
    )
    return theta, latent


def _masked_softmax64(logits: tf.Tensor, mask: tf.Tensor, *, axis: int) -> tf.Tensor:
    logits64 = tf.cast(logits, tf.float64)
    mask = tf.cast(mask, tf.bool)
    masked = tf.where(mask, logits64, tf.constant(-1e300, tf.float64))
    maximum = tf.reduce_max(masked, axis=axis, keepdims=True)
    exponent = tf.where(mask, tf.exp(masked - maximum), 0.0)
    normalized = tf.math.divide_no_nan(
        exponent, tf.reduce_sum(exponent, axis=axis, keepdims=True)
    )
    return tf.cast(normalized, logits.dtype)


def _masked_mean64(values: tf.Tensor, mask: tf.Tensor, *, axis: int) -> tf.Tensor:
    dtype = values.dtype
    values64 = tf.cast(values, tf.float64)
    mask64 = tf.cast(mask, tf.float64)
    while mask64.shape.rank < values64.shape.rank:
        mask64 = mask64[..., None]
    result = tf.math.divide_no_nan(
        tf.reduce_sum(values64 * mask64, axis=axis),
        tf.reduce_sum(mask64, axis=axis),
    )
    return tf.cast(result, dtype)


def _masked_max(values: tf.Tensor, mask: tf.Tensor, *, axis: int) -> tf.Tensor:
    mask = tf.cast(mask, tf.bool)
    while mask.shape.rank < values.shape.rank:
        mask = mask[..., None]
    masked = tf.where(mask, values, tf.cast(-1e9, values.dtype))
    maximum = tf.reduce_max(masked, axis=axis)
    any_valid = tf.reduce_any(mask, axis=axis)
    return tf.where(any_valid, maximum, tf.zeros_like(maximum))


def _head(outputs: int, width: int, name: str) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(width, activation="gelu"),
            tf.keras.layers.Dense(width, activation="gelu"),
            tf.keras.layers.Dense(outputs),
        ],
        name=name,
    )


def _validated_inputs(
    inputs: dict[str, tf.Tensor],
    max_sites: int,
    max_species: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    required = {"X", "Y", "response_mask", "site_mask", "species_mask"}
    missing = sorted(required.difference(inputs))
    if missing:
        raise ValueError(f"generative iid v2 inputs missing: {missing}")
    X = tf.cast(inputs["X"], tf.float32)
    Y = tf.cast(inputs["Y"], tf.float32)
    response_mask = tf.cast(inputs["response_mask"], tf.bool)
    site_mask = tf.cast(inputs["site_mask"], tf.bool)
    species_mask = tf.cast(inputs["species_mask"], tf.bool)
    tf.debugging.assert_shapes(
        [
            (X, ("B", max_sites, 2)),
            (Y, ("B", max_sites, max_species)),
            (response_mask, ("B", max_sites, max_species)),
            (site_mask, ("B", max_sites)),
            (species_mask, ("B", max_species)),
        ]
    )
    tf.debugging.assert_near(
        tf.boolean_mask(X[..., 0], site_mask),
        tf.ones_like(tf.boolean_mask(X[..., 0], site_mask)),
        atol=1e-6,
    )
    return X, Y, response_mask, site_mask, species_mask


def _validate_model_bounds(max_sites: int, max_species: int) -> None:
    if not GENERATIVE_IID_MIN_SITES <= int(max_sites) <= GENERATIVE_IID_MAX_SITES:
        raise ValueError("v2 max_sites is outside the frozen support")
    if not (
        GENERATIVE_IID_MIN_SPECIES <= int(max_species) <= GENERATIVE_IID_MAX_SPECIES
    ):
        raise ValueError("v2 max_species is outside the frozen support")
    if GENERATIVE_IID_N_FACTORS != 2:
        raise AssertionError("the orbit density is frozen to two factors")
