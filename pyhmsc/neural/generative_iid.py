"""Generative Neural-HMSC iid latent-factor probit model.

This module implements the frozen ``generative_neural_hmsc_iid_probit_v1``
representation. It is intentionally independent of the legacy fixed-effect
posterior and iid residual-SVD prototypes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.special import ndtr
import tensorflow as tf


GENERATIVE_IID_PROTOCOL = "generative_neural_hmsc_iid_probit_v1"
GENERATIVE_IID_SCHEMA_VERSION = 1
GENERATIVE_IID_N_COVARIATES = 2
GENERATIVE_IID_N_FACTORS = 2
GENERATIVE_IID_POSTERIOR_RANK = 16
GENERATIVE_IID_MIN_SITES = 24
GENERATIVE_IID_MAX_SITES = 96
GENERATIVE_IID_MIN_SPECIES = 12
GENERATIVE_IID_MAX_SPECIES = 75
GENERATIVE_IID_PREREGISTRATION_SHA256 = (
    "09c6a195ca139bdf168816b4f50db321c789bfdd061628e4f99a28cca81cea3f"
)
GENERATIVE_IID_SEED_AUDIT_SHA256 = (
    "39e8763bf8a4fd525dc624570cd2f2b3392dbd1f62d7fa2e3c326f9340194cd6"
)
GENERATIVE_IID_DESIGN_REVIEW_SHA256 = (
    "d271caed64dc1346b1f8d9e192534949adedd3122c1e311638e912ca868990cc"
)

_LOADING_STRATA = {
    "weak": (0.15, 0.35),
    "medium": (0.50, 0.80),
    "strong": (0.95, 1.30),
}
_PREVALENCE_STRATA = {
    "rare": (0.08, 0.20),
    "moderate": (0.30, 0.50),
    "common": (0.60, 0.78),
}
_COVARIATE_SHAPES = {"normal", "right_skewed"}


@dataclass(frozen=True)
class GenerativeIidDataset:
    """One prior-conditional iid latent-factor probit community."""

    X: np.ndarray
    Y: np.ndarray
    response_mask: np.ndarray
    truth_alpha: float
    truth_beta: np.ndarray
    truth_eta: np.ndarray
    truth_lambda: np.ndarray
    truth_log_tau: float
    truth_random_effect: np.ndarray
    truth_association: np.ndarray
    truth_association_correlation: np.ndarray
    probabilities: np.ndarray
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        n_sites, n_covariates = self.X.shape
        if n_covariates != GENERATIVE_IID_N_COVARIATES:
            raise ValueError("generative iid X must contain Intercept and x1")
        if self.Y.shape[0] != n_sites:
            raise ValueError("X and Y site dimensions differ")
        n_species = self.Y.shape[1]
        expected = {
            "response_mask": (n_sites, n_species),
            "truth_beta": (GENERATIVE_IID_N_COVARIATES, n_species),
            "truth_eta": (n_sites, GENERATIVE_IID_N_FACTORS),
            "truth_lambda": (GENERATIVE_IID_N_FACTORS, n_species),
            "truth_random_effect": (n_sites, n_species),
            "truth_association": (n_species, n_species),
            "truth_association_correlation": (n_species, n_species),
            "probabilities": (n_sites, n_species),
        }
        for name, shape in expected.items():
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} shape differs: expected {shape}")
        if self.response_mask.dtype != bool:
            raise ValueError("response_mask must be Boolean")
        if not np.all(np.isin(self.Y, [0.0, 1.0])):
            raise ValueError("generative iid Y must be binary")
        if not np.allclose(self.X[:, 0], 1.0):
            raise ValueError("the first X column must be the intercept")
        if not all(
            np.all(np.isfinite(np.asarray(value)))
            for value in (
                self.X,
                self.Y,
                self.truth_beta,
                self.truth_eta,
                self.truth_lambda,
                self.probabilities,
            )
        ):
            raise ValueError("generative iid dataset contains non-finite values")


@dataclass(frozen=True)
class GenerativeIidBatch:
    """Padded variable-shape tensors and simulation truths."""

    X: np.ndarray
    Y: np.ndarray
    response_mask: np.ndarray
    site_mask: np.ndarray
    species_mask: np.ndarray
    alpha: np.ndarray
    Beta: np.ndarray
    Eta: np.ndarray
    Lambda: np.ndarray
    log_tau: np.ndarray
    random_effect: np.ndarray
    association: np.ndarray
    association_correlation: np.ndarray
    metadata: tuple[dict[str, object], ...]

    def model_inputs(self) -> dict[str, np.ndarray]:
        return {
            "X": self.X,
            "Y": self.Y,
            "response_mask": self.response_mask,
            "site_mask": self.site_mask,
            "species_mask": self.species_mask,
        }


@dataclass(frozen=True)
class JointStateLayout:
    """Fixed padded layout for one model configuration."""

    max_sites: int
    max_species: int
    n_covariates: int = GENERATIVE_IID_N_COVARIATES
    n_factors: int = GENERATIVE_IID_N_FACTORS

    @property
    def alpha_slice(self) -> slice:
        return slice(0, 1)

    @property
    def beta_slice(self) -> slice:
        start = 1
        return slice(start, start + self.n_covariates * self.max_species)

    @property
    def eta_slice(self) -> slice:
        start = self.beta_slice.stop
        return slice(start, start + self.max_sites * self.n_factors)

    @property
    def lambda_slice(self) -> slice:
        start = self.eta_slice.stop
        return slice(start, start + self.n_factors * self.max_species)

    @property
    def log_tau_slice(self) -> slice:
        start = self.lambda_slice.stop
        return slice(start, start + 1)

    @property
    def size(self) -> int:
        return int(self.log_tau_slice.stop)

    def unpack(self, state: tf.Tensor) -> dict[str, tf.Tensor]:
        """Unpack leading-dimension state tensors into padded parameters."""
        state = tf.convert_to_tensor(state)
        leading = tf.shape(state)[:-1]
        beta = tf.reshape(
            state[..., self.beta_slice],
            tf.concat(
                [leading, [self.n_covariates, self.max_species]], axis=0
            ),
        )
        eta = tf.reshape(
            state[..., self.eta_slice],
            tf.concat([leading, [self.max_sites, self.n_factors]], axis=0),
        )
        loadings = tf.reshape(
            state[..., self.lambda_slice],
            tf.concat([leading, [self.n_factors, self.max_species]], axis=0),
        )
        return {
            "alpha": state[..., self.alpha_slice][..., 0],
            "Beta": beta,
            "Eta": eta,
            "Lambda": loadings,
            "log_tau": state[..., self.log_tau_slice][..., 0],
        }


@dataclass(frozen=True)
class JointLowRankPosterior:
    """Masked joint Normal with diagonal plus low-rank covariance."""

    mean: tf.Tensor
    diagonal_scale: tf.Tensor
    low_rank_factor: tf.Tensor
    state_mask: tf.Tensor
    layout: JointStateLayout
    site_mask: tf.Tensor
    species_mask: tf.Tensor

    def sample(self, draws: int, seed: int | None = None) -> tf.Tensor:
        if draws <= 0:
            raise ValueError("draws must be positive")
        dtype = self.mean.dtype
        batch = tf.shape(self.mean)[0]
        dimension = tf.shape(self.mean)[1]
        rank = tf.shape(self.low_rank_factor)[2]
        if seed is None:
            diagonal_noise = tf.random.normal(
                [draws, batch, dimension], dtype=dtype
            )
            rank_noise = tf.random.normal([draws, batch, rank], dtype=dtype)
        else:
            diagonal_noise = tf.random.stateless_normal(
                [draws, batch, dimension],
                seed=[int(seed), 1],
                dtype=dtype,
            )
            rank_noise = tf.random.stateless_normal(
                [draws, batch, rank],
                seed=[int(seed), 2],
                dtype=dtype,
            )
        low_rank = tf.einsum(
            "bdr,kbr->kbd", self.low_rank_factor, rank_noise
        )
        values = (
            self.mean[None, ...]
            + self.diagonal_scale[None, ...] * diagonal_noise
            + low_rank
        )
        mask = tf.cast(self.state_mask[None, ...], dtype)
        return values * mask + self.mean[None, ...] * (1.0 - mask)

    def log_prob(self, value: tf.Tensor) -> tf.Tensor:
        """Evaluate log q with Woodbury and determinant-lemma algebra."""
        value = tf.cast(value, self.mean.dtype)
        if value.shape.rank == 2:
            value = value[None, ...]
        mask = tf.cast(self.state_mask, self.mean.dtype)
        variance = tf.square(self.diagonal_scale) * mask + (1.0 - mask)
        inverse_variance = tf.math.reciprocal(variance)
        factor = self.low_rank_factor * mask[..., None]
        residual = (value - self.mean[None, ...]) * mask[None, ...]
        weighted_factor = factor * inverse_variance[..., None]
        small = tf.eye(
            tf.shape(factor)[-1],
            batch_shape=[tf.shape(factor)[0]],
            dtype=self.mean.dtype,
        ) + tf.einsum("bdr,bds->brs", factor, weighted_factor)
        chol = tf.linalg.cholesky(small)
        weighted_residual = residual * inverse_variance[None, ...]
        projected = tf.einsum("bdr,kbd->kbr", factor, weighted_residual)
        solved = tf.linalg.cholesky_solve(chol[None, ...], projected[..., None])
        correction = tf.reduce_sum(projected * solved[..., 0], axis=-1)
        quadratic = (
            tf.reduce_sum(residual * weighted_residual, axis=-1) - correction
        )
        logdet = tf.reduce_sum(
            tf.math.log(variance) * mask, axis=-1
        ) + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
        dimensions = tf.reduce_sum(mask, axis=-1)
        return -0.5 * (
            quadratic
            + logdet[None, ...]
            + dimensions[None, ...]
            * tf.math.log(tf.cast(2.0 * math.pi, self.mean.dtype))
        )


@dataclass(frozen=True)
class GenerativeTrainingHistory:
    loss: list[float]
    iwelbo: list[float]
    gradient_norm: list[float]


def simulate_generative_iid_dataset(
    *,
    n_sites: int,
    n_species: int,
    covariate_shape: str,
    loading_stratum: str,
    prevalence_stratum: str,
    seed: int,
    response_realization: int = 0,
    max_attempts: int = 512,
    response_mask: np.ndarray | None = None,
) -> GenerativeIidDataset:
    """Draw one community from the frozen prior conditional on factorial bins."""
    _validate_supported_shape(n_sites, n_species)
    if covariate_shape not in _COVARIATE_SHAPES:
        raise ValueError(f"unsupported covariate_shape: {covariate_shape!r}")
    if loading_stratum not in _LOADING_STRATA:
        raise ValueError(f"unsupported loading_stratum: {loading_stratum!r}")
    if prevalence_stratum not in _PREVALENCE_STRATA:
        raise ValueError(f"unsupported prevalence_stratum: {prevalence_stratum!r}")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if response_realization < 0:
        raise ValueError("response_realization must be non-negative")

    covariate_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 101])
    )
    raw_x = (
        covariate_rng.normal(size=n_sites)
        if covariate_shape == "normal"
        else np.exp(covariate_rng.normal(scale=0.75, size=n_sites))
    )
    x_sd = float(np.std(raw_x, ddof=1))
    if not np.isfinite(x_sd) or x_sd <= 0.0:
        raise RuntimeError("simulated covariate has zero or non-finite scale")
    x1 = (raw_x - np.mean(raw_x)) / x_sd
    X = np.column_stack([np.ones(n_sites), x1]).astype(np.float32)

    tau_low, tau_high = _LOADING_STRATA[loading_stratum]
    prevalence_low, prevalence_high = _PREVALENCE_STRATA[prevalence_stratum]
    parameter_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 202])
    )
    accepted = None
    for attempt in range(1, max_attempts + 1):
        alpha = float(parameter_rng.normal(loc=-0.50, scale=0.85))
        log_tau = float(
            parameter_rng.normal(loc=math.log(0.65), scale=0.45)
        )
        tau = math.exp(log_tau)
        beta = np.vstack(
            [
                parameter_rng.normal(alpha, 0.35, size=n_species),
                parameter_rng.normal(0.0, 0.50, size=n_species),
            ]
        )
        eta = parameter_rng.normal(
            size=(n_sites, GENERATIVE_IID_N_FACTORS)
        )
        loadings = parameter_rng.normal(
            scale=tau, size=(GENERATIVE_IID_N_FACTORS, n_species)
        )
        random_effect = eta @ loadings
        linear = X @ beta + random_effect
        probabilities = ndtr(linear)
        expected_prevalence = float(np.mean(probabilities))
        if (
            tau_low <= tau <= tau_high
            and prevalence_low <= expected_prevalence <= prevalence_high
        ):
            accepted = (
                alpha,
                log_tau,
                beta,
                eta,
                loadings,
                random_effect,
                probabilities,
                expected_prevalence,
                attempt,
            )
            break
    if accepted is None:
        raise RuntimeError(
            "failed to draw the requested loading/prevalence stratum within "
            f"{max_attempts} attempts for seed {seed}"
        )

    (
        alpha,
        log_tau,
        beta,
        eta,
        loadings,
        random_effect,
        probabilities,
        expected_prevalence,
        attempt,
    ) = accepted
    response_rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), 303, int(response_realization)]
        )
    )
    Y = response_rng.binomial(1, probabilities).astype(np.float32)
    if response_mask is None:
        observed = np.ones_like(Y, dtype=bool)
    else:
        observed = np.asarray(response_mask, dtype=bool)
        if observed.shape != Y.shape:
            raise ValueError("response_mask shape differs from Y")
    association = loadings.T @ loadings
    association_correlation = association_to_correlation(association)
    return GenerativeIidDataset(
        X=X,
        Y=Y,
        response_mask=observed,
        truth_alpha=alpha,
        truth_beta=beta.astype(np.float32),
        truth_eta=eta.astype(np.float32),
        truth_lambda=loadings.astype(np.float32),
        truth_log_tau=log_tau,
        truth_random_effect=random_effect.astype(np.float32),
        truth_association=association.astype(np.float32),
        truth_association_correlation=association_correlation.astype(np.float32),
        probabilities=probabilities.astype(np.float32),
        metadata={
            "protocol": GENERATIVE_IID_PROTOCOL,
            "seed": int(seed),
            "response_realization": int(response_realization),
            "n_sites": int(n_sites),
            "n_species": int(n_species),
            "n_covariates": GENERATIVE_IID_N_COVARIATES,
            "n_factors": GENERATIVE_IID_N_FACTORS,
            "covariate_shape": covariate_shape,
            "loading_stratum": loading_stratum,
            "prevalence_stratum": prevalence_stratum,
            "tau": float(math.exp(log_tau)),
            "expected_prevalence": expected_prevalence,
            "observed_prevalence": float(np.mean(Y)),
            "parameter_attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "distribution": "probit",
            "formula": "~ x1",
            "random_level": "iid_site",
        },
    )


def make_stratified_response_mask(
    n_sites: int,
    n_species: int,
    *,
    seed: int,
    holdout_fraction: float = 0.20,
) -> np.ndarray:
    """Create an outcome-blind cell mask with observed/hidden row/column support."""
    if n_sites < 2 or n_species < 2:
        raise ValueError("stratified masking requires at least two sites/species")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 404]))
    scores = rng.uniform(size=(n_sites, n_species))
    hidden = scores < float(holdout_fraction)
    for site in range(n_sites):
        hidden[site, np.argmin(scores[site])] = True
        hidden[site, np.argmax(scores[site])] = False
    for species in range(n_species):
        hidden[np.argmin(scores[:, species]), species] = True
        hidden[np.argmax(scores[:, species]), species] = False
    observed = ~hidden
    if not (
        np.all(observed.any(axis=0))
        and np.all(observed.any(axis=1))
        and np.all(hidden.any(axis=0))
        and np.all(hidden.any(axis=1))
    ):
        raise RuntimeError("failed to construct stratified response mask")
    return observed


def batch_generative_iid_datasets(
    datasets: Sequence[GenerativeIidDataset],
    *,
    max_sites: int | None = None,
    max_species: int | None = None,
) -> GenerativeIidBatch:
    """Pad variable communities while retaining all structural truths."""
    if not datasets:
        raise ValueError("datasets must not be empty")
    observed_sites = max(dataset.Y.shape[0] for dataset in datasets)
    observed_species = max(dataset.Y.shape[1] for dataset in datasets)
    max_sites = int(max_sites if max_sites is not None else observed_sites)
    max_species = int(max_species if max_species is not None else observed_species)
    if observed_sites > max_sites or observed_species > max_species:
        raise ValueError("dataset exceeds requested padding dimensions")
    batch = len(datasets)
    X = np.zeros((batch, max_sites, 2), dtype=np.float32)
    Y = np.zeros((batch, max_sites, max_species), dtype=np.float32)
    response_mask = np.zeros((batch, max_sites, max_species), dtype=bool)
    site_mask = np.zeros((batch, max_sites), dtype=bool)
    species_mask = np.zeros((batch, max_species), dtype=bool)
    alpha = np.zeros(batch, dtype=np.float32)
    beta = np.zeros((batch, 2, max_species), dtype=np.float32)
    eta = np.zeros((batch, max_sites, 2), dtype=np.float32)
    loadings = np.zeros((batch, 2, max_species), dtype=np.float32)
    log_tau = np.zeros(batch, dtype=np.float32)
    random_effect = np.zeros(
        (batch, max_sites, max_species), dtype=np.float32
    )
    association = np.zeros(
        (batch, max_species, max_species), dtype=np.float32
    )
    correlation = np.zeros_like(association)
    for index, dataset in enumerate(datasets):
        n_sites, n_species = dataset.Y.shape
        X[index, :n_sites] = dataset.X
        Y[index, :n_sites, :n_species] = dataset.Y
        response_mask[index, :n_sites, :n_species] = dataset.response_mask
        site_mask[index, :n_sites] = True
        species_mask[index, :n_species] = True
        alpha[index] = dataset.truth_alpha
        beta[index, :, :n_species] = dataset.truth_beta
        eta[index, :n_sites] = dataset.truth_eta
        loadings[index, :, :n_species] = dataset.truth_lambda
        log_tau[index] = dataset.truth_log_tau
        random_effect[index, :n_sites, :n_species] = dataset.truth_random_effect
        association[index, :n_species, :n_species] = dataset.truth_association
        correlation[
            index, :n_species, :n_species
        ] = dataset.truth_association_correlation
    return GenerativeIidBatch(
        X=X,
        Y=Y,
        response_mask=response_mask,
        site_mask=site_mask,
        species_mask=species_mask,
        alpha=alpha,
        Beta=beta,
        Eta=eta,
        Lambda=loadings,
        log_tau=log_tau,
        random_effect=random_effect,
        association=association,
        association_correlation=correlation,
        metadata=tuple(dict(dataset.metadata) for dataset in datasets),
    )


class GenerativeIidPosteriorModel(tf.keras.Model):
    """Permutation-equivariant structured posterior for the frozen iid family."""

    def __init__(
        self,
        *,
        max_sites: int = GENERATIVE_IID_MAX_SITES,
        max_species: int = GENERATIVE_IID_MAX_SPECIES,
        hidden_width: int = 64,
        message_rounds: int = 3,
        posterior_rank: int = GENERATIVE_IID_POSTERIOR_RANK,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if hidden_width != 64 or message_rounds != 3 or posterior_rank != 16:
            raise ValueError(
                "v1 requires width 64, three rounds, and posterior rank 16"
            )
        _validate_model_bounds(max_sites, max_species)
        self.max_sites = int(max_sites)
        self.max_species = int(max_species)
        self.hidden_width = int(hidden_width)
        self.message_rounds = int(message_rounds)
        self.posterior_rank = int(posterior_rank)
        self.layout = JointStateLayout(self.max_sites, self.max_species)
        self.site_initial = tf.keras.layers.Dense(64, activation="gelu")
        self.species_initial = tf.keras.layers.Dense(64, activation="gelu")
        self.site_message_layers = [
            _message_mlp(f"site_message_{index}")
            for index in range(self.message_rounds)
        ]
        self.species_message_layers = [
            _message_mlp(f"species_message_{index}")
            for index in range(self.message_rounds)
        ]
        self.site_update_layers = [
            _update_mlp(f"site_update_{index}")
            for index in range(self.message_rounds)
        ]
        self.species_update_layers = [
            _update_mlp(f"species_update_{index}")
            for index in range(self.message_rounds)
        ]
        self.site_norms = [
            tf.keras.layers.LayerNormalization(name=f"site_norm_{index}")
            for index in range(self.message_rounds)
        ]
        self.species_norms = [
            tf.keras.layers.LayerNormalization(name=f"species_norm_{index}")
            for index in range(self.message_rounds)
        ]
        self.coefficient_embedding = tf.keras.layers.Embedding(2, 8)
        self.factor_embedding = tf.keras.layers.Embedding(2, 8)
        self.beta_head = _posterior_head("beta_head")
        self.eta_head = _posterior_head("eta_head")
        self.lambda_head = _posterior_head("lambda_head")
        self.global_head = _global_posterior_head("global_head")

    def get_config(self) -> dict[str, object]:
        return {
            "max_sites": self.max_sites,
            "max_species": self.max_species,
            "hidden_width": self.hidden_width,
            "message_rounds": self.message_rounds,
            "posterior_rank": self.posterior_rank,
        }

    def call(
        self, inputs: dict[str, tf.Tensor], training: bool = False
    ) -> JointLowRankPosterior:
        X, Y, response_mask, site_mask, species_mask = _validated_inputs(
            inputs, self.max_sites, self.max_species
        )
        dtype = X.dtype
        observation = tf.cast(response_mask, dtype)
        pair_mask = (
            observation
            * tf.cast(site_mask[:, :, None], dtype)
            * tf.cast(species_mask[:, None, :], dtype)
        )
        observed_count_site = tf.reduce_sum(pair_mask, axis=2, keepdims=True)
        observed_count_species = tf.reduce_sum(pair_mask, axis=1)
        site_prevalence = tf.math.divide_no_nan(
            tf.reduce_sum(Y * pair_mask, axis=2, keepdims=True),
            observed_count_site,
        )
        site_fraction = tf.math.divide_no_nan(
            observed_count_site,
            tf.reduce_sum(
                tf.cast(species_mask, dtype), axis=1, keepdims=True
            )[:, None, :],
        )
        species_prevalence = tf.math.divide_no_nan(
            tf.reduce_sum(Y * pair_mask, axis=1), observed_count_species
        )
        species_fraction = tf.math.divide_no_nan(
            observed_count_species,
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
        edge = tf.stack([Y * observation, observation], axis=-1)
        for index in range(self.message_rounds):
            site_tokens = tf.broadcast_to(
                site_state[:, :, None, :],
                [
                    tf.shape(X)[0],
                    self.max_sites,
                    self.max_species,
                    self.hidden_width,
                ],
            )
            species_tokens = tf.broadcast_to(
                species_state[:, None, :, :],
                [
                    tf.shape(X)[0],
                    self.max_sites,
                    self.max_species,
                    self.hidden_width,
                ],
            )
            pair_features = tf.concat(
                [site_tokens, species_tokens, edge], axis=-1
            )
            site_edge_message = self.site_message_layers[index](
                pair_features, training=training
            )
            species_edge_message = self.species_message_layers[index](
                pair_features, training=training
            )
            site_message = _masked_mean(
                site_edge_message, pair_mask[..., None], axis=2
            )
            species_message = _masked_mean(
                species_edge_message, pair_mask[..., None], axis=1
            )
            site_delta = self.site_update_layers[index](
                tf.concat([site_state, site_message], axis=-1),
                training=training,
            )
            species_delta = self.species_update_layers[index](
                tf.concat([species_state, species_message], axis=-1),
                training=training,
            )
            site_state = self.site_norms[index](site_state + site_delta)
            species_state = self.species_norms[index](
                species_state + species_delta
            )
            site_state *= tf.cast(site_mask[..., None], dtype)
            species_state *= tf.cast(species_mask[..., None], dtype)

        community = tf.concat(
            [
                _masked_mean(
                    site_state, tf.cast(site_mask[..., None], dtype), axis=1
                ),
                _masked_max(
                    site_state, tf.cast(site_mask[..., None], dtype), axis=1
                ),
                _masked_mean(
                    species_state,
                    tf.cast(species_mask[..., None], dtype),
                    axis=1,
                ),
                _masked_max(
                    species_state,
                    tf.cast(species_mask[..., None], dtype),
                    axis=1,
                ),
            ],
            axis=-1,
        )
        design_mean = _masked_mean(
            X, tf.cast(site_mask[..., None], dtype), axis=1
        )
        centered = X - design_mean[:, None, :]
        design_variance = _masked_mean(
            tf.square(centered),
            tf.cast(site_mask[..., None], dtype),
            axis=1,
        )
        design_summary = tf.stack(
            [design_mean, tf.sqrt(tf.maximum(design_variance, 1e-8))],
            axis=-1,
        )

        coefficient_ids = tf.range(2)
        factor_ids = tf.range(2)
        coefficient_embedding = self.coefficient_embedding(coefficient_ids)
        factor_embedding = self.factor_embedding(factor_ids)
        beta_tokens = tf.concat(
            [
                tf.broadcast_to(
                    species_state[:, None, :, :],
                    [tf.shape(X)[0], 2, self.max_species, 64],
                ),
                tf.broadcast_to(
                    community[:, None, None, :],
                    [tf.shape(X)[0], 2, self.max_species, 256],
                ),
                tf.broadcast_to(
                    coefficient_embedding[None, :, None, :],
                    [tf.shape(X)[0], 2, self.max_species, 8],
                ),
                tf.broadcast_to(
                    design_summary[:, :, None, :],
                    [tf.shape(X)[0], 2, self.max_species, 2],
                ),
            ],
            axis=-1,
        )
        eta_tokens = tf.concat(
            [
                tf.broadcast_to(
                    site_state[:, :, None, :],
                    [tf.shape(X)[0], self.max_sites, 2, 64],
                ),
                tf.broadcast_to(
                    community[:, None, None, :],
                    [tf.shape(X)[0], self.max_sites, 2, 256],
                ),
                tf.broadcast_to(
                    factor_embedding[None, None, :, :],
                    [tf.shape(X)[0], self.max_sites, 2, 8],
                ),
            ],
            axis=-1,
        )
        lambda_tokens = tf.concat(
            [
                tf.broadcast_to(
                    species_state[:, None, :, :],
                    [tf.shape(X)[0], 2, self.max_species, 64],
                ),
                tf.broadcast_to(
                    community[:, None, None, :],
                    [tf.shape(X)[0], 2, self.max_species, 256],
                ),
                tf.broadcast_to(
                    factor_embedding[None, :, None, :],
                    [tf.shape(X)[0], 2, self.max_species, 8],
                ),
            ],
            axis=-1,
        )
        beta_raw = self.beta_head(beta_tokens, training=training)
        eta_raw = self.eta_head(eta_tokens, training=training)
        lambda_raw = self.lambda_head(lambda_tokens, training=training)
        global_raw = tf.reshape(
            self.global_head(community, training=training),
            [tf.shape(X)[0], 2, 18],
        )
        return _assemble_joint_posterior(
            beta_raw=beta_raw,
            eta_raw=eta_raw,
            lambda_raw=lambda_raw,
            global_raw=global_raw,
            site_mask=site_mask,
            species_mask=species_mask,
            layout=self.layout,
        )


def generative_log_prior(
    state: tf.Tensor,
    *,
    layout: JointStateLayout,
    site_mask: tf.Tensor,
    species_mask: tf.Tensor,
) -> tf.Tensor:
    """Evaluate the exact frozen prior on padded state samples."""
    state = tf.convert_to_tensor(state)
    if not state.dtype.is_floating:
        state = tf.cast(state, tf.float32)
    if state.shape.rank == 2:
        state = state[None, ...]
    parameters = layout.unpack(state)
    alpha = parameters["alpha"]
    beta = parameters["Beta"]
    eta = parameters["Eta"]
    loadings = parameters["Lambda"]
    log_tau = parameters["log_tau"]
    species = tf.cast(species_mask[None, :, :], state.dtype)
    sites = tf.cast(site_mask[None, :, :], state.dtype)
    logp = _normal_log_prob(alpha, -0.50, 0.85)
    logp += _normal_log_prob(log_tau, math.log(0.65), 0.45)
    logp += tf.reduce_sum(
        _normal_log_prob(beta[..., 0, :], alpha[..., None], 0.35)
        * species,
        axis=-1,
    )
    logp += tf.reduce_sum(
        _normal_log_prob(beta[..., 1, :], 0.0, 0.50) * species,
        axis=-1,
    )
    logp += tf.reduce_sum(
        _normal_log_prob(eta, 0.0, 1.0) * sites[..., None],
        axis=(-2, -1),
    )
    tau = tf.exp(log_tau)
    logp += tf.reduce_sum(
        _normal_log_prob(loadings, 0.0, tau[..., None, None])
        * species[..., None, :],
        axis=(-2, -1),
    )
    return logp


def probit_log_likelihood(
    state: tf.Tensor,
    *,
    layout: JointStateLayout,
    X: tf.Tensor,
    Y: tf.Tensor,
    response_mask: tf.Tensor,
    site_mask: tf.Tensor,
    species_mask: tf.Tensor,
) -> tf.Tensor:
    """Evaluate observed-cell Bernoulli-probit log likelihood."""
    state = tf.convert_to_tensor(state)
    if not state.dtype.is_floating:
        state = tf.cast(state, tf.float32)
    if state.shape.rank == 2:
        state = state[None, ...]
    parameters = layout.unpack(state)
    beta = parameters["Beta"]
    eta = parameters["Eta"]
    loadings = parameters["Lambda"]
    fixed = tf.einsum("bni,kbis->kbns", tf.cast(X, state.dtype), beta)
    random = tf.einsum("kbnh,kbhs->kbns", eta, loadings)
    probabilities = 0.5 * (
        1.0
        + tf.math.erf(
            (fixed + random)
            / tf.sqrt(tf.cast(2.0, state.dtype))
        )
    )
    probabilities = tf.clip_by_value(probabilities, 1e-6, 1.0 - 1e-6)
    Y = tf.cast(Y, state.dtype)[None, ...]
    mask = (
        tf.cast(response_mask, state.dtype)
        * tf.cast(site_mask[:, :, None], state.dtype)
        * tf.cast(species_mask[:, None, :], state.dtype)
    )[None, ...]
    point = Y * tf.math.log(probabilities) + (1.0 - Y) * tf.math.log(
        1.0 - probabilities
    )
    return tf.reduce_sum(point * mask, axis=(-2, -1))


def importance_weighted_variational_loss(
    posterior: JointLowRankPosterior,
    inputs: dict[str, tf.Tensor | np.ndarray],
    *,
    draws: int = 8,
    kl_weight: float = 1.0,
    seed: int | None = None,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Return negative IWELBO and diagnostic components."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    if not 0.0 < kl_weight <= 1.0:
        raise ValueError("kl_weight must be in (0, 1]")
    samples = posterior.sample(draws, seed=seed)
    log_likelihood = probit_log_likelihood(
        samples,
        layout=posterior.layout,
        X=tf.convert_to_tensor(inputs["X"], dtype=tf.float32),
        Y=tf.convert_to_tensor(inputs["Y"], dtype=tf.float32),
        response_mask=tf.convert_to_tensor(inputs["response_mask"], dtype=tf.bool),
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
    iwelbo_by_batch = tf.reduce_logsumexp(weights, axis=0) - tf.math.log(
        tf.cast(draws, weights.dtype)
    )
    loss = -tf.reduce_mean(iwelbo_by_batch)
    return loss, {
        "iwelbo": tf.reduce_mean(iwelbo_by_batch),
        "log_likelihood": tf.reduce_mean(log_likelihood),
        "log_prior": tf.reduce_mean(log_prior),
        "log_q": tf.reduce_mean(log_q),
    }


def train_generative_iid_model(
    model: GenerativeIidPosteriorModel,
    batch: GenerativeIidBatch,
    *,
    epochs: int = 200,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    final_learning_rate: float = 3e-5,
    weight_decay: float = 1e-5,
    gradient_clip_norm: float = 5.0,
    model_seed: int = 501900001,
    importance_draws: int = 8,
) -> GenerativeTrainingHistory:
    """Train with the frozen schedule; reduced values are allowed for unit smoke."""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0.0 or final_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
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
                name: value[indices]
                for name, value in batch.model_inputs().items()
            }
            with tf.GradientTape() as tape:
                posterior = model(inputs, training=True)
                loss, diagnostics = importance_weighted_variational_loss(
                    posterior,
                    inputs,
                    draws=importance_draws,
                    kl_weight=kl_weight,
                )
            gradients = tape.gradient(loss, model.trainable_variables)
            finite_gradients = [
                gradient
                for gradient in gradients
                if gradient is not None
            ]
            if not finite_gradients or not all(
                bool(
                    tf.reduce_all(
                        tf.math.is_finite(tf.convert_to_tensor(gradient))
                    )
                )
                for gradient in finite_gradients
            ):
                raise FloatingPointError("non-finite generative iid gradient")
            clipped, norm = tf.clip_by_global_norm(
                finite_gradients, gradient_clip_norm
            )
            variables = [
                variable
                for gradient, variable in zip(
                    gradients, model.trainable_variables
                )
                if gradient is not None
            ]
            optimizer.apply_gradients(zip(clipped, variables))
            if not bool(tf.math.is_finite(loss)):
                raise FloatingPointError("non-finite generative iid loss")
            epoch_loss.append(float(loss.numpy()))
            epoch_iwelbo.append(float(diagnostics["iwelbo"].numpy()))
            epoch_norm.append(float(norm.numpy()))
        losses.append(float(np.mean(epoch_loss)))
        iwelbos.append(float(np.mean(epoch_iwelbo)))
        gradient_norms.append(float(np.mean(epoch_norm)))
    return GenerativeTrainingHistory(losses, iwelbos, gradient_norms)


def state_vector_from_truth(
    batch: GenerativeIidBatch, layout: JointStateLayout
) -> tf.Tensor:
    """Pack padded simulation truths in the exact model layout."""
    if batch.X.shape[1] != layout.max_sites:
        raise ValueError("batch site padding does not match layout")
    if batch.Y.shape[2] != layout.max_species:
        raise ValueError("batch species padding does not match layout")
    return tf.concat(
        [
            tf.convert_to_tensor(batch.alpha[:, None], dtype=tf.float32),
            tf.reshape(
                tf.convert_to_tensor(batch.Beta, dtype=tf.float32),
                [len(batch.metadata), -1],
            ),
            tf.reshape(
                tf.convert_to_tensor(batch.Eta, dtype=tf.float32),
                [len(batch.metadata), -1],
            ),
            tf.reshape(
                tf.convert_to_tensor(batch.Lambda, dtype=tf.float32),
                [len(batch.metadata), -1],
            ),
            tf.convert_to_tensor(batch.log_tau[:, None], dtype=tf.float32),
        ],
        axis=-1,
    )


def posterior_mean_invariants(
    posterior: JointLowRankPosterior,
) -> dict[str, tf.Tensor]:
    """Compute deterministic mean-coordinate invariant summaries."""
    parameters = posterior.layout.unpack(posterior.mean)
    random_effect = tf.einsum(
        "bnh,bhs->bns", parameters["Eta"], parameters["Lambda"]
    )
    association = tf.einsum(
        "bhs,bht->bst", parameters["Lambda"], parameters["Lambda"]
    )
    diagonal = tf.maximum(tf.linalg.diag_part(association), 1e-8)
    correlation = association / tf.sqrt(
        diagonal[:, :, None] * diagonal[:, None, :]
    )
    return {
        "Beta": parameters["Beta"],
        "R": random_effect,
        "A": association,
        "C": correlation,
    }


def association_to_correlation(association: np.ndarray) -> np.ndarray:
    association = np.asarray(association, dtype=float)
    diagonal = np.maximum(np.diag(association), 1e-8)
    return association / np.sqrt(diagonal[:, None] * diagonal[None, :])


def gauge_fix_factors(
    eta: np.ndarray, loadings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply deterministic orientation while preserving invariant products."""
    eta = np.asarray(eta, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    if eta.ndim != 2 or loadings.ndim != 2:
        raise ValueError("eta and loadings must be matrices")
    if eta.shape[1] != loadings.shape[0]:
        raise ValueError("eta and loading factor dimensions differ")
    eigenvalues, eigenvectors = np.linalg.eigh(loadings @ loadings.T)
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order]
    fixed_eta = eta @ rotation
    fixed_loadings = rotation.T @ loadings
    for factor in range(fixed_loadings.shape[0]):
        pivot = int(np.argmax(np.abs(fixed_loadings[factor])))
        if fixed_loadings[factor, pivot] < 0.0:
            fixed_loadings[factor] *= -1.0
            fixed_eta[:, factor] *= -1.0
    return fixed_eta, fixed_loadings


def _assemble_joint_posterior(
    *,
    beta_raw: tf.Tensor,
    eta_raw: tf.Tensor,
    lambda_raw: tf.Tensor,
    global_raw: tf.Tensor,
    site_mask: tf.Tensor,
    species_mask: tf.Tensor,
    layout: JointStateLayout,
) -> JointLowRankPosterior:
    beta_raw = tf.convert_to_tensor(beta_raw)
    eta_raw = tf.convert_to_tensor(eta_raw)
    lambda_raw = tf.convert_to_tensor(lambda_raw)
    global_raw = tf.convert_to_tensor(global_raw)
    beta_flat = tf.reshape(beta_raw, [tf.shape(beta_raw)[0], -1, 18])
    eta_flat = tf.reshape(eta_raw, [tf.shape(eta_raw)[0], -1, 18])
    lambda_flat = tf.reshape(
        lambda_raw, [tf.shape(lambda_raw)[0], -1, 18]
    )
    ordered = tf.concat(
        [
            global_raw[:, 0:1],
            beta_flat,
            eta_flat,
            lambda_flat,
            global_raw[:, 1:2],
        ],
        axis=1,
    )
    species_mask = tf.cast(species_mask, tf.bool)
    site_mask = tf.cast(site_mask, tf.bool)
    state_mask = tf.concat(
        [
            tf.ones([tf.shape(ordered)[0], 1], dtype=tf.bool),
            tf.reshape(
                tf.broadcast_to(
                    species_mask[:, None, :],
                    [tf.shape(ordered)[0], 2, layout.max_species],
                ),
                [tf.shape(ordered)[0], -1],
            ),
            tf.reshape(
                tf.broadcast_to(
                    site_mask[:, :, None],
                    [tf.shape(ordered)[0], layout.max_sites, 2],
                ),
                [tf.shape(ordered)[0], -1],
            ),
            tf.reshape(
                tf.broadcast_to(
                    species_mask[:, None, :],
                    [tf.shape(ordered)[0], 2, layout.max_species],
                ),
                [tf.shape(ordered)[0], -1],
            ),
            tf.ones([tf.shape(ordered)[0], 1], dtype=tf.bool),
        ],
        axis=1,
    )
    state_float = tf.cast(state_mask, ordered.dtype)
    mean = ordered[..., 0] * state_float
    scale = (tf.nn.softplus(ordered[..., 1]) + 1e-4) * state_float + (
        1.0 - state_float
    )
    factor = ordered[..., 2:] * state_float[..., None]
    return JointLowRankPosterior(
        mean=mean,
        diagonal_scale=scale,
        low_rank_factor=factor,
        state_mask=state_mask,
        layout=layout,
        site_mask=site_mask,
        species_mask=species_mask,
    )


def _validated_inputs(
    inputs: dict[str, tf.Tensor],
    max_sites: int,
    max_species: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    required = {
        "X",
        "Y",
        "response_mask",
        "site_mask",
        "species_mask",
    }
    missing = sorted(required.difference(inputs))
    if missing:
        raise ValueError(f"generative iid model inputs missing: {missing}")
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


def _message_mlp(name: str) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(64, activation="gelu"),
        ],
        name=name,
    )


def _update_mlp(name: str) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(64, activation="gelu"),
        ],
        name=name,
    )


def _posterior_head(name: str) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(18),
        ],
        name=name,
    )


def _global_posterior_head(name: str) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(36),
        ],
        name=name,
    )


def _masked_mean(
    values: tf.Tensor, mask: tf.Tensor, *, axis: int
) -> tf.Tensor:
    mask = tf.cast(mask, values.dtype)
    return tf.math.divide_no_nan(
        tf.reduce_sum(values * mask, axis=axis),
        tf.reduce_sum(mask, axis=axis),
    )


def _masked_max(
    values: tf.Tensor, mask: tf.Tensor, *, axis: int
) -> tf.Tensor:
    mask = tf.cast(mask, tf.bool)
    masked = tf.where(mask, values, tf.cast(-1e9, values.dtype))
    maximum = tf.reduce_max(masked, axis=axis)
    any_valid = tf.reduce_any(mask, axis=axis)
    return tf.where(any_valid, maximum, tf.zeros_like(maximum))


def _normal_log_prob(
    value: tf.Tensor,
    location: tf.Tensor | float,
    scale: tf.Tensor | float,
) -> tf.Tensor:
    value = tf.convert_to_tensor(value)
    location = tf.cast(location, value.dtype)
    scale = tf.cast(scale, value.dtype)
    standardized = (value - location) / scale
    return (
        -0.5 * tf.square(standardized)
        - tf.math.log(scale)
        - 0.5 * tf.math.log(tf.cast(2.0 * math.pi, value.dtype))
    )


def _validate_supported_shape(n_sites: int, n_species: int) -> None:
    if not GENERATIVE_IID_MIN_SITES <= int(n_sites) <= GENERATIVE_IID_MAX_SITES:
        raise ValueError(
            f"n_sites must be in [{GENERATIVE_IID_MIN_SITES}, "
            f"{GENERATIVE_IID_MAX_SITES}]"
        )
    if not (
        GENERATIVE_IID_MIN_SPECIES
        <= int(n_species)
        <= GENERATIVE_IID_MAX_SPECIES
    ):
        raise ValueError(
            f"n_species must be in [{GENERATIVE_IID_MIN_SPECIES}, "
            f"{GENERATIVE_IID_MAX_SPECIES}]"
        )


def _validate_model_bounds(max_sites: int, max_species: int) -> None:
    if not (
        GENERATIVE_IID_MIN_SITES
        <= int(max_sites)
        <= GENERATIVE_IID_MAX_SITES
    ):
        raise ValueError("model max_sites is outside the preregistered support")
    if not (
        GENERATIVE_IID_MIN_SPECIES
        <= int(max_species)
        <= GENERATIVE_IID_MAX_SPECIES
    ):
        raise ValueError(
            "model max_species is outside the preregistered support"
        )
