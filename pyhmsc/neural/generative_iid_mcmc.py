"""Exact-model MCMC reference for generative Neural-HMSC iid probit v1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import tensorflow as tf

from pyhmsc.neural.generative_iid import (
    GENERATIVE_IID_PROTOCOL,
    GenerativeIidDataset,
    JointStateLayout,
    batch_generative_iid_datasets,
    generative_log_prior,
    probit_log_likelihood,
)


@dataclass(frozen=True)
class ExactMcmcResult:
    """Exact-model state draws and sampler diagnostics."""

    samples: np.ndarray
    layout: JointStateLayout
    acceptance_probability: np.ndarray
    step_size: np.ndarray
    split_rhat_max: float
    bulk_ess_min: float
    diagnostics: dict[str, Any]

    @property
    def chains(self) -> int:
        return int(self.samples.shape[0])

    @property
    def draws(self) -> int:
        return int(self.samples.shape[1])


def exact_model_log_joint(
    state: tf.Tensor,
    dataset: GenerativeIidDataset,
) -> tf.Tensor:
    """Evaluate the exact preregistered target for one unpadded community."""
    n_sites, n_species = dataset.Y.shape
    layout = JointStateLayout(n_sites, n_species)
    batch = batch_generative_iid_datasets(
        [dataset], max_sites=n_sites, max_species=n_species
    )
    state = tf.cast(tf.convert_to_tensor(state), tf.float64)
    if state.shape.rank == 1:
        state = state[None, :]
    state_with_draw = state[:, None, :]
    prior = generative_log_prior(
        state_with_draw,
        layout=layout,
        site_mask=tf.convert_to_tensor(batch.site_mask),
        species_mask=tf.convert_to_tensor(batch.species_mask),
    )[:, 0]
    likelihood = probit_log_likelihood(
        state_with_draw,
        layout=layout,
        X=tf.convert_to_tensor(batch.X),
        Y=tf.convert_to_tensor(batch.Y),
        response_mask=tf.convert_to_tensor(batch.response_mask),
        site_mask=tf.convert_to_tensor(batch.site_mask),
        species_mask=tf.convert_to_tensor(batch.species_mask),
    )[:, 0]
    return prior + likelihood


def initial_exact_mcmc_state(
    dataset: GenerativeIidDataset,
    *,
    chains: int,
    seed: int,
) -> np.ndarray:
    """Create prior-scale, non-neural initial states for exact MCMC."""
    if chains <= 0:
        raise ValueError("chains must be positive")
    n_sites, n_species = dataset.Y.shape
    layout = JointStateLayout(n_sites, n_species)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 701]))
    states = np.zeros((chains, layout.size), dtype=np.float64)
    for chain in range(chains):
        alpha = float(rng.normal(-0.50, 0.20))
        log_tau = float(rng.normal(math.log(0.65), 0.10))
        beta = np.vstack(
            [
                rng.normal(alpha, 0.10, size=n_species),
                rng.normal(0.0, 0.10, size=n_species),
            ]
        )
        eta = rng.normal(0.0, 0.10, size=(n_sites, 2))
        loadings = rng.normal(
            0.0, 0.10 * math.exp(log_tau), size=(2, n_species)
        )
        states[chain] = np.concatenate(
            [
                [alpha],
                beta.ravel(),
                eta.ravel(),
                loadings.ravel(),
                [log_tau],
            ]
        )
    return states


def run_exact_model_mcmc(
    dataset: GenerativeIidDataset,
    *,
    chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    seed: int,
    target_acceptance: float = 0.85,
    step_size: float = 0.02,
    initial_state: np.ndarray | None = None,
) -> ExactMcmcResult:
    """Run the independent TensorFlow Probability NUTS reference."""
    if chains <= 0 or warmup <= 0 or draws <= 0:
        raise ValueError("chains, warmup, and draws must be positive")
    if not 0.0 < target_acceptance < 1.0:
        raise ValueError("target_acceptance must be between zero and one")
    try:
        import tensorflow_probability as tfp
    except ImportError as error:
        raise RuntimeError(
            "tensorflow-probability is required for exact-model MCMC"
        ) from error

    if initial_state is None:
        initial_values = initial_exact_mcmc_state(
            dataset,
            chains=chains,
            seed=seed,
        )
    else:
        initial_values = np.asarray(initial_state, dtype=np.float64)
        expected = (chains, JointStateLayout(*dataset.Y.shape).size)
        if initial_values.shape != expected:
            raise ValueError(
                "continued exact-MCMC initial state must have shape "
                f"{expected}"
            )
        if not np.isfinite(initial_values).all():
            raise ValueError("continued exact-MCMC initial state is non-finite")
    initial = tf.convert_to_tensor(initial_values, dtype=tf.float64)

    def target_log_prob(state: tf.Tensor) -> tf.Tensor:
        return exact_model_log_joint(state, dataset)

    inner = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob,
        step_size=tf.cast(step_size, tf.float64),
        max_tree_depth=6,
    )
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=inner,
        num_adaptation_steps=max(int(0.8 * warmup), 1),
        target_accept_prob=tf.cast(target_acceptance, tf.float64),
    )

    state_draws, trace = tfp.mcmc.sample_chain(
        num_results=draws,
        num_burnin_steps=warmup,
        current_state=initial,
        kernel=kernel,
        seed=tf.constant([int(seed), 702], dtype=tf.int32),
        trace_fn=lambda _, results: (
            results.inner_results.log_accept_ratio,
            results.new_step_size,
        ),
    )
    log_accept_ratio, adapted_step = trace
    samples = np.transpose(np.asarray(state_draws), (1, 0, 2))
    acceptance = np.minimum(1.0, np.exp(np.asarray(log_accept_ratio)))
    registered = registered_mcmc_diagnostics(samples, JointStateLayout(*dataset.Y.shape))
    split_rhat = split_rhat_values(registered)
    bulk_ess = bulk_ess_values(registered)
    return ExactMcmcResult(
        samples=samples,
        layout=JointStateLayout(*dataset.Y.shape),
        acceptance_probability=acceptance,
        step_size=np.asarray(adapted_step),
        split_rhat_max=float(np.max(split_rhat)),
        bulk_ess_min=float(np.min(bulk_ess)),
        diagnostics={
            "protocol": GENERATIVE_IID_PROTOCOL,
            "chains": int(chains),
            "warmup": int(warmup),
            "draws": int(draws),
            "target_acceptance": float(target_acceptance),
            "continued_from_supplied_chain_states": (
                initial_state is not None
            ),
            "acceptance_probability_mean": float(np.mean(acceptance)),
            "split_rhat_max": float(np.max(split_rhat)),
            "bulk_ess_min": float(np.min(bulk_ess)),
            "registered_feature_count": int(registered.shape[-1]),
        },
    )


def registered_mcmc_diagnostics(
    samples: np.ndarray,
    layout: JointStateLayout,
) -> np.ndarray:
    """Return non-gauge scalars plus fixed invariant projections."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[-1] != layout.size:
        raise ValueError("MCMC samples must be chains x draws x state")
    state = tf.convert_to_tensor(samples, dtype=tf.float32)
    parameters = layout.unpack(state)
    beta = np.asarray(parameters["Beta"], dtype=np.float64)
    eta = np.asarray(parameters["Eta"], dtype=np.float64)
    loadings = np.asarray(parameters["Lambda"], dtype=np.float64)
    random_effect = np.einsum("cdnh,cdhs->cdns", eta, loadings)
    association = np.einsum("cdhs,cdht->cdst", loadings, loadings)
    diagonal = np.maximum(
        np.diagonal(association, axis1=-2, axis2=-1), 1e-8
    )
    correlation = association / np.sqrt(
        diagonal[..., :, None] * diagonal[..., None, :]
    )
    features = [
        np.asarray(parameters["alpha"], dtype=np.float64)[..., None],
        np.asarray(parameters["log_tau"], dtype=np.float64)[..., None],
        beta.reshape(samples.shape[0], samples.shape[1], -1),
    ]
    for family, values in (
        ("Beta", beta),
        ("R", random_effect),
        ("C", correlation),
    ):
        flattened = values.reshape(samples.shape[0], samples.shape[1], -1)
        projections = fixed_rademacher_projections(
            family, flattened.shape[-1], count=16
        )
        features.append(np.einsum("cdf,pf->cdp", flattened, projections))
    return np.concatenate(features, axis=-1)


def fixed_rademacher_projections(
    family: str,
    dimension: int,
    *,
    count: int = 16,
) -> np.ndarray:
    """Generate protocol-hash-owned unit Rademacher projections."""
    if dimension <= 0 or count <= 0:
        raise ValueError("dimension and count must be positive")
    rows = []
    for index in range(count):
        key = f"{GENERATIVE_IID_PROTOCOL}:{family}:{index}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        row = rng.choice([-1.0, 1.0], size=dimension)
        rows.append(row / np.linalg.norm(row))
    return np.asarray(rows, dtype=np.float64)


def split_rhat_values(values: np.ndarray) -> np.ndarray:
    """Compute ordinary split-Rhat for chains x draws x features."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("split R-hat needs >=2 chains and >=4 draws")
    half = values.shape[1] // 2
    split = np.concatenate(
        [values[:, :half], values[:, -half:]], axis=0
    )
    n = split.shape[1]
    chain_means = np.mean(split, axis=1)
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = n * np.var(chain_means, axis=0, ddof=1)
    variance = ((n - 1.0) / n) * within + between / n
    return np.sqrt(
        np.maximum(variance, np.finfo(float).eps)
        / np.maximum(within, np.finfo(float).eps)
    )


def bulk_ess_values(values: np.ndarray) -> np.ndarray:
    """Conservative initial-positive-sequence bulk ESS estimate."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("ESS needs >=2 chains and >=4 draws")
    chains, draws, features = values.shape
    centered = values - np.mean(values, axis=1, keepdims=True)
    variance = np.mean(np.var(values, axis=1, ddof=1), axis=0)
    ess = np.empty(features, dtype=np.float64)
    for feature in range(features):
        if variance[feature] <= np.finfo(float).eps:
            ess[feature] = chains * draws
            continue
        autocorrelations = []
        for lag in range(1, draws):
            covariance = np.mean(
                centered[:, :-lag, feature]
                * centered[:, lag:, feature]
            )
            autocorrelations.append(covariance / variance[feature])
        positive_sum = 0.0
        for index in range(0, len(autocorrelations) - 1, 2):
            pair = autocorrelations[index] + autocorrelations[index + 1]
            if pair <= 0.0:
                break
            positive_sum += pair
        ess[feature] = min(
            chains * draws,
            chains * draws / max(1.0 + 2.0 * positive_sum, 1.0),
        )
    return ess
