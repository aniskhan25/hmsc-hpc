"""Comparator adapters for generative Neural-HMSC iid qualification."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr
import tensorflow as tf

from pyhmsc.model import HmscModel
from pyhmsc.neural.generative_iid import (
    GenerativeIidDataset,
    JointStateLayout,
    batch_generative_iid_datasets,
    zero_latent_state_draws,
)
from pyhmsc.neural.generative_iid_artifact import GenerativeIidInference
from pyhmsc.neural.generative_iid_evaluation import (
    deterministic_new_site_mask,
    evaluate_state_draws,
)
from pyhmsc.neural.generative_iid_mcmc import (
    bulk_ess_values,
    registered_mcmc_diagnostics,
    run_exact_model_mcmc,
    split_rhat_values,
)
from pyhmsc.neural.posterior_heads import sample_beta_posterior
from pyhmsc.neural.release import NeuralHmscRelease


def evaluate_neural_contexts(
    inference: GenerativeIidInference,
    datasets: Sequence[GenerativeIidDataset],
    *,
    draws: int = 256,
    zero_latent: bool = False,
    method: str = "candidate",
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate candidate or R=0 ablation contexts one at a time."""
    rows = []
    condition_estimates = []
    jittered_draws = 0
    total_draws = 0
    for dataset in datasets:
        batch = batch_generative_iid_datasets(
            [dataset],
            max_sites=inference.model.max_sites,
            max_species=inference.model.max_species,
        )
        started = time.perf_counter()
        posterior = inference.model(batch.model_inputs(), training=False)
        state = posterior.sample(
            draws,
            seed=_child_seed(int(dataset.metadata["seed"]), 801),
        )[:, 0, :]
        log_q = posterior.log_prob(state[:, None, :])
        if not bool(tf.reduce_all(tf.math.is_finite(log_q))):
            raise FloatingPointError("candidate posterior density is non-finite")
        if zero_latent:
            state = zero_latent_state_draws(state, posterior.layout)
        elapsed = time.perf_counter() - started
        condition_estimates.append(_posterior_condition_bound(posterior))
        total_draws += draws
        rows.append(
            evaluate_state_draws(
                dataset,
                np.asarray(state),
                layout=posterior.layout,
                method=method,
                inference_seconds=elapsed,
                peak_device_memory_bytes=_peak_device_memory_bytes(),
            )
        )
    return rows, {
        "covariance_jitter_fraction": (
            float(jittered_draws / total_draws) if total_draws else 0.0
        ),
        "covariance_condition_max": float(max(condition_estimates, default=0.0)),
        "max_shape_inference_seconds": float(
            max(
                (
                    row["inference_seconds"]
                    for row in rows
                    if int(row["n_sites"]) == 96
                    and int(row["n_species"]) == 75
                ),
                default=0.0,
            )
        ),
        "peak_device_memory_bytes": float(
            max(
                (
                    row["peak_device_memory_bytes"] or 0
                    for row in rows
                ),
                default=0,
            )
        ),
    }


def evaluate_exact_mcmc_contexts(
    datasets: Sequence[GenerativeIidDataset],
    *,
    warmup: int = 1000,
    draws: int = 1000,
    chains: int = 4,
    output_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Run the exact reference with its one preauthorized continuation."""
    rows = []
    diagnostics = []
    elapsed_total = 0.0
    root = (
        None
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    if root is not None:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"exact-MCMC output is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        seed = int(dataset.metadata["seed"])
        started = time.perf_counter()
        result = run_exact_model_mcmc(
            dataset,
            chains=chains,
            warmup=warmup,
            draws=draws,
            seed=_child_seed(seed, 702),
            target_acceptance=0.85,
        )
        continued = False
        samples = result.samples
        rhat = result.split_rhat_max
        ess = result.bulk_ess_min
        if rhat > 1.05 or ess < 200.0:
            continuation = run_exact_model_mcmc(
                dataset,
                chains=chains,
                warmup=warmup,
                draws=draws,
                seed=_child_seed(seed, 702),
                target_acceptance=0.85,
                initial_state=result.samples[:, -1, :],
            )
            samples = np.concatenate(
                [result.samples, continuation.samples],
                axis=1,
            )
            registered = registered_mcmc_diagnostics(samples, result.layout)
            rhat = float(np.max(split_rhat_values(registered)))
            ess = float(np.min(bulk_ess_values(registered)))
            continued = True
        elapsed = time.perf_counter() - started
        elapsed_total += elapsed
        rows.append(
            evaluate_state_draws(
                dataset,
                samples,
                layout=result.layout,
                method="exact_model_mcmc",
                inference_seconds=elapsed,
            )
        )
        diagnostics.append(
            {
                "seed": seed,
                "chains": chains,
                "retained_draws_per_chain": int(samples.shape[1]),
                "split_rhat_max": rhat,
                "bulk_ess_min": ess,
                "continuation_used": continued,
            }
        )
        if root is not None:
            np.savez_compressed(
                root / f"{seed}.npz",
                state_samples=samples,
                split_rhat_max=np.asarray(rhat),
                bulk_ess_min=np.asarray(ess),
                continuation_used=np.asarray(continued),
            )
    return rows, diagnostics, elapsed_total


def evaluate_python_hmsc_contexts(
    datasets: Sequence[GenerativeIidDataset],
    *,
    output_root: str | Path,
    python: str = sys.executable,
    samples: int = 1000,
    transient: int = 1000,
    chains: int = 4,
) -> tuple[list[dict[str, Any]], float]:
    """Fit qualified Python-native HMSC-HPC to the fixed comparator subset."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    elapsed_total = 0.0
    for dataset in datasets:
        seed = int(dataset.metadata["seed"])
        context_root = root / str(seed)
        if context_root.exists():
            raise FileExistsError(f"Python HMSC context exists: {context_root}")
        n_sites, n_species = dataset.Y.shape
        species = [f"sp{index + 1}" for index in range(n_species)]
        index = [f"site{site + 1}" for site in range(n_sites)]
        Y = np.asarray(dataset.Y, dtype=float).copy()
        Y[~dataset.response_mask] = np.nan
        y_frame = pd.DataFrame(Y, index=index, columns=species)
        x_frame = pd.DataFrame(
            {"x1": dataset.X[:, 1]},
            index=index,
        )
        study = pd.DataFrame({"site": index}, index=index)
        model = HmscModel(
            Y=y_frame,
            X=x_frame,
            x_formula="~ x1",
            distr="probit",
            study_design=study,
            random_levels={
                "site": {
                    "column": "site",
                    "type": "iid",
                    "nf": 2,
                    "nfMin": 2,
                    "nfMax": 2,
                }
            },
        )
        started = time.perf_counter()
        fit = model.sample(
            samples=samples,
            transient=transient,
            thin=1,
            chains=chains,
            backend="hmsc-hpc",
            init="python-native",
            verbose=max(samples + transient, 1),
            workdir=context_root,
            keep_workdir=True,
            python=python,
            rng_seed=_child_seed(seed, 703),
            fp=64,
        )
        elapsed_total += time.perf_counter() - started
        prediction_frame = x_frame.assign(site=index)
        probabilities = fit.predict_samples(
            prediction_frame,
            response=True,
            random_effects="known",
            rng_seed=_child_seed(seed, 704),
        )
        mean_probability = np.mean(probabilities, axis=(0, 1))
        hidden = ~dataset.response_mask
        masked_brier, masked_log_loss = _proper_scores(
            dataset.Y[hidden], mean_probability[hidden]
        )
        new_sites = deterministic_new_site_mask(n_sites, seed=seed)
        marginal = fit.predict_samples(
            prediction_frame.iloc[np.flatnonzero(new_sites)],
            response=True,
            random_effects="marginal",
            rng_seed=_child_seed(seed, 705),
        ).mean(axis=(0, 1))
        new_brier, new_log_loss = _proper_scores(
            dataset.Y[new_sites].ravel(),
            marginal.ravel(),
        )
        association = fit.species_associations(
            level=0,
            correlation=True,
        ).to_numpy(dtype=float)
        off_diagonal = np.triu_indices(n_species, k=1)
        rows.append(
            {
                "seed": seed,
                "method": "qualified_python_hmsc_hpc",
                "masked_cell_brier": masked_brier,
                "masked_cell_log_loss": masked_log_loss,
                "new_site_brier": new_brier,
                "new_site_log_loss": new_log_loss,
                "association_vector_mean": association[off_diagonal].tolist(),
                "all_finite": bool(
                    np.isfinite(mean_probability).all()
                    and np.isfinite(association).all()
                ),
            }
        )
    return rows, elapsed_total


def evaluate_v0_1_contexts(
    release: NeuralHmscRelease,
    datasets: Sequence[GenerativeIidDataset],
    *,
    draws: int = 256,
) -> list[dict[str, Any]]:
    """Evaluate immutable v0.1 only on its matched 40x75x2 support."""
    rows = []
    checkpoints = [
        release.load_checkpoint(seed=seed) for seed in release.seeds
    ]
    for dataset in datasets:
        if dataset.Y.shape != (40, 75) or dataset.X.shape[1] != 2:
            raise ValueError("v0.1 comparator received an unmatched context")
        member_draws = []
        for index, checkpoint in enumerate(checkpoints):
            posterior = checkpoint.predict_beta_posterior(
                {
                    "X": dataset.X,
                    "Y": dataset.Y,
                    "formula": checkpoint.formula,
                    "distribution": "probit",
                    "covariate_names": checkpoint.covariate_names,
                    "species_names": checkpoint.species_names,
                }
            )
            member_draws.append(
                np.asarray(
                    sample_beta_posterior(
                        posterior,
                        draws=draws,
                        seed=_child_seed(
                            int(dataset.metadata["seed"]),
                            900 + index,
                        ),
                    )[:, 0]
                )
            )
        beta = np.concatenate(member_draws, axis=0)
        probability = np.mean(
            ndtr(np.einsum("ni,dis->dns", dataset.X, beta)),
            axis=0,
        )
        hidden = ~dataset.response_mask
        brier, log_loss = _proper_scores(
            dataset.Y[hidden], probability[hidden]
        )
        coverage = _coverage95(beta, dataset.truth_beta)
        rows.append(
            {
                "seed": int(dataset.metadata["seed"]),
                "method": "immutable_neural_hmsc_v0_1",
                "marginal": {
                    "Beta": {
                        "coverage_95": coverage,
                        "element_count": int(dataset.truth_beta.size),
                    }
                },
                "masked_cell_brier": brier,
                "masked_cell_log_loss": log_loss,
                "all_finite": bool(np.isfinite(beta).all()),
            }
        )
    return rows


def _posterior_condition_bound(posterior: Any) -> float:
    diagonal = np.square(np.asarray(posterior.diagonal_scale))
    factor = np.asarray(posterior.low_rank_factor)
    mask = np.asarray(posterior.state_mask, dtype=bool)
    active_diagonal = diagonal[mask]
    if not active_diagonal.size:
        return 1.0
    minimum = max(float(np.min(active_diagonal)), np.finfo(float).eps)
    maximum = float(np.max(active_diagonal))
    factor_norm = float(
        np.max(np.sum(np.square(factor), axis=(1, 2)))
    )
    return (maximum + factor_norm) / minimum


def _peak_device_memory_bytes() -> int:
    try:
        devices = tf.config.list_logical_devices("GPU")
        if not devices:
            return 0
        info = tf.config.experimental.get_memory_info("GPU:0")
        return int(info.get("peak", 0))
    except (RuntimeError, ValueError):
        return 0


def _child_seed(seed: int, tag: int) -> int:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(tag)]))
    return int(rng.integers(1, np.iinfo(np.int32).max))


def _proper_scores(
    y: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    target = np.asarray(y, dtype=float)
    prediction = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return (
        float(np.mean(np.square(prediction - target))),
        float(
            -np.mean(
                target * np.log(prediction)
                + (1.0 - target) * np.log(1.0 - prediction)
            )
        ),
    )


def _coverage95(draws: np.ndarray, truth: np.ndarray) -> float:
    values = np.asarray(draws, dtype=float)
    lower, upper = np.quantile(values, [0.025, 0.975], axis=0)
    target = np.asarray(truth, dtype=float)
    return float(np.mean((lower <= target) & (target <= upper)))
