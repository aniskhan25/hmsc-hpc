#!/usr/bin/env python3
"""No-ledger numerical diagnostic for the generative iid v2 posterior."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
from typing import Callable

import numpy as np
import tensorflow as tf

from pyhmsc.neural.generative_iid import (
    batch_generative_iid_datasets,
    simulate_generative_iid_dataset,
)
from pyhmsc.neural import generative_iid_v2 as v2


ORDINARY_SEEDS = tuple(range(983_001, 983_019))
MODEL_INITIALIZATION_SEED = 511_900_001


def _require_no_opening_tokens() -> None:
    present = sorted(
        name
        for name, value in os.environ.items()
        if name.startswith("OPEN_GENERATIVE_IID") and value
    )
    if present:
        raise RuntimeError(f"numerical diagnostic refuses opening tokens: {present}")


def _ordinary_batch():
    combinations = list(
        itertools.product(
            ("normal", "right_skewed"),
            ("weak", "medium", "strong"),
            ("rare", "moderate", "common"),
        )
    )
    shapes = ((24, 12), (40, 36), (96, 75))
    datasets = []
    for index, (covariate_shape, loading, prevalence) in enumerate(combinations):
        n_sites, n_species = shapes[index % len(shapes)]
        datasets.append(
            simulate_generative_iid_dataset(
                n_sites=n_sites,
                n_species=n_species,
                covariate_shape=covariate_shape,
                loading_stratum=loading,
                prevalence_stratum=prevalence,
                seed=ORDINARY_SEEDS[index],
            )
        )
    return batch_generative_iid_datasets(
        datasets,
        max_sites=96,
        max_species=75,
    )


def _replacement_terms(
    *,
    full_float64: bool,
    cpu_cholesky: bool,
) -> Callable:
    def terms(log_diagonal_scale, low_rank_factor, mask):
        output_dtype = log_diagonal_scale.dtype
        work_dtype = tf.float64 if full_float64 else output_dtype
        log_scale = tf.cast(log_diagonal_scale, work_dtype)
        factor = tf.cast(low_rank_factor, work_dtype)
        mask_float = tf.cast(mask, work_dtype)
        variance = tf.exp(2.0 * log_scale) * mask_float + (1.0 - mask_float)
        inverse_variance = tf.math.reciprocal(variance)
        factor *= mask_float[..., None]
        weighted_factor = factor * inverse_variance[..., None]
        rank = tf.shape(factor)[-1]
        small = tf.eye(
            rank,
            batch_shape=[tf.shape(factor)[0]],
            dtype=work_dtype,
        ) + tf.einsum("bdr,bds->brs", factor, weighted_factor)
        small = 0.5 * (small + tf.linalg.matrix_transpose(small))
        if cpu_cholesky:
            with tf.device("/CPU:0"):
                chol = tf.linalg.cholesky(small)
        else:
            chol = tf.linalg.cholesky(small)
        logdet = tf.reduce_sum(tf.math.log(variance) * mask_float, axis=-1)
        logdet += 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(chol)), axis=-1
        )
        return tuple(
            tf.cast(value, output_dtype)
            for value in (inverse_variance, weighted_factor, chol, logdet)
        )

    return terms


def _install_mode(mode: str) -> None:
    if mode == "frozen":
        return
    if mode == "symmetric_float64":
        v2._masked_low_rank_terms = _replacement_terms(
            full_float64=True,
            cpu_cholesky=False,
        )
        return
    if mode == "symmetric_float64_cpu":
        v2._masked_low_rank_terms = _replacement_terms(
            full_float64=True,
            cpu_cholesky=True,
        )
        return
    raise ValueError(f"unknown diagnostic mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("frozen", "symmetric_float64", "symmetric_float64_cpu"),
        required=True,
    )
    args = parser.parse_args()
    _require_no_opening_tokens()
    _install_mode(args.mode)
    batch = _ordinary_batch()
    tf.keras.utils.set_random_seed(MODEL_INITIALIZATION_SEED)
    model = v2.GenerativeIidOrbitPosteriorModel()
    result = {
        "status": "failed",
        "mode": args.mode,
        "protocol": v2.GENERATIVE_IID_V2_PROTOCOL,
        "ledger_seeds_opened": False,
        "ordinary_seed_range": [ORDINARY_SEEDS[0], ORDINARY_SEEDS[-1]],
        "model_initialization_seed": MODEL_INITIALIZATION_SEED,
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
    }
    try:
        history = v2.train_generative_iid_orbit_model(
            model,
            batch,
            epochs=2,
            model_seed=MODEL_INITIALIZATION_SEED,
        )
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise
    result.update(
        {
            "status": "finite",
            "loss": history.loss,
            "iwelbo": history.iwelbo,
            "gradient_norm": history.gradient_norm,
            "all_finite": bool(
                np.isfinite(history.loss).all()
                and np.isfinite(history.iwelbo).all()
                and np.isfinite(history.gradient_norm).all()
            ),
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
