from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from examples import diagnose_generative_neural_hmsc_iid_v2_numerics as diagnostic
from pyhmsc.neural import generative_iid_v2 as v2


def test_v2_numerical_diagnostic_uses_only_ordinary_seeds():
    assert diagnostic.ORDINARY_SEEDS == tuple(range(983_001, 983_019))
    assert all(seed < 1_000_000 for seed in diagnostic.ORDINARY_SEEDS)
    source = Path(
        "examples/diagnose_generative_neural_hmsc_iid_v2_numerics.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("593000", "594000", "511000", "512000", "513000", "514000", "515000"):
        assert forbidden not in source


def test_v2_numerical_diagnostic_refuses_opening_tokens(monkeypatch):
    monkeypatch.setenv("OPEN_GENERATIVE_IID_V2_511M_TRAINING", "forbidden")
    try:
        diagnostic._require_no_opening_tokens()
    except RuntimeError as error:
        assert "refuses opening tokens" in str(error)
    else:
        raise AssertionError("diagnostic accepted an opening token")


def test_v2_numerical_review_scheduler_is_no_ledger_and_hash_pinned():
    source = Path(
        "docs/lumi_generative_neural_hmsc_iid_v2_numerical_review_sbatch.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=dev-g" in source
    assert "e3ca2876bdad56ce6c42d45720f88435742567e8977c292c530316cdd32e973b" in source
    assert "^OPEN_GENERATIVE_IID" in source
    assert "frozen symmetric_float64 symmetric_float64_cpu" in source
    assert "runpy.run_path" in source
    assert "GENERATE_593M" not in source
    assert "OPEN_GENERATIVE_IID_V2_511M" not in source


def test_v2_woodbury_kernel_uses_exact_symmetric_float64_cpu_factorization():
    source = Path("pyhmsc/neural/generative_iid_v2.py").read_text(encoding="utf-8")
    assert 'work_dtype = tf.float64' in source
    assert 'with tf.device("/CPU:0")' in source
    assert "0.5 * (small + tf.linalg.matrix_transpose(small))" in source

    rng = np.random.default_rng(983_101)
    log_scale = tf.Variable(rng.uniform(-5.0, 0.5, size=(2, 64)), dtype=tf.float32)
    factor = tf.Variable(rng.normal(0.0, 2.0, size=(2, 64, 16)), dtype=tf.float32)
    mask = tf.sequence_mask([64, 37], 64)
    with tf.GradientTape() as tape:
        inverse, weighted, chol, logdet = v2._masked_low_rank_terms(
            log_scale,
            factor,
            mask,
        )
        objective = (
            tf.reduce_sum(inverse)
            + tf.reduce_sum(weighted)
            + tf.reduce_sum(chol)
            + tf.reduce_sum(logdet)
        )
    gradients = tape.gradient(objective, (log_scale, factor))
    for value in (inverse, weighted, chol, logdet, objective, *gradients):
        assert bool(tf.reduce_all(tf.math.is_finite(value)))


@pytest.mark.slow
def test_v2_mixed_shape_two_epoch_training_is_finite_on_ordinary_corpus():
    batch = diagnostic._ordinary_batch()
    tf.keras.utils.set_random_seed(diagnostic.MODEL_INITIALIZATION_SEED)
    model = v2.GenerativeIidOrbitPosteriorModel()
    history = v2.train_generative_iid_orbit_model(
        model,
        batch,
        epochs=2,
        model_seed=diagnostic.MODEL_INITIALIZATION_SEED,
    )
    assert np.isfinite(history.loss).all()
    assert np.isfinite(history.iwelbo).all()
    assert np.isfinite(history.gradient_norm).all()


def test_v2_numerical_review_keeps_every_ledger_block_sealed():
    review = Path(
        "docs/generative_neural_hmsc_iid_v2_numerical_review_2026-08-01.md"
    ).read_text(encoding="utf-8")
    assert "implementation_repair_accepted_keep_511m_515m_sealed" in review
    assert "87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc" in review
    assert "ledger_seeds_opened = false" in review
    assert "Blocks 511M-515M remain sealed" in review
    assert "does not retroactively pass" in review
