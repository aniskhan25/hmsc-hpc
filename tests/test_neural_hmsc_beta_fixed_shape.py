import numpy as np
import pytest
import tensorflow as tf

from pyhmsc.neural.evaluation import evaluate_beta_posterior, predict_beta_posterior
from pyhmsc.neural.models import FixedShapeBetaPosteriorModel
from pyhmsc.neural.posterior_heads import sample_beta_posterior
from pyhmsc.neural.simulator import simulate_fixed_effect_dataset
from pyhmsc.neural.train import fixed_shape_training_data


def _fixed_gaussian_datasets(count: int, seed_offset: int) -> list:
    return [
        simulate_fixed_effect_dataset(
            n_sites=32,
            n_species=3,
            distribution="normal",
            seed=seed_offset + idx,
            beta_scale=0.75,
            gaussian_sigma=0.25,
        )
        for idx in range(count)
    ]


def test_fixed_shape_beta_model_outputs_positive_scales_and_samples():
    data = fixed_shape_training_data(_fixed_gaussian_datasets(2, 100))
    model = FixedShapeBetaPosteriorModel(n_sites=32, n_covariates=3, n_species=3)

    posterior = predict_beta_posterior(model, data)
    samples = sample_beta_posterior(posterior, draws=5, seed=8)

    assert posterior.mean.shape == (2, 3, 3)
    assert posterior.scale.shape == (2, 3, 3)
    assert samples.shape == (5, 2, 3, 3)
    assert float(tf.reduce_min(posterior.scale)) > 0.0


def test_full_covariance_beta_model_outputs_valid_cholesky_and_correlated_samples():
    data = fixed_shape_training_data(_fixed_gaussian_datasets(2, 200))
    model = FixedShapeBetaPosteriorModel(
        n_sites=32,
        n_covariates=3,
        n_species=3,
        posterior_family="full_covariance_normal",
    )

    posterior = predict_beta_posterior(model, data)
    samples = sample_beta_posterior(posterior, draws=500, seed=9)

    assert posterior.posterior_family == "full_covariance_normal"
    assert posterior.scale_tril is not None
    assert posterior.scale_tril.shape == (2, 3, 3, 3)
    assert samples.shape == (500, 2, 3, 3)
    diagonal = tf.linalg.diag_part(posterior.scale_tril)
    assert float(tf.reduce_min(diagonal)) > 0.0
    covariance = np.cov(samples.numpy()[:, 0, :, 0], rowvar=False)
    expected = posterior.scale_tril.numpy()[0, 0] @ posterior.scale_tril.numpy()[0, 0].T
    np.testing.assert_allclose(covariance, expected, atol=0.08)


def test_fixed_shape_beta_model_starts_with_nontrivial_beta_posterior_means():
    tf.keras.utils.set_random_seed(2026)
    test_data = fixed_shape_training_data(_fixed_gaussian_datasets(8, 5000))
    model = FixedShapeBetaPosteriorModel(
        n_sites=32,
        n_covariates=3,
        n_species=3,
        hidden_units=(96, 96),
        min_scale=1e-3,
    )

    posterior = predict_beta_posterior(model, test_data)
    metrics = evaluate_beta_posterior(posterior, test_data.Beta)

    assert metrics.beta_mean_rmse_truth < 0.85 * metrics.zero_baseline_rmse_truth
    assert metrics.beta_scale_min > 1e-4
    assert np.isfinite(metrics.beta_interval_coverage_truth_95)
    assert metrics.beta_interval_width_mean_95 > 0.0


def test_poisson_fixed_shape_model_uses_log_response_ridge_anchor():
    X = tf.ones((1, 8, 1), dtype=tf.float32)
    Y = tf.ones((1, 8, 1), dtype=tf.float32) * (np.exp(2.0) - 1.0)
    model = FixedShapeBetaPosteriorModel(
        n_sites=8,
        n_covariates=1,
        n_species=1,
        distribution="poisson",
    )

    posterior = model({"X": X, "Y": Y}, training=False)

    assert posterior.mean.numpy()[0, 0, 0] == pytest.approx(2.0, abs=0.01)


def test_fixed_shape_training_data_rejects_mixed_shapes():
    datasets = [
        simulate_fixed_effect_dataset(n_sites=16, n_species=2, distribution="normal", seed=1),
        simulate_fixed_effect_dataset(n_sites=20, n_species=2, distribution="normal", seed=2),
    ]

    try:
        fixed_shape_training_data(datasets)
    except ValueError as exc:
        assert "same fixed shapes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mixed fixed-shape training data")
