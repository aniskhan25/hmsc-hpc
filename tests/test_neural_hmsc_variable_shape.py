import numpy as np
import tensorflow as tf

from pyhmsc.neural.evaluation import evaluate_masked_beta_posterior, predict_variable_beta_posterior
from pyhmsc.neural.models import VariableShapeBetaPosteriorModel
from pyhmsc.neural.simulator import FixedEffectDataset, simulate_fixed_effect_dataset
from pyhmsc.neural.train import variable_shape_training_data


def _variable_gaussian_datasets() -> list[FixedEffectDataset]:
    return [
        simulate_fixed_effect_dataset(n_sites=16, n_species=2, distribution="normal", seed=11),
        simulate_fixed_effect_dataset(n_sites=24, n_species=4, distribution="normal", seed=12),
        simulate_fixed_effect_dataset(n_sites=20, n_species=3, distribution="normal", seed=13),
    ]


def test_variable_shape_training_data_pads_sites_species_and_masks():
    data = variable_shape_training_data(_variable_gaussian_datasets())

    assert data.X.shape == (3, 24, 3)
    assert data.Y.shape == (3, 24, 4)
    assert data.Beta.shape == (3, 3, 4)
    assert data.site_mask.sum(axis=1).tolist() == [16, 24, 20]
    assert data.species_mask.sum(axis=1).tolist() == [2, 4, 3]
    assert np.all(data.Y[0, :, 2:] == 0.0)
    assert np.all(data.Beta[2, :, 3] == 0.0)


def test_variable_shape_model_outputs_masked_positive_scales_and_nontrivial_means():
    tf.keras.utils.set_random_seed(3100)
    data = variable_shape_training_data(_variable_gaussian_datasets())
    model = VariableShapeBetaPosteriorModel(n_covariates=3)

    posterior = predict_variable_beta_posterior(model, data)
    metrics = evaluate_masked_beta_posterior(posterior, data.Beta, data.species_mask)

    assert posterior.mean.shape == (3, 3, 4)
    assert posterior.scale.shape == (3, 3, 4)
    assert np.allclose(posterior.mean.numpy()[0, :, 2:], 0.0)
    assert np.allclose(posterior.scale.numpy()[0, :, 2:], 0.0)
    assert metrics.beta_mean_rmse_truth < 0.85 * metrics.zero_baseline_rmse_truth
    assert metrics.beta_scale_min > 1e-4


def test_variable_shape_model_is_site_order_invariant():
    data = variable_shape_training_data([_variable_gaussian_datasets()[1]])
    permutation = np.array([5, 2, 1, 8, 0, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    permuted = type(data)(
        X=data.X[:, permutation, :],
        Y=data.Y[:, permutation, :],
        Beta=data.Beta,
        site_mask=data.site_mask[:, permutation],
        species_mask=data.species_mask,
    )
    model = VariableShapeBetaPosteriorModel(n_covariates=3)

    left = predict_variable_beta_posterior(model, data)
    right = predict_variable_beta_posterior(model, permuted)

    np.testing.assert_allclose(left.mean.numpy(), right.mean.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(left.scale.numpy(), right.scale.numpy(), rtol=1e-5, atol=1e-5)


def test_variable_shape_model_is_species_order_equivariant():
    data = variable_shape_training_data([_variable_gaussian_datasets()[1]])
    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted = type(data)(
        X=data.X,
        Y=data.Y[:, :, permutation],
        Beta=data.Beta[:, :, permutation],
        site_mask=data.site_mask,
        species_mask=data.species_mask[:, permutation],
    )
    model = VariableShapeBetaPosteriorModel(n_covariates=3)

    left = predict_variable_beta_posterior(model, data)
    right = predict_variable_beta_posterior(model, permuted)

    np.testing.assert_allclose(left.mean.numpy(), right.mean.numpy()[:, :, inverse], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(left.scale.numpy(), right.scale.numpy()[:, :, inverse], rtol=1e-5, atol=1e-5)
