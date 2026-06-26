import numpy as np

from pyhmsc.neural.evaluation import (
    evaluate_spatial_latent_posterior,
    predict_spatial_latent_posterior,
    spatial_holdout_random_effect_rmse,
)
from pyhmsc.neural.models import SpatialLatentFactorPosteriorModel
from pyhmsc.neural.simulator import simulate_spatial_latent_effect_dataset
from pyhmsc.neural.storage import write_spatial_latent_posterior_hdf5
from pyhmsc.neural.train import spatial_latent_training_data
from pyhmsc.posterior import HmscFit


def _spatial_dataset(seed: int = 2000):
    return simulate_spatial_latent_effect_dataset(
        n_sites=36,
        n_species=4,
        n_factors=1,
        distribution="normal",
        seed=seed,
        beta_scale=0.02,
        spatial_range=0.28,
        spatial_sd=1.0,
        gaussian_sigma=0.01,
        holdout_stride=5,
    )


def test_simulate_spatial_latent_effect_dataset_shapes_and_split():
    dataset = _spatial_dataset(2001)

    assert dataset.Y.shape == (36, 4)
    assert dataset.coords.shape == (36, 2)
    assert dataset.study_design[["xcoord", "ycoord"]].shape == (36, 2)
    assert dataset.truth_eta.shape == (36, 1)
    assert dataset.truth_lambda.shape == (1, 4)
    assert dataset.train_mask.sum() + dataset.test_mask.sum() == 36
    assert dataset.train_mask.any()
    assert dataset.test_mask.any()
    expected = dataset.truth_eta.to_numpy() @ dataset.truth_lambda.to_numpy()
    np.testing.assert_allclose(dataset.truth_random_effect.to_numpy(), expected)


def test_spatial_latent_model_evaluates_holdout_modes():
    dataset = _spatial_dataset(2002)
    data = spatial_latent_training_data([dataset])
    model = SpatialLatentFactorPosteriorModel(
        n_sites=36,
        n_covariates=3,
        n_species=4,
        n_factors=1,
        spatial_range=0.28,
        eta_scale=0.02,
        lambda_scale=0.02,
    )

    posterior = predict_spatial_latent_posterior(model, data)
    metrics = evaluate_spatial_latent_posterior(posterior, data, spatial_range=0.28)
    nearest = spatial_holdout_random_effect_rmse(posterior, data, mode="nearest", spatial_range=0.28)
    conditional = spatial_holdout_random_effect_rmse(posterior, data, mode="conditional", spatial_range=0.28)

    assert posterior.eta_mean.shape == (1, 36, 1)
    assert posterior.lambda_mean.shape == (1, 1, 4)
    assert metrics.random_effect_rmse_truth < 1.5
    assert metrics.association_correlation_truth > 0.4
    assert np.isfinite(metrics.residual_nearest_correlation)
    assert nearest >= 0.0
    assert conditional >= 0.0
    assert metrics.holdout_nearest_rmse_truth == nearest
    assert metrics.holdout_conditional_rmse_truth == conditional


def test_write_spatial_latent_posterior_hdf5_loads_with_hmscfit(tmp_path):
    dataset = _spatial_dataset(2003)
    data = spatial_latent_training_data([dataset])
    model = SpatialLatentFactorPosteriorModel(
        n_sites=36,
        n_covariates=3,
        n_species=4,
        n_factors=1,
        spatial_range=0.28,
    )
    posterior = predict_spatial_latent_posterior(model, data)

    path = write_spatial_latent_posterior_hdf5(
        posterior,
        tmp_path / "spatial_latent.h5",
        covariate_names=list(dataset.truth_beta.index),
        species_names=list(dataset.truth_beta.columns),
        site_names=list(dataset.truth_eta.index),
        coords=dataset.coords.to_numpy(dtype=float),
        chains=2,
        draws=5,
        seed=2004,
    )
    fit = HmscFit.from_file(path)

    assert fit.beta_samples().shape == (2, 5, 3, 4)
    assert fit.eta_samples(0).shape == (2, 5, 36, 1)
    assert fit.lambda_samples(0).shape == (2, 5, 1, 4)
    assert fit.metadata["random_levels"][0]["type"] == "spatial_full"
    assert fit.metadata["random_levels"][0]["coords"] == ["xcoord", "ycoord"]
    assert len(fit.metadata["random_levels"][0]["coordinate_values"]) == 36
