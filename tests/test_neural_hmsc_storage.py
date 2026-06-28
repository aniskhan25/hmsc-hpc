import numpy as np

from pyhmsc.neural.evaluation import predict_beta_posterior
from pyhmsc.neural.models import FixedShapeBetaPosteriorModel
from pyhmsc.neural.simulator import simulate_fixed_effect_dataset
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.neural.train import fixed_shape_training_data
from pyhmsc.posterior import HmscFit
from pyhmsc.storage import inspect_posterior_storage


def test_write_beta_posterior_hdf5_loads_with_hmscfit(tmp_path):
    dataset = simulate_fixed_effect_dataset(
        n_sites=24,
        n_species=3,
        distribution="normal",
        seed=404,
    )
    data = fixed_shape_training_data([dataset])
    model = FixedShapeBetaPosteriorModel(n_sites=24, n_covariates=3, n_species=3)
    posterior = predict_beta_posterior(model, data)
    path = write_beta_posterior_hdf5(
        posterior,
        tmp_path / "neural_posterior.h5",
        covariate_names=list(dataset.truth_beta.index),
        species_names=list(dataset.truth_beta.columns),
        chains=2,
        draws=5,
        seed=99,
    )

    fit = HmscFit.from_file(path)
    beta = fit.beta_samples()

    assert beta.shape == (2, 5, 3, 3)
    assert fit.metadata["model_type"] == "neural-hmsc"
    assert fit.metadata["inference"]["engine"] == "amortized-neural"
    assert list(fit.beta_mean().index) == ["Intercept", "x1", "x2"]
    assert list(fit.beta_mean().columns) == ["sp1", "sp2", "sp3"]
    np.testing.assert_allclose(
        fit.beta_mean().to_numpy(),
        beta.mean(axis=(0, 1)),
    )
    ci = fit.beta_ci(level=0.8)
    assert set(ci) == {"lower", "upper"}
    assert ci["lower"].shape == (3, 3)


def test_neural_posterior_storage_info_reports_beta_and_metadata(tmp_path):
    dataset = simulate_fixed_effect_dataset(
        n_sites=16,
        n_species=2,
        distribution="normal",
        seed=405,
    )
    data = fixed_shape_training_data([dataset])
    model = FixedShapeBetaPosteriorModel(n_sites=16, n_covariates=3, n_species=2)
    posterior = predict_beta_posterior(model, data)
    path = write_beta_posterior_hdf5(
        posterior,
        tmp_path / "neural_posterior.h5",
        covariate_names=list(dataset.truth_beta.index),
        species_names=list(dataset.truth_beta.columns),
        chains=1,
        draws=7,
        seed=100,
    )

    info = inspect_posterior_storage(path)
    by_name = {dataset.name: dataset for dataset in info.datasets}

    assert info.metadata_present
    assert info.n_chains == 1
    assert info.n_draws == 7
    assert by_name["Beta"].shape == (1, 7, 3, 2)
    assert info.attrs["nChains"] == "1"


def test_full_covariance_beta_storage_preserves_correlated_draws(tmp_path):
    dataset = simulate_fixed_effect_dataset(
        n_sites=16,
        n_species=1,
        distribution="normal",
        seed=406,
    )
    data = fixed_shape_training_data([dataset])
    model = FixedShapeBetaPosteriorModel(
        n_sites=16,
        n_covariates=3,
        n_species=1,
        posterior_family="full_covariance_normal",
    )
    posterior = predict_beta_posterior(model, data)
    path = write_beta_posterior_hdf5(
        posterior,
        tmp_path / "full_covariance.h5",
        covariate_names=list(dataset.truth_beta.index),
        species_names=list(dataset.truth_beta.columns),
        chains=1,
        draws=2000,
        seed=101,
    )

    fit = HmscFit.from_file(path)
    samples = fit.beta_samples()[0, :, :, 0]
    expected = posterior.scale_tril.numpy()[0, 0] @ posterior.scale_tril.numpy()[0, 0].T

    assert fit.metadata["inference"]["posterior_family"] == "full_covariance_normal"
    np.testing.assert_allclose(np.cov(samples, rowvar=False), expected, atol=0.05)
