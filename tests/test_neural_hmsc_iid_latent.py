import json

import h5py
import numpy as np

from pyhmsc.neural.benchmark import compare_iid_association_posterior_files
from pyhmsc.neural.evaluation import evaluate_iid_latent_posterior, predict_iid_latent_posterior
from pyhmsc.neural.models import IidLatentFactorPosteriorModel
from pyhmsc.neural.simulator import simulate_iid_latent_effect_dataset
from pyhmsc.neural.storage import write_iid_latent_posterior_hdf5
from pyhmsc.neural.train import iid_latent_training_data
from pyhmsc.posterior import HmscFit


def _iid_dataset(seed: int = 1000):
    return simulate_iid_latent_effect_dataset(
        n_sites=28,
        n_species=4,
        n_groups=28,
        n_factors=1,
        distribution="normal",
        seed=seed,
        beta_scale=0.02,
        gaussian_sigma=0.01,
    )


def test_simulate_iid_latent_effect_dataset_shapes_and_truth():
    dataset = _iid_dataset(1001)

    assert dataset.Y.shape == (28, 4)
    assert dataset.study_design.shape == (28, 1)
    assert list(dataset.study_design.columns) == ["plot"]
    assert dataset.group_codes.shape == (28,)
    assert dataset.truth_eta.shape == (28, 1)
    assert dataset.truth_lambda.shape == (1, 4)
    expected = dataset.truth_eta.to_numpy()[dataset.group_codes] @ dataset.truth_lambda.to_numpy()
    np.testing.assert_allclose(dataset.truth_random_effect.to_numpy(), expected)


def test_iid_latent_model_recovers_random_effect_and_association_invariants():
    data = iid_latent_training_data([_iid_dataset(1002)])
    model = IidLatentFactorPosteriorModel(
        n_sites=28,
        n_covariates=3,
        n_species=4,
        n_groups=28,
        n_factors=1,
        eta_scale=0.02,
        lambda_scale=0.02,
    )

    posterior = predict_iid_latent_posterior(model, data)
    metrics = evaluate_iid_latent_posterior(posterior, data)

    assert posterior.eta_mean.shape == (1, 28, 1)
    assert posterior.lambda_mean.shape == (1, 1, 4)
    assert metrics.random_effect_rmse_truth < 0.2
    assert metrics.association_rmse_truth < 0.5
    assert metrics.association_correlation_truth > 0.8
    assert metrics.eta_scale_mean > 0.0
    assert metrics.lambda_scale_mean > 0.0


def test_write_iid_latent_posterior_hdf5_loads_with_hmscfit(tmp_path):
    dataset = _iid_dataset(1003)
    data = iid_latent_training_data([dataset])
    model = IidLatentFactorPosteriorModel(
        n_sites=28,
        n_covariates=3,
        n_species=4,
        n_groups=28,
        n_factors=1,
    )
    posterior = predict_iid_latent_posterior(model, data)

    path = write_iid_latent_posterior_hdf5(
        posterior,
        tmp_path / "iid_latent.h5",
        covariate_names=list(dataset.truth_beta.index),
        species_names=list(dataset.truth_beta.columns),
        group_names=list(dataset.truth_eta.index),
        chains=2,
        draws=5,
        seed=1004,
    )
    fit = HmscFit.from_file(path)

    assert fit.beta_samples().shape == (2, 5, 3, 4)
    assert fit.eta_samples(0).shape == (2, 5, 28, 1)
    assert fit.lambda_samples(0).shape == (2, 5, 1, 4)
    assert list(fit.eta_mean(0).index) == list(dataset.truth_eta.index)
    associations = fit.species_associations(level=0, correlation=False)
    assert associations.shape == (4, 4)


def test_compare_iid_association_posterior_files_is_sign_invariant(tmp_path):
    loadings = np.array([[0.5, -1.0, 0.25]], dtype=float)
    neural_lambda = np.stack([loadings, -loadings], axis=0)[None, ...]
    mcmc_lambda = np.stack([loadings, loadings * 1.02], axis=0)[None, ...]
    eta = np.ones((1, 2, 2, 1), dtype=float)
    neural_path = _write_latent(tmp_path / "neural.h5", eta, neural_lambda)
    mcmc_path = _write_latent(tmp_path / "mcmc.h5", eta, mcmc_lambda)

    row = compare_iid_association_posterior_files(
        neural_posterior=neural_path,
        mcmc_posterior=mcmc_path,
        truth_lambda=loadings,
        dataset="iid",
    )

    assert row["parameter"] == "Associations"
    assert row["association_rmse_mcmc"] < 0.05
    assert row["neural_association_rmse_truth"] < 1e-12


def _write_latent(path, eta, loadings):
    metadata = {
        "names": {"species": ["sp1", "sp2", "sp3"][: loadings.shape[-1]]},
        "random_levels": [{"levels": ["plot_a", "plot_b"][: eta.shape[2]]}],
    }
    with h5py.File(path, "w") as handle:
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=eta)
        level.create_dataset("Lambda", data=loadings)
        handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
    return path
