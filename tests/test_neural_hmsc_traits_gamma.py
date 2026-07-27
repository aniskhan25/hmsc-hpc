import json

import h5py
import numpy as np
import tensorflow as tf

from pyhmsc.neural.benchmark import compare_gamma_posterior_files
from pyhmsc.neural.evaluation import evaluate_gamma_posterior, predict_gamma_posterior
from pyhmsc.neural.models import TraitGammaPosteriorModel
from pyhmsc.neural.simulator import simulate_trait_effect_dataset
from pyhmsc.neural.storage import write_gamma_posterior_hdf5
from pyhmsc.neural.train import compiled_trait_effect_training_data, trait_effect_training_data
from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.posterior import HmscFit


def _trait_datasets(count: int, seed_offset: int):
    return [
        simulate_trait_effect_dataset(
            n_sites=36,
            n_species=5,
            distribution="normal",
            seed=seed_offset + idx,
            beta_residual_scale=0.0,
            gaussian_sigma=0.2,
        )
        for idx in range(count)
    ]


def test_simulate_trait_effect_dataset_shapes_and_trait_mediation():
    dataset = simulate_trait_effect_dataset(
        n_sites=12,
        n_species=4,
        distribution="normal",
        seed=700,
        beta_residual_scale=0.0,
    )

    assert dataset.Y.shape == (12, 4)
    assert dataset.traits.shape == (4, 1)
    assert list(dataset.trait_design.columns) == ["Intercept", "body"]
    assert dataset.truth_gamma.shape == (3, 2)
    expected_beta = dataset.truth_gamma.to_numpy() @ dataset.trait_design.to_numpy().T
    np.testing.assert_allclose(dataset.truth_beta.to_numpy(), expected_beta)


def test_trait_gamma_model_outputs_nontrivial_gamma_posterior():
    tf.keras.utils.set_random_seed(710)
    data = trait_effect_training_data(_trait_datasets(4, 800))
    model = TraitGammaPosteriorModel(
        n_sites=36,
        n_covariates=3,
        n_species=5,
        n_traits=2,
        hidden_units=(64, 64),
    )

    posterior = predict_gamma_posterior(model, data)
    metrics = evaluate_gamma_posterior(posterior, data.Gamma)

    assert posterior.mean.shape == (4, 3, 2)
    assert posterior.scale.shape == (4, 3, 2)
    assert metrics.gamma_scale_min > 0.0
    assert metrics.gamma_mean_rmse_truth < 0.9 * metrics.zero_baseline_rmse_truth
    assert np.isfinite(metrics.gamma_interval_coverage_truth_95)


def test_compiled_trait_effect_training_data_uses_compiler_trait_design(tmp_path):
    dataset = simulate_trait_effect_dataset(
        n_sites=10,
        n_species=4,
        distribution="normal",
        seed=850,
        beta_residual_scale=0.0,
    )
    compiled = compile_hmsc_model(
        Y=dataset.Y,
        X=dataset.X,
        formula="~ x1 + x2",
        distr="normal",
        chains=1,
        output=tmp_path / "compiled",
        traits=dataset.traits,
        trait_formula="~ body",
    )

    data = compiled_trait_effect_training_data(
        compiled.init_json,
        dataset.truth_gamma.to_numpy(dtype=np.float32),
        beta_true=dataset.truth_beta.to_numpy(dtype=np.float32),
    )

    assert data.X.shape == (1, 10, 3)
    assert data.Y.shape == (1, 10, 4)
    assert data.T.shape == (1, 4, 1)
    body = dataset.traits["body"].to_numpy(dtype=np.float32)
    expected_body = (body - body.mean()) / body.std(ddof=1)
    np.testing.assert_allclose(data.T[0, :, 0], expected_body, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(data.Beta[0], dataset.truth_beta.to_numpy(dtype=np.float32), atol=1e-5)


def test_write_gamma_posterior_hdf5_loads_with_hmscfit(tmp_path):
    dataset = _trait_datasets(1, 900)[0]
    data = trait_effect_training_data([dataset])
    model = TraitGammaPosteriorModel(n_sites=36, n_covariates=3, n_species=5, n_traits=2)
    posterior = predict_gamma_posterior(model, data)

    path = write_gamma_posterior_hdf5(
        posterior,
        tmp_path / "gamma_posterior.h5",
        covariate_names=list(dataset.truth_gamma.index),
        trait_names=list(dataset.truth_gamma.columns),
        distribution="normal",
        chains=2,
        draws=5,
        seed=901,
    )
    fit = HmscFit.from_file(path)

    assert fit.gamma_samples().shape == (2, 5, 3, 2)
    assert list(fit.gamma_mean().index) == ["Intercept", "x1", "x2"]
    assert list(fit.gamma_mean().columns) == ["Intercept", "body"]
    summary = fit.gamma_summary(level=0.8)
    assert set(["covariate", "trait", "mean", "lower", "upper"]).issubset(summary.columns)


def test_compare_gamma_posterior_files_reports_metrics(tmp_path):
    truth = np.array([[0.0, 0.5], [1.0, -0.5]], dtype=float)
    neural = np.array([[[[0.0, 0.45], [1.05, -0.45]], [[0.1, 0.55], [0.95, -0.55]]]], dtype=float)
    mcmc = np.array(
        [
            [[[0.0, 0.50], [1.00, -0.50]], [[0.05, 0.52], [0.98, -0.48]]],
            [[[-0.05, 0.48], [1.02, -0.52]], [[0.02, 0.51], [0.99, -0.51]]],
        ],
        dtype=float,
    )
    neural_path = _write_gamma(tmp_path / "neural_gamma.h5", neural)
    mcmc_path = _write_gamma(tmp_path / "mcmc_gamma.h5", mcmc)

    row = compare_gamma_posterior_files(
        neural_posterior=neural_path,
        mcmc_posterior=mcmc_path,
        truth_gamma=truth,
        dataset="traits",
        distribution="normal",
    )

    assert row["parameter"] == "Gamma"
    assert row["n_covariates"] == 2
    assert row["n_traits"] == 2
    assert row["gamma_mean_rmse_mcmc"] >= 0.0
    assert 0.0 <= row["gamma_ci_overlap_95"] <= 1.0
    assert row["neural_gamma_mean_rmse_truth"] < 0.1


def _write_gamma(path, gamma_samples):
    metadata = {
        "names": {
            "covariates": ["Intercept", "x1"][: gamma_samples.shape[2]],
            "traits": ["Intercept", "body"][: gamma_samples.shape[3]],
        },
        "formula": {"X": "~ x1", "T": "~ body"},
        "distribution": "normal",
    }
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Gamma", data=gamma_samples)
        handle.attrs["nChains"] = int(gamma_samples.shape[0])
        handle.attrs["nDraws"] = int(gamma_samples.shape[1])
        handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
    return path
