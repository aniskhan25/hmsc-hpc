import numpy as np
import pandas as pd
import pytest

from pyhmsc import HmscModel, simulate_fixed_effect_data


def _assert_sign_recovered(fit, truth, covariate="x"):
    beta = fit.beta_mean()
    for species in truth.columns:
        expected = np.sign(truth.loc[covariate, species])
        if expected != 0:
            assert np.sign(beta.loc[covariate, species]) == expected


@pytest.mark.slow
def test_native_gaussian_fixed_effect_recovery_signs(tmp_path):
    beta = np.array([[0.0, 0.0], [1.5, -1.5]])
    Y, X, truth = simulate_fixed_effect_data(n_sites=50, beta=beta, distr="normal", seed=101)
    model = HmscModel(Y=Y, X=X, x_formula="~ x", distr="normal")

    fit = model.sample(
        samples=20,
        transient=20,
        thin=1,
        chains=1,
        init="python-native",
        verbose=20,
        workdir=tmp_path / "gaussian",
    )

    _assert_sign_recovered(fit, truth)


@pytest.mark.slow
def test_native_poisson_fixed_effect_recovery_signs(tmp_path):
    beta = np.array([[0.0, 0.0], [0.9, -0.9]])
    Y, X, truth = simulate_fixed_effect_data(n_sites=60, beta=beta, distr="poisson", seed=102)
    model = HmscModel(Y=Y, X=X, x_formula="~ x", distr="poisson")

    fit = model.sample(
        samples=20,
        transient=20,
        thin=1,
        chains=1,
        init="python-native",
        verbose=20,
        workdir=tmp_path / "poisson",
    )

    _assert_sign_recovered(fit, truth)


@pytest.mark.slow
def test_native_traits_phylogeny_slow_smoke(tmp_path):
    beta = np.array([[0.1, -0.1], [0.8, -0.8]])
    Y, X, _truth = simulate_fixed_effect_data(n_sites=20, beta=beta, distr="poisson", seed=103)
    traits = pd.DataFrame({"body_size": [1.0, 2.0]}, index=Y.columns)
    phylo = pd.DataFrame([[1.0, 0.25], [0.25, 1.0]], index=Y.columns, columns=Y.columns)
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        traits=traits,
        trait_formula="~ body_size",
        phylo_cov=phylo,
    )

    fit = model.sample(
        samples=5,
        transient=5,
        thin=1,
        chains=1,
        init="python-native",
        verbose=5,
        workdir=tmp_path / "traits_phylogeny",
    )

    assert fit.beta_mean().shape == (2, 2)
    assert fit.gamma_samples().shape[-1] == 2
    assert np.isfinite(fit.beta_mean().to_numpy()).all()


@pytest.mark.slow
def test_native_iid_random_intercept_slow_smoke(tmp_path):
    beta = np.array([[0.2, -0.2], [0.7, -0.7]])
    Y, X, _truth = simulate_fixed_effect_data(n_sites=24, beta=beta, distr="poisson", seed=104)
    study_design = pd.DataFrame({"plot": [f"plot_{idx % 4}" for idx in range(len(Y))]})
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={"plot": {"column": "plot", "type": "iid"}},
    )

    fit = model.sample(
        samples=5,
        transient=5,
        thin=1,
        chains=1,
        init="python-native",
        verbose=5,
        workdir=tmp_path / "iid",
    )

    assert fit.eta_samples(0).shape[2] == 4
    assert fit.lambda_samples(0).shape[-1] == 2
    assert np.isfinite(fit.eta_samples(0)).all()


@pytest.mark.slow
def test_native_spatial_full_slow_smoke(tmp_path):
    beta = np.array([[0.2, -0.2], [0.7, -0.7]])
    Y, X, _truth = simulate_fixed_effect_data(n_sites=12, beta=beta, distr="poisson", seed=105)
    study_design = pd.DataFrame(
        {
            "plot": [f"plot_{idx}" for idx in range(len(Y))],
            "xcoord": np.linspace(0.0, 1.0, len(Y)),
            "ycoord": np.linspace(1.0, 0.0, len(Y)),
        }
    )
    model = HmscModel(
        Y=Y,
        X=X,
        x_formula="~ x",
        distr="poisson",
        study_design=study_design,
        random_levels={"plot": {"column": "plot", "type": "spatial_full", "coords": ["xcoord", "ycoord"]}},
    )

    fit = model.sample(
        samples=5,
        transient=5,
        thin=1,
        chains=1,
        init="python-native",
        verbose=5,
        workdir=tmp_path / "spatial_full",
    )

    assert fit.eta_samples(0).shape[2] == len(Y)
    assert fit.lambda_samples(0).shape[-1] == 2
    assert np.isfinite(fit.eta_samples(0)).all()
