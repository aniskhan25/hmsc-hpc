import numpy as np
import pytest

from pyhmsc import HmscModel, simulate_fixed_effect_data


@pytest.mark.slow
def test_gaussian_fixed_effect_recovery_signs():
    beta = np.array([[0.0, 0.0], [1.0, -1.0]])
    Y, X, truth = simulate_fixed_effect_data(n_sites=30, beta=beta, distr="normal", seed=12)
    model = HmscModel(Y=Y, X=X, x_formula="~ x", distr="normal")
    fit = model.sample(samples=5, transient=5, thin=1, chains=1, init="python-native", verbose=5)
    observed = fit.beta_mean().loc["x"]
    assert np.sign(observed["sp1"]) == np.sign(truth.loc["x", "sp1"])
